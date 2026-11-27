"""GPT-2 forward pass that reads and writes a paged KV cache.

The whole running batch is processed in one call, with no padding in the
token dimension: the new tokens of all sequences (1 for a decode, up to the
chunk size for a prefill) are concatenated into one flat ``[T, E]`` tensor,
so the QKV projection, output projection and MLP run as single matmuls.

Only attention needs per-sequence structure. Sequences are split into
*attention groups* (decodes, and prefill chunks) and each group is padded
to its own longest query length, so a 512-token prefill chunk never forces
the 1-token decodes in the same step to be padded to 512. For each layer
the new keys/values are scattered into their cache slots, each sequence's
context is gathered through its block table, and a causal mask built from
token positions is applied. Prefill, chunked prefill, recomputation after
preemption and decode all share this code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


@dataclass
class AttnGroup:
    """Sequences attended together (Bg sequences, Qg = longest query).

    index: [Bg, Qg] position of each query token in the flat batch (pad 0).
    valid: [Bg, Qg] False for padding.
    positions: [Bg, Qg] absolute token positions (pad 0).
    block_tables: [Bg, max_blocks] physical block ids (padded with 0).
    context_lens: [Bg] tokens in the cache after this step.
    """

    index: torch.Tensor
    valid: torch.Tensor
    positions: torch.Tensor
    block_tables: torch.Tensor
    context_lens: torch.Tensor


@dataclass
class BatchInput:
    """One engine step. tokens/positions/slot_mapping are flat ``[T]``;
    ``last_index[b]`` is the flat index of sequence b's last new token."""

    tokens: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    groups: List[AttnGroup]
    last_index: torch.Tensor


class KVCache:
    """Per-layer key/value block pools shaped [num_blocks, block_size, H, D]."""

    def __init__(self, cfg: ModelConfig, num_blocks: int, block_size: int,
                 device: str = "cpu", dtype: torch.dtype = torch.float32) -> None:
        shape = (num_blocks, block_size, cfg.n_head, cfg.head_dim)
        self.block_size = block_size
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layer)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layer)]

    @property
    def bytes_per_block(self) -> int:
        t = self.k[0]
        return 2 * len(self.k) * t[0].numel() * t.element_size()


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.ln_1 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_eps)
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_eps)
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.mlp_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_proj(F.gelu(self.c_fc(self.ln_2(x)), approximate="tanh"))

    def forward_paged(self, x: torch.Tensor, batch: BatchInput, k_cache: torch.Tensor,
                      v_cache: torch.Tensor, masks: List[torch.Tensor]) -> torch.Tensor:
        T, E = x.shape
        H, D = self.cfg.n_head, self.cfg.head_dim
        q, k, v = self.c_attn(self.ln_1(x)).split(E, dim=-1)
        q, k, v = q.reshape(T, H, D), k.reshape(T, H, D), v.reshape(T, H, D)

        # 1) scatter the new K/V into their physical slots
        k_cache.view(-1, H, D)[batch.slot_mapping] = k
        v_cache.view(-1, H, D)[batch.slot_mapping] = v

        out = torch.empty_like(q)
        bs = k_cache.shape[1]
        for g, mask in zip(batch.groups, masks):
            Bg, Qg = g.index.shape
            nb = g.block_tables.shape[1]
            # 2) gather each sequence's whole context through its block table
            keys = k_cache[g.block_tables].view(Bg, nb * bs, H, D).transpose(1, 2)
            vals = v_cache[g.block_tables].view(Bg, nb * bs, H, D).transpose(1, 2)
            # 3) masked attention: a query at position p sees keys 0..p
            qg = q[g.index].transpose(1, 2)  # [Bg, H, Qg, D]
            og = F.scaled_dot_product_attention(qg, keys, vals, attn_mask=mask).transpose(1, 2)
            out[g.index[g.valid]] = og[g.valid]

        x = x + self.c_proj(out.reshape(T, E))
        return x + self._mlp(x)

    def forward_dense(self, x: torch.Tensor) -> torch.Tensor:
        """Reference attention over the full sequence with no cache."""
        B, T, E = x.shape
        H, D = self.cfg.n_head, self.cfg.head_dim
        q, k, v = self.c_attn(self.ln_1(x)).split(E, dim=-1)
        q, k, v = (t.view(B, T, H, D).transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.c_proj(out.transpose(1, 2).reshape(B, T, E))
        return x + self._mlp(x)


class GPT2(nn.Module):
    """GPT-2 decoder with tied input/output embeddings."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.n_positions, cfg.n_embd)
        self.h = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_eps)

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln_f(x) @ self.wte.weight.T

    @torch.inference_mode()
    def forward_paged(self, batch: BatchInput, cache: KVCache) -> torch.Tensor:
        """Run one step and return logits for each sequence's last token: [B, V]."""
        masks = []
        for g in batch.groups:
            L = g.block_tables.shape[1] * cache.block_size
            key_pos = torch.arange(L, device=batch.tokens.device).view(1, 1, L)
            m = (key_pos <= g.positions.unsqueeze(-1)) & (key_pos < g.context_lens.view(-1, 1, 1))
            masks.append(m.unsqueeze(1))  # [Bg, 1, Qg, L]
        x = self.wte(batch.tokens) + self.wpe(batch.positions)
        for i, blk in enumerate(self.h):
            x = blk.forward_paged(x, batch, cache.k[i], cache.v[i], masks)
        return self._logits(x[batch.last_index])

    @torch.inference_mode()
    def forward_dense(self, tokens: torch.Tensor) -> torch.Tensor:
        """Full-sequence logits [B, T, V] without any cache (reference path)."""
        pos = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.wte(tokens) + self.wpe(pos)
        for blk in self.h:
            x = blk.forward_dense(x)
        return self._logits(x)


def init_random(cfg: ModelConfig, seed: int = 0) -> GPT2:
    """Random weights (std 0.02, like GPT-2) for tests that need no download."""
    torch.manual_seed(seed)
    model = GPT2(cfg)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.normal_(p, std=0.02)
    return model.eval()


def load_gpt2(repo_id: str = "openai-community/gpt2", device: str = "cpu",
              path: Optional[str] = None) -> GPT2:
    """Load pretrained GPT-2 weights from a safetensors file.

    The file is fetched from the Hugging Face Hub (cached under
    ``~/.cache/huggingface``) unless ``path`` points to a local copy.
    HF stores GPT-2 projections as Conv1D ([in, out]); they are transposed
    into ``nn.Linear`` layout here.
    """
    from safetensors.torch import load_file

    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id, "model.safetensors")
    raw: Dict[str, torch.Tensor] = {k.removeprefix("transformer."): v for k, v in load_file(path).items()}
    n_layer = 1 + max(int(k.split(".")[1]) for k in raw if k.startswith("h."))
    vocab, n_embd = raw["wte.weight"].shape
    cfg = ModelConfig(vocab_size=vocab, n_positions=raw["wpe.weight"].shape[0], n_embd=n_embd,
                      n_layer=n_layer, n_head=n_embd // 64)
    model = GPT2(cfg)
    rename = {"attn.c_attn": "c_attn", "attn.c_proj": "c_proj", "mlp.c_fc": "c_fc", "mlp.c_proj": "mlp_proj"}
    state: Dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if k.endswith(".attn.bias") or k.endswith(".attn.masked_bias"):
            continue  # causal-mask buffers, not weights
        for old, new in rename.items():
            if f".{old}." in k:
                k = k.replace(f".{old}.", f".{new}.")
                if k.endswith(".weight"):
                    v = v.t().contiguous()
                break
        state[k] = v
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def kv_bytes_per_token(cfg: ModelConfig, dtype_bytes: int = 4) -> int:
    return 2 * cfg.n_layer * cfg.n_embd * dtype_bytes


__all__ = ["AttnGroup", "BatchInput", "GPT2", "KVCache", "init_random", "load_gpt2", "param_count", "kv_bytes_per_token"]
