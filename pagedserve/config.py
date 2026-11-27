"""Configuration dataclasses for the model and the serving engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Shape of a GPT-2 style decoder-only transformer."""

    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    layer_norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @classmethod
    def tiny(cls) -> "ModelConfig":
        """A small random-weight config used by unit tests and CI."""
        return cls(vocab_size=256, n_positions=256, n_embd=64, n_layer=2, n_head=4)


@dataclass
class EngineConfig:
    """Knobs that control memory layout and scheduling.

    Attributes:
        block_size: tokens stored per KV-cache block (page).
        num_blocks: total KV-cache blocks in the pool (the memory budget).
        max_num_seqs: maximum sequences in the running batch.
        max_num_batched_tokens: token budget per engine step. Decode tokens
            are scheduled first; prefill chunks fill what is left (chunked
            prefill). Must be >= 1.
        policy: "continuous" (iteration-level scheduling) or "static"
            (admit a batch only when the previous batch fully finished).
        reserve_max: if True, reserve KV blocks for prompt + max_tokens at
            admission, mimicking a contiguous pre-allocated KV cache. If
            False, blocks are allocated lazily as tokens are produced.
        enable_prefix_caching: share full KV blocks between requests whose
            token prefixes are identical.
        max_model_len: hard cap on prompt + generated tokens.
    """

    block_size: int = 16
    num_blocks: int = 512
    max_num_seqs: int = 32
    max_num_batched_tokens: int = 512
    policy: str = "continuous"
    reserve_max: bool = False
    enable_prefix_caching: bool = False
    max_model_len: int = 1024

    def __post_init__(self) -> None:
        if self.policy not in ("continuous", "static"):
            raise ValueError(f"unknown policy {self.policy!r}")
        if self.block_size < 1 or self.num_blocks < 1:
            raise ValueError("block_size and num_blocks must be positive")
        if self.max_num_batched_tokens < 1 or self.max_num_seqs < 1:
            raise ValueError("max_num_batched_tokens and max_num_seqs must be positive")
