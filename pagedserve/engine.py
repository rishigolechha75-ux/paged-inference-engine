"""The engine loop: schedule -> build batch -> forward -> sample -> update."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import torch

from .block_manager import BlockManager
from .config import EngineConfig
from .model import GPT2, BatchInput, KVCache
from .sampler import Sampler
from .scheduler import Scheduler, SchedulerOutput
from .sequence import SamplingParams, Sequence, SeqStatus


@dataclass
class StepOutput:
    """Tokens produced in one step, keyed by sequence id."""

    new_tokens: Dict[int, int] = field(default_factory=dict)
    finished: List[Sequence] = field(default_factory=list)


@dataclass
class EngineStats:
    steps: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    generated_tokens: int = 0
    padded_query_slots: int = 0
    busy_time_s: float = 0.0
    kv_util_samples: List[float] = field(default_factory=list)
    kv_used_samples: List[int] = field(default_factory=list)
    batch_size_samples: List[int] = field(default_factory=list)


class LLMEngine:
    """Offline + online inference engine around a paged-KV GPT-2.

    ``step()`` runs exactly one forward pass over the scheduled batch.
    The API server calls it in a background loop; ``generate()`` drives it
    to completion for offline use and benchmarks.
    """

    def __init__(self, model: GPT2, cfg: EngineConfig, device: str = "cpu") -> None:
        if cfg.max_model_len > model.cfg.n_positions:
            cfg.max_model_len = model.cfg.n_positions
        self.model = model
        self.cfg = cfg
        self.device = device
        self.cache = KVCache(model.cfg, cfg.num_blocks, cfg.block_size, device=device)
        self.block_manager = BlockManager(cfg.num_blocks, cfg.block_size, cfg.enable_prefix_caching)
        self.scheduler = Scheduler(cfg, self.block_manager)
        self.sampler = Sampler()
        self.stats = EngineStats()
        self._ids = itertools.count()
        self.sequences: Dict[int, Sequence] = {}

    # --------------------------------------------------------------- requests
    def add_request(self, prompt_token_ids: List[int], params: Optional[SamplingParams] = None,
                    arrival_time: Optional[float] = None) -> Sequence:
        params = params or SamplingParams()
        if not prompt_token_ids:
            raise ValueError("prompt must contain at least one token")
        if any(t < 0 or t >= self.model.cfg.vocab_size for t in prompt_token_ids):
            raise ValueError("token id out of vocabulary range")
        total = len(prompt_token_ids) + params.max_tokens
        if total > self.cfg.max_model_len:
            raise ValueError(f"prompt + max_tokens = {total} exceeds max_model_len {self.cfg.max_model_len}")
        if self.block_manager.blocks_for(total) > self.cfg.num_blocks:
            raise ValueError("request can never fit in the KV cache")
        seq = Sequence(seq_id=next(self._ids), prompt_token_ids=list(prompt_token_ids), params=params)
        if arrival_time is not None:
            seq.arrival_time = arrival_time
        self.sequences[seq.seq_id] = seq
        self.scheduler.add(seq)
        return seq

    def has_unfinished(self) -> bool:
        return self.scheduler.has_work()

    # ------------------------------------------------------------------ batch
    def _build_batch(self, sched: SchedulerOutput) -> BatchInput:
        bs = self.cfg.block_size
        B = len(sched.scheduled)
        Q = max(n for _, n in sched.scheduled)
        max_blocks = max(len(s.block_table) for s, _ in sched.scheduled)
        tokens = torch.zeros(B, Q, dtype=torch.long)
        positions = torch.zeros(B, Q, dtype=torch.long)
        slots = torch.full((B, Q), -1, dtype=torch.long)
        tables = torch.zeros(B, max_blocks, dtype=torch.long)
        ctx = torch.zeros(B, dtype=torch.long)
        last = torch.zeros(B, dtype=torch.long)
        for b, (seq, n) in enumerate(sched.scheduled):
            start = seq.num_computed_tokens
            ids = seq.all_token_ids[start:start + n]
            pos = torch.arange(start, start + n)
            table = torch.tensor(seq.block_table, dtype=torch.long)
            tokens[b, :n] = torch.tensor(ids, dtype=torch.long)
            positions[b, :n] = pos
            slots[b, :n] = table[pos // bs] * bs + pos % bs
            tables[b, :len(seq.block_table)] = table
            ctx[b] = start + n
            last[b] = n - 1
        dev = self.device
        return BatchInput(tokens.to(dev), positions.to(dev), slots.to(dev), tables.to(dev), ctx.to(dev), last.to(dev))

    # ------------------------------------------------------------------- step
    def step(self) -> StepOutput:
        out = StepOutput()
        sched = self.scheduler.schedule()
        for seq in sched.rejected:
            seq.finish_time = time.perf_counter()
            out.finished.append(seq)
        if sched.is_empty:
            return out

        t0 = time.perf_counter()
        batch = self._build_batch(sched)
        logits = self.model.forward_paged(batch, self.cache)
        now = time.perf_counter()

        st = self.stats
        st.steps += 1
        st.busy_time_s += now - t0
        st.padded_query_slots += batch.tokens.numel()
        st.batch_size_samples.append(len(sched.scheduled))

        for b, (seq, n) in enumerate(sched.scheduled):
            if seq.is_prefill:
                st.prefill_tokens += n
            else:
                st.decode_tokens += n
            seq.num_computed_tokens += n
            self.block_manager.commit(seq)
            if seq.num_computed_tokens < seq.num_tokens:
                continue  # mid-prefill chunk: nothing to sample yet
            token = self.sampler.sample(logits[b], seq.params, seq.seq_id)
            seq.output_token_ids.append(token)
            seq.token_times.append(now)
            if seq.first_token_time is None:
                seq.first_token_time = now
            st.generated_tokens += 1
            out.new_tokens[seq.seq_id] = token
            reason = self._finish_reason(seq, token)
            if reason:
                seq.finish_reason = reason
                seq.finish_time = now
                self.scheduler.finish(seq)
                self.sampler.forget(seq.seq_id)
                out.finished.append(seq)

        used = self.block_manager.num_used_blocks
        stored = sum(s.num_computed_tokens for s in self.scheduler.running)
        st.kv_used_samples.append(used)
        st.kv_util_samples.append(stored / (used * self.cfg.block_size) if used else 0.0)
        return out

    def _finish_reason(self, seq: Sequence, token: int) -> Optional[str]:
        p = seq.params
        if not p.ignore_eos and p.eos_token_id is not None and token == p.eos_token_id:
            return "stop"
        if len(seq.output_token_ids) >= p.max_tokens:
            return "length"
        if seq.num_tokens >= self.cfg.max_model_len:
            return "length"
        return None

    # ---------------------------------------------------------------- offline
    def generate(self, prompts: Iterable[List[int]], params: Optional[SamplingParams] = None) -> List[Sequence]:
        """Run a list of prompts to completion and return them in input order."""
        seqs = [self.add_request(p, params) for p in prompts]
        while self.has_unfinished():
            self.step()
        return seqs

    def reset_stats(self) -> None:
        self.stats = EngineStats()


def generate_dense(model: GPT2, prompt: List[int], max_tokens: int) -> List[int]:
    """Greedy decoding by re-running the full sequence each step (no cache).

    Slow but obviously correct; tests use it as the ground truth.
    """
    ids = list(prompt)
    for _ in range(max_tokens):
        logits = model.forward_dense(torch.tensor([ids], device=next(model.parameters()).device))
        ids.append(int(torch.argmax(logits[0, -1]).item()))
    return ids[len(prompt):]


__all__ = ["LLMEngine", "StepOutput", "EngineStats", "generate_dense", "SeqStatus"]
