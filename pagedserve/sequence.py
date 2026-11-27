"""Request/sequence state tracked by the scheduler."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SamplingParams:
    """Per-request decoding options.

    temperature == 0 means greedy decoding. ``ignore_eos`` forces exactly
    ``max_tokens`` tokens, which benchmarks use to get deterministic lengths.
    """

    max_tokens: int = 32
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    seed: Optional[int] = None
    ignore_eos: bool = False
    eos_token_id: Optional[int] = 50256

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")


class SeqStatus(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Sequence:
    """One generation request and its KV-cache bookkeeping.

    ``num_computed_tokens`` counts tokens whose K/V are already in the cache.
    A sequence is in the *prefill* phase while that is below the prompt
    length, and in *decode* afterwards (exactly one new token per step).
    """

    seq_id: int
    prompt_token_ids: List[int]
    params: SamplingParams
    arrival_time: float = field(default_factory=time.perf_counter)
    output_token_ids: List[int] = field(default_factory=list)
    status: SeqStatus = SeqStatus.WAITING
    block_table: List[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    num_cached_prompt_tokens: int = 0
    num_preemptions: int = 0
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None
    finish_reason: Optional[str] = None
    token_times: List[float] = field(default_factory=list)

    @property
    def all_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def is_prefill(self) -> bool:
        # More than one token without K/V means a (possibly chunked or
        # recomputed) prefill; exactly one means a plain decode step.
        return self.num_uncomputed_tokens > 1

    @property
    def num_uncomputed_tokens(self) -> int:
        """Tokens that still need a forward pass before the next sample."""
        return self.num_tokens - self.num_computed_tokens

    @property
    def max_total_tokens(self) -> int:
        return len(self.prompt_token_ids) + self.params.max_tokens

    @property
    def is_finished(self) -> bool:
        return self.status == SeqStatus.FINISHED
