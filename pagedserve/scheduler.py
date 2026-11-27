"""Iteration-level scheduler.

Every engine step the scheduler decides which sequences run and how many new
tokens each contributes, under two limits: a per-step token budget and the
free KV-cache blocks.

``continuous`` policy (Orca / vLLM style):
  1. Running sequences go first, oldest first. Decodes cost one token;
     sequences still in prefill get a chunk of whatever budget remains
     (chunked prefill), so a long prompt cannot stall ongoing decodes.
  2. If a running sequence needs a new block and none is free, the newest
     running sequence is *preempted*: its blocks are freed and it goes back
     to the front of the waiting queue to be recomputed later.
  3. Leftover budget admits waiting requests in FIFO order.

``static`` policy (the classic baseline): a new batch is admitted only once
every sequence of the previous batch has finished, so short requests wait
for the longest one in their batch.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Tuple

from .block_manager import BlockManager
from .config import EngineConfig
from .sequence import Sequence, SeqStatus


@dataclass
class SchedulerOutput:
    scheduled: List[Tuple[Sequence, int]] = field(default_factory=list)
    preempted: List[Sequence] = field(default_factory=list)
    rejected: List[Sequence] = field(default_factory=list)

    @property
    def num_tokens(self) -> int:
        return sum(n for _, n in self.scheduled)

    @property
    def is_empty(self) -> bool:
        return not self.scheduled


class Scheduler:
    def __init__(self, cfg: EngineConfig, block_manager: BlockManager) -> None:
        self.cfg = cfg
        self.bm = block_manager
        self.waiting: Deque[Sequence] = deque()
        self.running: List[Sequence] = []
        self.num_preemptions = 0

    def add(self, seq: Sequence) -> None:
        seq.status = SeqStatus.WAITING
        self.waiting.append(seq)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def finish(self, seq: Sequence) -> None:
        self.bm.free(seq)
        if seq in self.running:
            self.running.remove(seq)
        seq.status = SeqStatus.FINISHED

    def _preempt(self, seq: Sequence) -> None:
        """Recompute-style preemption: drop the KV cache, requeue at the front."""
        self.bm.free(seq)
        self.running.remove(seq)
        seq.num_computed_tokens = 0
        seq.num_preemptions += 1
        seq.status = SeqStatus.WAITING
        self.waiting.appendleft(seq)
        self.num_preemptions += 1

    def _reserve(self, seq: Sequence) -> int:
        """Tokens of KV memory to reserve when a sequence is admitted."""
        if self.cfg.reserve_max:
            return seq.max_total_tokens
        return seq.num_tokens

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()
        budget = self.cfg.max_num_batched_tokens

        # ---- 1. running sequences (oldest first)
        i = 0
        while i < len(self.running) and budget > 0:
            seq = self.running[i]
            n = min(seq.num_uncomputed_tokens, budget)
            target = seq.num_computed_tokens + n
            while not self.bm.can_grow(seq, target):
                victim = self.running[-1]
                self._preempt(victim)
                out.preempted.append(victim)
                if victim is seq:
                    break
            if seq.status != SeqStatus.RUNNING:
                break  # preempted itself; everything after it is gone too
            self.bm.ensure(seq, target)
            out.scheduled.append((seq, n))
            budget -= n
            i += 1

        # ---- 2. admit waiting sequences
        can_admit = not out.preempted
        if self.cfg.policy == "static":
            can_admit = can_admit and not self.running
        while can_admit and self.waiting and budget > 0 and len(self.running) < self.cfg.max_num_seqs:
            seq = self.waiting[0]
            reserve = self._reserve(seq)
            if not self.bm.can_admit(seq, reserve):
                if not self.running and not out.scheduled:
                    # cannot fit even into an empty cache: reject it
                    self.waiting.popleft()
                    seq.status = SeqStatus.FINISHED
                    seq.finish_reason = "rejected"
                    out.rejected.append(seq)
                    continue
                break
            self.waiting.popleft()
            self.bm.admit(seq, reserve)
            seq.status = SeqStatus.RUNNING
            self.running.append(seq)
            n = min(seq.num_uncomputed_tokens, budget)
            out.scheduled.append((seq, n))
            budget -= n
        return out
