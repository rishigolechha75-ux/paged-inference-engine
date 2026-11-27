"""Token sampling: greedy, temperature, top-k and nucleus (top-p)."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from .sequence import SamplingParams


class Sampler:
    """Samples one token per row. Seeded requests get their own generator so
    their output does not depend on what else is in the batch."""

    def __init__(self) -> None:
        self._generators: Dict[int, torch.Generator] = {}

    def forget(self, seq_id: int) -> None:
        self._generators.pop(seq_id, None)

    def _generator(self, seq_id: int, seed: Optional[int]) -> Optional[torch.Generator]:
        if seed is None:
            return None
        gen = self._generators.get(seq_id)
        if gen is None:
            gen = torch.Generator().manual_seed(seed)
            self._generators[seq_id] = gen
        return gen

    def sample(self, logits: torch.Tensor, params: SamplingParams, seq_id: int) -> int:
        """Sample from a single row of logits ``[V]``."""
        if params.temperature == 0:
            return int(torch.argmax(logits).item())
        logits = logits.float() / params.temperature
        if params.top_k > 0:
            k = min(params.top_k, logits.numel())
            kth = torch.topk(logits, k).values[-1]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        if params.top_p < 1.0:
            sorted_probs, idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            # keep the smallest prefix whose mass reaches top_p (always >= 1 token)
            drop = cum - sorted_probs >= params.top_p
            sorted_probs = sorted_probs.masked_fill(drop, 0.0)
            probs = torch.zeros_like(probs).scatter(0, idx, sorted_probs)
            probs = probs / probs.sum()
        gen = self._generator(seq_id, params.seed)
        return int(torch.multinomial(probs.cpu(), 1, generator=gen).item())
