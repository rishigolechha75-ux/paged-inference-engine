"""Decode throughput vs batch size (why batching matters at all).

Each run: B random 100-token prompts, 64 output tokens each, all at t=0,
ample KV memory. Writes reports/batch_scaling.json.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

from pagedserve.config import EngineConfig
from pagedserve.engine import LLMEngine
from pagedserve.model import load_gpt2
from pagedserve.sequence import SamplingParams


def main() -> None:
    model = load_gpt2()
    rng = random.Random(0)
    out = {}
    for b in (1, 2, 4, 8, 16, 32, 64):
        engine = LLMEngine(model, EngineConfig(num_blocks=1024, max_num_seqs=64, max_num_batched_tokens=8192))
        prompts = [[rng.randrange(50000) for _ in range(100)] for _ in range(b)]
        t0 = time.perf_counter()
        engine.generate(prompts, SamplingParams(max_tokens=64, ignore_eos=True))
        dt = time.perf_counter() - t0
        out[str(b)] = {"output_tok_s": round(b * 64 / dt, 1), "seconds": round(dt, 3)}
        print(b, out[str(b)])
    path = Path(__file__).resolve().parent.parent / "reports" / "batch_scaling.json"
    path.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
