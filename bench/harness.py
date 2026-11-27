"""Replays a workload against the engine and collects latency metrics.

Arrivals follow a Poisson process (exponential inter-arrival gaps) in wall
clock time. ``rate=inf`` submits everything at t=0 (offline throughput).
"""

from __future__ import annotations

import math
import random
import time
from typing import Dict, List

import numpy as np

from pagedserve.config import EngineConfig
from pagedserve.engine import LLMEngine
from pagedserve.sequence import SamplingParams

from .workload import Request


def pct(xs: List[float], q: float) -> float:
    return float(np.percentile(xs, q)) if xs else float("nan")


def run(model, cfg: EngineConfig, requests: List[Request], rate: float = math.inf,
        max_tokens: int = 256, seed: int = 0) -> Dict:
    engine = LLMEngine(model, EngineConfig(**vars(cfg)))
    rng = random.Random(seed)
    gaps = [0.0 if math.isinf(rate) else rng.expovariate(rate) for _ in requests]
    offsets = np.cumsum(gaps).tolist()

    t0 = time.perf_counter()
    seqs = []
    nxt = 0
    while nxt < len(requests) or engine.has_unfinished():
        now = time.perf_counter() - t0
        while nxt < len(requests) and offsets[nxt] <= now:
            r = requests[nxt]
            params = SamplingParams(max_tokens=max_tokens, stop_after=r.output_len, ignore_eos=True)
            seqs.append(engine.add_request(r.prompt_ids, params, arrival_time=t0 + offsets[nxt]))
            nxt += 1
        if engine.has_unfinished():
            engine.step()
        elif nxt < len(requests):
            time.sleep(max(0.0, offsets[nxt] - (time.perf_counter() - t0)))
    makespan = time.perf_counter() - t0

    done = [s for s in seqs if s.finish_reason in ("stop", "length")]
    ttft = [s.first_token_time - s.arrival_time for s in done]
    e2e = [s.finish_time - s.arrival_time for s in done]
    tpot = [(s.finish_time - s.first_token_time) / (len(s.output_token_ids) - 1)
            for s in done if len(s.output_token_ids) > 1]
    out_tokens = sum(len(s.output_token_ids) for s in done)
    st = engine.stats
    return {
        "requests": len(done),
        "rejected": len(seqs) - len(done),
        "rate_req_s": None if math.isinf(rate) else rate,
        "makespan_s": round(makespan, 3),
        "output_tokens": out_tokens,
        "output_tok_s": round(out_tokens / makespan, 1),
        "req_s": round(len(done) / makespan, 3),
        "ttft_p50_s": round(pct(ttft, 50), 4),
        "ttft_p99_s": round(pct(ttft, 99), 4),
        "tpot_mean_ms": round(1000 * float(np.mean(tpot)), 2),
        "e2e_p50_s": round(pct(e2e, 50), 3),
        "e2e_p99_s": round(pct(e2e, 99), 3),
        "steps": st.steps,
        "mean_batch_size": round(float(np.mean(st.batch_size_samples)), 2),
        "max_batch_size": int(max(st.batch_size_samples)),
        "mean_kv_util": round(float(np.mean(st.kv_util_samples)), 4),
        "peak_kv_blocks": int(max(st.kv_used_samples)),
        "preemptions": engine.scheduler.num_preemptions,
        "prefill_tokens_computed": st.prefill_tokens,
        "prefix_hit_rate": round(engine.block_manager.prefix_hit_rate, 4),
    }
