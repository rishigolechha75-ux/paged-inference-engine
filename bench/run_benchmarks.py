"""Run all benchmarks and write reports/*.json and reports/*.png.

    python -m bench.run_benchmarks            # full run (~15 min on a laptop CPU)
    python -m bench.run_benchmarks --quick    # smaller workload for a smoke test
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch

from pagedserve.config import EngineConfig
from pagedserve.model import kv_bytes_per_token, load_gpt2
from pagedserve.tokenizer import GPT2Tokenizer

from .harness import run
from .workload import dolly_requests, shared_prefix_requests

REPORTS = Path(__file__).resolve().parent.parent / "reports"

MAX_TOKENS = 256          # client-side cap every request sends
NUM_BLOCKS = 384          # KV budget: 384 x 16 = 6144 token slots
BLOCK = 16

SYSTEMS = {
    "static_reserve": dict(policy="static", reserve_max=True),
    "continuous_reserve": dict(policy="continuous", reserve_max=True),
    "continuous_paged": dict(policy="continuous", reserve_max=False),
}


def base_cfg(**kw) -> EngineConfig:
    args = dict(block_size=BLOCK, num_blocks=NUM_BLOCKS, max_num_seqs=64, max_num_batched_tokens=512)
    args.update(kw)
    return EngineConfig(**args)


def save(name: str, obj) -> None:
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"{name}.json").write_text(json.dumps(obj, indent=2) + "\n")
    print(f"wrote reports/{name}.json")


def fragmentation(requests, block_sizes=(1, 8, 16, 32, 64, 128)):
    """Exact KV slot waste at request completion for each allocation scheme."""
    final = [len(r.prompt_ids) + r.output_len for r in requests]
    out = {"reserve_max_tokens": {"allocated": sum(len(r.prompt_ids) + MAX_TOKENS for r in requests)}}
    for bs in block_sizes:
        out[f"paged_block_{bs}"] = {"allocated": sum(-(-n // bs) * bs for n in final)}
    used = sum(final)
    for v in out.values():
        v["used"] = used
        v["waste_pct"] = round(100 * (1 - used / v["allocated"]), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n = 40 if args.quick else 160
    rates = [2.0, 8.0] if args.quick else [2.0, 4.0, 8.0, 16.0]

    torch.manual_seed(0)
    model = load_gpt2()
    tok = GPT2Tokenizer()
    reqs = dolly_requests(tok, n)
    plens = [len(r.prompt_ids) for r in reqs]
    olens = [r.output_len for r in reqs]

    env = {
        "date": time.strftime("%Y-%m-%d"),
        "machine": platform.machine(), "processor": platform.processor() or platform.platform(),
        "python": platform.python_version(), "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(), "device": "cpu", "dtype": "float32",
        "model": "openai-community/gpt2 (124M)",
        "kv_bytes_per_token": kv_bytes_per_token(model.cfg),
        "kv_budget_tokens": NUM_BLOCKS * BLOCK,
        "kv_budget_mib": round(NUM_BLOCKS * BLOCK * kv_bytes_per_token(model.cfg) / 2**20, 1),
        "max_tokens_cap": MAX_TOKENS,
        "workload": {"dataset": "databricks-dolly-15k (CC BY-SA 3.0)", "requests": len(reqs),
                     "prompt_len_mean": round(float(np.mean(plens)), 1),
                     "prompt_len_p50": float(np.median(plens)), "prompt_len_max": max(plens),
                     "output_len_mean": round(float(np.mean(olens)), 1),
                     "output_len_p50": float(np.median(olens)), "output_len_max": max(olens)},
    }
    save("environment", env)
    save("fragmentation", fragmentation(reqs))

    # 1. offline throughput: everything arrives at t=0
    offline = {name: run(model, base_cfg(**kw), reqs) for name, kw in SYSTEMS.items()}
    save("offline_throughput", offline)

    # 2. online serving: Poisson arrivals
    online = {name: [run(model, base_cfg(**kw), reqs, rate=r) for r in rates] for name, kw in SYSTEMS.items()}
    save("online_latency", online)

    # 3. block size sweep for the paged engine (offline)
    sweep = {}
    for bs in (8, 16, 32, 64):
        cfg = base_cfg(block_size=bs, num_blocks=NUM_BLOCKS * BLOCK // bs)
        sweep[str(bs)] = run(model, cfg, reqs)
    save("block_size_sweep", sweep)

    # 4. prefix caching on a shared few-shot prefix
    preqs = shared_prefix_requests(tok, 48 if args.quick else 96)
    prefix = {
        "prefix_len_tokens": min(len(r.prompt_ids) for r in preqs),
        "no_prefix_cache": run(model, base_cfg(), preqs, rate=8.0, max_tokens=64),
        "prefix_cache": run(model, base_cfg(enable_prefix_caching=True), preqs, rate=8.0, max_tokens=64),
    }
    save("prefix_caching", prefix)

    from .plots import make_plots
    make_plots(REPORTS)


if __name__ == "__main__":
    main()
