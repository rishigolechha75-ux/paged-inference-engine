"""Command line interface: ``pagedserve generate`` and ``pagedserve serve``."""

from __future__ import annotations

import argparse
import sys
import time

from .config import EngineConfig
from .engine import LLMEngine
from .sequence import SamplingParams


def _engine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="openai-community/gpt2", help="HF repo id with GPT-2 safetensors")
    p.add_argument("--device", default="cpu")
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--num-blocks", type=int, default=1024)
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--max-batched-tokens", type=int, default=512)
    p.add_argument("--policy", choices=["continuous", "static"], default="continuous")
    p.add_argument("--reserve-max", action="store_true", help="pre-reserve KV for prompt+max_tokens")
    p.add_argument("--prefix-caching", action="store_true")


def _build(args: argparse.Namespace):
    from .model import load_gpt2
    from .tokenizer import GPT2Tokenizer

    model = load_gpt2(args.model, device=args.device)
    tok = GPT2Tokenizer(args.model)
    cfg = EngineConfig(block_size=args.block_size, num_blocks=args.num_blocks, max_num_seqs=args.max_num_seqs,
                       max_num_batched_tokens=args.max_batched_tokens, policy=args.policy,
                       reserve_max=args.reserve_max, enable_prefix_caching=args.prefix_caching)
    return LLMEngine(model, cfg, device=args.device), tok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="pagedserve")
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate completions for one or more prompts")
    _engine_args(g)
    g.add_argument("prompts", nargs="+")
    g.add_argument("--max-tokens", type=int, default=40)
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--top-p", type=float, default=1.0)
    g.add_argument("--seed", type=int, default=None)

    s = sub.add_parser("serve", help="start the HTTP API")
    _engine_args(s)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    engine, tok = _build(args)

    if args.cmd == "generate":
        params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p,
                                seed=args.seed, eos_token_id=tok.eos_token_id)
        t0 = time.perf_counter()
        seqs = engine.generate([tok.encode(p) for p in args.prompts], params)
        dt = time.perf_counter() - t0
        for p, seq in zip(args.prompts, seqs):
            print(f"--- {p!r}\n{tok.decode(seq.output_token_ids)}\n")
        n = sum(len(s.output_token_ids) for s in seqs)
        print(f"{n} tokens in {dt:.2f}s ({n / dt:.1f} tok/s), {engine.stats.steps} steps", file=sys.stderr)
        return 0

    import uvicorn

    from .server import create_app

    uvicorn.run(create_app(engine, tok, model_name=args.model), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
