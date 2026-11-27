"""Check token-for-token agreement with Hugging Face transformers.

Greedy-decodes Dolly prompts (stopping at EOS, as HF does) with this engine (batched, chunked prefill,
paged KV) and with ``transformers`` GPT2LMHeadModel.generate (one prompt
at a time) and writes reports/parity.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from pagedserve.config import EngineConfig
from pagedserve.engine import LLMEngine
from pagedserve.model import load_gpt2
from pagedserve.sequence import SamplingParams
from pagedserve.tokenizer import GPT2Tokenizer

from .workload import dolly_requests

N_PROMPTS = 24
N_TOKENS = 64


def main() -> None:
    from transformers import GPT2LMHeadModel

    ours = load_gpt2()
    hf = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
    tok = GPT2Tokenizer()
    prompts = [r.prompt_ids for r in dolly_requests(tok, N_PROMPTS, seed=7, max_prompt=256)]
    engine = LLMEngine(ours, EngineConfig(block_size=16, num_blocks=512, max_num_batched_tokens=128))
    seqs = engine.generate(prompts, SamplingParams(max_tokens=N_TOKENS, eos_token_id=50256))

    exact, first_div, max_diff = 0, [], 0.0
    for ids, seq in zip(prompts, seqs):
        with torch.no_grad():
            ref = hf.generate(torch.tensor([ids]), max_new_tokens=N_TOKENS, do_sample=False,
                              pad_token_id=50256)[0, len(ids):].tolist()
            full = torch.tensor([ids + ref])
            max_diff = max(max_diff, (ours.forward_dense(full) - hf(full).logits).abs().max().item())
        if seq.output_token_ids == ref:
            exact += 1
        else:
            pairs = zip(seq.output_token_ids, ref)
            first_div.append(next((i for i, (a, b) in enumerate(pairs) if a != b),
                                  min(len(seq.output_token_ids), len(ref))))
    report = {"prompts": N_PROMPTS, "tokens_per_prompt": N_TOKENS, "exact_match": exact,
              "first_divergence_positions": first_div, "max_abs_logit_diff_dense_vs_hf": max_diff}
    out = Path(__file__).resolve().parent.parent / "reports" / "parity.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
