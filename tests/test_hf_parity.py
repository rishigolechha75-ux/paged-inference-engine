"""Token-for-token parity with Hugging Face transformers on real GPT-2.

Needs the 548 MB GPT-2 checkpoint, so it only runs when
PAGEDSERVE_HF_TESTS=1 is set.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("PAGEDSERVE_HF_TESTS") != "1",
                                reason="set PAGEDSERVE_HF_TESTS=1 to run real GPT-2 parity tests")

PROMPTS = [
    "The capital of France is",
    "In 1969, astronauts",
    "def quicksort(arr):",
    "Machine learning is a field of study that",
    "Once upon a time",
]


@pytest.fixture(scope="module")
def models():
    transformers = pytest.importorskip("transformers")
    from pagedserve.model import load_gpt2
    from pagedserve.tokenizer import GPT2Tokenizer

    hf = transformers.GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
    return load_gpt2(), hf, GPT2Tokenizer()


def test_greedy_generation_matches_transformers(models):
    from pagedserve.config import EngineConfig
    from pagedserve.engine import LLMEngine
    from pagedserve.sequence import SamplingParams

    ours, hf, tok = models
    engine = LLMEngine(ours, EngineConfig(block_size=16, num_blocks=256, max_num_batched_tokens=16))
    prompts = [tok.encode(p) for p in PROMPTS]
    seqs = engine.generate(prompts, SamplingParams(max_tokens=32, ignore_eos=True))
    for ids, seq in zip(prompts, seqs):
        with torch.no_grad():
            ref = hf.generate(torch.tensor([ids]), max_new_tokens=32, do_sample=False,
                              pad_token_id=50256)[0, len(ids):].tolist()
        assert seq.output_token_ids == ref


def test_logits_close_to_transformers(models):
    ours, hf, tok = models
    ids = torch.tensor([tok.encode(PROMPTS[3])])
    with torch.no_grad():
        diff = (ours.forward_dense(ids) - hf(ids).logits).abs().max().item()
    assert diff < 1e-3
