import random

import pytest
import torch

from pagedserve.config import EngineConfig
from pagedserve.engine import LLMEngine, generate_dense
from pagedserve.sampler import Sampler
from pagedserve.sequence import SamplingParams

MAX_TOKENS = 12
GREEDY = SamplingParams(max_tokens=MAX_TOKENS, ignore_eos=True)


@pytest.fixture(scope="module")
def reference(tiny_model, prompts):
    return [generate_dense(tiny_model, p, MAX_TOKENS) for p in prompts]


CONFIGS = {
    "paged-bs16": EngineConfig(block_size=16, num_blocks=64, max_model_len=256),
    "paged-bs1": EngineConfig(block_size=1, num_blocks=1024, max_model_len=256),
    "paged-bs3-chunked": EngineConfig(block_size=3, num_blocks=200, max_num_batched_tokens=7, max_model_len=256),
    "static-reserve": EngineConfig(block_size=4, num_blocks=200, policy="static", reserve_max=True,
                                   max_num_seqs=3, max_model_len=256),
    "continuous-reserve": EngineConfig(block_size=4, num_blocks=200, reserve_max=True, max_model_len=256),
    "prefix-cache": EngineConfig(block_size=4, num_blocks=200, enable_prefix_caching=True, max_model_len=256),
    # 15 blocks of 4 tokens cannot hold all 8 sequences at once -> preemption
    "tight-memory": EngineConfig(block_size=4, num_blocks=15, max_num_batched_tokens=64, max_model_len=256),
}


@pytest.mark.parametrize("name", list(CONFIGS))
def test_paged_engine_matches_dense_reference(tiny_model, prompts, reference, name):
    cfg = CONFIGS[name]
    engine = LLMEngine(tiny_model, EngineConfig(**vars(cfg)))
    seqs = engine.generate(prompts, GREEDY)
    for seq, ref in zip(seqs, reference):
        assert seq.output_token_ids == ref, f"seq {seq.seq_id} diverged under {name}"
        assert seq.finish_reason == "length"
    assert engine.block_manager.num_free_blocks == cfg.num_blocks


def test_tight_memory_actually_preempts(tiny_model, prompts):
    engine = LLMEngine(tiny_model, EngineConfig(**vars(CONFIGS["tight-memory"])))
    engine.generate(prompts, GREEDY)
    assert engine.scheduler.num_preemptions > 0


def test_static_policy_waits_for_whole_batch(tiny_model):
    engine = LLMEngine(tiny_model, EngineConfig(block_size=4, num_blocks=100, policy="static",
                                                 max_num_seqs=2, max_model_len=256))
    a = engine.add_request([1, 2], SamplingParams(max_tokens=2, ignore_eos=True))
    b = engine.add_request([3, 4], SamplingParams(max_tokens=10, ignore_eos=True))
    c = engine.add_request([5, 6], SamplingParams(max_tokens=2, ignore_eos=True))
    while not a.is_finished:
        engine.step()
    engine.step()
    assert c.num_computed_tokens == 0  # slot is free but static batching waits for b
    while engine.has_unfinished():
        engine.step()
    assert c.first_token_time > b.finish_time


def test_continuous_policy_backfills_freed_slot(tiny_model):
    engine = LLMEngine(tiny_model, EngineConfig(block_size=4, num_blocks=100, max_num_seqs=2, max_model_len=256))
    a = engine.add_request([1, 2], SamplingParams(max_tokens=2, ignore_eos=True))
    b = engine.add_request([3, 4], SamplingParams(max_tokens=10, ignore_eos=True))
    c = engine.add_request([5, 6], SamplingParams(max_tokens=2, ignore_eos=True))
    while engine.has_unfinished():
        engine.step()
    assert c.first_token_time < b.finish_time


def test_prefix_cache_hits_shared_system_prompt(tiny_model):
    rng = random.Random(1)
    system = [rng.randrange(256) for _ in range(40)]
    prompts = [system + [rng.randrange(256) for _ in range(5)] for _ in range(4)]
    refs = [generate_dense(tiny_model, p, 4) for p in prompts]
    engine = LLMEngine(tiny_model, EngineConfig(block_size=8, num_blocks=100, enable_prefix_caching=True,
                                                 max_num_seqs=1, max_model_len=256))
    seqs = engine.generate(prompts, SamplingParams(max_tokens=4, ignore_eos=True))
    assert [s.output_token_ids for s in seqs] == refs
    assert seqs[0].num_cached_prompt_tokens == 0
    assert all(s.num_cached_prompt_tokens == 40 for s in seqs[1:])


def test_eos_stops_generation(tiny_model):
    engine = LLMEngine(tiny_model, EngineConfig(block_size=4, num_blocks=50, max_model_len=256))
    prompt = [7, 8, 9]
    first = generate_dense(tiny_model, prompt, 1)[0]
    seq = engine.generate([prompt], SamplingParams(max_tokens=10, eos_token_id=first))[0]
    assert seq.output_token_ids == [first] and seq.finish_reason == "stop"


def test_request_validation(tiny_model):
    engine = LLMEngine(tiny_model, EngineConfig(block_size=4, num_blocks=4, max_model_len=256))
    with pytest.raises(ValueError):
        engine.add_request([], SamplingParams())
    with pytest.raises(ValueError):
        engine.add_request([1] * 10, SamplingParams(max_tokens=10))  # 20 tokens > 16 slots
    with pytest.raises(ValueError):
        engine.add_request([999], SamplingParams())
    with pytest.raises(ValueError):
        engine.add_request([1] * 250, SamplingParams(max_tokens=10))  # > max_model_len


def test_seeded_sampling_is_batch_independent(tiny_model, prompts):
    params = SamplingParams(max_tokens=8, temperature=0.9, top_k=50, top_p=0.9, seed=123, ignore_eos=True)
    alone = LLMEngine(tiny_model, EngineConfig(num_blocks=64, max_model_len=256)).generate([prompts[3]], params)[0]
    together = LLMEngine(tiny_model, EngineConfig(num_blocks=64, max_model_len=256)).generate(prompts, params)[3]
    assert alone.output_token_ids == together.output_token_ids


def test_top_p_and_top_k_restrict_support():
    s = Sampler()
    logits = torch.tensor([5.0, 4.0, -10.0, -10.0])
    seen = {s.sample(logits, SamplingParams(temperature=1.0, top_p=0.5), 0) for _ in range(50)}
    assert seen == {0}
    seen = {s.sample(logits, SamplingParams(temperature=5.0, top_k=2), 0) for _ in range(200)}
    assert seen <= {0, 1}


def test_stats_and_kv_utilization(tiny_model, prompts):
    engine = LLMEngine(tiny_model, EngineConfig(block_size=4, num_blocks=200, max_model_len=256))
    engine.generate(prompts, GREEDY)
    st = engine.stats
    assert st.generated_tokens == len(prompts) * MAX_TOKENS
    assert st.prefill_tokens == sum(map(len, prompts))
    assert all(0 <= u <= 1 for u in st.kv_util_samples)
