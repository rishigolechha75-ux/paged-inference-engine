import math

from bench.harness import run
from bench.run_benchmarks import fragmentation
from bench.workload import Request
from pagedserve.config import EngineConfig


def _reqs():
    return [Request(list(range(1, 1 + n)), out) for n, out in [(5, 3), (20, 10), (9, 1), (33, 7)]]


def test_harness_offline_and_poisson(tiny_model):
    cfg = EngineConfig(block_size=4, num_blocks=64, max_model_len=256)
    off = run(tiny_model, cfg, _reqs(), rate=math.inf, max_tokens=16)
    assert off["requests"] == 4 and off["output_tokens"] == 3 + 10 + 1 + 7
    assert off["ttft_p50_s"] >= 0 and off["preemptions"] == 0
    onl = run(tiny_model, cfg, _reqs(), rate=200.0, max_tokens=16)
    assert onl["requests"] == 4 and onl["rate_req_s"] == 200.0


def test_reserving_engine_has_lower_kv_utilisation(tiny_model):
    paged = run(tiny_model, EngineConfig(block_size=4, num_blocks=64, max_model_len=256), _reqs(), max_tokens=40)
    reserved = run(tiny_model, EngineConfig(block_size=4, num_blocks=64, reserve_max=True, max_model_len=256),
                   _reqs(), max_tokens=40)
    assert paged["mean_kv_util"] > reserved["mean_kv_util"]


def test_fragmentation_is_exact():
    frag = fragmentation([Request([1] * 10, 6)], block_sizes=(1, 16))
    assert frag["paged_block_1"]["waste_pct"] == 0.0
    assert frag["paged_block_16"] == {"allocated": 16, "used": 16, "waste_pct": 0.0}
    assert frag["reserve_max_tokens"]["allocated"] == 10 + 256
