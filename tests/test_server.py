import json

import pytest
from fastapi.testclient import TestClient

from pagedserve.config import EngineConfig
from pagedserve.engine import LLMEngine, generate_dense
from pagedserve.server import create_app
from pagedserve.tokenizer import ByteTokenizer


@pytest.fixture()
def client(tiny_model):
    engine = LLMEngine(tiny_model, EngineConfig(block_size=4, num_blocks=128, max_model_len=256))
    with TestClient(create_app(engine, ByteTokenizer(), model_name="tiny")) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model": "tiny"}


def test_completion_matches_reference(client, tiny_model):
    r = client.post("/v1/completions", json={"prompt": "hello", "max_tokens": 6, "ignore_eos": True})
    assert r.status_code == 200
    body = r.json()
    ref = generate_dense(tiny_model, list(b"hello"), 6)
    assert body["choices"][0]["text"] == ByteTokenizer().decode(ref)
    assert body["usage"] == {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11,
                             "cached_prompt_tokens": 0}
    assert body["choices"][0]["finish_reason"] == "length"


def test_streaming_emits_one_event_per_token(client):
    with client.stream("POST", "/v1/completions",
                       json={"prompt": "abc", "max_tokens": 5, "stream": True, "ignore_eos": True}) as r:
        lines = [ln for ln in r.iter_lines() if ln.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(ln[6:]) for ln in lines[:-1]]
    assert len(chunks) == 6  # 5 tokens + final finish_reason chunk
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"


def test_concurrent_requests_and_metrics(client):
    import concurrent.futures as cf

    def call(i):
        return client.post("/v1/completions", json={"prompt": f"req {i}", "max_tokens": 8, "ignore_eos": True})

    with cf.ThreadPoolExecutor(8) as pool:
        results = list(pool.map(call, range(8)))
    assert all(r.status_code == 200 for r in results)
    text = client.get("/metrics").text
    assert "pagedserve_generated_tokens_total 64" in text
    assert "pagedserve_kv_blocks_used 0" in text


def test_invalid_requests(client):
    assert client.post("/v1/completions", json={"prompt": "", "max_tokens": 4}).status_code == 422
    assert client.post("/v1/completions", json={"prompt": "x", "max_tokens": 0}).status_code == 422
    r = client.post("/v1/completions", json={"prompt": "x" * 250, "max_tokens": 50})
    assert r.status_code == 400
