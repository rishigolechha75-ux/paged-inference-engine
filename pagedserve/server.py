"""OpenAI-style completions API with token streaming and Prometheus metrics.

A single background thread owns the engine and calls ``step()`` in a loop.
HTTP handlers never touch the model: they enqueue requests under a lock and
receive tokens through per-request ``asyncio.Queue`` objects, which the
engine thread fills with ``loop.call_soon_threadsafe``.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from .engine import LLMEngine
from .sequence import SamplingParams
from .tokenizer import Tokenizer


class CompletionRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(default=32, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    seed: Optional[int] = None
    stream: bool = False
    ignore_eos: bool = False


_DONE = object()


class AsyncEngine:
    """Runs :class:`LLMEngine` on a background thread."""

    def __init__(self, engine: LLMEngine, idle_sleep_s: float = 0.002) -> None:
        self.engine = engine
        self.lock = threading.Lock()
        self.idle_sleep_s = idle_sleep_s
        self._streams: Dict[int, Tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.started_at = time.time()

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="engine", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def submit(self, prompt_ids: List[int], params: SamplingParams) -> Tuple[int, asyncio.Queue]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        with self.lock:
            seq = self.engine.add_request(prompt_ids, params)
            self._streams[seq.seq_id] = (loop, queue)
        return seq.seq_id, queue

    def _run(self) -> None:
        while not self._stop.is_set():
            with self.lock:
                busy = self.engine.has_unfinished()
                out = self.engine.step() if busy else None
            if out is None:
                time.sleep(self.idle_sleep_s)
                continue
            for seq_id, token in out.new_tokens.items():
                self._push(seq_id, token)
            for seq in out.finished:
                self._push(seq.seq_id, _DONE)
                self._streams.pop(seq.seq_id, None)

    def _push(self, seq_id: int, item: object) -> None:
        target = self._streams.get(seq_id)
        if target is not None:
            loop, queue = target
            loop.call_soon_threadsafe(queue.put_nowait, item)


def render_metrics(engine: LLMEngine) -> str:
    """Prometheus text exposition of engine counters and gauges."""
    bm = engine.block_manager
    st = engine.stats
    lines = [
        ("pagedserve_requests_running", "gauge", len(engine.scheduler.running)),
        ("pagedserve_requests_waiting", "gauge", len(engine.scheduler.waiting)),
        ("pagedserve_kv_blocks_used", "gauge", bm.num_used_blocks),
        ("pagedserve_kv_blocks_total", "gauge", engine.cfg.num_blocks),
        ("pagedserve_kv_block_size_tokens", "gauge", engine.cfg.block_size),
        ("pagedserve_prefix_cache_hit_rate", "gauge", round(bm.prefix_hit_rate, 6)),
        ("pagedserve_preemptions_total", "counter", engine.scheduler.num_preemptions),
        ("pagedserve_steps_total", "counter", st.steps),
        ("pagedserve_generated_tokens_total", "counter", st.generated_tokens),
        ("pagedserve_prefill_tokens_total", "counter", st.prefill_tokens),
        ("pagedserve_busy_seconds_total", "counter", round(st.busy_time_s, 6)),
    ]
    out = []
    for name, kind, value in lines:
        out.append(f"# TYPE {name} {kind}")
        out.append(f"{name} {value}")
    return "\n".join(out) + "\n"


def create_app(engine: LLMEngine, tokenizer: Tokenizer, model_name: str = "gpt2") -> FastAPI:
    runner = AsyncEngine(engine)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        runner.start()
        yield
        runner.stop()

    app = FastAPI(title="pagedserve", version="0.1.0", lifespan=lifespan)
    app.state.runner = runner

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": model_name}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        with runner.lock:
            return render_metrics(engine)

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        prompt_ids = tokenizer.encode(req.prompt)
        params = SamplingParams(max_tokens=req.max_tokens, temperature=req.temperature, top_p=req.top_p,
                                top_k=req.top_k, seed=req.seed, ignore_eos=req.ignore_eos,
                                eos_token_id=tokenizer.eos_token_id)
        try:
            seq_id, queue = runner.submit(prompt_ids, params)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        created = int(time.time())

        if req.stream:
            async def events():
                while True:
                    item = await queue.get()
                    if item is _DONE:
                        seq = engine.sequences.pop(seq_id)
                        chunk = {"id": f"cmpl-{seq_id}", "object": "text_completion", "model": model_name,
                                 "choices": [{"index": 0, "text": "", "finish_reason": seq.finish_reason}]}
                        yield f"data: {json.dumps(chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    chunk = {"id": f"cmpl-{seq_id}", "object": "text_completion", "created": created,
                             "model": model_name,
                             "choices": [{"index": 0, "text": tokenizer.decode([item]), "finish_reason": None}]}
                    yield f"data: {json.dumps(chunk)}\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")

        tokens: List[int] = []
        while True:
            item = await queue.get()
            if item is _DONE:
                break
            tokens.append(item)
        seq = engine.sequences.pop(seq_id)
        ttft = (seq.first_token_time - seq.arrival_time) if seq.first_token_time else None
        return {
            "id": f"cmpl-{seq_id}",
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "text": tokenizer.decode(tokens), "finish_reason": seq.finish_reason}],
            "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(tokens),
                      "total_tokens": len(prompt_ids) + len(tokens),
                      "cached_prompt_tokens": seq.num_cached_prompt_tokens},
            "timing": {"ttft_s": ttft, "e2e_s": (seq.finish_time or time.perf_counter()) - seq.arrival_time},
        }

    return app
