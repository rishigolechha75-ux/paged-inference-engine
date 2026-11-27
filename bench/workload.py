"""Request workloads built from the databricks-dolly-15k dataset.

Source: https://huggingface.co/datasets/databricks/databricks-dolly-15k
License: CC BY-SA 3.0. The file is downloaded at runtime into ``data/``
and never committed.
"""

from __future__ import annotations

import json
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List

DOLLY_URL = ("https://huggingface.co/datasets/databricks/databricks-dolly-15k/"
             "resolve/main/databricks-dolly-15k.jsonl")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Request:
    prompt_ids: List[int]
    output_len: int  # real response length in tokens (replayed via stop_after)


def load_dolly() -> List[dict]:
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / "databricks-dolly-15k.jsonl"
    if not path.exists():
        urllib.request.urlretrieve(DOLLY_URL, path)
    with path.open() as f:
        return [json.loads(line) for line in f]


def _prompt(row: dict) -> str:
    text = row["instruction"].strip()
    if row.get("context"):
        text += "\n\n" + row["context"].strip()
    return text + "\n\nResponse:"


def dolly_requests(tokenizer, n: int, seed: int = 0, max_prompt: int = 512,
                   max_output: int = 256, min_output: int = 4) -> List[Request]:
    """Sample ``n`` requests with real prompt and response lengths (clipped)."""
    rows = load_dolly()
    rng = random.Random(seed)
    rng.shuffle(rows)
    out: List[Request] = []
    for row in rows:
        prompt = tokenizer.encode(_prompt(row))
        resp = len(tokenizer.encode(row["response"]))
        if not prompt or len(prompt) > max_prompt:
            continue
        out.append(Request(prompt, min(max(resp, min_output), max_output)))
        if len(out) == n:
            break
    return out


def shared_prefix_requests(tokenizer, n: int, shots: int = 8, output_len: int = 32,
                           seed: int = 1) -> List[Request]:
    """Few-shot workload: every request starts with the same ``shots`` worked
    examples (a long shared prefix) followed by a distinct short question."""
    rows = [r for r in load_dolly() if not r.get("context") and len(r["response"]) < 600]
    rng = random.Random(seed)
    rng.shuffle(rows)
    prefix = "".join(f"Instruction: {r['instruction'].strip()}\nResponse: {r['response'].strip()}\n\n"
                     for r in rows[:shots])
    reqs = []
    for r in rows[shots:shots + n]:
        reqs.append(Request(tokenizer.encode(prefix + f"Instruction: {r['instruction'].strip()}\nResponse:"),
                            output_len))
    return reqs
