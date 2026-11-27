"""Charts from the saved JSON reports."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"static_reserve": "#b5523b", "continuous_reserve": "#d9a441", "continuous_paged": "#2f6f73"}
LABELS = {"static_reserve": "static batching + reserved KV",
          "continuous_reserve": "continuous batching + reserved KV",
          "continuous_paged": "continuous batching + paged KV"}


def _load(d: Path, name: str):
    return json.loads((d / f"{name}.json").read_text())


def make_plots(d: Path) -> None:
    off = _load(d, "offline_throughput")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    names = list(off)
    for ax, key, title in zip(axes, ["output_tok_s", "mean_batch_size", "mean_kv_util"],
                              ["output tokens / s", "mean running batch", "mean KV slot utilisation"]):
        vals = [off[n][key] for n in names]
        ax.bar(range(len(names)), vals, color=[COLORS[n] for n in names])
        ax.set_xticks(range(len(names)), ["static\nreserved", "continuous\nreserved", "continuous\npaged"])
        ax.set_title(title)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:g}", ha="center", va="bottom", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Offline throughput, {off[names[0]]['requests']} Dolly requests, GPT-2 124M on CPU, 6144-token KV budget")
    fig.tight_layout()
    fig.savefig(d / "offline_throughput.png", dpi=130)
    plt.close(fig)

    onl = _load(d, "online_latency")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, key, title in zip(axes, ["ttft_p50_s", "ttft_p99_s", "e2e_p50_s"],
                              ["TTFT p50 (s)", "TTFT p99 (s)", "end-to-end latency p50 (s)"]):
        for n, runs in onl.items():
            ax.plot([r["rate_req_s"] for r in runs], [r[key] for r in runs], "o-", color=COLORS[n], label=LABELS[n])
        ax.set_xlabel("arrival rate (req/s)")
        ax.set_title(title)
        ax.set_yscale("log")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(d / "online_latency.png", dpi=130)
    plt.close(fig)
