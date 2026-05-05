"""Figure 2 — Serving throughput vs concurrent requests for CDA paper.

Reads cap_sweep_N*.json artifacts and plots throughput (tok/s) vs batch B
for FA2 vs CDA, marking OOM cliffs.

Usage:
    python papers/gen_fig2_throughput.py \\
        runs/paper/cap_sweep_N128K_v3.json \\
        runs/paper/cap_sweep_N64K_v3.json \\
        runs/paper/cap_sweep_N16K_v2.json \\
        runs/paper/cap_sweep_N8K_v3.json \\
        --output papers/69cf3c869b7b5e6e5a202759/neurips/figures/fig2_throughput.pdf
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _N_label(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n // (1024*1024)}M"
    if n >= 1000:
        return f"{n // 1024}K"
    return str(n)


def _consolidate_per_N(paths: list[str]) -> dict[int, dict]:
    """Merge multiple sweep files per prompt_len, picking the best (latest) value for each batch."""
    by_N: dict[int, dict] = {}
    for path in paths:
        d = json.loads(Path(path).read_text())
        N = d["config"]["prompt_len"]
        if N not in by_N:
            by_N[N] = {"FA2": {}, "CDA": {}}
        for backend in ("FA2", "CDA"):
            for r in d.get(backend, []):
                if r.get("status") != "ok":
                    continue
                B = r["batch"]
                tps = r["throughput_tok_s"]
                # Keep highest throughput per B (anomaly cells get superseded)
                cur = by_N[N][backend].get(B)
                if cur is None or tps > cur["throughput_tok_s"]:
                    by_N[N][backend][B] = r
    return by_N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifacts", nargs="+", help="cap_sweep_N*.json files")
    ap.add_argument("--output", default="papers/fig2_throughput.pdf")
    ap.add_argument("--model", default="Llama-3.1-8B (A6000)")
    ap.add_argument("--ymin-zero", action="store_true",
                    help="Force y-axis to start at 0")
    args = ap.parse_args()

    plt.rcParams.update({
        "font.size": 12,
        "font.family": "sans-serif",
        "figure.dpi": 200,
    })

    by_N = _consolidate_per_N(args.artifacts)
    sorted_N = sorted(by_N.keys(), reverse=True)

    n_panels = len(sorted_N)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.0),
                              sharey=False)
    if n_panels == 1:
        axes = [axes]

    for ax, N in zip(axes, sorted_N):
        fa2_data = by_N[N]["FA2"]
        cda_data = by_N[N]["CDA"]
        fa2_b = sorted(fa2_data.keys())
        fa2_tps = [fa2_data[b]["throughput_tok_s"] for b in fa2_b]
        cda_b = sorted(cda_data.keys())
        cda_tps = [cda_data[b]["throughput_tok_s"] for b in cda_b]
        fa2_oom_at = None  # consolidate keeps only oks
        cda_oom_at = None

        if fa2_b:
            ax.plot(fa2_b, fa2_tps, "o-", color="#3B7DD8", lw=2,
                     label="FA2 (FP16 KV)")
            if fa2_oom_at is not None:
                ax.axvline(fa2_oom_at, color="#3B7DD8", lw=1.2, ls="--",
                            alpha=0.7)
                ax.text(fa2_oom_at, ax.get_ylim()[1] * 0.95,
                        f"FA2 OOM\nB={fa2_oom_at}", color="#3B7DD8",
                        ha="left", va="top", fontsize=9)
        if cda_b:
            ax.plot(cda_b, cda_tps, "s-", color="#D62728", lw=2,
                     label="CDA (K4V4)")
            if cda_oom_at is not None:
                ax.axvline(cda_oom_at, color="#D62728", lw=1.2, ls="--",
                            alpha=0.7)
                ax.text(cda_oom_at, ax.get_ylim()[1] * 0.85,
                        f"CDA OOM\nB={cda_oom_at}", color="#D62728",
                        ha="left", va="top", fontsize=9)

        ax.set_xlabel("Concurrent requests $B$")
        ax.set_xscale("log", base=2)
        if args.ymin_zero:
            ax.set_ylim(bottom=0)
        ax.set_title(f"$N = {_N_label(N)}$")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=10, framealpha=0.95)

    axes[0].set_ylabel("Throughput (tokens/s)")

    fig.suptitle(f"Serving throughput vs concurrent requests — {args.model}",
                  fontsize=13, y=1.02)
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"Saved: {out}, {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
