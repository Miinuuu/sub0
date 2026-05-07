"""Figure 2 — Serving throughput vs concurrent requests for CDA paper.

Reads cap_sweep_N*.json artifacts and plots throughput (tok/s) vs batch B
for FA2, dequant+FA2 (Class (a)), and CDA (Class (d)).

Usage (consolidated layout — see runs/paper/fig2_throughput/README.md):
    python benchmarks/paper/gen_fig2_throughput.py \\
        runs/paper/fig2_throughput/raw/4K/*.json \\
        runs/paper/fig2_throughput/raw/16K/*.json \\
        runs/paper/fig2_throughput/raw/64K/*.json \\
        runs/paper/fig2_throughput/raw/128K/*.json \\
        --output papers/69cf3c869b7b5e6e5a202759/neurips/figures/fig2_throughput.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_colors import METHOD_COLORS


def _N_label(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n // (1024*1024)}M"
    if n >= 1000:
        return f"{n // 1024}K"
    return str(n)


_BACKENDS = ("FA2", "CDA", "DEQUANT_FA2", "FUSED_DEQUANT_FA2")


def _consolidate_per_N(paths: list[str]) -> dict[int, dict]:
    """Merge multiple sweep files per total context length (prompt+decode), picking the best (latest) value for each batch."""
    by_N: dict[int, dict] = {}
    for path in paths:
        d = json.loads(Path(path).read_text())
        # N = total context length (prompt + decode), matching paper convention
        # Nctx in {4, 16, 64, 128}K
        N = d["config"]["prompt_len"] + d["config"]["decode_len"]
        if N not in by_N:
            by_N[N] = {b: {} for b in _BACKENDS}
        for backend in _BACKENDS:
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
        "font.family": "serif",
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.dpi": 200,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    by_N = _consolidate_per_N(args.artifacts)
    sorted_N = sorted(by_N.keys(), reverse=True)

    n_panels = len(sorted_N)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.0),
                              sharey=False)
    if n_panels == 1:
        axes = [axes]

    backend_style = {
        "FA2":               ("o", "-",  METHOD_COLORS["FA2"], "FA2 (FP16 KV)"),
        "FUSED_DEQUANT_FA2": ("D", "--", METHOD_COLORS["DEQUANT_FA2"],
                              "dequant+FA2 (K4V4 → FP16, Class (a))"),
        "CDA":               ("s", "-",  METHOD_COLORS["CDA"], "CDA (K4V4, Class (d))"),
    }

    for ax, N in zip(axes, sorted_N):
        for backend, (mk, ls, color, label) in backend_style.items():
            data = by_N[N].get(backend, {})
            if not data:
                continue
            bs = sorted(data.keys())
            # Decode-only throughput: (B * decode_len) / total_wall_s.
            # Matches Table A3 convention (excludes prefill tokens) so
            # paper text and figure cite identical numbers per cell.
            tps = []
            for b in bs:
                r = data[b]
                D = r["decode_len"]
                wall = r["total_wall_s"]
                tps.append((b * D) / wall)
            ax.plot(bs, tps, marker=mk, linestyle=ls, color=color, lw=2,
                    label=label)

        ax.set_xlabel("Concurrent requests $B$")
        ax.set_xscale("log", base=2)
        if args.ymin_zero:
            ax.set_ylim(bottom=0)
        ax.set_title(f"$N = {_N_label(N)}$")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Decode-only throughput (tokens/s)")

    legend_handles = [
        Line2D([0], [0], marker=mk, linestyle=ls, color=color, lw=2,
               markersize=6, label=label)
        for mk, ls, color, label in backend_style.values()
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=len(legend_handles),
               frameon=False, handlelength=2.4, columnspacing=1.2)

    # fig.suptitle(f"Serving throughput vs concurrent requests — {args.model}",
    #               fontsize=13, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"Saved: {out}, {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
