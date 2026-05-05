"""Generate tab:memory LaTeX (KV/req, max concurrent, throughput) from cap sweeps.

Reads cap_sweep_N*.json artifacts and emits a paper-ready LaTeX table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _N_label(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n // (1024*1024)}M"
    if n >= 1000:
        return f"{n // 1024}K"
    return str(n)


def _kv_bytes_per_token(layers: int, h_kv: int, head_dim: int,
                        bytes_per_elem: float, kv_count: int = 2) -> int:
    return layers * h_kv * head_dim * bytes_per_elem * kv_count


def _kv_bytes_per_token_cda(layers: int, h_kv: int, slot_w: int) -> int:
    return layers * h_kv * slot_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifacts", nargs="+",
                    help="cap_sweep_N*.json files")
    ap.add_argument("--model-layers", type=int, default=32,
                    help="Llama-3.1-8B = 32")
    ap.add_argument("--h-kv", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--cda-slot-bytes", type=int, default=144,
                    help="K4V4 = 144 B (with 16B alignment padding)")
    args = ap.parse_args()

    fa2_kv_bytes = _kv_bytes_per_token(
        args.model_layers, args.h_kv, args.head_dim, 2)  # fp16
    cda_kv_bytes = _kv_bytes_per_token_cda(
        args.model_layers, args.h_kv, args.cda_slot_bytes)
    print(f"% Per-token KV bytes: FA2={fa2_kv_bytes:,} CDA={cda_kv_bytes:,} "
          f"ratio={fa2_kv_bytes/cda_kv_bytes:.2f}x")
    print()

    # Header
    print(r"\begin{tabular}{@{}rrrrrrrr@{}}")
    print(r"\toprule")
    print(r"$\nctx$ & FA2 KV/req & FA2 max $B$ & FA2 tok/s & "
          r"\ours{} KV/req & \ours{} max $B$ & \ours{} tok/s & $\Delta$ \\")
    print(r"\midrule")

    # Aggregate per-N across multiple files (fold ext/ext2/ext3/anomredo)
    by_N: dict = {}
    for path in args.artifacts:
        d = json.loads(Path(path).read_text())
        N = d["config"]["prompt_len"]
        bag = by_N.setdefault(N, {"FA2": [], "CDA": []})
        for backend in ("FA2", "CDA"):
            for r in d.get(backend, []):
                if r.get("status") == "ok":
                    bag[backend].append(r)

    rows = []
    for N, bag in sorted(by_N.items()):
        fa2_kv_gb = fa2_kv_bytes * N / (1024**3)
        cda_kv_gb = cda_kv_bytes * N / (1024**3)
        fa2_max_b = max((r["batch"] for r in bag["FA2"]), default=0)
        cda_max_b = max((r["batch"] for r in bag["CDA"]), default=0)
        fa2_max_tps = max((r["throughput_tok_s"] for r in bag["FA2"]), default=0)
        cda_max_tps = max((r["throughput_tok_s"] for r in bag["CDA"]), default=0)
        rows.append((N, fa2_kv_gb, fa2_max_b, fa2_max_tps,
                     cda_kv_gb, cda_max_b, cda_max_tps))
    for N, fa2_kv, fa2_b, fa2_tps, cda_kv, cda_b, cda_tps in rows:
        delta = (cda_tps - fa2_tps) / fa2_tps * 100 if fa2_tps else 0
        print(f"  {_N_label(N)} & "
              f"{fa2_kv:.2f}\\,GB & {fa2_b} & {fa2_tps:.0f} & "
              f"{cda_kv:.2f}\\,GB & {cda_b} & {cda_tps:.0f} & "
              f"$\\mathbf{{{delta:+.1f}\\%}}$ \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
