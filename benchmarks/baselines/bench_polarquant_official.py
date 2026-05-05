"""Bench PolarQuant's official Triton decode-score kernel across (B, N) grid.

Matches the structure of bench_quarot_official.py / bench_kivi_official.py
so the resulting JSON feeds the same ``tab:gold-baselines`` / ``app:compressed-index-baselines``
comparison. PolarQuant's public kernel is score-only
(Q-K inner product on polar-quantized K); its V pass is not part of the
released Triton kernel, so this measurement matches QJL's score-kernel-only
comparison.

Shape parameterization follows PolarQuant's own benchmark_matmul.py:
- B = batch
- N = 32 (H_q query heads per layer; GQA repeats K into 32 copies in the public kernel)
- D = 64 (half of head_dim=128; polar transform doubles last dim)
- group_size = 32
- L = context length (Nb = L // group_size blocks)
- rbits = tbits = 4 (4-bit per polar coordinate)

Usage:
    python experiments/bench_polarquant_official.py \
        --Bs 1,16,32,64 --Ls 4096,16384,32768,65536,131072 \
        --output runs/bench_polarquant_official.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
from statistics import median

sys.path.insert(0, "<HOME>/ing/research/REF/PolarQuant")

import torch
from models.kernel4group import attention_decode_forward_triton_impl  # noqa: E402


def bench(B: int, L: int, N: int = 32, D: int = 64, group_size: int = 32,
          rbits: int = 4, tbits: int = 4, warmup: int = 20, iters: int = 50) -> float:
    assert L % group_size == 0, f"L={L} must be divisible by group_size={group_size}"
    device = torch.device("cuda:0")
    torch.manual_seed(0xCDA + B + L)

    Nb = L // group_size
    query_states = torch.randn(B, N, 1, 2 * D, dtype=torch.float16, device=device)
    indices = torch.randint(0, 250, (B, N, Nb, group_size, D),
                             dtype=torch.uint8, device=device)
    rscale = torch.randn(B, N, Nb, 1, D, dtype=torch.float16, device=device)
    rmn    = torch.randn(B, N, Nb, 1, D, dtype=torch.float16, device=device)
    tscale = torch.randn(B, N, Nb, 1, D, dtype=torch.float16, device=device)
    tmn    = torch.randn(B, N, Nb, 1, D, dtype=torch.float16, device=device)

    for _ in range(warmup):
        attention_decode_forward_triton_impl(
            query_states, indices, rscale, rmn, tscale, tmn,
            rbits=rbits, tbits=tbits,
        )
    torch.cuda.synchronize()

    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        attention_decode_forward_triton_impl(
            query_states, indices, rscale, rmn, tscale, tmn,
            rbits=rbits, tbits=tbits,
        )
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)  # ms -> us
    return median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Bs", default="1,16,32,64")
    ap.add_argument("--Ls", default="4096,16384,32768,65536,131072")
    ap.add_argument("--output", default="runs/bench_polarquant_official.json")
    args = ap.parse_args()

    Bs = [int(x) for x in args.Bs.split(",")]
    Ls = [int(x) for x in args.Ls.split(",")]

    results = []
    print(f"{'B':>4} {'L':>7} {'PolarQuant us':>15}")
    print("-" * 30)
    for B in Bs:
        for L in Ls:
            try:
                us = bench(B, L)
                print(f"{B:>4d} {L:>7d} {us:>15.1f}")
                results.append({"B": B, "L": L, "us": us, "ok": True})
            except Exception as exc:
                msg = str(exc)[:180]
                print(f"{B:>4d} {L:>7d}  FAIL: {msg[:60]}")
                results.append({"B": B, "L": L, "ok": False, "error": msg})
            # always save running results
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
