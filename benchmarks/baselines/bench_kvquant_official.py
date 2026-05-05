"""Official KVQuant kernel bench — Q@K (4-bit) + softmax + P@V (4-bit).

Uses KVQuant's `quant_cuda` extension (built from
REF/KVQuant/deployment/kvquant/). Their released kernel is MHA-only
with B=1 hardcoded (kcache/vcache have no batch dim), so we report B=1
across N. This matches their paper's kernel claims.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from statistics import median

import torch

sys.path.insert(0, "<HOME>/ing/research/REF/KVQuant/deployment/kvquant")
import quant_cuda


def bench_kvquant(N, num_heads=32, head_dim=128, bits=4, rope_theta=10000.0,
                   warmup=10, iters=40):
    device = torch.device("cuda:0")
    torch.manual_seed(0xCDA + N)

    # kcache / vcache: [num_heads, (head_dim/32)*bits, N] int32.
    int_per_head = (head_dim // 32) * bits  # 4-bit + head_dim=128 → 16
    kcache = torch.randint(-(2**30), 2**30, (num_heads, int_per_head, N),
                            dtype=torch.int32, device=device)
    vcache = torch.randint(-(2**30), 2**30, (num_heads, int_per_head, N),
                            dtype=torch.int32, device=device)

    # K-path LUT: [num_heads, head_dim, 2^bits] fp32 per-channel.
    k_lut = (torch.rand(num_heads, head_dim, 2 ** bits,
                           dtype=torch.float32, device=device) - 0.5) * 2.0
    # V-path LUT: [N, 2^bits] fp32 per-token.
    v_lut = (torch.rand(N, 2 ** bits,
                           dtype=torch.float32, device=device) - 0.5) * 2.0

    # Kernel reads mul.size(0) as batch_size and loops over it internally;
    # layout expected is (batch, num_heads, ...) flat.
    q = torch.randn(1, num_heads, head_dim, dtype=torch.float32, device=device)
    score_out = torch.zeros(1, num_heads, N, dtype=torch.float32, device=device)
    mul_out = torch.zeros(1, num_heads, head_dim, dtype=torch.float32, device=device)

    def call():
        # Q @ K (4-bit, non-sparse, RoPE-fused).
        quant_cuda.vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt(
            q, kcache, score_out, k_lut, N, rope_theta, 0
        )
        # fp16 softmax (match KVQuant's own model code).
        p = torch.softmax(score_out.half().float(), dim=-1).contiguous()
        # P @ V (4-bit, non-sparse).
        quant_cuda.vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt(
            p, vcache, mul_out, v_lut, N
        )
        return mul_out

    for _ in range(warmup): call()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); call(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    return median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", default="4096,16384,32768,65536")
    ap.add_argument("--output", default="runs/bench_kvquant_official.json")
    args = ap.parse_args()
    Ns = [int(x) for x in args.Ns.split(",")]

    print(f"{'N':>6}  {'KVQuant μs':>11}")
    print("-" * 22)
    results = []
    for N in Ns:
        try:
            us = bench_kvquant(N)
            print(f"{N:>6d}  {us:>11.1f}")
            results.append({"B": 1, "N": N, "kvquant_us": round(us, 1)})
        except Exception as e:
            print(f"{N:>6d}  FAILED: {str(e)[:120]}")
            results.append({"B": 1, "N": N, "error": str(e)[:200]})
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
