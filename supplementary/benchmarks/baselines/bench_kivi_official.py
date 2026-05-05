"""Official KIVI kernel bench — score + softmax + value with int2/int4 gemv.

Uses KIVI's `kivi_gemv.gemv_forward_cuda_outer_dim` (built from
REF/KIVI/quant/). Unlike QuaRot and KVQuant, KIVI's kernel supports
GQA (nh_q / nh_kv separate), so we bench at Llama-3.1-8B shape
(H_q=32, H_kv=8, D=128) directly — same shape as CDA.

KIVI stores K quant-along-outer-dim (tokens) and V quant-along-last-dim
(channels). Both use the same gemv kernel with transposed layouts.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from statistics import median

import torch

sys.path.insert(0, "<HOME>/ing/research/REF/KIVI")
sys.path.insert(0, "<HOME>/ing/research/REF/KIVI/quant")
import kivi_gemv
from quant.matmul import cuda_bmm_fA_qB_outer

GROUP_SIZE = 32


def bench_kivi(B, N, H_q=32, H_kv=8, D=128, k_bits=2, v_bits=2,
               warmup=10, iters=40):
    device = torch.device("cuda:0")
    torch.manual_seed(0xCDA + N + B)

    feat_per_int_k = 32 // k_bits
    feat_per_int_v = 32 // v_bits

    # Q: (B, H_q, 1, D) fp16
    q = torch.randn(B, H_q, 1, D, dtype=torch.float16, device=device) * 0.3

    # K_quant (transposed): (B, H_kv, D, N/feat_per_int_k) int32.
    # scales/zeros: (B, H_kv, D, N/GROUP_SIZE) fp16.
    k_pack_N = N // feat_per_int_k
    k_groups = N // GROUP_SIZE
    k_quant = torch.randint(0, 2**30, (B, H_kv, D, k_pack_N),
                              dtype=torch.int32, device=device)
    k_scale = torch.randn(B, H_kv, D, k_groups, dtype=torch.float16, device=device).abs() + 1e-3
    k_zero  = torch.zeros(B, H_kv, D, k_groups, dtype=torch.float16, device=device)

    # V_quant: (B, H_kv, N, D/feat_per_int_v) int32.
    # scales/zeros: (B, H_kv, N, D/GROUP_SIZE) fp16.
    v_pack_D = D // feat_per_int_v
    v_groups_D = D // GROUP_SIZE
    v_quant = torch.randint(0, 2**30, (B, H_kv, N, v_pack_D),
                              dtype=torch.int32, device=device)
    v_scale = torch.randn(B, H_kv, N, v_groups_D, dtype=torch.float16, device=device).abs() + 1e-3
    v_zero  = torch.zeros(B, H_kv, N, v_groups_D, dtype=torch.float16, device=device)

    def call():
        scores = cuda_bmm_fA_qB_outer(GROUP_SIZE, q, k_quant, k_scale, k_zero, k_bits)
        # scores: (B, H_q, 1, N) fp16
        # softmax (fp32 upcast matching KIVI model code)
        scale = 1.0 / (D ** 0.5)
        p = torch.softmax((scores * scale).float(), dim=-1).to(torch.float16)
        # Value: P @ V_quant → (B, H_q, 1, D)
        out = cuda_bmm_fA_qB_outer(GROUP_SIZE, p, v_quant, v_scale, v_zero, v_bits)
        return out

    # Warmup (includes first-call wrapper import)
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
    ap.add_argument("--Bs", default="1,4,16,32")
    ap.add_argument("--Ns", default="4096,16384,32768,65536")
    ap.add_argument("--k_bits", type=int, default=2)
    ap.add_argument("--v_bits", type=int, default=2)
    ap.add_argument("--output", default="runs/bench_kivi_official.json")
    args = ap.parse_args()
    Bs = [int(x) for x in args.Bs.split(",")]
    Ns = [int(x) for x in args.Ns.split(",")]

    print(f"KIVI K{args.k_bits}V{args.v_bits}, group_size={GROUP_SIZE}, H_q=32 H_kv=8 D=128")
    print(f"{'B':>3} {'N':>6}  {'KIVI μs':>10}")
    print("-" * 25)
    results = []
    for B in Bs:
        for N in Ns:
            try:
                us = bench_kivi(B, N, k_bits=args.k_bits, v_bits=args.v_bits)
                print(f"{B:>3d} {N:>6d}  {us:>10.1f}")
                results.append({"B": B, "N": N,
                                "k_bits": args.k_bits, "v_bits": args.v_bits,
                                "kivi_us": round(us, 1)})
            except Exception as e:
                print(f"{B:>3d} {N:>6d}  FAILED: {str(e)[:120]}")
                results.append({"B": B, "N": N, "error": str(e)[:200]})
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
