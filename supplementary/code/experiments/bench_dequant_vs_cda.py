"""CDA direct HMMA vs dequantize-then-attend (FA2) — kernel-level comparison.

Both paths store K/V in the SAME 4-bit K + 2-bit V + fp32 norm layout (so
memory footprint is identical). The only difference:

  * CDA path     — cda_decode_full_hmma reads compressed indices + norms
                    directly and synthesises Tensor-Core operand fragments
                    in registers. No fp16 K/V matrix is ever materialised.

  * Dequant path — explicit HadamardQuantCompressor.dequantize() to fp16 K
                    and fp16 V tensors, then flash_attn_varlen_func.
                    Mimics what KIVI / GEAR / QuaRot / KVQuant pipelines do.

Measures: kernel-level μs per call across (B, N) × head_dim=128, group=4.
This is the apples-to-apples "dequant overhead" isolation.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
from statistics import median

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core.compression import HadamardQuantCompressor, CompressedTensor
from core.cda_attn import cda_decode_full_hmma as jit_cda
import vllm._custom_ops  # triggers torch.ops._C load


def _build_synthetic_cache(B, N, H_kv, D, device):
    """Generate random fp16 K/V, quantise via HadamardQuantCompressor,
    pack into CDA 104 B/slot layout + the same indices/norms separately for
    the dequant path. Returns both representations."""
    ck = HadamardQuantCompressor(dim=D, bit_width=4, half_rotation=True)
    cv = HadamardQuantCompressor(dim=D, bit_width=2, half_rotation=True)
    ck._ensure_tensors(device); cv._ensure_tensors(device)

    block_size = 16
    num_blk = (N + block_size - 1) // block_size
    total_slots = B * num_blk * block_size

    # Fresh random K, V.
    K_fresh = torch.randn(total_slots, H_kv, D, dtype=torch.float16, device=device) * 0.3
    V_fresh = torch.randn(total_slots, H_kv, D, dtype=torch.float16, device=device) * 0.3

    cK = ck.quantize(K_fresh.reshape(-1, D).contiguous())
    cV = cv.quantize(V_fresh.reshape(-1, D).contiguous())

    # Pack into CDA's 104 B/slot layout.
    cache = torch.zeros(total_slots, H_kv, 104, dtype=torch.uint8, device=device)
    cache[:, :, 0:64]   = cK.indices.view(total_slots, H_kv, 64)
    cache[:, :, 64:96]  = cV.indices.view(total_slots, H_kv, 32)
    cache[:, :, 96:100] = (cK.norms.float().view(total_slots, H_kv, 1)
                            .contiguous().view(torch.uint8).view(total_slots, H_kv, 4))
    cache[:, :, 100:104]= (cV.norms.float().view(total_slots, H_kv, 1)
                            .contiguous().view(torch.uint8).view(total_slots, H_kv, 4))

    cb_k = (ck._centroids * 2.0 - 1.0).float().contiguous()
    cb_v = (cv._centroids * 2.0 - 1.0).float().contiguous()
    rot_fp32 = ck._rotation.float().contiguous()
    rot_fp16 = rot_fp32.to(torch.float16).contiguous()

    block_tbl = torch.arange(B * num_blk, dtype=torch.int32,
                               device=device).view(B, num_blk)
    seq_lens = torch.full((B,), N, dtype=torch.int32, device=device)
    return dict(ck=ck, cv=cv, cache=cache, cK=cK, cV=cV,
                 block_tbl=block_tbl, seq_lens=seq_lens,
                 cb_k=cb_k, cb_v=cb_v, rot_fp32=rot_fp32, rot_fp16=rot_fp16,
                 total_slots=total_slots, block_size=block_size, num_blk=num_blk)


def _bench_one(B, N, H_q=32, H_kv=8, D=128, warmup=10, iters=50):
    device = torch.device("cuda:0")
    torch.manual_seed(0xCDA)
    ctx = _build_synthetic_cache(B, N, H_kv, D, device)

    group_size = H_q // H_kv
    tile_N = 1024 if N >= 8192 else 512
    scale = 1.0 / (D ** 0.5)
    max_seq_bound = ctx["num_blk"] * ctx["block_size"]

    q_fp16 = torch.randn(B, H_q, D, dtype=torch.float16, device=device) * 0.3
    out = torch.empty(B, H_q, D, dtype=torch.float16, device=device)

    # ---- Path A: CDA direct ----
    def run_cda():
        torch.ops._C.cda_decode_full_hmma(
            q_fp16, ctx["cache"], ctx["block_tbl"], ctx["seq_lens"],
            ctx["cb_k"], ctx["cb_v"], ctx["rot_fp32"], ctx["rot_fp16"], out,
            group_size, ctx["block_size"], tile_N, max_seq_bound, scale,
        )

    # ---- Path B: Dequantize-then-FA2 ----
    # Dequantise K/V to fp16 at actual seq positions (single request in this
    # microbench; extend trivially to multi-request). Then run FA2
    # flash_attn_varlen_func.
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    # Pre-compute sequential slot ids (single request spans slots 0..N-1
    # block-aligned).
    slot_ids = torch.arange(N, dtype=torch.long, device=device)
    # Reshape once: flat_cache[B*num_blk*block_size, H_kv, 104]. For the
    # synthetic data we generated, slots 0..N-1 are the first sequence's
    # cached K/V (block_tbl[0] = [0..num_blk-1]).

    def dequant_kv():
        K_flat = ctx["ck"].dequantize(ctx["cK"]).reshape(-1, H_kv, D)[:N]
        V_flat = ctx["cv"].dequantize(ctx["cV"]).reshape(-1, H_kv, D)[:N]
        return K_flat.to(torch.float16), V_flat.to(torch.float16)

    K_fp16, V_fp16 = dequant_kv()

    cu_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, N], dtype=torch.int32, device=device)
    q_fa = q_fp16[0].contiguous().unsqueeze(0)  # (1, H_q, D)

    def run_fa2_preequantised():
        # Dequantise first, then attend.
        K, V = dequant_kv()
        return flash_attn_varlen_func(
            q=q_fa, k=K, v=V,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=1, max_seqlen_k=N,
            softmax_scale=scale, causal=False,
        )

    # Warmup
    for _ in range(warmup):
        run_cda(); run_fa2_preequantised()
    torch.cuda.synchronize()

    def _time(fn):
        times = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            times.append(s.elapsed_time(e) * 1000.0)
        return median(times)

    us_cda = _time(run_cda)
    us_deq = _time(run_fa2_preequantised)
    ratio = us_deq / us_cda
    return us_cda, us_deq, ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="1:4096,1:8192,1:16384,1:32768,4:32768,16:32768")
    ap.add_argument("--output", default="runs/bench_dequant_vs_cda.json")
    args = ap.parse_args()
    configs = [tuple(int(x) for x in c.split(":")) for c in args.configs.split(",")]

    print(f"{'B':>3s} {'N':>6s}  {'CDA (μs)':>10s}  {'DEQUANT+FA2 (μs)':>18s}  "
          f"{'speedup':>8s}")
    print("-" * 60)
    results = []
    for (B, N) in configs:
        try:
            us_cda, us_deq, ratio = _bench_one(B, N)
            print(f"{B:>3d} {N:>6d}  {us_cda:>10.1f}  {us_deq:>18.1f}  "
                  f"{ratio:>7.2f}×")
            results.append({"B": B, "N": N,
                             "cda_us": round(us_cda, 1),
                             "dequant_fa2_us": round(us_deq, 1),
                             "speedup": round(ratio, 2)})
        except Exception as e:
            print(f"{B:>3d} {N:>6d}  FAILED: {str(e)[:80]}")
            results.append({"B": B, "N": N, "error": str(e)[:200]})
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
