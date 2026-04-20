"""Bench three CDA kernel paths side-by-side.

  (a) 3-stage LUT      — cuda_hw_attention_gqa
                         score_4b + softmax + v_2b_full, materialized score GMEM
  (b) Flash-LUT (scalar) — cuda_hw_attention_flash_gqa -> k_cda_flash_split_k4v2_gqa
                         streaming online softmax, multiplier-free inner loops, NO Tensor Core
  (c) HMMA (Tensor Core) — torch.ops._C.cda_decode_full_hmma
                         mma.sync.m16n8k16 + ldmatrix.sync, paged cache

All three on Llama-3.1-8B shape (H_q=32, H_kv=8, D=128) single-request decode (B·1 query),
sweeping the (B, N) grid used in paper Table 4.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from statistics import median

import torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from core.compression import HadamardQuantCompressor
from core.cda_attn import (
    cuda_hw_attention_gqa, cuda_hw_attention_flash_gqa,
    cda_decode_full_hmma,
)


def _prep_per_request(B, N, H_q=32, H_kv=8, D=128, device="cuda:0"):
    torch.manual_seed(0xCDA + N + B)
    dev = torch.device(device)
    ck = HadamardQuantCompressor(dim=D, bit_width=4, half_rotation=True)
    cv = HadamardQuantCompressor(dim=D, bit_width=2, half_rotation=True)
    ck._ensure_tensors(dev); cv._ensure_tensors(dev)
    cb_k = (ck._centroids * 2.0 - 1.0).float().contiguous()
    cb_v = (cv._centroids * 2.0 - 1.0).float().contiguous()
    rot32 = ck._rotation.float().contiguous()

    Q = (torch.randn(B, H_q, D, device=dev, dtype=torch.float32) * 0.3)
    Q_rot = Q @ ck._rotation_t.float().contiguous()

    pK = torch.randint(0, 256, (B, H_kv, N, D // 2), dtype=torch.uint8, device=dev)
    pV = torch.randint(0, 256, (B, H_kv, N, D // 4), dtype=torch.uint8, device=dev)
    nK = (torch.rand(B, H_kv, N, device=dev) + 0.5).float().contiguous()
    nV = (torch.rand(B, H_kv, N, device=dev) + 0.5).float().contiguous()
    return {"Q_rot": Q_rot, "pK": pK, "nK": nK, "pV": pV, "nV": nV,
            "cb_k": cb_k, "cb_v": cb_v, "rotation": rot32,
            "B": B, "N": N, "D": D, "H_q": H_q, "H_kv": H_kv,
            "group_size": H_q // H_kv, "ck": ck, "cv": cv}


def _prep_hmma_paged(B, N, H_q=32, H_kv=8, D=128, device="cuda:0"):
    """HMMA kernel expects paged cache layout."""
    torch.manual_seed(0xCDA + N + B)
    dev = torch.device(device)
    block_size = 16
    num_blk = (N + block_size - 1) // block_size
    total_slots = B * num_blk * block_size

    cache = torch.zeros(total_slots, H_kv, 104, dtype=torch.uint8, device=dev)
    cache[:, :, :96] = torch.randint(0, 256, (total_slots, H_kv, 96), dtype=torch.uint8, device=dev)
    nK = torch.rand(total_slots, H_kv, device=dev) + 0.5
    nV = torch.rand(total_slots, H_kv, device=dev) + 0.5
    cache[:, :, 96:100]  = nK.contiguous().view(torch.uint8).view(total_slots, H_kv, 4)
    cache[:, :, 100:104] = nV.contiguous().view(torch.uint8).view(total_slots, H_kv, 4)

    ck = HadamardQuantCompressor(dim=D, bit_width=4, half_rotation=True)
    cv = HadamardQuantCompressor(dim=D, bit_width=2, half_rotation=True)
    ck._ensure_tensors(dev); cv._ensure_tensors(dev)
    cb_k  = (ck._centroids * 2.0 - 1.0).float().contiguous()
    cb_v  = (cv._centroids * 2.0 - 1.0).float().contiguous()
    rot32 = ck._rotation.float().contiguous()
    rot16 = rot32.to(torch.float16).contiguous()
    block_tbl = torch.arange(B * num_blk, dtype=torch.int32, device=dev).view(B, num_blk)
    seq_lens  = torch.full((B,), N, dtype=torch.int32, device=dev)
    q = torch.randn(B, H_q, D, dtype=torch.float16, device=dev) * 0.3
    out = torch.empty(B, H_q, D, dtype=torch.float16, device=dev)
    return {"q": q, "cache": cache, "block_tbl": block_tbl, "seq_lens": seq_lens,
            "cb_k": cb_k, "cb_v": cb_v, "rot32": rot32, "rot16": rot16, "out": out,
            "block_size": block_size, "num_blk": num_blk,
            "max_seq_bound": num_blk * block_size,
            "B": B, "H_q": H_q, "D": D, "group_size": H_q // H_kv}


def bench_3stage_lut(B, N, warmup=10, iters=30):
    ctx = _prep_per_request(B, N)
    scale = 1.0 / (ctx["D"] ** 0.5)
    def call():
        for b in range(B):
            cuda_hw_attention_gqa(
                ctx["Q_rot"][b], ctx["pK"][b], ctx["nK"][b],
                ctx["pV"][b], ctx["nV"][b],
                ctx["cb_k"], ctx["cb_v"], ctx["rotation"],
                scale, N, ctx["group_size"], k_bits=4, v_bits=2, topk=0)
    for _ in range(warmup): call()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); call(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    return median(ts)


def bench_flash_lut(B, N, tile_N=None, warmup=10, iters=30):
    """Scalar flash-decoding LUT kernel (streaming online softmax, no TC)."""
    ctx = _prep_per_request(B, N)
    scale = 1.0 / (ctx["D"] ** 0.5)
    if tile_N is None:
        tile_N = 1024 if N >= 8192 else 512
    def call():
        for b in range(B):
            cuda_hw_attention_flash_gqa(
                ctx["Q_rot"][b], ctx["pK"][b], ctx["nK"][b],
                ctx["pV"][b], ctx["nV"][b],
                ctx["cb_k"], ctx["cb_v"], ctx["rotation"],
                scale, N, ctx["group_size"], tile_N=tile_N)
    for _ in range(warmup): call()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); call(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    return median(ts)


def bench_hmma_tc(B, N, tile_N=None, warmup=10, iters=30):
    """Actual Tensor-Core HMMA kernel (mma.sync + ldmatrix, paged)."""
    ctx = _prep_hmma_paged(B, N)
    scale = 1.0 / (ctx["D"] ** 0.5)
    if tile_N is None:
        tile_N = 1024 if N >= 8192 else 512
    def call():
        cda_decode_full_hmma(
            ctx["q"], ctx["cache"], ctx["block_tbl"], ctx["seq_lens"],
            ctx["cb_k"], ctx["cb_v"], ctx["rot32"], ctx["rot16"], ctx["out"],
            ctx["group_size"], ctx["block_size"], tile_N, ctx["max_seq_bound"], scale)
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
    ap.add_argument("--Bs", default="1,16,32,64")
    ap.add_argument("--Ns", default="4096,16384,32768,65536,131072")
    ap.add_argument("--output", default="runs/bench_cda_three_paths.json")
    args = ap.parse_args()

    Bs = [int(x) for x in args.Bs.split(",")]
    Ns = [int(x) for x in args.Ns.split(",")]

    rows = []
    hdr = f" {'B':>4} {'N':>7} | {'3-stage LUT':>13} | {'Flash-LUT':>12} | {'HMMA TC':>10} | {'HMMA vs F-LUT':>14}"
    print(hdr); print("-" * len(hdr))
    for B in Bs:
        for N in Ns:
            res = {"B": B, "N": N}
            try:
                res["three_stage_us"] = bench_3stage_lut(B, N)
            except Exception as ex:
                res["three_stage_us"] = None; print(f"  3stage FAIL {B=} {N=}: {ex}")
            try:
                res["flash_lut_us"] = bench_flash_lut(B, N)
            except Exception as ex:
                res["flash_lut_us"] = None; print(f"  flash-LUT FAIL {B=} {N=}: {ex}")
            try:
                res["hmma_tc_us"] = bench_hmma_tc(B, N)
            except Exception as ex:
                res["hmma_tc_us"] = None; print(f"  HMMA-TC FAIL {B=} {N=}: {ex}")

            ts = res.get("three_stage_us"); fl = res.get("flash_lut_us"); hm = res.get("hmma_tc_us")
            spd = (fl / hm) if (fl and hm) else None
            res["hmma_vs_flash_lut"] = round(spd, 2) if spd else None
            rows.append(res)
            t_s = f"{ts:>13.1f}" if ts else "   " + "FAIL".rjust(10)
            f_s = f"{fl:>12.1f}" if fl else "  " + "FAIL".rjust(10)
            h_s = f"{hm:>10.1f}" if hm else "  " + "FAIL".rjust( 8)
            s_s = f"{spd:>14.2f}" if spd else "".rjust(14)
            print(f" {B:>4} {N:>7} | {t_s} | {f_s} | {h_s} | {s_s}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(rows, indent=2))
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
