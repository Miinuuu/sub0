"""Head-to-head kernel speed comparison.

Compares three production candidates on the same workload (B=1, decode):
  1. **deprecated `multiq_paged_full_hmma_v1`** — P34/R15 hand-tuned production
     line (101KB .cu, NCU-driven 14+ rounds)
  2. **current Lloyd `flash_attn_varlen_compressed_kv_fused_func`** —
     PR1 Lloyd-W4A16 (with num_splits=1 workaround for correctness)
  3. **production-FA2 baseline** — `dequant_compressed_kv` + FA2
     (= what real production would do without compressed-attention path)
  4. **FA2-fp16 reference** — FA2 on already-materialized fp16 K/V

Uses the same Compressor-encoded data for all paths. Cosine validated
against attention_compressed_rotated_reference.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F


def _time_call(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    vals = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        vals.append(start.elapsed_time(end) * 1000.0)  # us
    return median(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="128,256,512,1024,2048,4096,8192,16384")
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--H-kv", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--scale-input", type=float, default=0.25)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)
    H_q = args.H_kv * args.group_size
    D = args.head_dim
    M_q = 1
    block_size = args.block_size
    scale = 1.0 / math.sqrt(D)
    lengths = [int(s) for s in args.lengths.split(",") if s.strip()]

    print(f"device = {torch.cuda.get_device_name(device)}")
    print(f"workload: B={args.B} H_q={H_q} H_kv={args.H_kv} group={args.group_size} "
          f"D={D} M_q={M_q} block_size={block_size}")
    print(f"warmup={args.warmup} iters={args.iters}")
    print()

    # Imports — defer so module loaders get fresh torch_load each run
    from cda.algorithm.compression import Compressor
    from cda.algorithm.attention import attention_compressed_rotated_reference
    # Deprecated kernels (moved to `deprecated/`) — still importable via the
    # `cda.*` namespace shim in `cda/_deprecated_alias.py`. Required only
    # for the historical 5-way comparison.
    from cda.kernels_vllm_fa2_fork import (
        flash_attn_varlen_compressed_kv_fused_func,
        dequantize_compressed_kv,
        flash_attn_varlen_func,
    )
    from cda.kernels_cuda.wrappers import decode_hmma_v1
    from cda.kernels_cuda.hmma_loader import load_paged_encode_fused

    encode_mod = load_paged_encode_fused()

    # Header — paper Table 3 columns: CDA decode (decode_hmma_v1),
    # Lloyd codebook FA2 path, dequant+FA2, vanilla FA2.
    cols = ["N",
            "p34_decode_hmma_us", "cos",
            "lloyd_us", "cos",
            "prod_FA2_us", "FA2_fp16_us"]
    print(",".join(cols))

    for N in lengths:
        torch.manual_seed(0xCDA + N + args.B * 97)

        # Random Q, K, V (unrotated)
        q = (torch.randn(args.B, H_q, M_q, D, dtype=torch.float16, device=device)
             * args.scale_input).contiguous()
        k = (torch.randn(args.B, args.H_kv, N, D, dtype=torch.float16, device=device)
             * args.scale_input).contiguous()
        v = (torch.randn(args.B, args.H_kv, N, D, dtype=torch.float16, device=device)
             * args.scale_input).contiguous()

        # Compressor-encoded K, V (Lloyd)
        cmp_k = Compressor(D, num_levels=16, codebook_mode="lloyd_beta", device=device)
        cmp_v = Compressor(D, num_levels=16, codebook_mode="lloyd_beta", device=device)
        cb_k_fp32 = cmp_k.codebook.float().contiguous()
        cb_v_fp32 = cmp_v.codebook.float().contiguous()
        rot_fp16 = cmp_k.rotation.to(torch.float16).contiguous()
        rot_fp32 = cmp_k.rotation.float().contiguous()
        q_rot = cmp_k.rotate(q).contiguous()

        # Reference (rotated frame)
        k_slot = cmp_k.encode(k)
        v_slot = cmp_v.encode(v)
        ref_rot = attention_compressed_rotated_reference(
            q_rot, k_slot, v_slot,
            cmp_K=cmp_k, cmp_V=cmp_v, group_size=args.group_size, scale=scale)
        ref_rot_v = ref_rot.permute(0, 2, 1, 3).reshape(args.B * M_q, H_q, D).contiguous()

        # ============================================================
        # Build paged cache (144B/slot, K4V4 = 136 data + 8 pad → 144 aligned)
        # ============================================================
        blocks_per_seq = (N + block_size - 1) // block_size
        num_blocks = args.B * blocks_per_seq
        slot_w = 144
        kv_cache_dep = torch.zeros(
            num_blocks * block_size, args.H_kv, slot_w,
            dtype=torch.uint8, device=device)
        block_table_dep = torch.arange(num_blocks, dtype=torch.int32,
                                        device=device).view(args.B, blocks_per_seq)
        seq_lens_dep = torch.full((args.B,), N, dtype=torch.int32, device=device)
        K_flat = k.permute(0, 2, 1, 3).reshape(args.B * N, args.H_kv, D).contiguous()
        V_flat = v.permute(0, 2, 1, 3).reshape(args.B * N, args.H_kv, D).contiguous()
        slot_mapping = torch.arange(args.B * N, dtype=torch.int32, device=device)
        encode_mod.cda_paged_encode_fused(
            K_flat, V_flat, slot_mapping, kv_cache_dep, cb_k_fp32, cb_v_fp32)

        # ============================================================
        # Path 1: decode_hmma_v1 (CDA decode kernel — paper's CDA column)
        # ============================================================
        out_decode = torch.empty(args.B, H_q, D, dtype=torch.float16, device=device)
        def call_decode():
            decode_hmma_v1(
                q.squeeze(2), kv_cache_dep, block_table_dep, seq_lens_dep, out_decode,
                cb_K=cb_k_fp32, cb_V=cb_v_fp32,
                rotation_fp32=rot_fp32, rotation_fp16=rot_fp16,
                group_size=args.group_size, block_size=block_size,
                scale=scale, max_seq_len=N)
        try:
            call_decode()
            out_decode_rot = torch.matmul(out_decode.float(), rot_fp16.float()).to(torch.float16)
            cos_decode = F.cosine_similarity(out_decode_rot.float().flatten(),
                                              ref_rot_v.float().flatten(), dim=0).item()
            us_decode = _time_call(call_decode, warmup=args.warmup, iters=args.iters)
        except Exception as e:
            us_decode, cos_decode = float('nan'), float('nan')
            print(f"  [N={N}] decode_hmma FAILED: {e}", file=sys.stderr)

        # ============================================================
        # Path 2: current Lloyd (num_splits=1 for correctness)
        # ============================================================
        k_idx_paged = torch.zeros(num_blocks, block_size, args.H_kv, D//2,
                                   dtype=torch.uint8, device=device)
        v_idx_paged = torch.zeros_like(k_idx_paged)
        k_norm_paged = torch.zeros(num_blocks, block_size, args.H_kv,
                                    dtype=torch.float32, device=device)
        v_norm_paged = torch.zeros_like(k_norm_paged)
        block_table_lloyd = torch.arange(num_blocks, dtype=torch.int32,
                                          device=device).view(args.B, blocks_per_seq)
        for b in range(args.B):
            for blk in range(blocks_per_seq):
                physical = b * blocks_per_seq + blk
                start = blk * block_size
                stop = min(N, start + block_size)
                width = stop - start
                k_idx_paged[physical, :width].copy_(k_slot.idx[b, :, start:stop].permute(1, 0, 2))
                v_idx_paged[physical, :width].copy_(v_slot.idx[b, :, start:stop].permute(1, 0, 2))
                k_norm_paged[physical, :width].copy_(k_slot.norm[b, :, start:stop].permute(1, 0))
                v_norm_paged[physical, :width].copy_(v_slot.norm[b, :, start:stop].permute(1, 0))

        q_varlen = q_rot.permute(0, 2, 1, 3).reshape(args.B * M_q, H_q, D).contiguous()
        cu_q = torch.arange(0, args.B + 1, dtype=torch.int32, device=device) * M_q
        seq_lens_lloyd = torch.full((args.B,), N, dtype=torch.int32, device=device)

        out_lloyd = torch.empty(args.B * M_q, H_q, D, dtype=torch.float16, device=device)
        def call_lloyd():
            flash_attn_varlen_compressed_kv_fused_func(
                q_varlen, k_idx_paged, k_norm_paged, v_idx_paged, v_norm_paged,
                cmp_k.codebook, cmp_v.codebook,
                M_q, cu_q, N, None,
                out=out_lloyd, softmax_scale=scale, num_splits=1,
                seqused_k=seq_lens_lloyd, block_table=block_table_lloyd,
                gqa_decode_swap=None, uniform_codebook=False)
        try:
            call_lloyd()
            cos_lloyd = F.cosine_similarity(out_lloyd.float().flatten(),
                                             ref_rot_v.float().flatten(), dim=0).item()
            us_lloyd = _time_call(call_lloyd, warmup=args.warmup, iters=args.iters)
        except Exception as e:
            us_lloyd, cos_lloyd = float('nan'), float('nan')
            print(f"  [N={N}] lloyd FAILED: {e}", file=sys.stderr)

        # ============================================================
        # Path 3: production-FA2 (dequant + FA2)
        # ============================================================
        def call_dequant():
            return dequantize_compressed_kv(
                k_slot.idx, k_slot.norm, v_slot.idx, v_slot.norm,
                cmp_k.codebook, cmp_v.codebook)
        K_mat, V_mat = call_dequant()
        # FA2 on dequantized fp16 K/V
        cu_k = torch.arange(0, args.B + 1, dtype=torch.int32, device=device) * N
        K_var = K_mat.permute(0, 2, 1, 3).reshape(args.B * N, args.H_kv, D).contiguous() if K_mat.dim() == 4 else K_mat
        V_var = V_mat.permute(0, 2, 1, 3).reshape(args.B * N, args.H_kv, D).contiguous() if V_mat.dim() == 4 else V_mat
        out_fa2 = torch.empty(args.B * M_q, H_q, D, dtype=torch.float16, device=device)
        def call_fa2():
            return flash_attn_varlen_func(
                q_rot.permute(0, 2, 1, 3).reshape(args.B * M_q, H_q, D).contiguous(),
                K_var, V_var,
                cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                max_seqlen_q=M_q, max_seqlen_k=N,
                softmax_scale=scale)
        try:
            us_dequant = _time_call(call_dequant, warmup=args.warmup, iters=args.iters)
            us_fa2 = _time_call(call_fa2, warmup=args.warmup, iters=args.iters)
            us_prod = us_dequant + us_fa2
        except Exception as e:
            us_dequant, us_fa2, us_prod = float('nan'), float('nan'), float('nan')
            print(f"  [N={N}] prod FA2 FAILED: {e}", file=sys.stderr)

        print(f"{N},"
              f"{us_decode:.1f},{cos_decode:.4f},"
              f"{us_lloyd:.1f},{cos_lloyd:.4f},"
              f"{us_prod:.1f},{us_fa2:.1f}", flush=True)


if __name__ == "__main__":
    main()
