"""Tab. 3 re-measurement in CUDA Graph mode for unified timing.

Measures CDA decode_hmma_v1 and dequant+FA2 production path on the SAME K4V4
cache, captured into CUDA graphs and replayed. Replaces the eager-mode
measurement in compute_path_isolation_eager.json with a graph-replay version
so that Tab. 2 (production-path) and Tab. 3 (compute-path-iso) share a
timing methodology.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402


def _time_graph(graph: torch.cuda.CUDAGraph, *, iters: int) -> float:
    torch.cuda.synchronize()
    vals = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        vals.append(start.elapsed_time(end) * 1000.0)  # us
    return median(vals)


def _capture(fn, *, warmup: int) -> torch.cuda.CUDAGraph:
    """Warm up + capture fn into a CUDA graph."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def main():
    import os
    cells_arg = os.environ.get("CELLS", "1,4096")
    cells = []
    for pair in cells_arg.split(";"):
        b, n = pair.split(",")
        cells.append((int(b), int(n)))
    H_kv = 8
    group_size = 4
    head_dim = 128
    block_size = 16
    M_q = 1
    scale_input = 0.25
    warmup = 5
    iters = 20
    device = torch.device("cuda:0")

    H_q = H_kv * group_size
    D = head_dim
    scale = 1.0 / math.sqrt(D)

    # Imports — matched to bench_kernel_head_to_head.py
    from cda.algorithm.compression import Compressor
    from cda.kernels_vllm_fa2_fork import (
        dequantize_compressed_kv,
        flash_attn_varlen_compressed_kv_fused_func,
        flash_attn_varlen_func,
    )
    from cda.kernels_cuda.wrappers import decode_hmma_v1
    from cda.kernels_cuda.hmma_loader import load_paged_encode_fused

    encode_mod = load_paged_encode_fused()

    results = []
    for B, N in cells:
        torch.manual_seed(0xCDA + N + B * 97)

        q = (
            torch.randn(B, H_q, M_q, D, dtype=torch.float16, device=device)
            * scale_input
        ).contiguous()
        k = (
            torch.randn(B, H_kv, N, D, dtype=torch.float16, device=device)
            * scale_input
        ).contiguous()
        v = (
            torch.randn(B, H_kv, N, D, dtype=torch.float16, device=device)
            * scale_input
        ).contiguous()

        cmp_k = Compressor(D, num_levels=16, codebook_mode="lloyd_beta", device=device)
        cmp_v = Compressor(D, num_levels=16, codebook_mode="lloyd_beta", device=device)
        cb_k_fp32 = cmp_k.codebook.float().contiguous()
        cb_v_fp32 = cmp_v.codebook.float().contiguous()
        rot_fp16 = cmp_k.rotation.to(torch.float16).contiguous()
        rot_fp32 = cmp_k.rotation.float().contiguous()
        q_rot = cmp_k.rotate(q).contiguous()

        k_slot = cmp_k.encode(k)
        v_slot = cmp_v.encode(v)

        # Paged cache (matches bench_kernel_head_to_head.py)
        blocks_per_seq = (N + block_size - 1) // block_size
        num_blocks = B * blocks_per_seq
        slot_w = 144
        kv_cache_dep = torch.zeros(
            num_blocks * block_size, H_kv, slot_w, dtype=torch.uint8, device=device
        )
        block_table_dep = (
            torch.arange(num_blocks, dtype=torch.int32, device=device)
            .view(B, blocks_per_seq)
        )
        seq_lens_dep = torch.full((B,), N, dtype=torch.int32, device=device)
        K_flat = k.permute(0, 2, 1, 3).reshape(B * N, H_kv, D).contiguous()
        V_flat = v.permute(0, 2, 1, 3).reshape(B * N, H_kv, D).contiguous()
        slot_mapping = torch.arange(B * N, dtype=torch.int32, device=device)
        encode_mod.cda_paged_encode_fused(
            K_flat, V_flat, slot_mapping, kv_cache_dep, cb_k_fp32, cb_v_fp32
        )

        # ============================================================
        # CDA path: single-kernel decode_hmma_v1
        # ============================================================
        out_decode = torch.empty(B, H_q, D, dtype=torch.float16, device=device)

        def call_cda():
            decode_hmma_v1(
                q.squeeze(2),
                kv_cache_dep,
                block_table_dep,
                seq_lens_dep,
                out_decode,
                cb_K=cb_k_fp32,
                cb_V=cb_v_fp32,
                rotation_fp32=rot_fp32,
                rotation_fp16=rot_fp16,
                group_size=group_size,
                block_size=block_size,
                scale=scale,
                max_seq_len=N,
            )

        cda_graph = _capture(call_cda, warmup=warmup)
        cda_us = _time_graph(cda_graph, iters=iters)

        # ============================================================
        # FA2 path: pure FP16 cache via standard FA2 (no quantization)
        # ============================================================
        K_flat_full = k.permute(0, 2, 1, 3).reshape(B * N, H_kv, D).contiguous()
        V_flat_full = v.permute(0, 2, 1, 3).reshape(B * N, H_kv, D).contiguous()
        cu_q_raw = (
            torch.arange(0, B + 1, dtype=torch.int32, device=device) * M_q
        )
        cu_k_raw = (
            torch.arange(0, B + 1, dtype=torch.int32, device=device) * N
        )
        q_raw_var = (
            q.permute(0, 2, 1, 3).reshape(B * M_q, H_q, D).contiguous()
        )

        def call_fa2():
            return flash_attn_varlen_func(
                q_raw_var,
                K_flat_full,
                V_flat_full,
                cu_seqlens_q=cu_q_raw,
                cu_seqlens_k=cu_k_raw,
                max_seqlen_q=M_q,
                max_seqlen_k=N,
                softmax_scale=scale,
            )

        fa2_graph = _capture(call_fa2, warmup=warmup)
        fa2_us = _time_graph(fa2_graph, iters=iters)

        # ============================================================
        # dequant+FA2 path: two kernels (dequant + FA2) captured together
        # ============================================================
        cu_q = (
            torch.arange(0, B + 1, dtype=torch.int32, device=device) * M_q
        )
        cu_k = (
            torch.arange(0, B + 1, dtype=torch.int32, device=device) * N
        )
        q_var = (
            q_rot.permute(0, 2, 1, 3).reshape(B * M_q, H_q, D).contiguous()
        )

        def call_path():
            K_mat, V_mat = dequantize_compressed_kv(
                k_slot.idx,
                k_slot.norm,
                v_slot.idx,
                v_slot.norm,
                cmp_k.codebook,
                cmp_v.codebook,
            )
            K_var = (
                K_mat.permute(0, 2, 1, 3).reshape(B * N, H_kv, D).contiguous()
                if K_mat.dim() == 4
                else K_mat
            )
            V_var = (
                V_mat.permute(0, 2, 1, 3).reshape(B * N, H_kv, D).contiguous()
                if V_mat.dim() == 4
                else V_mat
            )
            return flash_attn_varlen_func(
                q_var,
                K_var,
                V_var,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=M_q,
                max_seqlen_k=N,
                softmax_scale=scale,
            )

        path_graph = _capture(call_path, warmup=warmup)
        dequant_fa2_us = _time_graph(path_graph, iters=iters)

        # ============================================================
        # FUSED dequant+FA2 path: single fused kernel (R1 in-kernel SMEM-tile dequant)
        # ============================================================
        def call_fused_path():
            return flash_attn_varlen_compressed_kv_fused_func(
                q_var,
                k_slot.idx,
                k_slot.norm,
                v_slot.idx,
                v_slot.norm,
                cmp_k.codebook,
                cmp_v.codebook,
                max_seqlen_q=M_q,
                cu_seqlens_q=cu_q,
                max_seqlen_k=N,
                cu_seqlens_k=cu_k,
                softmax_scale=scale,
            )

        fused_graph = _capture(call_fused_path, warmup=warmup)
        fused_dequant_fa2_us = _time_graph(fused_graph, iters=iters)

        ratio = cda_us / dequant_fa2_us
        ratio_fused = cda_us / fused_dequant_fa2_us
        ratio_fa2 = cda_us / fa2_us
        results.append(
            {
                "B": B,
                "N": N,
                "FA2_us": fa2_us,
                "CDA_us": cda_us,
                "dequant_FA2_us": dequant_fa2_us,
                "fused_dequant_FA2_us": fused_dequant_fa2_us,
                "ratio_CDA_vs_FA2": ratio_fa2,
                "ratio_CDA_vs_dequantFA2": ratio,
                "ratio_CDA_vs_fused_dequantFA2": ratio_fused,
            }
        )
        print(
            f"B={B}, N={N}: FA2={fa2_us:.1f} us, CDA={cda_us:.1f} us, "
            f"dequant+FA2={dequant_fa2_us:.1f} us, fused={fused_dequant_fa2_us:.1f} us, "
            f"r_FA2={ratio_fa2:.3f}, r_deq={ratio:.3f}, r_fused={ratio_fused:.3f}",
            flush=True,
        )

    out_name = os.environ.get("OUT_NAME", "compute_path_isolation_cuda_graph")
    out_path = Path(f"runs/paper/{out_name}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "config": {
                    "purpose": (
                        "Tab. 3 re-measurement in CUDA Graph mode for unified timing"
                        " methodology with Tab. 2 production-path."
                    ),
                    "framework": "benchmarks/bench_compute_path_iso_cuda_graph.py",
                    "mode": "cuda_graph_replay",
                    "warmup": warmup,
                    "iters": iters,
                    "device": torch.cuda.get_device_name(device),
                    "torch_version": torch.__version__,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M"),
                    "model_shape": (
                        f"H_kv={H_kv}, group={group_size}, head_dim={head_dim},"
                        f" block_size={block_size}"
                    ),
                },
                "results": results,
            },
            indent=2,
        )
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
