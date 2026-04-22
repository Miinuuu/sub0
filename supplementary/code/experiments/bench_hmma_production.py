"""Phase 5 bench — HMMA production kernel vs baseline coop vs FlashInfer FA2.

Reports μs-per-kernel at realistic vLLM decode shapes (Llama-3.1-8B: H_q=32,
H_kv=8, D=128). Target: HMMA ≤ 0.5 × FA2 at B=16 N=32K (current coop is
2.05× slower than FA2).

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/bench_hmma_production.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import flashinfer

from core.cda_attn import (
    cda_flash_split_k4v2_gqa_paged_batched_coop,
    cda_flash_reduce_batched,
)
from core.compression import HadamardQuantCompressor
from core.vllm_integration.cda_backend import (
    _BYTES_PER_TOKEN, _K_OFFSET, _K_NBYTES, _V_OFFSET, _V_NBYTES,
    _NORM_K_OFFSET, _NORM_V_OFFSET,
)


D = 128
H_q = 32
H_kv = 8
GROUP = 4
BLOCK = 16
SCALE = 1.0 / D ** 0.5
ITERS = 20
WARMUP = 5


def _load():
    # Supplementary: load pre-built .so (no CUDA source redistributed).
    from core._load_prebuilt import load_hmma
    return load_hmma()


def _time_us(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1e6


def _build_cda_cache(B, N, device):
    num_blk = (N + BLOCK - 1) // BLOCK
    num_blocks = num_blk + 4
    kv_cache = torch.zeros(num_blocks, BLOCK, H_kv, _BYTES_PER_TOKEN,
                            dtype=torch.uint8, device=device)
    k = torch.randn(N, H_kv, D, dtype=torch.float16, device=device) * 0.3
    v = torch.randn(N, H_kv, D, dtype=torch.float16, device=device) * 0.3
    kc = HadamardQuantCompressor(dim=D, bit_width=4, half_rotation=True)
    vc = HadamardQuantCompressor(dim=D, bit_width=2, half_rotation=True)
    kc._ensure_tensors(device); vc._ensure_tensors(device)
    cK = kc.quantize(k.reshape(-1, D).contiguous())
    cV = vc.quantize(v.reshape(-1, D).contiguous())
    cb_k = (kc._centroids * 2.0 - 1.0).float().contiguous()
    cb_v = (vc._centroids * 2.0 - 1.0).float().contiguous()
    rotation = kc._rotation.to(torch.float32)
    flat = kv_cache.view(num_blocks * BLOCK, H_kv, _BYTES_PER_TOKEN)
    slots = torch.arange(N, dtype=torch.long, device=device)
    flat[slots, :, _K_OFFSET:_K_OFFSET + _K_NBYTES] = cK.indices.view(N, H_kv, _K_NBYTES)
    flat[slots, :, _V_OFFSET:_V_OFFSET + _V_NBYTES] = cV.indices.view(N, H_kv, _V_NBYTES)
    flat[slots, :, _NORM_K_OFFSET:_NORM_K_OFFSET + 4] = (
        cK.norms.float().view(N, H_kv, 1).view(torch.uint8).view(N, H_kv, 4))
    flat[slots, :, _NORM_V_OFFSET:_NORM_V_OFFSET + 4] = (
        cV.norms.float().view(N, H_kv, 1).view(torch.uint8).view(N, H_kv, 4))

    block_table = torch.arange(num_blk, dtype=torch.int32, device=device) \
        .unsqueeze(0).expand(B, -1).contiguous()
    seq_lens = torch.full((B,), N, dtype=torch.int32, device=device)
    return flat, block_table, seq_lens, cb_k, cb_v, rotation, num_blk


def _fi_wrapper_for(B, N, num_blk, workspace, prev_wrapper):
    # FlashInfer FA2 paged decode.
    num_pages = num_blk + 4
    device = workspace.device
    kv_cache_fp16 = torch.randn(num_pages, 2, H_kv, BLOCK, D,
                                 dtype=torch.float16, device=device) * 0.3
    kv_page_indices = torch.arange(num_blk, dtype=torch.int32, device=device) \
                           .unsqueeze(0).expand(B, -1).contiguous().view(-1)
    kv_page_indptr  = torch.arange(0, B * num_blk + 1, num_blk,
                                    dtype=torch.int32, device=device)
    kv_last_page_len = torch.full((B,), BLOCK, dtype=torch.int32, device=device)
    if N % BLOCK != 0:
        kv_last_page_len[:] = N % BLOCK
    Q_fp16 = torch.randn(B, H_q, D, dtype=torch.float16, device=device) * 0.3

    wrapper = prev_wrapper if prev_wrapper is not None \
        else flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "HND")
    wrapper.plan(
        indptr=kv_page_indptr,
        indices=kv_page_indices,
        last_page_len=kv_last_page_len,
        num_qo_heads=H_q, num_kv_heads=H_kv, head_dim=D,
        page_size=BLOCK,
        q_data_type=torch.float16, kv_data_type=torch.float16,
    )
    return wrapper, Q_fp16, kv_cache_fp16


def main():
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    ext = _load()
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    fi_wrapper = None

    print()
    print("Phase 5 bench — HMMA vs coop vs FA2 (FlashInfer)")
    print(f"  Llama-3.1-8B shapes: H_q={H_q}, H_kv={H_kv}, D={D}, group={GROUP}")
    print()
    print(f"  {'(B, N)':<14}  {'tile':>5}  "
          f"{'FA2 (μs)':>10}  {'coop (μs)':>10}  {'HMMA (μs)':>11}  "
          f"{'HMMA/FA2':>9}  {'HMMA/coop':>10}")
    print("  " + "-" * 80)

    results = []
    for B, N in [(1, 4096), (1, 16384), (1, 32768),
                  (4, 4096), (4, 16384), (4, 32768),
                  (16, 4096), (16, 16384), (16, 32768)]:
        flat, block_table, seq_lens, cb_k, cb_v, rotation, num_blk = \
            _build_cda_cache(B, N, device)
        Q = torch.randn(B, H_q, D, dtype=torch.float16, device=device) * 0.3
        Q_rot = (Q.to(torch.float32) @ rotation).contiguous()
        tile_coop = 1024 if N >= 8192 else 512
        # HMMA favors smaller tiles (more splits → more blocks). Post-optimization
        # sweep shows tile=512 marginally best at N≥16K (1822 μs vs 1833 for 256
        # at B=16 N=32K); kept tile=512 for consistency with coop.
        tile_hmma = 512 if N >= 16384 else tile_coop
        tile = tile_coop  # (column header display)

        # FlashInfer FA2.
        fi_wrapper, Q_fi, kv_fi = _fi_wrapper_for(B, N, num_blk, workspace, fi_wrapper)
        f_fa2 = lambda: fi_wrapper.run(Q_fi, kv_fi)

        # Baseline coop at its optimal tile.
        f_coop = lambda: cda_flash_split_k4v2_gqa_paged_batched_coop(
            Q_rot, flat, block_table, seq_lens, cb_k, cb_v,
            GROUP, BLOCK, tile_coop, N, SCALE,
        )

        # HMMA production at its optimal tile.
        f_hmma = lambda: ext.cda_flash_split_k4v2_gqa_paged_batched_coop_hmma(
            Q_rot, flat, block_table, seq_lens, cb_k, cb_v,
            GROUP, BLOCK, tile_hmma, N, SCALE,
        )

        us_fa2  = _time_us(f_fa2)
        us_coop = _time_us(f_coop)
        us_hmma = _time_us(f_hmma)

        hmma_fa2  = us_hmma / us_fa2
        hmma_coop = us_hmma / us_coop
        print(f"  B={B:<2} N={N:<6}   coop_t={tile_coop} hmma_t={tile_hmma}   "
              f"{us_fa2:>8.1f}  {us_coop:>10.1f}  {us_hmma:>11.1f}  "
              f"{hmma_fa2:>8.2f}×  {hmma_coop:>9.2f}×",
              flush=True)

        results.append({
            "B": B, "N": N, "tile_coop": tile_coop, "tile_hmma": tile_hmma,
            "fa2_us": round(us_fa2, 1),
            "coop_us": round(us_coop, 1),
            "hmma_us": round(us_hmma, 1),
            "hmma_over_fa2": round(hmma_fa2, 3),
            "hmma_over_coop": round(hmma_coop, 3),
        })

    Path("runs").mkdir(exist_ok=True)
    Path("runs/bench_hmma_production.json").write_text(json.dumps(results, indent=2))
    print("\n  → runs/bench_hmma_production.json")


if __name__ == "__main__":
    main()
