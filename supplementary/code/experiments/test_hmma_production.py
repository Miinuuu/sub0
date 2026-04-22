"""Phase 4 — test the production HMMA paged-batched coop kernel vs baseline coop.

Builds a realistic (B, H_q, H_kv, D) attention config with paged KV and
Hadamard-quantized K/V, then checks:
  1. HMMA output matches baseline coop output (cosine > 0.998).
  2. Both outputs match a pure-PyTorch compressed-domain reference.

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/test_hmma_production.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

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


def _load():
    # Supplementary: load pre-built .so (no CUDA source redistributed).
    from core._load_prebuilt import load_hmma
    return load_hmma()


def _build_cache(B, N, device):
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
    return flat, block_table, seq_lens, cb_k, cb_v, rotation


def main():
    device = torch.device("cuda:0")
    torch.manual_seed(0)

    ext = _load()

    print()
    print("Phase 4 correctness — HMMA production vs baseline coop")
    print()
    print(f"  {'(B, N)':<14}  {'tile_N':>6}  {'cos(HMMA,coop)':>15}  "
          f"{'max|abs|':>12}  result")
    print("  " + "-" * 65)

    all_pass = True
    for B, N in [(1, 4096), (1, 32768), (4, 8192), (4, 32768), (16, 4096), (16, 32768)]:
        flat, block_table, seq_lens, cb_k, cb_v, rotation = _build_cache(B, N, device)

        Q = torch.randn(B, H_q, D, dtype=torch.float16, device=device) * 0.3
        Q_rot = (Q.to(torch.float32) @ rotation).contiguous()

        tile = 1024 if N >= 8192 else 512

        # Baseline coop.
        p_c, m_c, l_c = cda_flash_split_k4v2_gqa_paged_batched_coop(
            Q_rot, flat, block_table, seq_lens, cb_k, cb_v,
            GROUP, BLOCK, tile, N, SCALE,
        )
        out_coop = cda_flash_reduce_batched(p_c, m_c, l_c)

        # HMMA production.
        p_h, m_h, l_h = ext.cda_flash_split_k4v2_gqa_paged_batched_coop_hmma(
            Q_rot, flat, block_table, seq_lens, cb_k, cb_v,
            GROUP, BLOCK, tile, N, SCALE,
        )
        out_hmma = cda_flash_reduce_batched(p_h, m_h, l_h)

        cos = torch.nn.functional.cosine_similarity(
            out_hmma.reshape(-1), out_coop.reshape(-1), dim=0).item()
        max_abs = (out_hmma - out_coop).abs().max().item()
        ok = cos > 0.998
        tag = "PASS" if ok else "FAIL"
        print(f"  B={B:<2} N={N:<6}    tile={tile:<4}  {cos:>15.6f}  "
              f"{max_abs:>12.4e}  {tag}")
        if not ok:
            all_pass = False

    if all_pass:
        print("\n[Phase 4] PASS — HMMA production matches coop across all configs.")
    else:
        print("\n[Phase 4] FAIL — debug output")
        sys.exit(1)


if __name__ == "__main__":
    main()
