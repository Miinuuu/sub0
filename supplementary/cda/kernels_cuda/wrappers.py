"""High-level wrappers for the paper-essential HMMA kernels (K4V4).

After the paper-cleanup pass this module exports two production-path
entry points:

* :func:`decode_hmma_v1` — single-Q decode HMMA (Tables 2/3, vLLM
  serving decode). Wraps ``cda_decode_full_hmma_v3`` from
  ``_decode_hmma.cu``.

* :func:`paged_encode_fused_cuda` — fused FHT + L2-quantize + paged
  scatter encoder (KV cache writes, b4 path). Wraps
  ``cda_paged_encode_fused`` from ``_paged_encode.cu``.

The earlier multi-Q / hybrid-prefill / fa2-direct wrappers and their
helpers were removed alongside their .cu sources during the
paper-cleanup pass. ``cda.kernels_cuda.vllm_integration.cda_attn_v2``
keeps the corresponding dispatcher branches behind
``CDA_PREFILL_BACKEND`` env-var guards; selecting a non-default
backend after the cleanup raises ``ImportError`` on the inline
import, which is the intended fail-loud behavior.
"""
import functools
import math
from typing import Optional

import torch
from torch import Tensor

from cda.kernels_cuda.hmma_loader import (
    load_decode_hmma,
    load_paged_encode_fused,
)


@functools.lru_cache(maxsize=8)
def _get_num_sms(device_idx: int) -> int:
    """Cached CUDA SM count (multi_processor_count) for a given device."""
    return torch.cuda.get_device_properties(device_idx).multi_processor_count


def _choose_tile_n(past_len: int, *, B: int = 1, H_q: int = 32,
                    device_idx: int = 0) -> int:
    """Pick tile_N for decode_hmma_v1 (FA2-style B-aware).

    Pick the LARGEST tile_N ∈ {512, 256, 1024} whose CTA grid yields
    ≥2 waves at the 2-blocks/SM in-flight target:

      * tile_N=512 — 2 blocks/SM occupancy (33 KB SMEM); preferred
        unless the grid is too small for ≥2 waves.
      * tile_N=256 — wins at low B short N (more N-axis parallelism).
      * tile_N=1024 — fallback when N is so long that smaller tiles
        violate the FR_MAX_SPLITS=256 cap from ``_decode_hmma.cu``.

    Per-call kwargs are kwarg-only so callers that pass only
    ``past_len`` keep B=1 (the historical default).
    """
    num_sms = _get_num_sms(device_idx)
    target_in_flight = 2 * num_sms       # 2 blocks/SM × num_SMs
    target_waves = 2.0
    bn = B * H_q                          # decode_hmma grid = B × H_q × splits

    # FR_MAX_SPLITS in _decode_hmma.cu (reduce kernel cap).
    FR_MAX_SPLITS = 256

    for tn in (256, 512, 1024):
        splits = (past_len + tn - 1) // tn
        if splits > FR_MAX_SPLITS:
            continue
        waves = (bn * splits) / target_in_flight
        if waves >= target_waves:
            return tn
    if (past_len + 255) // 256 <= FR_MAX_SPLITS:
        return 256
    if (past_len + 511) // 512 <= FR_MAX_SPLITS:
        return 512
    return 1024


def decode_hmma_v1(
    Q_fp16: Tensor,           # (B, H_q, D) fp16 — UNROTATED
    kv_cache: Tensor,         # (num_total_slots, H_kv, 136) uint8 K4V4
    block_table: Tensor,      # (B, max_blocks) int32
    seq_lens: Tensor,         # (B,) int32 — full seq_len (past + 1)
    output: Tensor,           # (B, H_q, D) fp16 — written in place
    *,
    cb_K: Tensor,
    cb_V: Tensor,
    rotation_fp32: Tensor,
    rotation_fp16: Tensor,
    group_size: int,
    block_size: int,
    scale: Optional[float] = None,
    max_seq_len: Optional[int] = None,
) -> None:
    """v1's HMMA single-Q decode kernel (K4V4 mode), in-place output.

    Wraps ``cda_decode_full_hmma_v3`` (non-graphable variant: kernel
    allocates internal scratch). For CG-safe decode call the graphable
    variant once it's wired through.
    """
    B, H_q, D = Q_fp16.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    if max_seq_len is None:
        max_seq_len = int(seq_lens.max().item())

    mod = load_decode_hmma(k4v2=False, group_size=group_size)
    tile_N = _choose_tile_n(max_seq_len, B=B, H_q=H_q,
                             device_idx=Q_fp16.device.index or 0)
    if Q_fp16.dtype != torch.float16:
        Q_fp16 = Q_fp16.to(torch.float16)
    if output.dtype != torch.float16:
        raise ValueError("decode_hmma_v1 requires fp16 output buffer")

    mod.cda_decode_full_hmma_v3(
        Q_fp16.contiguous(),
        kv_cache,
        block_table.to(torch.int32) if block_table.dtype != torch.int32 else block_table,
        seq_lens.to(torch.int32) if seq_lens.dtype != torch.int32 else seq_lens,
        cb_K.contiguous(), cb_V.contiguous(),
        rotation_fp32.contiguous(), rotation_fp16.contiguous(),
        output,
        group_size, block_size, tile_N, max_seq_len, scale,
    )


def paged_encode_fused_cuda(
    K: Tensor,                      # (T, H_kv, D=128) fp16, UN-rotated
    V: Tensor,
    slot_mapping: Tensor,           # (T,) int32
    flat_cache: Tensor,             # (num_slots, H_kv, slot_w) uint8
    *,
    cb_K: Tensor,                   # (16,) fp32
    cb_V: Tensor,
) -> None:
    """Single-CUDA-kernel fused FHT + L2-quantize + scatter encode (b4).

    Drop-in replacement for the Triton + pre-FHT trio with identical
    semantics: K4V4 layout, MSB-first nibble packing, L2-norm scaling,
    ε=1e-12. Caller passes raw fp16 K/V — kernel does Hadamard rotation
    internally via in-SMEM Sylvester butterfly (1/sqrt(D) scaled).
    """
    if K.dtype != torch.float16 or V.dtype != torch.float16:
        raise TypeError("K, V must be fp16")
    if flat_cache.dtype != torch.uint8:
        raise TypeError("flat_cache must be uint8")
    if K.shape[-1] != 128:
        raise ValueError(f"D must be 128, got {K.shape[-1]}")
    mod = load_paged_encode_fused()
    sm_i32 = (slot_mapping
              if slot_mapping.dtype == torch.int32
              else slot_mapping.to(torch.int32))
    mod.cda_paged_encode_fused(
        K.contiguous(), V.contiguous(), sm_i32, flat_cache,
        cb_K.contiguous(), cb_V.contiguous(),
    )


__all__ = [
    "decode_hmma_v1",
    "paged_encode_fused_cuda",
]
