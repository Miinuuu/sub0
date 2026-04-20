"""Pure-PyTorch reference for the Flash-Decoding split-K CDA kernel.

Two flavours are provided:

* :func:`flash_attention_reference` — streaming online-softmax loop over N,
  producing the final output in *original* (unrotated) space. Matches
  :func:`core.compressed_attention.hw_attention` within floating-point
  tolerance. Use this to sanity-check the streaming math.

* :func:`flash_split_reference` + :func:`flash_reduce_reference` — mirrors
  the two-kernel CUDA layout:
    * Kernel 1 emits per-split ``(partial_out_rot, m_split, l_split)``.
    * Kernel 2 merges splits via the canonical online-softmax rescale
      and applies the inverse Hadamard rotation.
  Use these to unit-test each CUDA kernel in isolation.

Contract (matches :func:`core.cda_attn.cuda_hw_attention_gqa`):

    Q_rot        : (H_q, D)          float32  — already rotated
    packed_K     : (H_kv, N, D/pk_k) uint8    — 4-bit or 2-bit indices
    norms_K      : (H_kv, N)         float32
    packed_V     : (H_kv, N, D/pk_v) uint8
    norms_V      : (H_kv, N)         float32
    codebook_k   : (2**k_bits,)      float32  — already ``2*centroid - 1``
    codebook_v   : (2**v_bits,)      float32  — already ``2*centroid - 1``
    rotation     : (D, D)            float32  — Hadamard; applied as a final
                                                ``out_rot @ rotation``

Return shape: ``(H_q, D)`` float32.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch

from core.compression import _gpu_unpack_2bit, _gpu_unpack_4bit


def _unpack(packed: torch.Tensor, bit_width: int) -> torch.Tensor:
    """Dispatch by bit-width; returns a long index tensor of shape ``(..., D)``."""
    if bit_width == 2:
        return _gpu_unpack_2bit(packed).long()
    elif bit_width == 4:
        return _gpu_unpack_4bit(packed).long()
    raise ValueError(f"bit_width must be 2 or 4, got {bit_width}")


def _dequantize_per_kv(
    packed: torch.Tensor,   # (H_kv, N, D/pack)
    codebook: torch.Tensor, # (2**bits,)
    bit_width: int,
) -> torch.Tensor:
    """Return ``(H_kv, N, D)`` float32 centroid values (pre-norm)."""
    indices = _unpack(packed, bit_width)            # (H_kv, N, D)
    return codebook.float()[indices]                # gather


def flash_attention_reference(
    Q_rot: torch.Tensor,
    packed_K: torch.Tensor, norms_K: torch.Tensor, codebook_k: torch.Tensor,
    packed_V: torch.Tensor, norms_V: torch.Tensor, codebook_v: torch.Tensor,
    rotation: torch.Tensor,
    scale: float, N: int, group_size: int,
    tile_N: int = 512,
    k_bits: int = 4, v_bits: int = 2,
) -> torch.Tensor:
    """Streaming online-softmax reference (single-pass, no materialised splits).

    Processes the KV sequence in ``tile_N`` chunks, maintaining a running
    ``(m, l, out_rot_accum)`` per Q head, and applies the inverse rotation
    once at the end. Mathematically identical to
    ``hw_attention(...)`` but traces the exact streaming accounting the CUDA
    Kernel 1 will perform.
    """
    H_q, D = Q_rot.shape
    H_kv = packed_K.shape[0]
    assert H_q == H_kv * group_size, \
        f"H_q={H_q} must equal H_kv={H_kv} * group_size={group_size}"
    assert Q_rot.dtype == torch.float32

    device = Q_rot.device
    K_deq = _dequantize_per_kv(packed_K, codebook_k, k_bits)   # (H_kv, N, D)
    V_deq = _dequantize_per_kv(packed_V, codebook_v, v_bits)   # (H_kv, N, D)

    out_rot = torch.zeros(H_q, D, dtype=torch.float32, device=device)
    neg_inf = torch.tensor(float("-inf"), dtype=torch.float32, device=device)

    for h_q in range(H_q):
        kv_h = h_q // group_size
        q = Q_rot[h_q]                                # (D,)

        m = neg_inf.clone()
        l = torch.zeros((), dtype=torch.float32, device=device)
        o = torch.zeros(D, dtype=torch.float32, device=device)

        s_start = 0
        while s_start < N:
            s_end = min(s_start + tile_N, N)
            K_split = K_deq[kv_h, s_start:s_end]      # (tile, D)
            V_split = V_deq[kv_h, s_start:s_end]      # (tile, D)
            nK_split = norms_K[kv_h, s_start:s_end].float()  # (tile,)
            nV_split = norms_V[kv_h, s_start:s_end].float()  # (tile,)

            raw = K_split @ q                          # (tile,)
            scores = raw * nK_split * scale            # (tile,)

            m_tile = scores.max()
            m_new = torch.maximum(m, m_tile)
            rescale = torch.exp(m - m_new) if torch.isfinite(m) else torch.zeros_like(m_new)
            m = m_new
            o = o * rescale
            l = l * rescale

            e = torch.exp(scores - m)                  # (tile,)
            l = l + e.sum()
            v_weighted = V_split * nV_split.unsqueeze(-1)            # (tile, D)
            o = o + (e.unsqueeze(-1) * v_weighted).sum(dim=0)        # (D,)

            s_start = s_end

        out_rot[h_q] = o / l

    return out_rot @ rotation


def flash_split_reference(
    Q_rot: torch.Tensor,
    packed_K: torch.Tensor, norms_K: torch.Tensor, codebook_k: torch.Tensor,
    packed_V: torch.Tensor, norms_V: torch.Tensor, codebook_v: torch.Tensor,
    scale: float, N: int, group_size: int,
    tile_N: int = 512,
    k_bits: int = 4, v_bits: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Kernel 1 reference — emits per-split partial state.

    Returns:
        partial_out_rot: (H_q, num_splits, D) float32
                         Σ_i exp(s_i - m_split) * v_i   (rotated space)
        m_vals:          (H_q, num_splits) float32
        l_vals:          (H_q, num_splits) float32
    """
    H_q, D = Q_rot.shape
    H_kv = packed_K.shape[0]
    device = Q_rot.device
    num_splits = (N + tile_N - 1) // tile_N

    K_deq = _dequantize_per_kv(packed_K, codebook_k, k_bits)
    V_deq = _dequantize_per_kv(packed_V, codebook_v, v_bits)

    partial = torch.zeros(H_q, num_splits, D, dtype=torch.float32, device=device)
    m_vals  = torch.full((H_q, num_splits), float("-inf"),
                          dtype=torch.float32, device=device)
    l_vals  = torch.zeros(H_q, num_splits, dtype=torch.float32, device=device)

    for h_q in range(H_q):
        kv_h = h_q // group_size
        q = Q_rot[h_q]

        for s in range(num_splits):
            s_start = s * tile_N
            s_end = min(s_start + tile_N, N)
            if s_start >= N:
                break

            K_split = K_deq[kv_h, s_start:s_end]
            V_split = V_deq[kv_h, s_start:s_end]
            nK_split = norms_K[kv_h, s_start:s_end].float()
            nV_split = norms_V[kv_h, s_start:s_end].float()

            scores = (K_split @ q) * nK_split * scale  # (tile,)
            m_split = scores.max()
            e = torch.exp(scores - m_split)            # (tile,)
            l_split = e.sum()

            v_weighted = V_split * nV_split.unsqueeze(-1)
            partial[h_q, s] = (e.unsqueeze(-1) * v_weighted).sum(dim=0)
            m_vals[h_q, s] = m_split
            l_vals[h_q, s] = l_split

    return partial, m_vals, l_vals


def flash_reduce_reference(
    partial_out_rot: torch.Tensor,  # (H_q, num_splits, D)
    m_vals: torch.Tensor,           # (H_q, num_splits)
    l_vals: torch.Tensor,           # (H_q, num_splits)
    rotation: torch.Tensor,
) -> torch.Tensor:
    """Kernel 2 reference — merge splits via online softmax, apply rotation.

    Returns ``(H_q, D)`` output in original space.
    """
    # NaN in m_vals means "split was empty" → treat as -inf.
    m_safe = torch.where(torch.isfinite(m_vals), m_vals,
                          torch.full_like(m_vals, float("-inf")))
    m_global, _ = m_safe.max(dim=-1, keepdim=True)       # (H_q, 1)

    w = torch.exp(m_safe - m_global)                     # (H_q, num_splits)
    w = torch.where(torch.isfinite(w), w, torch.zeros_like(w))
    l_total = (w * l_vals).sum(dim=-1, keepdim=True)     # (H_q, 1)

    weighted = w.unsqueeze(-1) * partial_out_rot         # (H_q, num_splits, D)
    out_rot = weighted.sum(dim=1) / l_total              # (H_q, D)

    return out_rot @ rotation


__all__ = [
    "flash_attention_reference",
    "flash_split_reference",
    "flash_reduce_reference",
]
