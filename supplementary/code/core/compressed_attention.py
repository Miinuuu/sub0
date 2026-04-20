"""Compressed-domain attention utilities.

Two modes optimized for different targets:

SW Mode (GPU):
  - KV stored compressed for memory savings
  - Decode: decompress → fp16 cuBLAS matmul (fastest on GPU)
  - Pre-rotation interpolation: N rotations → 1 (when blending multiple KVs)

HW Mode (FPGA/ASIC):
  - KV stored compressed, NEVER decompressed
  - Decode: attention computed directly on compressed indices
  - Q rotation 1x + codebook lookup + accumulate → score
  - 7.5x bandwidth reduction, 37x compute reduction vs decompress path
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from core.compression import (
    TurboQuantCompressor,
    HadamardQuantCompressor,
    CompressedTensor,
    _gpu_unpack_4bit,
    _gpu_unpack_2bit,
)


# ---------------------------------------------------------------------------
# Attention gating helper
# ---------------------------------------------------------------------------

def _topk_gate_attn(attn_weights: torch.Tensor, topk: int) -> torch.Tensor:
    """Zero out all but top-K attention weights per query, then renormalize."""
    if topk <= 0 or topk >= attn_weights.shape[-1]:
        return attn_weights
    _, topk_idx = attn_weights.topk(topk, dim=-1)
    mask = torch.zeros_like(attn_weights)
    mask.scatter_(-1, topk_idx, 1.0)
    gated = attn_weights * mask
    return gated / gated.sum(dim=-1, keepdim=True).clamp(min=1e-12)


# ---------------------------------------------------------------------------
# SW Mode: GPU-optimized (decompress + cuBLAS)
# ---------------------------------------------------------------------------

def sw_attention(
    Q: torch.Tensor,
    compressed_K: CompressedTensor,
    compressed_V: CompressedTensor,
    compressor: TurboQuantCompressor,
    scale: Optional[float] = None,
    v_compressor: Optional[TurboQuantCompressor] = None,
    attn_gate_topk: int = 0,
) -> torch.Tensor:
    """GPU-optimized: decompress KV → standard fp16 attention.

    Best for GPU where cuBLAS fp16 matmul is fastest.
    Benefit: memory savings from compressed storage.

    Args:
        Q: (batch, heads, seq_q, dim)
        compressed_K/V: compressed KV cache
        compressor: TurboQuant or Hadamard compressor
        scale: attention scale (default: 1/√dim)

    Returns:
        output: (batch, heads, seq_q, dim)
    """
    if scale is None:
        scale = compressor.dim ** -0.5

    K = compressor.dequantize(compressed_K)  # full decompress
    V = (v_compressor or compressor).dequantize(compressed_V)  # full decompress

    scores = (Q @ K.T) * scale
    attn = F.softmax(scores, dim=-1)
    if attn_gate_topk > 0:
        attn = _topk_gate_attn(attn, attn_gate_topk)
    output = attn @ V

    return output


def sw_attention_pre_rotation(
    Q: torch.Tensor,
    compressed_K_list: list,
    compressed_V_list: list,
    weights: torch.Tensor,
    compressor: TurboQuantCompressor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """GPU-optimized weighted blend of multiple compressed KVs.

    Pre-rotation interpolation: blend in rotated space, rotate once.
    Saves (N-1) rotation operations when blending N compressed KVs.

    Used by KVCOMM for delta interpolation.
    """
    if scale is None:
        scale = compressor.dim ** -0.5

    # Blend in pre-rotation space
    K_blended_rot = None
    V_blended_rot = None
    for i, (cK, cV, w) in enumerate(zip(compressed_K_list, compressed_V_list, weights)):
        unmapped_K, norms_K = compressor.dequantize_pre_rotation(cK)
        unmapped_V, norms_V = compressor.dequantize_pre_rotation(cV)
        pre_rot_K = unmapped_K.float() * norms_K
        pre_rot_V = unmapped_V.float() * norms_V
        if K_blended_rot is None:
            K_blended_rot = w * pre_rot_K
            V_blended_rot = w * pre_rot_V
        else:
            K_blended_rot += w * pre_rot_K
            V_blended_rot += w * pre_rot_V

    # ONE rotation at the end
    K_restored = compressor.apply_inverse_rotation(K_blended_rot)
    V_restored = compressor.apply_inverse_rotation(V_blended_rot)

    # Standard attention
    scores = (Q @ K_restored.T) * scale
    attn = F.softmax(scores, dim=-1)
    output = attn @ V_restored

    return output


# ---------------------------------------------------------------------------
# HW Mode: FPGA/ASIC-optimized (compressed-domain, no decompress)
# ---------------------------------------------------------------------------

def _unpack_indices(compressed: CompressedTensor, dim: int) -> torch.Tensor:
    """Unpack bit-packed indices to long tensor."""
    packed = compressed.indices
    if compressed.bit_width == 4 and packed.shape[-1] == dim // 2:
        return _gpu_unpack_4bit(packed).long()
    elif compressed.bit_width == 2 and packed.shape[-1] == dim // 4:
        return _gpu_unpack_2bit(packed).long()
    return packed.long()


def hw_attention_score(
    Q: torch.Tensor,
    compressed_K: CompressedTensor,
    compressor: TurboQuantCompressor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """FPGA-optimized: compute Q·K^T without decompressing K.

    K stays compressed. Q is rotated once. Score computed via
    codebook lookup + dot product in rotated space.

    Bandwidth: 7.5x reduction (2-bit) vs fp16.
    Compute: no rotation of K (37x reduction vs decompress path).

    Returns:
        scores: (M, N) attention scores (pre-softmax)
    """
    if scale is None:
        scale = compressor.dim ** -0.5

    compressor._ensure_tensors(Q.device)

    indices = _unpack_indices(compressed_K, compressor.dim)
    norms = compressed_K.norms.float()

    # Q rotation: always fp32 for precision (Hadamard = additions only)
    Q_rot = Q.float() @ compressor._rotation_t.float()

    # Codebook lookup (no decompress), fp32
    c_k = compressor._centroids[indices].float() * 2.0 - 1.0

    # Dot product in rotated space
    scores = (Q_rot @ c_k.T) * norms.unsqueeze(0) * scale

    return scores


def hw_attention_output(
    attn_weights: torch.Tensor,
    compressed_V: CompressedTensor,
    compressor: TurboQuantCompressor,
    v_compressor: Optional[TurboQuantCompressor] = None,
) -> torch.Tensor:
    """FPGA-optimized: compute attn·V without decompressing V.

    Weighted sum in rotated space + ONE inverse rotation at the end.

    Returns:
        output: (M, dim) attention output
    """
    vc = v_compressor or compressor
    vc._ensure_tensors(attn_weights.device)

    indices = _unpack_indices(compressed_V, vc.dim)
    norms = compressed_V.norms.float()

    # Codebook lookup (no decompress), fp32
    c_v = vc._centroids[indices].float() * 2.0 - 1.0
    scaled_v = c_v * norms.unsqueeze(-1)

    # Weighted sum in rotated space
    output_rot = attn_weights.float() @ scaled_v

    # ONE inverse rotation, fp32
    output = output_rot @ vc._rotation.float()

    return output.float()


def hw_attention(
    Q: torch.Tensor,
    compressed_K: CompressedTensor,
    compressed_V: CompressedTensor,
    compressor: TurboQuantCompressor,
    scale: Optional[float] = None,
    v_compressor: Optional[TurboQuantCompressor] = None,
    attn_gate_topk: int = 0,
) -> torch.Tensor:
    """FPGA-optimized full attention: K and V never decompressed.

    Complete pipeline:
      1. Q_rot = H · Q                              (Hadamard 1x)
      2. score = Q_rot · centroids[idx_K] × norm_K   (K compressed-domain)
      3. attn = softmax(score / √d)
      4. out_rot = Σ attn × centroids[idx_V] × norm_V (V compressed-domain)
      5. output = H^T · out_rot                       (Hadamard 1x)

    Total rotations: 2 (vs 2N for decompress path)
    Bandwidth: 7.5x reduction at 2-bit
    Compute: 37x reduction vs decompress path
    """
    scores = hw_attention_score(Q, compressed_K, compressor, scale)
    attn_weights = F.softmax(scores, dim=-1)
    if attn_gate_topk > 0:
        attn_weights = _topk_gate_attn(attn_weights, attn_gate_topk)
    output = hw_attention_output(attn_weights, compressed_V, compressor, v_compressor=v_compressor)
    return output
