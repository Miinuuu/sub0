"""Python bindings for the compiled CDA CUDA kernels.

The kernels live in ``csrc/cda_kernels.cu`` and are built into the
``cda._cda_kernels`` extension module by ``setup.py``. Install the package
(``pip install -e .`` or ``python setup.py build_ext --inplace``) before
importing this module.

Two kernels are exposed per bit-width (2-bit and 4-bit):
  * ``cuda_compressed_score``      — Q_rot @ compressed_K  (scaled, with norms)
  * ``cuda_compressed_v``          — attn @ compressed_V   (norm-weighted)
  * ``cuda_compressed_score_4bit`` / ``cuda_compressed_v_4bit``

``cuda_hw_attention_batched`` composes these into a full compressed-domain
attention step: score → softmax → optional top-K gating → V output → inverse
rotation.
"""
from __future__ import annotations

import torch

try:
    from cda import _cda_kernels as _mod  # compiled via setup.py
except ImportError as exc:
    raise ImportError(
        "cda._cda_kernels is not built. Run `pip install -e .` or "
        "`python setup.py build_ext --inplace` from the repository root."
    ) from exc


def cuda_hw_attention_batched(
    Q_rot_all: torch.Tensor,       # (B, D) float32, Q already rotated
    packed_indices_K: torch.Tensor,  # (B*N, D//pack_k) uint8
    norms_K: torch.Tensor,         # (B*N,) float32
    packed_indices_V: torch.Tensor,  # (B*N, D//pack_v) uint8
    norms_V: torch.Tensor,         # (B*N,) float32
    codebook: torch.Tensor,        # (L,) float32 — unmapped K centroids
    rotation: torch.Tensor,        # (D, D) float32 — inverse rotation
    scale: float,
    N: int,
    codebook_v: torch.Tensor | None = None,
    attn_gate_topk: int = 0,
    bit_width_k: int = 0,  # 0 → auto-detect from packed shape
    bit_width_v: int = 0,
) -> torch.Tensor:
    """Full compressed-domain attention using the compiled CUDA kernels.

    Supports 2-bit, 4-bit, and asymmetric K/V bit-widths (e.g. K=4, V=2).
    Bit-widths auto-detect from packed tensor shape when set to 0.

    Args:
        Q_rot_all: Rotated query, shape (B, D), float32 on CUDA.
        packed_indices_K: Bit-packed K indices (MSB-first, see compression.py).
        norms_K: Per-token K norms.
        packed_indices_V: Bit-packed V indices.
        norms_V: Per-token V norms.
        codebook: Unmapped Lloyd-Max K centroids (length 2^B_k).
        rotation: (D, D) inverse rotation matrix (fp32). For CDA, this is
            the Hadamard transpose (== Hadamard / sqrt(D)); for TurboQuant it
            is the random orthogonal QR matrix transpose.
        scale: Attention scale (usually 1/sqrt(D)).
        N: Context length (KV cache token count).
        codebook_v: Optional separate V codebook (for asymmetric bits).
        attn_gate_topk: Optional top-K gating on softmax output. 0 disables.
        bit_width_k / bit_width_v: 2 or 4. 0 means auto-detect from packed shape.

    Returns:
        Output tensor of shape (B, D) in fp32.
    """
    D = Q_rot_all.shape[1]
    assert D <= 128, f"D={D} exceeds CUDA kernel shared memory limit (128)"
    cb_v = codebook_v if codebook_v is not None else codebook

    if bit_width_k == 0:
        bit_width_k = 2 if packed_indices_K.shape[-1] == D // 4 else 4
    if bit_width_v == 0:
        bit_width_v = 2 if packed_indices_V.shape[-1] == D // 4 else 4

    if bit_width_k == 2:
        scores = _mod.cuda_compressed_score(
            Q_rot_all, packed_indices_K, norms_K, codebook, N, scale
        )
    else:
        scores = _mod.cuda_compressed_score_4bit(
            Q_rot_all, packed_indices_K, norms_K, codebook, N, scale
        )

    attn = torch.softmax(scores, dim=-1)

    if attn_gate_topk > 0:
        from cda.compressed_attention import _topk_gate_attn
        attn = _topk_gate_attn(attn, attn_gate_topk)

    if bit_width_v == 2:
        output_rot = _mod.cuda_compressed_v(
            attn, packed_indices_V, norms_V, cb_v, N, D
        )
    else:
        output_rot = _mod.cuda_compressed_v_4bit(
            attn, packed_indices_V, norms_V, cb_v, N, D
        )

    return output_rot @ rotation


# Re-export low-level kernel bindings for power users.
cuda_compressed_score = _mod.cuda_compressed_score
cuda_compressed_v = _mod.cuda_compressed_v
cuda_compressed_score_4bit = _mod.cuda_compressed_score_4bit
cuda_compressed_v_4bit = _mod.cuda_compressed_v_4bit

__all__ = [
    "cuda_hw_attention_batched",
    "cuda_compressed_score",
    "cuda_compressed_v",
    "cuda_compressed_score_4bit",
    "cuda_compressed_v_4bit",
]
