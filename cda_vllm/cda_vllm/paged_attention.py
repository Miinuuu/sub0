"""Python wrapper for the paged compressed-domain attention CUDA kernels.

Ops come from ``cda_vllm._cda_paged_kernels`` (built from the private
``csrc/cda_paged_kernels.cu`` source by ``setup.py``). The kernel layout
follows vLLM's paged KV convention:

  * ``block_tables``        :  (B, max_logical_blocks) int32 — logical→physical
  * ``packed_*_blocks``     :  (num_physical_blocks, block_size, D//4) uint8
  * ``norms_*_blocks``      :  (num_physical_blocks, block_size) float32

The 2-bit packed layout matches the host-side compressor
(``cda.HadamardQuantCompressor(bit_width=2)``) with MSB-first byte order.

Three ops are exposed:

  * ``score_paged2b_forward``   — Q · compressed_K → raw scores
  * ``vfull_paged2b_forward``   — softmax(scores) · compressed_V, all tokens
  * ``vsparse_paged2b_forward`` — softmax(scores) · compressed_V, Top-K only

``cuda_cda_paged`` composes them into a full end-to-end attention step with
an optional Top-K sparsity gate.
"""
from __future__ import annotations

import torch

try:
    from cda_vllm import _cda_paged_kernels as _mod
except ImportError as exc:
    raise ImportError(
        "cda_vllm._cda_paged_kernels is not built. Run `pip install -e .` "
        "or `python setup.py build_ext --inplace` from the repo root."
    ) from exc


def cuda_cda_paged(
    Q_rot: torch.Tensor,              # (B, D) float32, rotated Q
    packed_K_blocks: torch.Tensor,    # (num_blocks, block_size, D/4) uint8
    norms_K_blocks: torch.Tensor,     # (num_blocks, block_size) float32
    packed_V_blocks: torch.Tensor,    # (num_blocks, block_size, D/4) uint8
    norms_V_blocks: torch.Tensor,     # (num_blocks, block_size) float32
    block_tables: torch.Tensor,       # (B, max_logical_blocks) int32
    codebook_k: torch.Tensor,         # (4,) float32 — K unmapped centroids
    codebook_v: torch.Tensor,         # (4,) float32 — V unmapped centroids
    rotation: torch.Tensor,           # (D, D) float32 — inverse rotation
    scale: float,
    N: int,                           # total sequence length
    *,
    block_size: int = 16,
    attn_gate_topk: int = 0,
) -> torch.Tensor:
    """End-to-end paged CDA attention step (2-bit K/V only).

    The kernel resolves each logical token's physical page through
    ``block_tables`` and computes attention directly against compressed
    indices — no decompress step.

    Args:
        Q_rot: Rotated query, shape (B, D), float32 on CUDA.
        packed_K_blocks, norms_K_blocks: Paged compressed K.
        packed_V_blocks, norms_V_blocks: Paged compressed V.
        block_tables: vLLM-style logical→physical block mapping.
        codebook_k, codebook_v: Unmapped Lloyd-Max centroids.
        rotation: (D, D) inverse rotation matrix (Hadamard^T for CDA).
        scale: Attention scale, usually ``1/sqrt(D)``.
        N: Total sequence length (≤ ``max_logical_blocks × block_size``).
        block_size: Tokens per physical page (default 16, matches vLLM).
        attn_gate_topk: If > 0, renormalise Top-K weights and route through
            the sparse V kernel. ``0`` selects the dense path.

    Returns:
        Output tensor shape (B, D) in fp32, already inverse-rotated.
    """
    D = Q_rot.shape[1]

    scores = _mod.score_paged2b_forward(
        Q_rot, packed_K_blocks, norms_K_blocks, block_tables,
        codebook_k, N, block_size, scale,
    )
    attn = torch.softmax(scores, dim=-1)

    if 0 < attn_gate_topk < N:
        vals, idx = attn.topk(attn_gate_topk, dim=-1)
        vals = vals / (vals.sum(dim=-1, keepdim=True) + 1e-12)
        output_rot = _mod.vsparse_paged2b_forward(
            vals, idx.int(), packed_V_blocks, norms_V_blocks,
            block_tables, codebook_v,
            attn_gate_topk, N, D, block_size,
        )
    else:
        output_rot = _mod.vfull_paged2b_forward(
            attn, packed_V_blocks, norms_V_blocks,
            block_tables, codebook_v, N, D, block_size,
        )

    return output_rot @ rotation


# Low-level handles for power users.
score_paged2b_forward = _mod.score_paged2b_forward
vfull_paged2b_forward = _mod.vfull_paged2b_forward
vsparse_paged2b_forward = _mod.vsparse_paged2b_forward


__all__ = [
    "cuda_cda_paged",
    "score_paged2b_forward",
    "vfull_paged2b_forward",
    "vsparse_paged2b_forward",
]
