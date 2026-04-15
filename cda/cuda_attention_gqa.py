"""Python bindings for the GQA-aware CDA CUDA kernels.

The kernels are built into the ``cda._cda_gqa_kernels`` binary extension
that ships with this package. CUDA sources are kept in the private
maintainer tree. Unlike the per-head kernels in :mod:`cda.cuda_attention`,
these launch with grid ``(token_block, query_head)`` and index KV as
``kv_head = q_head // group_size`` inside the kernel — there is no
Python-side ``repeat_interleave`` for GQA.

Exposed kernels:
  * ``score_2b_gqa`` / ``score_4b_gqa``     — Q · compressed K  (scaled, GQA)
  * ``vfull_2b_gqa``                        — attn · compressed V  (dense)
  * ``vsparse_2b_gqa``                      — attn · compressed V  (TopK sparse)

:func:`cuda_hw_attention_gqa` composes these into a full per-layer
attention step on the compressed KV of one layer.

The shipped ``.so`` matches the tested platform (Python 3.10,
PyTorch 2.5+cu121, sm_86 / RTX A6000). Contact the authors for a
platform-specific binary drop if your stack differs.
"""
from __future__ import annotations

import torch

try:
    from cda import _cda_gqa_kernels as _gqa  # ships as binary extension
except ImportError as exc:
    raise ImportError(
        "cda._cda_gqa_kernels binary not found. The shipped .so matches the "
        "tested platform (Python 3.10, PyTorch 2.5+cu121, sm_86). Contact the "
        "authors for a platform-specific binary drop."
    ) from exc


def cuda_hw_attention_gqa(
    Q_rot: torch.Tensor,          # (H_q, D) float32
    packed_K: torch.Tensor,       # (H_kv, N, D/pack_k) uint8
    norms_K: torch.Tensor,        # (H_kv, N) float32
    packed_V: torch.Tensor,       # (H_kv, N, D/4) uint8  (2-bit V)
    norms_V: torch.Tensor,        # (H_kv, N) float32
    codebook_k: torch.Tensor,     # (2^k_bits,) float32 — signed Lloyd-Max centroids
    codebook_v: torch.Tensor,     # (4,) float32 — 2-bit V centroids
    rotation: torch.Tensor,       # (D, D) float32 — inverse rotation
    scale: float,
    N: int,
    group_size: int,
    k_bits: int = 4,
    topk: int = 0,
) -> torch.Tensor:
    """Full GQA-aware compressed-domain attention for one layer.

    Emits one score launch + one V launch (dense or TopK sparse), both with
    grid ``(token_block, H_q)``. KV is read per query head as
    ``kv_head = query_head // group_size`` — no ``repeat_interleave``.

    Returns ``(H_q, D)`` float32.
    """
    H_q, D = Q_rot.shape
    assert D <= 128, f"D={D} exceeds kernel shared memory (128)"
    assert H_q % group_size == 0, f"H_q={H_q} not divisible by group_size={group_size}"

    k_pack = D // (8 // k_bits)  # 2-bit → D/4, 4-bit → D/2
    pk = packed_K.reshape(-1, k_pack).contiguous()
    nk = norms_K.reshape(-1).contiguous()
    pv = packed_V.reshape(-1, D // 4).contiguous()
    nv = norms_V.reshape(-1).contiguous()

    if k_bits == 2:
        scores = _gqa.score_2b_gqa(Q_rot, pk, nk, codebook_k, N, group_size, scale)
    elif k_bits == 4:
        scores = _gqa.score_4b_gqa(Q_rot, pk, nk, codebook_k, N, group_size, scale)
    else:
        raise ValueError(f"unsupported k_bits={k_bits} (expected 2 or 4)")

    attn = torch.softmax(scores, dim=-1)

    if 0 < topk < N:
        vals, idx = attn.topk(topk, dim=-1)
        vals = vals / (vals.sum(dim=-1, keepdim=True) + 1e-12)
        out_rot = _gqa.vsparse_2b_gqa(
            vals, idx.int(), pv, nv, codebook_v, topk, N, D, group_size,
        )
    else:
        out_rot = _gqa.vfull_2b_gqa(attn, pv, nv, codebook_v, N, D, group_size)

    return out_rot @ rotation


# Re-export low-level bindings for power users.
score_2b_gqa = _gqa.score_2b_gqa
score_4b_gqa = _gqa.score_4b_gqa
vfull_2b_gqa = _gqa.vfull_2b_gqa
vsparse_2b_gqa = _gqa.vsparse_2b_gqa

__all__ = [
    "cuda_hw_attention_gqa",
    "score_2b_gqa",
    "score_4b_gqa",
    "vfull_2b_gqa",
    "vsparse_2b_gqa",
]
