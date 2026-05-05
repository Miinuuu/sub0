"""Compressed-domain attention reference.

Pure PyTorch implementation that defines correctness for
``cda/kernels/*``. The kernel implementations must match this module's
output at ``cos ≥ 0.9999`` across the test sweep.

Two regimes:
    decode  (M_q = 1)   — single query token × N keys ; bandwidth-bound
                          on GPU; this is where compression delivers
                          a ~5× speedup vs fp16.
    prefill (M_q > 1)   — chunked-prefill / multi-token Q × N keys ;
                          compute-bound; compression has neutral or
                          negative effect on speed (memory savings only).

Both regimes share the same math:

    Q_rot  = Q · H              (Hadamard rotation)
    K_dec  = cb_K[idx_K] · norm_K          (decode in rotated frame)
    V_dec  = cb_V[idx_V] · norm_V
    P      = softmax(Q_rot · K_dec^T · scale)        (causal mask if prefill)
    O_rot  = P · V_dec
    O      = O_rot · H^T                              (un-rotate)

Reference uses fp32 internals for stability; the result is cast to fp16
for comparison with kernels.

GQA: ``H_q = group_size · H_kv``. Each q-head ``h`` consumes K/V from
``h // group_size``. Implemented via einsum / explicit indexing rather
than ``repeat_interleave`` (saves an O(group_size·N·D) materialization).
"""

import math
from typing import Optional

import torch
from torch import Tensor

from .compression import Compressor, CompressedSlot, unpack_4bit


# ---------------------------------------------------------------------------
# Core attention math (fp32 reference)
# ---------------------------------------------------------------------------


def causal_mask(m_q: int, m_k: int, *, q_offset: int = 0,
                  device: torch.device | str | None = None) -> Tensor:
    """Lower-right causal mask: query token ``q`` may attend to keys
    ``[0, q + q_offset]``.

    Returns shape (M_q, M_k) bool, ``True`` where attention is allowed.
    Use the standard "set masked = -inf before softmax" pattern when
    consuming.
    """
    q = torch.arange(m_q, device=device).unsqueeze(1) + q_offset
    k = torch.arange(m_k, device=device).unsqueeze(0)
    return k <= q


def attention_reference(
    Q: Tensor,                # (B, H_q, M_q, D) fp32 / fp16
    K: Tensor,                # (B, H_kv, M_k, D)
    V: Tensor,                # (B, H_kv, M_k, D)
    *,
    group_size: int,
    scale: Optional[float] = None,
    causal: bool = False,
    q_offset: int = 0,
) -> Tensor:
    """Standard GQA attention reference (no compression).

    Returns (B, H_q, M_q, D) fp16. Used as a baseline for compressed-domain
    comparisons.
    """
    B, H_q, M_q, D = Q.shape
    H_kv = K.shape[1]
    if H_q != H_kv * group_size:
        raise ValueError(f"H_q={H_q} != H_kv * group_size = {H_kv * group_size}")
    M_k = K.shape[2]
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    Qf = Q.float()
    Kf = K.float()
    Vf = V.float()
    # GQA: reshape Q to (B, H_kv, group_size, M_q, D) so we can einsum vs (B, H_kv, M_k, D)
    Q_g = Qf.view(B, H_kv, group_size, M_q, D)
    scores = torch.einsum("bhgmd,bhkd->bhgmk", Q_g, Kf) * scale
    if causal:
        mask = causal_mask(M_q, M_k, q_offset=q_offset, device=Q.device)
        scores = scores.masked_fill(~mask, float("-inf"))
    P = torch.softmax(scores, dim=-1)                       # (B, H_kv, g, M_q, M_k)
    O = torch.einsum("bhgmk,bhkd->bhgmd", P, Vf)            # (B, H_kv, g, M_q, D)
    return O.reshape(B, H_q, M_q, D).to(torch.float16)


# ---------------------------------------------------------------------------
# Compressed-domain attention (the cda primitive)
# ---------------------------------------------------------------------------


def qk_scores_uniform_4bit_without_k_decode(
    Q_rot: Tensor,                    # (B, H_q, M_q, D), rotated frame
    K_slot: CompressedSlot,           # idx (B, H_kv, M_k, D//2), norm (B, H_kv, M_k)
    *,
    group_size: int,
    scale: Optional[float] = None,
    causal: bool = False,
    q_offset: int = 0,
    key_start: int = 0,
    key_end: Optional[int] = None,
) -> Tensor:
    """Compute uniform 4-bit QK scores without materializing decoded K.

    For the canonical 16-level uniform codebook,

        K[k, d] = norm[k] * (code[k, d] / 8 - 15 / 16)

    so

        Q @ K[k] = norm[k] * ((Q @ code[k]) / 8 - 15 / 16 * sum(Q)).

    This is the score-domain contract for an aggressive QK path: K is used as
    integer codes plus row norm, not as an fp16/fp32 decoded tensor. The current
    implementation unpacks codes in PyTorch for validation; a kernel should keep
    codes packed and lower the ``Q @ code`` term to an integer-friendly mainloop.
    """
    B, H_q, M_q, D = Q_rot.shape
    H_kv = K_slot.idx.shape[1]
    if H_q != H_kv * group_size:
        raise ValueError(f"H_q={H_q} != H_kv * group_size = {H_kv * group_size}")
    if K_slot.idx.size(-1) * 2 != D:
        raise ValueError(f"K packed dim {K_slot.idx.size(-1)} does not match D={D}")
    n_keys = K_slot.norm.size(-1)
    if K_slot.idx.size(-2) != n_keys:
        raise ValueError("K_slot idx/norm key dimensions do not match")
    if key_end is None:
        key_end = n_keys
    if not (0 <= key_start <= key_end <= n_keys):
        raise ValueError(f"invalid key window [{key_start}, {key_end}) for length {n_keys}")
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    K_idx = K_slot.idx[..., key_start:key_end, :]
    K_norm = K_slot.norm[..., key_start:key_end].float()
    codes = unpack_4bit(K_idx, d=D).float()
    Q_g = Q_rot.float().view(B, H_kv, group_size, M_q, D)
    q_code = torch.einsum("bhgmd,bhkd->bhgmk", Q_g, codes)
    q_sum = Q_g.sum(dim=-1).unsqueeze(-1)
    scores = K_norm.unsqueeze(2).unsqueeze(3) * (q_code * 0.125 - q_sum * 0.9375)
    scores = scores * scale
    if causal:
        local_q_offset = q_offset - key_start
        mask = causal_mask(M_q, key_end - key_start, q_offset=local_q_offset, device=Q_rot.device)
        scores = scores.masked_fill(~mask, float("-inf"))
    return scores


def attention_compressed_rotated_reference_qk_without_k_decode(
    Q_rot: Tensor,
    K_slot: CompressedSlot,
    V_slot: CompressedSlot,
    *,
    cmp_V: Compressor,
    group_size: int,
    scale: Optional[float] = None,
    causal: bool = False,
    q_offset: int = 0,
    key_start: int = 0,
    key_end: Optional[int] = None,
) -> Tensor:
    """Compressed attention using score-domain QK and the existing decoded V path."""
    K_win, V_win, key_start, key_end = _compressed_key_window(
        K_slot, V_slot, key_start=key_start, key_end=key_end,
    )
    scores = qk_scores_uniform_4bit_without_k_decode(
        Q_rot,
        K_win,
        group_size=group_size,
        scale=scale,
        causal=causal,
        q_offset=q_offset - key_start if causal else q_offset,
    )
    V = cmp_V.decode(V_win, dtype=torch.float16, rotated=True)
    P = torch.softmax(scores, dim=-1)
    B, H_q, M_q, D = Q_rot.shape
    out = torch.einsum("bhgmk,bhkd->bhgmd", P, V.float())
    return out.reshape(B, H_q, M_q, D).to(torch.float16)


def attention_partial_reference(
    Q: Tensor,                # (B, H_q, M_q, D) fp32 / fp16 — IN ROTATED FRAME
    K: Tensor,                # (B, H_kv, M_k, D) — IN ROTATED FRAME (decoded)
    V: Tensor,                # (B, H_kv, M_k, D) — IN ROTATED FRAME (decoded)
    *,
    group_size: int,
    scale: Optional[float] = None,
    causal: bool = False,
    q_offset: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Reference forward that returns the un-normalized O + softmax stats.

    This is the **API contract** for any split-K decode kernel: each
    block produces these three outputs, and the host (or a reduce kernel)
    merges them via online softmax. Operates entirely in the rotated
    frame so kernels can produce partials without un-rotating per split.

    Returns:
        O_unnorm : (B, H_q, M_q, D) fp32 — sum_k exp(s_k - m) * V_k
        m        : (B, H_q, M_q)   fp32 — running row max
        l        : (B, H_q, M_q)   fp32 — sum_k exp(s_k - m)
    """
    B, H_q, M_q, D = Q.shape
    H_kv = K.shape[1]
    if H_q != H_kv * group_size:
        raise ValueError(f"H_q={H_q} != H_kv * group_size = {H_kv * group_size}")
    M_k = K.shape[2]
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    Qf = Q.float()
    Kf = K.float()
    Vf = V.float()
    Q_g = Qf.view(B, H_kv, group_size, M_q, D)
    scores = torch.einsum("bhgmd,bhkd->bhgmk", Q_g, Kf) * scale       # (B, H_kv, g, M_q, M_k)
    if causal:
        mask = causal_mask(M_q, M_k, q_offset=q_offset, device=Q.device)
        scores = scores.masked_fill(~mask, float("-inf"))
    m, _ = scores.max(dim=-1, keepdim=False)                           # (B, H_kv, g, M_q)
    P_unnorm = torch.exp(scores - m.unsqueeze(-1))                     # (B, H_kv, g, M_q, M_k)
    P_unnorm = torch.where(P_unnorm.isnan(), torch.zeros_like(P_unnorm), P_unnorm)
    l = P_unnorm.sum(dim=-1)                                            # (B, H_kv, g, M_q)
    O_unnorm = torch.einsum("bhgmk,bhkd->bhgmd", P_unnorm, Vf)         # (B, H_kv, g, M_q, D)
    return (
        O_unnorm.reshape(B, H_q, M_q, D),
        m.reshape(B, H_q, M_q),
        l.reshape(B, H_q, M_q),
    )


def merge_partials(
    partials: list[tuple[Tensor, Tensor, Tensor]],
) -> Tensor:
    """Online-softmax merge across split-K partials.

    Each partial is ``(O_unnorm, m, l)`` from
    :func:`attention_partial_reference` over a disjoint key range. The
    merge produces the normalized output as if the full key range had
    been processed in one pass.

    Math:
        m*  = max_i m_i
        l_i' = l_i * exp(m_i - m*)
        l*  = sum_i l_i'
        O_unnorm* = sum_i O_unnorm_i * exp(m_i - m*)
        O = O_unnorm* / l*
    """
    if not partials:
        raise ValueError("no partials to merge")
    if len(partials) == 1:
        O_un, m, l = partials[0]
        return (O_un / l.unsqueeze(-1).clamp_min(1e-30)).to(torch.float16)
    # Stack along a new "split" axis so reductions are vectorized.
    Os, ms, ls = zip(*partials)
    O_stack = torch.stack(Os, dim=0)              # (S, B, H_q, M_q, D)
    m_stack = torch.stack(ms, dim=0)              # (S, B, H_q, M_q)
    l_stack = torch.stack(ls, dim=0)
    m_global = m_stack.max(dim=0).values          # (B, H_q, M_q)
    correction = (m_stack - m_global.unsqueeze(0)).exp()  # (S, B, H_q, M_q)
    correction = torch.where(correction.isnan(), torch.zeros_like(correction), correction)
    l_global = (l_stack * correction).sum(dim=0)
    O_unnorm = (O_stack * correction.unsqueeze(-1)).sum(dim=0)
    O = O_unnorm / l_global.unsqueeze(-1).clamp_min(1e-30)
    return O.to(torch.float16)


def _compressed_key_window(
    K_slot: CompressedSlot,
    V_slot: CompressedSlot,
    *,
    key_start: int = 0,
    key_end: Optional[int] = None,
) -> tuple[CompressedSlot, CompressedSlot, int, int]:
    """Slice compressed K/V along the key axis without changing slot layout."""
    n_k = K_slot.norm.size(-1)
    n_v = V_slot.norm.size(-1)
    if K_slot.idx.size(-2) != n_k:
        raise ValueError("K_slot idx/norm key dimensions do not match")
    if V_slot.idx.size(-2) != n_v:
        raise ValueError("V_slot idx/norm key dimensions do not match")
    if n_k != n_v:
        raise ValueError(f"K/V key lengths differ: {n_k} vs {n_v}")
    if key_end is None:
        key_end = n_k
    if not (0 <= key_start <= key_end <= n_k):
        raise ValueError(f"invalid key window [{key_start}, {key_end}) for length {n_k}")
    K_win = CompressedSlot(
        idx=K_slot.idx[..., key_start:key_end, :].contiguous(),
        norm=K_slot.norm[..., key_start:key_end].contiguous(),
    )
    V_win = CompressedSlot(
        idx=V_slot.idx[..., key_start:key_end, :].contiguous(),
        norm=V_slot.norm[..., key_start:key_end].contiguous(),
    )
    return K_win, V_win, key_start, key_end


def attention_compressed_rotated_reference(
    Q_rot: Tensor,                    # (B, H_q, M_q, D) fp16/fp32, rotated frame
    K_slot: CompressedSlot,
    V_slot: CompressedSlot,
    *,
    cmp_K: Compressor,
    cmp_V: Compressor,
    group_size: int,
    scale: Optional[float] = None,
    causal: bool = False,
    q_offset: int = 0,
    key_start: int = 0,
    key_end: Optional[int] = None,
) -> Tensor:
    """Kernel-friendly compressed attention in the rotated frame.

    This is the dense-output contract for vLLM/FA2-style kernels. The
    kernel should receive already-rotated Q, synthesize K/V for the active
    key window from packed indices + codebooks + norms, and produce O in
    the rotated frame. No final Hadamard un-rotation is included here.

    ``key_start``/``key_end`` describe the absolute key range used by the
    local slot window. For causal windows, ``q_offset`` is adjusted so the
    mask is equivalent to applying lower-right causal masking in full-K
    coordinates.
    """
    K_win, V_win, key_start, _ = _compressed_key_window(
        K_slot, V_slot, key_start=key_start, key_end=key_end,
    )
    K = cmp_K.decode(K_win, dtype=torch.float16, rotated=True)
    V = cmp_V.decode(V_win, dtype=torch.float16, rotated=True)
    local_q_offset = q_offset - key_start if causal else q_offset
    return attention_reference(
        Q_rot,
        K,
        V,
        group_size=group_size,
        scale=scale,
        causal=causal,
        q_offset=local_q_offset,
    )


def attention_compressed_partial_reference(
    Q_rot: Tensor,                    # (B, H_q, M_q, D) fp16/fp32, rotated frame
    K_slot: CompressedSlot,
    V_slot: CompressedSlot,
    *,
    cmp_K: Compressor,
    cmp_V: Compressor,
    group_size: int,
    scale: Optional[float] = None,
    causal: bool = False,
    q_offset: int = 0,
    key_start: int = 0,
    key_end: Optional[int] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """FA2 split-K contract over compressed K/V.

    Returns the same ``(O_unnorm, m, l)`` partial as
    :func:`attention_partial_reference`, but accepts packed compressed
    slots and decodes only ``[key_start:key_end)``. This is the reference
    a compressed FA2 tile kernel should match before merge.
    """
    K_win, V_win, key_start, _ = _compressed_key_window(
        K_slot, V_slot, key_start=key_start, key_end=key_end,
    )
    K = cmp_K.decode(K_win, dtype=torch.float16, rotated=True)
    V = cmp_V.decode(V_win, dtype=torch.float16, rotated=True)
    local_q_offset = q_offset - key_start if causal else q_offset
    return attention_partial_reference(
        Q_rot,
        K,
        V,
        group_size=group_size,
        scale=scale,
        causal=causal,
        q_offset=local_q_offset,
    )


def attention_compressed_split_reference(
    Q_rot: Tensor,                    # (B, H_q, M_q, D) fp16/fp32, rotated frame
    K_slot: CompressedSlot,
    V_slot: CompressedSlot,
    *,
    cmp_K: Compressor,
    cmp_V: Compressor,
    group_size: int,
    k_splits: int,
    scale: Optional[float] = None,
    causal: bool = False,
    q_offset: int = 0,
) -> Tensor:
    """Split-K compressed attention reference in FA2 partial/merge form."""
    if k_splits < 1:
        raise ValueError(f"k_splits must be >= 1, got {k_splits}")
    n_keys = K_slot.norm.size(-1)
    if n_keys <= 0:
        raise ValueError("compressed attention requires at least one key")
    chunk = (n_keys + k_splits - 1) // k_splits
    partials = [
        attention_compressed_partial_reference(
            Q_rot,
            K_slot,
            V_slot,
            cmp_K=cmp_K,
            cmp_V=cmp_V,
            group_size=group_size,
            scale=scale,
            causal=causal,
            q_offset=q_offset,
            key_start=start,
            key_end=min(start + chunk, n_keys),
        )
        for start in range(0, n_keys, chunk)
    ]
    return merge_partials(partials)


def attention_compressed(
    Q: Tensor,                       # (B, H_q, M_q, D) fp16 — un-rotated query
    K_slot: CompressedSlot,          # idx (B, H_kv, M_k, D//2) uint8;
                                      # norm (B, H_kv, M_k) fp32
    V_slot: CompressedSlot,          # idx (B, H_kv, M_k, D/dims_per_byte) uint8;
                                      # norm (B, H_kv, M_k) fp32
    *,
    cmp_K: Compressor,
    cmp_V: Compressor,
    group_size: int,
    scale: Optional[float] = None,
    causal: bool = False,
    q_offset: int = 0,
    rotated: bool = False,
) -> Tensor:
    """Compressed-domain attention reference.

    Decodes K, V from their compressed form once (pure-PyTorch), then runs
    the standard attention math. This is the **correctness oracle** for
    every kernel that operates on compressed K/V — they must match this
    output at ``cos ≥ 0.9999`` across the test sweep.

    Args:
        Q: query tensor, un-rotated unless ``rotated=True``. Either way
            the rotation is applied / skipped consistently across Q and the
            decoded K/V so the answer matches the un-rotated baseline.
        K_slot, V_slot: compressed K and V (output of
            :meth:`Compressor.encode`).
        cmp_K, cmp_V: the compressors used to encode K_slot/V_slot. We
            need them for the rotation matrix and codebooks.
        group_size: GQA group size (4 for Llama-3.1-8B, 8 for -70B).
        scale: softmax scale (default 1/sqrt(D)).
        causal: apply lower-right causal mask.
        q_offset: query position offset (for chunked prefill).
        rotated: if True, ``Q`` is already in rotated coords and the output
            stays in rotated coords (skip the final un-rotation). Used by
            CuTeDSL kernels that operate fully in the rotated frame.

    Returns:
        (B, H_q, M_q, D) fp16, in original coords (or rotated coords if
        ``rotated=True``).
    """
    Q_rot = Q if rotated else cmp_K.rotate(Q)
    O_rot = attention_compressed_rotated_reference(
        Q_rot,
        K_slot,
        V_slot,
        cmp_K=cmp_K,
        cmp_V=cmp_V,
        group_size=group_size,
        scale=scale,
        causal=causal,
        q_offset=q_offset,
    )
    if rotated:
        return O_rot
    return cmp_K.unrotate(O_rot).to(torch.float16)
