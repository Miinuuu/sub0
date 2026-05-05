"""FlashAttention-csrc based CDA prefill prototype.

This is the first FA2-skeleton implementation path:

1. Rotate Q with the same Hadamard transform used by the compressed cache.
2. Use a CUDA CDA loader to materialize dense rotated K/V from paged slots.
3. Call ``flash_attn_2_cuda.varlen_fwd`` unchanged.
4. Inverse-rotate the output back to the model frame.

The dense K/V materialization is intentionally a stage-0 bridge. It gives us a
correct FA2-csrc baseline and a clean loader contract before moving the loader
inside FA2's global-to-shared K/V copy path.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch
from torch import Tensor

from cda.kernels_cuda.hmma_loader import (
    load_fa2_cda_hadamard,
    load_fa2_cda_loader,
    load_fa2_cda_varlen_fused,
)

try:
    import fast_hadamard_transform as _FHT
except ImportError:  # pragma: no cover - depends on local optional package
    _FHT = None


def _hadamard_or_matmul(
    x: Tensor,
    rotation_fp32: Tensor,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    D = x.shape[-1]
    if (
        D == 128
        and x.is_cuda
        and x.dtype == torch.float16
        and x.is_contiguous()
        and (out is None or (out.is_cuda and out.dtype == torch.float16 and out.is_contiguous()))
    ):
        y = torch.empty_like(x) if out is None else out
        mod = load_fa2_cda_hadamard()
        mod.hadamard128_fp16(x, y, 1.0 / math.sqrt(D))
        return y
    if _FHT is not None and (D & (D - 1)) == 0:
        y = _FHT.hadamard_transform(x.contiguous(), scale=1.0 / math.sqrt(D))
    else:
        y = torch.matmul(x.float(), rotation_fp32).to(torch.float16)
    if out is not None:
        out.copy_(y)
        return out
    return y


def paged_cache_to_dense_rotated(
    kv_cache: Tensor,
    block_table: Tensor,
    seq_lens: Tensor,
    *,
    cb_K: Tensor,
    cb_V: Tensor,
    block_size: int,
    max_seq_len: int,
    k4v2: bool = False,
) -> tuple[Tensor, Tensor]:
    """Materialize FA2-compatible dense rotated K/V from CDA paged cache.

    Returns tensors shaped ``(B * max_seq_len, H_kv, 128)``. Rows beyond each
    request's ``seq_lens[b]`` are zero-filled and hidden from FA2 by passing
    ``seqused_k=seq_lens``.
    """
    block_table_i32 = (
        block_table.to(torch.int32) if block_table.dtype != torch.int32 else block_table
    )
    seq_i32 = seq_lens.to(torch.int32) if seq_lens.dtype != torch.int32 else seq_lens
    mod = load_fa2_cda_loader(k4v2=k4v2)
    k_rot, v_rot = mod.cda_fa2_paged_cache_to_dense_rotated(
        kv_cache,
        block_table_i32,
        seq_i32,
        cb_K.contiguous(),
        cb_V.contiguous(),
        int(block_size),
        int(max_seq_len),
    )
    return k_rot, v_rot


def fa2_cda_prefill_varlen_v1(
    Q_fp16: Tensor,           # (T, H_q, D) fp16, unrotated
    kv_cache: Tensor,         # (num_slots, H_kv, slot_w) uint8 K4V4
    block_table: Tensor,      # (B, max_blocks) int32
    cu_seqlens_q: Tensor,     # (B+1,) int32
    seq_lens: Tensor,         # (B,) int32, full key lengths
    output: Tensor,           # (T, H_q, D) fp16, written in place
    *,
    cb_K: Tensor,
    cb_V: Tensor,
    rotation_fp32: Tensor,
    group_size: int,
    block_size: int,
    max_query_len: int,
    max_seq_len: int,
    scale: Optional[float] = None,
    causal: bool = True,
    k4v2: bool = False,
    return_dense_kv: bool = False,
    fuse_fa2_epilogue: bool = False,
) -> tuple[Tensor, Tensor] | None:
    """Run full-compressed prefill through FlashAttention csrc.

    This mirrors vLLM's varlen call shape.  ``max_query_len`` is the current
    chunk length and ``seq_lens`` is the full key length after cache update.
    With ``causal=True``, FA2's bottom-right causal alignment gives the same
    mask as ``k <= seq_len - max_query_len + q``.
    """
    if Q_fp16.dim() != 3:
        raise ValueError("Q_fp16 must be (T, H_q, D)")
    if output.shape != Q_fp16.shape:
        raise ValueError("output shape must match Q_fp16")
    if output.dtype != torch.float16:
        raise ValueError("output must be fp16")
    if Q_fp16.dtype != torch.float16:
        Q_in = Q_fp16.to(torch.float16)
    else:
        Q_in = Q_fp16

    T, H_q, D = Q_in.shape
    B = int(cu_seqlens_q.numel()) - 1
    if B <= 0:
        return None
    if H_q % group_size != 0:
        raise ValueError("H_q must be divisible by group_size")
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    cu_q = cu_seqlens_q.to(torch.int32) if cu_seqlens_q.dtype != torch.int32 else cu_seqlens_q
    seq_i32 = seq_lens.to(torch.int32) if seq_lens.dtype != torch.int32 else seq_lens
    k_rot, v_rot = paged_cache_to_dense_rotated(
        kv_cache,
        block_table,
        seq_i32,
        cb_K=cb_K,
        cb_V=cb_V,
        block_size=block_size,
        max_seq_len=max_seq_len,
        k4v2=k4v2,
    )

    cu_k = torch.arange(
        0,
        B + 1,
        dtype=torch.int32,
        device=Q_in.device,
    ) * int(max_seq_len)

    if fuse_fa2_epilogue:
        if not causal:
            raise NotImplementedError("fused FA2-CDA epilogue path is causal-only")
        mod = load_fa2_cda_varlen_fused()
        mod.cda_fa2_varlen_fwd_hadamard(
            Q_in.contiguous(),
            k_rot,
            v_rot,
            output,
            cu_q.contiguous(),
            cu_k.contiguous(),
            seq_i32.contiguous(),
            int(max_query_len),
            int(max_seq_len),
            float(scale),
        )
    else:
        q_rot = _hadamard_or_matmul(Q_in, rotation_fp32)
        fa2_out = output if output.is_contiguous() else None
        try:
            from vllm.vllm_flash_attn import flash_attn_varlen_func
            use_cu_k = bool(torch.all(seq_i32 == int(max_seq_len)).item())
            out_rot = flash_attn_varlen_func(
                q_rot,
                k_rot,
                v_rot,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k if use_cu_k else None,
                seqused_k=None if use_cu_k else seq_i32,
                max_seqlen_q=int(max_query_len),
                max_seqlen_k=int(max_seq_len),
                causal=bool(causal),
                softmax_scale=float(scale),
                out=fa2_out,
            )
            if isinstance(out_rot, tuple):
                out_rot = out_rot[0]
        except (AttributeError, ImportError):
            import flash_attn_2_cuda

            out_rot, *_ = flash_attn_2_cuda.varlen_fwd(
                q_rot,
                k_rot,
                v_rot,
                fa2_out,    # out: reuse final buffer for rotated output when possible
                cu_q,
                cu_k,
                seq_i32,    # seqused_k
                None,       # block_table: dense K/V bridge has already paged-gathered
                None,       # alibi_slopes
                int(max_query_len),
                int(max_seq_len),
                0.0,        # dropout_p
                float(scale),
                False,      # zero_tensors
                bool(causal),
                -1,         # window_size_left
                -1,         # window_size_right
                False,      # return_softmax
                None,       # generator
            )
        _hadamard_or_matmul(out_rot.contiguous(), rotation_fp32, out=output)
    if return_dense_kv:
        return k_rot, v_rot
    return None


def _raw_kv_to_dense_rotated(
    K_fp16: Tensor,
    V_fp16: Tensor,
    cu_seqlens_q: Tensor,
    *,
    rotation_fp32: Tensor,
    max_query_len: int,
    max_seq_len: int,
) -> tuple[Tensor, Tensor]:
    """Rotate fresh fp16 K/V into the dense FA2 varlen K layout.

    The fast path is full/nocache prefill where ``max_query_len ==
    max_seq_len`` and K/V are already request-major dense.  The padded path is
    kept for correctness experiments, but chunked prefill with compressed past
    should use the hybrid raw-current + compressed-past path in the vLLM
    integration instead of this helper.
    """
    if K_fp16.dim() != 3 or V_fp16.dim() != 3:
        raise ValueError("K_fp16 and V_fp16 must be (T, H_kv, D)")
    if K_fp16.shape != V_fp16.shape:
        raise ValueError("K_fp16 and V_fp16 shapes must match")
    B = int(cu_seqlens_q.numel()) - 1
    if B <= 0:
        return K_fp16, V_fp16

    K_in = K_fp16.to(torch.float16) if K_fp16.dtype != torch.float16 else K_fp16
    V_in = V_fp16.to(torch.float16) if V_fp16.dtype != torch.float16 else V_fp16
    k_rot_flat = _hadamard_or_matmul(K_in.contiguous(), rotation_fp32)
    v_rot_flat = _hadamard_or_matmul(V_in.contiguous(), rotation_fp32)

    dense_tokens = B * int(max_seq_len)
    if k_rot_flat.shape[0] == dense_tokens:
        return k_rot_flat.contiguous(), v_rot_flat.contiguous()
    if int(max_query_len) == int(max_seq_len):
        raise ValueError(
            "raw K/V token count does not match B * max_seq_len for full prefill"
        )

    H_kv, D = k_rot_flat.shape[1], k_rot_flat.shape[2]
    k_rot = torch.zeros(
        dense_tokens, H_kv, D, dtype=torch.float16, device=K_in.device
    )
    v_rot = torch.zeros_like(k_rot)
    cu_cpu = cu_seqlens_q.detach().to("cpu", non_blocking=False)
    for b in range(B):
        src0 = int(cu_cpu[b].item())
        src1 = int(cu_cpu[b + 1].item())
        dst0 = b * int(max_seq_len)
        length = src1 - src0
        if length > int(max_seq_len):
            raise ValueError("raw K/V request length exceeds max_seq_len")
        k_rot[dst0:dst0 + length].copy_(k_rot_flat[src0:src1])
        v_rot[dst0:dst0 + length].copy_(v_rot_flat[src0:src1])
    return k_rot, v_rot


def fa2_cda_prefill_rawkv_varlen_v1(
    Q_fp16: Tensor,           # (T, H_q, D) fp16, unrotated
    K_fp16: Tensor,           # (T, H_kv, D) fp16, fresh uncompressed K
    V_fp16: Tensor,           # (T, H_kv, D) fp16, fresh uncompressed V
    cu_seqlens_q: Tensor,     # (B+1,) int32
    seq_lens: Tensor,         # (B,) int32, key lengths for dense K layout
    output: Tensor,           # (T, H_q, D) fp16, written in place
    *,
    rotation_fp32: Tensor,
    group_size: int,
    max_query_len: int,
    max_seq_len: int,
    scale: Optional[float] = None,
    causal: bool = True,
) -> None:
    """Run no-past prefill through the fused FA2-CDA path using raw K/V.

    This bypasses the compressed-cache bridge for TTFT-sensitive first-chunk
    prefill while preserving the fused Q Hadamard and output inverse-Hadamard
    kernel.  It is intentionally not the chunked-past implementation: when
    ``seq_lens > max_query_len`` the vLLM integration routes to a hybrid path
    that attends raw current-chunk K/V plus compressed paged past K/V.
    """
    if Q_fp16.dim() != 3:
        raise ValueError("Q_fp16 must be (T, H_q, D)")
    if K_fp16.dim() != 3 or V_fp16.dim() != 3:
        raise ValueError("K_fp16 and V_fp16 must be (T, H_kv, D)")
    if K_fp16.shape != V_fp16.shape:
        raise ValueError("K_fp16 and V_fp16 shapes must match")
    if output.shape != Q_fp16.shape:
        raise ValueError("output shape must match Q_fp16")
    if output.dtype != torch.float16:
        raise ValueError("output must be fp16")
    if not causal:
        raise NotImplementedError("fa2_cda_prefill_rawkv_varlen_v1 is causal-only")

    Q_in = Q_fp16.to(torch.float16) if Q_fp16.dtype != torch.float16 else Q_fp16
    T, H_q, D = Q_in.shape
    if K_fp16.shape[0] != T:
        raise ValueError("raw K/V must have the same token count as Q")
    if K_fp16.shape[2] != D:
        raise ValueError("Q/K/V head dimensions must match")
    H_kv = K_fp16.shape[1]
    if H_q != H_kv * int(group_size):
        raise ValueError("H_q must equal H_kv * group_size")
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    cu_q = cu_seqlens_q.to(torch.int32) if cu_seqlens_q.dtype != torch.int32 else cu_seqlens_q
    seq_i32 = seq_lens.to(torch.int32) if seq_lens.dtype != torch.int32 else seq_lens
    B = int(cu_q.numel()) - 1
    if B <= 0:
        return

    k_rot, v_rot = _raw_kv_to_dense_rotated(
        K_fp16,
        V_fp16,
        cu_q,
        rotation_fp32=rotation_fp32,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
    )
    cu_k = torch.arange(
        0,
        B + 1,
        dtype=torch.int32,
        device=Q_in.device,
    ) * int(max_seq_len)

    mod = load_fa2_cda_varlen_fused()
    mod.cda_fa2_varlen_fwd_hadamard(
        Q_in.contiguous(),
        k_rot,
        v_rot,
        output,
        cu_q.contiguous(),
        cu_k.contiguous(),
        seq_i32.contiguous(),
        int(max_query_len),
        int(max_seq_len),
        float(scale),
    )


def fa2_raw_prefill_varlen_v1(
    Q_fp16: Tensor,           # (T, H_q, D) fp16, unrotated
    K_fp16: Tensor,           # (T, H_kv, D) fp16, fresh uncompressed K
    V_fp16: Tensor,           # (T, H_kv, D) fp16, fresh uncompressed V
    cu_seqlens_q: Tensor,     # (B+1,) int32
    output: Tensor,           # (T, H_q, D) fp16, written in place
    *,
    group_size: int,
    max_query_len: int,
    scale: Optional[float] = None,
    causal: bool = True,
) -> None:
    """Run no-past prefill exactly like vLLM FA2 over fresh raw Q/K/V.

    For a full prompt chunk with no compressed prefix, CDA's Hadamard rotation
    is orthonormal and raw attention is mathematically equivalent to rotating
    Q/K/V and inverse-rotating the output. This path avoids all prefill FHT and
    compressed-cache bridge work; the cache is still updated separately for
    subsequent compressed decode.
    """
    if Q_fp16.dim() != 3:
        raise ValueError("Q_fp16 must be (T, H_q, D)")
    if K_fp16.dim() != 3 or V_fp16.dim() != 3:
        raise ValueError("K_fp16 and V_fp16 must be (T, H_kv, D)")
    if K_fp16.shape != V_fp16.shape:
        raise ValueError("K_fp16 and V_fp16 shapes must match")
    if output.shape != Q_fp16.shape:
        raise ValueError("output shape must match Q_fp16")
    if output.dtype != torch.float16:
        raise ValueError("output must be fp16")
    if not causal:
        raise NotImplementedError("fa2_raw_prefill_varlen_v1 is causal-only")

    Q_in = Q_fp16.to(torch.float16) if Q_fp16.dtype != torch.float16 else Q_fp16
    K_in = K_fp16.to(torch.float16) if K_fp16.dtype != torch.float16 else K_fp16
    V_in = V_fp16.to(torch.float16) if V_fp16.dtype != torch.float16 else V_fp16
    T, H_q, D = Q_in.shape
    if K_in.shape[0] != T:
        raise ValueError("raw K/V must have the same token count as Q")
    if K_in.shape[2] != D:
        raise ValueError("Q/K/V head dimensions must match")
    H_kv = K_in.shape[1]
    if H_q != H_kv * int(group_size):
        raise ValueError("H_q must equal H_kv * group_size")
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    cu_q = cu_seqlens_q.to(torch.int32) if cu_seqlens_q.dtype != torch.int32 else cu_seqlens_q
    out_arg = output if output.is_contiguous() else None
    try:
        from vllm.vllm_flash_attn import flash_attn_varlen_func
        out = flash_attn_varlen_func(
            Q_in.contiguous(),
            K_in.contiguous(),
            V_in.contiguous(),
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_q,
            max_seqlen_q=int(max_query_len),
            max_seqlen_k=int(max_query_len),
            causal=True,
            softmax_scale=float(scale),
            out=out_arg,
        )
    except (AttributeError, ImportError):
        import flash_attn_2_cuda

        ret = flash_attn_2_cuda.varlen_fwd(
            Q_in.contiguous(),
            K_in.contiguous(),
            V_in.contiguous(),
            out_arg,
            cu_q,
            cu_q,
            None,       # seqused_k
            None,       # block_table
            None,       # alibi_slopes
            int(max_query_len),
            int(max_query_len),
            0.0,        # dropout_p
            float(scale),
            False,      # zero_tensors
            True,       # causal
            -1,         # window_size_left
            -1,         # window_size_right
            False,      # return_softmax
            None,       # generator
        )
        out = ret[0]
    if out.data_ptr() != output.data_ptr():
        output.copy_(out)


__all__ = [
    "fa2_cda_prefill_rawkv_varlen_v1",
    "fa2_cda_prefill_varlen_v1",
    "fa2_raw_prefill_varlen_v1",
    "paged_cache_to_dense_rotated",
]
