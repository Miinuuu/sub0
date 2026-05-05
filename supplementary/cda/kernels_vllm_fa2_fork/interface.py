"""vLLM-compatible Python ABI for the local FA2 varlen fork."""

from __future__ import annotations

from collections.abc import Sequence

import torch

_OP_NAMESPACE = "_cda_vllm_fa2_fork_C"
# W3 EXPERIMENT: was 2048. Lowering to see if swap helps N=512-1024.
_GQA_DECODE_SWAP_MIN_CONTEXT = 256


# W1: cu_seqlens cache for gqa-decode-swap path. Avoids two GPU kernels per
# call (torch.arange + mul) by reusing pre-built tensors keyed by
# (batch_size, group_size, device_str).
_CU_SEQLENS_SWAP_CACHE: dict[tuple[int, int, str], torch.Tensor] = {}


def _cu_seqlens_swap_cached(batch_size: int, group_size: int,
                             device: torch.device) -> torch.Tensor:
    key = (int(batch_size), int(group_size), str(device))
    cached = _CU_SEQLENS_SWAP_CACHE.get(key)
    if cached is None:
        cached = (
            torch.arange(batch_size + 1, dtype=torch.int32, device=device)
            * group_size
        )
        _CU_SEQLENS_SWAP_CACHE[key] = cached
    return cached


def _maybe_contiguous(x: torch.Tensor | None) -> torch.Tensor | None:
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _normalize_window_size(window_size: Sequence[int] | None) -> tuple[int, int]:
    if window_size is None:
        return -1, -1
    if len(window_size) != 2:
        raise ValueError("window_size must contain exactly two values")
    return int(window_size[0]), int(window_size[1])


def _is_fp8_dtype(dtype: torch.dtype) -> bool:
    return dtype in {
        getattr(torch, "float8_e4m3fn", object()),
        getattr(torch, "float8_e5m2", object()),
    }


def _resolve_gqa_decode_swap(enabled: bool | None, max_seqlen_k: int) -> bool:
    if enabled is not None:
        return bool(enabled)
    return int(max_seqlen_k) >= _GQA_DECODE_SWAP_MIN_CONTEXT


def _load_varlen_op():
    namespace = getattr(torch.ops, _OP_NAMESPACE, None)
    if namespace is not None and hasattr(namespace, "varlen_fwd"):
        return namespace.varlen_fwd

    from .build import load

    load()
    namespace = getattr(torch.ops, _OP_NAMESPACE, None)
    if namespace is None or not hasattr(namespace, "varlen_fwd"):
        raise RuntimeError(f"{_OP_NAMESPACE}.varlen_fwd was not registered")
    return namespace.varlen_fwd


def _load_dequant_op():
    namespace = getattr(torch.ops, _OP_NAMESPACE, None)
    if namespace is not None and hasattr(namespace, "dequantize_compressed_kv"):
        return namespace.dequantize_compressed_kv

    from .build import load

    load()
    namespace = getattr(torch.ops, _OP_NAMESPACE, None)
    if namespace is None or not hasattr(namespace, "dequantize_compressed_kv"):
        raise RuntimeError(f"{_OP_NAMESPACE}.dequantize_compressed_kv was not registered")
    return namespace.dequantize_compressed_kv


def _load_varlen_compressed_op():
    namespace = getattr(torch.ops, _OP_NAMESPACE, None)
    if namespace is not None and hasattr(namespace, "varlen_fwd_compressed_kv"):
        return namespace.varlen_fwd_compressed_kv

    from .build import load

    load()
    namespace = getattr(torch.ops, _OP_NAMESPACE, None)
    if namespace is None or not hasattr(namespace, "varlen_fwd_compressed_kv"):
        raise RuntimeError(f"{_OP_NAMESPACE}.varlen_fwd_compressed_kv was not registered")
    return namespace.varlen_fwd_compressed_kv


def _load_varlen_compressed_fused_op():
    namespace = getattr(torch.ops, _OP_NAMESPACE, None)
    if namespace is not None and hasattr(namespace, "varlen_fwd_compressed_kv_fused"):
        return namespace.varlen_fwd_compressed_kv_fused

    from .build import load

    load()
    namespace = getattr(torch.ops, _OP_NAMESPACE, None)
    if namespace is None or not hasattr(namespace, "varlen_fwd_compressed_kv_fused"):
        raise RuntimeError(f"{_OP_NAMESPACE}.varlen_fwd_compressed_kv_fused was not registered")
    return namespace.varlen_fwd_compressed_kv_fused


def _flatten_compressed_kv(
    idx: torch.Tensor,
    norm: torch.Tensor,
    *,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if idx.dim() == 3:
        if norm.dim() != 2:
            raise ValueError(f"{name}_norm must be (total_k, H_kv) for 3D idx")
        return idx.contiguous(), norm.contiguous()
    if idx.dim() == 4:
        if norm.dim() != 3:
            raise ValueError(f"{name}_norm must be (B, H_kv, N) for 4D idx")
        if idx.shape[:3] != norm.shape:
            raise ValueError(f"{name}_idx and {name}_norm shape mismatch")
        # Reference compressor layout is (B, H_kv, N, D/2). FA2 varlen expects
        # contiguous keys as (total_k, H_kv, D), so use B-major token order.
        return (
            idx.permute(0, 2, 1, 3).reshape(-1, idx.shape[1], idx.shape[3]).contiguous(),
            norm.permute(0, 2, 1).reshape(-1, norm.shape[1]).contiguous(),
        )
    raise ValueError(f"{name}_idx must be 3D or 4D, got {idx.dim()}D")


def dequantize_compressed_kv(
    k_idx: torch.Tensor,
    k_norm: torch.Tensor,
    v_idx: torch.Tensor,
    v_norm: torch.Tensor,
    cb_k: torch.Tensor,
    cb_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize rotated-frame compressed K/V for the FA2 production body."""
    k_idx_flat, k_norm_flat = _flatten_compressed_kv(k_idx, k_norm, name="k")
    v_idx_flat, v_norm_flat = _flatten_compressed_kv(v_idx, v_norm, name="v")
    if k_idx_flat.shape != v_idx_flat.shape:
        raise ValueError("K/V idx shape mismatch after flattening")
    if k_norm_flat.shape != v_norm_flat.shape:
        raise ValueError("K/V norm shape mismatch after flattening")
    dequant_op = _load_dequant_op()
    k, v = dequant_op(
        k_idx_flat,
        k_norm_flat,
        v_idx_flat,
        v_norm_flat,
        cb_k.contiguous(),
        cb_v.contiguous(),
    )
    return k, v


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int,
    cu_seqlens_k: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None,
    q_v: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: Sequence[int] | None = None,
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    block_table: torch.Tensor | None = None,
    return_softmax_lse: bool = False,
    out: torch.Tensor | None = None,
    scheduler_metadata: torch.Tensor | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    num_splits: int = 0,
    fa_version: int | None = 2,
    s_aux: torch.Tensor | None = None,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_tot_seqused_k: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Call the forked FA2 varlen op with vLLM's public FA ABI."""
    if fa_version not in (None, 2):
        raise ValueError(f"FA2 fork only supports fa_version=2, got {fa_version}")
    if cu_seqlens_k is None and seqused_k is None:
        raise ValueError("cu_seqlens_k or seqused_k must be provided")
    if cu_seqlens_k is not None and seqused_k is not None:
        raise ValueError("cu_seqlens_k and seqused_k cannot both be provided")
    if block_table is not None and seqused_k is None:
        raise ValueError("seqused_k must be provided when block_table is provided")
    if q_v is not None:
        raise NotImplementedError("FA2 fork does not support q_v")
    if scheduler_metadata is not None:
        raise NotImplementedError("FA2 fork does not support scheduler_metadata")
    if q_descale is not None or k_descale is not None or v_descale is not None:
        if _is_fp8_dtype(q.dtype) or _is_fp8_dtype(k.dtype) or _is_fp8_dtype(v.dtype):
            raise NotImplementedError("FA2 fork does not support FP8 descale tensors")
        # vLLM passes unity descale tensors on the ordinary FP16/BF16 path.
        # The forked FA2 op has no descale ABI, so non-FP8 descales are ignored.
    if s_aux is not None:
        raise NotImplementedError("FA2 fork does not support sink tokens")
    if cp_world_size != 1 or cp_rank != 0 or cp_tot_seqused_k is not None:
        raise NotImplementedError("FA2 fork does not support context-parallel ABI args")
    if num_splits > 1:
        raise NotImplementedError("FA2 fork initial path does not support split-KV")
    if return_attn_probs:
        raise NotImplementedError("FA2 fork does not return attention probabilities")

    _ = deterministic

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    real_window_size = _normalize_window_size(window_size)
    q, k, v = [_maybe_contiguous(x) for x in (q, k, v)]
    dummy_cu_seqlens_k = torch.empty_like(cu_seqlens_q)
    varlen_fwd = _load_varlen_op()

    out, softmax_lse = varlen_fwd(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        dummy_cu_seqlens_k if cu_seqlens_k is None else cu_seqlens_k,
        seqused_k,
        None,
        block_table,
        alibi_slopes,
        int(max_seqlen_q),
        int(max_seqlen_k),
        float(dropout_p),
        float(softmax_scale),
        False,
        bool(causal),
        real_window_size[0],
        real_window_size[1],
        float(softcap),
        bool(return_softmax_lse and dropout_p > 0.0),
        int(num_splits),
        None,
    )
    return (out, softmax_lse) if return_softmax_lse else out


def flash_attn_varlen_compressed_kv_func(
    q: torch.Tensor,
    k_idx: torch.Tensor,
    k_norm: torch.Tensor,
    v_idx: torch.Tensor,
    v_norm: torch.Tensor,
    cb_k: torch.Tensor,
    cb_v: torch.Tensor,
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int,
    cu_seqlens_k: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: Sequence[int] | None = None,
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    return_softmax_lse: bool = False,
    num_splits: int = 0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run compressed-KV attention through the existing FA2 varlen body.

    Inputs are already in the rotated frame. This adapter materializes fp16 K/V
    scratch tensors and then calls ``flash_attn_varlen_func`` unchanged. It is
    a correctness and benchmarking bridge before fusing decode into FA2's K/V
    tile load path.
    """
    if dropout_p != 0.0:
        raise NotImplementedError("compressed-KV adapter is inference-only")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    real_window_size = _normalize_window_size(window_size)
    k_idx_flat, k_norm_flat = _flatten_compressed_kv(k_idx, k_norm, name="k")
    v_idx_flat, v_norm_flat = _flatten_compressed_kv(v_idx, v_norm, name="v")
    if k_idx_flat.shape != v_idx_flat.shape:
        raise ValueError("K/V idx shape mismatch after flattening")
    if k_norm_flat.shape != v_norm_flat.shape:
        raise ValueError("K/V norm shape mismatch after flattening")

    varlen_fwd_compressed = _load_varlen_compressed_op()
    out, softmax_lse = varlen_fwd_compressed(
        q,
        k_idx_flat,
        k_norm_flat,
        v_idx_flat,
        v_norm_flat,
        cb_k.contiguous(),
        cb_v.contiguous(),
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        alibi_slopes,
        int(max_seqlen_q),
        int(max_seqlen_k),
        float(dropout_p),
        float(softmax_scale),
        False,
        bool(causal),
        real_window_size[0],
        real_window_size[1],
        float(softcap),
        bool(return_softmax_lse and dropout_p > 0.0),
        int(num_splits),
        None,
    )
    return (out, softmax_lse) if return_softmax_lse else out


def flash_attn_varlen_compressed_kv_fused_func(
    q: torch.Tensor,
    k_idx: torch.Tensor,
    k_norm: torch.Tensor,
    v_idx: torch.Tensor,
    v_norm: torch.Tensor,
    cb_k: torch.Tensor,
    cb_v: torch.Tensor,
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int,
    cu_seqlens_k: torch.Tensor | None = None,
    *,
    seqused_k: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: Sequence[int] | None = None,
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    return_softmax_lse: bool = False,
    num_splits: int = 0,
    gqa_decode_swap: bool | None = None,
    uniform_codebook: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Experimental direct compressed-tile-load FA2 path.

    This path avoids materializing fp16 K/V scratch tensors. It currently
    targets the narrow correctness slice needed before deeper vLLM integration:
    fp16, D=128, flattened varlen K/V or paged KV cache.
    """
    if dropout_p != 0.0:
        raise NotImplementedError("compressed-KV fused path is inference-only")
    if block_table is not None and seqused_k is None:
        raise ValueError("seqused_k must be provided when block_table is provided")
    if cu_seqlens_k is None:
        if seqused_k is None:
            raise ValueError("cu_seqlens_k or seqused_k must be provided")
        cu_seqlens_k = torch.zeros(
            int(cu_seqlens_q.numel()), dtype=torch.int32, device=cu_seqlens_q.device
        )
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    real_window_size = _normalize_window_size(window_size)
    paged_kv = block_table is not None
    if paged_kv:
        if k_idx.dim() != 4 or v_idx.dim() != 4:
            raise ValueError("paged k_idx/v_idx must be (num_blocks, block_size, H_kv, D/2)")
        if k_norm.dim() != 3 or v_norm.dim() != 3:
            raise ValueError("paged k_norm/v_norm must be (num_blocks, block_size, H_kv)")
        if k_idx.shape != v_idx.shape:
            raise ValueError("K/V idx shape mismatch")
        if k_norm.shape != v_norm.shape or k_norm.shape != k_idx.shape[:3]:
            raise ValueError("K/V idx/norm shape mismatch")
        k_idx_arg = k_idx.contiguous()
        k_norm_arg = k_norm.contiguous()
        v_idx_arg = v_idx.contiguous()
        v_norm_arg = v_norm.contiguous()
    else:
        k_idx_arg, k_norm_arg = _flatten_compressed_kv(k_idx, k_norm, name="k")
        v_idx_arg, v_norm_arg = _flatten_compressed_kv(v_idx, v_norm, name="v")
        if k_idx_arg.shape != v_idx_arg.shape:
            raise ValueError("K/V idx shape mismatch after flattening")
        if k_norm_arg.shape != v_norm_arg.shape:
            raise ValueError("K/V norm shape mismatch after flattening")

    batch_size = int(cu_seqlens_q.numel() - 1)
    h_q = int(q.shape[1])
    h_kv = int(k_idx_arg.shape[2] if paged_kv else k_idx_arg.shape[1])
    group_size = h_q // h_kv if h_kv > 0 else 0
    use_gqa_decode_swap = (
        _resolve_gqa_decode_swap(gqa_decode_swap, max_seqlen_k)
        and not return_softmax_lse
        and max_seqlen_q == 1
        and group_size > 1
        and h_q == h_kv * group_size
        and q.shape[0] == batch_size
        and window_size is None
        and alibi_slopes is None
    )
    original_out = out
    if use_gqa_decode_swap:
        # Match vLLM FA2 decode: turn (B, 1, H_kv * group, D) into
        # (B, group, H_kv, D), so each KV head is loaded once per group block.
        # W2: drop trailing .contiguous() — reshape() after transpose forces
        # a contiguous copy already (incompatible strides), so the explicit
        # .contiguous() is a no-op.
        q = q.reshape(batch_size, h_kv, group_size, q.shape[-1]).transpose(1, 2)
        q = q.reshape(batch_size * group_size, h_kv, q.shape[-1])
        cu_seqlens_q = _cu_seqlens_swap_cached(batch_size, group_size, q.device)
        max_seqlen_q = group_size
        causal = False
        out = None

    fused = _load_varlen_compressed_fused_op()
    # W1: skip .contiguous() on codebooks. They are 16-entry fp32 constants
    # provided by the caller (Compressor.codebook), always contiguous on the
    # cda code path. Eliminates 2 GPU memcpy launches per call (~10 µs total).
    cb_k_arg = cb_k if cb_k.is_contiguous() else cb_k.contiguous()
    cb_v_arg = cb_v if cb_v.is_contiguous() else cb_v.contiguous()
    out, softmax_lse = fused(
        q,
        k_idx_arg,
        k_norm_arg,
        v_idx_arg,
        v_norm_arg,
        cb_k_arg,
        cb_v_arg,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        block_table,
        alibi_slopes,
        int(max_seqlen_q),
        int(max_seqlen_k),
        float(dropout_p),
        float(softmax_scale),
        False,
        bool(causal),
        real_window_size[0],
        real_window_size[1],
        float(softcap),
        bool(return_softmax_lse and dropout_p > 0.0),
        int(num_splits),
        bool(uniform_codebook),
        None,
    )
    if use_gqa_decode_swap:
        # W2: same as Q swap — reshape after transpose copies; trailing
        # .contiguous() is a no-op.
        out = out.reshape(batch_size, group_size, h_kv, out.shape[-1]).transpose(1, 2)
        out = out.reshape(batch_size, h_q, out.shape[-1])
        if original_out is not None:
            original_out.copy_(out)
            out = original_out
    return (out, softmax_lse) if return_softmax_lse else out


__all__ = [
    "dequantize_compressed_kv",
    "flash_attn_varlen_func",
    "flash_attn_varlen_compressed_kv_func",
    "flash_attn_varlen_compressed_kv_fused_func",
]
