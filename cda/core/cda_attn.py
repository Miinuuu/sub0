"""CDA single-file integration — CUDA kernels + HF decoder patch.

Loading strategy:
  1. Try the pre-compiled ``_cda_gqa_kernels.so`` that ships with the repo
     (fast startup, matches the reviewed binary drop).
  2. Fall back to JIT-compiling the CUDA source embedded in this module
     via :func:`torch.utils.cpp_extension.load_inline` — useful when the
     binary is missing for the current Python / CUDA stack, or when the
     kernels themselves need to be modified.

Public API (unchanged from the previous split between
``cuda_attention_gqa.py`` / ``cuda_attention_e2e.py`` / ``cda_attn.py``):

  * Kernel-level bindings:
        ``score_2b_gqa``, ``score_4b_gqa``,
        ``vfull_2b_gqa``, ``vsparse_2b_gqa``

  * Layer-level orchestration:
        ``cuda_hw_attention_gqa(...)`` — full per-layer GQA attention,
        ``cda_attn_layer(...)`` — alias kept for backward compatibility,
        ``_get_gqa()`` — lazily returns the loaded extension module.

  * HuggingFace integration:
        ``_compress_kv_cache_cuda(kv_cache, k_comp, v_comp)``,
        ``_cda_decode_attention(...)``,
        ``patch_model_compressed_attn(model, k_comp, v_comp)``,
        ``unpatch_model(model)``,
        ``_PositionOnlyCache`` (dummy cache for decode).

Typical use::

    from core.compression import HadamardQuantCompressor
    from core.cda_attn import (
        _compress_kv_cache_cuda, patch_model_compressed_attn,
        unpatch_model, _PositionOnlyCache,
    )

    compressed = _compress_kv_cache_cuda(fp16_kv, k_comp, v_comp)
    patch_model_compressed_attn(model, k_comp, v_comp)
    for li in range(n_layers):
        model._cda_compressed[li] = compressed[li]
    for k in ("_rotation", "_codebook_k", "_codebook_v"):
        model._cda_compressed[k] = compressed[k]
    pc = _PositionOnlyCache(n_layers, S, device, torch.float16)
    model(nxt, past_key_values=pc, use_cache=True)
    unpatch_model(model)
"""
from __future__ import annotations

import math

import torch
from transformers import DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


# =============================================================================
# Lazy module loader (.so first, JIT fallback via private ``_kernels_jit``)
# =============================================================================
#
# The actual CUDA source is kept in :mod:`core._kernels_jit`, which is NOT
# shipped with submission artefacts (submission ships only the compiled
# ``_cda_gqa_kernels`` binary).  If the binary import succeeds we never touch
# the JIT path — so this module is submission-safe on its own.
# =============================================================================

_gqa_mod = None


def _get_gqa():
    """Return the compiled GQA kernel module (import-cached).

    Resolution order:
      1. Pre-compiled ``core._cda_gqa_kernels`` binary (shipped with the repo).
      2. JIT compile from :mod:`core._kernels_jit` (maintainer-only source).

    Raises:
        ImportError: if both the binary and the private JIT source are absent.
    """
    global _gqa_mod
    if _gqa_mod is not None:
        return _gqa_mod
    try:
        from core import _cda_gqa_kernels as _gqa_mod  # pre-compiled .so
        return _gqa_mod
    except ImportError:
        pass
    try:
        from core._kernels_jit import load_jit
    except ImportError as e:
        raise ImportError(
            "core._cda_gqa_kernels.so not found and maintainer JIT source "
            "(core._kernels_jit) is unavailable — contact the authors for a "
            "platform-specific binary drop."
        ) from e
    _gqa_mod = load_jit()
    return _gqa_mod


# Eager-load at import so accidental graph captures or tight loops don't pay
# the compile cost inside the timed region.
_get_gqa()


# Kernel-level bindings (thin wrappers so the call site does not need to know
# whether .so or JIT provided the symbols).
def score_2b_gqa(Q, pk, nm, cb, N, group_size, scale):
    return _get_gqa().score_2b_gqa(Q, pk, nm, cb, N, group_size, scale)


def score_4b_gqa(Q, pk, nm, cb, N, group_size, scale):
    return _get_gqa().score_4b_gqa(Q, pk, nm, cb, N, group_size, scale)


def vfull_2b_gqa(a, pk, nm, cb, N, D, group_size):
    return _get_gqa().vfull_2b_gqa(a, pk, nm, cb, N, D, group_size)


def vsparse_2b_gqa(atk, idx, pk, nm, cb, K, N, D, group_size):
    return _get_gqa().vsparse_2b_gqa(atk, idx, pk, nm, cb, K, N, D, group_size)


def vfull_4b_gqa(a, pk, nm, cb, N, D, group_size):
    return _get_gqa().vfull_4b_gqa(a, pk, nm, cb, N, D, group_size)


def vsparse_4b_gqa(atk, idx, pk, nm, cb, K, N, D, group_size):
    return _get_gqa().vsparse_4b_gqa(atk, idx, pk, nm, cb, K, N, D, group_size)


# --- Paged variants (opt-in; requires block_tables + block_size) --------------

def score_2b_gqa_paged(Q, pk, nm, bt, cb, N, group_size, block_size, num_blocks, scale):
    return _get_gqa().score_2b_gqa_paged(Q, pk, nm, bt, cb, N, group_size,
                                          block_size, num_blocks, scale)


def score_4b_gqa_paged(Q, pk, nm, bt, cb, N, group_size, block_size, num_blocks, scale):
    return _get_gqa().score_4b_gqa_paged(Q, pk, nm, bt, cb, N, group_size,
                                          block_size, num_blocks, scale)


def vfull_2b_gqa_paged(a, pk, nm, bt, cb, N, D, group_size, block_size, num_blocks):
    return _get_gqa().vfull_2b_gqa_paged(a, pk, nm, bt, cb, N, D, group_size,
                                          block_size, num_blocks)


def vsparse_2b_gqa_paged(atk, idx, pk, nm, bt, cb, K, N, D, group_size,
                          block_size, num_blocks):
    return _get_gqa().vsparse_2b_gqa_paged(atk, idx, pk, nm, bt, cb, K, N, D,
                                            group_size, block_size, num_blocks)


def vfull_4b_gqa_paged(a, pk, nm, bt, cb, N, D, group_size, block_size, num_blocks):
    return _get_gqa().vfull_4b_gqa_paged(a, pk, nm, bt, cb, N, D, group_size,
                                          block_size, num_blocks)


def vsparse_4b_gqa_paged(atk, idx, pk, nm, bt, cb, K, N, D, group_size,
                          block_size, num_blocks):
    return _get_gqa().vsparse_4b_gqa_paged(atk, idx, pk, nm, bt, cb, K, N, D,
                                            group_size, block_size, num_blocks)


# =============================================================================
# Layer-level orchestration
# =============================================================================

def cuda_hw_attention_gqa(
    Q_rot: torch.Tensor,          # (H_q, D) float32
    packed_K: torch.Tensor,       # (H_kv, N, D/pack_k) uint8
    norms_K: torch.Tensor,        # (H_kv, N) float32
    packed_V: torch.Tensor,       # (H_kv, N, D/pack_v) uint8
    norms_V: torch.Tensor,        # (H_kv, N) float32
    codebook_k: torch.Tensor,     # (2^k_bits,)
    codebook_v: torch.Tensor,     # (2^v_bits,)
    rotation: torch.Tensor,
    scale: float,
    N: int,
    group_size: int,
    k_bits: int = 4,
    v_bits: int = 2,
    topk: int = 0,
) -> torch.Tensor:
    """Full GQA-aware compressed-domain attention for one layer.

    Supports K ∈ {2, 4} bits and V ∈ {2, 4} bits independently. Emits one
    score launch + one V launch (dense or TopK sparse). Returns
    ``(H_q, D)`` float32 after applying the inverse rotation.
    """
    H_q, D = Q_rot.shape
    assert D <= 128, f"D={D} exceeds kernel shared-memory limit (128)"
    assert H_q % group_size == 0, f"H_q={H_q} not divisible by group_size={group_size}"

    k_pack = D // (8 // k_bits)
    v_pack = D // (8 // v_bits)
    pk = packed_K.reshape(-1, k_pack).contiguous()
    nk = norms_K.reshape(-1).contiguous()
    pv = packed_V.reshape(-1, v_pack).contiguous()
    nv = norms_V.reshape(-1).contiguous()

    if k_bits == 2:
        scores = score_2b_gqa(Q_rot, pk, nk, codebook_k, N, group_size, scale)
    elif k_bits == 4:
        scores = score_4b_gqa(Q_rot, pk, nk, codebook_k, N, group_size, scale)
    else:
        raise ValueError(f"unsupported k_bits={k_bits} (expected 2 or 4)")

    attn = torch.softmax(scores, dim=-1)

    if 0 < topk < N:
        vals, idx = attn.topk(topk, dim=-1)
        vals = vals / (vals.sum(dim=-1, keepdim=True) + 1e-12)
        if v_bits == 2:
            out_rot = vsparse_2b_gqa(vals, idx.int(), pv, nv, codebook_v,
                                      topk, N, D, group_size)
        elif v_bits == 4:
            out_rot = vsparse_4b_gqa(vals, idx.int(), pv, nv, codebook_v,
                                      topk, N, D, group_size)
        else:
            raise ValueError(f"unsupported v_bits={v_bits} (expected 2 or 4)")
    else:
        if v_bits == 2:
            out_rot = vfull_2b_gqa(attn, pv, nv, codebook_v, N, D, group_size)
        elif v_bits == 4:
            out_rot = vfull_4b_gqa(attn, pv, nv, codebook_v, N, D, group_size)
        else:
            raise ValueError(f"unsupported v_bits={v_bits} (expected 2 or 4)")

    return out_rot @ rotation


# Backward-compat alias — some callers import ``cda_attn_layer`` directly.
cda_attn_layer = cuda_hw_attention_gqa


def cuda_hw_attention_gqa_paged(
    Q_rot: torch.Tensor,          # (H_q, D) float32
    packed_K: torch.Tensor,       # (H_kv, num_blocks, block_size, D/pack_k) uint8
    norms_K: torch.Tensor,        # (H_kv, num_blocks, block_size) float32
    packed_V: torch.Tensor,       # (H_kv, num_blocks, block_size, D/pack_v) uint8
    norms_V: torch.Tensor,        # (H_kv, num_blocks, block_size) float32
    block_tables: torch.Tensor,   # (max_logical_blocks,) int32 — shared across KV heads
    codebook_k: torch.Tensor,
    codebook_v: torch.Tensor,
    rotation: torch.Tensor,
    scale: float,
    N: int,
    group_size: int,
    block_size: int = 16,
    k_bits: int = 4,
    v_bits: int = 2,
    topk: int = 0,
) -> torch.Tensor:
    """Paged-KV variant of :func:`cuda_hw_attention_gqa`.

    Supports K ∈ {2, 4} bits and V ∈ {2, 4} bits (same matrix as the
    contiguous path). Block tables are shared across KV heads — every KV
    head reads the same logical→physical mapping but from its own row in
    the packed pool (shape ``(H_kv, num_blocks, …)``).

    Returns ``(H_q, D)`` float32 after the inverse rotation.
    """
    H_q, D = Q_rot.shape
    assert D <= 128
    assert H_q % group_size == 0
    num_blocks = packed_K.shape[1]

    if k_bits == 2:
        scores = score_2b_gqa_paged(Q_rot, packed_K, norms_K, block_tables,
                                     codebook_k, N, group_size,
                                     block_size, num_blocks, scale)
    elif k_bits == 4:
        scores = score_4b_gqa_paged(Q_rot, packed_K, norms_K, block_tables,
                                     codebook_k, N, group_size,
                                     block_size, num_blocks, scale)
    else:
        raise ValueError(f"paged path only supports k_bits ∈ {{2,4}}, got {k_bits}")

    attn = torch.softmax(scores, dim=-1)

    if 0 < topk < N:
        vals, idx = attn.topk(topk, dim=-1)
        vals = vals / (vals.sum(dim=-1, keepdim=True) + 1e-12)
        if v_bits == 2:
            out_rot = vsparse_2b_gqa_paged(vals, idx.int(), packed_V, norms_V,
                                            block_tables, codebook_v,
                                            topk, N, D, group_size,
                                            block_size, num_blocks)
        elif v_bits == 4:
            out_rot = vsparse_4b_gqa_paged(vals, idx.int(), packed_V, norms_V,
                                            block_tables, codebook_v,
                                            topk, N, D, group_size,
                                            block_size, num_blocks)
        else:
            raise ValueError(f"paged path only supports v_bits ∈ {{2,4}}, got {v_bits}")
    else:
        if v_bits == 2:
            out_rot = vfull_2b_gqa_paged(attn, packed_V, norms_V, block_tables,
                                          codebook_v, N, D, group_size,
                                          block_size, num_blocks)
        elif v_bits == 4:
            out_rot = vfull_4b_gqa_paged(attn, packed_V, norms_V, block_tables,
                                          codebook_v, N, D, group_size,
                                          block_size, num_blocks)
        else:
            raise ValueError(f"paged path only supports v_bits ∈ {{2,4}}, got {v_bits}")

    return out_rot @ rotation


def pack_kv_to_blocks(compressed_layer: dict, block_size: int = 16) -> dict:
    """Convert a contiguous ``_compress_kv_cache_cuda`` layer dict to paged layout.

    Returns a new dict with ``packed_K / norms_K / packed_V / norms_V`` reshaped
    to ``(H_kv, num_blocks, block_size, …)`` and an identity ``block_tables``
    mapping (logical == physical). Pads the last block with zeros if
    ``N % block_size != 0``.
    """
    pK = compressed_layer["packed_K"]
    nK = compressed_layer["norms_K"]
    pV = compressed_layer["packed_V"]
    nV = compressed_layer["norms_V"]
    H_kv, N, pack_k = pK.shape
    pack_v = pV.shape[-1]
    num_blocks = (N + block_size - 1) // block_size
    pad = num_blocks * block_size - N
    device = pK.device

    def _pad_then_reshape(t, pack):
        if pad > 0:
            pad_t = torch.zeros(H_kv, pad, pack, dtype=t.dtype, device=device)
            t = torch.cat([t, pad_t], dim=1)
        return t.reshape(H_kv, num_blocks, block_size, pack).contiguous()

    def _pad_then_reshape_1d(t):
        if pad > 0:
            pad_t = torch.zeros(H_kv, pad, dtype=t.dtype, device=device)
            t = torch.cat([t, pad_t], dim=1)
        return t.reshape(H_kv, num_blocks, block_size).contiguous()

    block_tables = torch.arange(num_blocks, dtype=torch.int32, device=device)
    return {
        "packed_K": _pad_then_reshape(pK, pack_k),
        "norms_K":  _pad_then_reshape_1d(nK),
        "packed_V": _pad_then_reshape(pV, pack_v),
        "norms_V":  _pad_then_reshape_1d(nV),
        "block_tables": block_tables,
        "block_size": block_size,
        "num_blocks": num_blocks,
        "H_kv": H_kv, "N": N, "D": compressed_layer["D"],
        "bit_width_k": compressed_layer["bit_width_k"],
        "bit_width_v": compressed_layer.get("bit_width_v", 2),
    }


def _get_e2e():
    """Legacy no-op kept for modules that still call it to trigger kernel load."""
    return _get_gqa()


# =============================================================================
# HuggingFace integration
# =============================================================================

def _compress_kv_cache_cuda(kv_cache, k_comp, v_comp):
    """Compress an HF DynamicCache into the layout the GQA kernel expects.

    Per layer, K/V are quantized per KV-head (no replication) and stored as::

        packed_K: (H_kv, N, D/pack_k)  uint8
        norms_K:  (H_kv, N)            float32
        packed_V: (H_kv, N, D/4)       uint8   (2-bit V)
        norms_V:  (H_kv, N)            float32

    Shared across layers: rotation + signed codebooks.
    """
    compressed: dict = {}
    rotation = cb_k = cb_v = None

    for li in range(len(kv_cache)):
        k = kv_cache.key_cache[li]   # (B, H_kv, N, D)
        v = kv_cache.value_cache[li]
        B, H_kv, N, D = k.shape
        assert B == 1, "fused path currently assumes batch=1"

        cK = k_comp.quantize(k.reshape(-1, D))
        cV = v_comp.quantize(v.reshape(-1, D))

        pack_k = cK.indices.shape[-1]
        pack_v = cV.indices.shape[-1]

        pK = cK.indices.view(B, H_kv, N, pack_k)[0].contiguous()
        nK = cK.norms.float().view(B, H_kv, N)[0].contiguous()
        pV = cV.indices.view(B, H_kv, N, pack_v)[0].contiguous()
        nV = cV.norms.float().view(B, H_kv, N)[0].contiguous()

        if rotation is None:
            k_comp._ensure_tensors(k.device)
            v_comp._ensure_tensors(k.device)
            rotation = k_comp._rotation.float().contiguous()
            cb_k = (k_comp._centroids * 2.0 - 1.0).float().contiguous()
            cb_v = (v_comp._centroids * 2.0 - 1.0).float().contiguous()

        compressed[li] = {
            "packed_K": pK, "norms_K": nK,
            "packed_V": pV, "norms_V": nV,
            "H_kv": H_kv, "N": N, "D": D,
            "bit_width_k": k_comp.bit_width,
            "bit_width_v": v_comp.bit_width,
        }

    compressed["_rotation"] = rotation
    compressed["_codebook_k"] = cb_k
    compressed["_codebook_v"] = cb_v
    return compressed


def _cda_decode_attention(hidden_states, attn_mod, position_embeddings,
                          layer_data, shared, topk):
    """Single decode step: compressed context + decode-token self-attention.

    Joint softmax over ``N + 1`` scores merges the kernel output with the
    current token's own fp16 self-attention — dropping this merge degrades
    PPL. Only 2-bit V dense is currently wired into the merged path.
    """
    B, S, _ = hidden_states.shape
    assert S == 1, "fused CDA path handles decode only (S=1)"
    D = attn_mod.head_dim
    num_heads = attn_mod.config.num_attention_heads
    num_kv_heads = attn_mod.config.num_key_value_heads
    group = num_heads // num_kv_heads
    scale = 1.0 / math.sqrt(D)
    N = layer_data["N"]
    k_bits = layer_data["bit_width_k"]
    v_bits = layer_data.get("bit_width_v", 2)    # backward-compat (old dicts)
    pack_k = D // (8 // k_bits)
    pack_v = D // (8 // v_bits)

    q = attn_mod.q_proj(hidden_states).view(B, S, num_heads, D).transpose(1, 2)
    k = attn_mod.k_proj(hidden_states).view(B, S, num_kv_heads, D).transpose(1, 2)
    v = attn_mod.v_proj(hidden_states).view(B, S, num_kv_heads, D).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    Q = q[0, :, 0, :].float()
    Q_rot = Q @ shared["_rotation"]

    pK = layer_data["packed_K"].reshape(-1, pack_k).contiguous()
    nK = layer_data["norms_K"].reshape(-1).contiguous()
    if k_bits == 2:
        scores_comp = score_2b_gqa(Q_rot, pK, nK, shared["_codebook_k"], N, group, scale)
    elif k_bits == 4:
        scores_comp = score_4b_gqa(Q_rot, pK, nK, shared["_codebook_k"], N, group, scale)
    else:
        raise ValueError(f"k_bits={k_bits} not supported (expected 2 or 4)")

    k_new = k[0, :, 0, :].float()
    v_new = v[0, :, 0, :].float()
    k_new_exp = k_new.repeat_interleave(group, dim=0)
    v_new_exp = v_new.repeat_interleave(group, dim=0)
    scores_self = (Q * k_new_exp).sum(dim=-1, keepdim=True) * scale

    scores_all = torch.cat([scores_comp, scores_self], dim=-1)
    attn_all = torch.softmax(scores_all, dim=-1)
    attn_comp = attn_all[:, :N].contiguous()
    attn_self = attn_all[:, N:]

    pV = layer_data["packed_V"].reshape(-1, pack_v).contiguous()
    nV = layer_data["norms_V"].reshape(-1).contiguous()
    if v_bits == 2:
        out_rot = vfull_2b_gqa(attn_comp, pV, nV, shared["_codebook_v"], N, D, group)
    elif v_bits == 4:
        out_rot = vfull_4b_gqa(attn_comp, pV, nV, shared["_codebook_v"], N, D, group)
    else:
        raise ValueError(f"v_bits={v_bits} not supported (expected 2 or 4)")
    out_comp = out_rot @ shared["_rotation"]
    out_self = attn_self * v_new_exp
    out = out_comp + out_self

    out = out.to(hidden_states.dtype).view(1, num_heads, 1, D)
    out = out.transpose(1, 2).reshape(B, S, num_heads * D)
    return attn_mod.o_proj(out), None


def patch_model_compressed_attn(model, k_comp, v_comp=None,
                                 attn_gate_topk: int = 0):
    """Monkey-patch ``LlamaAttention.forward`` to use the GQA CDA kernel on decode.

    Prefill (``S > 1``) falls through to the original forward. After calling
    this, populate ``model._cda_compressed[li]`` with per-layer dicts from
    :func:`_compress_kv_cache_cuda` and set the shared
    ``_rotation`` / ``_codebook_k`` / ``_codebook_v`` keys on the same dict
    before running decode.
    """
    vc = v_comp or k_comp
    model._cda_compressed = {}
    model._cda_k_comp = k_comp
    model._cda_v_comp = vc
    model._cda_topk = attn_gate_topk
    model._cda_originals = {}

    for li, layer in enumerate(model.model.layers):
        attn_mod = layer.self_attn
        model._cda_originals[li] = attn_mod.forward

        def _make_patched(orig, layer_idx, attn_ref):
            def patched_forward(hidden_states, position_embeddings,
                                attention_mask=None, past_key_value=None,
                                cache_position=None, **kwargs):
                layer_data = model._cda_compressed.get(layer_idx)
                if layer_data is not None and hidden_states.shape[1] == 1:
                    return _cda_decode_attention(
                        hidden_states, attn_ref, position_embeddings,
                        layer_data, model._cda_compressed,
                        topk=model._cda_topk,
                    )
                return orig(hidden_states, position_embeddings, attention_mask,
                            past_key_value, cache_position, **kwargs)
            return patched_forward

        attn_mod.forward = _make_patched(attn_mod.forward, li, attn_mod)


def unpatch_model(model):
    """Remove patches installed by :func:`patch_model_compressed_attn`."""
    if hasattr(model, "_cda_originals"):
        for li, orig in model._cda_originals.items():
            model.model.layers[li].self_attn.forward = orig
    for attr in ("_cda_compressed", "_cda_originals",
                 "_cda_k_comp", "_cda_v_comp", "_cda_topk"):
        if hasattr(model, attr):
            delattr(model, attr)


class _PositionOnlyCache(DynamicCache):
    """Thin DynamicCache that only tracks seq length (no real KV data).

    The patched attention pulls compressed KV from ``model._cda_compressed``
    instead of the HF cache, so this dummy cache exists only so RoPE sees
    the correct position index.
    """

    def __init__(self, num_layers, seq_len, device, dtype):
        super().__init__()
        for li in range(num_layers):
            dummy = torch.zeros(1, 1, seq_len, 1, device=device, dtype=dtype)
            self.update(dummy, dummy, li)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if len(self.key_cache) <= layer_idx:
            for _ in range(len(self.key_cache), layer_idx + 1):
                self.key_cache.append([])
                self.value_cache.append([])
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        elif len(self.key_cache[layer_idx]) == 0:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


__all__ = [
    # contiguous kernel bindings
    "score_2b_gqa", "score_4b_gqa",
    "vfull_2b_gqa", "vsparse_2b_gqa",
    "vfull_4b_gqa", "vsparse_4b_gqa",
    # paged kernel bindings
    "score_2b_gqa_paged", "score_4b_gqa_paged",
    "vfull_2b_gqa_paged", "vsparse_2b_gqa_paged",
    "vfull_4b_gqa_paged", "vsparse_4b_gqa_paged",
    # layer orchestration
    "cuda_hw_attention_gqa", "cuda_hw_attention_gqa_paged",
    "cda_attn_layer", "_get_gqa", "_get_e2e",
    "pack_kv_to_blocks",
    # HF integration
    "_compress_kv_cache_cuda", "_cda_decode_attention",
    "patch_model_compressed_attn", "unpatch_model",
    "_PositionOnlyCache",
]
