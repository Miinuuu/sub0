"""Fused compressed-domain attention patch using sub0's GQA-aware CUDA kernels.

Wraps :func:`cda.cuda_attention_gqa.cuda_hw_attention_gqa` (built from
``csrc/cda_gqa_kernels.cu`` into the ``cda._cda_gqa_kernels`` binary
extension) so it can be dropped into HuggingFace Llama models via
monkey-patching. No CUDA source is compiled at import time — everything
runs through the pre-built ``.so``.

The GQA kernels index ``kv_head = q_head // group_size`` inside CUDA, so
the per-step ``repeat_interleave`` that a per-head kernel would need on
the Python side (~28 ms at N=32768 across 32 layers) disappears. This
reproduces the paper's Figure 5(c) timings.

Typical use (see ``experiments/bench_cda_integrated_single.py``)::

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

from cda.cuda_attention_gqa import (
    cuda_hw_attention_gqa,
    score_2b_gqa,
    score_4b_gqa,
    vfull_2b_gqa,
)


def _compress_kv_cache_cuda(kv_cache, k_comp, v_comp):
    """Compress an HF DynamicCache into the layout the GQA kernel expects.

    Per layer, K/V are quantized per KV-head (no replication) and stored as::

        packed_K: (H_kv, N, D/pack_k)  uint8
        norms_K:  (H_kv, N)            float32
        packed_V: (H_kv, N, D/4)       uint8   (2-bit V)
        norms_V:  (H_kv, N)            float32

    The GQA kernel reads these per query head via ``kv_head = q_head // group``.
    Shared across layers: rotation + signed codebooks.
    """
    compressed: dict = {}
    rotation = cb_k = cb_v = None

    for li in range(len(kv_cache)):
        k = kv_cache.key_cache[li]   # (B, H_kv, N, D)
        v = kv_cache.value_cache[li]
        B, H_kv, N, D = k.shape
        assert B == 1, "fused path currently assumes batch=1"

        cK = k_comp.quantize(k.reshape(-1, D))  # indices: (B*H_kv*N, D/pack_k)
        cV = v_comp.quantize(v.reshape(-1, D))

        pack_k = cK.indices.shape[-1]
        pack_v = cV.indices.shape[-1]

        pK = cK.indices.view(B, H_kv, N, pack_k)[0].contiguous()  # (H_kv, N, pack_k)
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
        }

    compressed["_rotation"] = rotation
    compressed["_codebook_k"] = cb_k
    compressed["_codebook_v"] = cb_v
    return compressed


def _cda_decode_attention(hidden_states, attn_mod, position_embeddings, layer_data, shared, topk):
    """Single decode step: compressed context + decode-token self-attention.

    Merges the fused GQA kernel output with the decode token's own (fp16)
    self-attention via a joint softmax over ``N + 1`` scores. Without this
    merge the causal self-attention score is dropped and PPL degrades.
    ``topk`` gating and 4-bit V are not yet supported on the merged path —
    the kernel currently ships with 2-bit V dense (``vfull_2b_gqa``) only.
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
    pack_k = D // (8 // k_bits)

    # Project Q, K_new, V_new (K_new & V_new needed for the self-attention term).
    q = attn_mod.q_proj(hidden_states).view(B, S, num_heads, D).transpose(1, 2)
    k = attn_mod.k_proj(hidden_states).view(B, S, num_kv_heads, D).transpose(1, 2)
    v = attn_mod.v_proj(hidden_states).view(B, S, num_kv_heads, D).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)   # q, k: (1, H, 1, D)

    Q = q[0, :, 0, :].float()                    # (H_q, D) — original space
    Q_rot = Q @ shared["_rotation"]              # rotated space for kernel

    # (a) scores over the N compressed tokens via GQA kernel
    pK = layer_data["packed_K"].reshape(-1, pack_k).contiguous()
    nK = layer_data["norms_K"].reshape(-1).contiguous()
    if k_bits == 2:
        scores_comp = score_2b_gqa(Q_rot, pK, nK, shared["_codebook_k"], N, group, scale)
    elif k_bits == 4:
        scores_comp = score_4b_gqa(Q_rot, pK, nK, shared["_codebook_k"], N, group, scale)
    else:
        raise ValueError(f"k_bits={k_bits} not supported (expected 2 or 4)")
    # scores_comp: (H_q, N)

    # (b) self-score in fp16 (decode token attends to itself)
    k_new = k[0, :, 0, :].float()                 # (H_kv, D)
    v_new = v[0, :, 0, :].float()                 # (H_kv, D)
    k_new_exp = k_new.repeat_interleave(group, dim=0)  # (H_q, D)
    v_new_exp = v_new.repeat_interleave(group, dim=0)
    scores_self = (Q * k_new_exp).sum(dim=-1, keepdim=True) * scale  # (H_q, 1)

    # (c) joint softmax over N + 1 tokens
    scores_all = torch.cat([scores_comp, scores_self], dim=-1)       # (H_q, N+1)
    attn_all = torch.softmax(scores_all, dim=-1)
    attn_comp = attn_all[:, :N].contiguous()                         # (H_q, N)
    attn_self = attn_all[:, N:]                                      # (H_q, 1)

    # (d) compressed V contribution (kernel) + self V contribution (fp16)
    pV = layer_data["packed_V"].reshape(-1, D // 4).contiguous()
    nV = layer_data["norms_V"].reshape(-1).contiguous()
    out_rot = vfull_2b_gqa(attn_comp, pV, nV, shared["_codebook_v"], N, D, group)
    out_comp = out_rot @ shared["_rotation"]                         # back to original space
    out_self = attn_self * v_new_exp                                 # (H_q, D) — broadcast
    out = out_comp + out_self                                        # (H_q, D) fp32

    out = out.to(hidden_states.dtype).view(1, num_heads, 1, D)
    out = out.transpose(1, 2).reshape(B, S, num_heads * D)
    return attn_mod.o_proj(out), None


def patch_model_compressed_attn(model, k_comp, v_comp=None, attn_gate_topk: int = 0):
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
            def patched_forward(hidden_states, position_embeddings, attention_mask=None,
                                past_key_value=None, cache_position=None, **kwargs):
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
    instead of the HF cache, so this dummy cache exists only so RoPE sees the
    correct position index.
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
    "_compress_kv_cache_cuda",
    "patch_model_compressed_attn",
    "unpatch_model",
    "_PositionOnlyCache",
]
