"""End-to-end generation with CUDA compressed-domain attention.

Monkey-patches HuggingFace LlamaAttention to use CUDA compressed kernel
during decode. Prefill runs normally, then KV is compressed and all
subsequent attention is computed directly on compressed indices.

Handles GQA (grouped-query attention) by repeating KV head indices.

Usage:
    from core.compressed_generate import compressed_generate
    result = compressed_generate(model, tokenizer, prompt, compressor)
"""
from __future__ import annotations

import gc
import time
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from core.compression import (
    HadamardQuantCompressor,
    CompressedTensor,
    _gpu_unpack_2bit,
    _gpu_unpack_4bit,
)
from core.compressed_attention import hw_attention_score, hw_attention_output, _topk_gate_attn

# Try to import GQA-aware CUDA kernel (falls back to Python if unavailable)
_USE_CUDA_E2E = False
try:
    from core.cda_attn import cda_attn_layer, _get_e2e
    _USE_CUDA_E2E = True
except ImportError:
    pass


def _compress_kv_cache(kv_cache, k_comp, v_comp, skip_sinks=4):
    """Compress a DynamicCache into per-layer CompressedTensors."""
    compressed = {}
    # transformers 4.57+ renamed: key_cache/value_cache → layers[i].keys/values.
    has_layers = hasattr(kv_cache, "layers") and len(kv_cache.layers) > 0
    for li in range(len(kv_cache)):
        if has_layers:
            k = kv_cache.layers[li].keys
            v = kv_cache.layers[li].values
        else:
            k = kv_cache.key_cache[li]
            v = kv_cache.value_cache[li]
        B, H, S, D = k.shape

        if skip_sinks > 0 and S > skip_sinks:
            k_sink = k[:, :, :skip_sinks, :]
            v_sink = v[:, :, :skip_sinks, :]
            k_rest = k[:, :, skip_sinks:, :].reshape(-1, D)
            v_rest = v[:, :, skip_sinks:, :].reshape(-1, D)
            cK = k_comp.quantize(k_rest)
            cV = v_comp.quantize(v_rest)
            compressed[li] = {
                "cK": cK, "cV": cV,
                "k_sink": k_sink, "v_sink": v_sink,
                "B": B, "H": H, "S_comp": S - skip_sinks, "S_sink": skip_sinks, "D": D,
            }
        else:
            k_flat = k.reshape(-1, D)
            v_flat = v.reshape(-1, D)
            cK = k_comp.quantize(k_flat)
            cV = v_comp.quantize(v_flat)
            compressed[li] = {
                "cK": cK, "cV": cV,
                "k_sink": None, "v_sink": None,
                "B": B, "H": H, "S_comp": S, "S_sink": 0, "D": D,
            }
    return compressed


def _compress_kv_cache_cuda(kv_cache, k_comp, v_comp, skip_sinks=4):
    """Compress KV cache into CUDA GQA kernel format.

    Returns dict with per-layer packed tensors shaped (H_kv, S_comp, pack_dim).
    """
    compressed = {}
    rotation = None
    for li in range(len(kv_cache)):
        if hasattr(kv_cache, "layers"):
            k = kv_cache.layers[li].keys
            v = kv_cache.layers[li].values
        else:
            k = kv_cache.key_cache[li]  # (B, H_kv, S, D)
            v = kv_cache.value_cache[li]
        B, H, S, D = k.shape

        S_sink = min(skip_sinks, S) if skip_sinks > 0 else 0
        S_comp = S - S_sink

        k_sink = k[:, :, :S_sink, :] if S_sink > 0 else None
        v_sink = v[:, :, :S_sink, :] if S_sink > 0 else None

        # Quantize per KV head separately to get (H_kv, S_comp, pack_dim) layout
        pk_list, nk_list, pv_list, nv_list = [], [], [], []
        for b in range(B):
            for h in range(H):
                k_h = k[b, h, S_sink:, :]  # (S_comp, D)
                v_h = v[b, h, S_sink:, :]
                ck = k_comp.quantize(k_h)
                cv = v_comp.quantize(v_h)
                pk_list.append(ck.indices)  # (S_comp, D/pack)
                nk_list.append(ck.norms)    # (S_comp,)
                pv_list.append(cv.indices)
                nv_list.append(cv.norms)

        # Stack: (B*H_kv, S_comp, pack_dim)
        packed_K = torch.stack(pk_list, dim=0)  # (B*H_kv, S_comp, D/pack_k)
        norms_K = torch.stack(nk_list, dim=0)   # (B*H_kv, S_comp)
        packed_V = torch.stack(pv_list, dim=0)
        norms_V = torch.stack(nv_list, dim=0)

        if rotation is None:
            rotation = torch.from_numpy(k_comp._rotation_np).float().to(k.device)

        compressed[li] = {
            "packed_K": packed_K, "norms_K": norms_K,
            "packed_V": packed_V, "norms_V": norms_V,
            "k_sink": k_sink, "v_sink": v_sink,
            "B": B, "H": H, "S_comp": S_comp, "S_sink": S_sink, "D": D,
        }

    compressed["_rotation"] = rotation
    compressed["_cb_k"] = torch.from_numpy(k_comp._centroids_np).float().to(k.device)
    compressed["_cb_v"] = torch.from_numpy(v_comp._centroids_np).float().to(k.device)
    return compressed


def _compressed_attention_cuda(query_states, layer_data, k_comp, v_comp, scale,
                                attn_gate_topk=0, k_new=None, v_new=None,
                                rotation=None, cb_k=None, cb_v=None):
    """CUDA GQA kernel attention for one layer. Single kernel launch per layer.

    query_states: (B, H_q, 1, D) float16
    Returns: (B, H_q, 1, D) float16
    """
    B = layer_data["B"]
    H_kv = layer_data["H"]
    S_comp = layer_data["S_comp"]
    S_sink = layer_data["S_sink"]
    D = layer_data["D"]
    k_sink = layer_data["k_sink"]
    v_sink = layer_data["v_sink"]

    H_q = query_states.shape[1]
    group_size = H_q // H_kv

    all_outputs = []
    for b in range(B):
        # Q for all heads: (H_q, D)
        Q = query_states[b, :, 0, :].float()  # (H_q, D)
        Q_rot = Q @ rotation

        # Packed KV for this batch: (H_kv, S_comp, pack_dim)
        pK = layer_data["packed_K"][b * H_kv:(b + 1) * H_kv]
        nK = layer_data["norms_K"][b * H_kv:(b + 1) * H_kv].float()
        pV = layer_data["packed_V"][b * H_kv:(b + 1) * H_kv]
        nV = layer_data["norms_V"][b * H_kv:(b + 1) * H_kv].float()

        topk = attn_gate_topk if attn_gate_topk > 0 and S_comp >= 16384 else 0

        # Single GQA kernel call: all H_q heads at once
        out_comp = cda_attn_layer(
            Q_rot, pK, nK, pV, nV,
            cb_k, cb_v, rotation, scale, S_comp,
            group_size=group_size, topk=topk,
        )  # (H_q, D)

        # Add sink + generated tokens contribution (FP16)
        if (k_sink is not None and S_sink > 0) or k_new is not None:
            # Concatenate sink + gen keys/values
            kv_parts_k, kv_parts_v = [], []
            if k_sink is not None and S_sink > 0:
                kv_parts_k.append(k_sink[b])   # (H_kv, S_sink, D)
                kv_parts_v.append(v_sink[b])
            if k_new is not None:
                kv_parts_k.append(k_new[b])    # (H_kv, gen_len, D)
                kv_parts_v.append(v_new[b])

            k_fp16 = torch.cat(kv_parts_k, dim=1).float()  # (H_kv, S_extra, D)
            v_fp16 = torch.cat(kv_parts_v, dim=1).float()
            S_extra = k_fp16.shape[1]

            # Expand for GQA: (H_q, S_extra, D)
            k_exp = k_fp16.repeat_interleave(group_size, dim=0)
            v_exp = v_fp16.repeat_interleave(group_size, dim=0)

            # FP16 attention scores for extra tokens
            scores_extra = torch.bmm(Q.unsqueeze(1), k_exp.transpose(1, 2)).squeeze(1) * scale
            # (H_q, S_extra)

            # Simple weighted combination: reweight compressed output
            # This is approximate — proper implementation needs joint softmax
            attn_extra = F.softmax(scores_extra, dim=-1)
            out_extra = torch.bmm(attn_extra.unsqueeze(1), v_exp).squeeze(1)  # (H_q, D)

            # Weighted merge: compressed tokens dominate (S_comp >> S_extra)
            w_comp = S_comp / (S_comp + S_extra)
            out = w_comp * out_comp + (1 - w_comp) * out_extra
        else:
            out = out_comp

        all_outputs.append(out.unsqueeze(0))  # (1, H_q, D)

    return torch.stack(all_outputs, dim=0).unsqueeze(2).half()  # (B, H_q, 1, D)


def _compressed_attention_for_layer(
    query_states,   # (B, num_q_heads, 1, D)
    layer_data,     # dict from _compress_kv_cache
    k_comp,
    v_comp,
    scale,
    attn_gate_topk=0,
    temperature=None,  # float, per-layer temperature (None = no correction)
    score_clamp=None,  # float, clamp pre-softmax scores to [-C, C] (None = no clamp)
    k_new=None,     # (B, H_kv, 1, D) current token's key (fp16, with RoPE)
    v_new=None,     # (B, H_kv, 1, D) current token's value (fp16)
):
    """Compute attention for one decode step using compressed KV."""
    B = layer_data["B"]
    H_kv = layer_data["H"]
    S_comp = layer_data["S_comp"]
    S_sink = layer_data["S_sink"]
    D = layer_data["D"]
    cK = layer_data["cK"]
    cV = layer_data["cV"]
    k_sink = layer_data["k_sink"]
    v_sink = layer_data["v_sink"]

    num_q_heads = query_states.shape[1]
    group_size = num_q_heads // H_kv

    all_outputs = []

    for b in range(B):
        head_outputs = []
        for kv_h in range(H_kv):
            # Slice compressed KV for this batch × kv_head
            idx_start = (b * H_kv + kv_h) * S_comp
            idx_end = idx_start + S_comp

            cK_h = CompressedTensor(
                indices=cK.indices[idx_start:idx_end],
                payload=None,
                norms=cK.norms[idx_start:idx_end],
                shape=torch.Size([S_comp, D]),
                dtype=cK.dtype, device=cK.device,
                bit_width=cK.bit_width, gpu_resident=True,
            )
            cV_h = CompressedTensor(
                indices=cV.indices[idx_start:idx_end],
                payload=None,
                norms=cV.norms[idx_start:idx_end],
                shape=torch.Size([S_comp, D]),
                dtype=cV.dtype, device=cV.device,
                bit_width=cV.bit_width, gpu_resident=True,
            )

            # Each Q head in this KV group
            for g in range(group_size):
                q_h = kv_h * group_size + g
                q = query_states[b, q_h, :, :]  # (1, D)

                # Compressed-domain attention
                scores_comp = hw_attention_score(q, cK_h, k_comp, scale)  # (1, S_comp)

                if k_sink is not None and S_sink > 0:
                    # Sink tokens: fp16 attention
                    k_s = k_sink[b, kv_h, :, :].float()  # (S_sink, D)
                    scores_sink = (q.float() @ k_s.T) * scale  # (1, S_sink)
                    scores_all = torch.cat([scores_sink, scores_comp.float()], dim=-1)
                else:
                    scores_all = scores_comp.float()

                # Current token's self-attention score (fp16)
                if k_new is not None:
                    k_cur = k_new[b, kv_h, :, :].float()  # (1, D)
                    score_cur = (q.float() @ k_cur.T) * scale  # (1, 1)
                    scores_all = torch.cat([scores_all, score_cur], dim=-1)

                if temperature is not None and temperature > 0:
                    scores_all = scores_all / temperature
                if score_clamp is not None:
                    scores_all = scores_all.clamp(-score_clamp, score_clamp)
                attn_w = F.softmax(scores_all, dim=-1)

                if attn_gate_topk > 0:
                    attn_w = _topk_gate_attn(attn_w, attn_gate_topk)

                # Split attention weights back
                n_prefix = S_sink + S_comp if (k_sink is not None and S_sink > 0) else S_comp
                attn_prefix = attn_w[:, :n_prefix]
                attn_cur = attn_w[:, n_prefix:] if k_new is not None else None

                if k_sink is not None and S_sink > 0:
                    attn_sink = attn_prefix[:, :S_sink]
                    attn_comp = attn_prefix[:, S_sink:]
                    v_s = v_sink[b, kv_h, :, :].float()  # (S_sink, D)
                    out_sink = attn_sink @ v_s  # (1, D)
                    out_comp = hw_attention_output(attn_comp, cV_h, v_comp)  # (1, D)
                    out_h = out_sink + out_comp.float()
                else:
                    out_h = hw_attention_output(attn_prefix, cV_h, v_comp).float()

                # Add current token's V contribution
                if attn_cur is not None and v_new is not None:
                    v_cur = v_new[b, kv_h, :, :].float()  # (1, D)
                    out_h = out_h + attn_cur @ v_cur

                head_outputs.append(out_h)  # (1, D)

        # Stack all Q heads: (num_q_heads, D)
        all_outputs.append(torch.stack([h.squeeze(0) for h in head_outputs], dim=0))

    # (B, num_q_heads, 1, D)
    return torch.stack(all_outputs, dim=0).unsqueeze(2)


def manual_decode_step(model, token_ids, kv_compressed, k_comp, v_comp,
                        position, attn_gate_topk=0, temperatures=None):
    """Run one decode step manually through all transformer layers.

    Bypasses HuggingFace's model() to avoid cache/mask conflicts.
    Uses compressed-domain attention for KV, standard MLP.
    Accumulates generated tokens' K/V in kv_compressed['gen_k'][li] / ['gen_v'][li].

    Args:
        model: HuggingFace LlamaForCausalLM
        token_ids: (B, 1) next token ids
        kv_compressed: dict from _compress_kv_cache (modified in-place)
        k_comp: key compressor
        v_comp: value compressor
        position: int, position of the decode token
        attn_gate_topk: top-K gating (0 = disabled)

    Returns:
        logits: (B, 1, vocab_size)
    """
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    vc = v_comp or k_comp
    scale = k_comp.dim ** -0.5

    # Initialize generated KV accumulator on first call
    if 'gen_k' not in kv_compressed:
        n_layers = len([k for k in kv_compressed.keys() if isinstance(k, int)])
        kv_compressed['gen_k'] = {li: [] for li in range(n_layers)}
        kv_compressed['gen_v'] = {li: [] for li in range(n_layers)}

    # Embedding
    hidden = model.model.embed_tokens(token_ids)  # (B, 1, hidden_size)

    # Position embeddings (RoPE)
    position_ids = torch.tensor([[position]], device=hidden.device)
    cos, sin = model.model.rotary_emb(hidden, position_ids)

    # Layer-by-layer forward
    for li, layer in enumerate(model.model.layers):
        residual = hidden

        # Pre-attention layernorm
        hidden_norm = layer.input_layernorm(hidden)

        # Compute Q, K_new, V_new
        attn = layer.self_attn
        input_shape = hidden_norm.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        query_states = attn.q_proj(hidden_norm).view(hidden_shape).transpose(1, 2)
        key_states = attn.k_proj(hidden_norm).view(
            *input_shape, -1, attn.head_dim).transpose(1, 2)
        value_states = attn.v_proj(hidden_norm).view(
            *input_shape, -1, attn.head_dim).transpose(1, 2)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Accumulate generated K/V (fp16, includes current + all previous gen tokens)
        kv_compressed['gen_k'][li].append(key_states)
        kv_compressed['gen_v'][li].append(value_states)
        # Concat all generated K/V for this layer
        k_all_gen = torch.cat(kv_compressed['gen_k'][li], dim=2)  # (B, H, gen_len, D)
        v_all_gen = torch.cat(kv_compressed['gen_v'][li], dim=2)

        # Compressed-domain attention + all generated tokens' fp16 K/V
        compressed = kv_compressed.get(li)
        if compressed is not None:
            if _USE_CUDA_E2E and "packed_K" in compressed:
                # Fast path: GQA-aware CUDA kernel
                rotation = kv_compressed.get("_rotation")
                cb_k = kv_compressed.get("_cb_k")
                cb_v = kv_compressed.get("_cb_v")
                attn_output = _compressed_attention_cuda(
                    query_states, compressed, k_comp, vc, scale, attn_gate_topk,
                    k_new=k_all_gen, v_new=v_all_gen,
                    rotation=rotation, cb_k=cb_k, cb_v=cb_v,
                )
            else:
                # Fallback: Python-level attention
                temp_l = temperatures[li] if temperatures is not None else None
                attn_output = _compressed_attention_for_layer(
                    query_states, compressed, k_comp, vc, scale, attn_gate_topk,
                    temperature=temp_l,
                    k_new=k_all_gen, v_new=v_all_gen,
                )
        else:
            attn_output = torch.zeros_like(query_states)

        attn_output = attn_output.to(hidden.dtype)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn.o_proj(attn_output)

        # Residual + attention
        hidden = residual + attn_output

        # MLP
        residual = hidden
        hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))

    # Final norm + LM head
    hidden = model.model.norm(hidden)
    logits = model.lm_head(hidden)

    return logits


def patch_model_compressed_attn(model, k_comp, v_comp=None, skip_sinks=4, attn_gate_topk=0):
    """Monkey-patch model to use compressed attention during decode.

    After calling this + setting model._cda_compressed, all decode steps
    use CUDA compressed-domain attention instead of fp16 matmul.

    Args:
        model: HuggingFace model
        k_comp: key compressor
        v_comp: value compressor (None = same as k_comp)
        skip_sinks: number of sink tokens kept in fp16
        attn_gate_topk: top-K gating (0 = disabled)
    """
    vc = v_comp or k_comp
    model._cda_compressed = {}
    model._cda_k_comp = k_comp
    model._cda_v_comp = vc
    model._cda_topk = attn_gate_topk
    model._cda_originals = {}

    scale = k_comp.dim ** -0.5

    for li, layer in enumerate(model.model.layers):
        attn_mod = layer.self_attn
        original_fwd = attn_mod.forward

        model._cda_originals[li] = original_fwd

        def _make_patched(orig, layer_idx):
            def patched_forward(hidden_states, position_embeddings, attention_mask=None,
                                past_key_value=None, cache_position=None, **kwargs):
                compressed = model._cda_compressed.get(layer_idx)

                # Decode with compressed attention
                if compressed is not None and hidden_states.shape[1] == 1:
                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, attn_mod.head_dim)

                    query_states = attn_mod.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    key_states = attn_mod.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    cos, sin = position_embeddings
                    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
                    query_states, _ = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                    if _USE_CUDA_E2E and "packed_K" in compressed:
                        rotation = model._cda_compressed.get("_rotation")
                        cb_k = model._cda_compressed.get("_cb_k")
                        cb_v = model._cda_compressed.get("_cb_v")
                        attn_output = _compressed_attention_cuda(
                            query_states, compressed,
                            model._cda_k_comp, model._cda_v_comp,
                            scale, model._cda_topk,
                            rotation=rotation, cb_k=cb_k, cb_v=cb_v,
                        )
                    else:
                        attn_output = _compressed_attention_for_layer(
                            query_states, compressed,
                            model._cda_k_comp, model._cda_v_comp,
                            scale, model._cda_topk,
                        )

                    attn_output = attn_output.to(hidden_states.dtype)
                    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                    attn_output = attn_mod.o_proj(attn_output)
                    return attn_output, None

                # Normal forward (prefill)
                return orig(hidden_states, position_embeddings, attention_mask,
                            past_key_value, cache_position, **kwargs)

            return patched_forward

        attn_mod.forward = _make_patched(original_fwd, li)


def patch_model_decompress_attn(model, k_comp, v_comp=None):
    """Monkey-patch model to decompress KV inside attention (for TQ/CDA SW-mode).

    Same as standard forward but decompresses compressed KV per layer
    inside model.forward(), eliminating Python-level external loop.
    """
    vc = v_comp or k_comp
    model._dc_compressed = {}
    model._dc_k_comp = k_comp
    model._dc_v_comp = vc
    model._dc_originals = {}

    for li, layer in enumerate(model.model.layers):
        attn_mod = layer.self_attn
        original_fwd = attn_mod.forward
        model._dc_originals[li] = original_fwd
        D = attn_mod.head_dim

        def _make_patched(orig, layer_idx):
            def patched_forward(hidden_states, position_embeddings, attention_mask=None,
                                past_key_value=None, cache_position=None, **kwargs):
                layer_data = model._dc_compressed.get(layer_idx)

                if layer_data is not None and hidden_states.shape[1] == 1:
                    from transformers.models.llama.modeling_llama import (
                        apply_rotary_pos_emb, repeat_kv,
                    )
                    import torch.nn.functional as F

                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, attn_mod.head_dim)

                    query_states = attn_mod.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    key_states = attn_mod.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_states = attn_mod.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    cos, sin = position_embeddings
                    query_states, key_states = apply_rotary_pos_emb(
                        query_states, key_states, cos, sin)

                    # Decompress cached KV
                    cK, cV, ks, vs = layer_data
                    kr = model._dc_k_comp.dequantize(cK).reshape(ks)
                    vr = model._dc_v_comp.dequantize(cV).reshape(vs)

                    # Standard attention with decompressed KV
                    B, H_kv, S, _D = kr.shape
                    H_q = query_states.shape[1]
                    group_size = H_q // H_kv
                    kr_exp = repeat_kv(kr, group_size)
                    vr_exp = repeat_kv(vr, group_size)

                    scale = attn_mod.head_dim ** -0.5
                    attn_weights = torch.matmul(
                        query_states, kr_exp.transpose(2, 3)) * scale
                    attn_weights = F.softmax(
                        attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                    attn_output = torch.matmul(attn_weights, vr_exp)

                    attn_output = attn_output.transpose(1, 2).contiguous()
                    attn_output = attn_output.reshape(*input_shape, -1)
                    attn_output = attn_mod.o_proj(attn_output)
                    return attn_output, None

                return orig(hidden_states, position_embeddings, attention_mask,
                            past_key_value, cache_position, **kwargs)

            return patched_forward

        attn_mod.forward = _make_patched(original_fwd, li)


def unpatch_model(model):
    """Remove compressed attention patches, restore original forwards."""
    # Handle both CDA and decompress patches
    for attr in ("_dc_originals", "_cda_originals"):
        if hasattr(model, attr):
            for li, orig in getattr(model, attr).items():
                model.model.layers[li].self_attn.forward = orig
    for attr in ("_cda_compressed", "_cda_originals", "_cda_k_comp", "_cda_v_comp", "_cda_topk",
                 "_dc_compressed", "_dc_originals", "_dc_k_comp", "_dc_v_comp"):
        if hasattr(model, attr):
            delattr(model, attr)


def _unpatch_model(model):
    """Legacy alias."""
    if not hasattr(model, "_cda_originals"):
        return
    for li, orig in model._cda_originals.items():
        model.model.layers[li].self_attn.forward = orig
    del model._cda_compressed, model._cda_originals, model._cda_k_comp, model._cda_v_comp


class _PositionOnlyCache(DynamicCache):
    """Thin DynamicCache that only tracks position (seq length)."""

    def __init__(self, num_layers, seq_len, device, dtype):
        super().__init__()
        if not hasattr(self, "key_cache"):
            self.key_cache = []
            self.value_cache = []
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


def calibrate_temperature(model, tokenizer, k_comp, v_comp=None, text=None,
                           max_tokens=512, skip_sinks=4):
    """Calibrate per-layer temperature for CDA score correction.

    Compares CDA attention score variance vs fp16 score variance per layer.
    Returns T_l = std(CDA_scores) / std(fp16_scores) per layer.

    Args:
        model: HuggingFace model
        tokenizer: tokenizer
        k_comp, v_comp: compressors
        text: calibration text (short is fine)
        max_tokens: max context for calibration
        skip_sinks: sink tokens

    Returns:
        list[float] of length num_layers
    """
    vc = v_comp or k_comp
    num_layers = model.config.num_hidden_layers
    num_kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    head_dim = getattr(model.config, "head_dim", 128)
    scale = head_dim ** -0.5

    if text is None:
        text = "The quick brown fox jumps over the lazy dog. " * 100
    inputs = tokenizer(text, return_tensors="pt", max_length=max_tokens, truncation=True).to(model.device)

    with torch.no_grad():
        out = model(**inputs, use_cache=True, return_dict=True)
        kv = out.past_key_values

    # Compress KV
    kv_compressed = _compress_kv_cache(kv, k_comp, vc, skip_sinks)

    # Capture Q per layer
    captured_q = {}
    hooks = []
    for li in range(num_layers):
        def make_hook(idx):
            def fn(mod, inp, out):
                captured_q[idx] = out.detach()
            return fn
        hooks.append(model.model.layers[li].self_attn.q_proj.register_forward_hook(make_hook(li)))

    with torch.no_grad():
        model(**inputs, use_cache=False)
    for h in hooks:
        h.remove()

    temperatures = []
    for li in range(num_layers):
        k_fp16 = kv.key_cache[li]  # (B, H_kv, S, D)
        B, H, S, D = k_fp16.shape
        q_all = captured_q[li].squeeze(0)  # (S, hidden)
        num_q_heads = model.config.num_attention_heads
        q_heads = q_all.reshape(S, num_q_heads, D)

        # Pick last token Q, first KV head
        q = q_heads[-1:, 0, :]  # (1, D)
        k = k_fp16[0, 0, :, :]  # (S, D)

        # fp16 scores
        scores_fp16 = (q.float() @ k.float().T) * scale  # (1, S)
        std_fp16 = scores_fp16.std().item()

        # CDA scores
        comp_data = kv_compressed[li]
        S_comp = comp_data["S_comp"]
        cK = comp_data["cK"]
        cK_h = CompressedTensor(
            indices=cK.indices[:S_comp], payload=None,
            norms=cK.norms[:S_comp], shape=torch.Size([S_comp, D]),
            dtype=cK.dtype, device=cK.device, bit_width=cK.bit_width, gpu_resident=True,
        )
        scores_cda = hw_attention_score(q.float(), cK_h, k_comp, scale)
        std_cda = scores_cda.float().std().item()

        # T = std_cda / std_fp16 (if CDA has higher variance, T > 1 → dividing reduces it)
        T = std_cda / max(std_fp16, 1e-8)
        temperatures.append(T)

    del kv, kv_compressed, captured_q
    gc.collect()
    torch.cuda.empty_cache()

    return temperatures


def compressed_eval_ppl(model, tokenizer, k_comp, v_comp, text,
                         max_length=2048, stride=512, skip_sinks=4,
                         attn_gate_topk=0, temperatures=None):
    """Evaluate PPL using compressed-domain attention (no decompress).

    Args:
        temperatures: optional list[float] per-layer temperature correction.
            Use calibrate_temperature() to compute. None = no correction.
    """
    import numpy as np

    vc = v_comp or k_comp
    enc = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    input_ids = enc.input_ids.to(model.device)
    S = input_ids.shape[1]
    positions = list(range(stride, min(S - 1, max_length), stride))
    if not positions:
        positions = [min(S - 2, max_length - 1)]

    nlls = []
    for pos in positions:
        ctx = input_ids[:, :pos]
        with torch.no_grad():
            out = model(ctx, use_cache=True, return_dict=True)
            kv = out.past_key_values

            kv_compressed = _compress_kv_cache(kv, k_comp, vc, skip_sinks)

            del kv, out
            gc.collect()
            torch.cuda.empty_cache()

            nxt = input_ids[:, pos:pos + 1]
            logits = manual_decode_step(
                model, nxt, kv_compressed, k_comp, vc,
                position=pos, attn_gate_topk=attn_gate_topk,
                temperatures=temperatures,
            )

            tgt = input_ids[:, pos + 1:pos + 2]
            if tgt.numel() > 0:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
                nlls.append(loss.item())

    return np.exp(np.mean(nlls)) if nlls else float("inf")