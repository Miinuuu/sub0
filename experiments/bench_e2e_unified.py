"""Unified E2E decode benchmark using sub0's packaged CUDA kernels only.

Reproduces **Figure 5(c)** (Llama-3.1-8B, A6000, TopK=0, all methods inside
``model.forward()``):

    - FP16      : standard HF forward with fp16 KV cache
    - KIVI 2-bit: HF built-in ``QuantoQuantizedCache`` (optional)
    - CDA K4/V2 : fused CUDA kernel (``cda.cuda_attention.cuda_hw_attention_batched``)
                  injected into each LlamaAttention.forward via monkey-patch

Unlike the research-tree version, this script uses only the sub0 public API:
``cda.HadamardQuantCompressor`` + ``cda.cuda_attention.cuda_hw_attention_batched``.
It inlines the attention replacement that the research tree's
``patch_model_compressed_attn`` provides, so it can be run directly from a
binary install of the sub0 package.

Methodology notes (matches the research script):
  - The compressed K/V stored per layer reflects the prefill context of length
    N. The current decode token's K/V is NOT folded in — we measure attention
    cost against N pre-compressed tokens per step (per-step microbench at
    fixed N, which is how Figure 5(c) is measured).
  - FP16 column uses the default SDPA path with the original attention.
  - All methods share the same ``model(nxt, past_key_values=cache)`` harness.

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/bench_e2e_unified.py --N 8192
    CUDA_VISIBLE_DEVICES=0 python experiments/bench_e2e_unified.py --N 65536 --skip-kivi
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb  # noqa: E402

from cda import HadamardQuantCompressor  # noqa: E402
from cda.cuda_attention import cuda_hw_attention_batched  # noqa: E402


# ---------------------------------------------------------------------------
# CDA monkey-patch: replace LlamaAttention.forward with a compressed-domain
# variant that calls the fused CUDA kernel against precomputed compressed K/V.
# ---------------------------------------------------------------------------

def _precompute_cda_kv(fp16_kv, c_k: HadamardQuantCompressor,
                       c_v: HadamardQuantCompressor, device):
    """Compress prefill KV per layer; pre-expand for GQA to kernel layout.

    Returns a dict with per-layer (packed_K, norms_K, packed_V, norms_V) already
    expanded along the num_heads axis, plus shared codebooks and rotation.
    """
    c_k._ensure_tensors(device)
    c_v._ensure_tensors(device)

    codebook_k = (c_k._centroids * 2.0 - 1.0).float().contiguous()
    codebook_v = (c_v._centroids * 2.0 - 1.0).float().contiguous()
    rotation = c_k._rotation.float().contiguous()
    rotation_t = c_k._rotation_t.float().contiguous()

    layers = []
    n_layers = len(fp16_kv.key_cache)
    for li in range(n_layers):
        k = fp16_kv.key_cache[li]   # (B, num_kv_heads, N, D)
        v = fp16_kv.value_cache[li]
        B, H_kv, N, D = k.shape
        cK = c_k.quantize(k.reshape(-1, D))
        cV = c_v.quantize(v.reshape(-1, D))
        layers.append({
            "pK": cK.indices.view(B, H_kv, N, -1).contiguous(),
            "nK": cK.norms.float().view(B, H_kv, N).contiguous(),
            "pV": cV.indices.view(B, H_kv, N, -1).contiguous(),
            "nV": cV.norms.float().view(B, H_kv, N).contiguous(),
            "N": N,
        })
    return {
        "layers": layers,
        "codebook_k": codebook_k,
        "codebook_v": codebook_v,
        "rotation": rotation,
        "rotation_t": rotation_t,
    }


def _make_cda_attn_forward(layer_idx: int, cda_data: dict):
    def cda_forward(self, hidden_states, position_embeddings,
                    attention_mask=None, past_key_value=None,
                    cache_position=None, **kwargs):
        B, S, _ = hidden_states.shape
        D = self.head_dim
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        group = num_heads // num_kv_heads
        scale = 1.0 / math.sqrt(D)

        # Project Q (and K to satisfy RoPE call shape; K value is discarded).
        q = self.q_proj(hidden_states).view(B, S, num_heads, D).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, S, num_kv_heads, D).transpose(1, 2)
        cos, sin = position_embeddings
        q, _ = apply_rotary_pos_emb(q, k, cos, sin)
        # q: (B, num_heads, S, D); decode has S=1.

        Q_flat = q.permute(0, 2, 1, 3).reshape(B * S * num_heads, D).float().contiguous()
        Q_rot = Q_flat @ cda_data["rotation_t"]   # (M, D)

        ld = cda_data["layers"][layer_idx]
        N = ld["N"]
        # GQA expand (H_kv → num_heads) once per call. pK stored (B, H_kv, N, pack).
        pK = ld["pK"].repeat_interleave(group, dim=1).reshape(B * num_heads * N, -1).contiguous()
        nK = ld["nK"].repeat_interleave(group, dim=1).reshape(B * num_heads * N).contiguous()
        pV = ld["pV"].repeat_interleave(group, dim=1).reshape(B * num_heads * N, -1).contiguous()
        nV = ld["nV"].repeat_interleave(group, dim=1).reshape(B * num_heads * N).contiguous()

        out = cuda_hw_attention_batched(
            Q_rot_all=Q_rot,
            packed_indices_K=pK, norms_K=nK,
            packed_indices_V=pV, norms_V=nV,
            codebook=cda_data["codebook_k"],
            codebook_v=cda_data["codebook_v"],
            rotation=cda_data["rotation"],
            scale=scale, N=N,
            bit_width_k=4, bit_width_v=2,
        )  # (B*S*num_heads, D) fp32

        out = out.to(hidden_states.dtype).view(B, S, num_heads, D).permute(0, 2, 1, 3).contiguous()
        out = out.transpose(1, 2).reshape(B, S, num_heads * D)
        return self.o_proj(out), None
    return cda_forward


def _install_cda_patch(model, cda_data):
    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        attn._cda_orig_forward = attn.forward
        attn.forward = types.MethodType(_make_cda_attn_forward(li, cda_data), attn)


def _uninstall_cda_patch(model):
    for layer in model.model.layers:
        attn = layer.self_attn
        if hasattr(attn, "_cda_orig_forward"):
            attn.forward = attn._cda_orig_forward
            del attn._cda_orig_forward


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------

def _measure(model, nxt, cache_factory, iters: int, warmup: int) -> float:
    cache = cache_factory()
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(nxt, past_key_values=cache, use_cache=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            _ = model(nxt, past_key_values=cache, use_cache=True)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=True, help="context length")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--skip-kivi", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    N = args.N
    iters = 5 if N >= 32768 else (10 if N >= 16384 else 20)
    warmup = 2 if N >= 32768 else (3 if N >= 16384 else 5)

    device = torch.device("cuda:0")
    print(f"Loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
        device_map=device, low_cpu_mem_usage=True,
    ).eval()
    D = getattr(model.config, "head_dim", None) or (
        model.config.hidden_size // model.config.num_attention_heads
    )

    # Build prefill input of exact length N.
    prompt = "The quick brown fox jumps over the lazy dog. " * 32768
    enc = tokenizer(prompt, return_tensors="pt", max_length=N,
                    truncation=True).to(device)
    input_ids = enc.input_ids
    ctx = input_ids[:, :-1]
    nxt = input_ids[:, -1:]

    row = {"N": N, "model": args.model}

    # --- 1. FP16 ------------------------------------------------------------
    with torch.no_grad():
        fp16_kv = model(ctx, use_cache=True, return_dict=True).past_key_values
    fp16_ms = _measure(model, nxt, lambda: fp16_kv, iters, warmup)
    row["fp16_ms"] = round(fp16_ms, 2)
    print(f"  FP16        : {fp16_ms:7.2f} ms")

    # --- 2. KIVI 2-bit (optional) ------------------------------------------
    if not args.skip_kivi:
        try:
            from transformers import QuantizedCacheConfig
            from transformers.cache_utils import QuantoQuantizedCache
            with torch.no_grad():
                kivi_cache = QuantoQuantizedCache(QuantizedCacheConfig(
                    nbits=2, backend="quanto", axis_key=0, axis_value=0))
                model(ctx, use_cache=True, return_dict=True,
                      past_key_values=kivi_cache)
            kivi_ms = _measure(model, nxt, lambda: kivi_cache, iters, warmup)
            row["kivi_ms"] = round(kivi_ms, 2)
            print(f"  KIVI 2-bit  : {kivi_ms:7.2f} ms")
            del kivi_cache; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            row["kivi_ms"] = f"FAIL:{type(e).__name__}"
            print(f"  KIVI        : FAIL {type(e).__name__}: {e}")

    # --- 3. CDA K4/V2 (fused CUDA kernel via sub0 public API) --------------
    c_k = HadamardQuantCompressor(dim=D, bit_width=4, half_rotation=True)
    c_v = HadamardQuantCompressor(dim=D, bit_width=2, half_rotation=True)
    cda_data = _precompute_cda_kv(fp16_kv, c_k, c_v, device)
    _install_cda_patch(model, cda_data)
    try:
        cda_ms = _measure(model, nxt, lambda: fp16_kv, iters, warmup)
        row["cda_k4v2_ms"] = round(cda_ms, 2)
        print(f"  CDA K4/V2   : {cda_ms:7.2f} ms")
    finally:
        _uninstall_cda_patch(model)

    # Speedups
    for k in ("kivi_ms", "cda_k4v2_ms"):
        if isinstance(row.get(k), (int, float)):
            row[k.replace("_ms", "_vs_fp16")] = round(row["fp16_ms"] / row[k], 2)

    print()
    print(json.dumps(row, indent=2))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(row, indent=2))
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
