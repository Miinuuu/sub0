"""Hadamard ablation: PPL with vs without rotation.

Monkey-patches HadamardQuantCompressor._rotation to identity so we can
measure the accuracy loss from dropping rotation while keeping bit-width
and codebook identical.

Config: Llama-3.1-8B-Instruct, WikiText-2, max_ctx=2048, K4V2.
Expected runtime: 10-20 min per variant × 2 variants.

Usage:
    CUDA_VISIBLE_DEVICES=1 python experiments/ablation_hadamard.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from core.compression import HadamardQuantCompressor


def _get_wikitext_text():
    """Load WikiText-2 test text (assumes it's cached)."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join(ds["text"])


def _make_compress_fn(model, kc, vc, skip_sinks, n_layers):
    def fn(ctx, nxt):
        # Prefill on ctx to build KV cache.
        out = model(ctx, use_cache=True, return_dict=True)
        kv_cache = out.past_key_values

        new_kv = DynamicCache()
        for li in range(n_layers):
            if hasattr(kv_cache, "layers") and len(kv_cache.layers) > li:
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
                ko_rest = kc.dequantize(kc.quantize(k_rest)).reshape(B, H, S - skip_sinks, D)
                vo_rest = vc.dequantize(vc.quantize(v_rest)).reshape(B, H, S - skip_sinks, D)
                ko = torch.cat([k_sink, ko_rest], dim=2)
                vo = torch.cat([v_sink, vo_rest], dim=2)
            else:
                ko = kc.dequantize(kc.quantize(k.reshape(-1, D))).reshape(k.shape)
                vo = vc.dequantize(vc.quantize(v.reshape(-1, D))).reshape(v.shape)
            new_kv.update(ko, vo, li)
        return model(nxt, past_key_values=new_kv, use_cache=False).logits
    return fn


@torch.no_grad()
def eval_ppl(model, tokenizer, fn, max_ctx=2048, stride=64, desc=""):
    """Evaluate PPL using the standard sliding-window approach.

    For each anchor position pos in the sliding window:
      - ctx = tokens [0, pos)
      - nxt = token at position pos
      - tgt = token at position pos+1 (what we predict)
    fn(ctx, nxt) must return logits predicting tgt.
    """
    text = _get_wikitext_text()
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    ids = ids.to(model.device)

    # Pick anchor positions: start at max_ctx, step by stride, limit 20 windows.
    positions = list(range(max_ctx, ids.shape[1] - 1, stride))[:20]

    nlls = []
    t0 = time.time()
    for pos in positions:
        ctx = ids[:, :pos]
        nxt = ids[:, pos:pos + 1]
        tgt = ids[:, pos + 1:pos + 2]
        if tgt.numel() == 0:
            break
        logits = fn(ctx, nxt)
        # logits shape: (1, 1, V). Predicts next token after nxt = position pos+1.
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
        nlls.append(float(loss.item()))

    t_elapsed = time.time() - t0
    ppl = float(np.exp(np.mean(nlls))) if nlls else float("inf")
    return {"ppl": round(ppl, 4), "n_pos": len(nlls),
            "time_s": round(t_elapsed, 1), "desc": desc}


def run_cda_variant(model, tokenizer, n_layers, head_dim, label, *,
                    patch_identity=False):
    kc = HadamardQuantCompressor(dim=head_dim, bit_width=4, half_rotation=True)
    vc = HadamardQuantCompressor(dim=head_dim, bit_width=2, half_rotation=True)
    kc._ensure_tensors(model.device); vc._ensure_tensors(model.device)

    if patch_identity:
        eye = torch.eye(head_dim, dtype=kc._rotation.dtype, device=model.device)
        kc._rotation   = eye
        kc._rotation_t = eye
        vc._rotation   = eye
        vc._rotation_t = eye
        label = label + " (no rotation)"

    fn = _make_compress_fn(model, kc, vc, skip_sinks=4, n_layers=n_layers)
    return eval_ppl(model, tokenizer, fn, max_ctx=2048, stride=64, desc=label)


def main():
    print("Loading Llama-3.1-8B-Instruct ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    head_dim = 128
    n_layers = model.config.num_hidden_layers
    print(f"  Loaded in {time.time()-t0:.0f}s. head_dim={head_dim}, n_layers={n_layers}")

    results = {}

    # FP16 baseline.
    print("\n=== FP16 baseline ===")
    def fp16_fn(ctx, nxt):
        out = model(ctx, use_cache=True, return_dict=True)
        return model(nxt, past_key_values=out.past_key_values, use_cache=False).logits
    results["fp16"] = eval_ppl(model, tokenizer, fp16_fn,
                                max_ctx=2048, stride=64, desc="FP16")
    print(f"  {results['fp16']}")

    # CDA K4V2 (with Hadamard rotation).
    print("\n=== CDA K4V2 (with Hadamard) ===")
    results["cda_k4v2"] = run_cda_variant(
        model, tokenizer, n_layers, head_dim,
        "CDA K4V2 (Hadamard)", patch_identity=False)
    print(f"  {results['cda_k4v2']}")

    # CDA K4V2 (no rotation — identity).
    print("\n=== CDA K4V2 (no rotation, identity) ===")
    results["cda_k4v2_noH"] = run_cda_variant(
        model, tokenizer, n_layers, head_dim,
        "CDA K4V2 (identity)", patch_identity=True)
    print(f"  {results['cda_k4v2_noH']}")

    # Summary.
    fp16_ppl = results["fp16"]["ppl"]
    print("\n" + "=" * 60)
    print("Hadamard ablation summary — Llama-3.1-8B WikiText-2:")
    print(f"  {'variant':<30}  {'PPL':>8}  {'Δ vs FP16':>10}")
    for k, r in results.items():
        delta = (r["ppl"] - fp16_ppl) / fp16_ppl * 100
        print(f"  {r['desc']:<30}  {r['ppl']:>8.4f}  {delta:>+9.2f}%")

    Path("runs").mkdir(exist_ok=True)
    Path("runs/ablation_hadamard.json").write_text(json.dumps(results, indent=2))
    print(f"\n  → runs/ablation_hadamard.json")


if __name__ == "__main__":
    main()
