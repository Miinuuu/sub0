"""Bit-width ablation: K4V2 vs K2V2 vs K4V4 vs FP16 baseline.

Shows the Pareto frontier of compression vs quality. Justifies K4V2 as the
sweet spot for CDA (matches Gemini Priority 3 recommendation #2).

Usage:
    CUDA_VISIBLE_DEVICES=1 python experiments/ablation_bitwidth.py
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
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join(ds["text"])


def _make_compress_fn(model, kc, vc, skip_sinks, n_layers):
    def fn(ctx, nxt):
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
    text = _get_wikitext_text()
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    ids = ids.to(model.device)
    positions = list(range(max_ctx, ids.shape[1] - 1, stride))[:20]
    nlls = []
    t0 = time.time()
    for pos in positions:
        ctx = ids[:, :pos]
        nxt = ids[:, pos:pos + 1]
        tgt = ids[:, pos + 1:pos + 2]
        if tgt.numel() == 0: break
        logits = fn(ctx, nxt)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
        nlls.append(float(loss.item()))
    t_elapsed = time.time() - t0
    ppl = float(np.exp(np.mean(nlls))) if nlls else float("inf")
    return {"ppl": round(ppl, 4), "n_pos": len(nlls),
            "time_s": round(t_elapsed, 1), "desc": desc}


def kv_bytes_per_slot(k_bits, v_bits, D=128):
    """Bytes per (position × H_kv) slot, including fp32 norms (8 bytes)."""
    return D * k_bits // 8 + D * v_bits // 8 + 8


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
    print(f"  Loaded in {time.time()-t0:.0f}s")

    results = {}

    print("\n=== FP16 baseline ===")
    def fp16_fn(ctx, nxt):
        out = model(ctx, use_cache=True, return_dict=True)
        return model(nxt, past_key_values=out.past_key_values, use_cache=False).logits
    results["fp16"] = eval_ppl(model, tokenizer, fp16_fn, desc="FP16 (512 B/slot)")
    results["fp16"]["kv_bytes_per_slot"] = 128 * 2 * 2  # K+V fp16
    print(f"  {results['fp16']}")

    # Bit-width configurations.
    configs = [
        ("k8v8", 8, 8, "K8V8"),
        ("k4v4", 4, 4, "K4V4"),
        ("k4v2", 4, 2, "K4V2 (ours)"),
        ("k2v2", 2, 2, "K2V2"),
    ]

    for key, kb, vb, label in configs:
        print(f"\n=== CDA {label} (K={kb}bit V={vb}bit) ===")
        kc = HadamardQuantCompressor(dim=head_dim, bit_width=kb, half_rotation=True)
        vc = HadamardQuantCompressor(dim=head_dim, bit_width=vb, half_rotation=True)
        kc._ensure_tensors(model.device); vc._ensure_tensors(model.device)
        fn = _make_compress_fn(model, kc, vc, skip_sinks=4, n_layers=n_layers)
        bpslot = kv_bytes_per_slot(kb, vb)
        results[key] = eval_ppl(model, tokenizer, fn, desc=f"CDA {label} ({bpslot} B/slot)")
        results[key]["kv_bytes_per_slot"] = bpslot
        results[key]["k_bits"] = kb
        results[key]["v_bits"] = vb
        print(f"  {results[key]}")

    # Summary.
    fp16_ppl = results["fp16"]["ppl"]
    fp16_bytes = results["fp16"]["kv_bytes_per_slot"]
    print("\n" + "=" * 70)
    print("Bit-width ablation — Llama-3.1-8B WikiText-2 (PPL, KV memory):")
    print(f"  {'variant':<24}  {'PPL':>8}  {'Δ vs FP16':>10}  "
          f"{'B/slot':>7}  {'Ratio':>7}")
    for k, r in results.items():
        delta = (r["ppl"] - fp16_ppl) / fp16_ppl * 100
        ratio = fp16_bytes / r["kv_bytes_per_slot"]
        print(f"  {r['desc']:<24}  {r['ppl']:>8.4f}  {delta:>+9.2f}%  "
              f"{r['kv_bytes_per_slot']:>6}B  {ratio:>6.2f}×")

    Path("runs").mkdir(exist_ok=True)
    Path("runs/ablation_bitwidth.json").write_text(json.dumps(results, indent=2))
    print("\n  → runs/ablation_bitwidth.json")


if __name__ == "__main__":
    main()
