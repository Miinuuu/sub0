"""QuaRot-style (Hadamard + uniform RTN) baseline vs CDA (Hadamard + Lloyd-Max).

Isolates the contribution of Lloyd-Max codebook by replacing the
Beta-optimal centroids with a uniform RTN grid — matching what QuaRot
uses on its Hadamard-rotated tensors. Rotation and bit-width are
identical; only the quantizer codebook differs.

This is the fairest PPL comparison against QuaRot that does not require
cloning the QuaRot repo: it runs through the same eval harness as
ablation_hadamard.py / ablation_bitwidth.py, so numbers are directly
comparable against FP16 and CDA variants.

Config: Llama-3.1-8B-Instruct, WikiText-2, max_ctx=4096, stride=64,
        20 sliding windows. ~15-20 min per variant.

Variants:
    FP16 baseline
    CDA K4V4  (Lloyd-Max, our recipe)
    QuaRot-style K4V4  (uniform RTN, same Hadamard + bit-width)
    CDA K4V2  (our ours)
    QuaRot-style K4V2  (uniform RTN, K=4 V=2 — QuaRot doesn't do asymmetric,
        so this probes 'does asymmetric still work without Lloyd-Max')

Usage:
    CUDA_VISIBLE_DEVICES=0 python \\
        experiments/ablation_quarot_style.py
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


def _patch_uniform_rtn(compressor: HadamardQuantCompressor) -> None:
    """Replace Lloyd-Max codebook with uniform RTN centroids in [0,1].

    Our pipeline maps rotated-unit coords from [-1,1] → [0,1] before
    quantization, so uniform RTN in the mapped space = uniform in the
    original rotated space. This matches QuaRot's symmetric RTN.
    """
    num_levels = 1 << compressor.bit_width
    # cell mid-points in [0, 1]: {1/(2N), 3/(2N), ..., (2N-1)/(2N)}
    centers = (np.arange(num_levels) + 0.5) / num_levels  # (N,)
    boundaries = np.concatenate([[0.0], (centers[:-1] + centers[1:]) / 2, [1.0]])

    compressor._centroids_np = centers.astype(np.float32)
    compressor._boundaries_np = boundaries.astype(np.float32)
    # Force tensor rebuild on next call
    compressor._centroids = None
    compressor._boundaries = None
    compressor._current_device = None


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
def eval_ppl(model, tokenizer, fn, max_ctx=4096, stride=64, desc=""):
    """Growing-context PPL (matches bench_ppl_scale.py used for Tab.~ppl)."""
    text = _get_wikitext_text()
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False,
                    max_length=max_ctx, truncation=True).input_ids
    ids = ids.to(model.device)
    positions = list(range(stride, min(ids.shape[1] - 1, max_ctx), stride))

    nlls = []
    t0 = time.time()
    for pos in positions:
        ctx = ids[:, :pos]
        nxt = ids[:, pos:pos + 1]
        tgt = ids[:, pos + 1:pos + 2]
        if tgt.numel() == 0:
            break
        logits = fn(ctx, nxt)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
        nlls.append(float(loss.item()))

    t_elapsed = time.time() - t0
    ppl = float(np.exp(np.mean(nlls))) if nlls else float("inf")
    return {"ppl": round(ppl, 4), "n_pos": len(nlls),
            "time_s": round(t_elapsed, 1), "desc": desc}


def run_variant(model, tokenizer, n_layers, head_dim, label, *,
                bk: int, bv: int, uniform_rtn: bool):
    kc = HadamardQuantCompressor(dim=head_dim, bit_width=bk, half_rotation=True)
    vc = HadamardQuantCompressor(dim=head_dim, bit_width=bv, half_rotation=True)
    if uniform_rtn:
        _patch_uniform_rtn(kc)
        _patch_uniform_rtn(vc)
    kc._ensure_tensors(model.device)
    vc._ensure_tensors(model.device)
    fn = _make_compress_fn(model, kc, vc, skip_sinks=4, n_layers=n_layers)
    return eval_ppl(model, tokenizer, fn, max_ctx=4096, stride=64, desc=label)


def main():
    print("Loading Llama-3.1-70B-Instruct (device_map=auto) ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-70B-Instruct",
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
                                max_ctx=4096, stride=64, desc="FP16 baseline")
    print(f"  {results['fp16']}")

    sweeps = [
        ("cda_k4v4",     "CDA K4V4 (Lloyd-Max)",    4, 4, False),
        ("quarot_k4v4",  "QuaRot-style K4V4 (RTN)", 4, 4, True),
        ("cda_k4v2",     "CDA K4V2 (Lloyd-Max)",    4, 2, False),
        ("quarot_k4v2",  "QuaRot-style K4V2 (RTN)", 4, 2, True),
    ]
    for key, label, bk, bv, rtn in sweeps:
        print(f"\n=== {label} ===")
        results[key] = run_variant(model, tokenizer, n_layers, head_dim, label,
                                   bk=bk, bv=bv, uniform_rtn=rtn)
        print(f"  {results[key]}")

    # Summary
    fp16_ppl = results["fp16"]["ppl"]
    print("\n" + "=" * 68)
    print("QuaRot-style vs CDA — Llama-3.1-70B WikiText-2:")
    print(f"  {'variant':<32}  {'PPL':>8}  {'Δ vs FP16':>10}")
    for k, r in results.items():
        delta = (r["ppl"] - fp16_ppl) / fp16_ppl * 100
        print(f"  {r['desc']:<32}  {r['ppl']:>8.4f}  {delta:>+9.2f}%")

    # Head-to-head gap (Lloyd-Max vs RTN at same bit-width)
    for pair in [("cda_k4v4", "quarot_k4v4"), ("cda_k4v2", "quarot_k4v2")]:
        cda, qr = pair
        if cda in results and qr in results:
            gap = results[qr]["ppl"] - results[cda]["ppl"]
            rel = gap / results[cda]["ppl"] * 100
            print(f"  Lloyd-Max gap  {cda:>16} → {qr:<16}  ΔPPL = {gap:+.4f} ({rel:+.2f}%)")

    Path("runs").mkdir(exist_ok=True)
    Path("runs/ablation_quarot_style.json").write_text(json.dumps(results, indent=2))
    print(f"\n  → runs/ablation_quarot_style.json")


if __name__ == "__main__":
    main()
