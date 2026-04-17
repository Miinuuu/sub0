"""Figure 4(c) memory footprint: peak GPU memory per config across N=1K..128K.

For each N, measures:
  * Model weights (baseline)
  * FP16 KV cache (SDPA/FA2 share same KV)
  * CDA K4V2 compressed KV (packed_K + norms + packed_V + norms + sinks)
  * CDA K2V2 compressed KV
  * Peak decode memory (model + KV + activations) per config

Uses torch.cuda.max_memory_allocated() around each phase.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.compression import HadamardQuantCompressor  # noqa: E402
from core.cda_attn import (  # noqa: E402  (ported from sub0 — uses _cda_gqa_kernels.so)
    _compress_kv_cache_cuda,
    patch_model_compressed_attn,
    unpatch_model,
)

# core.cda_attn now provides _PositionOnlyCache; fall back to the bench helper
# if running against older checkouts that still define it locally.
# integrated bench.
from experiments.bench_cda_integrated_single import _PositionOnlyCache  # noqa: E402


def _mb(bytes_):
    return round(bytes_ / (1024 * 1024), 2)


def _kv_cache_bytes(kv_cache):
    """Total bytes occupied by an HF DynamicCache."""
    tot = 0
    for k in kv_cache.key_cache:
        tot += k.numel() * k.element_size()
    for v in kv_cache.value_cache:
        tot += v.numel() * v.element_size()
    return tot


def _cda_cache_bytes(compressed, n_layers):
    """Total bytes for the CDA-repo compressed cache."""
    tot = 0
    for li in range(n_layers):
        d = compressed[li]
        for key in ("packed_K", "norms_K", "packed_V", "norms_V",
                    "k_sink", "v_sink"):
            t = d.get(key)
            if t is None: continue
            tot += t.numel() * t.element_size()
    for key in ("_rotation", "_codebook_k", "_codebook_v"):
        t = compressed.get(key)
        if t is None: continue
        tot += t.numel() * t.element_size()
    return tot


def measure_N(N, model_name, device):
    print(f"\n{'='*60}\n  N = {N}\n{'='*60}")
    row = {"N": N}

    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        attn_implementation="sdpa", device_map=device, low_cpu_mem_usage=True).eval()

    torch.cuda.synchronize()
    weights_bytes = torch.cuda.memory_allocated()
    row["model_weights_MB"] = _mb(weights_bytes)
    print(f"  Model weights     : {row['model_weights_MB']:>10.2f} MB")

    prompt = "The quick brown fox jumps over the lazy dog. " * 32768
    enc = tokenizer(prompt, return_tensors="pt", max_length=N, truncation=True).to(device)
    ctx = enc.input_ids[:, :-1]; nxt = enc.input_ids[:, -1:]
    S = ctx.shape[1]

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        kv = model(ctx, use_cache=True, return_dict=True,
                   num_logits_to_keep=1).past_key_values

    fp16_kv_bytes = _kv_cache_bytes(kv)
    prefill_peak = torch.cuda.max_memory_allocated()
    row["fp16_kv_MB"]      = _mb(fp16_kv_bytes)
    row["prefill_peak_MB"] = _mb(prefill_peak)
    print(f"  FP16 KV cache     : {row['fp16_kv_MB']:>10.2f} MB")
    print(f"  Prefill peak      : {row['prefill_peak_MB']:>10.2f} MB")

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(3):
            _ = model(nxt, past_key_values=kv, use_cache=True)
    row["sdpa_decode_peak_MB"] = _mb(torch.cuda.max_memory_allocated())
    print(f"  SDPA decode peak  : {row['sdpa_decode_peak_MB']:>10.2f} MB")

    D = getattr(model.config, "head_dim", None) or (
        model.config.hidden_size // model.config.num_attention_heads)
    n_layers = model.config.num_hidden_layers

    configs = {
        "K4V2": (HadamardQuantCompressor(dim=D, bit_width=4, half_rotation=True),
                 HadamardQuantCompressor(dim=D, bit_width=2, half_rotation=True)),
        "K2V2": (HadamardQuantCompressor(dim=D, bit_width=2, half_rotation=True),
                 HadamardQuantCompressor(dim=D, bit_width=2, half_rotation=True)),
    }
    for label, (kc, vc) in configs.items():
        # Reuse the already-prefilled kv (no re-prefill → fits at 128K)
        compressed = _compress_kv_cache_cuda(kv, kc, vc)
        cda_kv_bytes = _cda_cache_bytes(compressed, n_layers)
        row[f"cda_{label.lower()}_kv_MB"]       = _mb(cda_kv_bytes)
        row[f"cda_{label.lower()}_kv_vs_fp16"]  = round(fp16_kv_bytes / cda_kv_bytes, 2)

        patch_model_compressed_attn(model, kc, vc, attn_gate_topk=0)
        for li in range(n_layers):
            model._cda_compressed[li] = compressed[li]
        for key in ("_rotation", "_codebook_k", "_codebook_v"):
            model._cda_compressed[key] = compressed[key]

        pc = _PositionOnlyCache(n_layers, S, device, torch.float16)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for _ in range(3):
                _ = model(nxt, past_key_values=pc, use_cache=True)
        row[f"cda_{label.lower()}_decode_peak_MB"] = _mb(torch.cuda.max_memory_allocated())

        print(f"  CDA {label} KV       : {row[f'cda_{label.lower()}_kv_MB']:>10.2f} MB  "
              f"({row[f'cda_{label.lower()}_kv_vs_fp16']:.2f}x reduction)")
        print(f"  CDA {label} decode pk: {row[f'cda_{label.lower()}_decode_peak_MB']:>10.2f} MB")

        unpatch_model(model)
        del compressed, pc
        gc.collect(); torch.cuda.empty_cache()

    del kv, model
    gc.collect(); torch.cuda.empty_cache()
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--Ns", default="1024,2048,4096,8192,16384,32768,65536,131072")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--output", default="runs/fig4c_memory.json")
    args = p.parse_args()

    device = torch.device("cuda:0")
    Ns = [int(n) for n in args.Ns.split(",")]
    rows = []
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for N in Ns:
        try:
            rows.append(measure_N(N, args.model, device))
        except torch.cuda.OutOfMemoryError as e:
            print(f"  !! OOM at N={N}: {e}")
            rows.append({"N": N, "error": "OOM"})
        gc.collect(); torch.cuda.empty_cache()
        out_path.write_text(json.dumps(rows, indent=2))
        print(f"  [saved {len(rows)} rows → {out_path}]")

    print("\n\n=== MEMORY SUMMARY (MB) ===")
    hdr = (f"{'N':>7}  {'FP16-KV':>10}  {'K4V2-KV':>10}  {'K2V2-KV':>10}  "
           f"{'K4V2/FP16':>10}  {'K2V2/FP16':>10}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r['N']:>7}  {'OOM':>10}")
            continue
        print(f"{r['N']:>7}  {r['fp16_kv_MB']:>10.1f}  "
              f"{r['cda_k4v2_kv_MB']:>10.1f}  {r['cda_k2v2_kv_MB']:>10.1f}  "
              f"{r['cda_k4v2_kv_vs_fp16']:>9.2f}x  {r['cda_k2v2_kv_vs_fp16']:>9.2f}x")


if __name__ == "__main__":
    main()
