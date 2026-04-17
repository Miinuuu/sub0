"""Strict re-measurement of Figure 5(c) at N=32K.

Differences vs bench_cda_integrated_single.py:
  * iters = 30 (vs 5) for low-variance statistics.
  * Per-iter KV reset via deep-copied cache (FP16 path stays at exactly N).
  * torch.cuda.Event timing (hardware clock) instead of perf_counter.
  * Reports mean / std / min / median.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from core.compression import HadamardQuantCompressor  # noqa: E402
from core.cda_attn import (  # noqa: E402
    _PositionOnlyCache,
    _compress_kv_cache_cuda,
    patch_model_compressed_attn,
    unpatch_model,
)


def _event_time_ms(fn, iters, warmup, setup=None):
    """Run fn() iters times, returning list of per-iter ms using cuda.Event.

    If setup is given, setup() runs OUTSIDE the timed window each iter
    (host-side sync before record ensures it fully completes first).
    """
    for _ in range(warmup):
        if setup is not None: setup()
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if setup is not None:
            setup()
            torch.cuda.synchronize()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return [starts[i].elapsed_time(ends[i]) for i in range(iters)]


def _stats(times):
    return {
        "mean": round(statistics.mean(times), 3),
        "std":  round(statistics.stdev(times) if len(times) > 1 else 0.0, 3),
        "min":  round(min(times), 3),
        "median": round(statistics.median(times), 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=32768)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    N = args.N
    device = torch.device("cuda:0")
    print(f"Loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
        attn_implementation="sdpa", device_map=device, low_cpu_mem_usage=True,
    ).eval()
    n_layers = model.config.num_hidden_layers
    D = getattr(model.config, "head_dim", None) or (
        model.config.hidden_size // model.config.num_attention_heads
    )

    prompt = "The quick brown fox jumps over the lazy dog. " * 32768
    enc = tokenizer(prompt, return_tensors="pt", max_length=N, truncation=True).to(device)
    ctx = enc.input_ids[:, :-1]
    nxt = enc.input_ids[:, -1:]
    S = ctx.shape[1]
    print(f"  N (context): {S}, warmup={args.warmup}, iters={args.iters}")

    # --- Prefill KV (fresh each iter via deep copy) ------------------------
    with torch.no_grad():
        fp16_kv_base = model(ctx, use_cache=True, return_dict=True).past_key_values

    # Save pristine K/V per layer so we can reset fp16_kv each iter.
    k_snap = [k.clone() for k in fp16_kv_base.key_cache]
    v_snap = [v.clone() for v in fp16_kv_base.value_cache]

    def reset_fp16_kv():
        for li in range(n_layers):
            fp16_kv_base.key_cache[li]   = k_snap[li].clone()
            fp16_kv_base.value_cache[li] = v_snap[li].clone()

    def fp16_step():
        with torch.no_grad():
            _ = model(nxt, past_key_values=fp16_kv_base, use_cache=True)

    fp16_times = _event_time_ms(fp16_step, args.iters, args.warmup, setup=reset_fp16_kv)
    fp16 = _stats(fp16_times)
    print(f"  FP16        : {fp16['mean']:7.3f} ± {fp16['std']:.3f} ms "
          f"(min {fp16['min']:.3f}, med {fp16['median']:.3f})")

    row = {"N": N, "iters": args.iters, "warmup": args.warmup,
           "model": args.model, "fp16": fp16}

    # --- CDA configs -------------------------------------------------------
    configs = {
        "K4V2": (HadamardQuantCompressor(dim=D, bit_width=4, half_rotation=True),
                 HadamardQuantCompressor(dim=D, bit_width=2, half_rotation=True)),
        "K2V2": (HadamardQuantCompressor(dim=D, bit_width=2, half_rotation=True),
                 HadamardQuantCompressor(dim=D, bit_width=2, half_rotation=True)),
    }

    for label, (kc, vc) in configs.items():
        compressed = _compress_kv_cache_cuda(fp16_kv_base, kc, vc)
        patch_model_compressed_attn(model, kc, vc, attn_gate_topk=0)
        for li in range(n_layers):
            model._cda_compressed[li] = compressed[li]
        for key in ("_rotation", "_codebook_k", "_codebook_v"):
            model._cda_compressed[key] = compressed[key]

        cda_pc = {"pc": None}
        def cda_setup():
            cda_pc["pc"] = _PositionOnlyCache(n_layers, S, device, torch.float16)
        def cda_step():
            with torch.no_grad():
                _ = model(nxt, past_key_values=cda_pc["pc"], use_cache=True)

        cda_times = _event_time_ms(cda_step, args.iters, args.warmup, setup=cda_setup)
        st = _stats(cda_times)
        st["vs_fp16"] = round(fp16["mean"] / st["mean"], 3)
        row[label.lower()] = st
        print(f"  CDA {label:5s}  : {st['mean']:7.3f} ± {st['std']:.3f} ms "
              f"(min {st['min']:.3f}, med {st['median']:.3f})  "
              f"→ {st['vs_fp16']:.2f}x vs FP16")

        unpatch_model(model)
        del compressed
        gc.collect()
        torch.cuda.empty_cache()

    print()
    print(json.dumps(row, indent=2))
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(row, indent=2))
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
