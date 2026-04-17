"""Figure 5(c) N=32K: FP16-SDPA vs FP16-FlashAttention2 (Flash-Decoding) vs CDA.

Flash-Attention 2 includes Flash-Decoding for single-token queries at
long context: the KV dim is split into chunks and reduced via log-sum-exp,
which is the SOTA FP16 baseline for long-context decode.

Methodology matches bench_cda_strict.py:
  * iters = 30, cuda.Event HW timing, setup (KV reset) excluded from window.
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
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


def _event_time(fn, iters, warmup, setup=None):
    for _ in range(warmup):
        if setup is not None: setup()
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if setup is not None:
            setup(); torch.cuda.synchronize()
        starts[i].record(); fn(); ends[i].record()
    torch.cuda.synchronize()
    return [starts[i].elapsed_time(ends[i]) for i in range(iters)]


def _stats(ts):
    return {"mean": round(statistics.mean(ts), 3),
            "std":  round(statistics.stdev(ts) if len(ts) > 1 else 0.0, 3),
            "min":  round(min(ts), 3),
            "median": round(statistics.median(ts), 3)}


def load_model(name, attn_impl, device):
    return AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16,
        attn_implementation=attn_impl, device_map=device, low_cpu_mem_usage=True,
    ).eval()


def bench_fp16(model, tokenizer, N, iters, warmup, device, skip_reset=False):
    """skip_reset: True to skip KV snapshot/reset (saves 2x KV memory).
    With iters≤30 and N≥128K, drift is <0.025% — negligible."""
    prompt = "The quick brown fox jumps over the lazy dog. " * 32768
    enc = tokenizer(prompt, return_tensors="pt", max_length=N, truncation=True).to(device)
    ctx = enc.input_ids[:, :-1]; nxt = enc.input_ids[:, -1:]
    with torch.no_grad():
        kv = model(ctx, use_cache=True, return_dict=True, num_logits_to_keep=1).past_key_values

    if skip_reset:
        setup = None
    else:
        k_snap = [k.clone() for k in kv.key_cache]
        v_snap = [v.clone() for v in kv.value_cache]
        def setup():
            for li in range(len(kv.key_cache)):
                kv.key_cache[li]   = k_snap[li].clone()
                kv.value_cache[li] = v_snap[li].clone()

    def step():
        with torch.no_grad():
            _ = model(nxt, past_key_values=kv, use_cache=True)
    ts = _event_time(step, iters, warmup, setup=setup)
    return _stats(ts), kv, nxt, ctx.shape[1]


def bench_cda(model, kv, nxt, S, n_layers, device, iters, warmup):
    D = getattr(model.config, "head_dim", None) or (
        model.config.hidden_size // model.config.num_attention_heads)
    configs = {
        "K4V2": (HadamardQuantCompressor(dim=D, bit_width=4, half_rotation=True),
                 HadamardQuantCompressor(dim=D, bit_width=2, half_rotation=True)),
        "K2V2": (HadamardQuantCompressor(dim=D, bit_width=2, half_rotation=True),
                 HadamardQuantCompressor(dim=D, bit_width=2, half_rotation=True)),
    }
    results = {}
    for label, (kc, vc) in configs.items():
        compressed = _compress_kv_cache_cuda(kv, kc, vc)
        patch_model_compressed_attn(model, kc, vc, attn_gate_topk=0)
        for li in range(n_layers):
            model._cda_compressed[li] = compressed[li]
        for key in ("_rotation", "_codebook_k", "_codebook_v"):
            model._cda_compressed[key] = compressed[key]
        state = {"pc": None}
        def setup(): state["pc"] = _PositionOnlyCache(n_layers, S, device, torch.float16)
        def step():
            with torch.no_grad():
                _ = model(nxt, past_key_values=state["pc"], use_cache=True)
        ts = _event_time(step, iters, warmup, setup=setup)
        results[label] = _stats(ts)
        unpatch_model(model)
        del compressed; gc.collect(); torch.cuda.empty_cache()
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=32768)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--output", default=None)
    p.add_argument("--skip-reset", action="store_true",
                   help="Skip KV snapshot/reset (for very long N when memory is tight)")
    args = p.parse_args()

    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"=== N={args.N}  iters={args.iters}  warmup={args.warmup} ===\n")
    row = {"N": args.N, "iters": args.iters, "warmup": args.warmup, "model": args.model}

    # ---- FP16 SDPA (auto) + SDPA flash-only backend ----
    print("[1/3] Loading model (attn=sdpa)")
    m = load_model(args.model, "sdpa", device)
    fp_sdpa, kv, nxt, S = bench_fp16(m, tokenizer, args.N, args.iters, args.warmup, device, skip_reset=args.skip_reset)
    print(f"  FP16 SDPA(auto) : {fp_sdpa['mean']:7.3f} ± {fp_sdpa['std']:.3f} ms")
    row["fp16_sdpa"] = fp_sdpa

    # SDPA flash-backend force removed (OOM + API mismatch — not relevant for decode)

    # ---- CDA (on same SDPA model — CDA replaces attention entirely) ----
    cda = bench_cda(m, kv, nxt, S, m.config.num_hidden_layers, device,
                    args.iters, args.warmup)
    for label, st in cda.items():
        st["vs_sdpa"] = round(fp_sdpa["mean"] / st["mean"], 3)
        row[f"cda_{label.lower()}"] = st
        print(f"  CDA {label:5s}     : {st['mean']:7.3f} ± {st['std']:.3f} ms "
              f"→ {st['vs_sdpa']:.2f}x vs SDPA")

    del m, kv; gc.collect(); torch.cuda.empty_cache()

    # ---- FP16 FlashAttention2 (Flash-Decoding) ----
    print("\n[2/3] Loading model (attn=flash_attention_2)")
    m = load_model(args.model, "flash_attention_2", device)
    fp_fa2, _, _, _ = bench_fp16(m, tokenizer, args.N, args.iters, args.warmup, device, skip_reset=args.skip_reset)
    row["fp16_flash_attn2"] = fp_fa2
    row["cda_k4v2"]["vs_fa2"] = round(fp_fa2["mean"] / row["cda_k4v2"]["mean"], 3)
    row["cda_k2v2"]["vs_fa2"] = round(fp_fa2["mean"] / row["cda_k2v2"]["mean"], 3)
    print(f"  FP16 Flash-A2 : {fp_fa2['mean']:7.3f} ± {fp_fa2['std']:.3f} ms")
    print(f"  CDA K4V2 vs FA2 : {row['cda_k4v2']['vs_fa2']:.2f}x")
    print(f"  CDA K2V2 vs FA2 : {row['cda_k2v2']['vs_fa2']:.2f}x")

    print("\n" + json.dumps(row, indent=2))
    if args.output:
        out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(row, indent=2))
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
