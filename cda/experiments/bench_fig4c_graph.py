"""Figure 4(c) E2E Decode (GPU evaluation of CDA) — CUDA Graphs version.

Fair kernel-to-kernel comparison: StaticCache + CUDA Graphs remove all
Python/DynamicCache overhead, leaving only the actual GPU work.

Per N, measures 4 configs back-to-back on the same GPU:
  * FP16 SDPA   (StaticCache + graph)
  * FP16 FA2    (StaticCache + graph)
  * CDA K4V2    (_PositionOnlyCache + graph)
  * CDA K2V2    (_PositionOnlyCache + graph)

This script lives in the cda research repo (core.compression +
core.cuda_attention_e2e). Helper functions (_compress_kv_cache_cuda,
patch_model_compressed_attn, _PositionOnlyCache, unpatch_model) are
imported from experiments/bench_cda_integrated_single.py.
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from pathlib import Path

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DynamicCache, StaticCache)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.compression import HadamardQuantCompressor  # noqa: E402
from core.cda_attn import (  # noqa: E402  (ported from sub0 — uses _cda_gqa_kernels.so)
    _compress_kv_cache_cuda,
    patch_model_compressed_attn,
    unpatch_model,
)


class _StaticPositionCache(DynamicCache):
    """No-op cache: reports fixed seq_length, never resizes.
    CDA's patched attention reads from model._cda_compressed, so actual K/V
    content is irrelevant. Fixed buffers → graph-capturable.
    """
    def __init__(self, num_layers, seq_len, device, dtype):
        super().__init__()
        self._seq_len = seq_len
        for _ in range(num_layers):
            dummy = torch.zeros(1, 1, seq_len, 1, device=device, dtype=dtype)
            self.key_cache.append(dummy)
            self.value_cache.append(dummy)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        return self._seq_len


def _stats(ts):
    return {"mean": round(statistics.mean(ts), 3),
            "std":  round(statistics.stdev(ts) if len(ts) > 1 else 0.0, 3),
            "min":  round(min(ts), 3),
            "median": round(statistics.median(ts), 3)}


def capture_and_time(fn, iters, warmup):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record(); g.replay(); ends[i].record()
    torch.cuda.synchronize()
    times = [starts[i].elapsed_time(ends[i]) for i in range(iters)]
    # Explicit cleanup — otherwise the graph's private memory pool accumulates
    # across consecutive captures and later calls hit cublas/kernel faults.
    del g, starts, ends, s
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return times


def _fp16_fn(model, tokenizer, N, device):
    """Prefill with DynamicCache (memory-safe for SDPA at large N), then
    transfer KV to a StaticCache so the decode step can be captured."""
    prompt = "The quick brown fox jumps over the lazy dog. " * 32768
    enc = tokenizer(prompt, return_tensors="pt", max_length=N, truncation=True).to(device)
    ctx = enc.input_ids[:, :-1]; nxt = enc.input_ids[:, -1:]
    S = ctx.shape[1]
    max_len = S + 8

    with torch.no_grad():
        dyn_kv = model(ctx, use_cache=True, return_dict=True,
                       num_logits_to_keep=1).past_key_values

    # Transit K/V through CPU to avoid holding dyn_kv and StaticCache on GPU
    # simultaneously (would OOM at 128K).
    cpu_k = [k.to("cpu") for k in dyn_kv.key_cache]
    cpu_v = [v.to("cpu") for v in dyn_kv.value_cache]
    del dyn_kv
    gc.collect(); torch.cuda.empty_cache()

    cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=max_len,
                        device=device, dtype=torch.float16)
    for li in range(len(cpu_k)):
        cache.key_cache[li][:, :, :S, :]   = cpu_k[li].to(device)
        cache.value_cache[li][:, :, :S, :] = cpu_v[li].to(device)
    del cpu_k, cpu_v
    gc.collect(); torch.cuda.empty_cache()

    static_nxt = nxt.clone()
    static_pos = torch.tensor([S], device=device)
    def fwd():
        with torch.no_grad():
            _ = model(static_nxt, past_key_values=cache, use_cache=True,
                      cache_position=static_pos)
    return fwd, cache, S


def bench_N(N, model_name, device, iters, warmup):
    print(f"\n{'='*60}\n  N = {N}   iters={iters}  warmup={warmup}\n{'='*60}")
    row = {"N": N, "iters": iters, "warmup": warmup}
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ---- FP16 SDPA ----
    print("[1/4] FP16 SDPA (graph)")
    m_sdpa = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        attn_implementation="sdpa", device_map=device, low_cpu_mem_usage=True).eval()
    fwd, cache, S = _fp16_fn(m_sdpa, tokenizer, N, device)
    row["fp16_sdpa"] = _stats(capture_and_time(fwd, iters, warmup))
    print(f"  FP16 SDPA  : {row['fp16_sdpa']['mean']:7.3f} ± {row['fp16_sdpa']['std']:.3f} ms")
    del m_sdpa, cache, fwd; gc.collect(); torch.cuda.empty_cache()

    # ---- FP16 FA2 ----
    print("[2/4] FP16 FlashAttention2 (graph)")
    m_fa2 = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        attn_implementation="flash_attention_2", device_map=device, low_cpu_mem_usage=True).eval()
    fwd, cache, S = _fp16_fn(m_fa2, tokenizer, N, device)
    row["fp16_fa2"] = _stats(capture_and_time(fwd, iters, warmup))
    print(f"  FP16 FA2   : {row['fp16_fa2']['mean']:7.3f} ± {row['fp16_fa2']['std']:.3f} ms")
    del m_fa2, cache, fwd; gc.collect(); torch.cuda.empty_cache()

    # ---- CDA (K4V2, K2V2) ----
    print("[3-4/4] CDA (graph)")
    m_cda = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        attn_implementation="sdpa", device_map=device, low_cpu_mem_usage=True).eval()
    n_layers = m_cda.config.num_hidden_layers
    D = getattr(m_cda.config, "head_dim", None) or (
        m_cda.config.hidden_size // m_cda.config.num_attention_heads)

    prompt = "The quick brown fox jumps over the lazy dog. " * 32768
    enc = tokenizer(prompt, return_tensors="pt", max_length=N, truncation=True).to(device)
    ctx = enc.input_ids[:, :-1]; nxt = enc.input_ids[:, -1:]
    S = ctx.shape[1]
    with torch.no_grad():
        kv = m_cda(ctx, use_cache=True, return_dict=True, num_logits_to_keep=1).past_key_values

    configs = {
        "K4V2": (HadamardQuantCompressor(dim=D, bit_width=4, half_rotation=True),
                 HadamardQuantCompressor(dim=D, bit_width=2, half_rotation=True)),
        "K2V2": (HadamardQuantCompressor(dim=D, bit_width=2, half_rotation=True),
                 HadamardQuantCompressor(dim=D, bit_width=2, half_rotation=True)),
        "K4V4": (HadamardQuantCompressor(dim=D, bit_width=4, half_rotation=True),
                 HadamardQuantCompressor(dim=D, bit_width=4, half_rotation=True)),
    }
    for label, (kc, vc) in configs.items():
        compressed = _compress_kv_cache_cuda(kv, kc, vc)
        patch_model_compressed_attn(m_cda, kc, vc, attn_gate_topk=0)
        for li in range(n_layers):
            m_cda._cda_compressed[li] = compressed[li]
        for key in ("_rotation", "_codebook_k", "_codebook_v"):
            m_cda._cda_compressed[key] = compressed[key]

        pc = _StaticPositionCache(n_layers, S, device, torch.float16)
        static_nxt = nxt.clone()
        def fwd():
            with torch.no_grad():
                _ = m_cda(static_nxt, past_key_values=pc, use_cache=True)
        row[f"cda_{label.lower()}"] = _stats(capture_and_time(fwd, iters, warmup))
        st = row[f"cda_{label.lower()}"]
        print(f"  CDA {label}   : {st['mean']:7.3f} ± {st['std']:.3f} ms")
        unpatch_model(m_cda)
        del compressed, pc; gc.collect(); torch.cuda.empty_cache()

    del m_cda, kv; gc.collect(); torch.cuda.empty_cache()

    # Speedups
    for label in ("k4v2", "k2v2", "k4v4"):
        st = row[f"cda_{label}"]
        st["vs_sdpa"] = round(row["fp16_sdpa"]["mean"] / st["mean"], 3)
        st["vs_fa2"]  = round(row["fp16_fa2"]["mean"]  / st["mean"], 3)

    print(f"\n  CDA K4V2 vs SDPA : {row['cda_k4v2']['vs_sdpa']:.2f}x  | vs FA2 : {row['cda_k4v2']['vs_fa2']:.2f}x")
    print(f"  CDA K2V2 vs SDPA : {row['cda_k2v2']['vs_sdpa']:.2f}x  | vs FA2 : {row['cda_k2v2']['vs_fa2']:.2f}x")
    print(f"  CDA K4V4 vs SDPA : {row['cda_k4v4']['vs_sdpa']:.2f}x  | vs FA2 : {row['cda_k4v4']['vs_fa2']:.2f}x")
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--Ns", default="32768,65536,131072")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    device = torch.device("cuda:0")
    Ns = [int(n) for n in args.Ns.split(",")]
    all_rows = []
    for N in Ns:
        iters = args.iters if N <= 65536 else max(10, args.iters // 2)
        try:
            all_rows.append(bench_N(N, args.model, device, iters, args.warmup))
        except Exception as e:
            print(f"  !! Error at N={N}: {e}")
            all_rows.append({"N": N, "error": str(e)})
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(all_rows, indent=2))

    print("\n\n=== SUMMARY (CUDA Graph mode) ===")
    hdr = f"{'N':>8}  {'SDPA':>8}  {'FA2':>8}  {'K4V2':>8}  {'K2V2':>8}  {'K4V2/FA2':>9}  {'K2V2/FA2':>9}"
    print(hdr); print("-" * len(hdr))
    for r in all_rows:
        if "error" in r: continue
        print(f"{r['N']:>8}  {r['fp16_sdpa']['mean']:>8.2f}  {r['fp16_fa2']['mean']:>8.2f}  "
              f"{r['cda_k4v2']['mean']:>8.2f}  {r['cda_k2v2']['mean']:>8.2f}  "
              f"{r['cda_k4v2']['vs_fa2']:>8.2f}x  {r['cda_k2v2']['vs_fa2']:>8.2f}x")


if __name__ == "__main__":
    main()
