"""CUDA Graphs bench: removes Python launch overhead.

Captures the full HF decode forward pass once, then replays via
cuda.CUDAGraph.replay() — no Python executes during replay, just
the pre-recorded kernel launches on the GPU's internal scheduler.

Strategy:
  * CDA : no-op _StaticPositionCache (patched attention doesn't read cache)
  * FP16: HF's built-in StaticCache (pre-allocates full KV buffer).

Only compares CDA K2V2 vs FP16-FA2 at a single N (paper-style).
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, StaticCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.compression import HadamardQuantCompressor  # noqa: E402
from core.cda_attn import (  # noqa: E402
    _compress_kv_cache_cuda,
    patch_model_compressed_attn,
    unpatch_model,
)


class _StaticPositionCache(DynamicCache):
    """No-op cache: reports fixed seq_length, never resizes.

    CDA's patched attention reads from model._cda_compressed, not this cache,
    so the actual K/V content is irrelevant. Fixed buffers → graph-capturable.
    """
    def __init__(self, num_layers, seq_len, device, dtype):
        super().__init__()
        self._seq_len = seq_len
        for li in range(num_layers):
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


def capture_and_time(forward_fn, iters, warmup):
    """Warm up, capture one invocation into a CUDA graph, replay + time."""
    # Warmup (stream capture prereq)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            forward_fn()
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        forward_fn()

    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        g.replay()
        ends[i].record()
    torch.cuda.synchronize()
    return [starts[i].elapsed_time(ends[i]) for i in range(iters)]


def eager_time(forward_fn, iters, warmup):
    for _ in range(warmup):
        forward_fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record(); forward_fn(); ends[i].record()
    torch.cuda.synchronize()
    return [starts[i].elapsed_time(ends[i]) for i in range(iters)]


def run_cda(N, model_name, device, iters, warmup):
    print(f"\n=== CDA K2V2  N={N}  (FP16→CDA, single GPU) ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        attn_implementation="sdpa", device_map=device, low_cpu_mem_usage=True,
    ).eval()
    n_layers = model.config.num_hidden_layers
    D = getattr(model.config, "head_dim", None) or (
        model.config.hidden_size // model.config.num_attention_heads)

    prompt = "The quick brown fox jumps over the lazy dog. " * 32768
    enc = tokenizer(prompt, return_tensors="pt", max_length=N, truncation=True).to(device)
    ctx = enc.input_ids[:, :-1]; nxt = enc.input_ids[:, -1:]
    S = ctx.shape[1]

    with torch.no_grad():
        kv = model(ctx, use_cache=True, return_dict=True, num_logits_to_keep=1).past_key_values

    kc = HadamardQuantCompressor(dim=D, bit_width=2, half_rotation=True)
    vc = HadamardQuantCompressor(dim=D, bit_width=2, half_rotation=True)
    compressed = _compress_kv_cache_cuda(kv, kc, vc)
    patch_model_compressed_attn(model, kc, vc, attn_gate_topk=0)
    for li in range(n_layers):
        model._cda_compressed[li] = compressed[li]
    for key in ("_rotation", "_codebook_k", "_codebook_v"):
        model._cda_compressed[key] = compressed[key]

    del kv, ctx; gc.collect(); torch.cuda.empty_cache()

    pc = _StaticPositionCache(n_layers, S, device, torch.float16)
    static_nxt = nxt.clone()

    def fwd():
        with torch.no_grad():
            _ = model(static_nxt, past_key_values=pc, use_cache=True)

    eager = _stats(eager_time(fwd, iters, warmup))
    print(f"  Eager          : {eager['mean']:7.3f} ± {eager['std']:.3f} ms")

    try:
        graph = _stats(capture_and_time(fwd, iters, warmup))
        print(f"  CUDA Graph     : {graph['mean']:7.3f} ± {graph['std']:.3f} ms  "
              f"(Python overhead removed: -{eager['mean'] - graph['mean']:.2f} ms)")
    except Exception as e:
        print(f"  CUDA Graph failed: {type(e).__name__}: {e}")
        graph = {"error": str(e)}

    unpatch_model(model)
    del model, pc, compressed; gc.collect(); torch.cuda.empty_cache()
    return {"eager": eager, "graph": graph}


def run_fa2(N, model_name, device, iters, warmup):
    print(f"\n=== FP16 FlashAttention2  N={N} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        attn_implementation="flash_attention_2", device_map=device, low_cpu_mem_usage=True,
    ).eval()

    prompt = "The quick brown fox jumps over the lazy dog. " * 32768
    enc = tokenizer(prompt, return_tensors="pt", max_length=N, truncation=True).to(device)
    ctx = enc.input_ids[:, :-1]; nxt = enc.input_ids[:, -1:]
    S = ctx.shape[1]
    max_len = S + 8  # small margin for the decode step

    # StaticCache pre-allocates the full (B, H, max_len, D) buffer — graph-friendly.
    cache = StaticCache(config=model.config, batch_size=1, max_cache_len=max_len,
                        device=device, dtype=torch.float16)
    # Prefill populates cache[:S] in-place
    with torch.no_grad():
        _ = model(ctx, past_key_values=cache, use_cache=True,
                  cache_position=torch.arange(S, device=device), num_logits_to_keep=1)

    static_nxt = nxt.clone()
    static_pos = torch.tensor([S], device=device)

    def fwd():
        with torch.no_grad():
            _ = model(static_nxt, past_key_values=cache, use_cache=True,
                      cache_position=static_pos)

    eager = _stats(eager_time(fwd, iters, warmup))
    print(f"  Eager          : {eager['mean']:7.3f} ± {eager['std']:.3f} ms")

    try:
        graph = _stats(capture_and_time(fwd, iters, warmup))
        print(f"  CUDA Graph     : {graph['mean']:7.3f} ± {graph['std']:.3f} ms  "
              f"(Python overhead removed: -{eager['mean'] - graph['mean']:.2f} ms)")
    except Exception as e:
        print(f"  CUDA Graph failed: {type(e).__name__}: {e}")
        graph = {"error": str(e)}

    del model, cache; gc.collect(); torch.cuda.empty_cache()
    return {"eager": eager, "graph": graph}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=32768)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    device = torch.device("cuda:0")
    out = {"N": args.N, "iters": args.iters, "warmup": args.warmup}

    out["cda_k2v2"] = run_cda(args.N, args.model, device, args.iters, args.warmup)
    out["fp16_fa2"] = run_fa2(args.N, args.model, device, args.iters, args.warmup)

    if out["cda_k2v2"].get("graph", {}).get("mean") and out["fp16_fa2"].get("graph", {}).get("mean"):
        sp = out["fp16_fa2"]["graph"]["mean"] / out["cda_k2v2"]["graph"]["mean"]
        print(f"\n  [Graph mode] CDA K2V2 vs FP16-FA2 : {sp:.2f}x")
        out["speedup_graph"] = round(sp, 3)

    print("\n" + json.dumps(out, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
