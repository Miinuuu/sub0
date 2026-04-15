"""Decode latency + peak KV memory for FP16 vs CDA compressed KV.

Compares three decode paths on a single LLM prefill:
  1. FP16 baseline (HuggingFace default KV cache).
  2. CDA-SW: compress KV with Hadamard rotation + Lloyd-Max, decompress each
     step for standard fp16 attention. Demonstrates memory savings without
     touching the attention math.
  3. CDA-HW (CUDA): compressed-domain attention via the fused CUDA kernels
     from ``cda.cuda_attention``. Runs only when the extension is built.

Usage::

    CUDA_VISIBLE_DEVICES=0 python experiments/benchmark_speed.py \\
        --model meta-llama/Llama-3.2-1B-Instruct --ctx 1024,4096,8192 \\
        --bits 2 --output runs/cda_speed.json
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache  # noqa: E402

from cda import HadamardQuantCompressor  # noqa: E402

try:
    from cda.cuda_attention import cuda_hw_attention_batched
    _HAS_CUDA_KERNEL = True
except ImportError:
    _HAS_CUDA_KERNEL = False


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def _timed_decode(model, nxt, past, iters: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            _ = model(nxt, past_key_values=past, use_cache=True)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000  # ms


def _peak_memory_mib(device: torch.device) -> float:
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


# ---------------------------------------------------------------------------
# CDA-SW: decompress per step, fp16 attention
# ---------------------------------------------------------------------------

def _build_compressed_cache(fp16_kv, comp: HadamardQuantCompressor, head_dim: int):
    compressed = []
    for li in range(len(fp16_kv.key_cache)):
        k = fp16_kv.key_cache[li]
        v = fp16_kv.value_cache[li]
        ck = comp.quantize(k.reshape(-1, head_dim))
        cv = comp.quantize(v.reshape(-1, head_dim))
        compressed.append((ck, cv, k.shape, v.shape))
    return compressed


def _run_cda_sw(model, nxt, compressed, comp: HadamardQuantCompressor, iters: int) -> float:
    def build_cache():
        kv = DynamicCache()
        for li, (ck, cv, ks, vs) in enumerate(compressed):
            kv.update(comp.dequantize(ck).reshape(ks),
                      comp.dequantize(cv).reshape(vs), li)
        return kv

    for _ in range(2):  # warmup
        with torch.no_grad():
            _ = model(nxt, past_key_values=build_cache(), use_cache=True)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            _ = model(nxt, past_key_values=build_cache(), use_cache=True)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


# ---------------------------------------------------------------------------
# CDA-HW: fused CUDA compressed-domain attention (single layer, illustrative)
# ---------------------------------------------------------------------------

def _run_cda_hw_microbench(comp, N: int, head_dim: int, heads: int,
                           device: torch.device, iters: int) -> float:
    """Time the fused CUDA kernel on a synthetic single-layer attention step.

    This isolates the kernel cost at fixed N: no prefill, no model forward.
    """
    if not _HAS_CUDA_KERNEL:
        return float("nan")

    B = heads
    D = head_dim
    Q = torch.randn(B, D, device=device, dtype=torch.float32)
    K = torch.randn(B * N, D, device=device, dtype=torch.float32)
    V = torch.randn(B * N, D, device=device, dtype=torch.float32)

    cK = comp.quantize(K.reshape(-1, D))
    cV = comp.quantize(V.reshape(-1, D))
    comp._ensure_tensors(device)
    codebook = (comp._centroids * 2.0 - 1.0).float()
    rotation = comp._rotation.float()
    Q_rot = Q @ comp._rotation_t.float()
    scale = 1.0 / math.sqrt(D)

    kwargs = dict(
        Q_rot_all=Q_rot,
        packed_indices_K=cK.indices.reshape(-1, cK.indices.shape[-1]),
        norms_K=cK.norms.float().reshape(-1),
        packed_indices_V=cV.indices.reshape(-1, cV.indices.shape[-1]),
        norms_V=cV.norms.float().reshape(-1),
        codebook=codebook,
        rotation=rotation,
        scale=scale,
        N=N,
    )

    for _ in range(5):
        cuda_hw_attention_batched(**kwargs)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        cuda_hw_attention_batched(**kwargs)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--ctx", default="1024,4096,8192",
                        help="comma-separated context lengths")
    parser.add_argument("--bits", type=int, default=2, choices=[2, 4])
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--output", default="runs/cda_speed.json")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    ctx_lens = [int(x) for x in args.ctx.split(",") if x.strip()]

    print(f"Loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=device,
        low_cpu_mem_usage=True,
    ).eval()
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    heads = model.config.num_key_value_heads or model.config.num_attention_heads
    print(f"  head_dim={head_dim}, kv_heads={heads}, layers={model.config.num_hidden_layers}")

    comp = HadamardQuantCompressor(dim=head_dim, bit_width=args.bits, half_rotation=True)

    # Text source for prefill (padded to longest ctx).
    prompt = ("The quick brown fox jumps over the lazy dog. " * 4096)
    enc = tokenizer(prompt, return_tensors="pt", max_length=max(ctx_lens),
                    truncation=True).to(device)

    results = {}
    for N in ctx_lens:
        print(f"\n=== N={N} ===")
        input_ids = enc.input_ids[:, :N]
        ctx_ids = input_ids[:, :-1]
        nxt = input_ids[:, -1:]

        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            fp16_kv = model(ctx_ids, use_cache=True, return_dict=True).past_key_values

        # FP16 baseline.
        for _ in range(2):
            _ = model(nxt, past_key_values=fp16_kv, use_cache=True)
        torch.cuda.synchronize()
        fp16_ms = _timed_decode(model, nxt, fp16_kv, args.iters)
        fp16_mib = _peak_memory_mib(device)
        print(f"  FP16 decode : {fp16_ms:6.2f} ms  peak={fp16_mib:.1f} MiB")

        # CDA-SW.
        torch.cuda.reset_peak_memory_stats(device)
        compressed = _build_compressed_cache(fp16_kv, comp, head_dim)
        sw_ms = _run_cda_sw(model, nxt, compressed, comp, args.iters)
        sw_mib = _peak_memory_mib(device)
        print(f"  CDA-SW {args.bits}b: {sw_ms:6.2f} ms  peak={sw_mib:.1f} MiB")

        # CDA-HW kernel microbench.
        torch.cuda.reset_peak_memory_stats(device)
        hw_ms = _run_cda_hw_microbench(comp, N=N, head_dim=head_dim,
                                       heads=heads, device=device,
                                       iters=args.iters)
        hw_mib = _peak_memory_mib(device) if _HAS_CUDA_KERNEL else None
        if _HAS_CUDA_KERNEL:
            print(f"  CDA-HW kern : {hw_ms:6.2f} ms  peak={hw_mib:.1f} MiB  (single-layer microbench)")
        else:
            print("  CDA-HW kern : [skipped — cda._cda_kernels not built]")

        results[str(N)] = {
            "N": N,
            "fp16_ms": round(fp16_ms, 2),
            "fp16_peak_mib": round(fp16_mib, 1),
            "cda_sw_ms": round(sw_ms, 2),
            "cda_sw_peak_mib": round(sw_mib, 1),
            "cda_hw_kernel_ms": None if not _HAS_CUDA_KERNEL else round(hw_ms, 2),
            "cda_hw_peak_mib": None if not _HAS_CUDA_KERNEL else round(hw_mib, 1),
        }

        del fp16_kv, compressed
        gc.collect()
        torch.cuda.empty_cache()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model": args.model,
        "bits": args.bits,
        "gpu": torch.cuda.get_device_name(device),
        "has_cuda_kernel": _HAS_CUDA_KERNEL,
        "results": results,
    }, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
