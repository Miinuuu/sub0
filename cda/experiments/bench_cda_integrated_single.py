"""CDA fused-path E2E decode benchmark (single N).

Uses sub0's GQA-aware packaged CUDA kernels (the ``core._cda_gqa_kernels``
binary extension that ships with this package). GQA indexing happens
inside CUDA — KV is read as ``kv_head = q_head // group_size`` — so
there is no per-step ``repeat_interleave`` on the Python side. This
reproduces the paper's Figure 5(c) timings on a single A6000.

Paper reference (Figure 5(c), Llama-3.1-8B, A6000):

    | N=32768 | FP16 | CDA K4/V2 | vs FP16 |
    | paper   | 110.8 ms | 35.1 ms | 3.16×   |

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/bench_cda_integrated_single.py --N 32768
    CUDA_VISIBLE_DEVICES=0 python experiments/bench_cda_integrated_single.py --N 65536 \
        --output runs/cda_fused_N65536.json
"""
from __future__ import annotations

import argparse
import gc
import json
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


def _measure_fp16(model, nxt, fp16_kv, iters, warmup):
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(nxt, past_key_values=fp16_kv, use_cache=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            _ = model(nxt, past_key_values=fp16_kv, use_cache=True)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def _measure_cda(model, nxt, n_layers, S, device, iters, warmup):
    for _ in range(warmup):
        with torch.no_grad():
            pc = _PositionOnlyCache(n_layers, S, device, torch.float16)
            _ = model(nxt, past_key_values=pc, use_cache=True)
        del pc
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            pc = _PositionOnlyCache(n_layers, S, device, torch.float16)
            _ = model(nxt, past_key_values=pc, use_cache=True)
        del pc
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=True, help="context length")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output", default=None)
    parser.add_argument("--configs", default="K4V2,K2V2",
                        help="comma-separated subset of {K4V2,K2V2}")
    args = parser.parse_args()

    N = args.N
    if N >= 32768:
        warmup, iters = 2, 5
    elif N >= 16384:
        warmup, iters = 3, 10
    else:
        warmup, iters = 5, 20

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
    input_ids = enc.input_ids
    ctx = input_ids[:, :-1]
    nxt = input_ids[:, -1:]
    S = ctx.shape[1]

    # --- FP16 baseline -----------------------------------------------------
    with torch.no_grad():
        fp16_kv = model(ctx, use_cache=True, return_dict=True).past_key_values
    fp16_ms = _measure_fp16(model, nxt, fp16_kv, iters, warmup)
    row = {"N": N, "model": args.model, "fp16_ms": round(fp16_ms, 2)}
    print(f"  FP16        : {fp16_ms:7.2f} ms")

    # --- CDA configs -------------------------------------------------------
    compressors = {
        "K4V2": (HadamardQuantCompressor(dim=D, bit_width=4, half_rotation=True),
                 HadamardQuantCompressor(dim=D, bit_width=2, half_rotation=True)),
        "K2V2": (HadamardQuantCompressor(dim=D, bit_width=2, half_rotation=True),
                 HadamardQuantCompressor(dim=D, bit_width=2, half_rotation=True)),
    }
    wanted = [c.strip() for c in args.configs.split(",") if c.strip()]
    for label in wanted:
        if label not in compressors:
            print(f"  Skipping unknown config: {label}")
            continue
        kc, vc = compressors[label]
        compressed = _compress_kv_cache_cuda(fp16_kv, kc, vc)
        patch_model_compressed_attn(model, kc, vc, attn_gate_topk=0)
        for li in range(n_layers):
            model._cda_compressed[li] = compressed[li]
        for key in ("_rotation", "_codebook_k", "_codebook_v"):
            model._cda_compressed[key] = compressed[key]

        cda_ms = _measure_cda(model, nxt, n_layers, S, device, iters, warmup)
        row[f"{label.lower()}_ms"] = round(cda_ms, 2)
        row[f"{label.lower()}_vs_fp16"] = round(fp16_ms / cda_ms, 2)
        print(f"  CDA {label:5s}   : {cda_ms:7.2f} ms   ({fp16_ms / cda_ms:.2f}x vs FP16)")

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
