"""Batch serving throughput: FP16 vs CDA K4/V2.

Measures tokens/s at different batch sizes and context lengths.
FP16: batch prefill then batch decode.
CDA: sequential decode (current kernel is batch=1), throughput = B / (B * cda_1_ms).

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/bench_batch_serving.py
    CUDA_VISIBLE_DEVICES=0 python experiments/bench_batch_serving.py --ns 4096,16384 --bs 1,4,8
"""
import os, sys, time, gc, json, argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from transformers import AutoModelForCausalLM, AutoTokenizer
from core.compression import HadamardQuantCompressor
from core.compressed_generate import (
    _compress_kv_cache_cuda, _compress_kv_cache,
    patch_model_compressed_attn, unpatch_model,
    _PositionOnlyCache, _USE_CUDA_E2E,
)
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--ns", type=str, default="1024,4096,8192,16384,32768")
parser.add_argument("--bs", type=str, default="1,2,4,8,16,32")
parser.add_argument("--warmup", type=int, default=3)
parser.add_argument("--iters", type=int, default=10)
args = parser.parse_args()

N_LIST = [int(x) for x in args.ns.split(",")]
B_LIST = [int(x) for x in args.bs.split(",")]
device = "cuda:0"
D = 128

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,
    device_map=device, low_cpu_mem_usage=True).eval()
n_layers = model.config.num_hidden_layers

c4 = HadamardQuantCompressor(dim=D, bit_width=4, half_rotation=True)
c2 = HadamardQuantCompressor(dim=D, bit_width=2, half_rotation=True)

from datasets import load_dataset
text = "\n\n".join([t for t in load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip()])

results = {}

print(f"Batch Serving Throughput: FP16 vs CDA K4/V2")
print(f"Model: {model_name}, Device: {device}")
print(f"{'N':>6} {'B':>4} | {'FP16 ms':>9} {'FP16 tok/s':>11} | {'CDA ms':>9} {'CDA tok/s':>11} | {'Speedup':>8}")
print("-" * 80)

for N in N_LIST:
    enc = tokenizer(text, return_tensors="pt", max_length=N, truncation=True)
    input_ids = enc.input_ids.to(device)
    ctx = input_ids[:, :-1]
    nxt = input_ids[:, -1:]

    # --- Measure CDA B=1 latency once per N (kernel is batch=1) ---
    # Chunked prefill
    fp16_kv = None
    with torch.no_grad():
        for i in range(0, ctx.shape[1], 4096):
            chunk = ctx[:, i:i + 4096]
            out = model(chunk, past_key_values=fp16_kv, use_cache=True, return_dict=True)
            fp16_kv = out.past_key_values
            del out

    # Compress
    if _USE_CUDA_E2E:
        compressed = _compress_kv_cache_cuda(fp16_kv, c4, c2, skip_sinks=0)
    else:
        compressed = _compress_kv_cache(fp16_kv, c4, c2, skip_sinks=0)

    # CDA decode B=1 timing
    patch_model_compressed_attn(model, c4, c2, skip_sinks=0, attn_gate_topk=0)
    for li in range(n_layers):
        model._cda_compressed[li] = compressed[li]
    if _USE_CUDA_E2E and "_rotation" in compressed:
        model._cda_compressed["_rotation"] = compressed["_rotation"]
        model._cda_compressed["_cb_k"] = compressed["_cb_k"]
        model._cda_compressed["_cb_v"] = compressed["_cb_v"]

    S = ctx.shape[1]
    for _ in range(args.warmup):
        with torch.no_grad():
            pc = _PositionOnlyCache(n_layers, S, device, torch.float16)
            _ = model(nxt, past_key_values=pc, use_cache=True)
        del pc
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        with torch.no_grad():
            pc = _PositionOnlyCache(n_layers, S, device, torch.float16)
            _ = model(nxt, past_key_values=pc, use_cache=True)
        del pc
    torch.cuda.synchronize()
    cda_1_ms = (time.perf_counter() - t0) / args.iters * 1000

    unpatch_model(model)
    del compressed; gc.collect(); torch.cuda.empty_cache()

    # --- Measure FP16 for each batch size ---
    for B in B_LIST:
        key = f"N{N}_B{B}"
        entry = {"N": N, "B": B}

        try:
            # Batch prefill: replicate input_ids
            ctx_batch = ctx.expand(B, -1).contiguous()
            nxt_batch = nxt.expand(B, -1).contiguous()

            fp16_kv_batch = None
            with torch.no_grad():
                for i in range(0, ctx_batch.shape[1], 4096):
                    chunk = ctx_batch[:, i:i + 4096]
                    out = model(chunk, past_key_values=fp16_kv_batch, use_cache=True, return_dict=True)
                    fp16_kv_batch = out.past_key_values
                    del out

            # FP16 batch decode timing
            for _ in range(args.warmup):
                with torch.no_grad():
                    _ = model(nxt_batch, past_key_values=fp16_kv_batch, use_cache=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.iters):
                with torch.no_grad():
                    _ = model(nxt_batch, past_key_values=fp16_kv_batch, use_cache=True)
            torch.cuda.synchronize()
            fp16_ms = (time.perf_counter() - t0) / args.iters * 1000
            fp16_tps = B / (fp16_ms / 1000)

            del fp16_kv_batch; gc.collect(); torch.cuda.empty_cache()

            # CDA: B sequential calls of cda_1_ms
            cda_ms = cda_1_ms * B
            cda_tps = B / (cda_ms / 1000)

            entry["fp16_ms"] = round(fp16_ms, 1)
            entry["fp16_tok_s"] = round(fp16_tps, 1)
            entry["cda_1_ms"] = round(cda_1_ms, 1)
            entry["cda_batch_ms"] = round(cda_ms, 1)
            entry["cda_tok_s"] = round(cda_tps, 1)
            entry["speedup"] = round(fp16_ms / cda_ms, 2)

            print(f"{N:>6} {B:>4} | {fp16_ms:>8.1f}ms {fp16_tps:>10.1f} | {cda_ms:>8.1f}ms {cda_tps:>10.1f} | {fp16_ms/cda_ms:>7.2f}x")

        except RuntimeError as e:
            entry["error"] = "OOM"
            print(f"{N:>6} {B:>4} | OOM")
            gc.collect(); torch.cuda.empty_cache()

        results[key] = entry

    del fp16_kv; gc.collect(); torch.cuda.empty_cache()

out_path = PROJECT_ROOT / "runs" / "batch_serving.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")
