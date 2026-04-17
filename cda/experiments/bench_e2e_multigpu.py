"""Unified E2E decode benchmark (Multi-GPU): all methods inside model.forward().

Every method uses model(nxt, past_key_values=cache) for fair comparison.
- FP16: standard forward
- KIVI: QuantoQuantizedCache (internal dequantize)
- TQ: monkey-patched attention (decompress inside forward)
- CDA: monkey-patched attention (CUDA E2E kernel inside forward)

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python experiments/bench_e2e_multigpu.py --N 8192
"""
import os, sys, gc, time, json, argparse
import torch
sys.path.insert(0, os.path.expanduser("~/ing/research/cda"))

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from core.compression import HadamardQuantCompressor, TurboQuantCompressor
from core.compressed_generate import (
    _compress_kv_cache_cuda, _compress_kv_cache,
    patch_model_compressed_attn, patch_model_decompress_attn,
    unpatch_model, _PositionOnlyCache, _USE_CUDA_E2E,
)

parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, required=True)
parser.add_argument("--model", type=str, default="llama8b", help="Model alias to use")
args = parser.parse_args()

MODELS = {
    "llama2-7b":  "meta-llama/Llama-2-7B-hf",
    "mistral7b":  "mistralai/Mistral-7B-v0.3",
    "llama8b":    "meta-llama/Llama-3.1-8B-Instruct",
    "llama2-13b": "meta-llama/Llama-2-13B-hf",
    "qwen32b":    "Qwen/Qwen2.5-32B",
    "llama70b":   "meta-llama/Llama-3.1-70B-Instruct",
}

N = args.N
iters = 5 if N >= 32768 else (10 if N >= 16384 else 20)
warmup = 2 if N >= 32768 else (3 if N >= 16384 else 5)

# Use HuggingFace Pipeline Parallelism
device = "auto"
model_name = MODELS.get(args.model, args.model)
D = 128

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,
    device_map=device, low_cpu_mem_usage=True).eval()

n_layers = model.config.num_hidden_layers

# For Multi-GPU, extract mapping so caches are placed on the correct GPUs
if hasattr(model, "hf_device_map"):
    layer_devices = {li: model.hf_device_map.get(f"model.layers.{li}", "cuda:0") for li in range(n_layers)}
    embed_device = model.hf_device_map.get("model.embed_tokens", "cuda:0")
else:
    layer_devices = {li: "cuda:0" for li in range(n_layers)}
    embed_device = "cuda:0"

from datasets import load_dataset
text = "\n\n".join([t for t in load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip()])

enc = tokenizer(text, return_tensors="pt", max_length=N, truncation=True)
input_ids = enc.input_ids.to(embed_device)
nxt = input_ids[:, -1:]
ctx = input_ids[:, :-1]

c4 = HadamardQuantCompressor(dim=D, bit_width=4, half_rotation=True)
c2 = HadamardQuantCompressor(dim=D, bit_width=2, half_rotation=True)
tq2 = TurboQuantCompressor(dim=D, bit_width=2, half_rotation=True)

row = {"N": N}


def measure(setup_fn, name):
    """Measure decode latency with setup_fn providing the cache."""
    cache = setup_fn()
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(nxt, past_key_values=cache, use_cache=True)
        if hasattr(cache, '_reset'):
            cache = setup_fn()
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            _ = model(nxt, past_key_values=cache, use_cache=True)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000
    return ms


# === 1. FP16 ===
with torch.no_grad():
    fp16_kv = model(ctx, use_cache=True, return_dict=True).past_key_values
    if isinstance(fp16_kv, tuple):
        tmp_cache = DynamicCache()
        for l_idx, layer_kv in enumerate(fp16_kv):
            tmp_cache.update(layer_kv[0], layer_kv[1], l_idx)
        fp16_kv = tmp_cache
fp16_ms = measure(lambda: fp16_kv, "FP16")
row["fp16_ms"] = round(fp16_ms, 1)

# === 2. KIVI ===
try:
    from transformers import QuantizedCacheConfig
    from transformers.cache_utils import QuantoQuantizedCache
    with torch.no_grad():
        kivi_cache = QuantoQuantizedCache(QuantizedCacheConfig(
            nbits=2, backend="quanto", axis_key=0, axis_value=0))
        model(ctx, use_cache=True, return_dict=True, past_key_values=kivi_cache)
    kivi_ms = measure(lambda: kivi_cache, "KIVI")
    row["kivi_ms"] = round(kivi_ms, 1)
    del kivi_cache; gc.collect(); torch.cuda.empty_cache()
except Exception as e:
    row["kivi_ms"] = f"FAIL:{e}"

# === 3. TQ 2-bit (monkey-patched decompress inside forward) ===
compressed_tq = {}
for li in range(n_layers):
    if hasattr(fp16_kv, "layers"):
        k = fp16_kv.layers[li].keys
        v = fp16_kv.layers[li].values
    else:
        k = fp16_kv.key_cache[li]
        v = fp16_kv.value_cache[li]
    cK = tq2.quantize(k.reshape(-1, D))
    cV = tq2.quantize(v.reshape(-1, D))
    compressed_tq[li] = (cK, cV, k.shape, v.shape)

patch_model_decompress_attn(model, tq2, tq2)
for li in range(n_layers):
    model._dc_compressed[li] = compressed_tq[li]
S = ctx.shape[1]

def tq_cache():
    return _PositionOnlyCache(n_layers, S, None, torch.float16, layer_devices=layer_devices)

tq_ms = measure(tq_cache, "TQ")
row["tq_ms"] = round(tq_ms, 1)
unpatch_model(model)
del compressed_tq; gc.collect(); torch.cuda.empty_cache()

# === 4. CDA K4/V2 (CUDA E2E kernel inside forward) ===
# Free fp16 KV to avoid memory fragmentation
with torch.no_grad():
    fp16_kv2 = model(ctx, use_cache=True, return_dict=True).past_key_values
    if isinstance(fp16_kv2, tuple):
        tmp_cache2 = DynamicCache()
        for l_idx, layer_kv in enumerate(fp16_kv2):
            tmp_cache2.update(layer_kv[0], layer_kv[1], l_idx)
        fp16_kv2 = tmp_cache2
if _USE_CUDA_E2E:
    compressed_cda = _compress_kv_cache_cuda(fp16_kv2, c4, c2, skip_sinks=0)
else:
    compressed_cda = _compress_kv_cache(fp16_kv2, c4, c2, skip_sinks=0)
del fp16_kv, fp16_kv2; gc.collect(); torch.cuda.empty_cache()

patch_model_compressed_attn(model, c4, c2, skip_sinks=0, attn_gate_topk=0)
for li in range(n_layers):
    model._cda_compressed[li] = compressed_cda[li]
if _USE_CUDA_E2E and "_rotation" in compressed_cda:
    model._cda_compressed["_rotation"] = compressed_cda["_rotation"]
    model._cda_compressed["_cb_k"] = compressed_cda["_cb_k"]
    model._cda_compressed["_cb_v"] = compressed_cda["_cb_v"]

def cda_cache():
    return _PositionOnlyCache(n_layers, S, None, torch.float16, layer_devices=layer_devices)

cda_ms = measure(cda_cache, "CDA")
row["cda_k4v2_ms"] = round(cda_ms, 1)
unpatch_model(model)

# Speedups
for k in ("kivi_ms", "tq_ms", "cda_k4v2_ms"):
    if isinstance(row.get(k), (int, float)):
        row[k.replace("_ms", "_vs")] = round(row["fp16_ms"] / row[k], 2)

print(json.dumps(row))
