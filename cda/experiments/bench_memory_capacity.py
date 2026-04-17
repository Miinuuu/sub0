"""Measure actual max batch size on GPU before OOM.

Allocates real KV caches (FP16 and CDA K4/V2) and finds max batch.
Uses binary search for efficiency.

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/bench_memory_capacity.py
"""
import os, sys, gc, json, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
device = "cuda:0"

# Llama-3.1-8B specs
n_layers = 32
n_kv_heads = 8
head_dim = 128
model_weight_gb = 16.0  # approximate FP16 weight size

# A6000 48GB
total_gpu_gb = 48.0
available_gb = total_gpu_gb - model_weight_gb  # ~32GB for KV

def try_alloc_fp16_kv(N, B):
    """Try to allocate FP16 KV cache for B requests of N tokens."""
    try:
        tensors = []
        for _ in range(n_layers):
            k = torch.zeros(B, n_kv_heads, N, head_dim, dtype=torch.float16, device=device)
            v = torch.zeros(B, n_kv_heads, N, head_dim, dtype=torch.float16, device=device)
            tensors.extend([k, v])
        del tensors
        torch.cuda.empty_cache()
        return True
    except RuntimeError:
        torch.cuda.empty_cache()
        return False

def try_alloc_cda_kv(N, B):
    """Try to allocate CDA K4/V2 compressed KV for B requests of N tokens."""
    try:
        tensors = []
        for _ in range(n_layers):
            # K: 4-bit packed → N * head_dim/2 bytes per head
            pk = torch.zeros(B * n_kv_heads, N, head_dim // 2, dtype=torch.uint8, device=device)
            nk = torch.zeros(B * n_kv_heads, N, dtype=torch.float16, device=device)
            # V: 2-bit packed → N * head_dim/4 bytes per head
            pv = torch.zeros(B * n_kv_heads, N, head_dim // 4, dtype=torch.uint8, device=device)
            nv = torch.zeros(B * n_kv_heads, N, dtype=torch.float16, device=device)
            tensors.extend([pk, nk, pv, nv])
        del tensors
        torch.cuda.empty_cache()
        return True
    except RuntimeError:
        torch.cuda.empty_cache()
        return False

def find_max_batch(alloc_fn, N, upper=2000):
    """Binary search for max batch size."""
    lo, hi = 1, upper
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        gc.collect()
        torch.cuda.empty_cache()
        if alloc_fn(N, mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best

# Pre-allocate dummy model weights to simulate real memory pressure
print(f"Simulating model weights ({model_weight_gb}GB)...")
weight_tensors = []
weight_bytes = int(model_weight_gb * 1024**3)
chunk = 256 * 1024 * 1024  # 256MB chunks
for _ in range(weight_bytes // chunk):
    weight_tensors.append(torch.zeros(chunk // 2, dtype=torch.float16, device=device))
print(f"Allocated {len(weight_tensors)} chunks = {len(weight_tensors)*256}MB")
torch.cuda.synchronize()

results = {}
for N in [1024, 4096, 8192, 16384, 32768, 65536, 131072]:
    print(f"\n=== N={N} ===")

    fp16_max = find_max_batch(try_alloc_fp16_kv, N, upper=500)
    cda_max = find_max_batch(try_alloc_cda_kv, N, upper=2000)

    fp16_kv_gb = round(n_layers * 2 * n_kv_heads * N * head_dim * 2 / 1024**3, 2)
    cda_kv_gb = round(n_layers * n_kv_heads * N * (head_dim//2 + 2 + head_dim//4 + 2) / 1024**3, 2)

    adv = round(cda_max / fp16_max, 1) if fp16_max > 0 else float('inf')

    print(f"  FP16: max_batch={fp16_max} (kv={fp16_kv_gb}GB/req)")
    print(f"  CDA:  max_batch={cda_max} (kv={cda_kv_gb}GB/req)")
    print(f"  Advantage: {adv}x")

    results[str(N)] = {
        "N": N,
        "fp16_kv_gb": fp16_kv_gb,
        "cda_kv_gb": cda_kv_gb,
        "fp16_max_batch": fp16_max,
        "cda_max_batch": cda_max,
        "batch_advantage": adv,
    }

out_path = PROJECT_ROOT / "runs" / "memory_capacity_measured.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")
