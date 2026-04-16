"""GPU Roofline Analysis: compute arithmetic intensity and plot on A6000 roofline.

Reads kernel timing from table5.json and computes ops/byte for FP16 and CDA.
No GPU needed — purely analytical from existing measurements.

A6000 specs:
  - Memory BW: 768 GB/s (GDDR6)
  - FP32 peak: 38.7 TFLOPS
  - FP16 peak: 77.4 TFLOPS (with tensor cores: 309.7 TFLOPS)
  - Ridge point (FP16): 77.4 / 0.768 = 100.8 ops/byte

Usage:
    python experiments/bench_gpu_roofline.py
"""
import json, os
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# A6000 specs
BW_GBs = 768        # GB/s
FP16_TFLOPS = 77.4  # TFLOPS (non-tensor-core)
RIDGE_POINT = FP16_TFLOPS * 1e3 / BW_GBs  # ops/byte = 100.8

# Model params (Llama-3.1-8B)
D = 128          # head_dim
H_Q = 32         # query heads
H_KV = 8         # KV heads (GQA group=4)
GQA_GROUP = H_Q // H_KV  # 4

# Per-token data sizes
FP16_BYTES_PER_TOK = D * 2 * 2  # K + V, FP16 = 512 bytes
CDA_K4V2_BYTES_PER_TOK = (D // 2 + 2) + (D // 4 + 2)  # K:4bit packed + norm, V:2bit packed + norm = 100 bytes

# Per-token operations
# FP16 attention: Q·K^T = D MACs = 2D FLOPs, then alpha·V = D MACs = 2D FLOPs
# Total per KV head per token: 4D FLOPs (score + output)
FP16_FLOPS_PER_TOK = 4 * D  # = 512 FLOPs

# CDA K-Score: D LUT lookups + D additions (no multiplications in hot loop)
# 1 scalar multiply (norm) + D additions = D+1 FLOPs equivalent
# CDA V-Output (TopK=128): for each selected token: 2^Bv additions per dim = D * 1 add
# But TopK makes V cost O(K) not O(N), so per-context-token: only K-Score matters
# K-Score: D additions + 1 multiply = D+1 ≈ D ops
# We count LUT lookup as equivalent to 1 FMA for roofline purposes
CDA_FLOPS_PER_TOK = 2 * D  # D lookups + D adds for K-Score (V is TopK-gated, amortized)

# Load measured kernel timings
table5_path = PROJECT_ROOT / "runs" / "table5.json"
with open(table5_path) as f:
    table5 = json.load(f)

results = {"a6000_specs": {"bw_gbs": BW_GBs, "fp16_tflops": FP16_TFLOPS, "ridge_point": round(RIDGE_POINT, 1)}}

print("=== GPU Roofline Analysis (A6000) ===")
print(f"BW: {BW_GBs} GB/s, FP16 peak: {FP16_TFLOPS} TFLOPS, Ridge: {RIDGE_POINT:.1f} ops/byte")
print()
print(f"{'N':>8} | {'FP16 AI':>9} {'CDA AI':>9} | {'FP16 GFLOPS':>12} {'CDA GFLOPS':>12} | {'BW limit':>10} {'Compute limit':>14}")
print("-" * 90)

for n_str in sorted(table5.keys(), key=int):
    d = table5[n_str]
    N = int(n_str)
    flash_ms = d["flash_ms"]
    k4v2_ms = d.get("k4v2_ms")

    if not k4v2_ms or flash_ms == "OOM":
        continue

    # Total data moved per kernel call (all heads)
    # FP16: read Q (H_Q * D * 2) + read K (H_KV * N * D * 2) + read V (H_KV * N * D * 2) + write O (H_Q * D * 2)
    fp16_bytes = H_KV * N * FP16_BYTES_PER_TOK + H_Q * D * 2 * 2  # KV read + Q/O
    # CDA: read Q (H_Q * D * 4) + read compressed KV (H_KV * N * CDA_bytes) + write O (H_Q * D * 2)
    cda_bytes = H_KV * N * CDA_K4V2_BYTES_PER_TOK + H_Q * D * (4 + 2)  # KV read + Q/O

    # Total FLOPs per kernel call (all heads)
    fp16_flops = H_Q * N * FP16_FLOPS_PER_TOK  # each query head attends to N tokens
    cda_flops = H_Q * N * CDA_FLOPS_PER_TOK

    # Arithmetic intensity
    fp16_ai = fp16_flops / fp16_bytes
    cda_ai = cda_flops / cda_bytes

    # Achieved throughput (GFLOPS)
    fp16_gflops = fp16_flops / (flash_ms * 1e-3) / 1e9
    cda_gflops = cda_flops / (k4v2_ms * 1e-3) / 1e9

    # Roofline bound
    fp16_bw_bound = fp16_ai * BW_GBs  # GFLOPS achievable under BW limit
    cda_bw_bound = cda_ai * BW_GBs

    regime_fp16 = "BW-limited" if fp16_ai < RIDGE_POINT else "Compute-limited"
    regime_cda = "BW-limited" if cda_ai < RIDGE_POINT else "Compute-limited"

    print(f"{N:>8} | {fp16_ai:>8.2f}  {cda_ai:>8.2f}  | {fp16_gflops:>10.1f}  {cda_gflops:>10.1f}  | {regime_fp16:>10} {regime_cda:>14}")

    results[n_str] = {
        "N": N,
        "fp16_ai": round(fp16_ai, 3),
        "cda_ai": round(cda_ai, 3),
        "fp16_gflops": round(fp16_gflops, 1),
        "cda_gflops": round(cda_gflops, 1),
        "fp16_regime": regime_fp16,
        "cda_regime": regime_cda,
    }

out_path = PROJECT_ROOT / "runs" / "gpu_roofline.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")
