"""Reproduce Table 5: Decode attention latency (FlashAttn vs CDA K2/V2 vs K4/V2).

Paper: Table 5, H=32, d=128, B=1, A6000.
FP16 baseline: PyTorch SDPA (FlashAttention backend).
CDA: TopK128 auto-enabled at N>=8K.

Usage:
    CUDA_VISIBLE_DEVICES=1 python experiments/bench_table5.py
    CUDA_VISIBLE_DEVICES=1 python experiments/bench_table5.py --ns 1024,4096,32768
"""
import os, sys, time, gc, json, argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.expanduser("~/ing/research"))
from core.core.compression import HadamardQuantCompressor
from core.core.cuda_attention_final import cuda_cda_final, _get_final
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--ns", type=str,
                    default="1024,4096,8192,16384,32768,65536,131072,262144,524288,1048576")
parser.add_argument("--heads", type=int, default=32, help="H_q (query heads)")
parser.add_argument("--iters", type=int, default=50)
parser.add_argument("--topk", type=str, default="auto",
                    help="TopK value or 'auto' (128 for N>=16K, 0 otherwise). Comma-sep for sweep.")
args = parser.parse_args()

N_LIST = [int(x) for x in args.ns.split(",")]
device = "cuda"
D = 128
H = args.heads
scale = D ** -0.5

print(f"Compiling CUDA kernels...")
_get_final()

comp2 = HadamardQuantCompressor(dim=D, bit_width=2, half_rotation=True)
comp4 = HadamardQuantCompressor(dim=D, bit_width=4, half_rotation=True)
comp2._ensure_tensors(torch.device(device))
comp4._ensure_tensors(torch.device(device))
cb2 = (comp2._centroids * 2.0 - 1.0).float().to(device)
cb4 = (comp4._centroids * 2.0 - 1.0).float().to(device)
rot = comp2._rotation.float().to(device)
rot_t = comp2._rotation_t.float().to(device)


def bench(fn, iters=None):
    iters = iters or args.iters
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def mem_bytes(N, mode):
    """Per-config KV memory in bytes (all heads combined)."""
    if mode == "fp16":
        return H * N * D * 2 * 2  # K+V, FP16
    elif mode == "k2v2":
        return H * N * ((D // 4 + 2) * 2)  # packed + FP16 norm, K+V
    elif mode == "k4v2":
        return H * N * ((D // 2 + 2) + (D // 4 + 2))  # K:4bit+norm, V:2bit+norm


def fmt_mem(b):
    if b >= 1024**3:
        return f"{b / 1024**3:.1f}\\,GB"
    else:
        return f"{b / 1024**2:.0f}\\,MB"


results = {}

print(f"\n{'='*90}")
print(f"Table 5: Decode attention latency (H={H}, D={D}, B=1)")
print(f"{'='*90}")
print(f"{'N':>8} | {'FlashAttn':>10} {'CDA K2/V2':>10} {'CDA K4/V2':>10} | "
      f"{'K2/V2 sp':>9} | {'FP16 KV':>10} {'K2/V2 KV':>10} {'K4/V2 KV':>10}")
print("-" * 95)

for N in N_LIST:
    entry = {"N": N}
    if args.topk == "auto":
        topk = 128 if N >= 16384 else 0
    else:
        topk = int(args.topk)

    # Memory calculation
    fp_mem = mem_bytes(N, "fp16")
    k2v2_mem = mem_bytes(N, "k2v2")
    k4v2_mem = mem_bytes(N, "k4v2")

    # --- FP16 FlashAttention (SDPA) ---
    fp_oom = False
    try:
        Qf = torch.randn(1, H, 1, D, device=device, dtype=torch.float16)
        Kf = torch.randn(1, H, N, D, device=device, dtype=torch.float16)
        Vf = torch.randn(1, H, N, D, device=device, dtype=torch.float16)
        t_fp = bench(lambda: F.scaled_dot_product_attention(Qf, Kf, Vf))
        entry["flash_ms"] = round(t_fp, 3)
        del Qf, Kf, Vf
        gc.collect(); torch.cuda.empty_cache()
    except RuntimeError:
        fp_oom = True
        entry["flash_ms"] = "OOM"
        gc.collect(); torch.cuda.empty_cache()

    # --- CDA K2/V2 ---
    try:
        Qr = torch.randn(H, D, device=device, dtype=torch.float32)
        Qr = (Qr @ rot_t).float()
        pk2 = torch.randint(0, 256, (H * N, D // 4), device=device, dtype=torch.uint8)
        nk2 = torch.randn(H * N, device=device, dtype=torch.float32).abs() * 10
        nv2 = torch.randn(H * N, device=device, dtype=torch.float32).abs() * 3
        t_k2v2 = bench(lambda: cuda_cda_final(Qr, pk2, nk2, pk2, nv2,
                                               cb2, cb2, rot, scale, N,
                                               k_bits=2, v_bits=2, attn_gate_topk=topk))
        entry["k2v2_ms"] = round(t_k2v2, 3)
        del Qr, pk2, nk2, nv2
        gc.collect(); torch.cuda.empty_cache()
    except RuntimeError:
        entry["k2v2_ms"] = "OOM"
        t_k2v2 = float('inf')
        gc.collect(); torch.cuda.empty_cache()

    # --- CDA K4/V2 ---
    try:
        Qr = torch.randn(H, D, device=device, dtype=torch.float32)
        Qr = (Qr @ rot_t).float()
        pk4 = torch.randint(0, 256, (H * N, D // 2), device=device, dtype=torch.uint8)
        nk4 = torch.randn(H * N, device=device, dtype=torch.float32).abs() * 10
        pv2 = torch.randint(0, 256, (H * N, D // 4), device=device, dtype=torch.uint8)
        nv2 = torch.randn(H * N, device=device, dtype=torch.float32).abs() * 3
        t_k4v2 = bench(lambda: cuda_cda_final(Qr, pk4, nk4, pv2, nv2,
                                               cb4, cb2, rot, scale, N,
                                               k_bits=4, v_bits=2, attn_gate_topk=topk))
        entry["k4v2_ms"] = round(t_k4v2, 3)
        del Qr, pk4, nk4, pv2, nv2
        gc.collect(); torch.cuda.empty_cache()
    except RuntimeError:
        entry["k4v2_ms"] = "OOM"
        t_k4v2 = float('inf')
        gc.collect(); torch.cuda.empty_cache()

    # Speedup
    if not fp_oom and t_k2v2 != float('inf'):
        entry["speedup_k2v2"] = round(t_fp / t_k2v2, 1)
    entry["fp16_kv"] = fp_mem
    entry["k2v2_kv"] = k2v2_mem
    entry["k4v2_kv"] = k4v2_mem

    results[str(N)] = entry

    # Print row
    fp_s = f"{t_fp:>8.3f}ms" if not fp_oom else "       OOM"
    k2_s = f"{t_k2v2:>8.3f}ms" if t_k2v2 != float('inf') else "       OOM"
    k4_s = f"{t_k4v2:>8.3f}ms" if t_k4v2 != float('inf') else "       OOM"
    sp_s = f"{t_fp/t_k2v2:>7.1f}x" if not fp_oom and t_k2v2 != float('inf') else "     ---"
    print(f"{N:>8} | {fp_s} {k2_s} {k4_s} | {sp_s} | "
          f"{fp_mem/1024**2:>8.0f}MB {k2v2_mem/1024**2:>8.0f}MB {k4v2_mem/1024**2:>8.0f}MB")

    # Save incrementally — merge with existing file to avoid overwriting other N values
    # Use per-N suffix when topk is explicit to prevent file conflicts across parallel runs
    if args.topk != "auto":
        suffix = f"_topk{topk}_N{N}"
        out_path = PROJECT_ROOT / "runs" / f"table5{suffix}.json"
    else:
        out_path = PROJECT_ROOT / "runs" / "table5.json"
    existing = {}
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
    existing.update(results)
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)

print(f"\nSaved to {out_path}")
print("Done.")
