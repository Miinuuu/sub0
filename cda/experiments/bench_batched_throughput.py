"""Batched CDA throughput benchmark.

Measures kernel-level throughput (tok/s) for batched CDA decode at
various batch sizes and context lengths, compared against FP16 SDPA.

Uses the GQA-aware kernels (_cda_gqa_kernels) which index KV as
kv_head = q_head // group_size inside CUDA. For batching, B_req requests
are stacked along the head dimension: total_heads = B_req * H_q.

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/bench_batched_throughput.py
"""
import gc, json
import torch
import torch.nn.functional as F
from pathlib import Path

from core.compression import HadamardQuantCompressor
from core.cda_attn import score_4b_gqa, vfull_2b_gqa, vsparse_2b_gqa

PROJECT_ROOT = Path(__file__).resolve().parents[1]
device = "cuda"

# Llama-3.1-8B specs
D = 128
H_q = 32
H_kv = 8
GROUP = H_q // H_kv  # 4
N_LAYERS = 32
TOPK = 128
ITERS = 50
WARMUP = 10


def cuda_time(fn, iters=ITERS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def bench_cda(B_req, N, cb4, cb2, scale):
    """Benchmark CDA K4/V2 GQA kernel for B_req requests."""
    total_q = B_req * H_q
    total_kv = B_req * H_kv

    Q_rot = torch.randn(total_q, D, dtype=torch.float32, device=device)
    # KV packed per KV-head (GQA kernel handles q→kv mapping)
    pk4 = torch.randint(0, 256, (total_kv * N, D // 2), dtype=torch.uint8, device=device)
    nk = torch.randn(total_kv * N, dtype=torch.float32, device=device).abs() + 0.1
    pv2 = torch.randint(0, 256, (total_kv * N, D // 4), dtype=torch.uint8, device=device)
    nv = torch.randn(total_kv * N, dtype=torch.float32, device=device).abs() + 0.1

    use_topk = (N >= 16384)

    def run():
        scores = score_4b_gqa(Q_rot, pk4, nk, cb4, N, GROUP, scale)
        attn = torch.softmax(scores, dim=-1)
        if use_topk:
            vals, idx = attn.topk(TOPK, dim=-1)
            vals = vals / (vals.sum(dim=-1, keepdim=True) + 1e-12)
            out = vsparse_2b_gqa(vals, idx.int(), pv2, nv, cb2, TOPK, N, D, GROUP)
        else:
            out = vfull_2b_gqa(attn, pv2, nv, cb2, N, D, GROUP)
        return out

    ms = cuda_time(run)

    del Q_rot, pk4, nk, pv2, nv
    gc.collect(); torch.cuda.empty_cache()
    return ms


def bench_fp16(B_req, N):
    """Benchmark FP16 attention via PyTorch SDPA (FlashAttention)."""
    Q = torch.randn(B_req, H_q, 1, D, dtype=torch.float16, device=device)
    K = torch.randn(B_req, H_q, N, D, dtype=torch.float16, device=device)
    V = torch.randn(B_req, H_q, N, D, dtype=torch.float16, device=device)

    def run():
        return F.scaled_dot_product_attention(Q, K, V, is_causal=False)

    ms = cuda_time(run)

    del Q, K, V
    gc.collect(); torch.cuda.empty_cache()
    return ms


def main():
    scale = D ** -0.5

    print("Loading GQA CUDA kernels...")

    comp4 = HadamardQuantCompressor(dim=D, bit_width=4, half_rotation=True)
    comp2 = HadamardQuantCompressor(dim=D, bit_width=2, half_rotation=True)
    comp4._ensure_tensors(torch.device(device))
    comp2._ensure_tensors(torch.device(device))
    cb4 = (comp4._centroids * 2.0 - 1.0).float().to(device)
    cb2 = (comp2._centroids * 2.0 - 1.0).float().to(device)

    context_lengths = [4096, 16384, 32768, 65536]
    batch_sizes = [1, 2, 4, 8]

    results = {}
    print(f"\n{'='*80}")
    print(f"  Batched CDA Kernel Throughput (Llama-3.1-8B config, A6000)")
    print(f"  H_q={H_q}, H_kv={H_kv}, D={D}, TopK={TOPK} (N>=16K), GQA group={GROUP}")
    print(f"{'='*80}")
    print(f"{'B':>4} {'N':>7} {'CDA ms':>9} {'FP16 ms':>9} "
          f"{'CDA tok/s':>10} {'FP16 tok/s':>11} {'Kernel SP':>10}")
    print(f"{'-'*80}")

    for N in context_lengths:
        for B in batch_sizes:
            key = f"N{N}_B{B}"

            try:
                cda_ms = bench_cda(B, N, cb4, cb2, scale)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    cda_ms = None; torch.cuda.empty_cache()
                else:
                    raise

            try:
                fp16_ms = bench_fp16(B, N)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    fp16_ms = None; torch.cuda.empty_cache()
                else:
                    raise

            cda_tps = B * 1000 / cda_ms if cda_ms else None
            fp16_tps = B * 1000 / fp16_ms if fp16_ms else None
            sp = fp16_ms / cda_ms if (cda_ms and fp16_ms) else None

            results[key] = {
                "N": N, "B": B,
                "cda_kernel_ms": round(cda_ms, 4) if cda_ms else "OOM",
                "fp16_kernel_ms": round(fp16_ms, 4) if fp16_ms else "OOM",
                "cda_tok_s": round(cda_tps, 1) if cda_tps else "OOM",
                "fp16_tok_s": round(fp16_tps, 1) if fp16_tps else "OOM",
                "kernel_speedup": round(sp, 2) if sp else None,
            }

            cs = f"{cda_ms:.4f}" if cda_ms else "OOM"
            fs = f"{fp16_ms:.4f}" if fp16_ms else "OOM"
            ct = f"{cda_tps:.1f}" if cda_tps else "OOM"
            ft = f"{fp16_tps:.1f}" if fp16_tps else "OOM"
            sps = f"{sp:.2f}x" if sp else "---"

            print(f"{B:>4} {N:>7} {cs:>9} {fs:>9} {ct:>10} {ft:>11} {sps:>10}")

        print(f"{'-'*80}")

    out_path = PROJECT_ROOT / "runs" / "batched_throughput.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
