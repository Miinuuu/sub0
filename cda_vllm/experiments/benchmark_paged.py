"""Paged CDA kernel throughput microbenchmark.

Measures the dense and sparse (Top-K) paged kernels across a range of
batch × context configurations, and prints per-iteration latency.

Usage::

    CUDA_VISIBLE_DEVICES=0 python experiments/benchmark_paged.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cda_vllm import cuda_cda_paged  # noqa: E402


def _synthetic_paged_inputs(B: int, N: int, D: int, block_size: int, device):
    k_pack = D // 4
    Q_rot = torch.randn(B, D, device=device, dtype=torch.float32)
    packed_K = torch.randint(0, 256, (B, N, k_pack), device=device, dtype=torch.uint8)
    norms_K = torch.randn(B, N, device=device, dtype=torch.float32).abs() + 0.1
    packed_V = torch.randint(0, 256, (B, N, k_pack), device=device, dtype=torch.uint8)
    norms_V = torch.randn(B, N, device=device, dtype=torch.float32).abs() + 0.1
    cb_k = torch.randn(4, device=device, dtype=torch.float32)
    cb_v = torch.randn(4, device=device, dtype=torch.float32)
    rotation = torch.eye(D, device=device, dtype=torch.float32)

    max_logical_blocks = (N + block_size - 1) // block_size
    num_phys = B * max_logical_blocks
    pk_blocks = torch.zeros((num_phys, block_size, k_pack), device=device, dtype=torch.uint8)
    nk_blocks = torch.zeros((num_phys, block_size), device=device, dtype=torch.float32)
    pv_blocks = torch.zeros((num_phys, block_size, k_pack), device=device, dtype=torch.uint8)
    nv_blocks = torch.zeros((num_phys, block_size), device=device, dtype=torch.float32)
    block_tables = torch.arange(num_phys, device=device, dtype=torch.int32) \
        .view(B, max_logical_blocks)

    phys = 0
    for b in range(B):
        for lb in range(max_logical_blocks):
            for off in range(block_size):
                n = lb * block_size + off
                if n < N:
                    pk_blocks[phys, off] = packed_K[b, n]
                    nk_blocks[phys, off] = norms_K[b, n]
                    pv_blocks[phys, off] = packed_V[b, n]
                    nv_blocks[phys, off] = norms_V[b, n]
            phys += 1

    return dict(
        Q_rot=Q_rot,
        packed_K_blocks=pk_blocks, norms_K_blocks=nk_blocks,
        packed_V_blocks=pv_blocks, norms_V_blocks=nv_blocks,
        block_tables=block_tables,
        codebook_k=cb_k, codebook_v=cb_v,
        rotation=rotation,
        scale=1.0 / math.sqrt(D), N=N, block_size=block_size,
    )


def _time(fn, iters: int) -> float:
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    ev0.record()
    for _ in range(iters):
        fn()
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / iters  # ms


def main() -> None:
    device = torch.device("cuda:0")
    scenarios = [
        {"B": 1,  "N": 16384},
        {"B": 4,  "N": 8192},
        {"B": 16, "N": 4096},
        {"B": 32, "N": 2048},
    ]
    D, block_size = 128, 16
    iters = 100
    topk = 128

    print(f"{'B':>3} {'N':>6}   dense(us)   sparse(us)   dense/sparse")
    for s in scenarios:
        args = _synthetic_paged_inputs(s["B"], s["N"], D, block_size, device)
        dense_ms = _time(lambda: cuda_cda_paged(**args, attn_gate_topk=0), iters)
        sparse_ms = _time(lambda: cuda_cda_paged(**args, attn_gate_topk=topk), iters)
        ratio = dense_ms / sparse_ms if sparse_ms else float("nan")
        print(f"{s['B']:>3} {s['N']:>6}   {dense_ms*1000:>9.2f}   "
              f"{sparse_ms*1000:>10.2f}   {ratio:>12.2f}x")


if __name__ == "__main__":
    main()
