"""Official QuaRot attention kernel bench across (B, N) grid.

Uses QuaRot's MultiLayerPagedKVCache4Bit as-is (MHA, int4+fp16had).
MHA shape (32, 32, 128) = Llama-2-7B attention shape. QuaRot's released
kernel does not support GQA, so this is the fairest we can measure
against their official code.

For direct CDA comparison, a companion bench at the same MHA shape is
needed (our production kernel targets GQA 32/8, so we report both and
note shape mismatch).
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from statistics import median

import torch

sys.path.insert(0, os.environ.get("REF_QuaRot_PATH", "REF/QuaRot").replace(" ", ""))
from quarot.transformers.kv_cache import MultiLayerPagedKVCache4Bit


def bench_quarot(B, N, num_heads=32, head_dim=128,
                 dtype="int4", hadamard_dtype=torch.float16,
                 warmup=10, iters=40):
    device = torch.device("cuda:0")
    torch.manual_seed(0xCDA + N + B)

    cache = MultiLayerPagedKVCache4Bit(
        batch_size=B,
        page_size=N,
        max_seq_len=N,
        device=device,
        n_layers=1,
        num_heads=num_heads,
        head_dim=head_dim,
        disable_quant=(dtype == "fp16"),
        hadamard_dtype=hadamard_dtype,
    )
    q = torch.rand((B, 1, num_heads, head_dim), dtype=torch.float16, device=device)
    k = torch.rand((B, 1, num_heads, head_dim), dtype=torch.float16, device=device)
    v = torch.rand((B, 1, num_heads, head_dim), dtype=torch.float16, device=device)

    def call():
        cache._needs_init = [False] * len(cache._needs_init)
        cache.length = N - 1
        forward_func = cache.update(k, v, layer_idx=0, cache_kwargs={})
        return forward_func(q)

    for _ in range(warmup): call()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); call(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    return median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Bs", default="1,4,16,32")
    ap.add_argument("--Ns", default="4096,16384,32768,65536")
    ap.add_argument("--output", default="runs/bench_quarot_official.json")
    args = ap.parse_args()
    Bs = [int(x) for x in args.Bs.split(",")]
    Ns = [int(x) for x in args.Ns.split(",")]

    print(f"{'B':>3} {'N':>6}  {'FP16 μs':>9}  {'Int4+Had μs':>13}  {'vs FP16':>8}")
    print("-" * 50)
    results = []
    for B in Bs:
        for N in Ns:
            try:
                fp16 = bench_quarot(B, N, dtype="fp16", hadamard_dtype=None)
                int4 = bench_quarot(B, N, dtype="int4", hadamard_dtype=torch.float16)
                ratio = fp16 / int4
                print(f"{B:>3d} {N:>6d}  {fp16:>9.1f}  {int4:>13.1f}  {ratio:>7.2f}×")
                results.append({"B": B, "N": N,
                                "fp16_us": round(fp16, 1),
                                "int4_had_us": round(int4, 1),
                                "int4_vs_fp16": round(ratio, 2)})
            except Exception as e:
                print(f"{B:>3d} {N:>6d}  FAILED: {str(e)[:80]}")
                results.append({"B": B, "N": N, "error": str(e)[:200]})
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
