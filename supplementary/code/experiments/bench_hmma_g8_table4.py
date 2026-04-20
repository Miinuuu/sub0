"""L2/L3-70B HMMA kernel bench — fills Table 4 cells previously marked N/A.

Shape: H_q=64, H_kv=8, D=128 (group_size=8). Measured via HMMA g8 kernel
(_hmma_production_g8.cu) with auto tile_N. Compared against FA2 baseline
at the same shape.

The kernel latency is independent of actual weight values — random
compressed K/V cache tensors suffice. This unblocks Table 4 column that
was marked N/A because the original HMMA kernel hard-coded group_size=4.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from statistics import median

import torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import vllm._custom_ops  # noqa (register FA2)
from core.compression import HadamardQuantCompressor
from core.cda_attn import cda_decode_full_hmma_g8, choose_tile_n_hmma


def bench_fa2(B, N, H_q=64, H_kv=8, D=128, warmup=10, iters=40):
    """FA2 at 70B shape — H_q=64 instead of 32."""
    device = torch.device("cuda:0"); torch.manual_seed(0xCDA + N + B)
    q = torch.randn(B, 1, H_q, D, dtype=torch.float16, device=device)
    k = torch.randn(B, N, H_kv, D, dtype=torch.float16, device=device)
    v = torch.randn(B, N, H_kv, D, dtype=torch.float16, device=device)
    cu_q = torch.arange(B + 1, dtype=torch.int32, device=device)
    cu_k = torch.arange(0, (B + 1) * N, N, dtype=torch.int32, device=device)
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    def call():
        return flash_attn_varlen_func(
            q.view(B, H_q, D), k.view(B * N, H_kv, D), v.view(B * N, H_kv, D),
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=1, max_seqlen_k=N, causal=False, softmax_scale=1.0/(D**0.5))

    for _ in range(warmup): call()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); call(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    return median(ts)


def prep_hmma(B, N, H_q=64, H_kv=8, D=128, device="cuda:0"):
    torch.manual_seed(0xCDA + N + B)
    block_size = 16
    num_blk = (N + block_size - 1) // block_size
    total_slots = B * num_blk * block_size
    cache = torch.zeros(total_slots, H_kv, 104, dtype=torch.uint8, device=device)
    cache[:, :, :96] = torch.randint(0, 256, (total_slots, H_kv, 96), dtype=torch.uint8, device=device)
    nK = torch.rand(total_slots, H_kv, device=device) + 0.5
    nV = torch.rand(total_slots, H_kv, device=device) + 0.5
    cache[:, :, 96:100]  = nK.contiguous().view(torch.uint8).view(total_slots, H_kv, 4)
    cache[:, :, 100:104] = nV.contiguous().view(torch.uint8).view(total_slots, H_kv, 4)
    ck = HadamardQuantCompressor(dim=D, bit_width=4, half_rotation=True)
    cv = HadamardQuantCompressor(dim=D, bit_width=2, half_rotation=True)
    ck._ensure_tensors(torch.device(device)); cv._ensure_tensors(torch.device(device))
    cb_k = (ck._centroids*2.-1.).float().contiguous()
    cb_v = (cv._centroids*2.-1.).float().contiguous()
    rot32 = ck._rotation.float().contiguous(); rot16 = rot32.to(torch.float16).contiguous()
    block_tbl = torch.arange(B*num_blk, dtype=torch.int32, device=device).view(B, num_blk)
    seq_lens = torch.full((B,), N, dtype=torch.int32, device=device)
    q = torch.randn(B, H_q, D, dtype=torch.float16, device=device) * 0.3
    out = torch.empty(B, H_q, D, dtype=torch.float16, device=device)
    return dict(q=q, cache=cache, block_tbl=block_tbl, seq_lens=seq_lens,
                cb_k=cb_k, cb_v=cb_v, rot32=rot32, rot16=rot16, out=out,
                block_size=block_size, max_seq_bound=num_blk*block_size,
                group_size=H_q // H_kv, D=D)


def bench_hmma(ctx, tile_N, warmup=10, iters=30):
    scale = 1.0/(ctx["D"]**0.5)
    def call():
        cda_decode_full_hmma_g8(ctx['q'], ctx['cache'], ctx['block_tbl'], ctx['seq_lens'],
                                 ctx['cb_k'], ctx['cb_v'], ctx['rot32'], ctx['rot16'], ctx['out'],
                                 ctx['group_size'], ctx['block_size'], tile_N, ctx['max_seq_bound'], scale)
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
    ap.add_argument("--Bs", default="1,8,16,32")
    ap.add_argument("--Ns", default="4096,16384,32768,65536,131072")
    ap.add_argument("--output", default="runs/bench_hmma_g8_table4.json")
    args = ap.parse_args()
    Bs = [int(x) for x in args.Bs.split(",")]; Ns = [int(x) for x in args.Ns.split(",")]

    rows = []
    hdr = f" {'B':>4} {'N':>7} | {'FA2 (μs)':>9} | {'HMMA g8 (μs)':>12} | {'tile':>4} | {'CDA/FA2':>8}"
    print(hdr); print("-" * len(hdr))
    for B in Bs:
        for N in Ns:
            try:
                t_fa2 = bench_fa2(B, N)
            except Exception as e:
                t_fa2 = None; print(f"  FA2 FAIL {B=} {N=}: {type(e).__name__}")
            tile = choose_tile_n_hmma(N, B)
            try:
                ctx = prep_hmma(B, N)
                t_hmma = bench_hmma(ctx, tile)
            except Exception as e:
                t_hmma = None; print(f"  HMMA FAIL {B=} {N=} tile={tile}: {type(e).__name__}")
            rows.append({"B": B, "N": N, "fa2_us": t_fa2, "hmma_g8_us": t_hmma, "tile_N": tile,
                         "cda_vs_fa2": (t_fa2/t_hmma) if (t_fa2 and t_hmma) else None})
            fa2_str = f"{t_fa2:>9.1f}" if t_fa2 else "   FAIL  "
            h_str = f"{t_hmma:>12.1f}" if t_hmma else "     FAIL   "
            r_str = f"{(t_fa2/t_hmma):>7.2f}×" if (t_fa2 and t_hmma) else "   -    "
            print(f" {B:>4} {N:>7} | {fa2_str} | {h_str} | {tile:>4} | {r_str}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(rows, indent=2))
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
