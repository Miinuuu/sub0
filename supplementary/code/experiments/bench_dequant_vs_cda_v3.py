"""CDA v3+auto HMMA vs dequant-then-FA2 — refresh of bench_dequant_vs_cda.

Same methodology; only the CDA call site uses v3 kernel with
choose_tile_n_hmma() auto selector. Used to refresh paper tab:same-compression
with v3+auto numbers (+ matching speedup ratios)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from statistics import median

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core.compression import HadamardQuantCompressor
from core.cda_attn import cda_decode_full_hmma_v3, choose_tile_n_hmma
import vllm._custom_ops  # noqa (register FA2)

# Re-use _build_synthetic_cache from the v1 bench to guarantee identical
# quantised state across runs. We import rather than duplicate.
from experiments.bench_dequant_vs_cda import _build_synthetic_cache


def _bench_one(B, N, H_q=32, H_kv=8, D=128, warmup=10, iters=50):
    device = torch.device("cuda:0")
    torch.manual_seed(0xCDA)
    ctx = _build_synthetic_cache(B, N, H_kv, D, device)
    group_size = H_q // H_kv
    tile_N = choose_tile_n_hmma(N, B)
    scale = 1.0 / (D ** 0.5)
    max_seq_bound = ctx["num_blk"] * ctx["block_size"]
    q_fp16 = torch.randn(B, H_q, D, dtype=torch.float16, device=device) * 0.3
    out = torch.empty(B, H_q, D, dtype=torch.float16, device=device)

    def run_cda():
        cda_decode_full_hmma_v3(
            q_fp16, ctx["cache"], ctx["block_tbl"], ctx["seq_lens"],
            ctx["cb_k"], ctx["cb_v"], ctx["rot_fp32"], ctx["rot_fp16"], out,
            group_size, ctx["block_size"], tile_N, max_seq_bound, scale,
        )

    from vllm.vllm_flash_attn import flash_attn_varlen_func

    def dequant_kv():
        K = ctx["ck"].dequantize(ctx["cK"]).reshape(-1, H_kv, D)[:N]
        V = ctx["cv"].dequantize(ctx["cV"]).reshape(-1, H_kv, D)[:N]
        return K.to(torch.float16), V.to(torch.float16)

    cu_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, N], dtype=torch.int32, device=device)
    q_fa = q_fp16[0].contiguous().unsqueeze(0)

    def run_dequant_fa2():
        K, V = dequant_kv()
        return flash_attn_varlen_func(
            q_fa, K, V, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=1, max_seqlen_k=N, causal=False, softmax_scale=scale)

    def _time(fn):
        for _ in range(warmup): fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) * 1000.0)
        return median(ts)

    t_cda = _time(run_cda)
    t_dq  = _time(run_dequant_fa2)
    return t_cda, t_dq, tile_N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="1:4096,1:8192,1:32768,4:8192,4:32768,16:8192,16:32768")
    ap.add_argument("--output", default="runs/bench_dequant_vs_cda_v3.json")
    args = ap.parse_args()
    cells = [tuple(int(x) for x in c.split(":")) for c in args.cells.split(",")]

    rows = []
    hdr = f" {'B':>4} {'N':>7} | {'tile':>4} | {'CDA v3 (μs)':>12} | {'dequant+FA2 (μs)':>17} | {'speedup':>8}"
    print(hdr); print("-" * len(hdr))
    for B, N in cells:
        t_cda, t_dq, tile = _bench_one(B, N)
        sp = t_dq / t_cda
        rows.append({"B": B, "N": N, "tile_N": tile,
                     "cda_v3_us": round(t_cda, 1),
                     "dequant_fa2_us": round(t_dq, 1),
                     "speedup": round(sp, 2)})
        print(f" {B:>4} {N:>7} | {tile:>4} | {t_cda:>12.1f} | {t_dq:>17.1f} | {sp:>7.2f}×")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(rows, indent=2))
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
