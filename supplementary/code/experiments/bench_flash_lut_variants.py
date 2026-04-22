"""Compare all existing Flash-LUT variants + my Phase-1 staged kernel."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from statistics import median

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.compression import HadamardQuantCompressor
from core.cda_attn import (
    cda_flash_split_k4v2_gqa,
    cda_flash_split_k4v2_gqa_coop,
    cda_flash_split_k4v2_gqa_ilp,
    cda_flash_split_k4v2_gqa_um,
    cda_flash_split_k4v2_gqa_streamk,
)
from core._lut_staged import cda_flash_staged_k4v2_gqa


def _prep(B, N, H_q=32, H_kv=8, D=128, device="cuda:0"):
    torch.manual_seed(0xCDA + N + B)
    dev = torch.device(device)
    ck = HadamardQuantCompressor(dim=D, bit_width=4, half_rotation=True)
    cv = HadamardQuantCompressor(dim=D, bit_width=2, half_rotation=True)
    ck._ensure_tensors(dev); cv._ensure_tensors(dev)
    cb_k = (ck._centroids * 2.0 - 1.0).float().contiguous()
    cb_v = (cv._centroids * 2.0 - 1.0).float().contiguous()
    Q = (torch.randn(B, H_q, D, device=dev, dtype=torch.float32) * 0.3)
    Q_rot = (Q @ ck._rotation_t.float().contiguous()).contiguous()
    pK = torch.randint(0, 256, (B, H_kv * N, D // 2), dtype=torch.uint8, device=dev)
    pV = torch.randint(0, 256, (B, H_kv * N, D // 4), dtype=torch.uint8, device=dev)
    nK = (torch.rand(B, H_kv * N, device=dev) + 0.5).float().contiguous()
    nV = (torch.rand(B, H_kv * N, device=dev) + 0.5).float().contiguous()
    return {"Q_rot": Q_rot, "pK": pK, "nK": nK, "pV": pV, "nV": nV,
            "cb_k": cb_k, "cb_v": cb_v,
            "B": B, "N": N, "D": D, "H_q": H_q, "H_kv": H_kv,
            "group_size": H_q // H_kv}


def bench_variant(name, fn, B, N, tile_N, extra_args=(), warmup=10, iters=30):
    ctx = _prep(B, N)
    scale = 1.0 / (ctx["D"] ** 0.5)

    def call():
        for b in range(B):
            fn(ctx["Q_rot"][b], ctx["pK"][b], ctx["nK"][b],
               ctx["pV"][b], ctx["nV"][b],
               ctx["cb_k"], ctx["cb_v"],
               N, ctx["group_size"], tile_N, scale, *extra_args)

    try:
        for _ in range(warmup): call()
    except Exception as e:
        return None, str(e)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); call(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    return median(ts), None


VARIANTS = [
    ("flat",    cda_flash_split_k4v2_gqa,         ()),
    ("coop",    cda_flash_split_k4v2_gqa_coop,    ()),
    ("ilp",     cda_flash_split_k4v2_gqa_ilp,     ()),
    ("um",      cda_flash_split_k4v2_gqa_um,      (10.0,)),  # phi upper bound
    ("streamk", cda_flash_split_k4v2_gqa_streamk, (0,)),    # num_ctas=auto
    ("staged",  cda_flash_staged_k4v2_gqa,        ()),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Bs", default="1,16,32,64")
    ap.add_argument("--Ns", default="4096,16384,32768,65536,131072")
    ap.add_argument("--output", default="runs/bench_flash_lut_variants.json")
    args = ap.parse_args()

    Bs = [int(x) for x in args.Bs.split(",")]
    Ns = [int(x) for x in args.Ns.split(",")]

    rows = []
    names = [v[0] for v in VARIANTS]
    hdr = f" {'B':>4} {'N':>7} | " + " | ".join(f"{n:>8}" for n in names)
    print(hdr); print("-" * len(hdr))
    for B in Bs:
        for N in Ns:
            tile_N = 1024 if N >= 8192 else 512
            row = {"B": B, "N": N, "tile_N": tile_N}
            cells = []
            for name, fn, extra in VARIANTS:
                t, err = bench_variant(name, fn, B, N, tile_N, extra)
                row[name] = t
                if t is None:
                    cells.append("   FAIL ")
                else:
                    cells.append(f"{t:>8.1f}")
            rows.append(row)
            print(f" {B:>4} {N:>7} | " + " | ".join(cells))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(rows, indent=2))
    print(f"\nSaved {args.output}")

    # Speedup summary: each variant relative to flat baseline
    print("\nRelative to 'flat' (baseline = 1.00×):")
    print(" " + " | ".join(f"{n:>8}" for n in names))
    for row in rows:
        base = row["flat"]
        if base:
            cells = [f"{(base/row[n]):>8.2f}×" if row.get(n) else "   FAIL " for n in names]
            print(f" B={row['B']:>2} N={row['N']:>6} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
