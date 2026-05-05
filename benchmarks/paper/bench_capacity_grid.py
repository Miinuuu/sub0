"""N × B capacity grid sweep — wraps bench_capacity_sweep.py over multiple N.

Drives the OOM-cliff data for paper Section 4 (memory efficiency).
For each prompt-len N, runs the per-batch sweep until OOM, then moves
to the next N. Aggregates everything into one JSON for the table
formatter.

Usage:
    python benchmarks/paper/bench_capacity_grid.py \\
        --shapes 4096:128 32768:128 65536:128 130816:128 \\
        --batches 1 2 4 8 16 32 64 128 \\
        --gpu 0 --gpu-mem-util 0.92 \\
        --output-dir runs/paper

    # Then format:
    python benchmarks/paper/format_paper_tables.py capacity \\
        --artifacts runs/paper/cap_grid_*.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run_one_shape(prompt_len: int, decode_len: int, batches: list[int],
                  gpu: int, gpu_mem_util: float, model: str,
                  max_model_len: int, output_dir: Path,
                  eager: bool, no_cda_memory_saving: bool) -> Path:
    """Invoke bench_capacity_sweep.py for one (prompt_len, decode_len)."""
    out_path = output_dir / f"cap_grid_N{prompt_len}_L{decode_len}.json"
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "bench_capacity_sweep.py"),
        "--prompt-len", str(prompt_len),
        "--decode-len", str(decode_len),
        "--batches", *map(str, batches),
        "--model", model,
        "--max-model-len", str(max_model_len),
        "--gpu-mem-util", str(gpu_mem_util),
        "--gpu", str(gpu),
        "--output", str(out_path),
    ]
    if eager:
        cmd.append("--eager")
    if no_cda_memory_saving:
        cmd.append("--no-cda-memory-saving")

    print(f"\n=== shape: prompt={prompt_len}, decode={decode_len} ===")
    print(f"  → {' '.join(cmd[:3])} ... --output {out_path.name}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, env=os.environ.copy())
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"  ! shape failed (exit {proc.returncode}); see logs", flush=True)
    else:
        print(f"  ✓ done in {dt:.0f}s", flush=True)
    return out_path


def parse_shape(s: str) -> tuple[int, int]:
    """Parse 'N:L' or 'N,L' into (prompt_len, decode_len)."""
    sep = ":" if ":" in s else ","
    parts = s.split(sep)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"shape must be 'prompt_len:decode_len', got {s!r}")
    return int(parts[0]), int(parts[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapes", type=parse_shape, nargs="+", required=True,
                    help="(prompt_len, decode_len) pairs, e.g. 32768:128 130816:128")
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--max-model-len", type=int, default=131072)
    ap.add_argument("--gpu-mem-util", type=float, default=0.92)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--eager", action="store_true",
                    help="Disable CUDA Graph (slower; default uses CG).")
    ap.add_argument("--no-cda-memory-saving", action="store_true",
                    help="Disable enable_cda_memory_saving() (CDA uses 512 B/slot).")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Where to write per-shape JSONs (cap_grid_N*_L*.json)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[Path] = []
    t_total0 = time.time()
    for prompt_len, decode_len in args.shapes:
        path = run_one_shape(prompt_len, decode_len, args.batches,
                              args.gpu, args.gpu_mem_util, args.model,
                              args.max_model_len, args.output_dir,
                              args.eager, args.no_cda_memory_saving)
        if path.exists():
            artifacts.append(path)

    elapsed = time.time() - t_total0
    print(f"\n=== Grid sweep complete in {elapsed/60:.1f} min ===")
    print(f"Artifacts ({len(artifacts)}):")
    for p in artifacts:
        try:
            d = json.loads(p.read_text())
            cfg = d.get("config", {})
            s = d.get("summary", {})
            print(f"  {p.name}: prompt={cfg.get('prompt_len')} decode={cfg.get('decode_len')} | "
                  f"FA2 max B={s.get('fa2_max_batch_ok')}, "
                  f"CDA max B={s.get('cda_max_batch_ok')}, "
                  f"× = {s.get('capacity_multiplier'):.2f}"
                  if s.get("capacity_multiplier") else
                  f"  {p.name}: prompt={cfg.get('prompt_len')} decode={cfg.get('decode_len')}")
        except Exception:
            print(f"  {p.name}: (could not read)")

    print(f"\nFormat tables:")
    paths_str = " ".join(str(p) for p in artifacts)
    print(f"  python benchmarks/paper/format_paper_tables.py capacity --artifacts {paths_str}")


if __name__ == "__main__":
    main()
