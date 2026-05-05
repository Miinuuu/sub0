"""Official QJL kernel latency bench — runs in QJL's own conda env (qjl).

Uses QJLSketch class directly for correct shape/dtype handling.

Usage: CUDA_VISIBLE_DEVICES=0 python \
         experiments/bench_qjl_official.py
"""
import sys, json
from pathlib import Path
from statistics import median
import torch

QJL_ROOT = "<HOME>/ing/research/REF/QJL"
sys.path.insert(0, QJL_ROOT)
from models.llama3_utils_qjl import QJLSketch

REPO = Path(__file__).resolve().parents[1]

# Llama-3.1-8B shapes matching Tab 4
SHAPES = [
    (1, 4096), (1, 8192), (1, 32768), (1, 65536),
    (16, 32768), (16, 65536), (32, 65536),
]
H_q, H_kv, D = 32, 8, 128
GROUP = 4

# QJL config per run_longbench.py defaults
SKETCH_BITS = 256      # 32 B/tok/head K sketch
OUTLIER_COUNT = 8
OUTLIER_SKETCH_DIM = SKETCH_BITS  # dim_outlier param

WARMUP = 10
ITERS = 40


def _time_us_events(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(ITERS):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    return median(ts)


def bench_qjl(B, N, device="cuda"):
    """Time QJL quantize + calc_score via QJLSketch."""
    sketch = QJLSketch(dim=(D, SKETCH_BITS), dim_outlier=OUTLIER_SKETCH_DIM, device=device, rot=True)
    # keep proj in fp32 (sketch's default) — dispatch will pick half_float kernel

    # K: (B, H_kv, N, GROUP, D)
    K = torch.randn(B, H_kv, N, GROUP, D, dtype=torch.half, device=device)
    # outlier indices (uint8): first OUTLIER_COUNT dims per token
    outlier_idx = torch.zeros(B, H_kv, N, OUTLIER_COUNT, dtype=torch.uint8, device=device)
    for i in range(OUTLIER_COUNT):
        outlier_idx[..., i] = i

    # Quantize: returns key_quant, outlier_quant, outlier_norm
    # In real decode, this is called only for new tokens (1 tok/step amortised);
    # we pre-build the full N-token sketch and then time only the score pass.
    key_quant, outlier_quant, outlier_norm = sketch.quantize(K, outlier_idx)

    # Synthetic inlier norms (shape matches key_quant[:4])
    norm_data = torch.randn(*key_quant.shape[:4], dtype=torch.half, device=device)

    # Q: (B, H_q, 1, D)
    Q = torch.randn(B, H_q, 1, D, dtype=torch.half, device=device)

    def _run_score():
        sketch.calc_score(Q, key_quant, outlier_quant, outlier_idx, norm_data, outlier_norm)
    # Verify once before timing
    _ = sketch.calc_score(Q, key_quant, outlier_quant, outlier_idx, norm_data, outlier_norm)
    torch.cuda.synchronize()
    t_score = _time_us_events(_run_score)

    # Add softmax for symmetric comparison with CommVQ/CDA path
    scores = sketch.calc_score(Q, key_quant, outlier_quant, outlier_idx, norm_data, outlier_norm)
    import torch.nn.functional as F
    def _run_sm():
        F.softmax(scores.to(torch.float32), dim=-1)
    t_sm = _time_us_events(_run_sm)

    return {"qjl_score_us": t_score, "qjl_softmax_us": t_sm, "qjl_total_us": t_score + t_sm}


def run_single(B, N):
    """Run single (B, N) in this process — called by subprocess wrapper."""
    torch.manual_seed(0xCDA)
    qjl = bench_qjl(B, N)
    return qjl


def main():
    """Sequential bench — one (B, N) per subprocess for crash isolation."""
    import subprocess, sys as _sys
    results = []
    this_script = str(Path(__file__).resolve())
    for B, N in SHAPES:
        print(f"\n=== B={B} N={N} ===", flush=True)
        row = {"B": B, "N": N}
        # invoke self as subprocess with --single B N
        try:
            r = subprocess.run(
                [_sys.executable, this_script, "--single", str(B), str(N)],
                capture_output=True, text=True, timeout=300,
                env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": "0"})
            if r.returncode == 0:
                # parse JSON line from stdout
                for line in r.stdout.splitlines():
                    if line.startswith("RESULT:"):
                        qjl = json.loads(line[len("RESULT:"):].strip())
                        row.update(qjl)
                        if qjl.get('qjl_score_us') is not None:
                            print(f"  QJL score={qjl['qjl_score_us']:8.1f} us")
                        else:
                            print(f"  QJL error: {qjl.get('error','unknown')[:200]}")
                        break
                else:
                    print(f"  no RESULT line. stdout: {r.stdout[-200:]}")
                    row["qjl_error"] = "no result"
            else:
                print(f"  subprocess failed exit {r.returncode}. stderr: {r.stderr[-200:]}")
                row["qjl_error"] = f"exit {r.returncode}: {r.stderr[-200:]}"
        except subprocess.TimeoutExpired:
            print(f"  timeout")
            row["qjl_error"] = "timeout"
        results.append(row)

    out = REPO / "runs" / "bench_qjl_official.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {
            "H_q": H_q, "H_kv": H_kv, "D": D, "group_size": GROUP,
            "qjl_sketch_bits": SKETCH_BITS, "qjl_outlier_count": OUTLIER_COUNT,
            "warmup": WARMUP, "iters": ITERS,
            "env": "qjl (torch 2.4.1+cu121)",
        },
        "results": results,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--single":
        B, N = int(sys.argv[2]), int(sys.argv[3])
        try:
            r = run_single(B, N)
            print("RESULT:", json.dumps(r), flush=True)
        except Exception as ex:
            print("RESULT:", json.dumps({"qjl_score_us": None, "error": str(ex)[:300]}), flush=True)
    else:
        main()
