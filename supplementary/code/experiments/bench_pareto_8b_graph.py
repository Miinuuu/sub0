"""Run the same QPS sweep as runs/pareto/ (8B, input=2048, output=128,
200 prompts, QPS ∈ {1,2,4,8,16}) but with CDA_ENABLE_CUDAGRAPH=1 so we can
compare the graph path against the existing eager baseline.

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/bench_pareto_8b_graph.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
OUT_DIR = REPO / "runs" / "pareto_8b_graph"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PY = "python"
PORT = 9200


def _boot_server():
    log = OUT_DIR / "_server.log"
    env = dict(os.environ)
    env.update({
        "CDA_ENABLE_MEMORY_SAVING": "1",
        "CDA_ENABLE_CUDAGRAPH": "1",
    })
    cmd = [
        PY, "-m", "vllm.entrypoints.cli.main", "serve", MODEL,
        "--gpu-memory-utilization", "0.60",
        "--dtype", "float16", "--max-model-len", "4096",
        "--tensor-parallel-size", "1",
        "--port", str(PORT), "--host", "127.0.0.1",
        "--attention-backend", "CUSTOM",
    ]
    with log.open("w") as f:
        server = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    t0 = time.time()
    while time.time() - t0 < 300:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
            break
        except Exception:
            time.sleep(3)
    else:
        server.kill()
        raise RuntimeError("server didn't start in 300s")
    print(f"  server ready after {time.time()-t0:.0f}s")
    # warmup
    urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/completions",
        data=json.dumps({"model": MODEL, "prompt": "hi", "max_tokens": 8}).encode(),
        headers={"Content-Type": "application/json"}))
    return server


def _run_qps(qps: int) -> Path:
    out = OUT_DIR / f"graph_qps{qps}.json"
    cmd = [
        PY, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
        "--backend", "vllm", "--model", MODEL,
        "--host", "127.0.0.1", "--port", str(PORT),
        "--endpoint", "/v1/completions",
        "--dataset-name", "random", "--input-len", "2048", "--output-len", "128",
        "--num-prompts", "200", "--request-rate", str(qps),
        "--ignore-eos", "--seed", "42",
        "--percentile-metrics", "ttft,tpot,itl", "--metric-percentiles", "50,99",
        "--save-result", "--result-filename", str(out),
        "--disable-tqdm",
    ]
    print(f"  [QPS={qps}] running bench (200 prompts)...")
    t0 = time.time()
    subprocess.run(cmd, check=False)
    print(f"  [QPS={qps}] done in {time.time()-t0:.0f}s → {out.name}")
    return out


def main():
    print("=== 8B Pareto QPS sweep w/ CUDA Graph enabled ===\n")
    server = _boot_server()
    try:
        for qps in [1, 2, 4, 8, 16]:
            _run_qps(qps)
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()

    # Summary
    print("\n=== Results (vs existing runs/pareto/ eager baseline) ===")
    for qps in [1, 2, 4, 8, 16]:
        g_path = OUT_DIR / f"graph_qps{qps}.json"
        e_path = REPO / "runs" / "pareto" / f"CDA_qps{qps}.json"
        if not g_path.exists() or not e_path.exists():
            continue
        g = json.loads(g_path.read_text())
        e = json.loads(e_path.read_text())
        ratio = g["median_tpot_ms"] / e["median_tpot_ms"]
        print(f"  QPS={qps:2d}  eager TPOT={e['median_tpot_ms']:6.1f}ms  "
              f"graph TPOT={g['median_tpot_ms']:6.1f}ms  "
              f"graph/eager={ratio:.2f}x  "
              f"graph tok/s={g['output_throughput']:6.1f}  "
              f"eager tok/s={e['output_throughput']:6.1f}")


if __name__ == "__main__":
    main()
