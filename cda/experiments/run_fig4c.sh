#!/bin/bash
# run_fig4c.sh
# Run the E2E Decode Latency multi-gpu benchmark for Figure 4c across varying sequence lengths and large models.

set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=("llama2-13b" "qwen32b" "llama70b")
SEQLENS=(1024 2048 4096 8192)

echo "=== Running Figure 4c E2E Latency Benchmark ==="

for model in "${MODELS[@]}"; do
    echo "----------------------------------------"
    echo "Evaluating Model: $model"
    echo "----------------------------------------"
    for N in "${SEQLENS[@]}"; do
        echo "[ Model: $model | N=$N ]"
        python experiments/bench_e2e_multigpu.py --model "$model" --N "$N" || echo "Run for $model (N=$N) Failed!"
    done
done

echo "=== DONE Figure 4c Benchmark ==="
