#!/bin/bash
# PPL benchmark: Llama-3.1-70B-Instruct on lab server
# Requires: 4x A6000 (192GB) or 3x A100-80GB (240GB)
#
# Usage:
#   bash experiments/run_ppl_70b.sh              # all methods, wikitext2
#   bash experiments/run_ppl_70b.sh wikitext2 c4 # both datasets

set -euo pipefail
cd "$(dirname "$0")/.."

DATASETS="${@:-wikitext2 c4}"

echo "=== Llama-3.1-70B PPL Benchmark ==="
echo "Datasets: ${DATASETS}"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l) detected"
echo ""

# Phase 1: FP16 baseline + CDA (our method) — 가장 중요
python experiments/bench_ppl_scale.py \
    --model llama70b \
    --dataset ${DATASETS} \
    --methods fp16 cda-4 cda-k4v2 cda-2 \
    --output-dir runs/ppl_scale_llama70b

# Phase 2: Competitors (KIVI, GEAR, TurboQuant)
python experiments/bench_ppl_scale.py \
    --model llama70b \
    --dataset ${DATASETS} \
    --methods kivi-4 kivi-2 gear-4 gear-2 turbo-2 \
    --output-dir runs/ppl_scale_llama70b

echo ""
echo "=== Done. Results in runs/ppl_scale_llama70b/ ==="
