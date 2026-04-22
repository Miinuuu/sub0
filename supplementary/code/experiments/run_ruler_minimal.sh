#!/usr/bin/env bash
# RULER minimal config — 4 NIAH variants × 50 samples × 4 contexts = 800 prompts.
# Expected: 2-3h on 1 GPU (ETA 8-13s/prompt × 800 = 6400s-10400s).
# Two-GPU split: FA2 on GPU 0, CDA on GPU 1.
#
# Usage (launched per-GPU via nohup):
#   BACKEND=fa2 GPU=0 bash experiments/run_ruler_minimal.sh
#   BACKEND=cda GPU=1 bash experiments/run_ruler_minimal.sh

set -e
BACKEND="${BACKEND:-fa2}"
GPU="${GPU:-0}"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$(cd "$REPO/.." && pwd)/data/accuracy_long_context/ruler_min"
mkdir -p "$OUTPUT_DIR"
cd "$REPO"

# Ensure nvcc is on PATH.
export PATH="/usr/local/cuda/bin:$PATH"

# 2 RULER tasks (QA + variable-tracking) — scaling-sensitive subset.
# This lm_eval build packages 6 ruler tasks (cwe, fwe, qa_hotpot, qa_squad, vt).
TASKS="ruler_qa_squad,ruler_vt"
CONTEXTS='[4096, 8192, 16384, 32768]'
LIMIT=100

if [ "$BACKEND" = "cda" ]; then
  export CDA_ENABLE_MEMORY_SAVING=1
  BACKEND_ARGS=",attention_backend=CUSTOM,dtype=float16"
  TAG="cda"
else
  unset CDA_ENABLE_MEMORY_SAVING
  BACKEND_ARGS=""
  TAG="fa2"
fi

CUDA_VISIBLE_DEVICES=$GPU python -m lm_eval --model vllm \
  --model_args "pretrained=${MODEL},tensor_parallel_size=1,max_model_len=32768,enforce_eager=True,gpu_memory_utilization=0.70${BACKEND_ARGS}" \
  --tasks "$TASKS" \
  --limit $LIMIT \
  --batch_size auto \
  --metadata="{\"max_seq_lengths\":${CONTEXTS}}" \
  --output_path "${OUTPUT_DIR}/ruler_${TAG}.json" \
  2>&1 | tee "${OUTPUT_DIR}/ruler_${TAG}.log"
