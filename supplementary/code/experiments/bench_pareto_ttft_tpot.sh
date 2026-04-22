#!/usr/bin/env bash
# B.1 Ultraplan: TTFT/TPOT Pareto curve via vLLM's `bench serve`.
#
# For each backend ∈ {FA2, CDA}, start a vLLM server, sweep --request-rate
# ∈ {1,2,4,8,16}, collect TTFT/TPOT (p50/p99) from the JSON save-result,
# then kill the server and repeat.
#
# Usage: CUDA_VISIBLE_DEVICES=0 bash experiments/bench_pareto_ttft_tpot.sh BACKEND
# BACKEND ∈ {FA2, CDA}.

set -eu

BACKEND="${1:-FA2}"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(cd "$REPO/.." && pwd)/data/system_pareto/pareto"
mkdir -p "$OUT_DIR"
mkdir -p "$OUT_DIR"
cd "$REPO"

INPUT_LEN=2048
OUTPUT_LEN=128
NUM_PROMPTS=200       # enough to get p99 stats
PORT=8765

VLLM="python -m vllm.entrypoints.cli.main"

# ---------------- start server in background ----------------
SERVER_ARGS=(
    serve "$MODEL"
    --enforce-eager
    --gpu-memory-utilization 0.85
    --dtype float16
    --max-model-len 4096
    --port "$PORT"
    --host 127.0.0.1
)
if [ "$BACKEND" = "CDA" ]; then
    export CDA_ENABLE_MEMORY_SAVING=1
    SERVER_ARGS+=(--attention-backend CUSTOM)
fi

echo "=== [${BACKEND}] starting server on port $PORT ==="
$VLLM "${SERVER_ARGS[@]}" \
    > "$OUT_DIR/_server_${BACKEND}.log" 2>&1 &
SERVER_PID=$!
echo "server pid=$SERVER_PID"

trap 'kill -TERM $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true' EXIT

# wait for health
for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "=== [${BACKEND}] server ready after ${i}s ==="
        break
    fi
    sleep 2
done

if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "ERROR: server never became ready. Log tail:"
    tail -40 "$OUT_DIR/_server_${BACKEND}.log"
    exit 1
fi

# ---------------- sweep request-rate ----------------
for QPS in 1 2 4 8 16; do
    echo "=== [${BACKEND}] bench at QPS=${QPS} ==="
    OUT="$OUT_DIR/${BACKEND}_qps${QPS}.json"
    $VLLM bench serve \
        --backend vllm \
        --model "$MODEL" \
        --host 127.0.0.1 --port "$PORT" \
        --endpoint /v1/completions \
        --dataset-name random \
        --input-len "$INPUT_LEN" \
        --output-len "$OUTPUT_LEN" \
        --num-prompts "$NUM_PROMPTS" \
        --request-rate "$QPS" \
        --ignore-eos \
        --seed 42 \
        --percentile-metrics "ttft,tpot,itl" \
        --metric-percentiles "50,99" \
        --save-result \
        --result-filename "$OUT" \
        --disable-tqdm \
        > "$OUT_DIR/_bench_${BACKEND}_qps${QPS}.log" 2>&1 || {
            echo "bench qps=$QPS failed, continuing"
            tail -5 "$OUT_DIR/_bench_${BACKEND}_qps${QPS}.log"
        }
done

echo "=== [${BACKEND}] done ==="
kill -TERM $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
