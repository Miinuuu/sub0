#!/usr/bin/env bash
# B.3 energy sweep driver. Runs all (backend, B, N) points sequentially on a
# single GPU. Skips points whose output JSON already exists.
#
# Usage: CUDA_VISIBLE_DEVICES=1 bash experiments/bench_energy_sweep.sh

set -eu
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="python"
OUT="$(cd "$REPO/.." && pwd)/data/system_energy/energy"
mkdir -p "$OUT"

# Representative (B, N) points covering weight-bound (B=1) and memory-bound (B=16+) regimes.
POINTS=(
    "1 4096"
    "1 16384"
    "1 32768"
    "16 4096"
    "16 8192"
    "16 32768"
)

for BK in FA2 CDA; do
    for pt in "${POINTS[@]}"; do
        B=${pt%% *}; N=${pt##* }
        NK=$((N/1024))K
        F="$OUT/${BK}_B${B}_N${NK}.json"
        if [ -f "$F" ]; then
            echo "skip $(basename $F) (exists)"
            continue
        fi
        echo "=== [${BK}] B=$B N=$N ==="
        $PY $REPO/experiments/bench_energy_nvml.py \
            --backend $BK --B $B --N $N --output-len 128 --iters 3 \
            --output $F \
            > $OUT/_log_${BK}_B${B}_N${NK}.txt 2>&1 \
            || echo "FAILED $BK B=$B N=$N, continuing"
    done
done
echo "=== energy sweep done ==="
