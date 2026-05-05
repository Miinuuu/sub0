#!/usr/bin/env bash
# Paper artifact reproducer for the CDA supplementary distribution.
#
# Modes:
#   smoke   ~10 s  Table 2 single cell (paper-essential decode kernel)
#   table2  ~3 min Table 2 full sweep B in {1,4,32}, N in {4K,16K,64K}
#   table3  ~2 min Table 3 CDA columns + Lloyd FA2-fork + dequant+FA2 + FA2_fp16
#   fig2    ~1 min Fig. 2 / Table A3 single cell (Llama-3.1-8B, vLLM)
#   tp2     ~2 min Table A4 proxy via TP=2 on 2 GPUs (paper used TP=4 on 4)
#   table4  ~2 min Table 4 PPL smoke (Mistral-7B FP16+CDA, max_ctx=512)
#   table5  ~2 min Table 5 RULER smoke (FA2 niah_single_1, max_ctx=4096)
#   figures ~5 s   Regenerate Fig 1, Fig 2, Table A3 from results/
#   all     all of the above
#
# Requires (cda env, see INSTALL.md):
#   * Python 3.10, PyTorch 2.10 (+cu128), SM86 GPU
#   * lm_eval (Table 5), datasets (Table 4)
#   * vLLM 0.19 (Fig 2 / TP=2 / Table 5)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"

mode="${1:-smoke}"
PY="${PYTHON:-python}"

case "$mode" in
  smoke)
    "$PY" benchmarks/bench_compute_path_iso_cuda_graph.py
    ;;
  table2)
    CELLS="1,4096;4,4096;32,4096;1,16384;4,16384;32,16384;1,65536;4,65536;32,65536" \
        "$PY" benchmarks/bench_compute_path_iso_cuda_graph.py
    ;;
  table3)
    "$PY" benchmarks/bench_kernel_head_to_head.py
    ;;
  fig2)
    "$PY" benchmarks/paper/bench_capacity_sweep.py \
        --prompt-len 4096 --decode-len 64 --batches 1 --gpu 0 \
        --output runs/paper/fig2_smoke.json --skip-fa2
    ;;
  tp2)
    "$PY" benchmarks/paper/bench_capacity_sweep.py \
        --prompt-len 4096 --decode-len 64 --batches 1 \
        --gpu 0,1 --tp 2 \
        --output runs/paper/tp2_smoke.json
    ;;
  table4)
    "$PY" benchmarks/baselines/bench_ppl_scale.py \
        --model mistral7b --methods fp16 cda-4 \
        --max-ctx 512 --stride 128 \
        --output-dir runs/ppl_smoke
    ;;
  table5)
    "$PY" benchmarks/baselines/bench_longbench.py \
        --backend FA2 --max-model-len 4096 --tasks niah_single_1 --limit 2 \
        --enforce-eager --output runs/ruler_smoke.json
    ;;
  figures)
    OUT_FIG="$HERE/figures"; OUT_TAB="$HERE/tables"
    mkdir -p "$OUT_FIG" "$OUT_TAB"
    CDA_FIG_OUT_DIR="$OUT_FIG" "$PY" benchmarks/paper/gen_fig1_4zone.py
    "$PY" benchmarks/paper/gen_fig2_throughput.py \
        results/tableA3_fig2_throughput/16K/*.json results/tableA3_fig2_throughput/64K/*.json results/tableA3_fig2_throughput/128K/*.json results/tableA3_fig2_throughput/4K/*.json \
        --output "$OUT_FIG/fig2_throughput.pdf"
    "$PY" benchmarks/paper/gen_tab_fig2_raw.py \
        results/tableA3_fig2_throughput/16K/*.json results/tableA3_fig2_throughput/64K/*.json results/tableA3_fig2_throughput/128K/*.json results/tableA3_fig2_throughput/4K/*.json \
        --output-tex "$OUT_TAB/tab_fig2_raw.tex"
    "$PY" benchmarks/paper/gen_tab_memory.py \
        results/tableA3_fig2_throughput/16K/*.json results/tableA3_fig2_throughput/64K/*.json results/tableA3_fig2_throughput/128K/*.json results/tableA3_fig2_throughput/4K/*.json \
        > "$OUT_TAB/tab_memory.md" || true
    echo "  -> $OUT_FIG/, $OUT_TAB/"
    ;;
  all)
    "$0" smoke; "$0" table3; "$0" figures
    ;;
  *)
    echo "modes: smoke|table2|table3|fig2|tp2|table4|table5|figures|all"
    exit 1
    ;;
esac
