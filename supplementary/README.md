# CDA — Supplementary Material

This folder contains the code, scripts, and raw experimental data
behind every numeric claim in the paper.

## Directory layout

```
supplementary/
├── README.md                      — this file
├── code/
│   ├── experiments/               — our benchmark scripts (used to produce paper numbers)
│   └── core/                      — our CDA library (Python bindings + compiled CUDA kernels)
│       ├── prebuilt_hmma/          — pre-built .so binaries (sm_86 / CUDA 12.8 / PyTorch 2.10)
│       ├── _cda_gqa_kernels.*.so   — CDA Flash-LUT + 3-stage kernels (pre-built)
│       ├── _load_prebuilt.py       — binary loader (replaces the JIT-compile path)
│       ├── cda_attn.py             — Python bindings + auto-dispatch
│       ├── compression.py          — Hadamard + Lloyd-Max quantiser
│       ├── compressed_attention.py — reference (pure PyTorch) compressed-domain attention
│       ├── compressed_generate.py  — long-context decode / PPL eval / calibration
│       ├── _flash_reference.py     — Python split-K oracle (kernel unit tests)
│       └── vllm_integration/cda_backend.py — vLLM 0.19 AttentionBackend (level-C path)
│
│   **CUDA source code is not redistributed.** Pre-built kernels target
│   sm_86 (NVIDIA RTX A6000); for other architectures contact the authors.
└── data/
    ├── accuracy_ppl/              — Table 2 WikiText-2 + C4 PPL raw JSON
    ├── accuracy_long_context/     — Appendix C RULER, App C reasoning (MMLU/GSM8K/Winogrande)
    ├── kernel_bench/              — Tables 3, 4 + Appendix F variants
    ├── system_vllm/               — Table 5 vLLM throughput JSONs
    ├── system_pareto/             — Appendix G TTFT/TPOT Pareto sweep
    ├── system_energy/             — Appendix H energy/token
    ├── system_prefill/            — Appendix E prefill latency + max-context
    ├── composability_awq/         — Appendix I AWQ × CDA K4V2 four-cell PPL
    └── misc/                      — HMMA vs FA2 MSE sanity
```

## Provenance of baseline numbers (IMPORTANT)

| Baseline | Measurement code | Source |
|---|---|---|
| FP16 reference | HuggingFace `transformers` forward | official |
| FA2 kernel | `vllm_flash_attn.flash_attn_varlen_func` (vLLM 0.19) | official |
| **KIVI** (PPL + kernel) | `REF/KIVI` — `LlamaForCausalLM_KIVI` + `cuda_bmm_fA_qB_outer` | **official**  |
| **GEAR** (PPL) | `REF/GEAR` — `compress_method=GEAR`, group=64, rank=2 | **official** |
| **KVQuant** (kernel) | `REF/KVQuant/deployment/kvquant/quant_cuda_kernel.cu` (fused NUQ) | **official** |
| **QuaRot** (PPL + kernel) | `REF/QuaRot/fake_quant/main.py` + `qattention_benchmark.py` | **official** |
| **TurboQuant**$^\dagger$ (PPL) | `turboquant_ours_reimpl/` — our faithful PyTorch re-implementation | **ours** |
| **CDA (ours)** | `code/experiments/bench_*.py` | **ours** |

$^\dagger$ TurboQuant has no public PyTorch evaluation harness; we cross-verified our re-implementation against the authors' reported numbers and in-place operators in `REF/turboquant-pytorch`.

**Only TurboQuant and CDA rows in Table 2 are our code.** All other
baselines run from their official repositories at the commits documented
below. We do **not** redistribute official-code checkouts — follow the
install instructions for each one.

### Reproducing baseline runs

| Repo | URL | Commit | Install |
|---|---|---|---|
| KIVI   | `github.com/jy-yuan/KIVI`        | main @ commit on install date | follow their README; env `kivi` (transformers 4.44.2) |
| GEAR   | `github.com/HaoKang-Timmy/GEAR`  | main | env shared with `kivi` |
| KVQuant| `github.com/SqueezeAILab/KVQuant`| main | env shared with `cda` |
| QuaRot | `github.com/spcl/QuaRot`         | main | env `quarot` (transformers 4.36.2); see `docs/QUAROT_ENV_SETUP.md` |

## Environment

- Python 3.10, PyTorch 2.10.0+cu128, CUDA 12.8 toolkit, Llama tokenizer
- vLLM 0.19.0 + FlashInfer 0.6.6, transformers 4.57.6 (for our code)
- GPU: NVIDIA RTX A6000 (48 GB, sm_86); tests pass on $2$ GPUs locally, $4$ GPUs on lab node
- **Our library** under `code/core/` is a standalone subset of the paper's implementation. The CDA kernel extension (`_cda_gqa_kernels.so`) and HMMA kernels (`_hmma_production_v3.cu`, `_hmma_production_g8.cu`) are compiled JIT at first import via `torch.utils.cpp_extension`.
- A minimal end-to-end run requires CUDA, the ~10 GB Llama-3.1-8B-Instruct checkpoint (from Hugging Face Hub), and ~48 GB GPU VRAM.

## Reproducing each paper claim

Every script below writes its output JSON under `data/<subdir>/`. The file
names match what the paper's figure / table generators consume.

### Table 2 — WikiText-2 PPL

```bash
# CDA (ours) — K4V2 across 7 LLMs
python code/experiments/bench_ppl_scale.py --model llama8b --dataset wikitext2
# ... per-model; see comments inside the script
```
Paired raw outputs in `data/accuracy_ppl/ppl_scale_*/`.

### Appendix C — RULER 13-task

```bash
bash code/experiments/run_ruler_minimal.sh  # uses lm-evaluation-harness RULER
```
Raw outputs in `data/accuracy_long_context/ruler/`.

### Appendix D — C4 cross-check

Same `bench_ppl_scale.py` with `--dataset c4`. Paired raw outputs live
under `data/accuracy_ppl/ppl_scale_*_c4*/`.

### Tables 3+4 — kernel microbench

```bash
python code/experiments/bench_dequant_vs_cda_v3.py   # Table 3 (same-compression isolation)
python code/experiments/bench_hmma_g8_table4.py     # Table 4 L2/L3-70B g8 cells
# Baseline kernels (QuaRot / KVQuant / KIVI) run from their official benches;
# the aggregated comparison JSON is in data/kernel_bench/GOLD_baselines_kernel_summary.json.
```

### Table 5 + Appendix G — vLLM serving

```bash
# Throughput (Table 5)
python code/experiments/bench_max_batch_tput.py --N 8192 --backend CDA    # repeat per N, backend
# TTFT/TPOT Pareto (Appendix G)
CUDA_VISIBLE_DEVICES=0 bash code/experiments/bench_pareto_ttft_tpot.sh FA2
CUDA_VISIBLE_DEVICES=0 bash code/experiments/bench_pareto_ttft_tpot.sh CDA
python code/figures/gen_fig_pareto.py   # produces Fig 3 in App G
```

### Appendix E — prefill + capacity

```bash
# Prefill latency
for N in 8192 16384 32768 65536 131064; do
  for BK in FA2 CDA; do
    CUDA_VISIBLE_DEVICES=0 python code/experiments/bench_prefill_memory.py \
        --backend $BK --N $N --output data/system_prefill/${BK}_N${N}.json
  done
done
# Max-context capacity sweep
for util in 0.35 0.5; do
  for BK in FA2 CDA; do
    CUDA_VISIBLE_DEVICES=0 python code/experiments/bench_max_context.py \
        --backend $BK --gpu-util $util --output data/system_prefill/maxctx_${BK}_util${util}.json
  done
done
```

### Appendix H — energy/token

```bash
CUDA_VISIBLE_DEVICES=0 bash code/experiments/bench_energy_sweep.sh
# or individual points:
python code/experiments/bench_energy_nvml.py --backend CDA --B 16 --N 32768 \
    --iters 3 --output data/system_energy/CDA_B16_N32K.json
```

### Appendix I — AWQ × CDA composability

```bash
for W in FP16 AWQ; do for A in FA2 CDA; do
  CUDA_VISIBLE_DEVICES=0 python code/experiments/bench_awq_cda_ppl.py \
      --weights $W --attn $A --output data/composability_awq/${W}_${A}.json
done; done
```

### HMMA vs FP16 FA2 numerical fidelity

```bash
python code/experiments/bench_hmma_vs_fa2_mse.py     # writes data/misc/hmma_vs_fa2_mse.json
python code/experiments/test_hmma_production.py     # cross-kernel sanity
```

