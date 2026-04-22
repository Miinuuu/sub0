# CDA — Supplementary Material

This folder contains the code, scripts, and raw experimental data behind
every numeric claim in the paper. The bundle is self-contained and reads
out-of-the-box on the target system (see `INSTALL.md`). **CUDA source
code is not redistributed**; we ship pre-built `.so` binaries only
(sm_86, CUDA 12.8, PyTorch 2.10). For a rebuild on another architecture,
contact the authors.

## Directory layout

```
supplementary/
├── README.md                               — this file
├── INSTALL.md                              — environment + sanity check
├── requirements.txt
├── code/
│   ├── experiments/                        — 31 benchmark / ablation scripts
│   └── core/
│       ├── prebuilt_hmma/                  — HMMA + fused .so binaries
│       │   ├── _hmma_production.so         —   baseline HMMA
│       │   ├── _cda_hmma_production_v3.so  —   v3 (cp.async V prefetch, 8B)
│       │   ├── _cda_hmma_production_g8.so  —   g8 (group_size=8, 70B)
│       │   ├── _cda_fused_update.so        —   fused quant+scatter
│       │   └── _cda_fused_reduce_rot.so    —   fused reduce+rotation+fp16
│       ├── _cda_gqa_kernels.*.so           — CDA flash (17 kernel bindings)
│       ├── _load_prebuilt.py               — .so loader (no JIT, no .cu)
│       ├── cda_attn.py                     — Python bindings + dispatch
│       │                                      (loads .so via _load_prebuilt)
│       ├── compression.py                  — Hadamard + Lloyd-Max quantizer
│       ├── compressed_attention.py         — pure-PyTorch reference
│       ├── compressed_generate.py          — long-context decode / PPL / calib
│       ├── _flash_reference.py             — Python split-K oracle
│       ├── _lut_staged.py                  — staged LUT variant (App F row)
│       └── vllm_integration/cda_backend.py — vLLM 0.19 CUSTOM backend
│
└── data/                                   — 230 + raw JSON files
    ├── accuracy_ppl/
    │   ├── ppl_scale_{llama8b,llama70b,qwen32b,...}/  — Table 2 (WikiText-2)
    │   ├── ppl_scale_*_c4*/                — Appendix D (C4 cross-check)
    │   ├── ablation_{hadamard,quarot_style,bitwidth}.json — Tab 9 (8B)
    │   ├── ablations_70b_max4k/            — Tab 28 (ablation-70b)
    │   ├── turboquant_ours_reimpl/         — our TurboQuant re-implementation
    │   ├── ppl_kivi_official{,_local}/     — KIVI official PPL log + summary
    │   └── gear_ppl/                       — GEAR PPL text dump
    │
    ├── accuracy_long_context/
    │   ├── ruler_min/                      — Tab 10 (8B RULER 4K)
    │   ├── ruler_70b/                      — Tab 10 (70B RULER)
    │   └── lm_eval_reasoning/              — MMLU/GSM8K/Winogrande
    │
    ├── kernel_bench/
    │   ├── bench_dequant_vs_cda_v3.json    — Tab 4 (same-compression)
    │   ├── bench_dequant_vs_cda{,_g8}.json  — related / 70B g8
    │   ├── bench_quarot_official{,_missing_cells}.json
    │   ├── bench_kivi_official{,_B64,_missing_cells}.json
    │   ├── bench_kvquant_official.json
    │   ├── bench_*vs_cda_summary.json      — per-baseline summaries
    │   ├── GOLD_baselines_kernel_summary.json — Tab 5 (gold-baselines)
    │   ├── bench_hmma_v3_kernel_sweep.json — Tab 15 (variants)
    │   ├── bench_hmma_tp_llama70b.json     — Tab 16 / Tab 25
    │   ├── bench_hmma_g8_table4.json       — Tab 16
    │   ├── bench_flash_lut_variants.json   — Appendix F (flash-LUT)
    │   └── hmma_v3_table4_cells.json
    │
    ├── system_vllm/
    │   ├── tput_graph/                     — Fig 2(c) full B-sweep (graph mode)
    │   ├── tput_cliff_graph/               — Tab 8 (cliff anchors)
    │   ├── tput_{cda,fa2}_N*.json          — eager-mode (legacy reference)
    │   └── tp4_llama70b/                   — Tab 26 (70B TP=4 B-sweep)
    │
    ├── system_pareto/
    │   ├── pareto_8b_graph/                — Tab 18/19 (graph)
    │   ├── pareto_70b/                     — Tab 27 (eager)
    │   └── pareto_70b_graph/               — Tab 20 (graph)
    │
    ├── system_prefill/                     — Tab 12/14 (prefill + capacity)
    │   ├── {FA2,CDA}_N{8K,16K,32K,64K,131K}.json
    │   └── maxctx_{FA2,CDA}_util*.json
    │
    ├── system_energy/                      — Tab 21/22 (J/tok)
    │   ├── energy/                         —   full sweep (B ∈ {1,16}, N ∈ 4K..32K)
    │   └── {CDA,FA2}_B*_N*.json            —   aggregated anchors
    │
    ├── composability_awq/                  — Tab 23/24 (AWQ × CDA)
    │   ├── {FP16,AWQ}_{FA2,CDA}.json       —   8B (4 cells)
    │   └── awq_70b/                        —   70B (4 cells)
    │
    └── misc/                               — HMMA↔FA2 MSE sanity
```

## Provenance of baseline numbers (IMPORTANT)

| Baseline              | Measurement code                                                   | Source       |
|-----------------------|--------------------------------------------------------------------|--------------|
| FP16 reference        | HuggingFace `transformers` forward                                 | official     |
| FA2 kernel            | `vllm_flash_attn.flash_attn_varlen_func` (vLLM 0.19)               | official     |
| **KIVI** (PPL+kernel) | `REF/KIVI` — `LlamaForCausalLM_KIVI` + `cuda_bmm_fA_qB_outer`      | **official** |
| **GEAR** (PPL)        | `REF/GEAR` — `compress_method=GEAR`, group=64, rank=2              | **official** |
| **KVQuant** (kernel)  | `REF/KVQuant/deployment/kvquant/quant_cuda_kernel.cu` (fused NUQ)  | **official** |
| **QuaRot** (PPL+kern) | `REF/QuaRot/fake_quant/main.py` + `qattention_benchmark.py`        | **official** |
| **TurboQuant**$^\dagger$ (PPL) | `turboquant_ours_reimpl/` — our PyTorch re-implementation   | **ours**     |
| **CDA (ours)**        | `code/experiments/bench_*.py`                                      | **ours**     |

$^\dagger$ TurboQuant has no public PyTorch evaluation harness; we
cross-verified our re-implementation against the authors' reported
numbers and in-place operators in `REF/turboquant-pytorch`.

**Only TurboQuant and CDA rows in Table 2 are our code.** All other
baselines run from their official repositories at the commits documented
below. We do **not** redistribute official-code checkouts — follow the
install instructions for each one.

### Reproducing baseline runs

| Repo    | URL                              | Commit | Install                                                    |
|---------|----------------------------------|--------|------------------------------------------------------------|
| KIVI    | `github.com/jy-yuan/KIVI`        | main   | follow their README; env `kivi` (transformers 4.44.2)      |
| GEAR    | `github.com/HaoKang-Timmy/GEAR`  | main   | env shared with `kivi`                                     |
| KVQuant | `github.com/SqueezeAILab/KVQuant`| main   | env shared with `cda`                                      |
| QuaRot  | `github.com/spcl/QuaRot`         | main   | env `quarot` (transformers 4.36.2); see `docs/QUAROT_ENV_SETUP.md` |

## Environment

- Python 3.10, PyTorch 2.10.0+cu128, CUDA 12.8 toolkit, Llama tokenizer
- vLLM 0.19.0 + FlashInfer 0.6.6, transformers 4.57.6 (for our code)
- GPU: NVIDIA RTX A6000 (48 GB, sm_86); 2 GPUs locally, 4 GPUs on lab node
- **CUDA kernels load from pre-built `.so`** under `code/core/prebuilt_hmma/`
  and `code/core/_cda_gqa_kernels.*.so`. No CUDA source is shipped.
- A minimal end-to-end run requires CUDA, the ~10 GB Llama-3.1-8B-Instruct
  checkpoint (from Hugging Face Hub), and ~48 GB GPU VRAM.

## Environment toggles

| Env var                    | Default | Effect                                                                            |
|----------------------------|:-------:|-----------------------------------------------------------------------------------|
| `CDA_ENABLE_MEMORY_SAVING` | unset   | Activates vLLM `CUSTOM` paged-KV backend with 4.92× shrunken slot                 |
| `CDA_ENABLE_CUDAGRAPH`     | unset   | Flips `_cudagraph_support` → UNIFORM\_SINGLE\_TOKEN\_DECODE; routes to graphable kernels |
| `CDA_GRAPH_MAX_B`          | 256     | Max batch for pre-allocated graph scratch (set 32 for 70B TP=4 memory budget)     |
| `CDA_GRAPH_MAX_SPLITS`     | 128     | Max `num_splits` in scratch                                                       |
| `CDA_PREFILL_BATCHED`      | unset   | Batch all per-request `_dequantize_past` calls in one compressor invocation       |

## Reproducing each paper claim

Every script below writes its output JSON under `data/<subdir>/`. The
file names match what the paper's figure / table generators consume.

### Table 2 — WikiText-2 PPL (7 LLMs)

```bash
# CDA (ours) — K4V2 across models
python code/experiments/bench_ppl_scale.py --model llama8b --dataset wikitext2
# ... per-model; see comments inside the script
```
Paired raw outputs in `data/accuracy_ppl/ppl_scale_*/`.

### Appendix D — C4 cross-check

```bash
python code/experiments/bench_ppl_scale.py --model llama70b --dataset c4
```
Paired raw outputs in `data/accuracy_ppl/ppl_scale_*_c4*/`.

### Tables 4 + 5 — kernel microbench (same-compression + gold-baselines)

```bash
# Tab 4 (same-compression isolation)
python code/experiments/bench_dequant_vs_cda.py       # legacy v1/v2
python code/experiments/bench_dequant_vs_cda_v3.py    # final HMMA v3 cells
# Tab 5 (gold baselines) — official kernels from their repos; see header
python code/experiments/bench_quarot_official.py
python code/experiments/bench_kvquant_official.py
python code/experiments/bench_kivi_official.py
# Aggregated summary:
#   data/kernel_bench/GOLD_baselines_kernel_summary.json
```

### Figure 2(c) — graph-mode full B-sweep (5 N × 5 curves)

```bash
# Run both backends in parallel on 2 GPUs
CUDA_VISIBLE_DEVICES=0 python code/experiments/bench_tput_sweep_graph.py \
    --backend CDA         --outdir data/system_vllm/tput_graph &
CUDA_VISIBLE_DEVICES=1 python code/experiments/bench_tput_sweep_graph.py \
    --backend FLASH_ATTN  --outdir data/system_vllm/tput_graph &
wait
```

### Table 8 — 8B throughput cliff (5 anchors)

```bash
CUDA_VISIBLE_DEVICES=0 python code/experiments/bench_tput_cliff_graph.py --backend FLASH_ATTN
CUDA_VISIBLE_DEVICES=1 python code/experiments/bench_tput_cliff_graph.py --backend CDA
# N=128K uses a separate script due to max_model_len cap:
python code/experiments/bench_tput_cliff_n128k.py --backend FLASH_ATTN
python code/experiments/bench_tput_cliff_n128k.py --backend CDA
```

### Appendix G — 8B TTFT/TPOT Pareto (graph vs eager)

```bash
# 8B graph-mode Pareto (Tab 18/19)
CUDA_VISIBLE_DEVICES=0 python code/experiments/bench_pareto_8b_graph.py
# 8B + 70B eager Pareto (Tab 27 70B; 8B CDA eager column in Tab 19)
CUDA_VISIBLE_DEVICES=0 bash code/experiments/bench_pareto_ttft_tpot.sh FA2
CUDA_VISIBLE_DEVICES=0 bash code/experiments/bench_pareto_ttft_tpot.sh CDA
```

### Appendix E — prefill + capacity (Tables 12/13/14)

```bash
# 8B prefill latency
for N in 8192 16384 32768 65536 131064; do
  for BK in FA2 CDA; do
    python code/experiments/bench_prefill_memory.py \
        --backend $BK --N $N --output data/system_prefill/${BK}_N${N}.json
  done
done
# Max-context capacity sweep
for util in 0.35 0.5; do
  for BK in FA2 CDA; do
    python code/experiments/bench_max_context.py \
        --backend $BK --gpu-util $util --output data/system_prefill/maxctx_${BK}_util${util}.json
  done
done
# 70B TP=4 prefill+energy (lab-server 4× A6000)
python code/experiments/bench_70b_prefill_energy.py
```

### Appendix H — energy/token (Tables 21/22)

```bash
bash code/experiments/bench_energy_sweep.sh
# or individual points:
python code/experiments/bench_energy_nvml.py --backend CDA --B 16 --N 32768 \
    --iters 3 --output data/system_energy/energy/CDA_B16_N32K.json
```

### Appendix I — AWQ × CDA composability (Tables 23/24)

```bash
# 8B (4 configs)
for W in FP16 AWQ; do for A in FA2 CDA; do
  python code/experiments/bench_awq_cda_ppl.py \
      --size 8b --weights $W --attn $A \
      --output data/composability_awq/${W}_${A}.json
done; done
# 70B TP=4 (4 configs; lab-server)
for W in FP16 AWQ; do for A in FA2 CDA; do
  CUDA_VISIBLE_DEVICES=0,1,2,3 python code/experiments/bench_awq_cda_ppl.py \
      --size 70b --weights $W --attn $A \
      --output data/composability_awq/awq_70b/${W,,}_${A,,}_70b.json
done; done
```

### Table 9 / Table 28 — ablations (8B and 70B)

```bash
# 8B (Tab 9)
python code/experiments/ablation_hadamard.py
python code/experiments/ablation_quarot_style.py
python code/experiments/ablation_bitwidth.py
# 70B TP=4 (Tab 28)
CUDA_VISIBLE_DEVICES=0,1,2,3 python code/experiments/ablation_hadamard_70b.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python code/experiments/ablation_quarot_style_70b.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python code/experiments/ablation_bitwidth_70b.py
```

### Appendix C — RULER 13-task (8B)

```bash
bash code/experiments/run_ruler_minimal.sh
```
Raw outputs in `data/accuracy_long_context/ruler_min/` (FP16 + CDA K4V2).

### Kernel variants + validation

```bash
# Flash-LUT vs HMMA variants (Tab 15)
python code/experiments/bench_flash_lut_variants.py
python code/experiments/bench_hmma_production.py
# Tab 4 / 25 HMMA 70B g8 cells
python code/experiments/bench_hmma_g8_table4.py
# Kernel correctness vs FP16 FA2
python code/experiments/bench_hmma_vs_fa2_mse.py
python code/experiments/test_hmma_production.py
```

## Supplementary policy

- **Code release is pre-built `.so` binaries only.** The Python bindings
  in `code/core/cda_attn.py` delegate every CUDA entry point to
  `code/core/_load_prebuilt.py`. The same `.so` files produced all paper
  numbers.
- **Data release is raw JSON** from the benchmark scripts. Figure
  generators and LaTeX tables consume the JSONs directly; no
  post-processing is hidden.

Last refreshed: 2026-04-22.
