# Supplementary Material — Compressed-Domain Attention (CDA)

Anonymized supplementary distribution for *"Compressed-Domain Attention:
Tensor-Core Decode for 4-Bit Codebook KV Caches"* (NeurIPS 2026).

This package supports **end-to-end reproduction** of every paper Table
and Figure that has CDA-measured numbers, using the same compiled CUDA
artifacts and Python entry points that produced the reported values.
CUDA source files are not redistributed — the prebuilt ``.so``
artifacts under ``cda/_prebuilt/`` are the verbatim binaries used for
the paper.

## Layout

```
supplementary/
├── README.md / INSTALL.md / reproduce.sh / LICENSE / pyproject.toml
├── cda/
│   ├── __init__.py
│   ├── _prebuilt_loader.py        # importlib + torch.ops.load_library wrappers
│   ├── _prebuilt/                 # 6 paper-essential CUDA artifacts (~52 MB)
│   ├── algorithm/                 # compression.py + attention.py (rotated-frame ref)
│   ├── eval/                      # canonical PPL / dataset fetchers (Table 4)
│   ├── kernels_cuda/              # decode/fa2_cda wrappers + cda_attn_v2 backend
│   ├── kernels_vllm_fa2_fork/     # FA2 fork interface; build.py is a prebuilt stub
│   ├── vllm_integration/          # vLLM backend registration
│   └── vllm_plugin.py             # cda_v2 plugin entry point
├── benchmarks/
│   ├── bench_compute_path_iso_cuda_graph.py   # Table 2
│   ├── bench_kernel_head_to_head.py           # Table 3 CDA + FA2 columns
│   ├── paper/
│   │   ├── bench_capacity_sweep.py            # Fig 2 + Table A3 + Table A4 (with --tp)
│   │   ├── bench_capacity_grid.py
│   │   ├── abl_rotation_multiseed.py          # Table A5 (requires CDA_V1_PATH)
│   │   ├── gen_fig1_4zone.py                  # Fig 1
│   │   ├── gen_fig2_throughput.py             # Fig 2 plot
│   │   ├── gen_tab_fig2_raw.py                # Table A3
│   │   ├── gen_tab_memory.py
│   │   └── format_paper_tables.py
│   └── baselines/
│       ├── bench_ppl_scale.py                 # Table 4 (FP16/CDA-{2,4}/QuaRot directly)
│       ├── bench_longbench.py                 # Table 5 RULER (lm-eval-harness wrapper)
│       ├── bench_kivi_official.py             # Table 3 KIVI col (requires REF/KIVI)
│       ├── bench_commvq_official.py           # Table 3 CommVQ col (REF/CommVQ)
│       ├── bench_qjl_official.py              # Table 3 QJL col (REF/QJL + qjl env)
│       ├── bench_polarquant_official.py       # Table 3 PolarQuant col (REF/PolarQuant)
│       ├── bench_quarot_official.py           # Table A2 QuaRot (REF/QuaRot)
│       ├── bench_kvquant_official.py          # Table A2 KVQuant (REF/KVQuant)
│       └── README.md                          # per-script REF/* dependencies
└── results/                        # paper backing data (anonymized)
    ├── compute_path_iso/  kernel_head_to_head/
    ├── ppl_scale/  ruler/
    ├── throughput/  tp4_70b/
    └── ablation_rotation/
```

### Paper-essential kernels (the 6 prebuilt .so)

| File                                    | Source kernel               | Used by                                  |
|-----------------------------------------|-----------------------------|------------------------------------------|
| `_v1_decode_hmma_k4v4_gs4.so`           | `_decode_hmma.cu`           | Table 2/3 + Fig 2/Table A4 decode        |
| `_v1_paged_encode_fused.so`             | `_paged_encode.cu`          | KV-cache encode                          |
| `_v1_fa2_cda_loader_k4v4.so`            | `_fa2_cda_loader.cu`        | vLLM prefill K/V materialization         |
| `_v1_fa2_cda_varlen_fused.so`           | `_fa2_cda_varlen_fused.cu`  | vLLM prefill (Hadamard epilogue)         |
| `_v1_fa2_cda_hadamard128.so`            | `_fa2_cda_hadamard.cu`      | Hadamard helper                          |
| `_cda_vllm_fa2_fork_C.so`               | vLLM-FA2 fork (TORCH_LIB)   | Cross-baseline (Tables 2/3) + vLLM prefill |

## Environment

The prebuilt ``.so`` artifacts have a fixed ABI:

- **Python**: 3.10 (cp310)
- **PyTorch**: 2.10.0 + CUDA 12.8
- **GPU**: NVIDIA Ampere SM86 (RTX A6000)
- **Optional for serving / RULER**: vLLM v0.19, lm-eval-harness ≥ 0.4.5
- **Optional for Table 4 PPL**: `transformers` ≥ 4.45, `datasets`

Loading on a non-matching environment raises ``ImportError``.

## Quick start

```bash
# Activate matching env, then
cd supplementary/
pip install -e ".[post]"           # numpy + matplotlib for figures
PYTHON=python bash reproduce.sh smoke
```

## Reproduction modes (`reproduce.sh <mode>`)

| Mode      | Time   | What it produces                                                             |
|-----------|--------|------------------------------------------------------------------------------|
| `smoke`   | ~10 s  | Table 2 single cell (B=1, N=4K)                                              |
| `table2`  | ~3 min | Table 2 full sweep B∈{1,4,32}, N∈{4K,16K,64K}                                |
| `table3`  | ~2 min | Table 3 CDA + Lloyd FA2-fork + dequant+FA2 + FA2_fp16 columns                |
| `fig2`    | ~1 min | Fig 2 / Table A3 single (B=1, N=4K, vLLM)                                    |
| `tp2`     | ~2 min | Table A4 proxy: TP=2 on 2 GPUs (paper used TP=4 on 4)                        |
| `table4`  | ~2 min | Table 4 PPL smoke (Mistral-7B, FP16+CDA, max_ctx=512)                        |
| `table5`  | ~2 min | Table 5 RULER smoke (FA2 niah_single_1, max_ctx=4096)                        |
| `figures` | ~5 s   | Regenerate Fig 1, Fig 2, Table A3 from `results/`                            |

## Claim → reproduction map

| Paper artifact                | Backing data                                  | How to reproduce                                                                           |
|-------------------------------|-----------------------------------------------|--------------------------------------------------------------------------------------------|
| Table 2 (compute-path iso.)   | `results/compute_path_iso/`                   | `bash reproduce.sh smoke` (1 cell) or `table2` (full sweep)                                |
| Table 3 (kernel head-to-head) | `results/kernel_head_to_head/`                | `bash reproduce.sh table3` for the 4 in-tree columns; KIVI/CommVQ/QJL/PolarQuant cols need `benchmarks/baselines/bench_*_official.py` + REF/* (see `benchmarks/baselines/README.md`) |
| Table 4 (WikiText-2 PPL)      | `results/ppl_scale/{model}/*.json`            | `bash reproduce.sh table4` (Mistral-7B sample) or full: `bench_ppl_scale.py --model llama8b --methods fp16 cda-4 ...` |
| Table 5 (RULER long-ctx)      | `results/ruler/{longbench,niah}/`             | `bash reproduce.sh table5` (smoke) or full: `bench_longbench.py --backend {FA2,CDA} --max-model-len {8192,16384,32768}` |
| Fig 2 + Table A3 (throughput) | `results/throughput/8b_a6000/`                | `bash reproduce.sh fig2` (single cell) or full: `bench_capacity_sweep.py --batches 1 2 4 8 ...`                  |
| Table A4 (TP=4 70B serving)   | `results/tp4_70b/`                            | `bash reproduce.sh tp2` (TP=2 proxy on 2 GPUs); paper config: `bench_capacity_sweep.py --tp 4 --gpu 0,1,2,3 --model meta-llama/Llama-3.1-70B-Instruct` |
| Table A5 (rotation ablation)  | `results/ablation_rotation/*.json`            | Backing JSON only; `abl_rotation_multiseed.py` requires the legacy v1 codebase (set `CDA_V1_PATH` env var) |
| Fig 1 / Fig 2 plots / Table A3 | (post-process from `results/`)               | `bash reproduce.sh figures`                                                                |

## Reproduction limitations (honest)

- **ABI lock**: prebuilt artifacts are SM86 / cp310 / cu128. Other
  environments cannot load them.
- **Table 3 baseline cols**: KIVI / CommVQ / QJL / PolarQuant scripts
  ship as references but require their respective `REF/*` reference
  implementations to run. Paper-time `runs/paper/cross_baseline_table3_refill.json`
  is preserved for verification.
- **Table A4**: paper's TP=4 on Llama-3.1-70B requires 4× A6000.
  ``reproduce.sh tp2`` runs the same code path on TP=2 + 8B as a
  smaller proxy.
- **Table A5**: rotation ablation depends on the legacy v1 codebase
  (Compressor v1 + ppl_eval). Backing JSONs are preserved.
- **Anonymization**: author identifiers and absolute paths have been
  removed from every shipped file.

## License

See `LICENSE`.
