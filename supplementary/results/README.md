# Paper backing data

Every numeric Table/Figure in the paper has a backing JSON/log under
`runs/paper/`, organized by paper artifact.

## Layout

```
runs/paper/
├── table2_compute_path_iso/        # Table 2 — kernel decode latency CDA vs dequant+FA2
│   └── compute_path_isolation_cuda_graph.json
├── table3_cross_baseline/          # Table 3 — cross-baseline kernel μs (fill-in cells)
│   └── cross_baseline_table3_refill.json
├── table4_ppl/                     # Table 4 — WikiText-2 PPL across 6 models
│   ├── llama8b/      (Llama-3.1-8B  : FP16 7.1216, CDA 7.335)
│   ├── llama70b/     (Llama-3.1-70B : FP16 2.3051, turbo-4 2.2989)
│   ├── llama2-7b/    (Llama-2-7B    : FP16 5.3997, turbo-4 5.4256)
│   ├── llama2-13b/
│   ├── llama2-70b/   (Llama-2-70B   : FP16 2.6042, turbo-4 2.5945)
│   ├── mistral7b/    (Mistral-7B    : FP16 5.3770, turbo-4 5.3867)
│   └── qwen32b-inst/ (Qwen-2.5-32B  : FP16 2.3031, turbo-4 2.3474)
├── table5_ruler/                   # Table 5 — RULER long-ctx Macro / NIAH @ {8K,16K,32K}
│   ├── 8k/{fa2,cda}/results_*.json
│   ├── 16k/{fa2,cda}/results_*.json
│   └── 32k/{fa2,cda}/results_*.json
├── tableA3_fig2_throughput/        # Fig 2 + Table A3 — vLLM throughput sweep (decode-bound)
│   ├── 4K/    (P=1024  D=3072)
│   ├── 16K/   (P=4096  D=12288)
│   ├── 64K/   (P=61440 D=4096)
│   └── 128K/  (P=126976 D=4096)
├── tableA4_tp4_70b/                # Table A4 — Llama-3.1-70B TP=4 serving
│   └── N{16K,64K}_b*_{cda,fa2}*.json
└── tableA5_rotation_ablation/      # Table A5 — rotation ablation (4 rows × 5 seeds)
    └── abl_rotation_multiseed.json
```

## Claim → file map

| Paper artifact | Reported number | File | Cell / key |
|---|---|---|---|
| Table 2 (B=1, N=4K)   | dequant=0.100ms / CDA=0.042ms / ratio=0.42×  | `table2_compute_path_iso/compute_path_isolation_cuda_graph.json` | results[0] |
| Table 2 (B=1, N=64K)  | dequant=1.360ms / CDA=0.290ms / ratio=0.21×  | same | results[1] |
| Table 2 (B=4, N=64K)  | dequant=5.554ms / CDA=0.984ms / ratio=0.18×  | same | results[2] |
| Table 2 (B=32, N=16K) | dequant=11.630ms / CDA=1.871ms / ratio=0.16× | same | results[3] |
| Table 3 PolarQuant B=4,N=64K  | 9{,}336 us  | `table3_cross_baseline/cross_baseline_table3_refill.json` | results[0] |
| Table 3 PolarQuant B=32,N=16K | 18{,}656 us | same | results[1] |
| Table 3 CommVQ B=4,N=64K      | 92{,}080 us | same | results[2] |
| Table 3 CommVQ B=32,N=16K     | 188{,}024 us | same | results[3] |
| Table 4 Llama-3.1-8B  | FP16 7.12, CDA 7.33 | `table4_ppl/llama8b/summary.json` | results.wikitext2.{fp16,cda-4}.ppl |
| Table 4 Llama-3.1-70B | FP16 2.31, dequant+FA2 2.30 | `table4_ppl/llama70b/summary.json` | results.wikitext2.{fp16,turbo-4}.ppl |
| Table 4 Llama-2-7B    | FP16 5.40, dequant+FA2 5.43 | `table4_ppl/llama2-7b/summary.json` | same |
| Table 4 Llama-2-70B   | FP16 2.60, dequant+FA2 2.59 | `table4_ppl/llama2-70b/summary.json` | same |
| Table 4 Mistral-7B    | FP16 5.38, dequant+FA2 5.39 | `table4_ppl/mistral7b/summary.json` | same |
| Table 4 Qwen-2.5-32B  | FP16 2.30, dequant+FA2 2.35 | `table4_ppl/qwen32b-inst/summary.json` | same |
| Table 5 8K Macro/NIAH  | FA2 0.945/0.998 ; CDA 0.937/0.993 | `table5_ruler/8k/{fa2,cda}/results_*.json` | results.<task>."8192,none" |
| Table 5 16K Macro/NIAH | FA2 0.932/0.995 ; CDA 0.923/0.992 | `table5_ruler/16k/...`  | results.<task>."16384,none" |
| Table 5 32K Macro/NIAH | FA2 0.864/0.993 ; CDA 0.854/0.987 | `table5_ruler/32k/...`  | results.<task>."32768,none" |
| Fig 2 / Table A3 (out_tps) | per cell tok/s | `tableA3_fig2_throughput/{4K,16K,64K,128K}/*.json` | out_tps = batch * decode_len / wall_s |
| Table A4 70B TP=4      | per (N,B,backend) tok/s | `tableA4_tp4_70b/N{16K,64K}_b*_{cda,fa2}*.json` | throughput_tok_s |
| Table A5 rotation      | identity 9.34 / hadamard 7.33 / random_orth 7.18±0.08 | `tableA5_rotation_ablation/abl_rotation_multiseed.json` | identity, hadamard, random_orthogonal_seeds |

For RULER scoring: **Macro** = mean over 13 subtasks
(`niah_single_{1,2,3}`, `niah_multikey_{1,2,3}`, `niah_multiquery`,
`niah_multivalue`, `ruler_cwe`, `ruler_fwe`, `ruler_qa_hotpot`,
`ruler_qa_squad`, `ruler_vt`); **NIAH** = mean over the 8 NIAH subtasks.
Per-context score lives at `results.<task>."<ctx>,none"` in each
results_*.json.

## Verification

All listed numbers were verified against the paper TeX
(`papers/69cf3c869b7b5e6e5a202759/neurips/`):

| Artifact | match |
|---|---|
| Table 2 (4 cells) | 100% EXACT |
| Table 3 fill-in (4 cells) | 100% EXACT |
| Table 4 (12 numbers across 6 models) | 100% EXACT |
| Table 5 (12 numbers, 3 ctx × 2 backend × 2 metric) | 100% EXACT |
| Table A3 (sampled 10 cells) | 9/10 EXACT (1 cell ±0.9 tok/s) |
| Table A5 (4 rows × seeds) | 100% EXACT |

## Provenance

- Tables 2 / 3 / A3 / A4 / A5 produced by the in-repo bench scripts on
  RTX A6000 (CUDA 12.8, PyTorch 2.10).
- Table 4 PPL evaluated under the canonical protocol (stride=64,
  max_ctx=4096, n_positions=63, filter_empty_lines=True,
  add_special_tokens=True). The `dequant+FA2` column is the
  TurboQuant-4 (`turbo-4`) decode path on the same K4V4 cache.
- Table 5 RULER evaluated via lm-evaluation-harness
  (`lm_eval --model vllm --tasks niah_* ruler_*`) on
  Llama-3.1-8B-Instruct, in line with the paper's lm-eval setup.

## Re-running

The bench scripts that produced each artifact are listed in
`benchmarks/baselines/README.md` and `supplementary/reproduce.sh`.
Paper Table 3's KIVI / FA2-FP16 / CDA columns and Table A4's full
TP=4 70B run require the original development environment plus
`REF/*` reference implementations.
