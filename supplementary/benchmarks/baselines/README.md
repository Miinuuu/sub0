# Baseline benchmark scripts (reference-only)

These scripts measure baseline kernels referenced by the paper. They
are kept for **provenance / reference**; most are not runnable in the
cda-v2 tree alone because they depend on the vendored baseline
implementations under `REF/` and (for Table 4) the legacy `cda.eval`
module that was never ported from cda-v1.

## Paper artifact map

| Script | Paper artifact | Native package | Backing data |
|---|---|---|---|
| `bench_kivi_official.py` | Table 3 KIVI col / Table 4 KIVI rows | `kivi_gemv`, `quant.matmul` (`REF/KIVI`) | (see Table 3 fill-in JSON) |
| `bench_commvq_official.py` | Table 3 CommVQ col | `commvq.triton_kernels` (`REF/CommVQ`) | `runs/paper/cross_baseline_table3_refill.json` |
| `bench_qjl_official.py` | Table 3 QJL col | `models.llama3_utils_qjl` (`REF/QJL`) | `runs/paper/cross_baseline_table3_refill.json` |
| `bench_polarquant_official.py` | Table 3 PolarQuant col | `models.kernel4group` (`REF/PolarQuant`) | `runs/paper/cross_baseline_table3_refill.json` |
| `bench_quarot_official.py` | Table A2 QuaRot (appendix native MHA) | `quarot.transformers.kv_cache` (`REF/QuaRot`) | (appendix) |
| `bench_kvquant_official.py` | Table A2 KVQuant (appendix native MHA) | `quant_cuda` (`REF/KVQuant`) | (appendix) |
| `bench_kivi_ppl.py`, `bench_ppl_kivi_official.py` | Table 4 KIVI PPL rows | `cda.eval.ppl` (v1 only) + `REF/KIVI` | `runs/baselines/v1_paper/ppl_scale/` |
| `bench_ppl_scale.py` | Table 4 (FP16 / KIVI / GEAR / QuaRot / TurboQuant PPL) | `cda.eval.ppl` (v1 only) | `runs/baselines/v1_paper/ppl_scale/` |
| `bench_longbench.py` | Table 5 reference (LongBench single-task) | `cda.eval.cda_decode` (v1 only) | `runs/baselines/v1_paper/longbench/` |

## Runtime status

- **Table 3 baseline kernels** (`bench_*_official.py` for KIVI / CommVQ /
  QJL / PolarQuant): each script imports its baseline's native Python
  package. To run, install the corresponding `REF/*` tree as an
  editable package or set up its conda env. The QJL paper data was
  collected in a separate `qjl` conda env with `torch 2.4.1+cu121`;
  CommVQ / PolarQuant were collected in the main `cda` env with
  `torch 2.10.0+cu128`.
- **Table A2** (`bench_quarot_official.py`, `bench_kvquant_official.py`):
  same pattern — install `REF/QuaRot` or `REF/KVQuant` first.
- **Table 4 PPL** (`bench_ppl_scale.py`): runnable in the cda env. The
  required `cda.eval.ppl` module ships with v2 (WikiText-2 / C4 text
  fetchers built on the `datasets` library). FP16 / CDA-{2,4} /
  QuaRot-style methods run from the in-tree `cda.algorithm.Compressor`
  alone. KIVI methods need `transformers.cache_utils.QuantoQuantizedCache`
  (transformers ≥ 4.45). GEAR methods need `REF/GEAR/...compress_function.py`
  on disk. The paper's reported numbers are also preserved under
  `runs/baselines/v1_paper/ppl_scale/` for direct comparison.
  (`bench_kivi_ppl.py`, `bench_ppl_kivi_official.py` are KIVI-specific
  variants that still import a v1 KIVI eval driver — kept as reference.)
- **Table 5 RULER** (`bench_longbench.py`): runnable in the cda env.
  Wraps `lm-evaluation-harness` (`lm_eval.simple_evaluate` Python API)
  to run the 13 RULER subtasks (8 NIAH + 5 RULER) on
  Llama-3.1-8B-Instruct under either FA2 or CDA, then aggregates Macro
  (mean over 13) and NIAH (mean over 8) into the paper Table 5 format.
  CDA backend requires the cda-v2 vLLM plugin (`VLLM_PLUGINS=cda_v2`)
  and the K4V4 KV-cache page-size patch (`CDA_V2_ENABLE_MEMORY_SAVING=1`)
  to reach every vLLM worker process; the script sets both env vars
  automatically when ``--backend CDA`` is passed.

## Paper provenance

The CDA column of Table 3 and the dequant+FA2 / FA2_fp16 columns of the
same table are produced by the in-repo `benchmarks/bench_kernel_head_to_head.py`
(no `cda.eval` dependency, runs in the `cda` env). The cross-baseline
columns (KIVI / CommVQ / QJL / PolarQuant) come from the per-baseline
official scripts in this directory; the paper's reported numbers are
preserved under `runs/paper/cross_baseline_table3_refill.json`.

To re-run a baseline cell, install the matching `REF/*` package and
invoke the corresponding script directly. To re-run all of Table 4 or
Table 5, restore the v1 codebase or the lm-evaluation-harness setup
referenced above; the in-repo scripts here are reference only.
