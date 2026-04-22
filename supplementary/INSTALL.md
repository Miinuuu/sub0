# Running the supplementary code

Target system: **NVIDIA RTX A6000 (sm_86), CUDA 12.8, Python 3.10,
PyTorch 2.10.0+cu128**. The bundled `.so` binaries are compiled for
exactly this ABI. For any other combination, contact the authors for a
rebuild — CUDA source is not redistributed.

## 1. Environment

```bash
conda create -n cda python=3.10 -y
conda activate cda

# PyTorch 2.10 + CUDA 12.8 (exact wheel required to match our .so ABI)
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# Remaining deps
pip install -r supplementary/requirements.txt
```

Optional, only for Appendix C (RULER):
```bash
pip install git+https://github.com/EleutherAI/lm-evaluation-harness@main
```

## 2. Model weights

Default model: `meta-llama/Llama-3.1-8B-Instruct` (HF Hub). Appendix
tables also use `meta-llama/Meta-Llama-3-70B-Instruct`,
`meta-llama/Llama-2-{7b,13b,70b}-hf`, `Qwen/Qwen2.5-32B`, and
`mistralai/Mistral-7B-v0.1`. Appendix I uses
`hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` and a 70B AWQ
counterpart. Log in with `huggingface-cli login` if you have not already.

## 3. Path setup

Every script locates `core/` relative to itself via `Path(__file__).parents`.
You can run from anywhere — just call the script by its path:

```bash
cd supplementary/
python code/experiments/bench_hmma_vs_fa2_mse.py
bash   code/experiments/bench_pareto_ttft_tpot.sh FA2
```

Shell scripts resolve `$REPO` to `supplementary/` automatically.

## 4. Sanity check (10 seconds)

Verify the pre-built kernels load:

```bash
python -c "
import sys
from pathlib import Path
sys.path.insert(0, 'supplementary/code')
from core._load_prebuilt import load_hmma_v3, load_hmma_g8, load_hmma, load_fused_update, load_fused_reduce_rot
for fn in (load_hmma, load_hmma_v3, load_hmma_g8, load_fused_update, load_fused_reduce_rot):
    mod = fn()
    print(f'  OK — {fn.__name__}() loaded ({mod})')
from core import cda_attn as _   # triggers _get_gqa() eager load of the flash .so
print('OK — all 5 HMMA/fused kernels + flash kernels loaded')
"
```

Then run the numerical fidelity bench (no model download needed):

```bash
python supplementary/code/experiments/bench_hmma_vs_fa2_mse.py
```

Expected runtime: ~3 minutes on a single A6000. Output JSON goes to
`data/misc/hmma_vs_fa2_mse.json` (creates the folder on first run).

## 5. Reproducing each paper table / figure

See `README.md` for per-appendix commands.

## Troubleshooting

- **`ImportError: Pre-built kernel missing`** — the `.so` file's ABI
  doesn't match your Python / PyTorch. Verify versions with
  `python -c "import torch; print(torch.__version__)"` (expected
  `2.10.0+cu128`).
- **`CUDA error: no kernel image is available for execution`** — your
  GPU is not sm_86. The bundled `.so` targets A6000 only; contact the
  authors for a sm_80 / sm_89 / sm_90 rebuild.
- **`ModuleNotFoundError: No module named 'core'`** — set
  `PYTHONPATH=supplementary/code` or run scripts by path (not module).
- **`undefined symbol: c10_cuda_check_implementation`** — handled
  internally by `_load_prebuilt._preload_torch_libs()` (forces libc10 /
  libc10_cuda with RTLD_GLOBAL before dlopening our `.so`). If you still
  see this, check that PyTorch's `lib/` directory is readable.
- **vLLM hangs on first run** — vLLM JIT-compiles its own attention
  backends; first startup can take 2–3 minutes.
- **70B runs require TP=4** — set `CUDA_VISIBLE_DEVICES=0,1,2,3` and
  pass `tensor_parallel_size=4` to vLLM. Capacity-bound scripts may also
  need `gpu_memory_utilization=0.85` or lower.
