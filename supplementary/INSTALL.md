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

All experiments use `meta-llama/Llama-3.1-8B-Instruct` (HF Hub) and, for
Appendix I, `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`. Log in
with `huggingface-cli login` if you have not already.

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

```python
python -c "
import sys
from pathlib import Path
sys.path.insert(0, 'supplementary/code')
from core.cda_attn import choose_tile_n_hmma
from core._load_prebuilt import load_hmma_v3
load_hmma_v3()
print('OK — HMMA v3 loaded')
"
```

Then run the numerical fidelity bench (no model download needed):

```bash
python supplementary/code/experiments/bench_hmma_vs_fa2_mse.py
```

Expected runtime: ~3 minutes on a single A6000. Output JSON goes to
`runs/hmma_vs_fa2_mse.json` (creates the folder on first run).

## 5. Reproducing each paper table / figure

See `README.md` for per-appendix commands.

## Troubleshooting

- **`ImportError: Pre-built kernel missing`** — the `.so` file's ABI
  doesn't match your Python / PyTorch. Verify versions with
  `python -c "import torch; print(torch.__version__)"`.
- **`CUDA error: no kernel image is available for execution`** — your
  GPU is not sm_86. The bundled `.so` targets A6000 only.
- **`ModuleNotFoundError: No module named 'core'`** — set
  `PYTHONPATH=supplementary/code` or run scripts by path (not module).
- **vLLM hangs on first run** — vLLM JIT-compiles its own attention
  backends; first startup can take 2–3 minutes.
