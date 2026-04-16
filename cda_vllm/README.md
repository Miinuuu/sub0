# CDA — vLLM Paged-Attention Variant

## Install

This package lives inside the main `cda_submission` repository. Install the parent package first, then this sub-package:

```bash
# from the cda_submission root
pip install -e .
pip install -e cda_vllm_submission
```

The main `cda` package is listed as a runtime dependency; `HadamardQuantCompressor` from there is used to prepare compressed inputs for the paged kernel.

---

## Repository layout (shipped)

```
cda_vllm_submission/
├── README.md
├── requirements.txt
├── setup.py                                 # CUDAExtension build
├── pyproject.toml
│
├── cda_vllm/
│   ├── __init__.py                          # Public API (source)
│   ├── paged_attention.py                   # Binding wrapper (source)
│   └── _cda_paged_kernels.cpython-310-x86_64-linux-gnu.so   # (binary, closed)
│
├── experiments/
│   └── benchmark_paged.py                   # Dense vs sparse throughput sweep
│
└── tests/
    └── test_paged.py                        # Paged vs PyTorch ref smoke test
```

---

## Quick start

```python
import math, torch
from cda import HadamardQuantCompressor
from cda_vllm import cuda_cda_paged

D, N, B, block_size = 128, 256, 2, 16
device = torch.device("cuda:0")

# Quantize K/V (flat, contiguous).
comp = HadamardQuantCompressor(dim=D, bit_width=2)
K_flat = torch.randn(B, N, D, device=device)
V_flat = torch.randn(B, N, D, device=device)
cK = comp.quantize(K_flat.reshape(-1, D))
cV = comp.quantize(V_flat.reshape(-1, D))

# Shuffle into paged layout — in production this is done by the vLLM block
# manager; here we construct it manually. See tests/test_paged.py.
# ...

# One fused paged attention step.
out = cuda_cda_paged(
    Q_rot, packed_K_blocks, norms_K_blocks,
    packed_V_blocks, norms_V_blocks, block_tables,
    codebook_k, codebook_v, rotation,
    scale=1.0 / math.sqrt(D), N=N,
    block_size=block_size, attn_gate_topk=0,
)
```

See `tests/test_paged.py` for a complete end-to-end example that constructs the paged layout from flat tensors and validates the kernel against a pure-PyTorch reference.

---

## Reproducing the paper numbers

```bash
# Smoke tests — validates kernel correctness against a PyTorch reference.
python tests/test_paged.py

# Microbenchmark — dense vs Top-K sparse latency across B × N.
CUDA_VISIBLE_DEVICES=0 python experiments/benchmark_paged.py
```

---

## Public API

| Symbol                                    | Layer        | What it does                                |
|-------------------------------------------|--------------|---------------------------------------------|
| `cda_vllm.cuda_cda_paged`                 | high-level   | Fused paged attention (score + softmax + V) |
| `cda_vllm.score_paged2b_forward`          | low-level    | Paged 2-bit Q·K score kernel                |
| `cda_vllm.vfull_paged2b_forward`          | low-level    | Paged 2-bit attn·V (all tokens)              |
| `cda_vllm.vsparse_paged2b_forward`        | low-level    | Paged 2-bit attn·V (Top-K sparse)            |

All kernels accept the same 2-bit packed layout and Lloyd-Max codebooks produced by the main `cda.HadamardQuantCompressor`.

---

## License

Released for artifact review under the ICCAD 2026 reviewer agreement.
