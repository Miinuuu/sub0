# CDA — Compressed-Domain Attention

Hardware-efficient KV cache compression for LLM decode. Companion artifact for the ICCAD 2026 submission _"Compressed-Domain Attention: Hadamard-Rotated KV Compression for FPGA/ASIC LLM Inference"_.

CDA replaces TurboQuant's random orthogonal rotation with a Sylvester Hadamard matrix. The rotation becomes a DSP-free butterfly on FPGA/ASIC, and because the rotation is shared and data-oblivious, attention can be computed **directly on compressed indices via codebook lookup** — the KV cache is never decompressed during inference.

| Metric (2-bit KV)              | vs FP16 KV | vs. decompress-then-matmul |
|-------------------------------|-----------:|----------------------------:|
| KV memory footprint           |      7.5× |                         — |
| MAC operations per decode step |         — |                        37× |

> **Companion artifact.** The [`cda_vllm_submission/`](cda_vllm_submission/) sub-package ships the paged-attention variant of CDA (vLLM `block_tables`-compatible CUDA kernels) under the same binary-distribution model. See the _Companion artifact_ section below.

---

## Distribution model

This repository ships as a **binary artifact**. Pre-compiled extensions for the tested reference platform are included so reviewers can reproduce all paper numbers without access to the private sources.

| Component                                  | Format              | Shipped here? |
|--------------------------------------------|---------------------|:-------------:|
| Public high-level wrappers & API stubs     | `.py` source        | ✅            |
| Quantizer / reference attention / cache    | Cython `.so`         | ✅            |
| Fused CUDA kernels (score + V output)      | CUDA `.so`           | ✅            |
| CUDA sources, Cython `.pyx`, maintainer tooling | —              | ❌ (private)  |

**Tested platform.** Python 3.10, PyTorch 2.5 + CUDA 12.1, NVIDIA RTX A6000 (sm_86), Linux x86_64. The bundled binaries are ABI-compatible with this stack only.

**Rebuilding for a different platform.** Contact the authors for a platform-specific binary drop or an NDA-gated source tree.

---

## Repository layout (shipped)

```
cda_submission/
├── README.md
├── requirements.txt
├── setup.py                      # Binary-only install
├── pyproject.toml
│
├── cda/                          # Importable Python package
│   ├── __init__.py               # Public API (source, open)
│   ├── compressed_model.py       # HuggingFace wrapper (source, open)
│   ├── cuda_attention.py         # CUDA kernel binding stubs (source, open)
│   ├── compression.cpython-310-x86_64-linux-gnu.so          # (binary, closed)
│   ├── compressed_attention.cpython-310-x86_64-linux-gnu.so # (binary, closed)
│   ├── patch_attention.cpython-310-x86_64-linux-gnu.so      # (binary, closed)
│   └── _cda_kernels.cpython-310-x86_64-linux-gnu.so         # (binary, closed)
│
├── experiments/
│   ├── benchmark_speed.py        # Decode latency + KV memory
│   └── benchmark_ppl.py          # WikiText-2 perplexity
│
└── tests/
    └── smoke_test.py             # Public-API smoke tests
```

---

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

No compilation step — the bundled `.so` files are dropped onto the import path directly.

---

## Quick start

### Compress a tensor and round-trip it

```python
import torch
from cda import HadamardQuantCompressor

comp = HadamardQuantCompressor(dim=128, bit_width=2, half_rotation=True)
x = torch.randn(64, 128, device="cuda")
compressed = comp.quantize(x)          # CompressedTensor (bit-packed indices + norms)
restored   = comp.dequantize(compressed)
print((x - restored).pow(2).mean())
```

### Swap in a compressed KV cache for any HF model

```python
from cda.compressed_model import CompressedKVModel

model = CompressedKVModel("meta-llama/Llama-3.1-8B-Instruct",
                          bit_width=2, skip_sinks=4)
print(model.generate("The meaning of life is ", max_new_tokens=64))
```

Prefill runs normally; KV is compressed in-place before decode.

### Call the fused CUDA kernels directly

```python
from cda.cuda_attention import cuda_hw_attention_batched
# See experiments/benchmark_speed.py and tests/smoke_test.py for complete call sites.
```

---

## Reproducing the paper numbers

### Smoke test

```bash
python tests/smoke_test.py
```

Exercises the public API: compressor round-trip at 2/4-bit, layer-adaptive schedules, SW vs dense attention agreement, HW == SW equivalence, and the fused CUDA kernel against the SW reference (skipped on CPU-only setups).

### Decode speed + KV memory

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/benchmark_speed.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --ctx 1024,4096,8192,16384 --bits 2 \
    --output runs/cda_speed.json
```

Reports, for each context length:

* FP16 baseline decode (ms) and peak memory (MiB)
* CDA-SW decode (ms / MiB) — compressed KV, decompress per step
* CDA-HW kernel microbenchmark (ms) — fused CUDA kernel on synthetic input

### WikiText-2 perplexity

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/benchmark_ppl.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --bits 2 --max-length 2048 --skip-sinks 4 \
    --output runs/cda_ppl.json
```

Outputs FP16 PPL, CDA PPL, and the delta on the WikiText-2 test split.

---

## Public API reference

| Symbol                                   | Layer        | What it does                           |
|------------------------------------------|--------------|----------------------------------------|
| `cda.TurboQuantCompressor`               | quantizer    | Random-orthogonal rotation + Lloyd-Max |
| `cda.HadamardQuantCompressor`            | quantizer    | DSP-free Sylvester Hadamard rotation   |
| `cda.LayerAdaptiveCompressor`            | quantizer    | Per-layer bit schedule                 |
| `cda.PCAQuantizedCompressor`             | quantizer    | PCA-projected baseline                 |
| `cda.sw_attention`                       | attention    | Decompress → fp16 cuBLAS matmul        |
| `cda.hw_attention(_score,_output)`       | attention    | Compressed-domain Python reference     |
| `cda.cuda_attention.cuda_hw_attention_batched` | attention | Fused CUDA end-to-end step          |
| `cda.compressed_model.CompressedKVModel` | integration  | HuggingFace model wrapper              |

See each symbol's docstring for argument conventions.

---

## Hardware notes

The paper's FPGA/ASIC numbers are generated from a separate RTL tree (not included here). This artifact exercises the reference CUDA implementation used to cross-validate the hardware numerically: identical Lloyd-Max codebook, identical Hadamard rotation, identical bit-packing.

---

## Companion artifact: `cda_vllm_submission/`

The `cda_vllm_submission/` sub-package, released alongside the main package, provides the **vLLM-style paged-attention variant** of CDA. It adds paged CUDA kernels that resolve compressed KV through vLLM's logical→physical `block_tables`:

| Kernel                       | Role                                                           |
|------------------------------|----------------------------------------------------------------|
| `score_paged2b_forward`      | Q · compressed K with bank-conflict-free shared-memory layout  |
| `vfull_paged2b_forward`      | Dense attn · compressed V over all paged tokens                |
| `vsparse_paged2b_forward`    | Top-K sparse attn · compressed V, indices→pages                |
| `cuda_cda_paged` (composed)  | Full fused decode step: score → softmax → (sparse\|dense) V    |

**Why ship it separately.** The main `cda` package focuses on the quantizer, SW/HW reference, and the contiguous-KV CUDA kernels that the ICCAD paper tables use. The paged variant is orthogonal — it demonstrates that the same 2-bit packed layout drops into a vLLM-style serving stack without re-encoding — so we keep it as a self-contained sub-package that depends on the main one.

**Install order.**

```bash
# 1) this package
pip install -e .

# 2) companion paged kernels
pip install -e cda_vllm_submission
```

**Validation.** `cda_vllm_submission/tests/test_paged.py` compares the paged kernel to a pure-PyTorch decode-then-attention reference (rel_err ≈ 8.5 × 10⁻⁸ in the current build) and to the dense paged path for the Top-K sparsity gate.

See [`cda_vllm_submission/README.md`](cda_vllm_submission/README.md) for the full API, layout, and reproduction instructions.

---

## License

Released for artifact review under the ICCAD 2026 reviewer agreement. A permissive open-source license will be attached upon acceptance, with the private source tree released alongside.
