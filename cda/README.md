# CDA — Compressed-Domain Attention

CDA computes attention directly on compressed KV indices via pre-computed
LUT lookups, bypassing the decompress-then-attend overhead entirely.

**Paper:** [Compressed-Domain Attention: Breaking the Memory Wall of Long-Context LLM Decoding via Lookup-Based Attention](https://arxiv.org/abs/XXXX.XXXXX)

---

## Install

```bash
pip install -r requirements.txt
pip install -e .
python tests/smoke_test.py        # smoke tests should pass
```

**Tested platform.** Python 3.10, PyTorch 2.5 + CUDA 12.1, NVIDIA RTX A6000
(sm_86), Linux x86_64. Pre-compiled binaries target this stack; contact the
authors for platform-specific `.so` drops.

---

## Running the paper experiments

All pre-computed results are in `runs/`; scripts live in `experiments/`.

```bash
# Figure 4(a)(c) speed — FP16 SDPA / FP16 FA2 / CDA K{2,4}V{2,4}, 1K→128K
CUDA_VISIBLE_DEVICES=0 python experiments/bench_fig4c_graph.py \
    --Ns 1024,2048,4096,8192,16384,32768,65536,131072

# Figure 4(c) memory — KV footprint per config, 1K→128K
CUDA_VISIBLE_DEVICES=0 python experiments/bench_fig4c_memory.py

# MSE verification — CDA kernel vs FP16 reference (all K/V combinations)
CUDA_VISIBLE_DEVICES=0 python experiments/bench_fig4c_mse.py

# Table 5 — multi-model PPL (wikitext-2 / C4)
CUDA_VISIBLE_DEVICES=0 python experiments/bench_ppl_scale.py \
    --model llama8b --dataset wikitext2

# Figure 5(c) single-N reference timing (reproduces paper 3.16×)
CUDA_VISIBLE_DEVICES=0 python experiments/bench_cda_integrated_single.py \
    --N 32768 --configs K4V2,K2V2

# Long-context PPL (Figure 4(b))
CUDA_VISIBLE_DEVICES=0 python experiments/bench_longctx_ppl_topk.py

# NIAH / LongBench / breakdown / batch-serving / memory-capacity …
ls experiments/
```

---

## Repository layout

```
sub0/cda/
├── core/                               # ★ Python package (4 .py + 3 .so)
│   ├── __init__.py                     #   public re-exports
│   ├── cda_attn.py                     #   GQA kernel wrappers + HF patch
│   ├── compressed_generate.py          #   long-ctx decode / PPL eval
│   ├── compression.*.so                #   quantizers (binary)
│   ├── compressed_attention.*.so       #   SW/HW reference (binary)
│   ├── _cda_gqa_kernels.*.so           #   fused CUDA GQA kernels (binary)
│   └── setup.py
├── experiments/                        # 21 benchmark scripts
├── runs/                               # cached JSON results
├── tests/smoke_test.py                 # correctness checks
├── requirements.txt
├── pyproject.toml
└── README.md
```

### The four-module public API (unified with the research tree)

| Module | Role |
|---|---|
| `core.compression` | `HadamardQuantCompressor`, `TurboQuantCompressor`, `CompressedTensor` |
| `core.compressed_attention` | Pure-PyTorch SW / HW reference attention on compressed KV |
| `core.cda_attn` | GQA-aware CUDA kernel wrappers (contiguous + paged, 2/4-bit K × V, dense + TopK) and HuggingFace Llama monkey-patch |
| `core.compressed_generate` | `manual_decode_step`, `compressed_eval_ppl`, `calibrate_temperature` — long-context accuracy path |

Pre-compiled CUDA kernels live in `core._cda_gqa_kernels`; `cda_attn.py`
loads it and exposes Python wrappers for every kernel. The full
support matrix:

| K bits | V bits | Dense V | Sparse V (TopK) | Contiguous KV | Paged KV |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | 2 | ✅ | ✅ | ✅ | ✅ |
| 2 | 4 | ✅ | ✅ | ✅ | ✅ |
| 4 | 2 | ✅ | ✅ | ✅ | ✅ |
| 4 | 4 | ✅ | ✅ | ✅ | ✅ |

Paged and contiguous paths are **bit-exactly identical** (only KV memory
layout differs).

### Binary distribution

Core CUDA / Cython sources are **not** included in this submission drop
— the shipped `.so` files are the only executable artefacts. The research
tree keeps a maintainer-only `_kernels_jit.py` alongside a build script,
but those are stripped before release. Contact the authors if the
platform binaries need to be rebuilt.

---

## Quick example

```python
import torch
from core import HadamardQuantCompressor, sw_attention  # public API
from core.cda_attn import (
    _compress_kv_cache_cuda, patch_model_compressed_attn,
    unpatch_model, _PositionOnlyCache,
)

# 1. Quantize a tensor
comp = HadamardQuantCompressor(dim=128, bit_width=4, half_rotation=True)
x = torch.randn(64, 128, device="cuda")
compressed = comp.quantize(x)
restored = comp.dequantize(compressed)
print(f"MSE: {(x - restored).pow(2).mean():.4e}")

# 2. Patch a HuggingFace Llama model for CDA decode
# (see experiments/bench_cda_integrated_single.py for the full flow)
```

---

## License

MIT License
