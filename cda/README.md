# CDA — Compressed-Domain Attention

CDA computes attention directly on compressed KV indices via pre-computed LUT lookups, bypassing the decompress-then-attend overhead entirely.

**Paper:** [Compressed-Domain Attention: Breaking the Memory Wall of Long-Context LLM Decoding via Lookup-Based Attention](https://arxiv.org/abs/XXXX.XXXXX)

---

## Install

```bash
pip install -r requirements.txt
pip install -e .
python tests/smoke_test.py   # 7/7 should pass
```

**Platform:** Python 3.10, PyTorch 2.5 + CUDA 12.1, NVIDIA RTX A6000 (sm_86), Linux x86_64.

---

## Experiments

All results are pre-computed in `runs/`. Benchmark scripts are in `experiments/`.

```bash
# Example: PPL evaluation
CUDA_VISIBLE_DEVICES=0 python experiments/bench_ppl_scale.py

# Example: End-to-end decode latency
CUDA_VISIBLE_DEVICES=0 python experiments/bench_cda_integrated_single.py --N 65536
```

---

## Repository Layout

```
sub0/
├── cda/                        # Package (binary + Python wrappers)
│   ├── __init__.py             # Public API
│   ├── compression.*.so        # Quantizers (binary)
│   ├── compressed_attention.*.so
│   ├── _cda_kernels.*.so       # Per-head CUDA kernels (binary)
│   ├── _cda_gqa_kernels.*.so   # GQA-aware CUDA kernels (binary)
│   ├── cuda_attention.py       # Per-head kernel wrapper
│   ├── cuda_attention_gqa.py   # GQA kernel wrapper
│   ├── cda_attn.py                  # HF model monkey-patch
│   └── compressed_model.py     # HF model wrapper
├── experiments/                # 12 scripts (one per paper figure/table)
├── runs/                       # 38 JSON result files
└── tests/smoke_test.py         # 7 correctness tests
```

**Binary distribution.** Core CUDA/Cython sources are not included. Pre-built `.so` files match the tested platform. Contact authors for other platforms.

---

## Quick Example

```python
import torch
from core.compression import HadamardQuantCompressor

comp = HadamardQuantCompressor(dim=128, bit_width=2, half_rotation=True)
x = torch.randn(64, 128, device="cuda")
compressed = comp.quantize(x)
restored = comp.dequantize(compressed)
print(f"MSE: {(x - restored).pow(2).mean():.4f}")
```

---

## License

MIT License
