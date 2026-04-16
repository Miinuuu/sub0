# CDA — Compressed-Domain Attention

Artifact for the ASPLOS 2027 submission: *"Compressed-Domain Attention: Breaking the Memory Wall of Long-Context LLM Decoding via Lookup-Based Attention"*.

CDA computes attention directly on compressed KV indices via pre-computed LUT lookups, bypassing the decompress-then-attend overhead entirely.

---

## Install

```bash
pip install -r requirements.txt
pip install -e .
python tests/smoke_test.py   # 7/7 should pass
```

**Platform:** Python 3.10, PyTorch 2.5 + CUDA 12.1, NVIDIA RTX A6000 (sm_86), Linux x86_64.

---

## Reproducing Paper Results

All results are pre-computed in `runs/`. To re-run:

| Paper Location | Script | Command |
|---|---|---|
| Table 2 (PPL) | `bench_ppl_scale.py` | `CUDA_VISIBLE_DEVICES=0 python experiments/bench_ppl_scale.py` |
| Table 2 (LongBench) | `bench_longbench_single.py` | `CUDA_VISIBLE_DEVICES=0 python experiments/bench_longbench_single.py --task narrativeqa` |
| Table 4 (Breakdown) | `bench_breakdown.py` | `CUDA_VISIBLE_DEVICES=0 python experiments/bench_breakdown.py` |
| Table 5 (Batched) | `bench_batched_throughput.py` | `CUDA_VISIBLE_DEVICES=0 python experiments/bench_batched_throughput.py` |
| Fig 4(a) Kernel speedup | `bench_table5.py` | `CUDA_VISIBLE_DEVICES=0 python experiments/bench_table5.py` |
| Fig 4(b) Long-ctx PPL | `bench_longctx_ppl_topk.py` | `CUDA_VISIBLE_DEVICES=0 python experiments/bench_longctx_ppl_topk.py` |
| Fig 4(c) E2E decode | `bench_cda_integrated_single.py` | `CUDA_VISIBLE_DEVICES=0 python experiments/bench_cda_integrated_single.py --N 65536` |
| Fig 4(d) TopK tradeoff | `bench_table5.py` | `CUDA_VISIBLE_DEVICES=0 python experiments/bench_table5.py --topk 128` |
| Fig 5(a) TopK ablation | `bench_table5.py` | `CUDA_VISIBLE_DEVICES=0 python experiments/bench_table5.py --topk 0,32,64,128,256,512` |
| Fig 5(b) Roofline | `bench_gpu_roofline.py` | `CUDA_VISIBLE_DEVICES=0 python experiments/bench_gpu_roofline.py` |
| Fig 5(c) Batch capacity | `bench_memory_capacity.py` | `CUDA_VISIBLE_DEVICES=0 python experiments/bench_memory_capacity.py` |
| Fig 5(d) Throughput | `bench_batch_serving.py` | `CUDA_VISIBLE_DEVICES=0 python experiments/bench_batch_serving.py` |
| S6.2 NIAH | `bench_niah_topk.py` | `CUDA_VISIBLE_DEVICES=0 python experiments/bench_niah_topk.py` |

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
│   ├── fused_attention.py      # HF model monkey-patch
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
from cda import HadamardQuantCompressor

comp = HadamardQuantCompressor(dim=128, bit_width=2, half_rotation=True)
x = torch.randn(64, 128, device="cuda")
compressed = comp.quantize(x)
restored = comp.dequantize(compressed)
print(f"MSE: {(x - restored).pow(2).mean():.4f}")
```

---

## License

Released for anonymous artifact review. Open-source license upon acceptance.
