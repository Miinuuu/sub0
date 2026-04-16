"""Compressed-Domain Attention — vLLM paged variant (ICCAD 2026 artifact).

Companion to the main ``cda`` package (installed separately). This package
adds the paged-attention CUDA kernels that plug into vLLM's logical→physical
block addressing. Use it when you want to integrate CDA into a serving stack
that already speaks vLLM's paged KV layout.

Public entry points:

  * :func:`cda_vllm.cuda_cda_paged` — fused paged attention step.
  * The low-level kernel bindings under :mod:`cda_vllm.paged_attention`.

The main ``cda`` package (quantizers, SW reference, HuggingFace wrapper)
is a required runtime dependency — install it first.
"""
from cda_vllm.paged_attention import (
    cuda_cda_paged,
    score_paged2b_forward,
    vfull_paged2b_forward,
    vsparse_paged2b_forward,
)

__version__ = "0.1.0"

__all__ = [
    "cuda_cda_paged",
    "score_paged2b_forward",
    "vfull_paged2b_forward",
    "vsparse_paged2b_forward",
]
