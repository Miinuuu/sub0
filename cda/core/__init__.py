"""Compressed-Domain Attention (CDA) — core library.

Same layout as the submission drop (``sub0/cda/core/``). Maintainer-only
additions are co-located here (``_kernels_jit.py``, ``build_kernels.py``)
and are stripped before shipping.

Public modules (mirrored across both trees):

  * :mod:`core.compression`          — quantizers (Hadamard, TurboQuant,
    layer-adaptive) + :class:`CompressedTensor`.
  * :mod:`core.compressed_attention` — pure-PyTorch SW / HW reference.
  * :mod:`core.cda_attn`             — GQA-aware CUDA kernel wrappers
    (contiguous + paged, 2/4-bit K×V, dense + TopK) + HF Llama patch.
  * :mod:`core.compressed_generate`  — long-context decode orchestration.

Pre-compiled CUDA kernels live in ``core._cda_gqa_kernels``.
"""
from core.compression import (
    CompressedTensor,
    HadamardQuantCompressor,
    LayerAdaptiveCompressor,
    PCAQuantizedCompressor,
    TurboQuantCompressor,
    TurboQuantProdCompressor,
    compute_norm_adaptive_schedule,
)
from core.compressed_attention import (
    hw_attention,
    hw_attention_output,
    hw_attention_score,
    sw_attention,
)

__version__ = "0.1.0"

__all__ = [
    "CompressedTensor",
    "HadamardQuantCompressor",
    "LayerAdaptiveCompressor",
    "PCAQuantizedCompressor",
    "TurboQuantCompressor",
    "TurboQuantProdCompressor",
    "compute_norm_adaptive_schedule",
    "hw_attention",
    "hw_attention_output",
    "hw_attention_score",
    "sw_attention",
]
