"""Compressed-Domain Attention (CDA).

Hardware-efficient KV cache compression for LLM decode. The public API is
divided into three layers:

  * :mod:`cda.compression` — quantizers (TurboQuant, CDA/Hadamard, layer-adaptive)
    and the :class:`CompressedTensor` container.
  * :mod:`cda.compressed_attention` — reference SW / HW attention modes on
    compressed KV (pure PyTorch).
  * :mod:`cda.cuda_attention` — fused CUDA kernels for compressed-domain
    attention (built by ``setup.py`` into ``cda._cda_kernels``).

High-level integration:

  * :mod:`cda.compressed_model` — drop-in HuggingFace wrapper.
  * :mod:`cda.patch_attention` — in-place cache + attention patching for
    existing models.
"""
from cda.compression import (
    CompressedTensor,
    HadamardQuantCompressor,
    LayerAdaptiveCompressor,
    PCAQuantizedCompressor,
    TurboQuantCompressor,
    TurboQuantProdCompressor,
    compute_norm_adaptive_schedule,
)
from cda.compressed_attention import (
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
