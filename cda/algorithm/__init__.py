"""Pure-PyTorch reference implementation.

Defines correctness for the compressed-domain attention pipeline. CuTeDSL
kernels in :mod:`cda.kernels` must match every function here at
``cos ≥ 0.9999`` across the standard sweep. See ``docs/design.md`` for
the layered architecture.

Modules:
    compression  — Hadamard rotation + low-bit quant (encode / decode)
    attention    — softmax(QK^T)V on compressed K/V (correctness reference)
    rotation_policy — independent QK/V rotations + weight folding references
    flash_ref    — FlashAttention-style streaming CDA oracle
    decode_ref   — single-Q × N keys (decode regime)
    prefill_ref  — multi-Q × N keys + causal (prefill regime)
    chunked_ref  — chunked-prefill orchestrator
"""
