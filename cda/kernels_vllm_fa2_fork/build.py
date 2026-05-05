"""Prebuilt loader stub for the vLLM FA2 fork (supplementary build).

The development tree compiles this extension from its CUDA sources.
The supplementary distribution ships the prebuilt
``_cda_vllm_fa2_fork_C.so`` instead, so this stub only registers the
TORCH_LIBRARY ops with ``torch.ops.load_library`` on first call.
"""
from __future__ import annotations

from cda._prebuilt_loader import load_torch_ops_library

EXTENSION_NAME = "_cda_vllm_fa2_fork_C"

_LOADED = False


def load(*args, **kwargs):
    """Idempotent: ensure the prebuilt FA2 fork ops are registered."""
    global _LOADED
    if _LOADED:
        return
    load_torch_ops_library(EXTENSION_NAME)
    _LOADED = True
