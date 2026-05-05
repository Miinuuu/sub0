"""vLLM general plugin entry point for cda-v2 backend registration.

Registered as ``vllm.general_plugins/cda_v2`` in pyproject.toml. vLLM
imports and runs all enabled general plugins at startup in EVERY worker
process (including TP-spawned subprocesses), which is what we need for
TP > 1 to make the CDA backend visible inside each worker's
AttentionBackendEnum.

To enable for a run: export VLLM_PLUGINS=cda_v2 (or include cda_v2 in a
comma-separated list). To disable: VLLM_PLUGINS="" (or omit cda_v2).
"""
from __future__ import annotations
import os


def register_cda_v2():
    """vLLM plugin entry: register CDA v2 backend in this process.

    Honors CDA_V2_ENABLE_MEMORY_SAVING=1 to additionally shrink vLLM's
    KV slot allocation to 144 B (full 3.56x KV memory saving).
    """
    # Import inside the function so plugin discovery is lightweight; the
    # heavy CUDA-load happens only when this is actually invoked.
    from cda.kernels_cuda.vllm_integration.cda_attn_v2 import (
        register_backend, enable_cda_memory_saving,
    )
    if os.environ.get("CDA_V2_ENABLE_MEMORY_SAVING") == "1":
        enable_cda_memory_saving()
    register_backend("CDA")
