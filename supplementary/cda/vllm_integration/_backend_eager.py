"""Eager class-path entry for vLLM's AttentionBackendEnum registry.

vLLM's ``register_backend`` resolves classes via fully-qualified path
(``module.attribute``). The four CDA-v2 backend classes live inside a
closure (``cda.vllm_integration.cda_attn_v2._build_backend_classes``)
so the module is import-safe outside vLLM environments. This shim
materializes the classes at import time and re-exports them so the
registry can find ``cda.vllm_integration._backend_eager.CDAv2AttentionBackend``.

Importing this module requires vLLM to be present (it triggers the
lazy-built classes).
"""

from cda.kernels_cuda.vllm_integration.cda_attn_v2 import get_backend_classes

(
    CDAv2AttentionBackend,
    CDAv2AttentionImpl,
    CDAv2AttentionMetadataBuilder,
    CDAv2AttentionMetadata,
) = get_backend_classes()


__all__ = [
    "CDAv2AttentionBackend",
    "CDAv2AttentionImpl",
    "CDAv2AttentionMetadataBuilder",
    "CDAv2AttentionMetadata",
]
