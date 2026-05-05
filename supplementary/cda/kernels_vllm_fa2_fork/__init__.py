"""vLLM FA2 varlen fork entry points."""

from .interface import (
    dequantize_compressed_kv,
    flash_attn_varlen_compressed_kv_fused_func,
    flash_attn_varlen_compressed_kv_func,
    flash_attn_varlen_func,
)


def register_vllm_backend(slot: str = "CUSTOM") -> None:
    from .backend import register_backend

    register_backend(slot)


__all__ = [
    "dequantize_compressed_kv",
    "flash_attn_varlen_compressed_kv_fused_func",
    "flash_attn_varlen_compressed_kv_func",
    "flash_attn_varlen_func",
    "register_vllm_backend",
]
