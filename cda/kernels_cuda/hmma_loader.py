"""Prebuilt loaders for the paper-essential CDA HMMA kernels.

In the development tree these functions JIT-compile their CUDA sources
via ``torch.utils.cpp_extension.load(...)``. The supplementary
distribution ships compiled artifacts under ``cda/_prebuilt/`` instead
of the sources, so each loader resolves to
``cda._prebuilt_loader.load_pybind_module(<name>)``.

The 6 paper-essential .so artifacts (paper Table 2/3 + Fig 2 + Table A4):

* ``_v1_decode_hmma_k4v4_gs4``        — Tables 2/3 + serving decode
* ``_v1_paged_encode_fused``          — KV cache encode
* ``_v1_fa2_cda_loader_k4v4``         — vLLM prefill K/V materialization
* ``_v1_fa2_cda_varlen_fused``        — vLLM prefill epilogue (Hadamard fused)
* ``_v1_fa2_cda_hadamard128``         — Hadamard helper
* ``_cda_vllm_fa2_fork_C``            — FA2 fork (TORCH_LIBRARY ops)

ABI lock: cp310 / PyTorch 2.10 (cu128) / SM86 (NVIDIA Ampere).
"""
from __future__ import annotations

from functools import lru_cache

from cda._prebuilt_loader import load_pybind_module


@lru_cache(maxsize=8)
def load_decode_hmma(k4v2: bool = False, group_size: int = 4):
    if group_size not in (4, 8):
        raise ValueError(f"group_size must be 4 or 8 (got {group_size})")
    v_tag = "k4v2" if k4v2 else "k4v4"
    return load_pybind_module(f"_v1_decode_hmma_{v_tag}_gs{group_size}")


@lru_cache(maxsize=2)
def load_paged_encode_fused():
    return load_pybind_module("_v1_paged_encode_fused")


@lru_cache(maxsize=4)
def load_kv_cache_update(k4v2: bool = False):
    name = "_v1_kv_update_k4v2" if k4v2 else "_v1_kv_update_k4v4"
    return load_pybind_module(name)


@lru_cache(maxsize=4)
def load_fa2_cda_loader(k4v2: bool = False):
    name = "_v1_fa2_cda_loader_k4v2" if k4v2 else "_v1_fa2_cda_loader_k4v4"
    return load_pybind_module(name)


@lru_cache(maxsize=1)
def load_fa2_cda_hadamard():
    return load_pybind_module("_v1_fa2_cda_hadamard128")


@lru_cache(maxsize=1)
def load_fa2_cda_varlen_fused():
    return load_pybind_module("_v1_fa2_cda_varlen_fused")


__all__ = [
    "load_decode_hmma",
    "load_paged_encode_fused",
    "load_kv_cache_update",
    "load_fa2_cda_loader",
    "load_fa2_cda_hadamard",
    "load_fa2_cda_varlen_fused",
]
