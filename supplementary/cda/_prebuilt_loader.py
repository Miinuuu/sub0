"""Prebuilt CUDA extension loader for the supplementary distribution.

Replaces the in-tree JIT loader calls used in the development build.
Source ``.cu`` files are not redistributed in this anonymized release;
instead, ``cda/_prebuilt/<name>.so`` ships compiled artifacts
that match the development environment listed in ``INSTALL.md``.

Two loading flavors are exposed because the project mixes two styles:

  1. ``load_pybind_module(name)`` — for extensions whose ``.so`` exposes
     a Python module via ``PYBIND11_MODULE`` (e.g., the HMMA decode/
     prefill kernels under ``cda/kernels_cuda/``). Returns a Python
     module-like object whose attributes are the exported functions.

  2. ``load_torch_ops_library(name)`` — for extensions whose ``.so``
     registers ops through ``TORCH_LIBRARY`` (e.g., the FA2 fork's
     ``_cda_vllm_fa2_fork_C`` namespace). Returns the populated
     ``torch.ops.<namespace>`` namespace.

Both functions are idempotent: subsequent calls with the same name
return the cached handle.

ABI requirement: ``.so`` artifacts were built for cp310 / PyTorch 2.10
(cu128) / SM86. Loading on a different ABI raises ``ImportError``.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import torch  # noqa: F401  — must be loaded so libc10 symbols resolve

_HERE = Path(__file__).resolve().parent
_PREBUILT_DIR = _HERE / "_prebuilt"


class PrebuiltMissing(ImportError):
    """Raised when no ``.so`` matches the requested logical name."""


def _so_path(name: str) -> Path:
    p = _PREBUILT_DIR / f"{name}.so"
    if not p.is_file():
        raise PrebuiltMissing(
            f"Prebuilt extension '{name}' not found at {p}.\n"
            "This supplementary requires Python 3.10 / PyTorch 2.10 "
            "(cu128) / SM86 (NVIDIA Ampere)."
        )
    return p


@lru_cache(maxsize=64)
def load_pybind_module(name: str) -> ModuleType:
    """Load a pybind11-style extension from ``cda/_prebuilt/<name>.so``."""
    so = _so_path(name)
    spec = importlib.util.spec_from_file_location(name, str(so))
    if spec is None or spec.loader is None:
        raise PrebuiltMissing(f"Could not build loader for {so}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=64)
def load_torch_ops_library(name: str):
    """Load a TORCH_LIBRARY-registered extension and return its torch.ops namespace."""
    so = _so_path(name)
    torch.ops.load_library(str(so))
    namespace = getattr(torch.ops, name, None)
    if namespace is None:
        raise PrebuiltMissing(
            f"torch.ops.{name} was not registered after loading {so}; "
            "the prebuilt artifact may be stale or for a different ABI."
        )
    return namespace
