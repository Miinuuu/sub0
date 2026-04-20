"""Load pre-built CUDA kernel extensions from prebuilt_hmma/ directory.

The supplementary bundle ships .so binaries only (no CUDA source). This
module replaces the source-based JIT-compile paths in cda_attn.py with
direct .so loading via importlib.

Tested on: Python 3.10, PyTorch 2.10.0+cu128, CUDA 12.8, NVIDIA sm_86.
For other architectures, contact the authors for a rebuild.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path


_DIR = Path(__file__).resolve().parent / "prebuilt_hmma"
_CACHE: dict[str, object] = {}


_TORCH_LIBS_LOADED = False

def _preload_torch_libs():
    """Force libc10 and libc10_cuda into the global symbol table.

    importlib's default (RTLD_LOCAL) hides PyTorch's CUDA symbols from our
    .so, causing ImportError: undefined symbol c10_cuda_check_implementation.
    We ctypes.CDLL them with RTLD_GLOBAL before dlopening our kernels.
    """
    global _TORCH_LIBS_LOADED
    if _TORCH_LIBS_LOADED:
        return
    import ctypes, os, torch
    torch_lib = Path(torch.__file__).parent / "lib"
    for name in ("libc10.so", "libc10_cuda.so"):
        p = torch_lib / name
        if p.exists():
            ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
    torch.cuda.is_available()  # also triggers PyTorch's own init
    _TORCH_LIBS_LOADED = True


def _load(name: str, filename: str):
    if name in _CACHE:
        return _CACHE[name]
    _preload_torch_libs()
    so_path = _DIR / filename
    if not so_path.exists():
        raise ImportError(
            f"Pre-built kernel missing: {so_path}. "
            f"This supplementary is compiled for Python 3.10 + PyTorch 2.10 "
            f"+ CUDA 12.8 + sm_86. Contact the authors for other targets."
        )
    spec = importlib.util.spec_from_file_location(name, str(so_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _CACHE[name] = mod
    return mod


def load_hmma():    return _load("_hmma_production",        "_hmma_production.so")
def load_hmma_v3(): return _load("_cda_hmma_production_v3", "_cda_hmma_production_v3.so")
def load_hmma_g8(): return _load("_cda_hmma_production_g8", "_cda_hmma_production_g8.so")

def load_fused_update():       return _load("_cda_fused_update",     "_cda_fused_update.so")
def load_fused_reduce_rot():   return _load("_cda_fused_reduce_rot", "_cda_fused_reduce_rot.so")
