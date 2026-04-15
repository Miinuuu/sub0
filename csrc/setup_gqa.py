"""Private maintainer build script for the GQA-aware CUDA extension.

Not shipped with the sub0 binary artifact — the compiled output
``cda/_cda_gqa_kernels*.so`` is. Kept here alongside the CUDA source so
the package can be rebuilt when the tested Python/PyTorch/CUDA combination
changes. Mirrors the existing ``_cda_kernels`` packaging approach.

Usage (from repo root)::

    python csrc/setup_gqa.py build_ext --inplace

This produces ``cda/_cda_gqa_kernels.cpython-3X-<arch>.so``. The
``cda.cuda_attention_gqa`` Python wrapper then loads it at import time.

Build environment tested: Python 3.10, PyTorch 2.5+cu121, CUDA 12.1,
sm_86 (RTX A6000).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")  # RTX A6000 / sm_86

# Place the built extension inside the ``cda`` package.
ext = CUDAExtension(
    name="cda._cda_gqa_kernels",
    sources=[str(ROOT / "csrc" / "cda_gqa_kernels.cu")],
    extra_compile_args={
        "cxx": ["-O3"],
        "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
    },
)

if __name__ == "__main__":
    # Default to in-place build when run without explicit args.
    if len(sys.argv) == 1:
        sys.argv += ["build_ext", "--inplace"]

    setup(
        name="cda-gqa-kernels",
        version="0.1.0",
        ext_modules=[ext],
        cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
        script_args=sys.argv[1:] or ["build_ext", "--inplace"],
    )
