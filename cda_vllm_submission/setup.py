"""Binary-only install script for the vLLM paged-attention variant of CDA.

Ships the pre-compiled ``cda_vllm._cda_paged_kernels`` extension; no source
build step is performed on the reviewer's machine. The CUDA source is kept
in a private maintainer tree.

Tested platform: Python 3.10, PyTorch 2.5 + CUDA 12.1, sm_86 (RTX A6000).

Install::

    pip install -e .
"""
from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent

setup(
    name="cda-vllm",
    version="0.1.0",
    description=(
        "CDA paged-attention CUDA kernels for vLLM-style serving "
        "(ICCAD 2026 companion artifact)."
    ),
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["cda_vllm", "cda_vllm.*"]),
    python_requires=">=3.10,<3.11",
    install_requires=[
        "torch>=2.1",
        "numpy>=1.24",
        "cda>=0.1.0",  # main package, provides HadamardQuantCompressor
    ],
    package_data={"cda_vllm": ["*.so"]},
    include_package_data=True,
    zip_safe=False,
)
