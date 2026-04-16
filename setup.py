"""Binary-only install script for the CDA submission.

This setup installs the pre-compiled ``.so`` extensions that ship with the
package; it does **not** build from source. The CUDA kernels
(``cda._cda_kernels``) and the private Python modules
(``cda.compression``, ``cda.compressed_attention``, ``cda.patch_attention``)
are shipped as platform-specific Cython/CUDA binaries and must match the
reviewer's Python version.

Tested combination: Python 3.10, PyTorch 2.5+cu121, CUDA 12.1, sm_86 (RTX A6000).

Install::

    pip install -e .
"""
from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent

setup(
    name="cda",
    version="0.1.0",
    description=(
        "Compressed-Domain Attention (CDA) — hardware-efficient KV cache "
        "compression for LLM decode (ICCAD 2026 artifact)."
    ),
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["cda", "cda.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1",
        "transformers>=4.40",
        "numpy>=1.24",
        "scipy>=1.10",
        "accelerate>=0.28",
    ],
    package_data={"cda": ["*.so"]},
    include_package_data=True,
    zip_safe=False,
)
