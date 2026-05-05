# Environment Setup

The prebuilt CUDA artifacts under ``cda/_prebuilt/`` are pinned to a
fixed ABI. Use the steps below to recreate that environment.

## Hardware

- Single-GPU paper experiments (Tables 2, 3, Fig. 2, Tables A3, A5):
  one NVIDIA RTX A6000 (Ampere SM86, 48 GB)
- Tensor-parallel paper experiments (Table A4): four NVIDIA RTX A6000
  with NVLink

## Software ABI

| Component      | Version                |
|----------------|------------------------|
| Python         | 3.10                   |
| PyTorch        | 2.10.0 (`+cu128`)      |
| CUDA Toolkit   | 12.8                   |
| Triton         | 3.6.0                  |
| vLLM           | 0.19                   |
| FlashAttention | 2.8.4                  |

Other configurations cannot load the prebuilt ``.so`` artifacts.

## Minimal install (kernel benchmarks only)

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install \
    "torch==2.10.0" --index-url https://download.pytorch.org/whl/cu128
pip install numpy
```

Verify::

```bash
python -c "import torch; \
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability(0))"
# Expected: 2.10.0+cu128 12.8 (8, 6)
```

Then run the kernel-level benchmarks::

```bash
cd supplementary/
PYTHONPATH=. python benchmarks/bench_compute_path_iso_cuda_graph.py
```

## End-to-end serving install (Fig. 2, Table A4)

To run the vLLM throughput benchmarks add::

```bash
pip install vllm==0.19.0
pip install flash-attn==2.8.4 --no-build-isolation
```

Some environment knobs that ``cda_attn_v2`` honours::

```bash
export VLLM_PLUGINS=""           # disable other vLLM plugins
export CDA_FA2_FORK_BUILD_SCOPE=smoke   # smaller FA2 fork build set
```

## Anonymization

This supplementary contains no author names, emails, or absolute paths
(verified before release). The ``cda/__init__.py``
preserves the development-tree namespace alias mechanism so that
``cda.kernels_vllm_fa2_fork`` resolves into the bundled
``deprecated/kernels_vllm_fa2_fork``.
