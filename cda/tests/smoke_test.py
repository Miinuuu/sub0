"""Smoke tests for the CDA submission package.

Run from the repository root::

    python -m pytest tests/smoke_test.py -v
    # or
    python tests/smoke_test.py

The CUDA kernel test is skipped gracefully if cda._cda_kernels was not built
(e.g. on a CPU-only machine). Build with ``pip install -e .`` to enable it.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Compressor round-trip
# ---------------------------------------------------------------------------

def test_turboquant_roundtrip():
    from core.compression import TurboQuantCompressor

    comp = TurboQuantCompressor(dim=128, bit_width=4)
    data = torch.randn(64, 128)
    restored = comp.dequantize(comp.quantize(data))
    assert restored.shape == data.shape
    mse = ((data - restored) ** 2).mean().item()
    assert mse < 1.0, f"TurboQuant 4-bit MSE too high: {mse}"
    print(f"  TurboQuant 4-bit roundtrip: MSE={mse:.5f}")


def test_hadamard_roundtrip():
    from core.compression import HadamardQuantCompressor

    comp = HadamardQuantCompressor(dim=128, bit_width=2)
    data = torch.randn(64, 128)
    restored = comp.dequantize(comp.quantize(data))
    assert restored.shape == data.shape
    mse = ((data - restored) ** 2).mean().item()
    assert mse < 2.0, f"Hadamard 2-bit MSE too high: {mse}"
    print(f"  Hadamard 2-bit roundtrip: MSE={mse:.5f}")


def test_asymmetric_kv_bits():
    from core.compression import HadamardQuantCompressor

    comp_k = HadamardQuantCompressor(dim=128, bit_width=4)
    comp_v = HadamardQuantCompressor(dim=128, bit_width=2)
    data = torch.randn(64, 128)
    rk = comp_k.dequantize(comp_k.quantize(data))
    rv = comp_v.dequantize(comp_v.quantize(data))
    mse_k = ((data - rk) ** 2).mean().item()
    mse_v = ((data - rv) ** 2).mean().item()
    assert mse_k < mse_v, "4-bit should beat 2-bit"
    print(f"  Asymmetric K4/V2: mse_k={mse_k:.5f}, mse_v={mse_v:.5f}")


def test_layer_adaptive():
    from cda import LayerAdaptiveCompressor

    schedule = [2] * 4 + [4] * 24 + [2] * 4  # 32-layer toy schedule
    comp = LayerAdaptiveCompressor(dim=128, layer_bit_schedule=schedule)
    data = torch.randn(32, 2, 4, 16, 128)  # (L, B, H, S, D)
    restored = comp.dequantize(comp.quantize(data))
    assert restored.shape == data.shape
    print(f"  LayerAdaptive roundtrip: shape={tuple(restored.shape)}")


# ---------------------------------------------------------------------------
# Attention parity
# ---------------------------------------------------------------------------

def _reference_attention(Q, K, V, scale):
    scores = (Q @ K.transpose(-1, -2)) * scale
    attn = torch.softmax(scores, dim=-1)
    return attn @ V


def test_sw_attention_matches_dense():
    """SW-mode compressed attention should stay close to dense attention.

    ``sw_attention`` expects K/V flat as (N, D); we reshape accordingly and
    compare against the dense reference on the same flat layout.
    """
    from core.compression import HadamardQuantCompressor, sw_attention

    torch.manual_seed(0)
    M, N, D = 4, 32, 128  # M queries attending to N keys
    Q = torch.randn(M, D)
    K = torch.randn(N, D)
    V = torch.randn(N, D)

    comp = HadamardQuantCompressor(dim=D, bit_width=4)
    cK = comp.quantize(K)
    cV = comp.quantize(V)

    scale = 1.0 / math.sqrt(D)
    ref = _reference_attention(Q, K, V, scale)
    out = sw_attention(Q, cK, cV, comp, scale=scale)
    rel = (ref - out).norm() / ref.norm()
    assert rel < 0.2, f"SW attention rel-error too high: {rel.item():.3f}"
    print(f"  sw_attention vs dense: rel_err={rel.item():.4f}")


def test_hw_sw_agreement():
    """HW and SW modes should agree (both read the same compressed KV)."""
    from core.compression import HadamardQuantCompressor, hw_attention, sw_attention

    torch.manual_seed(1)
    M, N, D = 2, 16, 128
    Q = torch.randn(M, D)
    K = torch.randn(N, D)
    V = torch.randn(N, D)

    comp = HadamardQuantCompressor(dim=D, bit_width=2)
    cK = comp.quantize(K)
    cV = comp.quantize(V)
    scale = 1.0 / math.sqrt(D)

    sw_out = sw_attention(Q, cK, cV, comp, scale=scale)
    hw_out = hw_attention(Q, cK, cV, comp, scale=scale)
    rel = (sw_out - hw_out).norm() / sw_out.norm()
    assert rel < 1e-2, f"HW vs SW mismatch: {rel.item():.4f}"
    print(f"  hw vs sw agreement: rel_err={rel.item():.2e}")


# ---------------------------------------------------------------------------
# CUDA kernel (optional — skipped if not built / no GPU)
# ---------------------------------------------------------------------------

def test_cuda_kernel_matches_sw():
    """Fused CUDA kernel result should match the pure-PyTorch SW reference."""
    if not torch.cuda.is_available():
        print("  [skip] no CUDA device")
        return
    try:
        from core.cuda_attention import cuda_hw_attention_batched
    except ImportError as exc:
        print(f"  [skip] _cda_kernels not built ({exc})")
        return

    from core.compression import HadamardQuantCompressor, sw_attention

    torch.manual_seed(2)
    device = torch.device("cuda:0")
    # Flat single-head layout matches sw_attention's usage contract.
    M, N, D = 2, 64, 128  # M queries attending to N compressed keys
    Q = torch.randn(M, D, device=device, dtype=torch.float32)
    K = torch.randn(N, D, device=device, dtype=torch.float32)
    V = torch.randn(N, D, device=device, dtype=torch.float32)

    comp = HadamardQuantCompressor(dim=D, bit_width=2)
    cK = comp.quantize(K)
    cV = comp.quantize(V)
    scale = 1.0 / math.sqrt(D)

    sw_out = sw_attention(Q, cK, cV, comp, scale=scale)

    # Replicate K/V across the M query dimension so the kernel's per-query
    # batch matches. Each query sees the same N keys/values.
    pK = cK.indices.unsqueeze(0).expand(M, N, -1).reshape(M * N, -1).contiguous()
    pV = cV.indices.unsqueeze(0).expand(M, N, -1).reshape(M * N, -1).contiguous()
    nK = cK.norms.float().unsqueeze(0).expand(M, N).reshape(-1).contiguous()
    nV = cV.norms.float().unsqueeze(0).expand(M, N).reshape(-1).contiguous()

    comp._ensure_tensors(device)
    codebook = (comp._centroids * 2.0 - 1.0).float()
    Q_rot = Q @ comp._rotation_t.float()
    cuda_out = cuda_hw_attention_batched(
        Q_rot_all=Q_rot,
        packed_indices_K=pK,
        norms_K=nK,
        packed_indices_V=pV,
        norms_V=nV,
        codebook=codebook,
        rotation=comp._rotation.float(),
        scale=scale,
        N=N,
    )
    rel = (sw_out - cuda_out).norm() / sw_out.norm()
    assert rel < 5e-2, f"CUDA vs SW mismatch: {rel.item():.4f}"
    print(f"  cuda kernel vs sw: rel_err={rel.item():.2e}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ("TurboQuant roundtrip", test_turboquant_roundtrip),
    ("Hadamard roundtrip", test_hadamard_roundtrip),
    ("Asymmetric K/V bits", test_asymmetric_kv_bits),
    ("LayerAdaptive", test_layer_adaptive),
    ("SW attn vs dense", test_sw_attention_matches_dense),
    ("HW == SW", test_hw_sw_agreement),
    ("CUDA kernel vs SW", test_cuda_kernel_matches_sw),
]


def main() -> int:
    print("=== CDA smoke tests ===")
    passed, failed = 0, 0
    for name, fn in TESTS:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{passed + failed} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
