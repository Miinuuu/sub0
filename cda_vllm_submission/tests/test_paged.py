"""Smoke test for the paged CDA CUDA kernels.

Compares the paged kernel output against:
  1. A pure-PyTorch decode-then-attention reference.
  2. The non-paged fused CUDA kernel from the sibling ``cda`` package.

Run from the repository root::

    python tests/test_paged.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_flat(packed_flat, norms_flat, cb, B, N, D):
    b0 = (packed_flat >> 6) & 3
    b1 = (packed_flat >> 4) & 3
    b2 = (packed_flat >> 2) & 3
    b3 = packed_flat & 3
    indices = torch.stack([b0, b1, b2, b3], dim=-1).reshape(B, N, D).long()
    return cb[indices].float() * norms_flat.unsqueeze(-1).float()


def _build_paged_layout(packed_flat, norms_flat, B, N, D, block_size):
    k_pack = D // 4
    max_logical_blocks = (N + block_size - 1) // block_size
    num_phys = B * max_logical_blocks

    packed_blocks = torch.zeros(
        (num_phys, block_size, k_pack), device=packed_flat.device, dtype=torch.uint8
    )
    norms_blocks = torch.zeros(
        (num_phys, block_size), device=norms_flat.device, dtype=torch.float32
    )
    block_tables = torch.zeros(
        (B, max_logical_blocks), device=packed_flat.device, dtype=torch.int32
    )

    phys = 0
    for b in range(B):
        for lb in range(max_logical_blocks):
            block_tables[b, lb] = phys
            for off in range(block_size):
                n = lb * block_size + off
                if n < N:
                    packed_blocks[phys, off] = packed_flat[b, n]
                    norms_blocks[phys, off] = norms_flat[b, n]
            phys += 1
    return packed_blocks, norms_blocks, block_tables


def _reference_attention(Q_rot, K_dec, V_dec, rotation, scale):
    """Per-batch fp32 reference: Q_rot · K_dec^T → softmax → · V_dec → rotate.

    Shapes: Q_rot (B, D), K_dec/V_dec (B, N, D). Returns (B, D).
    """
    # (B, 1, D) @ (B, D, N) → (B, 1, N)
    scores = (Q_rot.float().unsqueeze(1) @ K_dec.float().transpose(-1, -2)).squeeze(1) * scale
    attn = torch.softmax(scores, dim=-1)
    # (B, 1, N) @ (B, N, D) → (B, 1, D)
    out_rot = (attn.unsqueeze(1) @ V_dec.float()).squeeze(1)
    return out_rot @ rotation


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_paged_matches_reference():
    if not torch.cuda.is_available():
        print("  [skip] no CUDA device")
        return
    try:
        from cda_vllm import cuda_cda_paged
    except ImportError as exc:
        print(f"  [skip] cda_vllm._cda_paged_kernels not built ({exc})")
        return

    torch.manual_seed(42)
    device = torch.device("cuda:0")
    B, N, D = 2, 120, 128
    block_size = 16
    k_pack = D // 4

    Q_rot = torch.randn(B, D, device=device, dtype=torch.float32)
    packed_K_flat = torch.randint(0, 256, (B, N, k_pack), device=device, dtype=torch.uint8)
    norms_K_flat = torch.randn(B, N, device=device, dtype=torch.float32).abs() + 0.1
    packed_V_flat = torch.randint(0, 256, (B, N, k_pack), device=device, dtype=torch.uint8)
    norms_V_flat = torch.randn(B, N, device=device, dtype=torch.float32).abs() + 0.1
    cb_k = torch.randn(4, device=device, dtype=torch.float32)
    cb_v = torch.randn(4, device=device, dtype=torch.float32)
    rotation = torch.eye(D, device=device, dtype=torch.float32)
    scale = 1.0 / math.sqrt(D)

    # Pure-PyTorch reference.
    K_dec = _decode_flat(packed_K_flat, norms_K_flat, cb_k, B, N, D)
    V_dec = _decode_flat(packed_V_flat, norms_V_flat, cb_v, B, N, D)
    ref = _reference_attention(Q_rot, K_dec, V_dec, rotation, scale)

    # Paged kernel.
    pk_blocks, nk_blocks, block_tables = _build_paged_layout(
        packed_K_flat, norms_K_flat, B, N, D, block_size
    )
    pv_blocks, nv_blocks, _ = _build_paged_layout(
        packed_V_flat, norms_V_flat, B, N, D, block_size
    )
    out = cuda_cda_paged(
        Q_rot, pk_blocks, nk_blocks, pv_blocks, nv_blocks,
        block_tables, cb_k, cb_v, rotation, scale, N,
        block_size=block_size, attn_gate_topk=0,
    )

    rel = (ref - out).norm() / ref.norm()
    assert rel < 1e-3, f"paged full vs PyTorch ref rel-err too high: {rel.item():.3e}"
    print(f"  paged full vs PyTorch ref: rel_err={rel.item():.2e}")


def test_paged_topk_gate():
    """Sparse Top-K path should stay close to the dense path when K is large."""
    if not torch.cuda.is_available():
        print("  [skip] no CUDA device")
        return
    try:
        from cda_vllm import cuda_cda_paged
    except ImportError as exc:
        print(f"  [skip] cda_vllm._cda_paged_kernels not built ({exc})")
        return

    torch.manual_seed(7)
    device = torch.device("cuda:0")
    B, N, D = 1, 256, 128
    block_size = 16
    k_pack = D // 4

    Q_rot = torch.randn(B, D, device=device, dtype=torch.float32)
    packed_K = torch.randint(0, 256, (B, N, k_pack), device=device, dtype=torch.uint8)
    norms_K = torch.randn(B, N, device=device, dtype=torch.float32).abs() + 0.1
    packed_V = torch.randint(0, 256, (B, N, k_pack), device=device, dtype=torch.uint8)
    norms_V = torch.randn(B, N, device=device, dtype=torch.float32).abs() + 0.1
    cb_k = torch.randn(4, device=device, dtype=torch.float32)
    cb_v = torch.randn(4, device=device, dtype=torch.float32)
    rotation = torch.eye(D, device=device, dtype=torch.float32)
    scale = 1.0 / math.sqrt(D)

    pk_blocks, nk_blocks, block_tables = _build_paged_layout(
        packed_K, norms_K, B, N, D, block_size
    )
    pv_blocks, nv_blocks, _ = _build_paged_layout(
        packed_V, norms_V, B, N, D, block_size
    )

    common = dict(
        Q_rot=Q_rot, packed_K_blocks=pk_blocks, norms_K_blocks=nk_blocks,
        packed_V_blocks=pv_blocks, norms_V_blocks=nv_blocks,
        block_tables=block_tables, codebook_k=cb_k, codebook_v=cb_v,
        rotation=rotation, scale=scale, N=N, block_size=block_size,
    )
    dense = cuda_cda_paged(**common, attn_gate_topk=0)
    sparse = cuda_cda_paged(**common, attn_gate_topk=N // 2)

    # TopK=N/2 should capture the bulk of the softmax mass; error bounded.
    rel = (dense - sparse).norm() / dense.norm()
    assert rel < 0.5, f"TopK sparse rel-err too high: {rel.item():.3f}"
    print(f"  paged Top-K@N/2 vs dense: rel_err={rel.item():.3e}")


TESTS = [
    ("paged vs PyTorch ref", test_paged_matches_reference),
    ("paged Top-K vs dense", test_paged_topk_gate),
]


def main() -> int:
    print("=== cda_vllm paged smoke tests ===")
    passed, failed = 0, 0
    for name, fn in TESTS:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{passed + failed} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
