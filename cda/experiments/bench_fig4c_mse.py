"""MSE verification for CDA kernels vs FP16 reference attention.

Separates the two error sources:

  1. **Quantization error** — CDA-SW (dequantize → FP16 attention)
     vs true FP16 attention.  Bounded by the codebook bit-width.
  2. **Kernel error** — CDA-HW (compiled CUDA kernel)
     vs CDA-SW.  Should be ~round-off; confirms kernel correctness.
  3. **Total error** = kernel + quantization.

For each (k_bits, v_bits, topk, contiguous/paged) config, reports:
  * max|·|, mean|·|, MSE, cosine-sim against FP16 reference.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.compression import HadamardQuantCompressor  # noqa: E402
from core.cda_attn import (  # noqa: E402
    cuda_hw_attention_gqa,
    cuda_hw_attention_gqa_paged,
    pack_kv_to_blocks,
)


def fp16_attention(Q_fp16, K_fp16, V_fp16, group_size):
    """Standard FP16 GQA attention. Returns (H_q, D)."""
    Q_f = Q_fp16.float()
    K_f = K_fp16.float().repeat_interleave(group_size, dim=0)   # (H_q, N, D)
    V_f = V_fp16.float().repeat_interleave(group_size, dim=0)
    D = Q_f.shape[-1]
    scale = 1.0 / (D ** 0.5)
    scores = (Q_f.unsqueeze(1) @ K_f.transpose(-1, -2)).squeeze(1) * scale  # (H_q, N)
    attn = torch.softmax(scores, dim=-1)
    out = (attn.unsqueeze(1) @ V_f).squeeze(1)                              # (H_q, D)
    return out


def cda_sw_attention(Q, K_dq, V_dq, group_size, topk=0):
    """CDA-SW: softmax(Q · K_dq^T) · V_dq using the *dequantized* KV.
    Isolates quantization error from kernel error.
    """
    K_e = K_dq.repeat_interleave(group_size, dim=0)
    V_e = V_dq.repeat_interleave(group_size, dim=0)
    D = Q.shape[-1]
    scale = 1.0 / (D ** 0.5)
    scores = (Q.unsqueeze(1) @ K_e.transpose(-1, -2)).squeeze(1) * scale
    attn = torch.softmax(scores, dim=-1)
    if topk > 0 and topk < attn.shape[-1]:
        vals, idx = attn.topk(topk, dim=-1)
        vals = vals / (vals.sum(dim=-1, keepdim=True) + 1e-12)
        out = torch.zeros_like(Q)
        for h in range(Q.shape[0]):
            out[h] = vals[h] @ V_e[h, idx[h]]
        return out
    return (attn.unsqueeze(1) @ V_e).squeeze(1)


def _metrics(ref, test):
    ref = ref.float(); test = test.float()
    diff = (ref - test).flatten()
    mse = (diff ** 2).mean().item()
    cos = torch.nn.functional.cosine_similarity(
        ref.flatten().unsqueeze(0), test.flatten().unsqueeze(0)).item()
    return {
        "max": diff.abs().max().item(),
        "mean": diff.abs().mean().item(),
        "mse": mse,
        "cos": cos,
    }


def run_config(k_bits, v_bits, topk, paged=False, block_size=16,
               H_q=32, H_kv=8, D=128, N=1024, device="cuda:0"):
    group_size = H_q // H_kv
    dev = torch.device(device)

    torch.manual_seed(0)
    # Realistic-ish FP16 inputs
    Q_fp16 = torch.randn(H_q, D, device=dev, dtype=torch.float16)
    K_fp16 = torch.randn(H_kv, N, D, device=dev, dtype=torch.float16)
    V_fp16 = torch.randn(H_kv, N, D, device=dev, dtype=torch.float16)

    # --- FP16 reference ---
    out_fp16 = fp16_attention(Q_fp16, K_fp16, V_fp16, group_size)

    # --- Compress KV via the canonical quantizer ---
    k_comp = HadamardQuantCompressor(dim=D, bit_width=k_bits, half_rotation=True)
    v_comp = HadamardQuantCompressor(dim=D, bit_width=v_bits, half_rotation=True)
    k_comp._ensure_tensors(dev); v_comp._ensure_tensors(dev)

    K_flat = K_fp16.reshape(-1, D)
    V_flat = V_fp16.reshape(-1, D)
    cK = k_comp.quantize(K_flat)
    cV = v_comp.quantize(V_flat)

    # Dequantized KV for SW path
    K_dq = k_comp.dequantize(cK).reshape(H_kv, N, D).float()
    V_dq = v_comp.dequantize(cV).reshape(H_kv, N, D).float()

    # --- CDA-SW: FP16 attention using dequantized KV ---
    Q_f = Q_fp16.float()
    out_sw = cda_sw_attention(Q_f, K_dq, V_dq, group_size, topk=topk)

    # --- CDA-HW: the actual CUDA kernel ---
    pack_k = cK.indices.shape[-1]
    pack_v = cV.indices.shape[-1]
    pK = cK.indices.view(1, H_kv, N, pack_k)[0].contiguous()
    nK = cK.norms.float().view(1, H_kv, N)[0].contiguous()
    pV = cV.indices.view(1, H_kv, N, pack_v)[0].contiguous()
    nV = cV.norms.float().view(1, H_kv, N)[0].contiguous()

    rotation = k_comp._rotation.float().contiguous()
    cb_k = (k_comp._centroids * 2.0 - 1.0).float().contiguous()
    cb_v = (v_comp._centroids * 2.0 - 1.0).float().contiguous()
    Q_rot = Q_f @ rotation
    scale = 1.0 / (D ** 0.5)

    if paged:
        layer = dict(packed_K=pK, norms_K=nK, packed_V=pV, norms_V=nV,
                     H_kv=H_kv, N=N, D=D,
                     bit_width_k=k_bits, bit_width_v=v_bits)
        p = pack_kv_to_blocks(layer, block_size=block_size)
        out_hw = cuda_hw_attention_gqa_paged(
            Q_rot, p["packed_K"], p["norms_K"], p["packed_V"], p["norms_V"],
            p["block_tables"], cb_k, cb_v, rotation, scale, N, group_size,
            block_size=block_size, k_bits=k_bits, v_bits=v_bits, topk=topk,
        )
    else:
        out_hw = cuda_hw_attention_gqa(
            Q_rot, pK, nK, pV, nV, cb_k, cb_v, rotation, scale, N, group_size,
            k_bits=k_bits, v_bits=v_bits, topk=topk,
        )

    quant = _metrics(out_fp16, out_sw)
    kernel = _metrics(out_sw, out_hw)
    total = _metrics(out_fp16, out_hw)
    return quant, kernel, total


def main():
    configs = [(k, v, tk, pg)
               for k in (2, 4) for v in (2, 4)
               for tk in (0, 128)
               for pg in (False, True)]

    hdr = (f"{'config':<20} {'layout':<10}"
           f" | {'quant MSE':>10} {'quant cos':>9}"
           f" | {'kernel max':>11} {'kernel MSE':>11} {'kernel cos':>10}"
           f" | {'total MSE':>10} {'total cos':>9}")
    print(hdr); print("-" * len(hdr))

    for k, v, tk, pg in configs:
        q, ker, tot = run_config(k, v, tk, paged=pg)
        lbl = f"K{k}V{v} tk={tk:>3}"
        layout = "paged" if pg else "contig"
        print(f"{lbl:<20} {layout:<10}"
              f" | {q['mse']:>10.3e} {q['cos']:>9.5f}"
              f" | {ker['max']:>11.3e} {ker['mse']:>11.3e} {ker['cos']:>10.5f}"
              f" | {tot['mse']:>10.3e} {tot['cos']:>9.5f}")


if __name__ == "__main__":
    main()
