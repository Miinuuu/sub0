"""Numerical fidelity: HMMA (K4V2) output vs FP16 FA2 reference.

Produces decode-time output MSE, cosine similarity, max|err|, L2 relative
error for a grid of (B, N) points on Llama-3.1-8B shapes (H_q=32, H_kv=8,
D=128).

Pipeline per trial:
  1. Sample random FP16 Q, K, V (seed-determined).
  2. Compute FP16 attention reference: softmax(QK^T / sqrt(d)) @ V.
  3. Quantise K, V via HadamardQuantCompressor (4-bit K, 2-bit V).
  4. Build the 104-B K4V2 paged cache slot.
  5. Run cda_decode_full_hmma_auto against that cache.
  6. Report error metrics relative to step 2.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("runs/hmma_vs_fa2_mse.json"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from core.compression import HadamardQuantCompressor
    from core.cda_attn import cda_decode_full_hmma_auto, choose_tile_n_hmma

    device = torch.device("cuda:0")
    H_q, H_kv, D = 32, 8, 128
    BLOCK = 16
    GROUP = H_q // H_kv  # 4

    # Pre-build compressors + codebooks (shared across trials)
    ck = HadamardQuantCompressor(dim=D, bit_width=4, half_rotation=True)
    cv = HadamardQuantCompressor(dim=D, bit_width=2, half_rotation=True)
    ck._ensure_tensors(device); cv._ensure_tensors(device)
    cb_k = (ck._centroids * 2.0 - 1.0).float().contiguous()
    cb_v = (cv._centroids * 2.0 - 1.0).float().contiguous()

    def trial(B, N, seed):
        g = torch.Generator(device=device).manual_seed(0xCDA + seed * 997 + N * 17 + B)
        # FP16 Q, K, V
        Q    = torch.randn(B, H_q, D, device=device, dtype=torch.float16, generator=g) * 0.3
        K_fp = torch.randn(B, H_kv, N, D, device=device, dtype=torch.float16, generator=g) * 0.3
        V_fp = torch.randn(B, H_kv, N, D, device=device, dtype=torch.float16, generator=g) * 0.3

        # --- FP16 FA2 reference: softmax(QK^T / sqrt(d)) @ V ---
        K_exp = K_fp.repeat_interleave(GROUP, dim=1)  # (B, H_q, N, D)
        V_exp = V_fp.repeat_interleave(GROUP, dim=1)
        scale = 1.0 / (D ** 0.5)
        scores = torch.einsum("bhd,bhnd->bhn", Q, K_exp) * scale
        probs = F.softmax(scores.float(), dim=-1)
        y_ref = torch.einsum("bhn,bhnd->bhd", probs, V_exp.float())  # FP32 output

        # --- Quantise K, V ---
        # quantize flattens last D; returns CompressedTensor with indices + norms.
        # Pass through whole K tensor (B, H_kv, N, D) reshaped to (-1, D).
        K_flat = K_fp.reshape(-1, D).contiguous()
        V_flat = V_fp.reshape(-1, D).contiguous()
        cK = ck.quantize(K_flat)
        cV = cv.quantize(V_flat)
        # cK.indices shape: (-1, D/2) for 4-bit, (-1, D/4) for 2-bit
        pK = cK.indices.view(B, H_kv, N, D // 2).contiguous()
        pV = cV.indices.view(B, H_kv, N, D // 4).contiguous()
        nK = cK.norms.float().view(B, H_kv, N).contiguous()
        nV = cV.norms.float().view(B, H_kv, N).contiguous()

        # --- Build 104-B paged K4V2 cache ---
        num_blk = (N + BLOCK - 1) // BLOCK
        total_slots = B * num_blk * BLOCK
        cache = torch.zeros(total_slots, H_kv, 104, dtype=torch.uint8, device=device)
        # Lay out as (B, num_blk * BLOCK, H_kv, 104) then pack bytes.
        for b in range(B):
            off = b * num_blk * BLOCK
            # Copy K/V bytes for slots t=0..N-1; padding slots stay zero.
            cache[off:off + N, :, :64]      = pK[b].transpose(0, 1)          # (N, H_kv, 64)
            cache[off:off + N, :, 64:96]    = pV[b].transpose(0, 1)          # (N, H_kv, 32)
            cache[off:off + N, :, 96:100]   = nK[b].transpose(0, 1).contiguous().view(torch.uint8).view(N, H_kv, 4)
            cache[off:off + N, :, 100:104]  = nV[b].transpose(0, 1).contiguous().view(torch.uint8).view(N, H_kv, 4)

        block_tbl = torch.arange(B * num_blk, dtype=torch.int32, device=device).view(B, num_blk)
        seq_lens = torch.full((B,), N, dtype=torch.int32, device=device)

        # --- Run HMMA ---
        tile_N = choose_tile_n_hmma(N, B)
        rot32 = ck._rotation.float().contiguous()
        rot16 = rot32.to(torch.float16).contiguous()
        y_cda_fp16 = torch.empty(B, H_q, D, dtype=torch.float16, device=device)
        cda_decode_full_hmma_auto(
            Q, cache, block_tbl, seq_lens, cb_k, cb_v,
            rot32, rot16, y_cda_fp16,
            GROUP, BLOCK, tile_N, num_blk * BLOCK, scale,
        )
        y_cda = y_cda_fp16.float()

        # --- Compare ---
        r = y_ref.reshape(-1)
        c = y_cda.reshape(-1)
        diff = c - r
        mse = (diff ** 2).mean().item()
        max_abs = diff.abs().max().item()
        l2_rel = (diff.norm() / r.norm()).item()
        cos = F.cosine_similarity(r, c, dim=0).item()
        return dict(mse=mse, max_abs=max_abs, l2_rel=l2_rel, cos=cos)

    results = {}
    for (B, N) in [(1, 1024), (1, 4096), (1, 8192), (1, 16384), (1, 32768), (1, 65536)]:
        torch.cuda.empty_cache()
        per_seed = []
        for seed in args.seeds:
            try:
                r = trial(B, N, seed)
                per_seed.append(r)
                print(f"  B={B} N={N:>6d} seed={seed}: mse={r['mse']:.3e} max={r['max_abs']:.3e} L2={r['l2_rel']:.4f} cos={r['cos']:.6f}")
            except Exception as e:
                print(f"  B={B} N={N} seed={seed}: FAILED {type(e).__name__}: {e}")
                per_seed.append({"error": f"{type(e).__name__}: {e}"})
        # Aggregate means
        valid = [r for r in per_seed if "error" not in r]
        if valid:
            avg = {k: sum(r[k] for r in valid) / len(valid) for k in ["mse", "max_abs", "l2_rel", "cos"]}
        else:
            avg = {}
        results[f"B{B}_N{N}"] = {"B": B, "N": N, "per_seed": per_seed, "mean": avg}
        print()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
