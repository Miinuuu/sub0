"""Multi-seed rotation ablation — Hadamard vs Random orthogonal vs Identity.

Resolves Codex Round-2 W2 / Round-3 / Round-4: prior single-seed result
showed Hadamard 0.225 PPL behind random orthogonal on Llama-3.1-8B
K4V4 — a gap that cannot be dismissed as seed noise without multi-seed
evidence. This script runs N independent Haar-random orthogonal seeds
and reports mean ± std vs Hadamard's deterministic point estimate.

Outputs (JSON):
  {
    "fp16": {"ppl": ...},
    "identity": {"ppl": ...},
    "hadamard": {"ppl": ...},
    "random_orthogonal_seeds": [
      {"seed": 0, "ppl": ...},
      ...
    ],
    "random_orthogonal_summary": {"mean": ..., "std": ..., "min": ..., "max": ...}
  }

Usage (single GPU):
    CUDA_VISIBLE_DEVICES=0 python benchmarks/paper/abl_rotation_multiseed.py \
        --n-seeds 5 \
        --output runs/paper/abl_rotation_multiseed.json

Imports cda-v1's compressor + ppl_eval (which produced the existing
v1_paper data). Quantizer is shared between v1 and v2 per CLAUDE.md.

NOTE: requires the cda-v1 codebase on sys.path (not redistributed in
v2). Set CDA_V1_PATH env var or edit the path below to point at a v1
clone. Backing data is preserved under
``runs/baselines/ablation_rotation_alternatives*.json``.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

# Add cda-v1 path for HadamardQuantCompressor + ppl_eval
CDA_V1 = Path(os.environ.get("CDA_V1_PATH", "../cda"))
if str(CDA_V1) not in sys.path:
    sys.path.insert(0, str(CDA_V1))

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from core.compression import HadamardQuantCompressor
from core.ppl_eval import eval_ppl


def _make_compress_fn(model, kc, vc, skip_sinks, n_layers):
    def fn(ctx, nxt):
        out = model(ctx, use_cache=True, return_dict=True)
        kv_cache = out.past_key_values

        new_kv = DynamicCache()
        for li in range(n_layers):
            if hasattr(kv_cache, "layers") and len(kv_cache.layers) > li:
                k = kv_cache.layers[li].keys
                v = kv_cache.layers[li].values
            else:
                k = kv_cache.key_cache[li]
                v = kv_cache.value_cache[li]
            B, H, S, D = k.shape
            if skip_sinks > 0 and S > skip_sinks:
                k_sink = k[:, :, :skip_sinks, :]
                v_sink = v[:, :, :skip_sinks, :]
                k_rest = k[:, :, skip_sinks:, :].reshape(-1, D)
                v_rest = v[:, :, skip_sinks:, :].reshape(-1, D)
                ko_rest = kc.dequantize(kc.quantize(k_rest)).reshape(B, H, S - skip_sinks, D)
                vo_rest = vc.dequantize(vc.quantize(v_rest)).reshape(B, H, S - skip_sinks, D)
                ko = torch.cat([k_sink, ko_rest], dim=2)
                vo = torch.cat([v_sink, vo_rest], dim=2)
            else:
                ko = kc.dequantize(kc.quantize(k.reshape(-1, D))).reshape(B, H, S, D)
                vo = vc.dequantize(vc.quantize(v.reshape(-1, D))).reshape(B, H, S, D)
            new_kv.update(ko, vo, li)
        return model(nxt, past_key_values=new_kv, use_cache=False).logits
    return fn


def _haar_orthogonal(d: int, dtype, device, seed: int):
    """Haar-random orthogonal matrix via QR of Gaussian iid (sign-corrected)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(d, d, generator=g, dtype=torch.float32)
    Q, R = torch.linalg.qr(A)
    sign = torch.sign(torch.diag(R))
    Q = Q * sign.unsqueeze(0)
    return Q.to(dtype=dtype, device=device)


def run_variant(model, tokenizer, n_layers, head_dim, label, *, rotation_kind, seed=None):
    kc = HadamardQuantCompressor(dim=head_dim, bit_width=4, half_rotation=True)
    vc = HadamardQuantCompressor(dim=head_dim, bit_width=4, half_rotation=True)
    kc._ensure_tensors(model.device); vc._ensure_tensors(model.device)

    if rotation_kind == "identity":
        eye = torch.eye(head_dim, dtype=kc._rotation.dtype, device=model.device)
        kc._rotation = eye; kc._rotation_t = eye
        vc._rotation = eye; vc._rotation_t = eye
    elif rotation_kind == "random_orthogonal":
        assert seed is not None, "random_orthogonal requires explicit seed"
        Q = _haar_orthogonal(head_dim, kc._rotation.dtype, model.device, seed=seed)
        kc._rotation = Q; kc._rotation_t = Q.T.contiguous()
        vc._rotation = Q; vc._rotation_t = Q.T.contiguous()
    elif rotation_kind == "hadamard":
        pass  # Default Hadamard already set by compressor
    else:
        raise ValueError(f"Unknown rotation_kind={rotation_kind}")

    fn = _make_compress_fn(model, kc, vc, skip_sinks=4, n_layers=n_layers)
    return eval_ppl(model, tokenizer, fn, max_ctx=4096, stride=64, desc=label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Number of random-orthogonal seeds to evaluate.")
    parser.add_argument("--seed-base", type=int, default=0xCDA,
                        help="Base seed; actual seeds = base, base+1, ..., base+n-1.")
    parser.add_argument("--output", required=True,
                        help="Output JSON path.")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.llm_name} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.llm_name, torch_dtype=torch.float16,
        device_map="auto", low_cpu_mem_usage=True,
    ).eval()
    head_dim = getattr(model.config, "head_dim", 128)
    n_layers = model.config.num_hidden_layers
    print(f"  Loaded in {time.time()-t0:.0f}s. head_dim={head_dim}, n_layers={n_layers}")

    results = {}

    print("\n=== FP16 baseline ===")
    def fp16_fn(ctx, nxt):
        out = model(ctx, use_cache=True, return_dict=True)
        return model(nxt, past_key_values=out.past_key_values, use_cache=False).logits
    results["fp16"] = eval_ppl(model, tokenizer, fp16_fn,
                                max_ctx=4096, stride=64, desc="FP16")
    print(f"  FP16 PPL = {results['fp16']['ppl']:.4f}")

    print("\n=== CDA K4V4 (Hadamard) ===")
    results["hadamard"] = run_variant(
        model, tokenizer, n_layers, head_dim,
        "CDA K4V4 (Hadamard)", rotation_kind="hadamard")
    print(f"  Hadamard PPL = {results['hadamard']['ppl']:.4f}")

    print("\n=== CDA K4V4 (identity) ===")
    results["identity"] = run_variant(
        model, tokenizer, n_layers, head_dim,
        "CDA K4V4 (identity)", rotation_kind="identity")
    print(f"  Identity PPL = {results['identity']['ppl']:.4f}")

    print(f"\n=== CDA K4V4 (Haar-random orthogonal × {args.n_seeds} seeds) ===")
    seed_results = []
    for i in range(args.n_seeds):
        seed = args.seed_base + i
        print(f"  seed {seed} ...")
        r = run_variant(
            model, tokenizer, n_layers, head_dim,
            f"CDA K4V4 (random orth., seed={seed})",
            rotation_kind="random_orthogonal", seed=seed)
        seed_results.append({"seed": seed, "ppl": r["ppl"], "n_pos": r["n_pos"]})
        print(f"    PPL = {r['ppl']:.4f}")
        # Incremental save
        results["random_orthogonal_seeds"] = seed_results
        out_path.write_text(json.dumps(results, indent=2))

    # Summary stats
    rand_ppls = [s["ppl"] for s in seed_results]
    rand_tensor = torch.tensor(rand_ppls)
    results["random_orthogonal_summary"] = {
        "n_seeds": len(rand_ppls),
        "mean": float(rand_tensor.mean()),
        "std": float(rand_tensor.std(unbiased=True)) if len(rand_ppls) > 1 else 0.0,
        "min": float(rand_tensor.min()),
        "max": float(rand_tensor.max()),
    }

    print("\n" + "=" * 72)
    print(f"Multi-seed rotation ablation — {args.llm_name}, K4V4, WikiText-2")
    print("=" * 72)
    fp16 = results["fp16"]["ppl"]
    print(f"  FP16 baseline: PPL = {fp16:.4f}")
    print(f"  Identity:      PPL = {results['identity']['ppl']:.4f}  (Δ = {results['identity']['ppl']-fp16:+.4f})")
    print(f"  Hadamard:      PPL = {results['hadamard']['ppl']:.4f}  (Δ = {results['hadamard']['ppl']-fp16:+.4f})")
    s = results["random_orthogonal_summary"]
    print(f"  Random ortho:  PPL = {s['mean']:.4f} ± {s['std']:.4f}  "
          f"(min {s['min']:.4f}, max {s['max']:.4f}, n={s['n_seeds']})")
    print(f"  Hadamard − Random mean: {results['hadamard']['ppl'] - s['mean']:+.4f} PPL")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
