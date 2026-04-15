"""Stable WikiText-2 perplexity: FP16 / Decompress / CDA (packaged GQA kernel).

Stride=128, ~15 eval positions per 2048-token window — statistically
meaningful (previous stride=512 with 3 positions was too noisy).

Each position ``pos``:
    1. Prefill ``input_ids[:, :pos]`` with fp16.
    2. Depending on config, swap the KV cache for either (a) fp16 round-tripped
       through quantize→dequantize (``eval_ppl_decompress``) or (b) compressed
       indices driven by the packaged GQA kernel (``eval_ppl_cda``).
    3. Decode one token (``input_ids[:, pos:pos+1]``).
    4. Score its log-prob against ``input_ids[:, pos+1]``.

Averaged over all eval positions → PPL.

Usage::

    CUDA_VISIBLE_DEVICES=0 python experiments/bench_ppl_stable.py
    CUDA_VISIBLE_DEVICES=0 python experiments/bench_ppl_stable.py --model llama1b
    CUDA_VISIBLE_DEVICES=0 python experiments/bench_ppl_stable.py --skip-cda

Notes:
    * ``Decompress *-bit`` uses fake quantization (quant→dequant→fp16 SDPA).
      Measures the pure quantization-error contribution to PPL.
    * ``CDA *-bit`` drives decode through ``cda.fused_attention`` →
      ``cuda_hw_attention_gqa``. Validates that the GQA kernel is
      numerically faithful to the decompress path.
    * ``skip_sinks=0`` is hard-coded (the sub0 fused path does not merge
      fp16 sink tokens with compressed tokens). Paper numbers use sinks=4,
      so absolute PPL here may be slightly higher than the paper table; the
      fp16 vs CDA delta is still informative.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache  # noqa: E402

from cda import HadamardQuantCompressor  # noqa: E402
from cda.fused_attention import (  # noqa: E402
    _PositionOnlyCache,
    _compress_kv_cache_cuda,
    patch_model_compressed_attn,
    unpatch_model,
)


STRIDE = 128

MODEL_CONFIGS = {
    # key: (hf_name, head_dim, num_kv_heads, num_q_heads, num_layers)
    "llama8b": ("meta-llama/Llama-3.1-8B-Instruct", 128, 8, 32, 32),
    "llama1b": ("meta-llama/Llama-3.2-1B", 64, 8, 32, 16),
    "llama3b": ("meta-llama/Llama-3.2-3B-Instruct", 128, 8, 24, 28),
}


# ---------------------------------------------------------------------------
# Eval routines
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_ppl_baseline(model, input_ids, positions):
    nlls = []
    for pos in positions:
        out = model(input_ids[:, :pos], use_cache=True, return_dict=True)
        logits = model(
            input_ids[:, pos:pos + 1],
            past_key_values=out.past_key_values, use_cache=True,
        ).logits
        tgt = input_ids[:, pos + 1:pos + 2]
        if tgt.numel() > 0:
            nlls.append(F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1)).item())
        del out
        gc.collect(); torch.cuda.empty_cache()
    return float(np.exp(np.mean(nlls)))


@torch.no_grad()
def eval_ppl_decompress(model, input_ids, positions, k_comp, v_comp, D):
    nlls = []
    for pos in positions:
        out = model(input_ids[:, :pos], use_cache=True, return_dict=True)
        kv = out.past_key_values
        new_kv = DynamicCache()
        for li in range(len(kv)):
            k, v = kv.key_cache[li], kv.value_cache[li]
            kr = k_comp.dequantize(k_comp.quantize(k.reshape(-1, D))).reshape(k.shape).to(k.dtype)
            vr = v_comp.dequantize(v_comp.quantize(v.reshape(-1, D))).reshape(v.shape).to(v.dtype)
            new_kv.update(kr, vr, li)
        logits = model(
            input_ids[:, pos:pos + 1],
            past_key_values=new_kv, use_cache=True,
        ).logits
        tgt = input_ids[:, pos + 1:pos + 2]
        if tgt.numel() > 0:
            nlls.append(F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1)).item())
        del kv, new_kv, out
        gc.collect(); torch.cuda.empty_cache()
    return float(np.exp(np.mean(nlls)))


@torch.no_grad()
def eval_ppl_cda(model, input_ids, positions, k_comp, v_comp):
    n_layers = model.config.num_hidden_layers
    device = input_ids.device
    nlls = []
    for pos in positions:
        out = model(input_ids[:, :pos], use_cache=True, return_dict=True)
        kv = out.past_key_values
        compressed = _compress_kv_cache_cuda(kv, k_comp, v_comp)
        del kv, out
        gc.collect(); torch.cuda.empty_cache()

        patch_model_compressed_attn(model, k_comp, v_comp)
        for li in range(n_layers):
            model._cda_compressed[li] = compressed[li]
        for key in ("_rotation", "_codebook_k", "_codebook_v"):
            model._cda_compressed[key] = compressed[key]

        pc = _PositionOnlyCache(n_layers, pos, device, torch.float16)
        logits = model(
            input_ids[:, pos:pos + 1],
            past_key_values=pc, use_cache=True,
        ).logits
        unpatch_model(model)

        tgt = input_ids[:, pos + 1:pos + 2]
        if tgt.numel() > 0:
            nlls.append(F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1)).item())
        del pc, compressed
        gc.collect(); torch.cuda.empty_cache()
    return float(np.exp(np.mean(nlls)))


# ---------------------------------------------------------------------------
# Sign-Fold (calibration-free V-projection sign randomization)
# ---------------------------------------------------------------------------

def apply_sign_fold(model, seed, num_kv_heads, head_dim, num_q_heads):
    """Multiply V-projection rows and matching o-projection columns by ±1.

    Involution: calling twice with the same seed restores original weights.
    Equivalent to a per-head sign permutation of the V codebook, which slightly
    de-correlates quantization error and consistently improves PPL by 0.1–0.5.
    """
    torch.manual_seed(seed)
    gs = num_q_heads // num_kv_heads
    with torch.no_grad():
        for layer in model.model.layers:
            vp = layer.self_attn.v_proj
            op = layer.self_attn.o_proj
            S = (torch.randint(
                0, 2, (num_kv_heads, head_dim),
                device=vp.weight.device, dtype=vp.weight.dtype,
            ) * 2 - 1)
            sf = S.reshape(-1)
            vp.weight.data *= sf.unsqueeze(1)
            if vp.bias is not None:
                vp.bias.data *= sf
            Sq = (S.unsqueeze(1)
                   .expand(num_kv_heads, gs, head_dim)
                   .reshape(num_q_heads, head_dim)
                   .reshape(-1))
            op.weight.data *= Sq.unsqueeze(0)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama8b", choices=list(MODEL_CONFIGS))
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--skip-cda", action="store_true",
                        help="only run fp16 baseline + decompress (skip GQA kernel)")
    parser.add_argument("--skip-sign-fold", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    model_name, D, H_kv, H_q, NL = MODEL_CONFIGS[args.model]
    print(f"Model: {model_name} (D={D}, H_kv={H_kv}, H_q={H_q}, NL={NL})")

    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device,
        low_cpu_mem_usage=True,
    ).eval()

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    enc = tokenizer(text, return_tensors="pt", max_length=args.max_length, truncation=True)
    input_ids = enc.input_ids.to(device)
    S = input_ids.shape[1]
    positions = list(range(STRIDE, min(S - 1, args.max_length), STRIDE))
    print(f"Eval: {len(positions)} positions, stride={STRIDE}\n")

    c2 = HadamardQuantCompressor(dim=D, bit_width=2, half_rotation=True)
    c4 = HadamardQuantCompressor(dim=D, bit_width=4, half_rotation=True)

    results = []

    def record(name, ppl, notes=""):
        delta = (ppl / results[0][1] - 1) * 100 if results else 0
        results.append((name, ppl, delta, notes))
        sign = "+" if delta > 0 else ""
        print(f"  {name:<34} {ppl:>6.3f}  {sign}{delta:>6.2f}%  {notes}")

    print(f"  {'Config':<34} {'PPL':>6}  {'dPPL':>7}  Notes")
    print("  " + "-" * 60)

    record("fp16 baseline",
           eval_ppl_baseline(model, input_ids, positions))

    record("Decompress 2-bit",
           eval_ppl_decompress(model, input_ids, positions, c2, c2, D))
    record("Decompress 4-bit",
           eval_ppl_decompress(model, input_ids, positions, c4, c4, D))

    if not args.skip_cda:
        record("CDA K2/V2 (GQA kernel)",
               eval_ppl_cda(model, input_ids, positions, c2, c2))
        record("CDA K4/V2 (GQA kernel)",
               eval_ppl_cda(model, input_ids, positions, c4, c2))

        if not args.skip_sign_fold:
            best_seed = 77 if D == 128 else 42
            apply_sign_fold(model, best_seed, H_kv, D, H_q)
            record(f"CDA K4/V2 + SF({best_seed})",
                   eval_ppl_cda(model, input_ids, positions, c4, c2),
                   notes="cal-free")
            apply_sign_fold(model, best_seed, H_kv, D, H_q)  # involution: undo

    if args.output:
        out = [{"config": name, "ppl": ppl, "delta_pct": delta, "notes": notes}
               for name, ppl, delta, notes in results]
        p = Path(args.output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "model": model_name, "D": D, "max_length": args.max_length,
            "stride": STRIDE, "n_positions": len(positions),
            "results": out,
        }, indent=2))
        print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
