"""PPL benchmark — Table 3 reproduction (paper-canonical protocol).

All table methods (FP16, KIVI, GEAR, CDA, TurboQuant, QuaRot-style) with
multi-GPU via ``device_map="auto"``. Outputs JSON per model per dataset.

Ported from cda-v1 ``experiments/bench_ppl_scale.py``. The protocol
(stride=64, max_ctx=4096, filter_empty_lines, add_special_tokens) is
**locked** to the paper's canonical settings via ``cda.eval.ppl``.

Usage:
    # Local 2x A6000 — Llama-3.1-8B, WikiText-2
    CUDA_VISIBLE_DEVICES=0,1 python benchmarks/bench_ppl_scale.py \\
        --model llama8b --dataset wikitext2

    # Lab server — Llama-3.1-70B, all datasets
    python benchmarks/bench_ppl_scale.py \\
        --model llama70b --dataset wikitext2 c4

    # Specific methods only
    python benchmarks/bench_ppl_scale.py \\
        --model qwen32b-inst --methods fp16 cda-4 kivi-4 quarot-4
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from cda.eval.ppl import get_wikitext_text, get_c4_text


# ── Model registry (paper-canonical) ────────────────────────────────
MODELS = {
    "llama2-7b":     "meta-llama/Llama-2-7b-hf",
    "mistral7b":     "mistralai/Mistral-7B-v0.3",
    "llama8b":       "meta-llama/Llama-3.1-8B-Instruct",
    "llama2-13b":    "meta-llama/Llama-2-13b-hf",
    "llama2-70b":    "meta-llama/Llama-2-70b-hf",
    "qwen32b":       "Qwen/Qwen2.5-32B",
    "qwen32b-inst":  "Qwen/Qwen2.5-32B-Instruct",
    "llama70b":      "meta-llama/Llama-3.1-70B-Instruct",
}

ALL_METHODS = [
    "fp16",
    "kivi-4", "kivi-2",
    "gear-4", "gear-2",
    "cda-4", "cda-2",
    "quarot-4",
]


def load_dataset_text(name: str) -> str:
    if name == "wikitext2":
        return get_wikitext_text(filter_empty_lines=True)
    if name == "c4":
        return get_c4_text(max_docs=500)
    raise ValueError(f"Unknown dataset: {name}")


# ── Core PPL evaluation ─────────────────────────────────────────────
def eval_ppl(model, input_ids, method_fn, positions, desc=""):
    """Evaluate PPL at given positions using ``method_fn``."""
    nlls = []
    t0 = time.time()
    for i, pos in enumerate(positions):
        ctx = input_ids[:, :pos]
        nxt = input_ids[:, pos:pos + 1]
        tgt = input_ids[:, pos + 1:pos + 2]
        if tgt.numel() == 0:
            continue
        with torch.no_grad():
            logits = method_fn(ctx, nxt, pos)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
            nlls.append(loss.item())
        gc.collect()
        torch.cuda.empty_cache()
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            ppl_so_far = np.exp(np.mean(nlls))
            print(f"    [{desc}] {i+1}/{len(positions)} "
                  f"PPL={ppl_so_far:.2f} ({elapsed:.0f}s)")
    ppl = np.exp(np.mean(nlls)) if nlls else float("inf")
    elapsed = time.time() - t0
    print(f"  {desc:20s} PPL={ppl:.4f}  ({len(nlls)} pos, {elapsed:.0f}s)")
    return {"ppl": round(float(ppl), 4),
            "n_pos": len(nlls),
            "time_s": round(elapsed, 1)}


# ── Method implementations ──────────────────────────────────────────
def run_fp16(model, input_ids, positions):
    def fn(ctx, nxt, pos):
        out = model(ctx, use_cache=True, return_dict=True)
        return model(nxt, past_key_values=out.past_key_values, use_cache=False).logits
    return eval_ppl(model, input_ids, fn, positions, desc="FP16")


def run_kivi(model, input_ids, positions, nbits):
    """KIVI-style quantization via transformers quanto backend."""
    try:
        from transformers.cache_utils import QuantoQuantizedCache
    except ImportError:
        print(f"  KIVI-{nbits}bit: quanto not available, skipping")
        return None

    def fn(ctx, nxt, pos):
        cache = QuantoQuantizedCache(model.config, nbits=nbits,
                                     axis_key=0, axis_value=0,
                                     q_group_size=32, residual_length=128)
        out = model(ctx, use_cache=True, return_dict=True, past_key_values=cache)
        return model(nxt, past_key_values=out.past_key_values, use_cache=False).logits
    return eval_ppl(model, input_ids, fn, positions, desc=f"KIVI-{nbits}bit")


def _load_gear_funcs():
    """Locate GEAR's compress_function. Prefers REF/ inside cda-v2."""
    candidates = [
        PROJECT_ROOT / "REF" / "GEAR" / "GenerationBench" / "GenerationTest"
            / "GEARLM" / "Simulated" / "compress_function.py",
        Path(os.path.expanduser(
            "~/ing/research/REF/GEAR/GenerationBench/GenerationTest/GEARLM/"
            "Simulated/compress_function.py")),
    ]
    for p in candidates:
        if p.exists():
            spec = importlib.util.spec_from_file_location("gear_compress", str(p))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return None


def run_gear(model, input_ids, positions, nbits, n_layers):
    mod = _load_gear_funcs()
    if mod is None:
        print(f"  GEAR-{nbits}bit: GEAR repo not found, skipping")
        return None

    gear_ch = mod.gearslkivi_channelQ_new
    gear_tok = mod.gearslkivi_tokenQ_new
    group_size = 128
    sparsity = 0.02
    rank, rankv, loop = 2, 2, 3

    def _get_kv(kv, li):
        if hasattr(kv, "layers"):
            return kv.layers[li].keys, kv.layers[li].values
        return kv.key_cache[li], kv.value_cache[li]

    def fn(ctx, nxt, pos):
        out = model(ctx, use_cache=True, return_dict=True)
        kv = out.past_key_values
        new_kv = DynamicCache()
        for li in range(n_layers):
            k, v = _get_kv(kv, li)
            k_q = gear_ch(k, nbits, group_size, sparsity=sparsity,
                          rank=rank, loop=loop).to(k.dtype)
            v_q = gear_tok(v, nbits, group_size, sparsity=sparsity,
                           rank=rankv, loop=loop).to(v.dtype)
            new_kv.update(k_q, v_q, li)
        return model(nxt, past_key_values=new_kv, use_cache=False).logits
    return eval_ppl(model, input_ids, fn, positions, desc=f"GEAR-{nbits}bit")


# ── CDA via cda-v2 algorithm.Compressor ─────────────────────────────
def _make_cda_compress_fn(model, k_comp, v_comp, skip_sinks, n_layers):
    """SW reference: encode → decode (round-trip via cda-v2 Compressor),
    then run the standard fp16 attention. Mimics the kernel's distortion.

    transformers 4.57+ exposes per-layer K/V as ``kv.layers[i].keys`` /
    ``kv.layers[i].values``.
    """
    def _get_kv(kv, li):
        if hasattr(kv, "layers"):
            return kv.layers[li].keys, kv.layers[li].values
        return kv.key_cache[li], kv.value_cache[li]

    D = k_comp.dim

    def _round_trip(comp, x):
        """Encode then decode along last dim ``D``. Returns same shape."""
        flat = x.reshape(-1, D).contiguous()
        # move compressor to data device on first call
        if comp.device != flat.device:
            comp = comp.to(flat.device)
        slot = comp.encode(flat)
        out = comp.decode(slot, dtype=flat.dtype)
        return out.reshape(x.shape), comp

    def fn(ctx, nxt, pos):
        nonlocal k_comp, v_comp
        out = model(ctx, use_cache=True, return_dict=True)
        kv = out.past_key_values
        new_kv = DynamicCache()
        for li in range(n_layers):
            k, v = _get_kv(kv, li)
            B, H, S, _ = k.shape
            if skip_sinks > 0 and S > skip_sinks:
                kk, kr = k[:, :, :skip_sinks], k[:, :, skip_sinks:]
                vk, vr = v[:, :, :skip_sinks], v[:, :, skip_sinks:]
                kr2, k_comp = _round_trip(k_comp, kr)
                vr2, v_comp = _round_trip(v_comp, vr)
                new_kv.update(torch.cat([kk, kr2], 2),
                              torch.cat([vk, vr2], 2), li)
            else:
                ko, k_comp = _round_trip(k_comp, k)
                vo, v_comp = _round_trip(v_comp, v)
                new_kv.update(ko, vo, li)
        return model(nxt, past_key_values=new_kv, use_cache=False).logits
    return fn


def run_cda(model, input_ids, positions, head_dim, n_layers,
            k_bits, v_bits, skip_sinks=4):
    from cda.algorithm.compression import Compressor
    kc = Compressor(dim=head_dim, num_levels=(16 if k_bits == 4 else 4))
    vc = Compressor(dim=head_dim, num_levels=(16 if v_bits == 4 else 4))
    tag = f"CDA-K{k_bits}V{v_bits}" if k_bits != v_bits else f"CDA-{k_bits}bit"
    fn = _make_cda_compress_fn(model, kc, vc, skip_sinks, n_layers)
    return eval_ppl(model, input_ids, fn, positions, desc=tag)


def run_quarot_style(model, input_ids, positions, head_dim, n_layers,
                     k_bits=4, v_bits=4, skip_sinks=4):
    """QuaRot-style: Hadamard rotation + uniform RTN (no Lloyd-Max)."""
    from cda.algorithm.compression import Compressor
    kc = Compressor(dim=head_dim,
                     num_levels=(16 if k_bits == 4 else 4),
                     codebook_mode="uniform")
    vc = Compressor(dim=head_dim,
                     num_levels=(16 if v_bits == 4 else 4),
                     codebook_mode="uniform")
    fn = _make_cda_compress_fn(model, kc, vc, skip_sinks, n_layers)
    tag = (f"QuaRot-style-K{k_bits}V{v_bits}"
           if k_bits != v_bits else f"QuaRot-style-{k_bits}bit")
    return eval_ppl(model, input_ids, fn, positions, desc=tag)


# ── Main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help=f"Model alias or HF path. Aliases: {list(MODELS.keys())}")
    parser.add_argument("--dataset", nargs="+", default=["wikitext2"],
                        choices=["wikitext2", "c4"])
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS,
                        choices=ALL_METHODS)
    parser.add_argument("--max-ctx", type=int, default=4096)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--skip-sinks", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    model_name = MODELS.get(args.model, args.model)
    model_alias = args.model if args.model in MODELS else model_name.split("/")[-1]

    if args.output_dir is None:
        args.output_dir = str(PROJECT_ROOT / "runs" / f"ppl_scale_{model_alias}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading {model_name} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  Loaded in {time.time()-t0:.0f}s")

    head_dim = getattr(model.config, "head_dim", None)
    if head_dim is None:
        head_dim = model.config.hidden_size // model.config.num_attention_heads
    n_layers = model.config.num_hidden_layers
    print(f"  head_dim={head_dim}, layers={n_layers}, "
          f"params={sum(p.numel() for p in model.parameters())/1e9:.1f}B")

    assert head_dim == 128, f"head_dim={head_dim}, codebook requires 128"

    all_results = {}

    for ds_name in args.dataset:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        text = load_dataset_text(ds_name)
        enc = tokenizer(text, return_tensors="pt",
                        max_length=args.max_ctx, truncation=True)
        first_device = next(model.parameters()).device
        input_ids = enc.input_ids.to(first_device)
        seq_len = input_ids.shape[1]

        positions = list(range(args.stride,
                               min(seq_len - 1, args.max_ctx),
                               args.stride))
        print(f"  Tokens: {seq_len}, eval positions: {len(positions)}")

        results = {}

        for method in args.methods:
            gc.collect()
            torch.cuda.empty_cache()
            print(f"\n── {method} ──")

            try:
                if method == "fp16":
                    r = run_fp16(model, input_ids, positions)
                elif method == "kivi-4":
                    r = run_kivi(model, input_ids, positions, 4)
                elif method == "kivi-2":
                    r = run_kivi(model, input_ids, positions, 2)
                elif method == "gear-4":
                    r = run_gear(model, input_ids, positions, 4, n_layers)
                elif method == "gear-2":
                    r = run_gear(model, input_ids, positions, 2, n_layers)
                elif method == "cda-4":
                    r = run_cda(model, input_ids, positions, head_dim, n_layers,
                                4, 4, args.skip_sinks)
                elif method == "cda-2":
                    r = run_cda(model, input_ids, positions, head_dim, n_layers,
                                2, 2, args.skip_sinks)
                elif method == "quarot-4":
                    r = run_quarot_style(model, input_ids, positions, head_dim,
                                          n_layers, k_bits=4, v_bits=4,
                                          skip_sinks=args.skip_sinks)
                else:
                    r = None

                if r is not None:
                    r["method"] = method
                    r["model"] = model_alias
                    r["dataset"] = ds_name
                    r["protocol"] = {
                        "harness": "benchmarks.bench_ppl_scale",
                        "max_ctx": args.max_ctx,
                        "stride": args.stride,
                        "skip_sinks": args.skip_sinks,
                        "filter_empty_lines": True,
                        "add_special_tokens": True,
                    }
                    results[method] = r
                    with open(out_dir / f"ppl_{ds_name}.json", "w") as f:
                        json.dump(results, f, indent=2)

            except Exception as e:
                print(f"  {method} FAILED: {e}")
                import traceback
                traceback.print_exc()

        all_results[ds_name] = results

    print(f"\n{'='*60}")
    print(f"Summary: {model_alias}")
    print(f"{'='*60}")
    for ds_name, results in all_results.items():
        print(f"\n  {ds_name}:")
        for method, r in results.items():
            print(f"    {method:20s}  PPL={r['ppl']:.4f}")

    summary = {
        "model": model_name,
        "alias": model_alias,
        "head_dim": head_dim,
        "n_layers": n_layers,
        "max_ctx": args.max_ctx,
        "stride": args.stride,
        "results": all_results,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
