"""Long-context PPL with TopK128 — using CDA core (manual_decode_step).

Measures K4/V2 and K2/V2 PPL with TopK128 gating at multiple context lengths.
Uses _compress_kv_cache + manual_decode_step from core, NOT monkey-patch SDPA.

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/bench_longctx_ppl_topk.py \
        --max-length 8192 16384 32768 65536
"""
from __future__ import annotations

import argparse, json, gc, os, sys, time
from pathlib import Path

import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoModelForCausalLM, AutoTokenizer
from core.compression import HadamardQuantCompressor
from core.compressed_generate import _compress_kv_cache, manual_decode_step


def run_one_length(model, input_ids, configs, max_len, stride, skip_sinks, topk):
    ids = input_ids[:, :max_len]
    seq_len = ids.shape[1]
    positions = list(range(stride, min(seq_len - 1, max_len), stride))
    n_pos = len(positions)
    print(f"\n--- max_length={max_len}, {n_pos} positions, topk={topk} ---")

    CHUNK = 4096
    nlls = {name: [] for name in configs}
    kv_cache = None
    last_end = 0
    t0 = time.time()

    for idx, pos in enumerate(positions):
        with torch.no_grad():
            # Extend uncompressed cache (backbone only)
            for start in range(last_end, pos, CHUNK):
                end = min(start + CHUNK, pos)
                out = model.model(input_ids=ids[:, start:end], past_key_values=kv_cache,
                                  use_cache=True, return_dict=True)
                kv_cache = out.past_key_values
                del out

            nxt = ids[:, pos:pos+1]
            tgt = ids[:, pos+1:pos+2]
            if tgt.numel() == 0:
                continue

            # CDA eval with TopK for each config
            for name, (kc, vc) in configs.items():
                compressed = _compress_kv_cache(kv_cache, kc, vc, skip_sinks=skip_sinks)
                logits = manual_decode_step(model, nxt, compressed, kc, vc,
                                           position=pos, attn_gate_topk=topk)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), tgt.view(-1))
                nlls[name].append(loss.item())
                del compressed, logits

            torch.cuda.empty_cache()

            # Extend cache for next position
            out = model.model(input_ids=nxt, past_key_values=kv_cache,
                              use_cache=True, return_dict=True)
            kv_cache = out.past_key_values
            del out
            last_end = pos + 1

        if (idx+1) % 100 == 0 or idx == n_pos - 1:
            elapsed = time.time() - t0
            parts = [f"[{idx+1}/{n_pos}] {elapsed:.0f}s"]
            for name, nl in nlls.items():
                if nl:
                    parts.append(f"{name}={np.exp(np.mean(nl)):.4f}")
            print("  " + "  ".join(parts))

    elapsed = time.time() - t0
    results = {}
    for name, nl in nlls.items():
        ppl = float(np.exp(np.mean(nl)))
        results[name] = {"ppl": round(ppl, 4), "n_pos": len(nl), "time_s": round(elapsed, 1)}
        print(f"  {name}: PPL={ppl:.4f}")

    del kv_cache
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--max-length", type=int, nargs='+', default=[8192, 16384, 32768, 65536])
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--skip-sinks", type=int, default=4)
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = str(Path(__file__).resolve().parents[1] / "runs" / "ppl_longctx_topk.json")

    print(f"Loading {args.llm_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.llm_name, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    ).eval()
    head_dim = getattr(model.config, "head_dim", 128)

    from datasets import load_dataset
    text = "\n\n".join([t for t in load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip()])
    max_tok = max(args.max_length)
    input_ids = tokenizer(text, return_tensors="pt", max_length=max_tok, truncation=True
                          ).input_ids.to(model.device)
    print(f"Tokenized: {input_ids.shape[1]} tokens")

    k4 = HadamardQuantCompressor(dim=head_dim, bit_width=4, half_rotation=True)
    v2 = HadamardQuantCompressor(dim=head_dim, bit_width=2, half_rotation=True)
    k2 = HadamardQuantCompressor(dim=head_dim, bit_width=2, half_rotation=True)
    configs = {"K4/V2": (k4, v2), "K2/V2": (k2, k2)}

    out_data = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            out_data = json.load(f)

    for ml in sorted(args.max_length):
        results = run_one_length(model, input_ids, configs, ml, args.stride,
                                 args.skip_sinks, args.topk)
        out_data[str(ml)] = results
        with open(args.output, "w") as f:
            json.dump(out_data, f, indent=2)

    print(f"\nAll saved to {args.output}")


if __name__ == "__main__":
    main()
