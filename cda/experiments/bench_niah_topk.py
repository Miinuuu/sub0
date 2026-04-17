"""Needle-In-A-Haystack with CDA K4/V2 + TopK128.

Tests fact recall at various context lengths and needle positions.
Uses _compress_kv_cache + manual_decode_step with topk=128.

Usage:
    CUDA_VISIBLE_DEVICES=1 python experiments/bench_niah_topk.py
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

NEEDLE = "The special magic number is: 7492158306."
QUERY = "What is the special magic number? The special magic number is:"
ANSWER = "7492158306"

HAYSTACK_UNIT = (
    "This is a passage of filler text used to pad the context. "
    "It contains no useful information and is only meant to increase the length. "
    "The weather today is sunny with a chance of rain later in the evening. "
    "In other news, scientists have discovered a new species of butterfly. "
    "The stock market showed mixed results, with technology stocks leading gains. "
)


def build_prompt(tokenizer, ctx_len, depth_pct):
    """Build a prompt with needle inserted at depth_pct of ctx_len."""
    # Tokenize haystack unit
    hay_tokens = tokenizer.encode(HAYSTACK_UNIT, add_special_tokens=False)
    needle_tokens = tokenizer.encode(NEEDLE, add_special_tokens=False)
    query_tokens = tokenizer.encode(QUERY, add_special_tokens=False)

    # Fill haystack to ctx_len - len(query) - len(needle) - margin
    available = ctx_len - len(query_tokens) - len(needle_tokens) - 10
    if available <= 0:
        available = ctx_len // 2

    n_repeats = (available // len(hay_tokens)) + 1
    hay_all = (hay_tokens * n_repeats)[:available]

    # Insert needle at depth_pct
    insert_pos = int(len(hay_all) * depth_pct / 100)
    prompt_tokens = hay_all[:insert_pos] + needle_tokens + hay_all[insert_pos:]

    # Truncate to ctx_len - query
    prompt_tokens = prompt_tokens[:ctx_len - len(query_tokens)]
    prompt_tokens = prompt_tokens + query_tokens

    return torch.tensor([prompt_tokens], dtype=torch.long)


def eval_niah(model, tokenizer, input_ids, k_comp, v_comp, topk, skip_sinks, max_gen=20):
    """Run CDA decode and check if answer contains the needle number."""
    device = model.device if hasattr(model, 'device') else next(model.parameters()).device
    input_ids = input_ids.to(device)
    seq_len = input_ids.shape[1]

    with torch.no_grad():
        # Prefill (backbone, chunked)
        CHUNK = 4096
        kv_cache = None
        for start in range(0, seq_len, CHUNK):
            end = min(start + CHUNK, seq_len)
            out = model.model(input_ids=input_ids[:, start:end], past_key_values=kv_cache,
                              use_cache=True, return_dict=True)
            kv_cache = out.past_key_values
            del out

        # Compress
        compressed = _compress_kv_cache(kv_cache, k_comp, v_comp, skip_sinks=skip_sinks)
        del kv_cache

        # Generate tokens
        generated = []
        pos = seq_len
        for _ in range(max_gen):
            if not generated:
                # First decode token: use last input token
                nxt = input_ids[:, -1:]
                logits = manual_decode_step(model, nxt, compressed, k_comp, v_comp,
                                           position=pos - 1, attn_gate_topk=topk)
            else:
                nxt = torch.tensor([[generated[-1]]], device=device)
                logits = manual_decode_step(model, nxt, compressed, k_comp, v_comp,
                                           position=pos, attn_gate_topk=topk)
            next_token = logits[0, -1].argmax().item()
            generated.append(next_token)
            pos += 1

            # Stop on EOS or newline
            decoded = tokenizer.decode([next_token])
            if next_token == tokenizer.eos_token_id or '\n' in decoded:
                break

        del compressed
        gc.collect()
        torch.cuda.empty_cache()

    answer_text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    found = ANSWER in answer_text
    return found, answer_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--ctx-lengths", type=int, nargs='+',
                        default=[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
    parser.add_argument("--depths", type=int, nargs='+', default=[10, 25, 50, 75, 90])
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--skip-sinks", type=int, default=4)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = str(Path(__file__).resolve().parents[1] / "runs" / "niah_topk128.json")

    print(f"Loading {args.llm_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.llm_name, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    ).eval()
    head_dim = getattr(model.config, "head_dim", 128)

    k4 = HadamardQuantCompressor(dim=head_dim, bit_width=4, half_rotation=True)
    v2 = HadamardQuantCompressor(dim=head_dim, bit_width=2, half_rotation=True)

    results = []
    total = len(args.ctx_lengths) * len(args.depths)
    done = 0
    t0 = time.time()

    for ctx_len in args.ctx_lengths:
        for depth in args.depths:
            done += 1
            input_ids = build_prompt(tokenizer, ctx_len, depth)
            actual_len = input_ids.shape[1]

            t1 = time.time()
            found, answer = eval_niah(model, tokenizer, input_ids, k4, v2,
                                      args.topk, args.skip_sinks)
            elapsed = time.time() - t1

            status = "PASS" if found else "FAIL"
            print(f"  [{done}/{total}] ctx={ctx_len:>6d} depth={depth:>2d}%  "
                  f"{status}  ({elapsed:.1f}s)  ans=\"{answer[:50]}\"")

            results.append({
                "ctx_len": ctx_len, "depth_pct": depth,
                "actual_tokens": actual_len,
                "found": found, "answer": answer[:100],
                "time_s": round(elapsed, 1)
            })

            # Save incrementally
            with open(args.output, "w") as f:
                json.dump({"config": "K4/V2+TopK128", "model": args.llm_name,
                           "results": results}, f, indent=2)

    elapsed = time.time() - t0
    n_pass = sum(1 for r in results if r["found"])
    print(f"\n=== NIAH K4/V2+TopK128: {n_pass}/{len(results)} passed ({elapsed:.0f}s) ===")

    # Summary per context length
    for ctx_len in args.ctx_lengths:
        ctx_results = [r for r in results if r["ctx_len"] == ctx_len]
        n_ok = sum(1 for r in ctx_results if r["found"])
        print(f"  ctx={ctx_len:>6d}: {n_ok}/{len(ctx_results)}")


if __name__ == "__main__":
    main()
