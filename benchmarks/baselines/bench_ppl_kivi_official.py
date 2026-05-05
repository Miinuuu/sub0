"""PPL with OFFICIAL KIVI (LlamaForCausalLM_KIVI + flash-attn path).

Uses KIVI's own Llama wrapper (supports GQA via LlamaFlashAttention_KIVI).
Requires flash-attn installed.

Usage:
    python experiments/bench_ppl_kivi_official.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --kbits 4 --vbits 4 \
        --output runs/ppl_kivi_official_l3-8b_k4v4.json
"""
from __future__ import annotations
import argparse, json, sys, os, time, math
from pathlib import Path

sys.path.insert(0, "<HOME>/ing/research/REF/KIVI")
import torch

from transformers import LlamaConfig, AutoTokenizer
from datasets import load_dataset


def eval_ppl(model, tokenizer, dataset_name: str = "wikitext-2-raw-v1",
             max_ctx: int = 4096, stride: int = 64, n_positions: int = 63):
    """KIVI PPL: sliding-window protocol (KIVI kernel GQA limitation).

    KIVI's LlamaForCausalLM_KIVI mishandles GQA when the prefill context
    is shorter than `residual_length` (128 tokens by default): it falls
    into an uncompressed code path at llama_kivi.py:380 that does not
    repeat_interleave K/V heads to match the query head count, raising
    `tensor a (32) vs b (8)` at the attention matmul.

    We therefore keep KIVI's original sliding-window eval protocol (each
    sample prefills a fixed 4096-token window, always exercising the
    compressed path) and only align the text-preprocessing (empty-line
    filter + BOS-prepended tokenization) with the canonical
    `core.ppl_eval` protocol. The caption on Tab.~ppl discloses this
    structural divergence.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.ppl_eval import get_wikitext_text
    text = get_wikitext_text(filter_empty_lines=True)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = enc.input_ids.to(model.device)
    n = input_ids.size(1)

    import torch.nn.functional as F
    nll = 0.0
    count = 0
    positions = list(range(max_ctx, min(n, max_ctx + n_positions * stride + 1), stride))[:n_positions]

    t0 = time.time()
    with torch.no_grad():
        for i, end in enumerate(positions):
            begin = max(0, end - max_ctx)
            if end + 1 >= n:
                break
            prefix = input_ids[:, begin:end]
            nxt    = input_ids[:, end:end + 1]
            tgt    = input_ids[:, end + 1:end + 2]

            out_p = model(prefix, use_cache=True)
            kv = out_p.past_key_values
            out_d = model(nxt, past_key_values=kv, use_cache=False)
            logits = out_d.logits[:, 0, :]

            nll += F.cross_entropy(logits, tgt[:, 0]).item()
            count += 1
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(positions)}] nll={nll/count:.4f}", flush=True)
    ppl = math.exp(nll / count)
    elapsed = time.time() - t0
    protocol = {
        "harness": "bench_ppl_kivi_official.eval_ppl",
        "dataset": "wikitext2",
        "add_special_tokens": True,
        "filter_empty_lines": True,
        "max_ctx": max_ctx,
        "stride": stride,
        "n_positions": n_positions,
        "window_policy": "sliding",  # non-canonical due to KIVI GQA limit
        "canonical": False,
    }
    return ppl, count, {"ppl": round(ppl, 4), "n_pos": count, "time_s": round(elapsed, 1),
                         "desc": "KIVI", "protocol": protocol}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--kbits", type=int, choices=[2, 4, 16])
    ap.add_argument("--vbits", type=int, choices=[2, 4, 16])
    ap.add_argument("--group_size", type=int, default=32)
    ap.add_argument("--residual_length", type=int, default=128)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    print(f"=== KIVI (official) PPL on {args.model} ===")
    print(f"    K{args.kbits}V{args.vbits}  group={args.group_size}  residual={args.residual_length}")

    config = LlamaConfig.from_pretrained(args.model)
    config.use_flash = True

    if args.kbits == 16 and args.vbits == 16:
        # Pure FP16 baseline
        from transformers import LlamaForCausalLM
        t0 = time.time()
        model = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16,
                                                 low_cpu_mem_usage=True, device_map="auto",
                                                 attn_implementation="flash_attention_2")
        print(f"  model loaded in {time.time()-t0:.0f}s (FP16 FA2)")
    else:
        config.k_bits = args.kbits
        config.v_bits = args.vbits
        config.group_size = args.group_size
        config.residual_length = args.residual_length
        from models.llama_kivi import LlamaForCausalLM_KIVI
        # device_map="auto" consults cls._no_split_modules to decide which
        # modules must stay on one GPU. KIVI's model inherits this list from
        # LlamaPreTrainedModel ("LlamaDecoderLayer") — it does NOT include
        # "LlamaDecoderLayer_KIVI", so accelerate happily splits KIVI's MLP
        # across GPUs, breaking gate_proj * up_proj. Pin it class-wide.
        LlamaForCausalLM_KIVI._no_split_modules = [
            "LlamaDecoderLayer_KIVI", "LlamaDecoderLayer"]
        # KIVI returns past_key_values as a 9-tuple containing uint8 packed
        # indices; accelerate's device-placement hook blindly casts outputs
        # to model dtype (fp16) when moving between GPUs, silently corrupting
        # the uint8 indices and producing NaN in subsequent attention. Tell
        # accelerate to leave past_key_values alone.
        LlamaForCausalLM_KIVI._skip_keys_device_placement = ["past_key_values"]
        t0 = time.time()
        model = LlamaForCausalLM_KIVI.from_pretrained(args.model, config=config,
                                                       low_cpu_mem_usage=True,
                                                       torch_dtype=torch.float16,
                                                       device_map="auto")
        print(f"  model loaded in {time.time()-t0:.0f}s (KIVI K{args.kbits}V{args.vbits})")

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=True)

    t0 = time.time()
    ppl, count, canonical_result = eval_ppl(model, tokenizer)
    dt = time.time() - t0

    result = {
        "model": args.model,
        "k_bits": args.kbits,
        "v_bits": args.vbits,
        "group_size": args.group_size,
        "residual_length": args.residual_length,
        "ppl": round(ppl, 4),
        "n_positions": count,
        "time_s": round(dt, 1),
        "protocol": canonical_result["protocol"],
    }
    print(json.dumps(result, indent=2))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
