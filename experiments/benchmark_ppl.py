"""WikiText-2 perplexity: FP16 baseline vs CDA compressed KV decode.

Measures perplexity in the same sliding-window style as GPTQ / QuaRot:
concatenate the test split, slice into non-overlapping windows of
``--max-length`` tokens, and average per-token NLL.

Usage::

    CUDA_VISIBLE_DEVICES=0 python experiments/benchmark_ppl.py \\
        --model meta-llama/Llama-3.2-1B-Instruct --bits 2 \\
        --max-length 2048 --stride 2048

The CDA path compresses the KV cache after prefill and decompresses per step.
For numerical parity with the paper, ``skip_sinks`` keeps the first N tokens
uncompressed (attention-sink tokens are disproportionately sensitive).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache  # noqa: E402

from cda import HadamardQuantCompressor  # noqa: E402


def _load_wikitext_tokens(tokenizer, device: torch.device) -> torch.Tensor:
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in data["text"] if t.strip())
    return tokenizer(text, return_tensors="pt").input_ids.to(device)


def _compress_kv_inplace(kv: DynamicCache, comp: HadamardQuantCompressor,
                         head_dim: int, skip_sinks: int) -> DynamicCache:
    """Round-trip each layer's KV through the compressor, preserving sinks."""
    for li in range(len(kv.key_cache)):
        for cache in (kv.key_cache, kv.value_cache):
            x = cache[li]
            if skip_sinks > 0 and x.shape[-2] > skip_sinks:
                sink = x[..., :skip_sinks, :]
                tail = x[..., skip_sinks:, :]
                tail_q = comp.dequantize(comp.quantize(tail.reshape(-1, head_dim))).reshape(tail.shape)
                cache[li] = torch.cat([sink, tail_q], dim=-2).to(x.dtype)
            else:
                cache[li] = comp.dequantize(comp.quantize(x.reshape(-1, head_dim))).reshape(x.shape).to(x.dtype)
    return kv


@torch.no_grad()
def evaluate_ppl(model, tokens: torch.Tensor, max_length: int, stride: int,
                 comp: HadamardQuantCompressor | None, head_dim: int,
                 skip_sinks: int) -> float:
    """Sliding-window PPL over the given token stream.

    * **FP16 path (``comp is None``)** — standard teacher-forced PPL over all
      in-window shifted logits.
    * **CDA path** — isolates the effect of compressed-KV decoding: for each
      window we prefill ``window[:-1]``, compress that KV, then forward the
      final token with the compressed cache and score only that one logit.
      The accumulated metric is therefore the PPL of _compressed-KV decoding_
      over the test stream (one evaluated token per window).
    """
    model.eval()
    nll_sum = 0.0
    token_count = 0
    total_len = tokens.shape[1]

    for begin in range(0, total_len, stride):
        end = min(begin + max_length, total_len)
        if end - begin < 2:
            break
        window = tokens[:, begin:end]

        # Decode-scenario eval: build a prefill that ends *before* the
        # penultimate token, compress, then take one decode step with the
        # penultimate token and score how well we predict window[-1].
        # FP16 path uses the same split but with an uncompressed KV — so the
        # two PPL figures differ only in the cache compression.
        if end - begin < 3:
            break
        ctx_ids = window[:, :-2]
        step_ids = window[:, -2:-1]
        target_id = window[:, -1]

        pre = model(ctx_ids, use_cache=True, return_dict=True)
        kv = pre.past_key_values
        if comp is not None:
            kv = _compress_kv_inplace(kv, comp, head_dim, skip_sinks)
        step = model(step_ids, past_key_values=kv, use_cache=True,
                     return_dict=True)
        pred_logits = step.logits[:, 0, :].float()
        nll = torch.nn.functional.cross_entropy(pred_logits, target_id)
        valid = 1

        nll_sum += nll.item() * valid
        token_count += valid
        if end == total_len:
            break

    return float(torch.exp(torch.tensor(nll_sum / max(token_count, 1))).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--bits", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=2048)
    parser.add_argument("--skip-sinks", type=int, default=4)
    parser.add_argument("--output", default="runs/cda_ppl.json")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=device,
        low_cpu_mem_usage=True,
    )
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    tokens = _load_wikitext_tokens(tokenizer, device)
    print(f"WikiText-2 test tokens: {tokens.shape[1]}  head_dim={head_dim}")

    fp16_ppl = evaluate_ppl(model, tokens, args.max_length, args.stride,
                            comp=None, head_dim=head_dim, skip_sinks=0)
    print(f"  FP16 baseline     : PPL={fp16_ppl:.3f}")

    comp = HadamardQuantCompressor(dim=head_dim, bit_width=args.bits,
                                   half_rotation=True)
    cda_ppl = evaluate_ppl(model, tokens, args.max_length, args.stride,
                           comp=comp, head_dim=head_dim,
                           skip_sinks=args.skip_sinks)
    print(f"  CDA {args.bits}-bit        : PPL={cda_ppl:.3f} "
          f"(Δ={cda_ppl - fp16_ppl:+.3f})")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model": args.model,
        "bits": args.bits,
        "max_length": args.max_length,
        "stride": args.stride,
        "skip_sinks": args.skip_sinks,
        "fp16_ppl": fp16_ppl,
        "cda_ppl": cda_ppl,
        "delta_ppl": cda_ppl - fp16_ppl,
    }, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
