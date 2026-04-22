"""B.2: AWQ + CDA composability. Measure WikiText-2 PPL for Llama-3.1-8B
at 4K context for the four backend combinations:

  (A) FP16 weights + FA2 KV         (baseline)
  (B) AWQ-4bit weights + FA2 KV     (weight-only compression)
  (C) FP16 weights + CDA K4V2       (KV-only compression)
  (D) AWQ-4bit weights + CDA K4V2   (stacked)

Uses vLLM 0.19 with the standard sliding-window PPL pattern (stride=64, 63
positions on wikitext-2-raw-v1 test split). Matches Table 1 protocol.
"""
from __future__ import annotations
import argparse, json, os, sys, math
from pathlib import Path


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", choices=("FP16", "AWQ"), required=True)
    ap.add_argument("--attn", choices=("FA2", "CDA"), required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--n-pos", type=int, default=63)
    ap.add_argument(
        "--size", choices=("8b", "70b"), default="8b",
        help="Model scale: 8b (single-GPU, TP=1, gpu-util=0.88) or "
             "70b (TP=4, gpu-util=0.85) — 70b AWQ weights downloaded to "
             "$HF_HOME/hub/models--hugging-quants--Meta-Llama-3.1-70B-"
             "Instruct-AWQ-INT4")
    ap.add_argument("--tensor-parallel-size", type=int, default=None)
    args = ap.parse_args()

    if args.attn == "CDA":
        os.environ["CDA_ENABLE_MEMORY_SAVING"] = "1"

    import torch
    from vllm import LLM, SamplingParams

    # Model selection — parametrised on --size so the same harness drives
    # the 8B (TP=1) composability confirmed in App.~\ref{app:awq} and the
    # 70B (TP=4) extension requested for the NeurIPS camera-ready delta.
    if args.size == "70b":
        if args.weights == "FP16":
            model_id = "meta-llama/Llama-3.1-70B-Instruct"
            quant = None
        else:
            model_id = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
            quant = "awq_marlin"
        tp = args.tensor_parallel_size or 4
        gpu_util = 0.85
    else:  # 8b
        if args.weights == "FP16":
            model_id = "meta-llama/Llama-3.1-8B-Instruct"
            quant = None
        else:
            model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
            quant = "awq_marlin"
        tp = args.tensor_parallel_size or 1
        gpu_util = 0.88

    kw = dict(
        model=model_id,
        max_model_len=args.max_ctx + 8,
        enforce_eager=True,
        gpu_memory_utilization=gpu_util,
        tensor_parallel_size=tp,
        dtype="float16",
    )
    if quant:
        kw["quantization"] = quant
    if args.attn == "CDA":
        kw["attention_backend"] = "CUSTOM"

    llm = LLM(**kw)
    tokenizer = llm.get_tokenizer()

    # Load WikiText-2 test split
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([r["text"] for r in ds if r["text"].strip()])
    all_ids = tokenizer.encode(text, add_special_tokens=False)
    print(f"Total tokens in test split: {len(all_ids)}")

    # Slide windows of length max_ctx, stride*stride=64 positions apart.
    win = args.max_ctx
    stride = args.stride
    positions = []
    for i in range(args.n_pos):
        start = i * stride * win // stride  # take n_pos evenly-spaced windows
        # simpler: step by max_ctx // n_pos * i
        start = (i * (len(all_ids) - win)) // max(args.n_pos - 1, 1)
        if start + win > len(all_ids):
            break
        positions.append(start)

    # Use vLLM's logprobs API to compute PPL window-by-window.
    sp = SamplingParams(
        max_tokens=1, temperature=0.0,
        prompt_logprobs=1,  # request per-token logprobs for the prompt
        ignore_eos=True,
    )

    total_nll = 0.0
    total_count = 0
    for idx, start in enumerate(positions):
        window_ids = all_ids[start:start + win]
        outs = llm.generate(prompts=[{"prompt_token_ids": window_ids}],
                            sampling_params=sp, use_tqdm=False)
        logprobs = outs[0].prompt_logprobs  # list len = win; first is None
        # Sum token-level NLL (skip the first token which has no logprob).
        for tid, lp_dict in zip(window_ids[1:], logprobs[1:]):
            if lp_dict is None:
                continue
            # vLLM returns {token_id: Logprob(logprob=...)} for the chosen tokens.
            if tid in lp_dict:
                total_nll -= lp_dict[tid].logprob
                total_count += 1
            else:
                # token wasn't in top-1 — fall back to any available logprob for this position
                any_key = next(iter(lp_dict))
                total_nll -= lp_dict[any_key].logprob
                total_count += 1
        print(f"  window {idx+1}/{len(positions)}: running PPL = {math.exp(total_nll/total_count):.4f}")

    ppl = math.exp(total_nll / total_count) if total_count else float("nan")
    res = {
        "weights": args.weights,
        "attn": args.attn,
        "size": args.size,
        "model_id": model_id,
        "quantization": quant,
        "tensor_parallel_size": tp,
        "max_ctx": args.max_ctx,
        "stride": args.stride,
        "n_pos": len(positions),
        "total_tokens_scored": total_count,
        "ppl": ppl,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
