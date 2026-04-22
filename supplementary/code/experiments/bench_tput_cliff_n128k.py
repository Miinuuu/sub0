"""Re-run only N=128K cliff point after max_model_len fix.

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/bench_tput_cliff_n128k.py \\
        --backend FLASH_ATTN --output runs/tput_cliff_graph/fa2_n128k.json
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=("FLASH_ATTN", "CDA"))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if args.backend == "CDA":
        os.environ["CDA_ENABLE_MEMORY_SAVING"] = "1"
        os.environ["CDA_ENABLE_CUDAGRAPH"] = "1"
        os.environ.setdefault("CDA_GRAPH_MAX_B", "8")
        os.environ.setdefault("CDA_GRAPH_MAX_SPLITS", "128")

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    B, N, gen = 2, 130944, 128  # 128K-ish, leaves gen+room under 131072
    kw = dict(
        model="meta-llama/Llama-3.1-8B-Instruct",
        max_model_len=131072,
        max_num_seqs=B,
        gpu_memory_utilization=0.85,
        dtype="float16",
        enforce_eager=False,
    )
    if args.backend == "CDA":
        kw["attention_backend"] = "CUSTOM"
    print(f"=== {args.backend} B={B} N={N} gen={gen} ===", flush=True)

    llm = LLM(**kw)
    sp = SamplingParams(max_tokens=gen, temperature=0.0, ignore_eos=True)

    VOCAB_SAFE = 100000
    prompts = [TokensPrompt(prompt_token_ids=[100 + ((i + j) % VOCAB_SAFE)
                                                 for j in range(N)])
                for i in range(B)]
    # Warmup
    llm.generate(prompts, sp, use_tqdm=False)
    llm.generate(prompts, sp, use_tqdm=False)

    t0 = time.time()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    t1 = time.time()
    dur = t1 - t0
    total_gen = sum(len(o.outputs[0].token_ids) for o in outs)
    tokps = total_gen / dur
    print(f"  dur={dur:.2f}s  gen={total_gen}  tok/s={tokps:.1f}", flush=True)

    res = [{
        "backend": args.backend, "B": B, "N": N, "gen_per_req": gen,
        "duration_s": dur, "total_gen_tokens": total_gen,
        "tok_per_s": tokps, "ok": True,
    }]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
