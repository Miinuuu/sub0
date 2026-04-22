"""L3-8B tab:tput cliff-throughput measurement with BOTH backends under
CUDA Graph capture. Complements the original tab:tput (which measured
CDA under enforce_eager); graph-mode updates let CDA's decode path amortise
Python dispatch cost just like FA2 already does by default.

(B*, N) cliff points from the paper:
    (32, 8192), (16, 16384), (8, 32768), (4, 65536), (2, 131072)

Protocol:
    - same prompt tokens replicated B times (broadcast prefill)
    - 128 output tokens per request
    - aggregate tok/s = B * 128 / duration

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/bench_tput_cliff_graph.py \\
        --backend CDA --output runs/tput_cliff_graph/cda.json
    CUDA_VISIBLE_DEVICES=0 python experiments/bench_tput_cliff_graph.py \\
        --backend FLASH_ATTN --output runs/tput_cliff_graph/fa2.json
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CLIFF_POINTS = [
    (32, 8192),
    (16, 16384),
    (8, 32768),
    (4, 65536),
    (2, 131072),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=("FLASH_ATTN", "CDA"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--gen", type=int, default=128)
    args = ap.parse_args()

    if args.backend == "CDA":
        os.environ["CDA_ENABLE_MEMORY_SAVING"] = "1"
        os.environ["CDA_ENABLE_CUDAGRAPH"] = "1"
        os.environ.setdefault("CDA_GRAPH_MAX_B", "64")
        os.environ.setdefault("CDA_GRAPH_MAX_SPLITS", "128")

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for B_star, N in CLIFF_POINTS:
        kw = dict(
            model="meta-llama/Llama-3.1-8B-Instruct",
            # Llama-3.1-8B native max_position_embeddings = 131072. Cap
            # at that minus the gen budget so we don't exceed the
            # model's declared limit at N=128K + gen.
            max_model_len=min(131072, N + args.gen + 256),
            max_num_seqs=B_star,
            gpu_memory_utilization=0.85,
            dtype="float16",
            enforce_eager=False,  # <-- graph mode
        )
        # Ensure prompt length leaves at least gen budget inside max_model_len.
        effective_N = min(N, kw["max_model_len"] - args.gen - 16)
        if args.backend == "CDA":
            kw["attention_backend"] = "CUSTOM"
        # Fresh LLM per (B*, N) so each gets its own graph capture at the
        # right max context length. Destroy to release GPU before next.
        print(f"\n=== {args.backend}  B={B_star}  N={N} ===", flush=True)
        try:
            llm = LLM(**kw)
        except Exception as e:
            print(f"  init FAILED: {str(e)[:150]}", flush=True)
            results.append({"backend": args.backend, "B": B_star, "N": N,
                             "ok": False, "error": str(e)[:200]})
            args.output.write_text(json.dumps(results, indent=2))
            continue

        sp = SamplingParams(max_tokens=args.gen, temperature=0.0, ignore_eos=True)

        VOCAB_SAFE = 100000
        prompts = [TokensPrompt(prompt_token_ids=[100 + ((i + j) % VOCAB_SAFE)
                                                     for j in range(effective_N)])
                    for i in range(B_star)]

        # Warmup
        try:
            llm.generate(prompts, sp, use_tqdm=False)
            llm.generate(prompts, sp, use_tqdm=False)
        except Exception as e:
            print(f"  warmup FAILED: {str(e)[:150]}", flush=True)
            results.append({"backend": args.backend, "B": B_star, "N": N,
                             "ok": False, "error": str(e)[:200]})
            del llm
            import gc; gc.collect()
            import torch; torch.cuda.empty_cache()
            args.output.write_text(json.dumps(results, indent=2))
            continue

        # Timed run
        t0 = time.time()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        t1 = time.time()
        dur = t1 - t0
        total_gen = sum(len(o.outputs[0].token_ids) for o in outs)
        tokps = total_gen / dur
        print(f"  dur={dur:.2f}s  gen={total_gen}  tok/s={tokps:.1f}", flush=True)

        results.append({
            "backend": args.backend, "B": B_star, "N": N,
            "gen_per_req": args.gen,
            "duration_s": dur, "total_gen_tokens": total_gen,
            "tok_per_s": tokps, "ok": True,
        })
        args.output.write_text(json.dumps(results, indent=2))

        del llm
        import gc; gc.collect()
        import torch; torch.cuda.empty_cache()

    # Summary
    print("\n=== Summary ===")
    for r in results:
        if r.get("ok"):
            print(f"  B={r['B']:>3}  N={r['N']:>6}  tok/s={r['tok_per_s']:>7.1f}")
        else:
            print(f"  B={r['B']:>3}  N={r['N']:>6}  FAILED: {r.get('error','')[:80]}")


if __name__ == "__main__":
    main()
