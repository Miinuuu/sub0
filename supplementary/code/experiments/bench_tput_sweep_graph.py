"""L3-8B Fig 2(c) full B-sweep under CUDA Graph capture (both backends).

Replaces the eager-mode tput_{cda,fa2}_N{nk}.json files that
bench_tput_sweep_graph's companion figure script
(runs/figures/gen_fig_gpu_kernel_3panel.py) reads. Writes graph-mode
analogues to runs/tput_graph/tput_{cda,fa2}_N{nk}.json with the same
schema so the figure script can swap data sources with a 2-char path
change.

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/bench_tput_sweep_graph.py \\
        --backend CDA --outdir runs/tput_graph
    CUDA_VISIBLE_DEVICES=0 python experiments/bench_tput_sweep_graph.py \\
        --backend FLASH_ATTN --outdir runs/tput_graph
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Match the existing eager B-grids so the graph data is a drop-in replacement.
SWEEP_CDA = {
    8192:   [1, 4, 8, 16, 32, 64],
    16384:  [1, 4, 8, 16, 32],
    32768:  [1, 2, 4, 8, 16, 32],
    65536:  [1, 2, 4, 8, 16],
    131072: [1, 2, 4, 8],
}
SWEEP_FA2 = {
    8192:   [1, 4, 8, 16, 32],
    16384:  [1, 4, 8, 16, 32],
    32768:  [1, 2, 4, 8, 16],
    65536:  [1, 2, 4, 8],
    131072: [1, 2, 4, 8],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=("FLASH_ATTN", "CDA"))
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--gen", type=int, default=128)
    args = ap.parse_args()

    if args.backend == "CDA":
        os.environ["CDA_ENABLE_MEMORY_SAVING"] = "1"
        os.environ["CDA_ENABLE_CUDAGRAPH"] = "1"
        os.environ.setdefault("CDA_GRAPH_MAX_B", "64")
        os.environ.setdefault("CDA_GRAPH_MAX_SPLITS", "128")
        sweep = SWEEP_CDA
    else:
        sweep = SWEEP_FA2

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    args.outdir.mkdir(parents=True, exist_ok=True)
    nk_map = {8192: "8K", 16384: "16K", 32768: "32K", 65536: "64K", 131072: "128K"}

    for N, B_list in sweep.items():
        nk = nk_map[N]
        tag = "cda" if args.backend == "CDA" else "fa2"
        outpath = args.outdir / f"tput_{tag}_N{nk}.json"
        max_B = max(B_list)

        print(f"\n=== {args.backend}  N={N} ({nk})  B∈{B_list} ===", flush=True)

        kw = dict(
            model="meta-llama/Llama-3.1-8B-Instruct",
            max_model_len=min(131072, N + args.gen + 256),
            max_num_seqs=max_B,
            gpu_memory_utilization=0.85,
            dtype="float16",
            enforce_eager=False,
        )
        if args.backend == "CDA":
            kw["attention_backend"] = "CUSTOM"
        kw["compilation_config"] = {"cudagraph_capture_sizes": B_list}

        effective_N = min(N, kw["max_model_len"] - args.gen - 16)

        try:
            llm = LLM(**kw)
        except Exception as e:
            print(f"  init FAILED: {str(e)[:180]}", flush=True)
            outpath.write_text(json.dumps([{
                "backend": args.backend, "N": N, "ok": False,
                "error": str(e)[:400]}], indent=2))
            continue

        sp = SamplingParams(max_tokens=args.gen, temperature=0.0, ignore_eos=True)

        VOCAB_SAFE = 100000
        # Build max_B prompts once; slice per B below.
        all_prompts = [
            TokensPrompt(prompt_token_ids=[100 + ((i + j) % VOCAB_SAFE)
                                           for j in range(effective_N)])
            for i in range(max_B)
        ]

        # Warmup: at max_B once, so graph capture hits the largest size.
        try:
            llm.generate(all_prompts, sp, use_tqdm=False)
        except Exception as e:
            print(f"  warmup FAILED: {str(e)[:180]}", flush=True)
            outpath.write_text(json.dumps([{
                "backend": args.backend, "N": N, "ok": False,
                "error": str(e)[:400]}], indent=2))
            del llm
            import gc; gc.collect()
            import torch; torch.cuda.empty_cache()
            continue

        results = []
        for B in B_list:
            prompts = all_prompts[:B]
            try:
                # one extra warmup at this B (graph replay path)
                llm.generate(prompts, sp, use_tqdm=False)
                t0 = time.time()
                outs = llm.generate(prompts, sp, use_tqdm=False)
                t1 = time.time()
                dur = t1 - t0
                total_gen = sum(len(o.outputs[0].token_ids) for o in outs)
                tokps = total_gen / dur
                print(f"  B={B:>3}  dur={dur:.2f}s  tok/s={tokps:>7.1f}", flush=True)
                results.append({
                    "backend": args.backend, "B": B, "N": N,
                    "gen_per_req": args.gen,
                    "duration_s": dur, "total_gen_tokens": total_gen,
                    "tok_per_s": tokps, "ok": True,
                })
            except Exception as e:
                print(f"  B={B:>3}  FAILED: {str(e)[:150]}", flush=True)
                results.append({
                    "backend": args.backend, "B": B, "N": N,
                    "ok": False, "error": str(e)[:200],
                })
            outpath.write_text(json.dumps(results, indent=2))

        del llm
        import gc; gc.collect()
        import torch; torch.cuda.empty_cache()

    print("\nDONE.")


if __name__ == "__main__":
    main()
