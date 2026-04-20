"""Prefill latency + peak GPU memory bench.

Measures:
  T_prefill  : time to first token (TTFT-style, single-request)
  peak_alloc : torch.cuda.max_memory_allocated (MB)
  peak_reserv: torch.cuda.max_memory_reserved  (MB)

Two backends:
  FA2 (baseline)             : default vLLM flash-attn
  CDA (our K4V2 HMMA kernel) : attention_backend=CUSTOM + CDA_ENABLE_MEMORY_SAVING=1

Single-request, greedy. One N per invocation so CUDA allocator state is clean.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path


def gpu_mem_used_mb(gpu_id: int = 0) -> float:
    """Query free GPU memory via nvidia-smi (sees all processes, not just current)."""
    r = subprocess.check_output(
        ["nvidia-smi", f"--id={gpu_id}",
         "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return float(r.strip())


def main():
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("FA2", "CDA"), required=True)
    ap.add_argument("--N", type=int, required=True, help="Prefill length (tokens)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    args = ap.parse_args()

    if args.backend == "CDA":
        os.environ["CDA_ENABLE_MEMORY_SAVING"] = "1"

    import torch
    from vllm import LLM, SamplingParams

    # Honest peak memory via nvidia-smi (captures EngineCore subprocess too).
    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    mem_before = gpu_mem_used_mb(gpu_id)

    kw = dict(
        model=args.model,
        max_model_len=min(args.N + 16, 131072),
        enforce_eager=True,
        gpu_memory_utilization=0.88,
        dtype="float16",
    )
    if args.backend == "CDA":
        kw["attention_backend"] = "CUSTOM"

    llm = LLM(**kw)

    mem_after_load = gpu_mem_used_mb(gpu_id)

    # Use token ids directly — wrap within Llama-3.1 vocab (128256) to avoid OOV.
    VOCAB_SAFE = 100000
    prompt_token_ids = [100 + (i % VOCAB_SAFE) for i in range(args.N)]
    actual_N = len(prompt_token_ids)

    sp1 = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)

    torch.cuda.synchronize()
    _ = llm.generate(prompts=[{"prompt_token_ids": prompt_token_ids}],
                     sampling_params=sp1, use_tqdm=False)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = llm.generate(prompts=[{"prompt_token_ids": prompt_token_ids}],
                        sampling_params=sp1, use_tqdm=False)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    t_prefill_ms = (t1 - t0) * 1000.0

    # Peak memory = after-prefill reading (monotonic up to decode-start).
    mem_after_prefill = gpu_mem_used_mb(gpu_id)

    res = {
        "backend": args.backend,
        "model": args.model,
        "N_requested": args.N,
        "N_actual": actual_N,
        "t_prefill_ms": t_prefill_ms,
        "mem_before_mb": mem_before,
        "mem_after_load_mb": mem_after_load,
        "mem_after_prefill_mb": mem_after_prefill,
        "model_load_mb": mem_after_load - mem_before,
        "kv_peak_delta_mb": mem_after_prefill - mem_after_load,
        "gen_token_ids": list(outs[0].outputs[0].token_ids)[:1] if outs[0].outputs else [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
