"""Concurrent-capacity sweep — direct test of the headline 3.56× claim.

Sweeps batch B at fixed prompt + decode, comparing FA2 (which OOMs at
some B*) against CDA (which continues past B*). The B-ratio at the
OOM cliff measures the user-visible concurrent-capacity multiplier.

This addresses the core paper claim: "≤ 3.56× concurrent-capacity
multiplier at the memory wall."

Usage (GPU 3 (multi-GPU server)):
    python \
      benchmarks/paper/bench_capacity_sweep.py \
        --batches 8 16 32 48 64 96 128 \
        --prompt-len 4096 --decode-len 128 \
        --gpu 3 \
        --output runs/paper/capacity_sweep.json
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run_one_config(backend: str, batch: int, prompt_len: int, decode_len: int,
                    model: str, max_model_len: int, gpu_id, tp: int,
                    memory_saving: bool, gpu_mem_util: float, eager: bool):
    """Spawn a vLLM subprocess for one (backend, batch) config.

    `gpu_id` may be an int (single GPU) or a comma-string like "0,1,2,3"
    (multi-GPU TP). When `tp > 1`, gpu_id should expose tp devices.
    """
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent.parent)
    # CDA backend registry is process-local in vLLM v1; the worker MUST run in
    # the same process group as where register_backend() was called. Disable
    # multiprocessing so the engine core stays in this process.
    env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    # vLLM TP requires CUDA aware multiprocessing init context for child workers
    if tp > 1:
        env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # For tp > 1 the CDA backend / KV-cache page-size patch must reach every
    # vLLM worker process via the cda_v2 plugin entry point. Inject the env
    # vars only on CDA runs so FA2 baselines keep their default KV layout.
    if backend == "CDA":
        env["VLLM_PLUGINS"] = "cda_v2"
        if memory_saving:
            env["CDA_V2_ENABLE_MEMORY_SAVING"] = "1"
    else:
        # Strip any inherited CDA env so FA2 never sees the patches.
        env.pop("VLLM_PLUGINS", None)
        env.pop("CDA_V2_ENABLE_MEMORY_SAVING", None)

    out_path = f"/tmp/_cap_{backend}_b{batch}_tp{tp}.json"
    cmd = [sys.executable, __file__, "--worker",
           "--backend", backend, "--batch", str(batch),
           "--prompt-len", str(prompt_len), "--decode-len", str(decode_len),
           "--model", model, "--max-model-len", str(max_model_len),
           "--gpu-mem-util", str(gpu_mem_util),
           "--tp", str(tp),
           "--out-tmp", out_path]
    if eager:
        cmd.append("--eager")
    if memory_saving:
        cmd.append("--memory-saving")

    print(f"  → {backend} B={batch} ...", flush=True)
    t0 = time.time()
    timeout_s = int(os.environ.get("CDA_BENCH_TIMEOUT_S", "1800"))
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                            timeout=timeout_s)
    dt = time.time() - t0
    if Path(out_path).exists():
        result = json.loads(Path(out_path).read_text())
        Path(out_path).unlink()
        result["wall_s"] = dt
        result["status"] = "ok"
    else:
        # Worker crashed (likely OOM)
        is_oom = ("CUDA out of memory" in proc.stderr or
                  "out of memory" in proc.stderr.lower() or
                  "OOM" in proc.stderr)
        result = {
            "backend": backend, "batch": batch, "status": "oom" if is_oom else "fail",
            "stderr_tail": proc.stderr[-1000:] if proc.stderr else "",
            "wall_s": dt,
        }
    return result


def worker_run(args):
    import sqlite3  # noqa: F401
    from diskcache import Cache  # noqa: F401
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    if args.backend == "CDA":
        from cda.kernels_cuda.vllm_integration.cda_attn_v2 import (
            register_backend, enable_cda_memory_saving,
        )
        if args.memory_saving:
            enable_cda_memory_saving()
        register_backend("CDA")
        attention_config = {"backend": "CDA"}
    else:
        attention_config = {"backend": "FLASH_ATTN", "flash_attn_version": 2}

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=args.eager,  # default False → CUDA Graph enabled
        dtype="float16",
        attention_config=attention_config,
        disable_log_stats=True,
        tensor_parallel_size=args.tp,
    )

    prompts = [
        TokensPrompt(prompt_token_ids=[(i + b) % 1000 + 1
                                            for i in range(args.prompt_len)])
        for b in range(args.batch)
    ]
    sampling = [SamplingParams(max_tokens=args.decode_len, temperature=0.0,
                                ignore_eos=True) for _ in prompts]

    t0 = time.time()
    outputs = llm.generate(prompts, sampling)
    t_total = time.time() - t0

    n_completed = sum(1 for o in outputs if o.outputs and o.outputs[0].token_ids)
    total_tokens = (sum(args.prompt_len for _ in outputs) +
                     sum(len(o.outputs[0].token_ids) if o.outputs else 0 for o in outputs))
    throughput = total_tokens / t_total

    result = {
        "backend": args.backend,
        "batch": args.batch,
        "prompt_len": args.prompt_len,
        "decode_len": args.decode_len,
        "n_completed": n_completed,
        "total_wall_s": t_total,
        "total_tokens": total_tokens,
        "throughput_tok_s": throughput,
    }
    Path(args.out_tmp).write_text(json.dumps(result, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[8, 16, 32, 48, 64, 96, 128])
    ap.add_argument("--prompt-len", type=int, default=4096)
    ap.add_argument("--decode-len", type=int, default=128)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--gpu", default="3",
                    help="single GPU (e.g. '3') or comma list for TP "
                         "(e.g. '0,1,2,3' for TP=4)")
    ap.add_argument("--tp", type=int, default=1,
                    help="tensor_parallel_size for vLLM (default 1)")
    ap.add_argument("--eager", action="store_true",
                    help="Disable CUDA Graph (slower; default uses CG).")
    ap.add_argument("--no-cda-memory-saving", action="store_true",
                    help="Disable enable_cda_memory_saving(); CDA will use "
                         "default 512 B/slot allocation. Use for *fair "
                         "throughput-at-equal-memory* comparison; OOM cliff "
                         "test should NOT use this.")
    ap.add_argument("--skip-fa2", action="store_true",
                    help="Skip FA2 sweep (use when FA2 results already exist).")
    ap.add_argument("--skip-cda", action="store_true",
                    help="Skip CDA sweep.")
    ap.add_argument("--output", required=False)
    # Worker mode
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--backend", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--batch", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out-tmp", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--memory-saving", action="store_true",
                    help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        worker_run(args)
        return

    if not args.output:
        print("ERROR: --output required", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {"config": vars(args), "FA2": [], "CDA": []}

    print(f"Capacity sweep: prompt={args.prompt_len}, decode={args.decode_len}")
    print(f"  batches: {args.batches}")
    print(f"  GPU: {args.gpu}\n")

    # FA2 sweep — stop at first OOM
    if args.skip_fa2:
        print("=== FA2 (skipped) ===")
        # Try to load existing FA2 data from output if file exists
        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text())
                results["FA2"] = existing.get("FA2", [])
                print(f"  loaded {len(results['FA2'])} existing FA2 entries")
            except Exception:
                pass
    print("=== FA2 ===" if not args.skip_fa2 else "")
    fa2_oom_at = None
    for b in (args.batches if not args.skip_fa2 else []):
        r = run_one_config("FLASH_ATTN", b, args.prompt_len, args.decode_len,
                            args.model, args.max_model_len, args.gpu, args.tp,
                            memory_saving=False, gpu_mem_util=args.gpu_mem_util,
                            eager=args.eager)
        results["FA2"].append(r)
        if r["status"] == "ok":
            print(f"  B={b}: OK  throughput={r.get('throughput_tok_s', 0):.0f} tok/s  ({r['wall_s']:.0f}s)")
        else:
            print(f"  B={b}: {r['status'].upper()}  ({r['wall_s']:.0f}s)")
        # Save incrementally
        out_path.write_text(json.dumps(results, indent=2))
        if r["status"] == "oom":
            fa2_oom_at = b
            print(f"  → FA2 OOM cliff at B={b}; stopping FA2 sweep.\n")
            break

    # CDA sweep — go through all batches
    print("=== CDA ===" if not args.skip_cda else "=== CDA (skipped) ===")
    for b in (args.batches if not args.skip_cda else []):
        r = run_one_config("CDA", b, args.prompt_len, args.decode_len,
                            args.model, args.max_model_len, args.gpu, args.tp,
                            memory_saving=not args.no_cda_memory_saving,
                            gpu_mem_util=args.gpu_mem_util,
                            eager=args.eager)
        results["CDA"].append(r)
        if r["status"] == "ok":
            print(f"  B={b}: OK  throughput={r.get('throughput_tok_s', 0):.0f} tok/s  ({r['wall_s']:.0f}s)")
        else:
            print(f"  B={b}: {r['status'].upper()}  ({r['wall_s']:.0f}s)")
        out_path.write_text(json.dumps(results, indent=2))
        if r["status"] == "oom":
            print(f"  → CDA OOM cliff at B={b}.\n")
            break

    # Summary
    fa2_max_ok = max([r["batch"] for r in results["FA2"] if r["status"] == "ok"], default=0)
    cda_max_ok = max([r["batch"] for r in results["CDA"] if r["status"] == "ok"], default=0)
    print("\n" + "=" * 60)
    print(f"Capacity multiplier (CDA / FA2): {cda_max_ok} / {fa2_max_ok} = {cda_max_ok/fa2_max_ok if fa2_max_ok else 'inf'}×")
    results["summary"] = {
        "fa2_max_batch_ok": fa2_max_ok,
        "cda_max_batch_ok": cda_max_ok,
        "capacity_multiplier": cda_max_ok / fa2_max_ok if fa2_max_ok else None,
    }
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
