"""Llama-3.1-70B TP=4 prefill latency + energy-per-token bench.

Parallel of bench_prefill_memory.py + bench_energy_nvml.py fused into one
vLLM session so the expensive 70B init is paid only once per backend.
Per-GPU power polling across all TP ranks.

Usage:
  python experiments/bench_70b_prefill_energy.py \
      --backend FA2 --B 1 --N 8192 --output-len 128 \
      --output runs/70b_prefill_energy_fa2.json
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, threading, time
from pathlib import Path


def gpu_mem_used_mb_all(gpu_ids):
    """Sum used memory across the given GPU ids."""
    total = 0.0
    for gpu_id in gpu_ids:
        r = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu_id}",
             "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        total += float(r.strip())
    return total


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("FA2", "CDA"), required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-70B-Instruct")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--iters", type=int, default=2,
                    help="Decode iters for energy averaging.")
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if args.backend == "CDA":
        os.environ["CDA_ENABLE_MEMORY_SAVING"] = "1"

    gpu_env = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    gpu_ids = [int(x) for x in gpu_env.split(",")]
    assert len(gpu_ids) >= args.tp, f"need >={args.tp} GPUs, got CUDA_VISIBLE_DEVICES={gpu_env}"
    gpu_ids = gpu_ids[: args.tp]

    import torch
    import pynvml
    from vllm import LLM, SamplingParams

    pynvml.nvmlInit()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in gpu_ids]

    # Multi-GPU power poller (sums instantaneous watts across all TP ranks).
    poll_events = []  # list of (t, total_power_W)
    running = [True]

    def poll():
        t0 = time.perf_counter()
        while running[0]:
            try:
                total_w = sum(
                    pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0 for h in handles
                )
            except pynvml.NVMLError:
                total_w = 0.0
            poll_events.append((time.perf_counter() - t0, total_w))
            time.sleep(0.050)

    mem_before = gpu_mem_used_mb_all(gpu_ids)

    kw = dict(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.N + args.output_len + 16,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_util,
        dtype="float16",
    )
    if args.backend == "CDA":
        kw["attention_backend"] = "CUSTOM"

    t_init0 = time.perf_counter()
    llm = LLM(**kw)
    t_init = time.perf_counter() - t_init0
    mem_after_load = gpu_mem_used_mb_all(gpu_ids)

    VOCAB_SAFE = 100000
    prompts = [
        {"prompt_token_ids": [100 + ((i * 257 + j) % VOCAB_SAFE) for j in range(args.N)]}
        for i in range(args.B)
    ]

    # ---------- Prefill latency (gen=1) ----------
    sp1 = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
    _ = llm.generate(prompts=prompts, sampling_params=sp1, use_tqdm=False)  # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = llm.generate(prompts=prompts, sampling_params=sp1, use_tqdm=False)
    torch.cuda.synchronize()
    t_prefill_ms = (time.perf_counter() - t0) * 1000.0
    mem_after_prefill = gpu_mem_used_mb_all(gpu_ids)

    # ---------- Energy per output token (gen=output_len) ----------
    sp = SamplingParams(max_tokens=args.output_len, temperature=0.0, ignore_eos=True)
    _ = llm.generate(prompts=prompts, sampling_params=sp, use_tqdm=False)  # warmup + populate prefix cache

    per_round = []
    for it in range(args.iters):
        poll_events.clear()
        running[0] = True
        thr = threading.Thread(target=poll, daemon=True)
        thr.start()

        torch.cuda.synchronize()
        t_start = time.perf_counter()
        outs = llm.generate(prompts=prompts, sampling_params=sp, use_tqdm=False)
        torch.cuda.synchronize()
        t_end = time.perf_counter()

        running[0] = False
        thr.join(timeout=1.0)

        elapsed_s = t_end - t_start
        events = [(t, p) for (t, p) in poll_events if t <= elapsed_s]
        if len(events) < 2:
            energy_j = 0.0; avg_w = 0.0
        else:
            energy_j = 0.0
            for i in range(1, len(events)):
                dt = events[i][0] - events[i-1][0]
                p_avg = 0.5 * (events[i][1] + events[i-1][1])
                energy_j += p_avg * dt
            avg_w = sum(p for _, p in events) / len(events)

        total_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
        per_round.append({
            "iter": it,
            "elapsed_s": elapsed_s,
            "energy_j": energy_j,
            "avg_power_w_total": avg_w,
            "n_samples": len(events),
            "total_tokens": total_tokens,
            "mj_per_tok": energy_j * 1000 / total_tokens if total_tokens else None,
        })

    pynvml.nvmlShutdown()

    mj_list = [r["mj_per_tok"] for r in per_round if r["mj_per_tok"]]
    res = {
        "backend": args.backend,
        "model": args.model,
        "tp": args.tp,
        "B": args.B, "N": args.N,
        "output_len": args.output_len,
        "gpu_ids": gpu_ids,
        "t_init_s": t_init,
        "t_prefill_ms": t_prefill_ms,
        "mem_before_mb_sum": mem_before,
        "mem_after_load_mb_sum": mem_after_load,
        "mem_after_prefill_mb_sum": mem_after_prefill,
        "model_load_mb_sum": mem_after_load - mem_before,
        "kv_peak_delta_mb_sum": mem_after_prefill - mem_after_load,
        "per_round": per_round,
        "mj_per_tok_mean": sum(mj_list) / len(mj_list) if mj_list else None,
        "mj_per_tok_min": min(mj_list) if mj_list else None,
        "mj_per_tok_max": max(mj_list) if mj_list else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
