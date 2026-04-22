"""B.3: Energy per token via NVML power polling.

Measures GPU-side energy consumption during vLLM decode and reports
J/token for FA2 vs CDA at a few (B, N) points.

For each backend and (B, N):
  1. Spawn a vLLM run that generates OUTPUT_LEN tokens per request for B requests.
  2. Poll pynvml power (W) every 50 ms from a background thread.
  3. Integrate P × dt over the bench window → total energy (J).
  4. Divide by total generated tokens → mJ/token.

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/bench_energy_nvml.py \
        --backend FA2 --B 1 --N 8192 --output runs/energy/FA2_B1_N8K.json
"""
from __future__ import annotations
import argparse, json, os, sys, threading, time
from pathlib import Path


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("FA2", "CDA"), required=True)
    ap.add_argument("--B", type=int, required=True, help="Batch size")
    ap.add_argument("--N", type=int, required=True, help="Prompt length per request")
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--iters", type=int, default=3, help="Repeat rounds for noise averaging")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if args.backend == "CDA":
        os.environ["CDA_ENABLE_MEMORY_SAVING"] = "1"

    import torch
    import pynvml
    from vllm import LLM, SamplingParams

    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

    # --------- power poller thread ---------
    poll_events = []  # list of (t_seconds, power_W)
    running = [True]

    def poll():
        t0 = time.perf_counter()
        while running[0]:
            try:
                p_mw = pynvml.nvmlDeviceGetPowerUsage(handle)  # milliwatts
                t = time.perf_counter() - t0
                poll_events.append((t, p_mw / 1000.0))
            except Exception:
                pass
            time.sleep(0.05)

    # --------- spin up vLLM ---------
    kw = dict(
        model=args.model,
        max_model_len=args.N + args.output_len + 16,
        enforce_eager=True,
        gpu_memory_utilization=0.88,
        dtype="float16",
    )
    if args.backend == "CDA":
        kw["attention_backend"] = "CUSTOM"

    llm = LLM(**kw)

    # Build prompts — B separate requests of length N each.
    VOCAB_SAFE = 100000
    prompts = [
        {"prompt_token_ids": [100 + ((i * 257 + j) % VOCAB_SAFE) for j in range(args.N)]}
        for i in range(args.B)
    ]
    sp = SamplingParams(max_tokens=args.output_len, temperature=0.0, ignore_eos=True)

    # Warmup (1 small round, no measurement)
    _ = llm.generate(prompts, sp, use_tqdm=False)

    # --------- measured rounds ---------
    per_round = []
    for it in range(args.iters):
        poll_events.clear()
        running[0] = True
        thr = threading.Thread(target=poll, daemon=True)
        thr.start()

        torch.cuda.synchronize()
        t_start = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        torch.cuda.synchronize()
        t_end = time.perf_counter()

        running[0] = False
        thr.join(timeout=1.0)

        elapsed_s = t_end - t_start
        # Trapezoid-integrate power over the run window.
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
            "avg_power_w": avg_w,
            "n_samples": len(events),
            "total_tokens": total_tokens,
            "mj_per_tok": energy_j * 1000 / total_tokens if total_tokens else None,
        })

    pynvml.nvmlShutdown()

    # --------- aggregate ---------
    mj_list = [r["mj_per_tok"] for r in per_round if r["mj_per_tok"]]
    res = {
        "backend": args.backend,
        "B": args.B, "N": args.N, "output_len": args.output_len,
        "gpu_id": gpu_id,
        "iters": args.iters,
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
