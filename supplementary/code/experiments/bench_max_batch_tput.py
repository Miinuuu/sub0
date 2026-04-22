"""Phase 4b/5: measure throughput at the MAX batch each backend can handle
at the same prompt length. CDA's 4.92× smaller KV should let it pack
significantly more concurrent requests, potentially beating FA2 on total
tok/s even if per-request latency is higher.
"""
import os, sys, time, argparse, gc
import torch
REPO = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO)

ap = argparse.ArgumentParser()
ap.add_argument("--backend", required=True, choices=("FLASH_ATTN", "CDA"))
ap.add_argument("--N", type=int, default=8192)
ap.add_argument("--gen", type=int, default=32)
ap.add_argument("--batches", type=str, default="1,4,8,16,32,64")
args = ap.parse_args()

if args.backend == "CDA":
    os.environ["CDA_ENABLE_MEMORY_SAVING"] = "1"

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

batches = [int(b) for b in args.batches.split(",")]
kw = dict(
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_model_len=args.N + args.gen + 256,
    max_num_seqs=max(batches),
    enforce_eager=True,
    gpu_memory_utilization=0.85,
    dtype="float16",
)
if args.backend == "CDA":
    kw["attention_backend"] = "CUSTOM"

llm = LLM(**kw)
sp = SamplingParams(max_tokens=args.gen, temperature=0.0, ignore_eos=True)

import json as _json_b
_out_path = f"/tmp/bench_max_batch_{args.backend}_N{args.N}.json"
results = []
for B in batches:
    # Wrap within a safe vocab-sized range so N>100K doesn't overflow (e.g. Llama-3.1 vocab=128256).
    VOCAB_SAFE = 100000
    prompts = [TokensPrompt(prompt_token_ids=[100 + ((i + j) % VOCAB_SAFE) for j in range(args.N)])
                for i in range(B)]
    # Warmup
    try:
        llm.generate(prompts, sp, use_tqdm=False)
        llm.generate(prompts, sp, use_tqdm=False)
    except Exception as e:
        print(f"[{args.backend}] B={B}  FAILED warmup: {str(e)[:100]}", flush=True)
        results.append({"backend": args.backend, "B": B, "N": args.N,
                         "ok": False, "error": str(e)[:200]})
        with open(_out_path, "w") as f: _json_b.dump(results, f, indent=2)
        gc.collect(); torch.cuda.empty_cache()
        continue
    torch.cuda.synchronize()
    t0 = time.time()
    out = llm.generate(prompts, sp, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    total_tok = B * args.gen
    per_tok_req = elapsed / args.gen * 1000  # ms per token per request
    tok_per_s = total_tok / elapsed
    results.append({"backend": args.backend, "B": B, "N": args.N, "ok": True,
                     "tok_per_s": round(tok_per_s, 1),
                     "per_req_tok_ms": round(per_tok_req, 2),
                     "elapsed_ms": round(elapsed*1000, 0)})
    with open(_out_path, "w") as f: _json_b.dump(results, f, indent=2)
    print(f"[{args.backend}] B={B:>3d}  tput={tok_per_s:>8.1f} tok/s  "
          f"per-req-tok={per_tok_req:.2f} ms  elapsed={elapsed*1000:.0f} ms",
          flush=True)
    gc.collect(); torch.cuda.empty_cache()
