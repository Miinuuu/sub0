"""Max-context-at-fixed-VRAM bench.

At gpu_memory_utilization={budget}, query vLLM's num_gpu_blocks * block_size
to determine the largest context length each backend can serve.

Key paper claim (Prop 2): CDA K4V2 = 832 B/tok, FP16 FA2 = 4096 B/tok → 4.92×.
At fixed VRAM budget we should therefore see ~5× more context from CDA.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path


def main():
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("FA2", "CDA"), required=True)
    ap.add_argument("--gpu-util", type=float, default=0.5)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    args = ap.parse_args()

    if args.backend == "CDA":
        os.environ["CDA_ENABLE_MEMORY_SAVING"] = "1"

    from vllm import LLM, SamplingParams

    # Start with a very permissive max_model_len — vLLM will allocate as many
    # KV blocks as fit and tell us how many it has.
    MAX_TRY = 131072  # Llama-3.1 native max

    kw = dict(
        model=args.model,
        max_model_len=MAX_TRY,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_util,
        dtype="float16",
    )
    if args.backend == "CDA":
        kw["attention_backend"] = "CUSTOM"

    try:
        llm = LLM(**kw)
        ok = True
        err = ""
        # Peek at engine-side block count.
        try:
            ec = llm.llm_engine.model_executor
            # Different vLLM 0.19 paths; fall back gracefully.
            num_blocks = None
            block_size = None
            for attr_path in [
                ("scheduler", "num_gpu_blocks"),
                ("kv_cache_config", "num_blocks"),
            ]:
                obj = ec
                try:
                    for a in attr_path:
                        obj = getattr(obj, a)
                    num_blocks = obj
                    break
                except AttributeError:
                    continue
            vc = llm.llm_engine.vllm_config
            block_size = getattr(vc.cache_config, "block_size", None)
        except Exception as e:
            num_blocks, block_size = None, None
            err = f"introspection: {e}"
    except Exception as e:
        ok = False
        err = str(e)
        num_blocks, block_size = None, None

    res = {
        "backend": args.backend,
        "gpu_memory_utilization": args.gpu_util,
        "max_model_len_tested": MAX_TRY,
        "ok": ok,
        "error": err[:300],
        "num_gpu_blocks": num_blocks,
        "block_size_tokens": block_size,
        "max_context_capacity": (num_blocks * block_size) if (num_blocks and block_size) else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
