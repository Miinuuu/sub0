"""bench_longbench.py — Table 5 RULER reproducer.

Runs the 13 RULER subtasks (8 NIAH + 5 RULER) on
``meta-llama/Llama-3.1-8B-Instruct`` under either FA2 or CDA, via
lm-evaluation-harness with the vLLM model interface. Aggregates
``Macro`` (mean over all 13 subtasks) and ``NIAH`` (mean over the 8
needle-in-haystack subtasks) into the paper Table 5 format.

Replaces the legacy LongBench-single driver that depended on
``cda.eval.cda_decode``; the paper used lm-evaluation-harness directly,
so this rewrite mirrors that path.

Usage::

    # FA2 reference, 32K context (paper Table 5 row N=32K, FA2 col)
    PYTHONPATH=. python benchmarks/baselines/bench_longbench.py \\
        --backend FA2 --max-model-len 32768 \\
        --output runs/longbench_ruler/fa2_32k.json

    # CDA, 32K context
    PYTHONPATH=. python benchmarks/baselines/bench_longbench.py \\
        --backend CDA --max-model-len 32768 \\
        --output runs/longbench_ruler/cda_32k.json

    # Smoke (limit 5 examples per task)
    PYTHONPATH=. python benchmarks/baselines/bench_longbench.py \\
        --backend FA2 --max-model-len 8192 --limit 5 \\
        --output /tmp/_t5_smoke.json

Requires (cda env):
* ``lm_eval`` (lm-evaluation-harness >= 0.4.x with RULER/NIAH tasks)
* vLLM 0.19 with the CDA backend registered for ``--backend CDA``
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# 13 RULER subtasks: 8 NIAH + 5 RULER (paper Table 5 caption)
NIAH_TASKS = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multiquery",
    "niah_multivalue",
]
RULER_NON_NIAH = [
    "ruler_cwe",
    "ruler_fwe",
    "ruler_qa_hotpot",
    "ruler_qa_squad",
    "ruler_vt",
]
ALL_TASKS = NIAH_TASKS + RULER_NON_NIAH


def _primary_metric(task_result: dict) -> float | None:
    """Pick a single scalar score from an lm-eval task result dict.

    Each RULER subtask exposes one or more ``metric,filter`` pairs
    (e.g. ``exact_match,none`` or ``contains,none``). We take the
    first numeric value that is not a stderr / alias entry.
    """
    skip_suffix = ("_stderr", ",std", "alias")
    for k, v in task_result.items():
        if any(s in k for s in skip_suffix):
            continue
        if isinstance(v, (int, float)):
            return float(v)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=["FA2", "CDA"], required=True,
        help="Attention backend (FA2 = vanilla FlashAttention 2; "
             "CDA = K4V4 compressed)",
    )
    parser.add_argument(
        "--model", default="meta-llama/Llama-3.1-8B-Instruct",
        help="HF model id (paper default: Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=32768,
        help="Max sequence length (paper Table 5 uses {8192, 16384, 32768})",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=ALL_TASKS,
        help="Subset of RULER subtasks (default: all 13)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Examples per task (smoke runs only — paper uses full set)",
    )
    parser.add_argument(
        "--gpu-mem-util", type=float, default=0.85,
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--enforce-eager", action="store_true",
    )
    parser.add_argument(
        "--no-cda-memory-saving", action="store_true",
        help="Skip the cda_attn_v2 KV-cache page-size patch (debug only). "
             "Without the patch vLLM allocates FA2-shaped KV buffers and the "
             "CDA backend cannot view them.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Aggregated JSON output path",
    )
    args = parser.parse_args()

    if args.backend == "CDA":
        # The CDA attention backend has two pieces that must reach every
        # vLLM worker process (TP and EngineCore subprocesses included):
        #   (1) backend class registration in vllm.v1.attention.backends
        #   (2) the AttentionSpec.real_page_size_bytes patch so vLLM
        #       allocates K4V4-shaped KV buffers (144 B/slot) instead of
        #       the FA2 default (512 B/slot)
        # ``cda/vllm_plugin.py`` is registered as the
        # ``vllm.general_plugins/cda_v2`` entry point and gates both
        # pieces on these env vars. Setting them BEFORE vLLM imports
        # ensures every subprocess inherits the configuration.
        os.environ.setdefault("VLLM_PLUGINS", "cda_v2")
        if not args.no_cda_memory_saving:
            os.environ["CDA_V2_ENABLE_MEMORY_SAVING"] = "1"
        from cda.kernels_cuda.vllm_integration.cda_attn_v2 import (
            register_backend, enable_cda_memory_saving,
        )
        if not args.no_cda_memory_saving:
            enable_cda_memory_saving()
        register_backend("CDA")
        attention_config = {"backend": "CDA"}
    else:
        attention_config = {"backend": "FLASH_ATTN", "flash_attn_version": 2}

    import lm_eval
    from lm_eval.models.vllm_causallms import VLLM

    print(f"[bench_longbench] backend={args.backend} "
          f"max_model_len={args.max_model_len} tasks={len(args.tasks)}")
    print(f"[bench_longbench] tasks: {args.tasks}")

    lm = VLLM(
        pretrained=args.model,
        dtype="float16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        tensor_parallel_size=1,
        seed=args.seed,
        enforce_eager=bool(args.enforce_eager),
        attention_config=attention_config,
    )

    # RULER tasks need (a) the tokenizer for haystack synthesis and
    # (b) the sequence length to bucket each sample. Both go through
    # the lm-eval metadata dict; see lm_eval/tasks/ruler/README.md.
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=list(args.tasks),
        limit=args.limit,
        batch_size="auto",
        apply_chat_template=True,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
        metadata={
            "tokenizer": args.model,
            "max_seq_lengths": [args.max_model_len],
        },
    )

    task_results = results["results"]
    niah_scores = [
        s for t in NIAH_TASKS
        if t in task_results and (s := _primary_metric(task_results[t])) is not None
    ]
    macro_scores = [
        s for t in ALL_TASKS
        if t in task_results and (s := _primary_metric(task_results[t])) is not None
    ]

    summary = {
        "config": {
            "backend": args.backend,
            "model": args.model,
            "max_model_len": args.max_model_len,
            "tasks": list(args.tasks),
            "limit": args.limit,
            "gpu_mem_util": args.gpu_mem_util,
            "seed": args.seed,
            "enforce_eager": bool(args.enforce_eager),
            "attention_config": attention_config,
        },
        "macro_13_subtasks": (sum(macro_scores) / len(macro_scores)
                              if macro_scores else None),
        "niah_8_subtasks": (sum(niah_scores) / len(niah_scores)
                            if niah_scores else None),
        "n_macro": len(macro_scores),
        "n_niah": len(niah_scores),
        "task_results": task_results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))

    print()
    macro = summary["macro_13_subtasks"]
    niah = summary["niah_8_subtasks"]
    print(f"[bench_longbench] Macro (13 subtasks): "
          + (f"{macro:.4f}" if macro is not None else "n/a"))
    print(f"[bench_longbench] NIAH  (8 subtasks):  "
          + (f"{niah:.4f}" if niah is not None else "n/a"))
    print(f"[bench_longbench] saved → {out_path}")


if __name__ == "__main__":
    main()
