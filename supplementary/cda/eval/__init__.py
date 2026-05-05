"""Evaluation utilities for paper Tables 4/5/A5.

Provides dataset text fetchers (``cda.eval.ppl``) used by
``benchmarks/baselines/bench_ppl_scale.py`` and friends. The ABI here
matches the legacy ``cda-v1`` ``cda.eval`` module so the canonical PPL
protocol (stride=64, max_ctx=4096, filter_empty_lines=True,
add_special_tokens=True) reproduces the paper's WikiText-2 / C4 numbers.
"""
