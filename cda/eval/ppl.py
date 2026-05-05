"""Canonical sliding-window perplexity evaluation.

All PPL-producing benchmarks (Tab. PPL, ablations, C4) share this single
``eval_ppl`` entry point so that the protocol (tokenization, text join,
stride, n_positions, forward loop) cannot drift between scripts.

Design notes
------------
* ``filter_empty_lines=True`` and ``add_special_tokens=True`` are canonical.
  The paper's Tab. PPL numbers were generated under these settings.
* A non-canonical run must pass ``canonical=False`` explicitly; otherwise
  the function asserts, preventing silent protocol divergence.
* The returned dict embeds a ``"protocol"`` field for audit trail.

Every entry in ``runs/`` from 2026-04-23 onwards should carry the
``protocol`` field so cross-run consistency can be verified after the fact.

Ported from legacy v1 ``core/ppl_eval.py`` — keeping the same API ensures
that runs/v1 cached JSONs are directly comparable.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F


_WIKITEXT_CACHE: dict[bool, str] = {}
_C4_CACHE: Optional[str] = None


def get_wikitext_text(filter_empty_lines: bool = True) -> str:
    """Return WikiText-2 test split joined into one string.

    With ``filter_empty_lines=True`` (canonical), empty lines are dropped
    before joining so the final text does not contain ``\\n\\n\\n\\n``
    runs that artificially deflate PPL. The paper's Tab. PPL uses this.
    """
    key = filter_empty_lines
    if key not in _WIKITEXT_CACHE:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        if filter_empty_lines:
            _WIKITEXT_CACHE[key] = "\n\n".join(t for t in ds["text"] if t.strip())
        else:
            _WIKITEXT_CACHE[key] = "\n\n".join(ds["text"])
    return _WIKITEXT_CACHE[key]


def get_c4_text(max_docs: int = 500) -> str:
    """Return the first ``max_docs`` C4 validation docs joined by ``\\n\\n``."""
    global _C4_CACHE
    if _C4_CACHE is None:
        from datasets import load_dataset
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        texts = []
        for i, item in enumerate(ds):
            if i >= max_docs:
                break
            texts.append(item["text"])
        _C4_CACHE = "\n\n".join(texts)
    return _C4_CACHE


def _load_dataset_text(name: str, filter_empty_lines: bool) -> str:
    if name == "wikitext2":
        return get_wikitext_text(filter_empty_lines=filter_empty_lines)
    if name == "c4":
        return get_c4_text()
    raise ValueError(f"unknown dataset: {name!r} (expected 'wikitext2' or 'c4')")


@torch.no_grad()
def eval_ppl(
    model,
    tokenizer,
    forward_fn: Callable,
    *,
    max_ctx: int = 4096,
    stride: int = 64,
    n_positions: Optional[int] = 63,
    add_special_tokens: bool = True,
    filter_empty_lines: bool = True,
    dataset: str = "wikitext2",
    desc: str = "",
    canonical: bool = True,
) -> dict:
    """Sliding-window per-position decode PPL.

    At each ``pos ∈ {stride, 2*stride, ...}`` up to ``n_positions`` steps:
        ctx  = ids[:, :pos]          # prefill
        nxt  = ids[:, pos:pos+1]     # the single decode token
        tgt  = ids[:, pos+1:pos+2]   # ground-truth next
        logits = forward_fn(ctx, nxt)        # (1, 1, V)
        nll  += cross_entropy(logits, tgt)
    The per-position mean of NLL is exp-ed to produce PPL.

    ``forward_fn(ctx, nxt)`` must return the logits of the single decode
    token. Typical callers wrap the model's cache-aware decode path
    (FP16 past_kv, CDA compressed cache, KIVI quantized cache, etc.).

    If the caller needs to deviate from the canonical protocol (for
    example, to reproduce a legacy script), ``canonical=False`` must be
    set explicitly. Otherwise this asserts and aborts.
    """
    if canonical:
        assert add_special_tokens is True, (
            "canonical protocol requires add_special_tokens=True; set "
            "canonical=False to override and document the divergence"
        )
        assert filter_empty_lines is True, (
            "canonical protocol requires filter_empty_lines=True; set "
            "canonical=False to override and document the divergence"
        )
        assert max_ctx == 4096 and stride == 64 and n_positions == 63, (
            "canonical protocol: max_ctx=4096, stride=64, n_positions=63"
        )

    text = _load_dataset_text(dataset, filter_empty_lines)
    enc = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
        max_length=max_ctx,
        truncation=True,
    )
    ids = enc.input_ids.to(next(model.parameters()).device)

    positions = list(range(stride, min(ids.shape[1] - 1, max_ctx), stride))
    if n_positions is not None:
        positions = positions[:n_positions]

    nlls = []
    t0 = time.time()
    for pos in positions:
        ctx = ids[:, :pos]
        nxt = ids[:, pos:pos + 1]
        tgt = ids[:, pos + 1:pos + 2]
        if tgt.numel() == 0:
            break
        logits = forward_fn(ctx, nxt)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
        nlls.append(float(loss.item()))
    elapsed = time.time() - t0
    ppl = float(np.exp(np.mean(nlls))) if nlls else float("inf")

    return {
        "ppl": round(ppl, 4),
        "n_pos": len(nlls),
        "time_s": round(elapsed, 1),
        "desc": desc,
        "protocol": {
            "harness": "cda.eval.ppl.eval_ppl",
            "dataset": dataset,
            "add_special_tokens": add_special_tokens,
            "filter_empty_lines": filter_empty_lines,
            "max_ctx": max_ctx,
            "stride": stride,
            "n_positions": n_positions,
            "canonical": canonical,
        },
    }


__all__ = ["eval_ppl", "get_wikitext_text", "get_c4_text"]
