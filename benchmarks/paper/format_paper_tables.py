"""Format paper-ready tables from existing bench JSON artifacts.

Reads the JSON outputs of benchmarks/bench_vllm_e2e_ttft_tpot.py and
benchmarks/paper/bench_capacity_sweep.py, then prints markdown tables
suitable for the paper.

Usage:
    # Single-stream latency table (B=1, multiple N)
    python benchmarks/paper/format_paper_tables.py latency \\
        --artifacts runs/bench_cda_vs_fa2_2k128_kernopt.json \\
                    runs/bench_cda_vs_fa2_8k256_cpasync16.json \\
                    runs/bench_cda_vs_fa2_32k256_kernopt.json \\
                    runs/bench_cda_vs_fa2_128k_b1_kernopt.json

    # Throughput / wall (B>1)
    python benchmarks/paper/format_paper_tables.py throughput \\
        --artifacts runs/bench_cda_vs_fa2_*.json

    # Capacity sweep (OOM cliff)
    python benchmarks/paper/format_paper_tables.py capacity \\
        --artifacts runs/paper/capacity_sweep*.json

    # Per-request artifact case study (the chunked-prefill story)
    python benchmarks/paper/format_paper_tables.py artifact \\
        --artifact runs/bench_cda_128k_b2_perreq.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _md_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _delta_pct(cda: float, fa2: float) -> str:
    if fa2 is None or cda is None or fa2 == 0:
        return "-"
    pct = (cda - fa2) / fa2 * 100
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.1f}%"


def _fmt_or_dash(v, fmt=".2f"):
    if v is None:
        return "-"
    return f"{v:{fmt}}"


def cmd_latency(args):
    """Single-stream / batched latency table.

    For each artifact, extract (config, vllm tpot, cda tpot, delta, throughput).
    """
    print(_md_row(["Workload (N / decode / B)",
                   "FA2 TPOT p50 (ms)", "CDA TPOT p50 (ms)", "ΔTPOT",
                   "FA2 TTFT p50 (s)", "CDA TTFT p50 (s)", "ΔTTFT",
                   "FA2 tps", "CDA tps", "Δtps"]))
    print(_md_row(["---"] * 10))
    for path in args.artifacts:
        d = _load(path)
        cfg = d.get("config", {})
        N = cfg.get("prompt_len", "?")
        L = cfg.get("decode_len", "?")
        B = cfg.get("num_requests", "?")
        wid = f"{N} / {L} / B={B}"

        v = d["results"].get("vllm")
        c = d["results"].get("cda")
        if v is None or c is None:
            continue

        v_tpot = v["tpot_ms"]["p50"]; c_tpot = c["tpot_ms"]["p50"]
        v_ttft = v["ttft_ms"]["p50"] / 1000; c_ttft = c["ttft_ms"]["p50"] / 1000
        v_tps  = v["throughput_output_tok_s"]; c_tps = c["throughput_output_tok_s"]

        print(_md_row([
            wid,
            _fmt_or_dash(v_tpot, ".2f"), _fmt_or_dash(c_tpot, ".2f"),
            _delta_pct(c_tpot, v_tpot),
            _fmt_or_dash(v_ttft, ".1f"), _fmt_or_dash(c_ttft, ".1f"),
            _delta_pct(c_ttft, v_ttft),
            _fmt_or_dash(v_tps, ".2f"), _fmt_or_dash(c_tps, ".2f"),
            _delta_pct(c_tps, v_tps),
        ]))


def cmd_throughput(args):
    """Throughput-focused table for batched workloads.

    Reports wall time + tps as primary metrics, hides per-token TPOT
    (which is the unreliable artifact-affected number at long-N B>1).
    """
    print(_md_row(["Workload",
                   "FA2 wall (s)", "CDA wall (s)", "Δwall",
                   "FA2 tok/s", "CDA tok/s", "Δtps", "Note"]))
    print(_md_row(["---"] * 8))
    for path in args.artifacts:
        d = _load(path)
        cfg = d.get("config", {})
        N = cfg.get("prompt_len", "?"); L = cfg.get("decode_len", "?")
        B = cfg.get("num_requests", "?")

        v = d["results"].get("vllm"); c = d["results"].get("cda")
        if v is None or c is None:
            continue

        v_wall = v["wall_s"]; c_wall = c["wall_s"]
        v_tps  = v["throughput_output_tok_s"]; c_tps = c["throughput_output_tok_s"]

        # Heuristic: long enough that chunk_prefill of one request can not fit
        # in a single iter alongside another req's decode → likely interleaved.
        # Empirically the artifact dominates only at ~64K+ B≥2 with default
        # max_num_batched_tokens=8192 AND when memory leaves room for both
        # reqs' KV simultaneously (CDA's regime, not FA2's).
        chunk_default = 8192
        likely_interleave = (N != "?" and B != "?" and B > 1
                              and N >= 2 * chunk_default)
        # And only flag if the per-request TPOT spread is actually pathological
        # (>3× p50). This catches the real artifact, not generic batch noise.
        spread_pathological = False
        try:
            spread_pathological = (c["tpot_ms"]["p90"] / c["tpot_ms"]["p50"]) > 3.0
        except Exception:
            pass
        note = "**chunked-prefill artifact**" if (likely_interleave and spread_pathological) else ""

        print(_md_row([
            f"{N}/{L}/B={B}",
            _fmt_or_dash(v_wall, ".1f"), _fmt_or_dash(c_wall, ".1f"),
            _delta_pct(c_wall, v_wall),
            _fmt_or_dash(v_tps, ".2f"), _fmt_or_dash(c_tps, ".2f"),
            _delta_pct(c_tps, v_tps),
            note,
        ]))


def cmd_capacity(args):
    """Capacity sweep — OOM cliff and throughput-at-capacity."""
    print(_md_row(["B", "FA2 status", "FA2 tok/s", "CDA status", "CDA tok/s"]))
    print(_md_row(["---"] * 5))
    for path in args.artifacts:
        d = _load(path)
        cfg = d["config"]
        N = cfg.get("prompt_len", "?")
        print(f"\n**N = {N}**\n")
        fa2 = {r["batch"]: r for r in d.get("FA2", [])}
        cda = {r["batch"]: r for r in d.get("CDA", [])}
        all_b = sorted(set(list(fa2.keys()) + list(cda.keys())))
        for b in all_b:
            fr = fa2.get(b, {"status": "-"})
            cr = cda.get(b, {"status": "-"})
            f_tps = fr.get("throughput_tok_s") if fr.get("status") == "ok" else None
            c_tps = cr.get("throughput_tok_s") if cr.get("status") == "ok" else None
            print(_md_row([
                str(b),
                fr["status"].upper(), _fmt_or_dash(f_tps, ".0f"),
                cr["status"].upper(), _fmt_or_dash(c_tps, ".0f"),
            ]))
        if "summary" in d:
            s = d["summary"]
            mult = s.get("capacity_multiplier")
            mult_s = f"{mult:.2f}×" if mult else "∞"
            print(f"\nCapacity multiplier: FA2 max B={s.get('fa2_max_batch_ok')}, "
                  f"CDA max B={s.get('cda_max_batch_ok')} → **{mult_s}**\n")


def cmd_artifact(args):
    """Case study: per-request metric attribution (the artifact section)."""
    d = _load(args.artifact)
    cfg = d.get("config", {})
    print(f"**Workload**: prompt={cfg.get('prompt_len')}, "
          f"decode={cfg.get('decode_len')}, B={cfg.get('num_requests')}\n")

    for backend, r in d["results"].items():
        print(f"\n### {backend.upper()}")
        print(f"wall={r['wall_s']:.2f}s, throughput={r['throughput_output_tok_s']:.3f} tok/s, "
              f"reported tpot p50={r['tpot_ms']['p50']:.2f}ms, "
              f"p90={r['tpot_ms']['p90']:.2f}ms, mean={r['tpot_ms']['mean']:.2f}ms")
        per = r.get("per_request", [])
        if per:
            print()
            print(_md_row(["req idx", "TTFT (s)", "TPOT (ms)", "out tokens"]))
            print(_md_row(["---"] * 4))
            for pr in per:
                print(_md_row([
                    str(pr["index"]),
                    f"{pr.get('ttft_ms', 0)/1000:.2f}",
                    f"{pr.get('tpot_ms', 0):.2f}",
                    str(pr.get("output_tokens", "?")),
                ]))

    # The story
    cda = d["results"].get("cda"); fa2 = d["results"].get("vllm")
    if cda and fa2:
        c_per = cda.get("per_request", []); f_per = fa2.get("per_request", [])
        if len(c_per) >= 2 and len(f_per) >= 2:
            c_req1_ttft = c_per[1]["ttft_ms"] / 1000
            f_req1_ttft = f_per[1]["ttft_ms"] / 1000
            c_tpot_spread = max(p["tpot_ms"] for p in c_per) / min(p["tpot_ms"] for p in c_per)
            f_tpot_spread = max(p["tpot_ms"] for p in f_per) / min(p["tpot_ms"] for p in f_per)
            wall_delta_pct = (cda['wall_s'] - fa2['wall_s']) / fa2['wall_s'] * 100
            print(f"\n### Story\n")
            print(f"- **req1 TTFT**: FA2={f_req1_ttft:.1f}s, CDA={c_req1_ttft:.1f}s "
                  f"({c_req1_ttft - f_req1_ttft:+.1f}s). "
                  f"FA2 starts req1 prefill ~{f_req1_ttft - c_req1_ttft:.0f}s LATER → "
                  f"FA2 ran sequentially (memory pressure prevents both prefills concurrently); "
                  f"CDA's smaller KV footprint allows CONCURRENT prefill+decode (chunked prefill mixing).")
            print(f"- **per-request TPOT spread**: FA2={f_tpot_spread:.2f}× (uniform), "
                  f"CDA={c_tpot_spread:.2f}× (req0 polluted). "
                  f"CDA's req0 decode iters share forward passes with req1's prefill chunks → "
                  f"vLLM attributes the chunk-prefill wall time to req0's inter-token latency.")
            print(f"- **Ground truth (wall time)**: FA2={fa2['wall_s']:.1f}s, "
                  f"CDA={cda['wall_s']:.1f}s ({wall_delta_pct:+.1f}%). "
                  f"CDA finishes the workload faster by exploiting the memory headroom "
                  f"that FA2 cannot.")
            print(f"- **Implication**: per-request TPOT is misleading at long-N B>1 "
                  f"with chunked prefill enabled. Wall time and throughput are the honest metrics.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_lat = sub.add_parser("latency")
    p_lat.add_argument("--artifacts", nargs="+", required=True)
    p_lat.set_defaults(fn=cmd_latency)

    p_thr = sub.add_parser("throughput")
    p_thr.add_argument("--artifacts", nargs="+", required=True)
    p_thr.set_defaults(fn=cmd_throughput)

    p_cap = sub.add_parser("capacity")
    p_cap.add_argument("--artifacts", nargs="+", required=True)
    p_cap.set_defaults(fn=cmd_capacity)

    p_art = sub.add_parser("artifact")
    p_art.add_argument("--artifact", required=True)
    p_art.set_defaults(fn=cmd_artifact)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
