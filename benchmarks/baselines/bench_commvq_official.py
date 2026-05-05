"""CommVQ Triton-kernel score latency bench.

Uses the 3 standalone wrappers from CommVQ/commvq/triton_kernels.py:
- calculate_QC_T_triton(query_states, centers)
- gatherC1C2_and_sum_over_residual_and_apply_rope_and_sum_triton(...)

Config (from triton_kernels.py example comments): H=32, DIM=head_dim//2=64, R=16, C=64.
Shapes for Llama-3.1-8B: H_q=32, H_kv=8, group=4, head_dim=128.

Usage: CUDA_VISIBLE_DEVICES=0 python experiments/bench_commvq_official.py
"""
import sys, json, subprocess
from pathlib import Path
from statistics import median
import torch

COMMVQ_ROOT = "<HOME>/ing/research/REF/CommVQ"
sys.path.insert(0, COMMVQ_ROOT)
from commvq.triton_kernels import (
    calculate_QC_T_triton,
    gatherC1C2_and_sum_over_residual_and_apply_rope_and_sum_triton,
    attn_weights_mul_value_triton,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cda_attn import _get_hmma
from core.compression import HadamardQuantCompressor
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]

SHAPES = [
    (1, 4096), (1, 8192), (1, 32768), (1, 65536),
    (16, 32768), (16, 65536), (32, 65536),
]

# Llama-3.1-8B
H_q, H_kv, D = 32, 8, 128
GROUP = 4
DIM_PAIR = D // 2  # 64 (paired dims)
R = 16  # residual layers per CommVQ example
C = 64  # codebook size
H_RATIO = 4
WARMUP = 10
ITERS = 40


def _time_us(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(ITERS):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    return median(ts)


def bench_commvq(B, N, device="cuda"):
    """Time CommVQ full decode-attention: QC_T + gather + softmax + V output."""
    dtype = torch.bfloat16

    # query_states pre-reshape: (B, H_q, q_len=1, DIM_PAIR, 2)
    Q = torch.randn(B, H_q, 1, DIM_PAIR, 2, dtype=dtype, device=device).contiguous()
    # centers: (H_q, DIM_PAIR, R, C, 2, 2)
    centers = torch.randn(H_q, DIM_PAIR, R, C, 2, 2, dtype=dtype, device=device).contiguous()
    # key_code: (B, H_kv, R, N)
    key_code = torch.randint(0, C, (B, H_kv, R, N), dtype=torch.int32, device=device)
    # Rotary sin/cos: (N, DIM_PAIR)
    qsin = torch.randn(N, DIM_PAIR, dtype=dtype, device=device).contiguous()
    qcos = torch.randn(N, DIM_PAIR, dtype=dtype, device=device).contiguous()
    # prescale: (B, N)
    prescale_k = torch.randn(B, N, dtype=dtype, device=device).contiguous()

    import math
    factor = math.sqrt(D)

    # Pre-compute QC_T for score bench (also times separately)
    QC_T = calculate_QC_T_triton(Q, centers)

    def _run_qc_t():
        calculate_QC_T_triton(Q, centers)
    t_qc = _time_us(_run_qc_t)

    # Attention scores: (B, H_q, N) after unflatten
    attn_weights_pre = gatherC1C2_and_sum_over_residual_and_apply_rope_and_sum_triton(
        key_code, QC_T, qsin, qcos, prescale_k, factor, H_RATIO=H_RATIO)

    def _run_gather():
        gatherC1C2_and_sum_over_residual_and_apply_rope_and_sum_triton(
            key_code, QC_T, qsin, qcos, prescale_k, factor, H_RATIO=H_RATIO)
    t_gather = _time_us(_run_gather)

    # --- Softmax (fp32) over attention scores ---
    # attn_weights_pre shape: (B, KV_H * H_RATIO, N) = (B, H_q, N)
    def _run_softmax():
        F.softmax(attn_weights_pre.to(torch.float32), dim=-1)
    t_softmax = _time_us(_run_softmax)

    # V-output skipped (attn_weights_mul_value_triton shape requirements complex for synthetic).
    # Caveat documented in caption: competitor full decode = score + softmax + V-kernel;
    # we report score + softmax here, which favours the competitor.
    total = t_qc + t_gather + t_softmax
    return {
        "commvq_qct_us": t_qc,
        "commvq_gather_us": t_gather,
        "commvq_softmax_us": t_softmax,
        "commvq_total_us": total,
    }


def bench_cda_full(B, N, device=None):
    if device is None or isinstance(device, str):
        device = torch.device("cuda")
    """Time CDA full attention wrapper (split + reduce + rotation)."""
    kc = HadamardQuantCompressor(dim=D, bit_width=4, half_rotation=True)
    vc = HadamardQuantCompressor(dim=D, bit_width=2, half_rotation=True)
    kc._ensure_tensors(device); vc._ensure_tensors(device)

    BLOCK = 16
    n_blk_per = (N + BLOCK - 1) // BLOCK
    num_blocks = B * n_blk_per + 4
    kv_cache = torch.zeros(num_blocks, BLOCK, H_kv, 104, dtype=torch.uint8, device=device)
    block_table = torch.zeros(B, n_blk_per, dtype=torch.int32, device=device)
    for b in range(B):
        block_table[b] = torch.arange(b*n_blk_per, (b+1)*n_blk_per, dtype=torch.int32, device=device)
    seq_lens = torch.full((B,), N, dtype=torch.int32, device=device)

    Q_rot = torch.randn(B, H_q, D, dtype=torch.float32, device=device)
    cb_k = (kc._centroids * 2.0 - 1.0).float().contiguous().to(device)
    cb_v = (vc._centroids * 2.0 - 1.0).float().contiguous().to(device)
    tile = 512 if N <= 16384 else 1024
    SCALE = 1.0/(D**0.5)
    flat = kv_cache.view(num_blocks, BLOCK, H_kv, 104)

    ext = _get_hmma()

    def _run():
        ext.cda_flash_split_k4v2_gqa_paged_batched_coop_hmma(
            Q_rot, flat, block_table, seq_lens, cb_k, cb_v,
            GROUP, BLOCK, tile, N, SCALE)
    try:
        _ = ext.cda_flash_split_k4v2_gqa_paged_batched_coop_hmma(
            Q_rot, flat, block_table, seq_lens, cb_k, cb_v,
            GROUP, BLOCK, tile, N, SCALE)
        torch.cuda.synchronize()
        t_split = _time_us(_run)
        # Add softmax-equivalent cost for fair symmetric comparison with competitors
        # (CDA's online softmax is already inside split kernel)
        return {"cda_split_us": t_split, "cda_total_us": t_split}
    except Exception as ex:
        return {"cda_split_us": None, "cda_err": str(ex)[:200]}


def run_single(B, N):
    torch.manual_seed(0xCDA)
    cm = bench_commvq(B, N)
    torch.cuda.empty_cache()
    torch.manual_seed(0xCDA)
    cda = bench_cda_full(B, N)
    out = {**cm, **cda}
    return out


def main():
    results = []
    this_script = str(Path(__file__).resolve())
    for B, N in SHAPES:
        print(f"\n=== B={B} N={N} ===", flush=True)
        row = {"B": B, "N": N}
        try:
            r = subprocess.run(
                [sys.executable, this_script, "--single", str(B), str(N)],
                capture_output=True, text=True, timeout=600,
                env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": "0"})
            if r.returncode == 0:
                for line in r.stdout.splitlines():
                    if line.startswith("RESULT:"):
                        cm = json.loads(line[len("RESULT:"):].strip())
                        row.update(cm)
                        if cm.get('commvq_total_us') is not None:
                            cda_v = cm.get('cda_total_us', -1)
                            ratio = cm['commvq_total_us'] / cda_v if cda_v else None
                            print(f"  CommVQ total={cm['commvq_total_us']:7.1f}  CDA split={cda_v if cda_v else 'NA':>7}  ratio={ratio:.2f}x" if ratio else f"  CommVQ total={cm['commvq_total_us']:7.1f}")
                        else:
                            print(f"  CommVQ err: {cm.get('error','')[:200]}")
                        break
                else:
                    print(f"  no RESULT. stdout: {r.stdout[-300:]}")
                    row["error"] = "no result"
            else:
                print(f"  subproc exit {r.returncode}. stderr: {r.stderr[-300:]}")
                row["error"] = f"exit {r.returncode}"
        except subprocess.TimeoutExpired:
            print("  timeout")
            row["error"] = "timeout"
        results.append(row)

    out = REPO / "runs" / "bench_commvq_official.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {
            "H_q": H_q, "H_kv": H_kv, "D": D, "group_size": GROUP,
            "R": R, "C": C, "H_RATIO": H_RATIO, "warmup": WARMUP, "iters": ITERS,
        },
        "results": results,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--single":
        B, N = int(sys.argv[2]), int(sys.argv[3])
        try:
            r = run_single(B, N)
            print("RESULT:", json.dumps(r), flush=True)
        except Exception as ex:
            print("RESULT:", json.dumps({"commvq_score_us": None, "error": str(ex)[:300]}), flush=True)
    else:
        main()
