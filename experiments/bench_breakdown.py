"""CDA decode latency breakdown: K-Score / Softmax+TopK / V-Output / Rotation.

Measures each stage separately using CUDA events.
Uses GQA-aware kernels (_cda_gqa_kernels) for accurate paper reproduction.

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/bench_breakdown.py
"""
import gc, json
import torch
import torch.nn.functional as F
from pathlib import Path

from cda import HadamardQuantCompressor
from cda.cuda_attention_gqa import score_4b_gqa, vfull_2b_gqa, vsparse_2b_gqa

PROJECT_ROOT = Path(__file__).resolve().parents[1]
device = "cuda"
D = 128
H_q = 32
H_kv = 8
GROUP = H_q // H_kv
scale = D ** -0.5
TOPK = 128
ITERS = 100

print("Loading GQA CUDA kernels...")

comp2 = HadamardQuantCompressor(dim=D, bit_width=2, half_rotation=True)
comp4 = HadamardQuantCompressor(dim=D, bit_width=4, half_rotation=True)
comp2._ensure_tensors(torch.device(device))
comp4._ensure_tensors(torch.device(device))
cb2 = (comp2._centroids * 2.0 - 1.0).float().to(device)
cb4 = (comp4._centroids * 2.0 - 1.0).float().to(device)
rot = comp2._rotation.float().to(device)
rot_t = comp2._rotation_t.float().to(device)


def cuda_time(fn, iters=ITERS):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


results = {}

for N in [1024, 4096, 16384, 32768, 65536]:
    topk = TOPK if N >= 16384 else 0
    print(f"\n=== N={N}, TopK={topk} ===")

    # Prepare data (GQA layout: Q=[H_q,D], KV=[H_kv,N,...])
    Qr = torch.randn(H_q, D, device=device, dtype=torch.float32)
    Qr = (Qr @ rot_t).float()
    pk4 = torch.randint(0, 256, (H_kv * N, D // 2), device=device, dtype=torch.uint8)
    nk4 = torch.randn(H_kv * N, device=device, dtype=torch.float32).abs() * 10
    pv2 = torch.randint(0, 256, (H_kv * N, D // 4), device=device, dtype=torch.uint8)
    nv2 = torch.randn(H_kv * N, device=device, dtype=torch.float32).abs() * 3

    # 1. K-Score (GQA)
    t_kscore = cuda_time(lambda: score_4b_gqa(Qr, pk4, nk4, cb4, N, GROUP, scale))
    scores = score_4b_gqa(Qr, pk4, nk4, cb4, N, GROUP, scale)

    # 2. Softmax
    t_softmax = cuda_time(lambda: torch.softmax(scores, dim=-1))
    attn = torch.softmax(scores, dim=-1)

    # 3. TopK (if enabled)
    if topk > 0:
        def topk_fn():
            vals, idx = attn.topk(topk, dim=-1)
            return vals / (vals.sum(dim=-1, keepdim=True) + 1e-12), idx.int()
        t_topk = cuda_time(topk_fn)
        vals, idx = attn.topk(topk, dim=-1)
        vals = vals / (vals.sum(dim=-1, keepdim=True) + 1e-12)
        idx = idx.int()
    else:
        t_topk = 0.0

    # 4. V-Output (GQA, sparse or full)
    if topk > 0:
        t_vout = cuda_time(lambda: vsparse_2b_gqa(vals, idx, pv2, nv2, cb2, topk, N, D, GROUP))
        output_rot = vsparse_2b_gqa(vals, idx, pv2, nv2, cb2, topk, N, D, GROUP)
    else:
        t_vout = cuda_time(lambda: vfull_2b_gqa(attn, pv2, nv2, cb2, N, D, GROUP))
        output_rot = vfull_2b_gqa(attn, pv2, nv2, cb2, N, D, GROUP)

    # 5. Rotation
    t_rot = cuda_time(lambda: output_rot @ rot)

    # FP16 FlashAttention reference
    try:
        Qf = torch.randn(1, H_q, 1, D, device=device, dtype=torch.float16)
        Kf = torch.randn(1, H_q, N, D, device=device, dtype=torch.float16)
        Vf = torch.randn(1, H_q, N, D, device=device, dtype=torch.float16)
        t_flash = cuda_time(lambda: F.scaled_dot_product_attention(Qf, Kf, Vf))
        del Qf, Kf, Vf
    except RuntimeError:
        t_flash = float('nan')

    total_cda = t_kscore + t_softmax + t_topk + t_vout + t_rot

    print(f"  K-Score:      {t_kscore:.3f} ms ({t_kscore/total_cda*100:.1f}%)")
    print(f"  Softmax:      {t_softmax:.3f} ms ({t_softmax/total_cda*100:.1f}%)")
    if topk > 0:
        print(f"  TopK:         {t_topk:.3f} ms ({t_topk/total_cda*100:.1f}%)")
    print(f"  V-Output:     {t_vout:.3f} ms ({t_vout/total_cda*100:.1f}%)")
    print(f"  Rotation:     {t_rot:.3f} ms ({t_rot/total_cda*100:.1f}%)")
    print(f"  ---")
    print(f"  CDA total:    {total_cda:.3f} ms")
    print(f"  FlashAttn:    {t_flash:.3f} ms")
    print(f"  Speedup:      {t_flash/total_cda:.2f}x")

    results[str(N)] = {
        "N": N, "topk": topk,
        "kscore_ms": round(t_kscore, 3),
        "softmax_ms": round(t_softmax, 3),
        "topk_ms": round(t_topk, 3),
        "voutput_ms": round(t_vout, 3),
        "rotation_ms": round(t_rot, 3),
        "cda_total_ms": round(total_cda, 3),
        "flash_ms": round(t_flash, 3),
    }

    del Qr, pk4, nk4, pv2, nv2, scores, attn, output_rot
    gc.collect(); torch.cuda.empty_cache()

out_path = PROJECT_ROOT / "runs" / "latency_breakdown.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")
