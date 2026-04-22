"""Flash-LUT with warp-cooperative SMEM staging for K (Phase-1 of ncu fix).

Ncu on the original ``k_cda_flash_split_k4v2_gqa`` reported 82% uncoalesced
global-memory sectors in the score-phase K load (per-thread stride of 64B,
one token per lane). This kernel restructures the K load as a cooperative
STAGE-wide uint4 DRAM→SMEM pull: 256 threads × 16B = 4096B = STAGE×64B,
single coalesced burst per sub-tile.

Only the K load path is modified. Softmax and V-accumulate are structurally
identical to the original. V coalescing fix is deferred to Phase-2.
"""
from __future__ import annotations

from pathlib import Path
from torch.utils.cpp_extension import load_inline


_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define FD_NT 256
#define FD_NW 8
#define FD_VD 16
#define STAGE 64        // tokens per cooperative K load

__global__ void k_cda_flash_staged_k4v2_gqa(
    const float* __restrict__ Q,
    const uint8_t* __restrict__ pK,
    const float* __restrict__ nK,
    const uint8_t* __restrict__ pV,
    const float* __restrict__ nV,
    const float* __restrict__ cb_k,
    const float* __restrict__ cb_v,
    float* __restrict__ partial_out,
    float* __restrict__ m_vals,
    float* __restrict__ l_vals,
    const int H_q, const int N, const int D, const int group_size,
    const int num_splits, const int tile_N, const float scale)
{
    const int split = blockIdx.x;
    const int h_q   = blockIdx.y;
    const int kv_h  = h_q / group_size;
    const int tid   = threadIdx.x;

    const int s_start = split * tile_N;
    int s_end = s_start + tile_N; if (s_end > N) s_end = N;
    const int tile = s_end - s_start;
    if (tile <= 0) {
        if (tid == 0) {
            m_vals[h_q * num_splits + split] = -INFINITY;
            l_vals[h_q * num_splits + split] = 0.f;
        }
        if (tid < D) partial_out[(h_q * num_splits + split) * D + tid] = 0.f;
        return;
    }

    extern __shared__ float smem[];
    float   *sqcb_k   = smem;                          // D*16 floats
    float   *s_scores = smem + D * 16;                 // tile_N floats
    float   *warp_buf = s_scores + tile_N;             // 64 floats
    // STAGE rows of 17 uint32 (68 bytes/row, bank-safe) = STAGE*17*4 bytes
    uint32_t *k_stage = reinterpret_cast<uint32_t*>(warp_buf + 64);

    // 1. Pre-compute sqcb_k[d][c] = Q[d] · cb_k[c]
    for (int j = tid; j < D * 16; j += FD_NT) {
        int d = j >> 4;
        int c = j & 0xF;
        sqcb_k[j] = Q[h_q * D + d] * cb_k[c];
    }
    __syncthreads();

    // 2. Score pass with warp-coop staged K load
    const int kv_base = kv_h * N;
    const int ppk = D / 2;   // 64 bytes per token

    for (int sub = 0; sub < tile; sub += STAGE) {
        int sub_tile = tile - sub; if (sub_tile > STAGE) sub_tile = STAGE;

        // 2a. Cooperative DRAM→SMEM: 256 threads × 16B = 4096B = STAGE*64B
        //     Each thread loads 4 consecutive uint32 (= 16B) from DRAM,
        //     writes to k_stage[n_local][col..col+3]. Stride 17 avoids
        //     the 2-way bank conflict that would hit at stride 16.
        {
            int byte_off = tid * 16;
            int total_bytes = sub_tile * 64;
            if (byte_off < total_bytes) {
                int n_local = byte_off >> 6;                 // /64
                int col_u32 = (byte_off & 0x3F) >> 2;        // uint32 column in token (0..15)
                const uint4* src_ptr = reinterpret_cast<const uint4*>(
                    pK + (kv_base + s_start + sub + n_local) * ppk + (byte_off & 0x3F));
                uint4 data = *src_ptr;
                k_stage[n_local * 17 + col_u32 + 0] = data.x;
                k_stage[n_local * 17 + col_u32 + 1] = data.y;
                k_stage[n_local * 17 + col_u32 + 2] = data.z;
                k_stage[n_local * 17 + col_u32 + 3] = data.w;
            }
        }
        __syncthreads();

        // 2b. Score decode from SMEM (identical body to original, but SMEM reads)
        for (int n_local = tid; n_local < sub_tile; n_local += FD_NT) {
            float raw = 0.f;
            #pragma unroll
            for (int w = 0; w < 16; w++) {
                uint32_t wd = k_stage[n_local * 17 + w];
                int j = w * 8;
                uint8_t b0 = wd & 0xFF;
                uint8_t b1 = (wd >> 8) & 0xFF;
                uint8_t b2 = (wd >> 16) & 0xFF;
                uint8_t b3 = (wd >> 24) & 0xFF;
                raw += sqcb_k[(j+0) * 16 + ((b0 >> 4) & 0xF)]
                     + sqcb_k[(j+1) * 16 + ( b0       & 0xF)]
                     + sqcb_k[(j+2) * 16 + ((b1 >> 4) & 0xF)]
                     + sqcb_k[(j+3) * 16 + ( b1       & 0xF)]
                     + sqcb_k[(j+4) * 16 + ((b2 >> 4) & 0xF)]
                     + sqcb_k[(j+5) * 16 + ( b2       & 0xF)]
                     + sqcb_k[(j+6) * 16 + ((b3 >> 4) & 0xF)]
                     + sqcb_k[(j+7) * 16 + ( b3       & 0xF)];
            }
            s_scores[sub + n_local] = raw * nK[kv_base + s_start + sub + n_local] * scale;
        }
        __syncthreads();
    }

    // 3. Block-wide max over s_scores[0..tile)
    float local_max = -INFINITY;
    for (int n_local = tid; n_local < tile; n_local += FD_NT) {
        float v = s_scores[n_local];
        if (v > local_max) local_max = v;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float o = __shfl_down_sync(0xFFFFFFFF, local_max, off);
        if (o > local_max) local_max = o;
    }
    if ((tid & 31) == 0) warp_buf[tid >> 5] = local_max;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (FD_NT / 32)) ? warp_buf[tid] : -INFINITY;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float o = __shfl_down_sync(0xFFFFFFFF, v, off);
            if (o > v) v = o;
        }
        if (tid == 0) warp_buf[0] = v;
    }
    __syncthreads();
    const float tile_m = warp_buf[0];

    // 4. Exp + sum
    float local_sum = 0.f;
    for (int n_local = tid; n_local < tile; n_local += FD_NT) {
        float e = __expf(s_scores[n_local] - tile_m);
        s_scores[n_local] = e;
        local_sum += e;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, off);
    if ((tid & 31) == 0) warp_buf[tid >> 5] = local_sum;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (FD_NT / 32)) ? warp_buf[tid] : 0.f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xFFFFFFFF, v, off);
        if (tid == 0) warp_buf[1] = v;
    }
    __syncthreads();
    const float tile_l = warp_buf[1];

    // 5. V accumulate — UNCHANGED from original (Phase-2 target for V-coalesce fix)
    const float c0 = cb_v[0], c1 = cb_v[1], c2 = cb_v[2], c3 = cb_v[3];
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int d_base  = warp_id * FD_VD;
    const int ppv     = D / 4;
    const int byte_off_warp = d_base >> 2;

    float ac[FD_VD];
    #pragma unroll
    for (int d = 0; d < FD_VD; d++) ac[d] = 0.f;

    #define CV(i) ((i)==0 ? w0 : ((i)==1 ? w1 : ((i)==2 ? w2 : w3)))
    for (int n_local = lane; n_local < tile; n_local += 32) {
        int n = s_start + n_local;
        float w  = s_scores[n_local] * nV[kv_base + n];
        float w0 = w * c0, w1 = w * c1, w2 = w * c2, w3 = w * c3;
        uint32_t wd = *reinterpret_cast<const uint32_t*>(pV + (kv_base + n) * ppv + byte_off_warp);
        uint8_t y0 = wd & 0xFF, y1 = (wd >> 8) & 0xFF, y2 = (wd >> 16) & 0xFF, y3 = (wd >> 24) & 0xFF;
        ac[ 0] += CV((y0 >> 6) & 3); ac[ 1] += CV((y0 >> 4) & 3);
        ac[ 2] += CV((y0 >> 2) & 3); ac[ 3] += CV( y0       & 3);
        ac[ 4] += CV((y1 >> 6) & 3); ac[ 5] += CV((y1 >> 4) & 3);
        ac[ 6] += CV((y1 >> 2) & 3); ac[ 7] += CV( y1       & 3);
        ac[ 8] += CV((y2 >> 6) & 3); ac[ 9] += CV((y2 >> 4) & 3);
        ac[10] += CV((y2 >> 2) & 3); ac[11] += CV( y2       & 3);
        ac[12] += CV((y3 >> 6) & 3); ac[13] += CV((y3 >> 4) & 3);
        ac[14] += CV((y3 >> 2) & 3); ac[15] += CV( y3       & 3);
    }
    #undef CV

    #pragma unroll
    for (int d = 0; d < FD_VD; d++) {
        float v = ac[d];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xFFFFFFFF, v, off);
        if (lane == 0) {
            partial_out[(h_q * num_splits + split) * D + d_base + d] = v;
        }
    }

    if (tid == 0) {
        m_vals[h_q * num_splits + split] = tile_m;
        l_vals[h_q * num_splits + split] = tile_l;
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> cda_flash_staged_k4v2_gqa(
    torch::Tensor Q, torch::Tensor pK, torch::Tensor nK,
    torch::Tensor pV, torch::Tensor nV,
    torch::Tensor cb_k, torch::Tensor cb_v,
    int N, int group_size, int tile_N, float scale)
{
    const int H_q = Q.size(0);
    const int D   = Q.size(1);
    const int num_splits = (N + tile_N - 1) / tile_N;
    TORCH_CHECK(D == 128, "staged kernel requires D == 128");
    TORCH_CHECK(tile_N % STAGE == 0 || tile_N < STAGE, "tile_N should be multiple of STAGE=64");

    auto opts_f = Q.options();
    auto partial = torch::zeros({H_q, num_splits, D}, opts_f);
    auto m_vals  = torch::full({H_q, num_splits}, -std::numeric_limits<float>::infinity(), opts_f);
    auto l_vals  = torch::zeros({H_q, num_splits}, opts_f);

    // smem layout: sqcb_k (D*16) + s_scores (tile_N) + warp_buf (64) + k_stage (STAGE*17)
    const size_t smem_bytes = (size_t)(D * 16 + tile_N + 64 + STAGE * 17) * sizeof(float);

    k_cda_flash_staged_k4v2_gqa<<<dim3(num_splits, H_q), FD_NT, smem_bytes>>>(
        Q.data_ptr<float>(), pK.data_ptr<uint8_t>(), nK.data_ptr<float>(),
        pV.data_ptr<uint8_t>(), nV.data_ptr<float>(),
        cb_k.data_ptr<float>(), cb_v.data_ptr<float>(),
        partial.data_ptr<float>(), m_vals.data_ptr<float>(), l_vals.data_ptr<float>(),
        H_q, N, D, group_size, num_splits, tile_N, scale);

    return std::make_tuple(partial, m_vals, l_vals);
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> cda_flash_staged_k4v2_gqa(
    torch::Tensor Q, torch::Tensor pK, torch::Tensor nK,
    torch::Tensor pV, torch::Tensor nV,
    torch::Tensor cb_k, torch::Tensor cb_v,
    int N, int group_size, int tile_N, float scale);
"""

_mod = None


def _get_staged_mod():
    global _mod
    if _mod is not None:
        return _mod
    _mod = load_inline(
        name="_cda_lut_staged",
        cpp_sources=_CPP_SRC,
        cuda_sources=_CUDA_SRC,
        functions=["cda_flash_staged_k4v2_gqa"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        verbose=False,
    )
    return _mod


def cda_flash_staged_k4v2_gqa(Q, pK, nK, pV, nV, cb_k, cb_v,
                               N, group_size, tile_N=512, scale=None):
    """Warp-cooperative staged Flash-LUT kernel (Phase-1 optimization).

    Same I/O as :func:`core.cda_attn.cda_flash_split_k4v2_gqa` — fix is internal.
    """
    if scale is None:
        scale = 1.0 / (Q.size(-1) ** 0.5)
    return _get_staged_mod().cda_flash_staged_k4v2_gqa(
        Q, pK, nK, pV, nV, cb_k, cb_v, N, group_size, tile_N, scale)
