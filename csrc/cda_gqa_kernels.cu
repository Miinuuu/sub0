// GQA-aware compressed-domain attention kernels for sub0.
//
// Kernel grid: blockIdx.y = query-head index, blockIdx.x = token block.
// KV for query head q is read from kv_head = q / group_size, so the GQA
// expansion that `cda.cuda_attention.cuda_hw_attention_batched` would do
// via `torch.repeat_interleave` is collapsed into a single index op inside
// the kernel. This is what lets per-step decode latency track the paper's
// Figure 5(c) numbers (Llama-3.1-8B, A6000, N=32768, K4/V2 = 35.1 ms).
//
// Exposed via `PYBIND11_MODULE(_cda_gqa_kernels, ...)`; built by
// `csrc/setup_gqa.py` as a side-by-side extension to the main
// ``cda._cda_kernels`` binary. Never exposed directly; see
// ``cda.cuda_attention_gqa`` for the Python wrapper.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// =====================================================================
// 2-bit K score (GQA): shared Q·codebook, per-byte 4×2-bit unpack.
// =====================================================================
__global__ void score_2b_gqa_kernel(
    const float* __restrict__ Q,    // (H_q, D)
    const uint8_t* __restrict__ pk, // (H_kv, N, D/4)
    const float* __restrict__ nm,   // (H_kv, N)
    const float* __restrict__ cb,   // (4,)
    float* __restrict__ sc,         // (H_q, N)
    const int H_q, const int N, const int D, const int group_size, const float scale)
{
    __shared__ float sqcb[128][4];
    const int b = blockIdx.y;
    const int kv_h = b / group_size;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int ppv = D / 4;

    float c0 = cb[0], c1 = cb[1], c2 = cb[2], c3 = cb[3];
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float q = Q[b * D + j];
        sqcb[j][0] = q * c0; sqcb[j][1] = q * c1;
        sqcb[j][2] = q * c2; sqcb[j][3] = q * c3;
    }
    __syncthreads();
    if (n >= N) return;

    float a = 0.f;
    const uint32_t* p = reinterpret_cast<const uint32_t*>(pk + (kv_h * N + n) * ppv);
    #pragma unroll
    for (int w = 0; w < D / 16; w++) {
        uint32_t wd = p[w]; int j = w * 16;
        uint8_t b0 = wd & 0xFF, b1 = (wd >> 8) & 0xFF,
                b2 = (wd >> 16) & 0xFF, b3 = (wd >> 24) & 0xFF;
        a += sqcb[j][(b0 >> 6) & 3] + sqcb[j + 1][(b0 >> 4) & 3]
           + sqcb[j + 2][(b0 >> 2) & 3] + sqcb[j + 3][b0 & 3]
           + sqcb[j + 4][(b1 >> 6) & 3] + sqcb[j + 5][(b1 >> 4) & 3]
           + sqcb[j + 6][(b1 >> 2) & 3] + sqcb[j + 7][b1 & 3]
           + sqcb[j + 8][(b2 >> 6) & 3] + sqcb[j + 9][(b2 >> 4) & 3]
           + sqcb[j + 10][(b2 >> 2) & 3] + sqcb[j + 11][b2 & 3]
           + sqcb[j + 12][(b3 >> 6) & 3] + sqcb[j + 13][(b3 >> 4) & 3]
           + sqcb[j + 14][(b3 >> 2) & 3] + sqcb[j + 15][b3 & 3];
    }
    sc[b * N + n] = a * nm[kv_h * N + n] * scale;
}

// =====================================================================
// 4-bit K score (GQA): 16-entry codebook, per-byte 2×4-bit unpack.
// =====================================================================
__global__ void score_4b_gqa_kernel(
    const float* __restrict__ Q,    // (H_q, D)
    const uint8_t* __restrict__ pk, // (H_kv, N, D/2)
    const float* __restrict__ nm,   // (H_kv, N)
    const float* __restrict__ cb,   // (16,)
    float* __restrict__ sc,         // (H_q, N)
    const int H_q, const int N, const int D, const int group_size, const float scale)
{
    __shared__ float sqcb[128][16];
    const int b = blockIdx.y;
    const int kv_h = b / group_size;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int ppv = D / 2;

    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float q = Q[b * D + j];
        for (int k = 0; k < 16; k++) sqcb[j][k] = q * cb[k];
    }
    __syncthreads();
    if (n >= N) return;

    float a = 0.f;
    const uint32_t* p = reinterpret_cast<const uint32_t*>(pk + (kv_h * N + n) * ppv);
    #pragma unroll
    for (int w = 0; w < D / 8; w++) {
        uint32_t wd = p[w]; int j = w * 8;
        uint8_t b0 = wd & 0xFF, b1 = (wd >> 8) & 0xFF,
                b2 = (wd >> 16) & 0xFF, b3 = (wd >> 24) & 0xFF;
        a += sqcb[j][(b0 >> 4) & 0xF] + sqcb[j + 1][b0 & 0xF]
           + sqcb[j + 2][(b1 >> 4) & 0xF] + sqcb[j + 3][b1 & 0xF]
           + sqcb[j + 4][(b2 >> 4) & 0xF] + sqcb[j + 5][b2 & 0xF]
           + sqcb[j + 6][(b3 >> 4) & 0xF] + sqcb[j + 7][b3 & 0xF];
    }
    sc[b * N + n] = a * nm[kv_h * N + n] * scale;
}

// =====================================================================
// 2-bit V output (GQA, full): softmax(scores) · V over all N tokens.
// =====================================================================
#define VD 32
#define VB 256
__device__ __forceinline__ float warp_reduce(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xFFFFFFFF, v, o);
    return v;
}

__global__ void v_2b_full_gqa_kernel(
    const float* __restrict__ at,    // (H_q, N)
    const uint8_t* __restrict__ pk,  // (H_kv, N, D/4)
    const float* __restrict__ nm,    // (H_kv, N)
    const float* __restrict__ cb,    // (4,)
    float* __restrict__ out,         // (H_q, D)
    const int H_q, const int N, const int D, const int group_size)
{
    const float c0 = cb[0], c1 = cb[1], c2 = cb[2], c3 = cb[3];
    __shared__ float sw[VD][VB / 32];
    const int b = blockIdx.y;
    const int kv_h = b / group_size;
    const int dg = blockIdx.x, tid = threadIdx.x;
    const int wid = tid / 32, lid = tid % 32, jb = dg * VD;
    if (jb >= D) return;
    const int ppv = D / 4;

    float ac[VD];
    for (int d = 0; d < VD; d++) ac[d] = 0.f;
#define CV(i) ((i) == 0 ? w0 : ((i) == 1 ? w1 : ((i) == 2 ? w2 : w3)))
    for (int n = tid; n < N; n += VB) {
        float w = at[b * N + n] * nm[kv_h * N + n];
        float w0 = w * c0, w1 = w * c1, w2 = w * c2, w3 = w * c3;
        const uint32_t* r = reinterpret_cast<const uint32_t*>(pk + (kv_h * N + n) * ppv + jb / 4);
        #pragma unroll
        for (int wi = 0; wi < VD / 16; wi++) {
            uint32_t wd = r[wi]; int do_ = wi * 16;
            uint8_t y0 = wd & 0xFF, y1 = (wd >> 8) & 0xFF,
                    y2 = (wd >> 16) & 0xFF, y3 = (wd >> 24) & 0xFF;
            ac[do_    ] += CV((y0 >> 6) & 3); ac[do_ + 1] += CV((y0 >> 4) & 3);
            ac[do_ + 2] += CV((y0 >> 2) & 3); ac[do_ + 3] += CV(y0 & 3);
            ac[do_ + 4] += CV((y1 >> 6) & 3); ac[do_ + 5] += CV((y1 >> 4) & 3);
            ac[do_ + 6] += CV((y1 >> 2) & 3); ac[do_ + 7] += CV(y1 & 3);
            ac[do_ + 8] += CV((y2 >> 6) & 3); ac[do_ + 9] += CV((y2 >> 4) & 3);
            ac[do_ + 10] += CV((y2 >> 2) & 3); ac[do_ + 11] += CV(y2 & 3);
            ac[do_ + 12] += CV((y3 >> 6) & 3); ac[do_ + 13] += CV((y3 >> 4) & 3);
            ac[do_ + 14] += CV((y3 >> 2) & 3); ac[do_ + 15] += CV(y3 & 3);
        }
    }
#undef CV
    for (int d = 0; d < VD; d++) {
        float v = warp_reduce(ac[d]);
        if (lid == 0) sw[d][wid] = v;
    }
    __syncthreads();
    if (wid == 0) {
        for (int d = 0; d < VD; d++) {
            float v = (lid < VB / 32) ? sw[d][lid] : 0.f;
            v = warp_reduce(v);
            if (lid == 0 && (jb + d) < D) out[b * D + jb + d] = v;
        }
    }
}

// =====================================================================
// 2-bit V output (GQA, TopK sparse).
// =====================================================================
#define VSB 128
__global__ void v_2b_sparse_gqa_kernel(
    const float* __restrict__ atk,   // (H_q, K) normalized TopK weights
    const int* __restrict__ idx,     // (H_q, K) TopK indices into N
    const uint8_t* __restrict__ pk,  // (H_kv, N, D/4)
    const float* __restrict__ nm,    // (H_kv, N)
    const float* __restrict__ cb,    // (4,)
    float* __restrict__ out,         // (H_q, D)
    const int H_q, const int K, const int N, const int D, const int group_size)
{
    const float c0 = cb[0], c1 = cb[1], c2 = cb[2], c3 = cb[3];
    __shared__ float sw[VD][VSB / 32];
    const int b = blockIdx.y;
    const int kv_h = b / group_size;
    const int dg = blockIdx.x, tid = threadIdx.x;
    const int wid = tid / 32, lid = tid % 32, jb = dg * VD;
    if (jb >= D) return;
    const int ppv = D / 4;

    float ac[VD];
    for (int d = 0; d < VD; d++) ac[d] = 0.f;
#define CV(i) ((i) == 0 ? w0 : ((i) == 1 ? w1 : ((i) == 2 ? w2 : w3)))
    for (int ki = tid; ki < K; ki += VSB) {
        int n = idx[b * K + ki];
        float w = atk[b * K + ki] * nm[kv_h * N + n];
        float w0 = w * c0, w1 = w * c1, w2 = w * c2, w3 = w * c3;
        const uint32_t* r = reinterpret_cast<const uint32_t*>(pk + (kv_h * N + n) * ppv + jb / 4);
        #pragma unroll
        for (int wi = 0; wi < VD / 16; wi++) {
            uint32_t wd = r[wi]; int do_ = wi * 16;
            uint8_t y0 = wd & 0xFF, y1 = (wd >> 8) & 0xFF,
                    y2 = (wd >> 16) & 0xFF, y3 = (wd >> 24) & 0xFF;
            ac[do_    ] += CV((y0 >> 6) & 3); ac[do_ + 1] += CV((y0 >> 4) & 3);
            ac[do_ + 2] += CV((y0 >> 2) & 3); ac[do_ + 3] += CV(y0 & 3);
            ac[do_ + 4] += CV((y1 >> 6) & 3); ac[do_ + 5] += CV((y1 >> 4) & 3);
            ac[do_ + 6] += CV((y1 >> 2) & 3); ac[do_ + 7] += CV(y1 & 3);
            ac[do_ + 8] += CV((y2 >> 6) & 3); ac[do_ + 9] += CV((y2 >> 4) & 3);
            ac[do_ + 10] += CV((y2 >> 2) & 3); ac[do_ + 11] += CV(y2 & 3);
            ac[do_ + 12] += CV((y3 >> 6) & 3); ac[do_ + 13] += CV((y3 >> 4) & 3);
            ac[do_ + 14] += CV((y3 >> 2) & 3); ac[do_ + 15] += CV(y3 & 3);
        }
    }
#undef CV
    for (int d = 0; d < VD; d++) {
        float v = warp_reduce(ac[d]);
        if (lid == 0) sw[d][wid] = v;
    }
    __syncthreads();
    if (wid == 0) {
        for (int d = 0; d < VD; d++) {
            float v = (lid < VSB / 32) ? sw[d][lid] : 0.f;
            v = warp_reduce(v);
            if (lid == 0 && (jb + d) < D) out[b * D + jb + d] = v;
        }
    }
}

// =====================================================================
// C++ wrappers (torch::Tensor entry points)
// =====================================================================
torch::Tensor score_2b_gqa(torch::Tensor Q, torch::Tensor pk, torch::Tensor nm,
                            torch::Tensor cb, int64_t N, int64_t group_size, double scale)
{
    int H_q = Q.size(0), D = Q.size(1);
    auto sc = torch::empty({H_q, N}, Q.options());
    score_2b_gqa_kernel<<<dim3((N + 255) / 256, H_q), 256>>>(
        Q.data_ptr<float>(), pk.data_ptr<uint8_t>(), nm.data_ptr<float>(),
        cb.data_ptr<float>(), sc.data_ptr<float>(),
        H_q, (int)N, D, (int)group_size, (float)scale);
    return sc;
}

torch::Tensor score_4b_gqa(torch::Tensor Q, torch::Tensor pk, torch::Tensor nm,
                            torch::Tensor cb, int64_t N, int64_t group_size, double scale)
{
    int H_q = Q.size(0), D = Q.size(1);
    auto sc = torch::empty({H_q, N}, Q.options());
    score_4b_gqa_kernel<<<dim3((N + 255) / 256, H_q), 256>>>(
        Q.data_ptr<float>(), pk.data_ptr<uint8_t>(), nm.data_ptr<float>(),
        cb.data_ptr<float>(), sc.data_ptr<float>(),
        H_q, (int)N, D, (int)group_size, (float)scale);
    return sc;
}

torch::Tensor vfull_2b_gqa(torch::Tensor a, torch::Tensor pk, torch::Tensor nm,
                            torch::Tensor cb, int64_t N, int64_t D, int64_t group_size)
{
    int H_q = a.size(0);
    auto o = torch::zeros({H_q, D}, a.options());
    v_2b_full_gqa_kernel<<<dim3(((int)D + VD - 1) / VD, H_q), VB>>>(
        a.data_ptr<float>(), pk.data_ptr<uint8_t>(), nm.data_ptr<float>(),
        cb.data_ptr<float>(), o.data_ptr<float>(),
        H_q, (int)N, (int)D, (int)group_size);
    return o;
}

torch::Tensor vsparse_2b_gqa(torch::Tensor atk, torch::Tensor idx, torch::Tensor pk,
                              torch::Tensor nm, torch::Tensor cb,
                              int64_t K, int64_t N, int64_t D, int64_t group_size)
{
    int H_q = atk.size(0);
    auto o = torch::zeros({H_q, D}, atk.options());
    v_2b_sparse_gqa_kernel<<<dim3(((int)D + VD - 1) / VD, H_q), VSB>>>(
        atk.data_ptr<float>(), idx.data_ptr<int>(), pk.data_ptr<uint8_t>(),
        nm.data_ptr<float>(), cb.data_ptr<float>(), o.data_ptr<float>(),
        H_q, (int)K, (int)N, (int)D, (int)group_size);
    return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("score_2b_gqa", &score_2b_gqa,
          "GQA 2-bit K score (Q rotated, per-KV-head packed K)");
    m.def("score_4b_gqa", &score_4b_gqa,
          "GQA 4-bit K score");
    m.def("vfull_2b_gqa", &vfull_2b_gqa,
          "GQA 2-bit V output (dense over all N)");
    m.def("vsparse_2b_gqa", &vsparse_2b_gqa,
          "GQA 2-bit V output (TopK sparse)");
}
