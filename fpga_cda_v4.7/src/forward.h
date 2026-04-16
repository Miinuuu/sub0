#include "typedefs.h"
#include "config.h"
#include <math.h>
#include <cstring>
// 6-port merged KV cache (K4V2): indices + norms packed per buffer
static constexpr long long k_cache_indices_size = (long long)n_layers * seq_len * (kv_dim / 2);
static constexpr long long v_cache_indices_size = (long long)n_layers * seq_len * (kv_dim / 4);
static constexpr long long cache_norms_size = (long long)n_layers * seq_len * n_kv_heads;

extern "C" void forward(
    Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *transformer,
    int token, int pos,
    uint8_t kv_k_cache[k_cache_indices_size + cache_norms_size * 2],
    uint8_t kv_v_cache[v_cache_indices_size + cache_norms_size * 2],
    uint16_t *out,
    void *wq_flat,
    void *ws_flat);

// ---- Codebook LUT (2-bit Lloyd-Max for Beta(31.5, 31.5), dim=64) ----
// Values in [-1, 1] (unmapped from [0,1] codebook)
static const float CODEBOOK[4] = {-0.1330f, -0.0400f, 0.0400f, 0.1330f};

inline float codebook_lut(uint8_t idx) {
    return CODEBOOK[idx & 0x3];
}

// ---- Hadamard butterfly transform (in-place, head_size=64, 6 stages) ----
inline void hadamard_transform_64(float *buf) {
    #pragma HLS INLINE
    for (int stride = 32; stride >= 1; stride >>= 1) {
        for (int i = 0; i < 64; i += stride * 2) {
            #pragma HLS UNROLL
            for (int j = 0; j < stride; j++) {
                #pragma HLS UNROLL
                float a = buf[i + j];
                float b = buf[i + j + stride];
                buf[i + j]          = a + b;
                buf[i + j + stride] = a - b;
            }
        }
    }
    // Normalize by 1/sqrt(64) = 1/8
    for (int i = 0; i < 64; i++) {
        #pragma HLS UNROLL
        buf[i] *= 0.125f;
    }
}

// ---- 2-bit quantization: float[64] → packed uint8[16] + norm ----
inline void quantize_kv_2bit(const float *vec, uint8_t *packed, float *norm_out) {
    #pragma HLS INLINE
    // Compute L2 norm
    float norm_sq = 0.0f;
    for (int i = 0; i < 64; i++) {
        #pragma HLS UNROLL
        norm_sq += vec[i] * vec[i];
    }
    float norm = sqrtf(norm_sq);
    *norm_out = norm;

    if (norm < 1e-12f) {
        for (int i = 0; i < 16; i++) packed[i] = 0;
        return;
    }

    float inv_norm = 1.0f / norm;

    // Quantize each coordinate: unit vector → find nearest codebook entry
    for (int i = 0; i < 64; i += 4) {
        #pragma HLS UNROLL
        uint8_t byte = 0;
        for (int j = 0; j < 4; j++) {
            float val = vec[i + j] * inv_norm;  // normalize to unit
            // Map to [0,1]: (val + 1) / 2, then find nearest codebook
            float mapped = (val + 1.0f) * 0.5f;
            if (mapped < 0.0f) mapped = 0.0f;
            if (mapped > 1.0f) mapped = 1.0f;
            // Simple nearest: 4 boundaries for 4 levels
            uint8_t idx;
            if (mapped < 0.2585f) idx = 0;
            else if (mapped < 0.5f) idx = 1;
            else if (mapped < 0.7415f) idx = 2;
            else idx = 3;
            // Pack MSB first (match Python _gpu_pack_2bit order)
            byte |= (idx << (6 - j * 2));
        }
        packed[i / 4] = byte;
    }
}
template <int S>
void dequantize(QuantizedTensor<S> *qx, uint16_t x[S], int GS)
{
  for (int i = 0; i < S; i++)
  {
    float val = (float)qx->q[i] * half_to_float(qx->s[i / GS]);
    x[i] = float_to_half(val);
  }
}

template <int S>
void quantize(QuantizedTensor<S> *qx, float x[S], int GS)
{
  constexpr int num_groups = S / 64;
  constexpr float Q_MAX = 127.0f;
  uint16_t scale_buffer[num_groups];
  int8_t quantized_buffer[S];

  for (int group = 0; group < num_groups; group++)
  {
    float wmax = 0.0;
    int base_idx = group * GS;

    for (int i = 0; i < GS; i++)
    {
      float val = fabs(x[base_idx + i]);
      if (val > wmax) wmax = val;
    }

    float scale = wmax / Q_MAX;
    if (scale < 1e-12f) scale = 1e-12f;
    scale_buffer[group] = float_to_half(scale);

    for (int i = 0; i < GS; i++)
    {
      float quant_value = x[base_idx + i] / scale;
      int8_t quantized = (int8_t)round(quant_value);
      quantized_buffer[base_idx + i] = quantized;
    }
  }

  std::memcpy(qx->q, quantized_buffer, S * sizeof(int8_t));
  std::memcpy(qx->s, scale_buffer, num_groups * sizeof(uint16_t));
}
