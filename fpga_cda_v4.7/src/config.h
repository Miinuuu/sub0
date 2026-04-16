#pragma once
#include "typedefs.h"

static constexpr int dim = 768;
static constexpr int hidden_dim = 2048;
static constexpr int n_layers = 12;
static constexpr int n_heads = 12;
static constexpr int n_kv_heads = 12;
static constexpr int vocab_size = 32000;
#ifdef SEQ_LEN_OVERRIDE
static constexpr int seq_len = SEQ_LEN_OVERRIDE;
#else
static constexpr int seq_len = 1024;
#endif
static constexpr int GS = 64;

// Compressed KV cache parameters
static constexpr int KV_BITS = 2;                           // 2-bit quantization
static constexpr int KV_LEVELS = 1 << KV_BITS;             // 4 codebook entries
static constexpr int head_size = dim / n_heads;             // 64
static constexpr int kv_dim = (dim * n_kv_heads) / n_heads; // 768
static constexpr int packed_head_size = head_size / 4;      // 16 bytes per head per token

constexpr Config config = {
    .dim = dim,
    .hidden_dim = hidden_dim,
    .n_layers = n_layers,
    .n_heads = n_heads,
    .n_kv_heads = n_kv_heads,
    .vocab_size = vocab_size,
    .seq_len = seq_len,
    .GS = GS,
};

// ---- Flat weight layout for burst_maxi ----
static constexpr int kv_dim_c2 = (dim * n_kv_heads) / n_heads;
static constexpr int LAYER_Q_SIZE = dim*dim + dim*kv_dim_c2 + dim*kv_dim_c2 + dim*dim
                                  + dim*hidden_dim + dim*hidden_dim + hidden_dim*dim;
static constexpr int CLS_Q_SIZE = dim * vocab_size;
static constexpr int WQ_FLAT_SIZE = LAYER_Q_SIZE * n_layers + CLS_Q_SIZE;

static constexpr int LAYER_S_SIZE = LAYER_Q_SIZE / GS;
static constexpr int CLS_S_SIZE = CLS_Q_SIZE / GS;
static constexpr int WS_FLAT_SIZE = LAYER_S_SIZE * n_layers + CLS_S_SIZE;
