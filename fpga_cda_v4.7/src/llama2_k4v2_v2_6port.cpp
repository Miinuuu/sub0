/* Inference for Llama-2 Transformer model in pure C, int8 quantized forward pass. */
// Host-side fp16 support
#ifdef __SYNTHESIS__
#include "hls_half.h"
typedef half act_t;
#else
#include <cstdint>
#include <cstring>
typedef uint16_t act_t;
static inline float half_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t expo = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FFu;
    if (expo == 0) { if (mant == 0) { uint32_t r = sign; float f; memcpy(&f,&r,4); return f; }
        while (!(mant & 0x400)) { mant <<= 1; expo--; } expo++; mant &= ~0x400u; }
    else if (expo == 31) { uint32_t r = sign | 0x7F800000u | (mant << 13); float f; memcpy(&f,&r,4); return f; }
    uint32_t r = sign | ((expo + 112) << 23) | (mant << 13);
    float f; memcpy(&f, &r, 4); return f;
}
static inline uint16_t float_to_half(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int expo = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (expo <= 0) return sign;
    if (expo >= 31) return sign | 0x7C00;
    return sign | (expo << 10) | (mant >> 13);
}
#endif
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string>
#include <iostream>
#include <cstring>
#include <fcntl.h>
#include "typedefs.h"
#include "forward.h"

// ---- Flatten weights for burst_maxi (original layout, no padding) ----
#include "config.h"
void flatten_weights(void *transformer_raw, int8_t *wq_flat, uint16_t *ws_flat) {
    auto *tr = (Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *)transformer_raw;
    auto *w = &tr->weights;
    constexpr int kv_dim = (dim * n_kv_heads) / n_heads;
    int qoff = 0, soff = 0;
    for (int l = 0; l < n_layers; l++) {
        memcpy(wq_flat+qoff, (w->wq+l)->q, dim*dim); qoff+=dim*dim;
        memcpy(ws_flat+soff, (w->wq+l)->s, dim*dim/GS*2); soff+=dim*dim/GS;
        memcpy(wq_flat+qoff, (w->wk+l)->q, dim*kv_dim); qoff+=dim*kv_dim;
        memcpy(ws_flat+soff, (w->wk+l)->s, dim*kv_dim/GS*2); soff+=dim*kv_dim/GS;
        memcpy(wq_flat+qoff, (w->wv+l)->q, dim*kv_dim); qoff+=dim*kv_dim;
        memcpy(ws_flat+soff, (w->wv+l)->s, dim*kv_dim/GS*2); soff+=dim*kv_dim/GS;
        memcpy(wq_flat+qoff, (w->wo+l)->q, dim*dim); qoff+=dim*dim;
        memcpy(ws_flat+soff, (w->wo+l)->s, dim*dim/GS*2); soff+=dim*dim/GS;
        memcpy(wq_flat+qoff, (w->w1+l)->q, dim*hidden_dim); qoff+=dim*hidden_dim;
        memcpy(ws_flat+soff, (w->w1+l)->s, dim*hidden_dim/GS*2); soff+=dim*hidden_dim/GS;
        memcpy(wq_flat+qoff, (w->w3+l)->q, dim*hidden_dim); qoff+=dim*hidden_dim;
        memcpy(ws_flat+soff, (w->w3+l)->s, dim*hidden_dim/GS*2); soff+=dim*hidden_dim/GS;
        memcpy(wq_flat+qoff, (w->w2+l)->q, hidden_dim*dim); qoff+=hidden_dim*dim;
        memcpy(ws_flat+soff, (w->w2+l)->s, hidden_dim*dim/GS*2); soff+=hidden_dim*dim/GS;
    }
    memcpy(wq_flat+qoff, w->wcls->q, dim*vocab_size);
    memcpy(ws_flat+soff, w->wcls->s, dim*vocab_size/GS*2);
    printf("Flattened: q=%d bytes, s=%d entries\n", qoff+dim*vocab_size, soff+dim*vocab_size/GS);
}

#include "config.h"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#if defined _WIN32
#include "win.h"
#else
#include <unistd.h>
#include <sys/mman.h>
#endif
// ----------------------------------------------------------------------------
// Globals

// void malloc_run_state(RunState *s, Config *p)
// {
//   // we calloc instead of malloc to keep valgrind happy
//   int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
//   s->x = (float *)calloc(p->dim, sizeof(float));
//   s->xb = (float *)calloc(p->dim, sizeof(float));
//   s->xb2 = (float *)calloc(p->dim, sizeof(float));
//   s->hb = (float *)calloc(p->hidden_dim, sizeof(float));
//   s->hb2 = (float *)calloc(p->hidden_dim, sizeof(float));
//   s->xq = (QuantizedTensor){.q = (int8_t *)calloc(p->dim, sizeof(int8_t)), .s = (float *)calloc(p->dim, sizeof(float))};
//   s->hq = (QuantizedTensor){.q = (int8_t *)calloc(p->hidden_dim, sizeof(int8_t)), .s = (float *)calloc(p->hidden_dim, sizeof(float))};
//   s->q = (float *)calloc(p->dim, sizeof(float));
//   s->k = (float *)calloc(kv_dim, sizeof(float));
//   s->v = (float *)calloc(kv_dim, sizeof(float));
//   s->att = (float *)calloc(p->n_heads * p->seq_len, sizeof(float));
//   s->logits = (float *)calloc(p->vocab_size, sizeof(float));
//   s->key_cache = (float *)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
//   s->value_cache = (float *)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
//   // ensure all mallocs went fine
//   if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->k || !s->v || !s->att || !s->logits || !s->key_cache || !s->value_cache)
//   {
//     fprintf(stderr, "malloc failed!\n");
//     exit(EXIT_FAILURE);
//   }
// }

void softmax(float *x, int size)
{
  // find max value (for numerical stability)
  float max_val = x[0];
  for (int i = 1; i < size; i++)
  {
    if (x[i] > max_val)
    {
      max_val = x[i];
    }
  }
  // exp and sum
  float sum = 0.0f;
  for (int i = 0; i < size; i++)
  {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  // normalize
  for (int i = 0; i < size; i++)
  {
    x[i] /= sum;
  }
}

template <int SIZE>
/* initialize `n` x quantized tensor (with `size_each` elements), starting from memory pointed at *ptr */
void init_quantized_tensors(void **ptr, QuantizedTensor<SIZE> *tensor, int n, int size_each)
{
  void *p = *ptr;
  for (int i = 0; i < n; i++)
  {
    /* map quantized int8 values*/
    std::memcpy(tensor[i].q, p, size_each * sizeof(int8_t));
    p = (int8_t *)p + size_each;
    /* map scale factors */
    { int nscales = size_each / GS; float *fp = (float*)p; for(int si=0;si<nscales;si++) tensor[i].s[si]=float_to_half(fp[si]); }

    p = (float *)p + size_each / GS;
  }
  *ptr = p; // advance ptr to current position
}

template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
void memory_map_weights(TransformerWeights<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *w, void *ptr, uint8_t shared_classifier)
{
  int head_size = dim / n_heads;
  // first are the parameters that are kept in fp32 (the rmsnorm (1D) weights)
  float *fptr = (float *)ptr; // cast our pointer to float*
  { for(int i=0;i<n_layers*dim;i++) w->rms_att_weight[i]=float_to_half(fptr[i]); }
  fptr += n_layers * dim;
  { for(int i=0;i<n_layers*dim;i++) w->rms_ffn_weight[i]=float_to_half(fptr[i]); }
  fptr += n_layers * dim;
  { for(int i=0;i<dim;i++) w->rms_final_weight[i]=float_to_half(fptr[i]); }
  fptr += dim;

  // now read all the quantized weights
  ptr = (void *)fptr; // now cast the pointer back to void*
  init_quantized_tensors(&ptr, w->q_tokens, 1, vocab_size * dim);
  // dequantize token embedding table
  dequantize<vocab_size * dim>(w->q_tokens, w->token_embedding_table, GS);

  init_quantized_tensors(&ptr, w->wq, n_layers, dim * (n_heads * head_size));
  init_quantized_tensors(&ptr, w->wk, n_layers, dim * (n_kv_heads * head_size));
  init_quantized_tensors(&ptr, w->wv, n_layers, dim * (n_kv_heads * head_size));
  init_quantized_tensors(&ptr, w->wo, n_layers, (n_heads * head_size) * dim);

  init_quantized_tensors(&ptr, w->w1, n_layers, dim * hidden_dim);
  init_quantized_tensors(&ptr, w->w2, n_layers, hidden_dim * dim);
  init_quantized_tensors(&ptr, w->w3, n_layers, dim * hidden_dim);

  if (shared_classifier)
  {
    std::memcpy(w->wcls, w->q_tokens, sizeof(QuantizedTensor<vocab_size * dim>));
  }
  else
  {
    init_quantized_tensors(&ptr, w->wcls, 1, dim * vocab_size);
  }
}

template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
void read_checkpoint(std::string checkpoint, Config *config, TransformerWeights<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *weights)
{
  FILE *file = fopen(checkpoint.c_str(), "rb");
  if (!file)
  {
    fprintf(stderr, "Couldn't open file %s\n", checkpoint.c_str());
    exit(EXIT_FAILURE);
  }
  // read in magic number (uint32), has to be 0x616b3432, i.e. "ak42" in ASCII
  uint32_t magic_number;
  if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1)
  {
    exit(EXIT_FAILURE);
  }
  if (magic_number != 0x616b3432)
  {
    fprintf(stderr, "Bad magic number\n");
    exit(EXIT_FAILURE);
  }
  // read in the version number (uint32), has to be 1
  int version;
  if (fread(&version, sizeof(int), 1, file) != 1)
  {
    exit(EXIT_FAILURE);
  }
  if (version != 2)
  {
    fprintf(stderr, "Bad version %d, need version 2\n", version);
    exit(EXIT_FAILURE);
  }
  int header_size = 256; // the header size for version 2 in bytes
  // read in the Config
  if (fread(config, sizeof(Config) - sizeof(int), 1, file) != 1)
  {
    exit(EXIT_FAILURE);
  }
  // read in flags
  uint8_t shared_classifier; // a byte to indicate if the classifier is shared
  if (fread(&shared_classifier, sizeof(uint8_t), 1, file) != 1)
  {
    exit(EXIT_FAILURE);
  }
  int group_size; // the group size used in quantization
  if (fread(&group_size, sizeof(int), 1, file) != 1)
  {
    exit(EXIT_FAILURE);
  }
  config->GS = GS;
  // figure out the file size
  fseek(file, 0, SEEK_END);     // move file pointer to end of file
  auto file_size = ftell(file); // get the file size, in bytes
  fclose(file);
  // memory map the Transformer weights into the data pointer
  auto fd = open(checkpoint.c_str(), O_RDONLY); // open in read only mode
  if (fd == -1)
  {
    fprintf(stderr, "open failed!\n");
    exit(EXIT_FAILURE);
  }
  auto data = (float *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (data == MAP_FAILED)
  {
    fprintf(stderr, "mmap failed!\n");
    exit(EXIT_FAILURE);
  }
  void *weights_ptr = ((char *)data) + header_size; // skip header bytes. char is 1 byte
  memory_map_weights(weights, weights_ptr, shared_classifier);
  close(fd);
  if (data != MAP_FAILED)
  {
    munmap(data, file_size);
  }
}

template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
void build_transformer(Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *t, std::string checkpoint_path)
{
  // read in the Config and the Weights from the checkpoint
  read_checkpoint(checkpoint_path, &t->config, &t->weights);
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct
{
  char *str;
  int id;
} TokenIndex;

typedef struct
{
  char **vocab;
  float *vocab_scores;
  TokenIndex *sorted_vocab;
  int vocab_size;
  unsigned int max_token_length;
  unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b)
{
  return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

void build_tokenizer(Tokenizer *t, std::string tokenizer_path, int vocab_size)
{
  // i should have written the vocab_size into the tokenizer file... sigh
  t->vocab_size = vocab_size;
  // malloc space to hold the scores and the strings
  t->vocab = (char **)malloc(vocab_size * sizeof(char *));
  t->vocab_scores = (float *)malloc(vocab_size * sizeof(float));
  t->sorted_vocab = NULL; // initialized lazily
  for (int i = 0; i < 256; i++)
  {
    t->byte_pieces[i * 2] = (unsigned char)i;
    t->byte_pieces[i * 2 + 1] = '\0';
  }
  // read in the file
  FILE *file = fopen(tokenizer_path.c_str(), "rb");
  if (!file)
  {
    fprintf(stderr, "couldn't load %s\n", tokenizer_path.c_str());
    exit(EXIT_FAILURE);
  }
  if (fread(&t->max_token_length, sizeof(int), 1, file) != 1)
  {
    fprintf(stderr, "failed read\n");
    exit(EXIT_FAILURE);
  }
  int len;
  for (int i = 0; i < vocab_size; i++)
  {
    if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1)
    {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    if (fread(&len, sizeof(int), 1, file) != 1)
    {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    t->vocab[i] = (char *)malloc(len + 1);
    if (fread(t->vocab[i], len, 1, file) != 1)
    {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    t->vocab[i][len] = '\0'; // add the string terminating token
  }
  fclose(file);
}

void free_tokenizer(Tokenizer *t)
{
  for (int i = 0; i < t->vocab_size; i++)
  {
    free(t->vocab[i]);
  }
  free(t->vocab);
  free(t->vocab_scores);
  free(t->sorted_vocab);
}

char *decode(Tokenizer *t, int prev_token, int token)
{
  char *piece = t->vocab[token];
  // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
  if (prev_token == 1 && piece[0] == ' ')
  {
    piece++;
  }
  // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
  // parse this and convert and return the actual byte
  unsigned char byte_val;
  if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1)
  {
    piece = (char *)t->byte_pieces + byte_val * 2;
  }
  return piece;
}

void safe_printf(char *piece)
{
  // piece might be a raw byte token, and we only want to print printable chars or whitespace
  // because some of the other bytes can be various control codes, backspace, etc.
  if (piece == NULL)
  {
    return;
  }
  if (piece[0] == '\0')
  {
    return;
  }
  if (piece[1] == '\0')
  {
    unsigned char byte_val = piece[0];
    if (!(isprint(byte_val) || isspace(byte_val)))
    {
      return; // bad byte, don't print it
    }
  }
  printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size)
{
  // efficiently find the perfect match for str in vocab, return its index or -1 if not found
  TokenIndex tok = {.str = str}; // acts as the key to search for
  TokenIndex *res = (TokenIndex *)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
  return res != NULL ? res->id : -1;
}

void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens)
{
  // encode the string text (input) into an upper-bound preallocated tokens[] array
  // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
  if (text == NULL)
  {
    fprintf(stderr, "cannot encode NULL text\n");
    exit(EXIT_FAILURE);
  }

  if (t->sorted_vocab == NULL)
  {
    // lazily malloc and sort the vocabulary
    t->sorted_vocab = (TokenIndex *)malloc(t->vocab_size * sizeof(TokenIndex));
    for (int i = 0; i < t->vocab_size; i++)
    {
      t->sorted_vocab[i].str = t->vocab[i];
      t->sorted_vocab[i].id = i;
    }
    qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
  }

  // create a temporary buffer that will store merge candidates of always two consecutive tokens
  // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
  char *str_buffer = (char *)malloc((t->max_token_length * 2 + 1 + 2) * sizeof(char));
  size_t str_len = 0;

  // start at 0 tokens
  *n_tokens = 0;

  // add optional BOS (=1) token, if desired
  if (bos)
    tokens[(*n_tokens)++] = 1;

  // add_dummy_prefix is true by default
  // so prepend a dummy prefix token to the input string, but only if text != ""
  // TODO: pretty sure this isn't correct in the general case but I don't have the
  // energy to read more of the sentencepiece code to figure out what it's doing
  if (text[0] != '\0')
  {
    int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
    tokens[(*n_tokens)++] = dummy_prefix;
  }

  // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
  // Code point ↔ UTF-8 conversion
  // First code point   Last code point Byte 1  Byte 2  Byte 3  Byte 4
  // U+0000     U+007F      0xxxxxxx
  // U+0080     U+07FF      110xxxxx    10xxxxxx
  // U+0800     U+FFFF      1110xxxx    10xxxxxx        10xxxxxx
  // U+10000    U+10FFFF    11110xxx    10xxxxxx        10xxxxxx        10xxxxxx

  // process the raw (UTF-8) byte sequence of the input string
  for (char *c = text; *c != '\0'; c++)
  {

    // reset buffer if the current byte is ASCII or a leading byte
    // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
    // 0x80 is 10000000
    // in UTF-8, all continuation bytes start with "10" in first two bits
    // so in English this is: "if this byte is not a continuation byte"
    if ((*c & 0xC0) != 0x80)
    {
      // this byte must be either a leading byte (11...) or an ASCII char (0x...)
      // => reset our location, as we're starting a new UTF-8 codepoint
      str_len = 0;
    }

    // append the current byte to the buffer
    str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
    str_buffer[str_len] = '\0';

    // while the next character is a continuation byte, continue appending
    // but if there are too many of them, just stop to avoid overruning str_buffer size.
    if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4)
    {
      continue;
    }

    // ok c+1 is not a continuation byte, so we've read in a full codepoint
    int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

    if (id != -1)
    {
      // we found this codepoint in vocab, add it as a token
      tokens[(*n_tokens)++] = id;
    }
    else
    {
      // byte_fallback encoding: just encode each byte as a token
      // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
      // so the individual bytes only start at index 3
      for (int i = 0; i < str_len; i++)
      {
        tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
      }
    }
    str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
  }

  // merge the best consecutive pair each iteration, according the scores in vocab_scores
  while (1)
  {
    float best_score = -1e10;
    int best_id = -1;
    int best_idx = -1;

    for (int i = 0; i < (*n_tokens - 1); i++)
    {
      // check if we can merge the pair (tokens[i], tokens[i+1])
      sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
      int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
      if (id != -1 && t->vocab_scores[id] > best_score)
      {
        // this merge pair exists in vocab! record its score and position
        best_score = t->vocab_scores[id];
        best_id = id;
        best_idx = i;
      }
    }

    if (best_idx == -1)
    {
      break; // we couldn't find any more pairs to merge, so we're done
    }

    // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
    tokens[best_idx] = best_id;
    // delete token at position best_idx+1, shift the entire sequence back 1
    for (int i = best_idx + 1; i < (*n_tokens - 1); i++)
    {
      tokens[i] = tokens[i + 1];
    }
    (*n_tokens)--; // token length decreased
  }

  // add optional EOS (=2) token, if desired
  if (eos)
    tokens[(*n_tokens)++] = 2;

  free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct
{
  float prob;
  int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct
{
  int vocab_size;
  ProbIndex *probindex; // buffer used in top-p sampling
  float temperature;
  float topp;
  unsigned long long rng_state;
} Sampler;

int sample_argmax(float *probabilities, int n)
{
  // return the index that has the highest probability
  int max_i = 0;
  float max_p = probabilities[0];
  for (int i = 1; i < n; i++)
  {
    if (probabilities[i] > max_p)
    {
      max_i = i;
      max_p = probabilities[i];
    }
  }
  return max_i;
}

int sample_mult(float *probabilities, int n, float coin)
{
  // sample index from probabilities (they must sum to 1!)
  // coin is a random number in [0, 1), usually from random_f32()
  float cdf = 0.0f;
  for (int i = 0; i < n; i++)
  {
    cdf += probabilities[i];
    if (coin < cdf)
    {
      return i;
    }
  }
  return n - 1; // in case of rounding errors
}

int compare(const void *a, const void *b)
{
  ProbIndex *a_ = (ProbIndex *)a;
  ProbIndex *b_ = (ProbIndex *)b;
  if (a_->prob > b_->prob)
    return -1;
  if (a_->prob < b_->prob)
    return 1;
  return 0;
}

int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex, float coin)
{
  // top-p sampling (or "nucleus sampling") samples from the smallest set of
  // tokens that exceed probability topp. This way we never sample tokens that
  // have very low probabilities and are less likely to go "off the rails".
  // coin is a random number in [0, 1), usually from random_f32()

  int n0 = 0;
  // quicksort indices in descending order of probabilities
  // values smaller than (1 - topp) / (n - 1) cannot be part of the result
  // so for efficiency we crop these out as candidates before sorting
  const float cutoff = (1.0f - topp) / (n - 1);
  for (int i = 0; i < n; i++)
  {
    if (probabilities[i] >= cutoff)
    {
      probindex[n0].index = i;
      probindex[n0].prob = probabilities[i];
      n0++;
    }
  }
  qsort(probindex, n0, sizeof(ProbIndex), compare);

  // truncate the list where cumulative probability exceeds topp
  float cumulative_prob = 0.0f;
  int last_idx = n0 - 1; // in case of rounding errors consider all elements
  for (int i = 0; i < n0; i++)
  {
    cumulative_prob += probindex[i].prob;
    if (cumulative_prob > topp)
    {
      last_idx = i;
      break; // we've exceeded topp by including last_idx
    }
  }

  // sample from the truncated list
  float r = coin * cumulative_prob;
  float cdf = 0.0f;
  for (int i = 0; i <= last_idx; i++)
  {
    cdf += probindex[i].prob;
    if (r < cdf)
    {
      return probindex[i].index;
    }
  }
  return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed)
{
  sampler->vocab_size = vocab_size;
  sampler->temperature = temperature;
  sampler->topp = topp;
  sampler->rng_state = rng_seed;
  // buffer only used with nucleus sampling; may not need but it's ~small
  sampler->probindex = (ProbIndex *)malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler *sampler)
{
  free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state)
{
  // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state)
{ // random float32 in [0,1)
  return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler *sampler, float *logits)
{
  // sample the token given the logits and some hyperparameters
  int next;
  if (sampler->temperature == 0.0f)
  {
    // greedy argmax sampling: take the token with the highest probability
    next = sample_argmax(logits, sampler->vocab_size);
  }
  else
  {
    // apply the temperature to the logits
    for (int q = 0; q < sampler->vocab_size; q++)
    {
      logits[q] /= sampler->temperature;
    }
    // apply softmax to the logits to get the probabilities for next token
    softmax(logits, sampler->vocab_size);
    // flip a (float) coin (this is our source of entropy for sampling)
    float coin = random_f32(&sampler->rng_state);
    // we sample from this distribution to get the next token
    if (sampler->topp <= 0 || sampler->topp >= 1)
    {
      // simply sample from the predicted probability distribution
      next = sample_mult(logits, sampler->vocab_size, coin);
    }
    else
    {
      // top-p (nucleus) sampling, clamping the least likely tokens to zero
      next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
    }
  }
  return next;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms()
{
  // return time in milliseconds, for benchmarking the model speed
  struct timespec time;
  clock_gettime(CLOCK_REALTIME, &time);
  return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop
template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
void generate(Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps, std::string &kernelpath, int start_pos = 0)
{
  char *empty_prompt = "";
  if (prompt == NULL)
  {
    prompt = empty_prompt;
  }

  // encode the (string) prompt into tokens sequence
  int num_prompt_tokens = 0;
  int *prompt_tokens = (int *)malloc((strlen(prompt) + 3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
  encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
  if (num_prompt_tokens < 1)
  {
    fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
    exit(EXIT_FAILURE);
  }

  std::cout << "Loading kernel..." << std::endl;
  auto device = xrt::device(0);
  auto uuid = device.load_xclbin(kernelpath);
  auto kernel = xrt::kernel(device, uuid, "forward");
  std::cout << "Out buffer size: " << vocab_size * sizeof(act_t) << std::endl;
  std::cout << "Transformer size: " << sizeof(*transformer) << std::endl;
  int cache_dim = n_layers * seq_len * ((dim * n_kv_heads) / n_heads);

  // kernel args (group_id includes scalars):
  //   0=transformer, 1=token(scalar), 2=pos(scalar),
  //   3=kv_k_cache, 4=kv_v_cache, 5=out, 6=wq_flat, 7=ws_flat
  constexpr int kv_dim_local = (dim * n_kv_heads) / n_heads;
  constexpr long long k_indices_size = (long long)n_layers * seq_len * (kv_dim_local / 2);
  constexpr long long v_indices_size = (long long)n_layers * seq_len * (kv_dim_local / 4);
  constexpr long long norms_size = (long long)n_layers * seq_len * n_kv_heads;

  std::cout << "Allocating transformer buffer (" << sizeof(*transformer) << " bytes)..." << std::endl;
  auto transformer_buffer = xrt::bo(device, sizeof(*transformer), kernel.group_id(0));
  std::cout << "  OK" << std::endl;

  std::cout << "Allocating output buffer (" << vocab_size * sizeof(float) << " bytes)..." << std::endl;
  auto out_buffer = xrt::bo(device, vocab_size * sizeof(act_t), kernel.group_id(5));
  std::cout << "  OK" << std::endl;

  // 6-port merged: kv_k = K_indices + K_norms, kv_v = V_indices + V_norms
  long long k_buf_size = (long long)k_indices_size + norms_size * sizeof(act_t);
  long long v_buf_size = (long long)v_indices_size + norms_size * sizeof(act_t);
  std::cout << "Allocating kv_k_buffer (" << k_buf_size << " B)..." << std::endl;
  auto kv_k_buffer = xrt::bo(device, k_buf_size, kernel.group_id(3));
  std::cout << "  OK" << std::endl;
  std::cout << "Allocating kv_v_buffer (" << v_buf_size << " B)..." << std::endl;
  auto kv_v_buffer = xrt::bo(device, v_buf_size, kernel.group_id(4));
  std::cout << "  OK" << std::endl;

  // Flatten weights for burst_maxi
  std::cout << "Flattening weights..." << std::endl;
  int8_t *wq_flat = (int8_t *)malloc(WQ_FLAT_SIZE);
  uint16_t *ws_flat = (uint16_t *)malloc(WS_FLAT_SIZE * 2);
  flatten_weights(transformer, wq_flat, ws_flat);

  std::cout << "Allocating wq_flat (" << WQ_FLAT_SIZE << " B)..." << std::endl;
  auto wq_buffer = xrt::bo(device, WQ_FLAT_SIZE, kernel.group_id(6));
  std::cout << "Allocating ws_flat (" << WS_FLAT_SIZE*2 << " B)..." << std::endl;
  auto ws_buffer_flat = xrt::bo(device, WS_FLAT_SIZE * 2, kernel.group_id(7));
  wq_buffer.write(wq_flat, WQ_FLAT_SIZE, 0);
  wq_buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  ws_buffer_flat.write(ws_flat, WS_FLAT_SIZE * 2, 0);
  ws_buffer_flat.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  free(wq_flat); free(ws_flat);

  std::cout << "Copying data to buffer" << std::endl;
  transformer_buffer.write(transformer, sizeof(*transformer), 0);
  transformer_buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  long start = 0;
  int next;
  int token = prompt_tokens[0];
  int pos = start_pos;
  if (start_pos > 0) {
    // Benchmark mode: skip to start_pos, generate 'steps' tokens from there
    int bench_steps = steps;
    steps = start_pos + steps;
    if (steps > seq_len) steps = seq_len;
    fprintf(stderr, "Benchmark mode: start_pos=%d, steps=%d, end=%d\n", start_pos, bench_steps, steps);
  }

  auto run = kernel(transformer_buffer, token, pos,
                    kv_k_buffer, kv_v_buffer,
                    out_buffer, wq_buffer, ws_buffer_flat);
  run.wait();

  // transformer_buffer.read(&transformer2, sizeof(*transformer), 0);
  // printf("tensor[i].q[0] = %d\n", transformer2.weights.w1[0].q[0]);

  act_t *logits_half = (act_t *)malloc(vocab_size * sizeof(act_t));
  float *logits = (float *)malloc(vocab_size * sizeof(float));
  out_buffer.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  out_buffer.read(logits_half, vocab_size * sizeof(act_t), 0);
    for(int i=0;i<vocab_size;i++) logits[i]=half_to_float(logits_half[i]);

  // advance the state state machine
  if (pos < num_prompt_tokens - 1)
  {
    // if we are still processing the input prompt, force the next prompt token
    next = prompt_tokens[pos + 1];
  }
  else
  {
    // otherwise sample the next token from the logits
    next = sample(sampler, logits);
  }
  pos++;

  // print the token as string, decode it with the Tokenizer object
  char *piece = decode(tokenizer, token, next);
  safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
  fflush(stdout);
  token = next;
  start = time_in_ms();
  // end first run
  while (pos < steps)
  {
    run.set_arg(1, token);
    run.set_arg(2, pos);
    run.start();
    run.wait();
    // transformer_buffer.read(&transformer2, sizeof(*transformer), 0);
    // printf("tensor[i].q[0] = %d\n", transformer2.weights.w1[0].q[0]);

    // float *logits = (float *)malloc(vocab_size * sizeof(float));

    out_buffer.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    out_buffer.read(logits_half, vocab_size * sizeof(act_t), 0);
    for(int i=0;i<vocab_size;i++) logits[i]=half_to_float(logits_half[i]);

    // advance the state state machine
    if (pos < num_prompt_tokens - 1)
    {
      // if we are still processing the input prompt, force the next prompt token
      next = prompt_tokens[pos + 1];
    }
    else
    {
      // otherwise sample the next token from the logits
      next = sample(sampler, logits);
    }
    pos++;
    
    if (pos == 256 || pos == 1024 || pos == 4096 || pos == 16384 || pos == 65536) {
      if (pos > start_pos + 1) {
        long intermediate_end = time_in_ms();
        int tokens_generated = pos - start_pos - 1;
        fprintf(stderr, "\n[INTERVAL] achieved tok/s: %f (tokens=%d, pos=%d)\n", tokens_generated / (double)(intermediate_end - start) * 1000, tokens_generated, pos);
      }
    }

    // data-dependent terminating condition: the BOS (=1) token delimits sequences
    if (next == 1)
    {
      // break; // Force exact N tokens for full 64K sweep evaluation
    }

    // print the token as string, decode it with the Tokenizer object
    char *piece = decode(tokenizer, token, next);
    safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
    fflush(stdout);
    token = next;
  }
  printf("\n");

  // report achieved tok/s (timer starts after first iteration)
  if (pos > start_pos + 1)
  {
    long end = time_in_ms();
    int tokens_generated = pos - start_pos - 1;
    fprintf(stderr, "achieved tok/s: %f (tokens=%d, pos=%d)\n", tokens_generated / (double)(end - start) * 1000, tokens_generated, pos);
  }

  free(prompt_tokens);
}

void read_stdin(const char *guide, char *buffer, size_t bufsize)
{
  // read a line from stdin, up to but not including \n
  printf("%s", guide);
  if (fgets(buffer, bufsize, stdin) != NULL)
  {
    size_t len = strlen(buffer);
    if (len > 0 && buffer[len - 1] == '\n')
    {
      buffer[len - 1] = '\0'; // strip newline
    }
  }
}

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage()
{
  fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
  fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
  fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
  fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
  fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
  fprintf(stderr, "  -i <string> input prompt\n");
  fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
  fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
  fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
  fprintf(stderr, "  -b <int>    benchmark start position, default 0 (normal generation)\n");
  exit(EXIT_FAILURE);
}

int main(int argc, char *argv[])
{
  std::cout << "start" << std::endl;
  // default parameters
  std::string checkpoint_path = ""; // e.g. out/model.bin
  std::string tokenizer_path = "tokenizer.bin";
  float temperature = 1.0f;        // 0.0 = greedy deterministic. 1.0 = original. don't set higher
  float topp = 0.9f;               // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
  int steps = 256;                 // number of steps to run for
  int start_pos = 0;               // starting position for benchmark
  char *prompt = NULL;             // prompt string
  unsigned long long rng_seed = 0; // seed rng with time by default
  const char *mode = "generate";   // generate|chat
  char *system_prompt = NULL;      // the (optional) system prompt to use in chat mode
  std::string kernelpath = "";

  // poor man's C argparse so we can override the defaults above from the command line
  if (argc >= 2)
  {
    checkpoint_path = argv[1];
  }
  else
  {
    std::cout << "test1" << std::endl;
    error_usage();
  }
  for (int i = 2; i < argc; i += 2)
  {
    // do some basic validation
    if (i + 1 >= argc)
    {
      error_usage();
    } // must have arg after flag
    if (argv[i][0] != '-')
    {
      error_usage();
    } // must start with dash
    if (strlen(argv[i]) != 2)
    {
      error_usage();
    } // must be -x (one dash, one letter)
    // read in the args
    if (argv[i][1] == 't')
    {
      temperature = atof(argv[i + 1]);
    }
    else if (argv[i][1] == 'p')
    {
      topp = atof(argv[i + 1]);
    }
    else if (argv[i][1] == 's')
    {
      rng_seed = atoi(argv[i + 1]);
    }
    else if (argv[i][1] == 'n')
    {
      steps = atoi(argv[i + 1]);
    }
    else if (argv[i][1] == 'i')
    {
      prompt = argv[i + 1];
    }
    else if (argv[i][1] == 'z')
    {
      tokenizer_path = argv[i + 1];
    }
    else if (argv[i][1] == 'm')
    {
      mode = argv[i + 1];
    }
    else if (argv[i][1] == 'y')
    {
      system_prompt = argv[i + 1];
    }
    else if (argv[i][1] == 'k')
    {
      kernelpath = argv[i + 1];
    }
    else if (argv[i][1] == 'b')
    {
      start_pos = atoi(argv[i + 1]);
    }
    else
    {
      error_usage();
    }
  }

  // parameter validation/overrides
  if (rng_seed <= 0)
    rng_seed = (unsigned int)time(NULL);
  if (temperature < 0.0)
    temperature = 0.0;
  if (topp < 0.0 || 1.0 < topp)
    topp = 0.9;
  if (steps < 0)
    steps = 0;

  // build the Transformer via the model .bin file
  static Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> transformer;
  build_transformer(&transformer, checkpoint_path);
  if (steps == 0)
    steps = transformer.config.seq_len; // only ovrerride if user asked for 0 (default)

  // build the Tokenizer via the tokenizer .bin file
  Tokenizer tokenizer;
  build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

  // build the Sampler
  Sampler sampler;
  build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

  // run!
  if (strcmp(mode, "generate") == 0)
  {
    generate(&transformer, &tokenizer, &sampler, prompt, steps, kernelpath, start_pos);
  }

  else
  {
    fprintf(stderr, "unknown mode: %s\n", mode);
    error_usage();
  }

  // memory and file handles cleanup
  free_sampler(&sampler);
  free_tokenizer(&tokenizer);
  return 0;
}
#endif
