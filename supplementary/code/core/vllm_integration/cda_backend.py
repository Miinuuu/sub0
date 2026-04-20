"""CDA AttentionBackend for vLLM 0.19 (v1 API).

Skeleton implementation. The four classes follow the
:mod:`vllm.v1.attention.backend` contract:

  * :class:`CDAAttentionBackend` — static metadata (name, impl class,
    KV cache shape, supported dtypes).
  * :class:`CDAAttentionMetadata` — per-batch state consumed by
    :meth:`CDAAttentionImpl.forward`. Built from
    :class:`CommonAttentionMetadata` by the builder below.
  * :class:`CDAAttentionMetadataBuilder` — transforms vLLM's scheduler
    output into our backend-specific metadata each step.
  * :class:`CDAAttentionImpl` — ``forward`` (compute) +
    ``do_kv_cache_update`` (write prefill K/V into paged cache).

Design principles:

* KV cache shape is ``(num_blocks, block_size, num_kv_heads, 104)`` uint8:
  bytes ``[0, 64)`` hold 4-bit K, ``[64, 96)`` hold 2-bit V, ``[96, 104)``
  store K+V fp32 norms. 4.9× smaller than the FP16 layout FlashAttn uses.
* Prefill (``query_len > 1``): compute attention on FP16 Q/K/V (reuse
  FA2 since we still have the uncompressed tensors in registers), then
  quantize K/V to CDA in ``do_kv_cache_update`` and write to slot mapping.
* Decode (``query_len == 1``): read CDA blocks via
  ``cuda_hw_attention_flash_gqa_paged`` (TODO: paged flash variant;
  for now the existing ``cuda_hw_attention_gqa_paged`` handles K4V2).
* GQA: `num_kv_heads < num_heads`; our kernels derive ``kv_head =
  q_head // group_size`` internally.

Status: Day 2 skeleton — methods raise ``NotImplementedError`` until
Day 3-4 wires in the real compute paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

import numpy as np
import torch
from torch.nn.attention.bias import causal_lower_right

from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.config import VllmConfig


# =============================================================================
# Cache layout constants (K4V2 with fp32 K,V norms)
# =============================================================================
# Per-token, per-KV-head byte layout inside the paged cache:
#   offset  size  content
#   ──────  ────  ─────────────────────────────
#        0    64  K 4-bit packed (D=128 → 64 B)
#       64    32  V 2-bit packed (D=128 → 32 B)
#       96     4  norm_K fp32
#      100     4  norm_V fp32
#      104              (total)
_BYTES_PER_TOKEN = 104
_K_OFFSET = 0
_K_NBYTES = 64           # D // 2
_V_OFFSET = 64
_V_NBYTES = 32           # D // 4
_NORM_K_OFFSET = 96
_NORM_V_OFFSET = 100
# vLLM's kv_cache_spec.dtype is fp16 for standard models → its allocator
# reserves block_size · num_kv_heads · 2 · head_size · 2 bytes per block
# (FP16 K and FP16 V). That's 512 bytes per (token, kv_head) for D=128 —
# 4.92× more than our 104 B CDA layout. To remain compatible with vLLM
# 0.19's ``_reshape_kv_cache_tensors`` sizing we return a shape whose
# total bytes match that allocation and write CDA data into the first
# 104 bytes of each slot, leaving the rest unused for now. Realising the
# full 4.92× memory saving inside vLLM requires teaching the spec to use
# ``dtype=uint8`` + smaller ``real_page_size_bytes`` — tracked as Day 6+.
_SLOT_BYTES_VLLM_ALIGNED = 512   # matches FP16 per-(token, kv_head) bytes
_SLOT_FP16_ELEMS_PER_TOKEN = 256  # 512 bytes / 2 B per fp16
# When the CDA spec patch is active (``enable_cda_memory_saving()``),
# vLLM's allocator reserves 104 bytes per (token, kv_head). 104 B
# does not split into whole fp16 elements — but since vLLM's
# ``_reshape_kv_cache_tensors`` uses the dtype from ``kv_cache_spec`` and
# our pretend-as-fp16 cast yields non-integer elements, we advertise the
# shape in uint8 units instead.  vLLM happily reshapes an fp16 buffer as
# uint8 since dtype-size ratio is 1/2 → element count doubles.
_SLOT_FP16_ELEMS_COMPACT = _BYTES_PER_TOKEN // 2  # 52 fp16 elements = 104 B


def _copy_out(output: torch.Tensor, s: int, e: int, out_b: torch.Tensor,
               H_q: int, D: int) -> None:
    """Copy ``out_b`` (shape ``(L, H_q, D)``) into ``output[s:e]``.
    vLLM 0.19 reshapes output to 3D ``(num_tokens, num_heads, head_size)``
    before calling backend.forward; older call sites still pass a 2D
    ``(num_tokens, num_heads * head_size)`` buffer. Handle both."""
    L = e - s
    if output.dim() == 3:
        output[s:e].copy_(out_b.to(output.dtype))
    else:
        output[s:e].copy_(out_b.reshape(L, H_q * D).to(output.dtype))


def _copy_out_single(output: torch.Tensor, b: int, out_b: torch.Tensor,
                       H_q: int, D: int) -> None:
    """Per-request decode: out_b is ``(H_q, D)``."""
    if output.dim() == 3:
        output[b].copy_(out_b.to(output.dtype))
    else:
        output[b].copy_(out_b.reshape(-1).to(output.dtype))


def _as_uint8_cache(kv_cache: torch.Tensor) -> torch.Tensor:
    """Reinterpret vLLM's fp16-dtype KV cache buffer as uint8 bytes.

    vLLM 0.19's allocator forces ``kv_cache_spec.dtype`` (typically fp16)
    when reshaping the raw buffer. We advertise a shape with 256 fp16
    elements per (token, kv_head) slot — matching FP16's 512-byte slot —
    and flip to uint8 for CDA packing inside the backend.
    """
    if kv_cache.dtype == torch.uint8:
        return kv_cache
    num_blocks, block_size, num_kv_heads, last = kv_cache.shape
    # fp16 → uint8 doubles the last dim (2 B per fp16).
    if last == _SLOT_FP16_ELEMS_PER_TOKEN:
        byte_width = _SLOT_BYTES_VLLM_ALIGNED              # 512
    elif last == _SLOT_FP16_ELEMS_COMPACT:
        byte_width = _BYTES_PER_TOKEN                      # 104
    else:
        raise ValueError(
            f"unexpected CDA cache last dim {last} "
            f"(expected {_SLOT_FP16_ELEMS_PER_TOKEN} or {_SLOT_FP16_ELEMS_COMPACT})")
    return kv_cache.view(torch.uint8).view(
        num_blocks, block_size, num_kv_heads, byte_width)


def unpack_paged_cache(
    kv_cache: torch.Tensor,       # (num_blocks, block_size, num_kv_heads, 512) uint8
    slot_mapping: torch.Tensor,   # (num_tokens,) int64
):
    """Read packed CDA slots back into per-token tensors.

    Helper used by the forward path (Day 4) and by correctness tests.
    Returns a 4-tuple ``(pK, nK, pV, nV)`` where:

      * ``pK`` shape ``(num_tokens, num_kv_heads, 64)`` uint8 — 4-bit K
      * ``nK`` shape ``(num_tokens, num_kv_heads)`` float32
      * ``pV`` shape ``(num_tokens, num_kv_heads, 32)`` uint8 — 2-bit V
      * ``nV`` shape ``(num_tokens, num_kv_heads)`` float32

    CDA data sits in the first ``_BYTES_PER_TOKEN`` bytes of each slot;
    the rest of the slot is padding so the full cache size matches what
    vLLM 0.19's FP16-centric allocator reserves.
    """
    # Accept both the raw uint8 (tests) and vLLM-allocated fp16 cache.
    kv_cache = _as_uint8_cache(kv_cache) if kv_cache.dtype != torch.uint8 else kv_cache
    num_blocks, block_size, num_kv_heads, width = kv_cache.shape
    assert width in (_BYTES_PER_TOKEN, _SLOT_BYTES_VLLM_ALIGNED), (
        f"unexpected CDA slot width {width}")
    flat = kv_cache.view(num_blocks * block_size, num_kv_heads, width)
    slots = flat.index_select(0, slot_mapping.to(torch.long))  # (N, H_kv, width)
    num_tokens = slots.shape[0]
    pK = slots[:, :, _K_OFFSET:_K_OFFSET + _K_NBYTES].contiguous()
    pV = slots[:, :, _V_OFFSET:_V_OFFSET + _V_NBYTES].contiguous()
    nK = (slots[:, :, _NORM_K_OFFSET:_NORM_K_OFFSET + 4].contiguous()
                 .view(num_tokens, num_kv_heads, 4)
                 .view(torch.float32).squeeze(-1)
                 .contiguous())
    nV = (slots[:, :, _NORM_V_OFFSET:_NORM_V_OFFSET + 4].contiguous()
                 .view(num_tokens, num_kv_heads, 4)
                 .view(torch.float32).squeeze(-1)
                 .contiguous())
    return pK, nK, pV, nV


# =============================================================================
# Metadata dataclass
# =============================================================================

@dataclass
class CDAAttentionMetadata(AttentionMetadata):
    """Per-batch metadata for CDA backend.

    Mirrors the fields FlashAttn uses plus book-keeping our kernels need:
    pack widths, codebook pointers (shared across layers), Hadamard rotation.
    """
    # Query indexing (same conventions as FlashAttentionMetadata).
    num_actual_tokens: int
    num_reqs: int
    max_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor       # (num_reqs+1,) int32
    seq_lens: torch.Tensor              # (num_reqs,) int32
    block_table: torch.Tensor           # (num_reqs, max_blocks_per_req) int32
    slot_mapping: torch.Tensor          # (num_tokens,) int64
    causal: bool = True

    # CDA-specific: codebook/rotation are shared across layers (set once by
    # the model init hook). Stored on the metadata only for plumbing; the
    # actual tensors are allocated by the backend init.
    codebook_k: Optional[torch.Tensor] = None   # (16,) float32
    codebook_v: Optional[torch.Tensor] = None   # (4,)  float32
    rotation:   Optional[torch.Tensor] = None   # (D, D) float32


# =============================================================================
# Metadata builder
# =============================================================================

class CDAAttentionMetadataBuilder(AttentionMetadataBuilder[CDAAttentionMetadata]):
    """Build :class:`CDAAttentionMetadata` from ``CommonAttentionMetadata``.

    Called once per forward step by vLLM's model runner. No heavy allocations
    here — everything is derived from the tensors already in ``common``.
    """
    # Attention runs eagerly between piecewise-captured MLP/norm chunks.
    # FULL-graph capture of our decode path produced gibberish output in
    # the Day 12 probe — our kernel wrapper still allocates partial_out /
    # m_vals / l_vals each call (torch::zeros inside the C++ binding),
    # and those allocations are not safe under a FULL captured-attention
    # replay. PIECEWISE is enough to get MLP + norms + sampler fusion
    # wins without the full-graph correctness risk; revisit once the
    # kernel wrapper is reworked to accept pre-allocated scratch buffers.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # Codebook + rotation are loaded lazily by the impl (the compressor
        # is a pure-Python object in core.compression; we instantiate the
        # shared tensors on first forward and cache them here).
        self._codebook_k: Optional[torch.Tensor] = None
        self._codebook_v: Optional[torch.Tensor] = None
        self._rotation:   Optional[torch.Tensor] = None

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CDAAttentionMetadata:
        c = common_attn_metadata
        return CDAAttentionMetadata(
            num_actual_tokens=c.num_actual_tokens,
            num_reqs=c.num_reqs,
            max_query_len=c.max_query_len,
            max_seq_len=c.max_seq_len,
            query_start_loc=c.query_start_loc,
            seq_lens=c.seq_lens,
            block_table=c.block_table_tensor,
            slot_mapping=c.slot_mapping,
            causal=c.causal,
            codebook_k=self._codebook_k,
            codebook_v=self._codebook_v,
            rotation=self._rotation,
        )


# =============================================================================
# Backend (static metadata)
# =============================================================================

class CDAAttentionBackend(AttentionBackend):
    """Static descriptor for the CDA backend.

    vLLM uses this to pick KV cache shape + dtype and route tensors to
    :class:`CDAAttentionImpl`. All logic lives in the Impl class.
    """
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[str]] = ["auto", "uint8"]
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Our paged kernel was designed around block_size=16 (matches vLLM's
        # default). Other multiples of 16 also work.
        return [MultipleOf(16)]

    @staticmethod
    def get_name() -> str:
        # vLLM looks up the backend by name against AttentionBackendEnum; we
        # slot into the CUSTOM entry which is reserved for third-party hooks.
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["CDAAttentionImpl"]:
        return CDAAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[CDAAttentionMetadataBuilder]:
        return CDAAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """Shape of the paged KV cache.

        To stay compatible with vLLM 0.19's allocator (which sizes pages
        assuming FP16 K + FP16 V = 2 · head_size · 2 = 512 B per token per
        KV head), we return a shape whose total byte count matches that
        allocation. We only **use** the first 104 bytes per slot for CDA
        data (K + V packed indices + norms); the remaining bytes are
        padding and will be reclaimed once the AttentionSpec is taught
        ``dtype=uint8``.
        """
        if head_size != 128:
            raise ValueError(
                f"CDAAttentionBackend currently supports head_size=128 only "
                f"(Llama-3.1-8B). Got head_size={head_size}."
            )
        # Shape in fp16 units. Two regimes:
        # (a) default — no spec patch applied, vLLM reserves FP16-aligned
        #     pages (512 B/slot). Return 256 fp16 elements = 512 B/slot;
        #     only the first 104 B are used. Wastes 4.92× memory.
        # (b) patched — enable_cda_memory_saving() has shrunk
        #     real_page_size_bytes to 104·block_size·num_kv_heads. vLLM
        #     then allocates at the compact size; return 52 fp16 = 104 B
        #     to match exactly.
        from vllm.v1.kv_cache_interface import AttentionSpec
        compact = getattr(AttentionSpec, "_cda_patched", False)
        last = _SLOT_FP16_ELEMS_COMPACT if compact else _SLOT_FP16_ELEMS_PER_TOKEN
        return (num_blocks, block_size, num_kv_heads, last)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128]

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        # Simple uint8 tensor copy; block layout is identical across GPUs.
        dst_kv_cache[src_to_dst[:, 1]] = src_kv_cache[src_to_dst[:, 0]]

    @staticmethod
    def copy_blocks(
        kv_caches: list[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        # Per-layer copy.
        for kv_cache in kv_caches:
            kv_cache[src_to_dists[:, 1]] = kv_cache[src_to_dists[:, 0]]


# =============================================================================
# AttentionImpl (forward + kv_cache_update)
# =============================================================================

class CDAAttentionImpl(AttentionImpl[CDAAttentionMetadata]):
    """Forward pass + KV cache write for CDA.

    Day 2 skeleton: methods raise ``NotImplementedError`` with TODOs.
    Day 3 lands ``do_kv_cache_update`` (prefill quantize + paged write).
    Day 4 lands ``forward`` (paged CDA read + kernel call).
    """

    can_return_lse_for_decode: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads or num_heads
        self.group_size = self.num_heads // self.num_kv_heads
        self.attn_type = attn_type
        if alibi_slopes is not None:
            raise NotImplementedError("CDA backend does not support ALiBi yet.")
        if sliding_window is not None:
            raise NotImplementedError(
                "CDA backend does not support sliding window yet.")
        if attn_type not in (AttentionType.DECODER,):
            raise NotImplementedError(
                f"CDA backend currently supports DECODER attention only, "
                f"got {attn_type}.")

        # Compressor / codebook / rotation are lazy-initialised on first call
        # — we don't have access to the compressor at __init__ time (vLLM
        # instantiates Impl before model weights land). See ``_ensure_init``.
        self._compressor_k = None
        self._compressor_v = None
        self._codebook_k: Optional[torch.Tensor] = None
        self._codebook_v: Optional[torch.Tensor] = None
        self._rotation: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Lazy init of compressor / codebook / rotation (called on first forward).
    # ------------------------------------------------------------------
    def _ensure_init(self, device: torch.device) -> None:
        if self._compressor_k is not None:
            return
        from core.compression import HadamardQuantCompressor
        self._compressor_k = HadamardQuantCompressor(
            dim=self.head_size, bit_width=4, half_rotation=True)
        self._compressor_v = HadamardQuantCompressor(
            dim=self.head_size, bit_width=2, half_rotation=True)
        self._compressor_k._ensure_tensors(device)
        self._compressor_v._ensure_tensors(device)
        self._codebook_k = (self._compressor_k._centroids * 2.0 - 1.0).float().contiguous()
        self._codebook_v = (self._compressor_v._centroids * 2.0 - 1.0).float().contiguous()
        self._rotation = self._compressor_k._rotation.float().contiguous()
        # Pre-computed fp16 copy for fused_reduce_rot_cast kernel (Tier 2
        # optimisation): avoids a per-layer fp32→fp16 cast.
        self._rotation_fp16 = self._rotation.to(torch.float16).contiguous()
        # Pre-computed Lloyd-Max boundaries (fp32) + Hadamard fp16 for the
        # fused quantise+scatter kernel. Computing these per layer per step
        # ate ~5-10 μs/layer in the hot path (32 layers × 64 decode = measurable).
        self._cb_k_bounds = self._compressor_k._boundaries[1:-1].to(torch.float32).contiguous()
        self._cb_v_bounds = self._compressor_v._boundaries[1:-1].to(torch.float32).contiguous()
        self._rotation_k_fp16 = (
            self._compressor_k._rotation.to(torch.float16).contiguous()
            if self._compressor_k._rotation.dtype != torch.float16
            else self._compressor_k._rotation)
        # Env flag read once at init (avoid per-call os.environ dict lookup).
        import os as _os
        self._use_fused_update = (_os.environ.get("CDA_DISABLE_FUSED_UPDATE") != "1")
        self._use_fused_tail   = (_os.environ.get("CDA_DISABLE_FUSED_TAIL")   != "1")
        self._use_hmma         = (_os.environ.get("CDA_DISABLE_HMMA")         != "1")
        # L1 fused decode — single C++ dispatch for Q_rot + HMMA + reduce+rot+cast.
        # Opt-in default ON; disable with CDA_DISABLE_FUSED_DECODE=1 for A/B.
        self._use_fused_decode = (_os.environ.get("CDA_DISABLE_FUSED_DECODE") != "1")
        # Diagnostic timing (profile B): collect per-call GPU time for
        # do_kv_cache_update and _forward_decode so we can attribute the vLLM
        # E2E gap against isolated kernel measurements. Enable with
        # CDA_PROFILE_KERNEL=1, dump stats at process exit via atexit.
        self._profile_kernel = (_os.environ.get("CDA_PROFILE_KERNEL") == "1")
        if self._profile_kernel:
            if not hasattr(CDAAttentionImpl, "_profile_events"):
                CDAAttentionImpl._profile_events = {"kv_update": [], "decode": []}
                # Periodic dump: every 32 decode events, flush stats to
                # /tmp/cda_profile.jsonl. Subprocess atexit is unreliable
                # under vLLM EngineCore, so we don't depend on it.
                CDAAttentionImpl._profile_dump_interval = 32 * 16  # ~16 tokens
                CDAAttentionImpl._profile_dump_counter = [0]
                def _dump_now():
                    import torch as _t, statistics as _st, json as _json
                    _t.cuda.synchronize()
                    lines = []
                    for name, pairs in CDAAttentionImpl._profile_events.items():
                        if not pairs: continue
                        us = [s.elapsed_time(e) * 1000.0 for s, e in pairs]
                        us.sort()
                        lines.append({
                            "phase": name, "n": len(us),
                            "median": _st.median(us), "sum_ms": sum(us)/1000.0,
                            "p10": us[len(us)//10], "p90": us[len(us)*9//10],
                        })
                    with open("/tmp/cda_profile.jsonl", "w") as _f:
                        for l in lines: _f.write(_json.dumps(l) + "\n")
                CDAAttentionImpl._profile_dump = staticmethod(_dump_now)

    # ------------------------------------------------------------------
    # KV cache write — called externally each layer per forward step.
    # Quantizes the new (prefill or decode) K/V and scatters into the paged
    # cache at the slot positions vLLM's scheduler assigned.
    # ------------------------------------------------------------------
    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,      # (num_tokens, num_kv_heads, head_size) fp16
        value: torch.Tensor,    # (num_tokens, num_kv_heads, head_size) fp16
        kv_cache: torch.Tensor, # (num_blocks, block_size, num_kv_heads, 104) uint8
        slot_mapping: torch.Tensor,  # (num_tokens,) int64
    ) -> None:
        if kv_cache.numel() == 0:
            return                                # profiling dummy
        self._ensure_init(kv_cache.device)
        if self._profile_kernel and key.shape[0] <= 64:  # decode calls only
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            ret = self._do_kv_cache_update_inner(
                layer, key, value, kv_cache, slot_mapping)
            e.record()
            CDAAttentionImpl._profile_events["kv_update"].append((s, e))
            return ret
        return self._do_kv_cache_update_inner(
            layer, key, value, kv_cache, slot_mapping)

    def _do_kv_cache_update_inner(
        self, layer, key, value, kv_cache, slot_mapping):
        # vLLM hands us a fp16-typed cache; flip to uint8 view for 104-B slots.
        kv_cache = _as_uint8_cache(kv_cache)
        # Minimise CPU-side work (per FA2 backend guidance): skip the shape
        # asserts (vLLM invariants) and the dtype check (vLLM ≥0.19 passes
        # int64 slot_mapping already). The fused kernel treats slot_mapping
        # == -1 as a no-op scatter, so we can pass padded key/value as-is
        # without slicing them.
        num_blocks, block_size, num_kv_heads, slot_width = kv_cache.shape
        flat_cache = kv_cache.view(num_blocks * block_size, num_kv_heads,
                                     slot_width)

        if self._use_fused_update:
            from core.cda_attn import cda_quantize_and_scatter
            cda_quantize_and_scatter(
                key, value, self._rotation_k_fp16,
                self._cb_k_bounds, self._cb_v_bounds,
                slot_mapping, flat_cache,
            )
            return
        # Slow path kept for A/B comparison — matches the FA-eager behaviour
        # of slicing to num_tokens and building slot_bytes in Python.
        num_tokens = slot_mapping.shape[0]
        if num_tokens == 0:
            return
        if key.shape[0] != num_tokens:
            key   = key[:num_tokens]
            value = value[:num_tokens]
        D = self.head_size

        # Reference path (fallback): Python pipeline via compressor +
        # multi-kernel scatter. Used when CDA_DISABLE_FUSED_UPDATE=1.
        # Flatten (tokens × heads) so the compressor can batch-quantize.
        key_flat   = key.reshape(-1, D).contiguous()
        value_flat = value.reshape(-1, D).contiguous()

        cK = self._compressor_k.quantize(key_flat)    # 4-bit K
        cV = self._compressor_v.quantize(value_flat)  # 2-bit V

        # Un-flatten back to (tokens, heads, packed_width).
        pK = cK.indices.view(num_tokens, num_kv_heads, _K_NBYTES)
        pV = cV.indices.view(num_tokens, num_kv_heads, _V_NBYTES)
        nK = cK.norms.float().view(num_tokens, num_kv_heads).contiguous()
        nV = cV.norms.float().view(num_tokens, num_kv_heads).contiguous()

        # Build the per-(token, head) 104-byte record.
        slot_bytes = torch.empty(
            num_tokens, num_kv_heads, _BYTES_PER_TOKEN,
            dtype=torch.uint8, device=kv_cache.device,
        )
        slot_bytes[:, :, _K_OFFSET:_K_OFFSET + _K_NBYTES] = pK
        slot_bytes[:, :, _V_OFFSET:_V_OFFSET + _V_NBYTES] = pV
        # Reinterpret fp32 norms as 4-byte groups. ``view(torch.uint8)``
        # preserves bit pattern — decode path reads them back as fp32.
        slot_bytes[:, :, _NORM_K_OFFSET:_NORM_K_OFFSET + 4] = \
            nK.contiguous().view(torch.uint8).view(num_tokens, num_kv_heads, 4)
        slot_bytes[:, :, _NORM_V_OFFSET:_NORM_V_OFFSET + 4] = \
            nV.contiguous().view(torch.uint8).view(num_tokens, num_kv_heads, 4)

        # Graph-safe scatter: dedicated CUDA kernel that skips rows whose
        # ``slot_mapping`` is negative or out of range (instead of
        # index_copy_'s device-side assert). Handles both patched (slot
        # stride 104 B) and unpatched (stride 512 B) layouts via the
        # ``slot_stride`` argument read from ``flat_cache.size(2)``. vLLM's
        # CUDA-Graph capture pads dummy decode batches with
        # slot_mapping=-1, so graph-safe no-op is essential.
        from core.cda_attn import _get_gqa
        _get_gqa().cda_paged_scatter(
            flat_cache, slot_bytes, slot_mapping.to(torch.long))

    # ------------------------------------------------------------------
    # Forward pass — prefill via SDPA on fresh fp16, decode via CDA kernel.
    # ------------------------------------------------------------------
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,    # (num_tokens, num_heads, head_size) fp16
        key: torch.Tensor,      # (num_tokens, num_kv_heads, head_size) fp16
        value: torch.Tensor,    # (num_tokens, num_kv_heads, head_size) fp16
        kv_cache: torch.Tensor, # (num_blocks, block_size, num_kv_heads, 104) uint8
        attn_metadata: CDAAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dispatch: prefill (query_len > 1) uses fresh fp16 SDPA,
        decode (query_len == 1) reads CDA paged cache via our kernel."""
        if output is None:
            raise ValueError("CDA backend requires an output buffer.")
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "CDA backend does not support fused output quantization yet.")
        if attn_metadata is None:        # profiling run
            return output.fill_(0)
        self._ensure_init(query.device)

        # Phase-4b outlier tracer: log prefill vs decode branch + duration
        # so we can see whether max_query_len > 1 at the first step of each
        # new request (which would send us down the slow _forward_prefill
        # dequantize-past path).
        import os as _os_fb
        _fb_on = _os_fb.environ.get("CDA_TRACE_FORWARD") == "1"
        if _fb_on:
            import time as _t_fb
            if not hasattr(CDAAttentionImpl, "_fb_log"):
                CDAAttentionImpl._fb_log = []
                CDAAttentionImpl._fb_count = [0]
            _fb_t0 = _t_fb.perf_counter_ns()

        # Reinterpret vLLM's fp16-typed cache as uint8 for CDA packing.
        kv_cache = _as_uint8_cache(kv_cache)

        if attn_metadata.max_query_len > 1:
            ret = self._forward_prefill(query, key, value, output,
                                            attn_metadata, kv_cache)
            _branch = "prefill"
        else:
            ret = self._forward_decode(query, kv_cache, output, attn_metadata)
            _branch = "decode"
        if _fb_on:
            import torch as _t_fb_t
            _t_fb_t.cuda.synchronize()
            _dt = (_t_fb.perf_counter_ns() - _fb_t0) / 1000.0  # μs
            CDAAttentionImpl._fb_count[0] += 1
            # Log only outliers (>5ms) to keep log small
            if _dt > 5000.0:
                CDAAttentionImpl._fb_log.append({
                    "call_idx": CDAAttentionImpl._fb_count[0],
                    "branch": _branch,
                    "max_query_len": int(attn_metadata.max_query_len),
                    "num_actual_tokens": int(attn_metadata.num_actual_tokens),
                    "num_reqs": int(attn_metadata.num_reqs),
                    "dt_us": round(_dt, 1),
                })
            # Dump every 128 logged
            if CDAAttentionImpl._fb_count[0] % 512 == 0:
                import json as _j_fb
                with open("/tmp/cda_forward_trace.json", "w") as _f_fb:
                    _j_fb.dump(CDAAttentionImpl._fb_log, _f_fb)
        return ret

    # ---- Prefill: fp32 SDPA on fresh Q/K/V (+ dequantised past) ---------
    def _forward_prefill(self, query, key, value, output, attn_metadata,
                           kv_cache):
        """fp32 SDPA on the fresh chunk. Supports both:

        * Initial prefill (query_len == seq_len): use ``key``/``value`` as-is.
        * Chunked prefill (query_len < seq_len): dequantise the already-
          written past from the paged cache and concatenate with the fresh
          fp16 K/V before SDPA.
        """
        import torch.nn.functional as F
        cu = attn_metadata.query_start_loc.tolist()
        seq_lens = attn_metadata.seq_lens.tolist()
        scale = self.scale
        H_q = self.num_heads
        D = self.head_size
        group = self.group_size

        for b in range(attn_metadata.num_reqs):
            s, e = cu[b], cu[b + 1]
            L_new = e - s
            L_full = seq_lens[b]
            past_len = L_full - L_new

            # Keep fp16 end-to-end for SDPA. Only compression (called
            # separately in do_kv_cache_update) needs fp32.
            q_b = query[s:e].transpose(0, 1)                          # (H_q, L_new, D)
            k_new = key[s:e].transpose(0, 1)                          # (H_kv, L_new, D)
            v_new = value[s:e].transpose(0, 1)

            if past_len > 0:
                # Dequantise past tokens from paged CDA cache.
                k_past, v_past = self._dequantize_past(
                    kv_cache, attn_metadata.block_table[b], past_len)
                # _dequantize_past returns fp32; cast to match fresh fp16.
                k_full = torch.cat([k_past.to(q_b.dtype), k_new], dim=1)
                v_full = torch.cat([v_past.to(q_b.dtype), v_new], dim=1)
            else:
                k_full, v_full = k_new, v_new

            if past_len == 0:
                # Initial prefill: pure lower-triangular causal mask — let
                # SDPA use its fused causal path (FA2 backend, no Python
                # loop). ``enable_gqa=True`` lets us skip the 4× K/V
                # repeat_interleave since the FA2 backend handles GQA
                # natively.
                out_b = F.scaled_dot_product_attention(
                    q_b.unsqueeze(0), k_full.unsqueeze(0), v_full.unsqueeze(0),
                    is_causal=True, scale=scale, enable_gqa=(group > 1),
                )[0].transpose(0, 1)                                   # (L_new, H_q, D)
            else:
                # Chunked prefill: row t attends to K positions 0..past_len+t.
                # ``causal_lower_right(L_new, L_full)`` encodes exactly that
                # semantic and is recognised by the FA2 backend, so we avoid
                # materialising the L_new × L_full attention matrix (the old
                # dense-mask path forced the math backend and OOM'd at
                # 11K-chunk prefill + 16-concurrent workloads).
                mask = causal_lower_right(L_new, L_full)
                out_b = F.scaled_dot_product_attention(
                    q_b.unsqueeze(0), k_full.unsqueeze(0), v_full.unsqueeze(0),
                    attn_mask=mask, scale=scale, enable_gqa=(group > 1),
                )[0].transpose(0, 1)
            _copy_out(output, s, e, out_b, H_q, D)
        return output

    # ---- Dequantise past tokens from paged cache (chunked prefill) ------
    def _dequantize_past(self, kv_cache, block_table_row, past_len):
        """Return ``(K_past, V_past)`` fp32 with shape ``(H_kv, past_len, D)``.

        Used only on the slow chunked-prefill path; kept out of ``forward``'s
        hot loop. Decompressor applies the Hadamard rotation internally so
        the returned tensors are in the **original** (pre-rotation) space.
        """
        num_blocks, block_size, H_kv, slot_width = kv_cache.shape
        D = self.head_size
        num_blk = (past_len + block_size - 1) // block_size
        req_blocks = block_table_row[:num_blk].to(torch.long)
        slot_ids = (req_blocks.unsqueeze(1) * block_size +
                     torch.arange(block_size, device=kv_cache.device)
                     ).reshape(-1)[:past_len].to(torch.long)
        flat = kv_cache.view(num_blocks * block_size, H_kv,
                               slot_width).index_select(0, slot_ids)

        # Reconstruct CompressedTensor for K and V, then dequantise.
        from core.compression import CompressedTensor
        pK_bytes = flat[:, :, _K_OFFSET:_K_OFFSET + _K_NBYTES].reshape(
            past_len * H_kv, _K_NBYTES).contiguous()
        pV_bytes = flat[:, :, _V_OFFSET:_V_OFFSET + _V_NBYTES].reshape(
            past_len * H_kv, _V_NBYTES).contiguous()
        nK = flat[:, :, _NORM_K_OFFSET:_NORM_K_OFFSET + 4].contiguous() \
                .view(torch.float32).reshape(past_len * H_kv).contiguous()
        nV = flat[:, :, _NORM_V_OFFSET:_NORM_V_OFFSET + 4].contiguous() \
                .view(torch.float32).reshape(past_len * H_kv).contiguous()

        cK = CompressedTensor(
            indices=pK_bytes, norms=nK, bit_width=4,
            shape=torch.Size([past_len * H_kv, D]),
            dtype=torch.float16, device=kv_cache.device,
            payload=None, gpu_resident=True, deflated=False,
        )
        cV = CompressedTensor(
            indices=pV_bytes, norms=nV, bit_width=2,
            shape=torch.Size([past_len * H_kv, D]),
            dtype=torch.float16, device=kv_cache.device,
            payload=None, gpu_resident=True, deflated=False,
        )
        K_flat = self._compressor_k.dequantize(cK).to(torch.float32)   # (L*H_kv, D)
        V_flat = self._compressor_v.dequantize(cV).to(torch.float32)
        # Reshape to (H_kv, L, D) — original stored order is (L, H_kv, D).
        K_past = K_flat.view(past_len, H_kv, D).transpose(0, 1).contiguous()
        V_past = V_flat.view(past_len, H_kv, D).transpose(0, 1).contiguous()
        return K_past, V_past

    # ---- Decode: unpack CDA paged cache, call kernel per request --------
    def _forward_decode(self, query, kv_cache, output, attn_metadata):
        if self._profile_kernel:
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            ret = self._forward_decode_inner(query, kv_cache, output, attn_metadata)
            e.record()
            CDAAttentionImpl._profile_events["decode"].append((s, e))
            CDAAttentionImpl._profile_dump_counter[0] += 1
            if CDAAttentionImpl._profile_dump_counter[0] % CDAAttentionImpl._profile_dump_interval == 0:
                CDAAttentionImpl._profile_dump()
            return ret
        return self._forward_decode_inner(query, kv_cache, output, attn_metadata)

    def _forward_decode_inner(self, query, kv_cache, output, attn_metadata):
        """All-requests-in-one kernel call via
        :func:`cuda_hw_attention_flash_paged_batched`. The batched paged
        kernel reads vLLM's paged KV cache directly — no Python-side
        gather / permute / contiguous() — and dispatches ``(num_splits,
        H_q, num_reqs)`` blocks, eliminating the per-request launch loop
        that dominated §27 Day 8 wall-clock.

        Output reshaped to vLLM's ``(num_tokens, num_heads, head_size)``
        layout. Uses the flash (split-K + online softmax) kernel — same
        math as §26 but fused and batched.

        Backend auto-selects between two kernel variants:

        * ``cuda_hw_attention_flash_paged_batched`` (flat) — one block per
          (split, Q-head, b). Wins at short contexts (N < 24K) where the
          per-Q-head grid minimises shared-memory pressure.
        * ``cuda_hw_attention_flash_paged_batched_coop`` (GQA-cooperative) —
          one block per (split, KV-head, b), handles all 4 Q heads in the
          same group together, sharing K/V HBM reads. Wins 1.33-1.45× at
          long contexts (N ≥ 24K) where HBM bandwidth dominates. See
          ``experiments/bench_batched_decode_kernel.py`` for the crossover.
        """
        from core.cda_attn import (
            cuda_hw_attention_flash_paged_batched,
            cuda_hw_attention_flash_paged_batched_coop,
            cuda_hw_attention_flash_paged_batched_coop_hmma,
        )

        # CPU-overhead minimisation (per FA2 backend guidance: views,
        # slices, and dtype checks are "surprisingly slow" in eager mode).
        # Assumptions enforced by vLLM 0.19 that let us skip checks:
        #   - num_actual_tokens == num_reqs in decode (vLLM invariant)
        #   - block_table and seq_lens are already int32 (verified via probe)
        #   - kv_cache shape is (num_blocks, block_size, H_kv, slot_width)
        block_table = attn_metadata.block_table
        seq_lens    = attn_metadata.seq_lens
        num_blocks, block_size, H_kv, slot_width = kv_cache.shape

        # flat_cache is a view; indexing by physical slot id directly.
        flat_cache = kv_cache.view(num_blocks * block_size, H_kv, slot_width)

        max_seq_bound = block_table.shape[1] * block_size

        # Day 14 microbench: coop wins ≥1.4× at N≥24K; flat is ~5% faster at
        # N=16K where coop's register pressure dominates. Group_size=4 is the
        # only coop variant we've written so far (Llama-3.1-8B layout).
        use_coop = (self.group_size == 4 and
                    attn_metadata.max_seq_len >= 24000)

        # Day 15 tile_N sweep (bench_tile_sweep.py):
        #   * Flat's optimum is tile_N=512 across all (B, N) we measured.
        #   * Coop benefits from LARGER tile_N at long contexts — at B=1
        #     N=32K, tile=2048 is 2.12× faster than tile=512 because the
        #     fixed per-block overhead dominates when num_splits is small
        #     already.
        if use_coop:
            # Day 18 retuning after __launch_bounds__(256, 2) doubled
            # occupancy: the fixed-overhead amortisation that favoured
            # tile=2048 pre-Day-18 is now dominated by the extra blocks a
            # smaller tile puts in flight across the now-healthier warp
            # scheduler. tile=1024 wins at B=1 N=32K by 8% (138→127 μs/layer)
            # and ties or wins at B≥4 N≥32K. tile=512 stays for short
            # contexts to avoid under-saturating.
            if max_seq_bound >= 8192:
                tile_N = 1024
            else:
                tile_N = 512
        else:
            tile_N = 1024 if max_seq_bound > 32768 else 512

        # HMMA Tensor-Core + fused-tail: Tier-2 fast path. Flags cached at
        # _ensure_init() so the hot loop avoids per-call os.environ lookups.
        use_hmma = (use_coop
                    and self.group_size == 4
                    and max_seq_bound >= 8192
                    and self._use_hmma)
        use_fused_tail = (use_hmma
                          and self._use_fused_tail
                          and output.dim() == 3)
        # L1: fold Q rotation + HMMA + reduce+rot+cast into one C++ call.
        # Requires fp16 `query` (vLLM default) and 3D output. Supersedes the
        # Tier-2 fused-tail path when enabled because it also absorbs the
        # Q_rot matmul that happens above this block.
        use_fused_decode = (use_fused_tail
                             and self._use_fused_decode
                             and query.dtype == torch.float16)

        if use_fused_decode:
            from core.cda_attn import cda_decode_full_hmma
            cda_decode_full_hmma(
                query, flat_cache, block_table, seq_lens,
                self._codebook_k, self._codebook_v,
                self._rotation, self._rotation_fp16, output,
                self.group_size, block_size, tile_N, max_seq_bound, self.scale,
            )
            return output

        # Non-fused-decode paths still need an explicit Q rotation tensor.
        # torch.matmul upcasts fp16 @ fp32 internally; avoids a separate
        # .to(fp32) on query.
        Q_rot = torch.matmul(query.to(torch.float32), self._rotation)

        if use_fused_tail:
            from core.cda_attn import (
                cda_flash_split_k4v2_gqa_paged_batched_coop_hmma,
                cda_fused_reduce_rot_cast,
            )
            partial, m_vals, l_vals = cda_flash_split_k4v2_gqa_paged_batched_coop_hmma(
                Q_rot, flat_cache, block_table, seq_lens,
                self._codebook_k, self._codebook_v, self.group_size,
                block_size, tile_N, max_seq_bound, self.scale,
            )
            cda_fused_reduce_rot_cast(
                partial, m_vals, l_vals, self._rotation_fp16, output)
            return output

        if use_hmma:
            flash_fn = cuda_hw_attention_flash_paged_batched_coop_hmma
        elif use_coop:
            flash_fn = cuda_hw_attention_flash_paged_batched_coop
        else:
            flash_fn = cuda_hw_attention_flash_paged_batched
        out_B = flash_fn(
            Q_rot, flat_cache, block_table, seq_lens,
            self._codebook_k, self._codebook_v, self._rotation,
            self.scale, self.group_size, block_size,
            max_seq_len=max_seq_bound, tile_N=tile_N,
        )                                                           # (B, H_q, D) fp32

        # Vectorised copy-out (single kernel instead of num_reqs copies).
        if output.dim() == 3:
            output.copy_(out_B.to(output.dtype))
        else:
            output.copy_(out_B.reshape(attn_metadata.num_reqs, -1).to(output.dtype))
        return output
