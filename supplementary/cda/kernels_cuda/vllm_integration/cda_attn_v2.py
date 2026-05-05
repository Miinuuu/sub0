"""CDA-v2 AttentionBackend for vLLM 0.19 (v1 API).

Production-grade integration of v1 HMMA CUDA kernels (vendored under
``cda/kernels_cuda/``) into vLLM's attention backend protocol. Mirrors
the structure of vLLM's existing ``cda_attn.py``.

Default dispatch (post `opt-toggles-cleanup`):

  - **Decode**  ← ``decode_hmma_v1`` (``_decode_hmma.cu``).
  - **Prefill** ← P34 full-HMMA only
                  (``_multiq_paged_full_hmma.cu``). No fallback path:
                  unsupported/non-uniform shapes fail loudly.
  - **Scatter** ← b4 ``paged_encode_fused_cuda`` (single CUDA kernel,
                  fused FHT + L2-quant + scatter).

Legacy CuTeDSL / Lloyd FA2-fork research kernels were moved to
``deprecated/`` (commit "B: deprecate Lloyd FA2-fork + CuTeDSL"); the
backend now exposes only the CUDA HMMA + FA2-CDA-bridge prefill paths.

K4V4 layout (slot_width = D + 8 = 136 B for D=128):
    offset  size  content
    ──────  ────  ─────────────────────────────
         0  D/2   K 4-bit packed (D=128 → 64 B)
       D/2  D/2   V 4-bit packed (D=128 → 64 B)
         D    4   norm_K fp32
       D+4    4   norm_V fp32
       D+8         (total)

vs cda-v1 K4V2 layout (104 B/slot): we trade 32 extra bytes/slot
(31% bigger) for a faster Tensor-Core MMA fp16 path (V at 4-bit fits
the same lookup pipeline as K, no separate 2-bit unpacking shader).

Supports group_size ∈ {4, 8} (Llama-3.1-8B + 70B). head_dim=128 fixed.

vLLM 0.19 imports are deferred to the class methods so this module is
import-safe even when vllm isn't installed (research environment).
"""
import math
import os
from dataclasses import dataclass
from typing import ClassVar

import torch

from cda.algorithm.compression import Compressor, CompressedSlot


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
HEAD_DIM = 128
# Raw K4V4 data: K(D/2) + V(D/2) + Knorm(4) + Vnorm(4) = D + 8 = 136 B for D=128.
# The paged decode kernel uses 128-bit cp.async, which requires per-slot
# addresses to be 16-byte aligned → slot allocation is rounded up to the
# next multiple of 16 (= 144 B for D=128). The 8 trailing bytes are padding.
# Compression vs FP16 KV (256 fp16 = 512 B): 144/512 = 28.1% (3.56× saving).
_DATA_BYTES_PER_TOKEN = HEAD_DIM + 8          # 136 — actual K4V4 payload
_BYTES_PER_TOKEN = (_DATA_BYTES_PER_TOKEN + 15) // 16 * 16   # 144 — allocated slot
_SLOT_FP16_ELEMS_COMPACT = _BYTES_PER_TOKEN // 2     # 72 — fp16 elements when memory-saving on
_SLOT_FP16_ELEMS_PER_TOKEN = 256                      # 512 B/slot — vLLM default FP16-aligned


def _as_uint8_cache(kv_cache: torch.Tensor) -> torch.Tensor:
    """Reinterpret vLLM's fp16-typed KV cache buffer as uint8 bytes so the
    cda-v2 paged adapter can read 136-byte slots directly."""
    if kv_cache.dtype == torch.uint8:
        return kv_cache
    num_blocks, block_size, num_kv_heads, last = kv_cache.shape
    if last == _SLOT_FP16_ELEMS_PER_TOKEN:
        byte_width = _SLOT_FP16_ELEMS_PER_TOKEN * 2     # 512 B/slot
    elif last == _SLOT_FP16_ELEMS_COMPACT:
        byte_width = _BYTES_PER_TOKEN                    # 136 B/slot
    else:
        raise ValueError(
            f"unexpected CDA cache last dim {last} "
            f"(expected {_SLOT_FP16_ELEMS_PER_TOKEN} or {_SLOT_FP16_ELEMS_COMPACT})")
    return kv_cache.view(torch.uint8).view(
        num_blocks, block_size, num_kv_heads, byte_width)


# ---------------------------------------------------------------------------
# Memory-saving spec patch
# ---------------------------------------------------------------------------
def enable_cda_memory_saving() -> None:
    """Monkey-patch ``AttentionSpec.real_page_size_bytes`` so vLLM accounts
    for cda-v2's smaller 136 B/slot vs default 512 B/slot.

    Idempotent. Call BEFORE constructing ``LLM`` if you want the KV memory
    savings (~3.76× vs FP16, vs ~4.92× for cda-v1 K4V2 — small regression
    in compression ratio, big improvement in kernel speed)."""
    from vllm.v1.kv_cache_interface import (
        AttentionSpec, FullAttentionSpec, ChunkedLocalAttentionSpec,
        SlidingWindowSpec,
    )
    if getattr(AttentionSpec, "_cda_v2_patched", False):
        return

    def _real_page_size_bytes(self):
        return self.block_size * self.num_kv_heads * _BYTES_PER_TOKEN

    prop = property(_real_page_size_bytes)
    for cls in (AttentionSpec, FullAttentionSpec, ChunkedLocalAttentionSpec,
                SlidingWindowSpec):
        if "real_page_size_bytes" in cls.__dict__:
            cls.real_page_size_bytes = prop
    AttentionSpec._cda_v2_patched = True


import os as _os
if _os.environ.get("CDA_V2_ENABLE_MEMORY_SAVING") == "1":
    enable_cda_memory_saving()


# ---------------------------------------------------------------------------
# Optional per-call cuda.Event timing (gated on CDA_V2_TIME_STEPS=1).
# Used to attribute the e2e TPOT gap to specific backend calls.
# Dump on shutdown via a hooked signal handler (writes to
# CDA_V2_TIME_STEPS_OUT or /tmp/cda_v2_step_timings.json).
# ---------------------------------------------------------------------------
_TIME_STEPS = _os.environ.get("CDA_V2_TIME_STEPS") == "1"
# Use prefix; dump at runtime suffixes os.getpid() so fork-inherited
# state still produces per-process files (parent vs EngineCore worker).
_TIME_STEPS_PREFIX = _os.environ.get(
    "CDA_V2_TIME_STEPS_OUT",
    "/tmp/cda_v2_step_timings",
)

# P28: dispatch toggles cached at module import — eliminates 4 per-call
# env lookups (≈ 5-10 µs each in eager) inside ``_forward_decode``.
# Toggles must be set BEFORE the backend module is imported (vLLM does
# that at engine init).
# HMMA V1 hybrid (legacy CUDA path). Opt-in via CDA_V2_USE_HMMA_V1=1 for
# bench comparison. Decode now uses the HMMA v1 decode kernel by default;
# prefill always uses the P34 full-HMMA multi-Q kernel.
_USE_HMMA_V1 = _os.environ.get("CDA_V2_USE_HMMA_V1") == "1"


def _time_steps_out():
    prefix = _os.environ.get("CDA_V2_TIME_STEPS_OUT", _TIME_STEPS_PREFIX)
    return f"{prefix}.pid{_os.getpid()}.json"


def _time_steps_enabled():
    return _TIME_STEPS or _os.environ.get("CDA_V2_TIME_STEPS") == "1"


_TIMINGS: dict = {
    "do_kv_cache_update": [],
    "_forward_decode": [],
    "_forward_prefill": [],
    "_forward_prefill_decode": [],
    "prefill_setup": [],
    "prefill_q_cast": [],
    "prefill_fa2_cda": [],
    "prefill_fa2_rawkv": [],
    "prefill_fa2_cda_rawkv": [],
    "prefill_hybrid_rawkv_chunked": [],
    "prefill_hmma": [],
    "prefill_output_copy": [],
    "decode_setup": [],
    "decode_q_cast": [],
    "decode_hmma": [],
    "decode_output_copy": [],
    # Sub-step timings inside _forward_decode (added to localize the
    # 1.21 ms / call wrapper overhead at 8K).
    "fd_decode_paged": [],         # the decode_paged wrapper call only
    "fd_output_copy": [],          # output buffer copy
}
# Periodic dump: every N total calls across all buckets, materialize
# cuda.Event pairs and write to file (overwrite). vLLM's EngineCore
# subprocess shuts down via SIGKILL → atexit doesn't fire, so we need
# to keep the file fresh.
_DUMP_EVERY = int(_os.environ.get("CDA_V2_TIME_STEPS_DUMP_EVERY", "16"))
_call_counter = [0]


def _dump_timings():
    """Materialize cuda.Event pairs to ms and dump to JSON (overwrite)."""
    import json as _json
    if not _time_steps_enabled():
        return
    if (torch.cuda.is_available()
            and hasattr(torch.cuda, "is_current_stream_capturing")
            and torch.cuda.is_current_stream_capturing()):
        return
    has_cuda_events = any(
        entry.get("events") is not None
        for evts in _TIMINGS.values()
        for entry in evts
    )
    if has_cuda_events:
        torch.cuda.synchronize()
    summary = {}
    for label, evts in _TIMINGS.items():
        if not evts:
            continue
        cpu_ms = [entry["cpu_ms"] for entry in evts
                  if entry.get("cpu_ms") is not None]
        cuda_ms = []
        for entry in evts:
            events = entry.get("events")
            if events is None:
                continue
            start, end = events
            cuda_ms.append(start.elapsed_time(end))
        n = len(evts)
        # Drop first 2 calls (cold-start: JIT compile + CG capture warmup).
        warm_cpu = cpu_ms[2:] if len(cpu_ms) > 2 else cpu_ms
        warm_cuda = cuda_ms[2:] if len(cuda_ms) > 2 else cuda_ms
        rec = {
            "n_total": n,
            "n_warm": len(warm_cpu),
        }
        if warm_cpu:
            rec.update({
                "mean_cpu_ms": sum(warm_cpu) / len(warm_cpu),
                "min_cpu_ms": min(warm_cpu),
                "max_cpu_ms": max(warm_cpu),
                "sum_cpu_ms": sum(warm_cpu),
                "p50_cpu_ms": sorted(warm_cpu)[len(warm_cpu) // 2],
            })
        if warm_cuda:
            rec.update({
                "mean_cuda_ms": sum(warm_cuda) / len(warm_cuda),
                "min_cuda_ms": min(warm_cuda),
                "max_cuda_ms": max(warm_cuda),
                "sum_cuda_ms": sum(warm_cuda),
                "p50_cuda_ms": sorted(warm_cuda)[len(warm_cuda) // 2],
            })
        summary[label] = rec
    path = _time_steps_out()
    parent = _os.path.dirname(path)
    if parent:
        _os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        _json.dump({
            "pid": _os.getpid(),
            "call_count": _call_counter[0],
            "summary": summary,
        }, f, indent=2)


def _time_call(label: str, fn):
    """Run fn() with CPU wall timing and optional cuda.Event timing."""
    if not _time_steps_enabled():
        return fn()
    import time as _time
    use_events = torch.cuda.is_available()
    if (use_events and hasattr(torch.cuda, "is_current_stream_capturing")
            and torch.cuda.is_current_stream_capturing()):
        use_events = False
    start = end = None
    if use_events:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    t0 = _time.perf_counter()
    try:
        out = fn()
    finally:
        cpu_ms = (_time.perf_counter() - t0) * 1e3
        if use_events:
            end.record()
        _TIMINGS.setdefault(label, []).append({
            "cpu_ms": cpu_ms,
            "events": (start, end) if use_events else None,
        })
        _call_counter[0] += 1
        if _DUMP_EVERY > 0 and _call_counter[0] % _DUMP_EVERY == 0:
            _dump_timings()
    return out


import atexit
atexit.register(_dump_timings)


# ---------------------------------------------------------------------------
# Compressor cache (one pair per device, lazily built)
# ---------------------------------------------------------------------------
_COMPRESSOR_CACHE: dict = {}
# Per-device cache of HMMA V1 wrapper artifacts (cb_K, cb_V, rotation_fp32,
# rotation_fp16). Avoids the .float()/.contiguous()/.to(fp16) chain on
# every layer's forward — these tensors are immutable across calls.
_HMMA_ARTIFACTS_CACHE: dict = {}


def _get_compressors(device: torch.device) -> tuple[Compressor, Compressor]:
    key = (device.type, getattr(device, "index", None))
    if key in _COMPRESSOR_CACHE:
        return _COMPRESSOR_CACHE[key]
    cmp_K = Compressor(HEAD_DIM, num_levels=16, device=device)
    cmp_V = Compressor(HEAD_DIM, num_levels=16, device=device)
    _COMPRESSOR_CACHE[key] = (cmp_K, cmp_V)
    return cmp_K, cmp_V


def _get_hmma_artifacts(cmp_K: Compressor, cmp_V: Compressor, device):
    """Cache cb_K, cb_V, rotation_fp32, rotation_fp16 for the HMMA V1
    wrapper. Keyed by ``(id(cmp_K), id(cmp_V), device)`` so a new
    Compressor (e.g. fresh test impl) produces fresh artifacts.

    Avoids the per-call .float()/.contiguous()/.to(fp16) chain when the
    compressor is stable (the typical vLLM serving case).
    """
    key = (id(cmp_K), id(cmp_V), device.type, getattr(device, "index", None))
    cached = _HMMA_ARTIFACTS_CACHE.get(key)
    if cached is not None:
        return cached
    cb_K = cmp_K.codebook.float().contiguous()
    cb_V = cmp_V.codebook.float().contiguous()
    rotation_fp32 = cmp_K.rotation.float().contiguous()
    rotation_fp16 = rotation_fp32.to(torch.float16).contiguous()
    artifacts = (cb_K, cb_V, rotation_fp32, rotation_fp16)
    _HMMA_ARTIFACTS_CACHE[key] = artifacts
    return artifacts


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
@dataclass
class CDAv2AttentionMetadata:
    num_actual_tokens: int
    num_reqs: int
    max_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool = True
    # P29: DCP fields (default 1 = single-GPU, no DCP).
    dcp_world_size: int = 1
    dcp_rank: int = 0
    # vLLM chunked-prefill may mix decode requests and prefill/extend
    # requests in one forward. Reordered batches are decode first,
    # prefill second.
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_decodes: int = 0


def _split_counts_from_query_start(
    query_start_loc, num_reqs: int, num_actual_tokens: int, max_query_len: int,
) -> tuple[int, int, int, int]:
    """Return vLLM-style decode/prefill split counts for reordered batches."""
    if max_query_len <= 1:
        return num_reqs, 0, num_actual_tokens, 0
    qs = query_start_loc
    if isinstance(qs, torch.Tensor):
        qs = qs.tolist()
    lens = [int(qs[i + 1]) - int(qs[i]) for i in range(num_reqs)]
    first_prefill = 0
    while first_prefill < num_reqs and lens[first_prefill] <= 1:
        first_prefill += 1
    num_decodes = first_prefill
    num_prefills = num_reqs - num_decodes
    num_decode_tokens = int(qs[first_prefill]) if num_decodes else 0
    num_prefill_tokens = num_actual_tokens - num_decode_tokens
    return num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens


# ---------------------------------------------------------------------------
# AttentionBackend / AttentionImpl shells (vLLM imports deferred)
# ---------------------------------------------------------------------------
def _build_backend_classes():
    """Lazy: construct the vLLM-AttentionBackend-shaped classes only when
    vLLM is actually present. Returns ``(Backend, Impl, Builder, Metadata)``.

    Letting the classes be built lazily means ``cda.vllm_integration.cda_attn_v2``
    can be imported even outside a vLLM environment (e.g. unit tests against
    the adapter alone)."""
    try:
        from vllm.v1.attention.backend import (
            AttentionBackend, AttentionCGSupport, AttentionImpl,
            AttentionMetadata, AttentionMetadataBuilder, AttentionType,
            CommonAttentionMetadata, MultipleOf,
        )
        from vllm.v1.kv_cache_interface import AttentionSpec
        from vllm.config import VllmConfig
    except ModuleNotFoundError:
        class _GenericBase:
            def __init__(self, *args, **kwargs):
                pass

            def __class_getitem__(cls, item):
                return cls

        class AttentionBackend:
            pass

        class AttentionImpl(_GenericBase):
            pass

        class AttentionMetadata:
            pass

        class AttentionMetadataBuilder(_GenericBase):
            pass

        class AttentionType:
            DECODER = "decoder"

        class AttentionCGSupport:
            UNIFORM_BATCH = "uniform_batch"

        class CommonAttentionMetadata:
            pass

        class MultipleOf(int):
            pass

        class AttentionSpec:
            pass

        class VllmConfig:
            pass

    @dataclass
    class CDAv2AttentionMetadataVLLM(AttentionMetadata):
        num_actual_tokens: int
        num_reqs: int
        max_query_len: int
        max_seq_len: int
        query_start_loc: torch.Tensor
        seq_lens: torch.Tensor
        block_table: torch.Tensor
        slot_mapping: torch.Tensor
        causal: bool = True
        # P29: DCP (Decode Context Parallelism). DCP=1 is the default
        # single-GPU case (no parallelism). DCP>1 shards the KV cache
        # across decode-context ranks; each rank gets only its local
        # slice of seq_lens / block_table / kv_cache. Llama-70B B=1 will
        # likely use DCP=2 or 4.
        dcp_world_size: int = 1
        dcp_rank: int = 0
        num_prefill_tokens: int = 0
        num_decode_tokens: int = 0
        num_prefills: int = 0
        num_decodes: int = 0

    class CDAv2AttentionMetadataBuilder(
        AttentionMetadataBuilder[CDAv2AttentionMetadataVLLM]
    ):
        # UNIFORM_BATCH: decode path is CG-safe, and the prefill forward
        # no longer performs per-layer GPU→CPU metadata syncs. vLLM still
        # avoids capturing prefill in piecewise CG, but the runtime path
        # now mirrors FA varlen metadata flow more closely.
        # NB: gpu_model_runner reads via get_cudagraph_support() classmethod
        # (vllm/v1/worker/gpu_model_runner.py:6270). The ClassVar alone is
        # not enough — must override the classmethod too (FA pattern,
        # flash_attn.py:267-280).
        _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
        # Request vLLM's decode -> short_extend -> long_extend -> prefill
        # ordering so split_decodes_and_prefills() can find a single boundary.
        reorder_batch_threshold: int | None = 1

        @classmethod
        def get_cudagraph_support(
            cls, vllm_config, kv_cache_spec,
        ) -> AttentionCGSupport:
            return cls._cudagraph_support

        def __init__(
            self,
            kv_cache_spec: AttentionSpec,
            layer_names: list[str],
            vllm_config: VllmConfig,
            device: torch.device,
        ) -> None:
            super().__init__(kv_cache_spec, layer_names, vllm_config, device)
            # P29: DCP (Decode Context Parallelism) infrastructure for
            # Llama-3.1-70B. DCP shards KV cache across decode-context
            # ranks; each rank computes partial attention over its KV
            # shard, then partials are reduced via online-softmax merge.
            # Mirrors FA backend's pattern (flash_attn.py:307-315).
            try:
                from vllm.distributed.parallel_state import get_dcp_group
                self.dcp_world_size = get_dcp_group().world_size
                self.dcp_rank = get_dcp_group().rank_in_group
            except (AssertionError, ImportError):
                # DCP not initialized (single-GPU or DCP=1) — default 1.
                self.dcp_world_size = 1
                self.dcp_rank = 0

        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ) -> CDAv2AttentionMetadataVLLM:
            c = common_attn_metadata
            try:
                from vllm.v1.attention.backends.utils import (
                    split_decodes_and_prefills,
                )
                num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                    split_decodes_and_prefills(c, decode_threshold=1)
                )
            except Exception:
                qsl = getattr(c, "query_start_loc_cpu", c.query_start_loc)
                num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                    _split_counts_from_query_start(
                        qsl, c.num_reqs, c.num_actual_tokens, c.max_query_len,
                    )
                )
            return CDAv2AttentionMetadataVLLM(
                num_actual_tokens=c.num_actual_tokens,
                num_reqs=c.num_reqs,
                max_query_len=c.max_query_len,
                max_seq_len=c.max_seq_len,
                query_start_loc=c.query_start_loc,
                seq_lens=c.seq_lens,
                block_table=c.block_table_tensor,
                slot_mapping=c.slot_mapping,
                causal=c.causal,
                dcp_world_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                num_prefill_tokens=num_prefill_tokens,
                num_decode_tokens=num_decode_tokens,
                num_prefills=num_prefills,
                num_decodes=num_decodes,
            )

    class CDAv2AttentionBackend(AttentionBackend):
        accept_output_buffer: bool = True
        supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16]
        supported_kv_cache_dtypes: ClassVar[list[str]] = ["auto", "uint8"]
        forward_includes_kv_cache_update: bool = False

        @staticmethod
        def get_supported_kernel_block_sizes() -> list:
            # cda-v2 decode kernel works with any block_size that's a
            # multiple of 16 (matches our N_BLOCK=32 alignment).
            return [MultipleOf(16)]

        @staticmethod
        def get_name() -> str:
            # vLLM does AttentionBackendEnum[get_name()] for the reverse
            # lookup that resolves the enum member from a constructed
            # backend instance. We override the existing ``CDA`` slot so
            # the name must match — the v1 CDA backend was previously at
            # this slot; cda-v2 takes its place via register_backend("CDA").
            return "CDA"

        @staticmethod
        def get_impl_cls() -> type:
            return CDAv2AttentionImpl

        @staticmethod
        def get_builder_cls() -> type:
            return CDAv2AttentionMetadataBuilder

        @staticmethod
        def get_kv_cache_shape(
            num_blocks: int,
            block_size: int,
            num_kv_heads: int,
            head_size: int,
            cache_dtype_str: str = "auto",
        ) -> tuple[int, ...]:
            if head_size != HEAD_DIM:
                raise ValueError(
                    f"CDA-v2 backend supports head_size={HEAD_DIM} only. "
                    f"Got head_size={head_size}.")
            # Always use compact 144B/slot. Earlier this conditioned on
            # `_cda_v2_patched` flag, but in environments where the
            # legacy `core.vllm_integration` module also loads it patches
            # AttentionSpec with `_cda_patched` (DIFFERENT flag, K4V2
            # ratio) — our flag check then fails and we'd return FP16
            # default 256, causing vLLM allocator shape mismatch.
            # We're the CDA backend; force-enable compact + return compact.
            if not getattr(AttentionSpec, "_cda_v2_patched", False):
                enable_cda_memory_saving()
            return (num_blocks, block_size, num_kv_heads, _SLOT_FP16_ELEMS_COMPACT)

        @classmethod
        def get_supported_head_sizes(cls) -> list[int]:
            return [HEAD_DIM]

        @staticmethod
        def swap_blocks(
            src_kv_cache: torch.Tensor,
            dst_kv_cache: torch.Tensor,
            src_to_dst: torch.Tensor,
        ) -> None:
            dst_kv_cache[src_to_dst[:, 1]] = src_kv_cache[src_to_dst[:, 0]]

        @staticmethod
        def copy_blocks(
            kv_caches: list[torch.Tensor],
            src_to_dists: torch.Tensor,
        ) -> None:
            for kv_cache in kv_caches:
                kv_cache[src_to_dists[:, 1]] = kv_cache[src_to_dists[:, 0]]

    class CDAv2AttentionImpl(AttentionImpl[CDAv2AttentionMetadataVLLM]):
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
                raise NotImplementedError("CDA-v2 backend does not support ALiBi.")
            if sliding_window is not None:
                raise NotImplementedError(
                    "CDA-v2 backend does not support sliding window.")
            if attn_type not in (AttentionType.DECODER,):
                raise NotImplementedError(
                    f"CDA-v2 backend is DECODER-only, got {attn_type}.")
            if self.group_size not in (4, 8):
                raise NotImplementedError(
                    f"CDA-v2 supports group_size ∈ {{4, 8}} (Llama-3.1-{{8B,70B}}). "
                    f"Got group_size={self.group_size}.")

        def do_kv_cache_update(
            self,
            layer: torch.nn.Module,
            key: torch.Tensor,        # (num_new_tokens, num_kv_heads, D) fp16
            value: torch.Tensor,
            kv_cache: torch.Tensor,
            slot_mapping: torch.Tensor,
        ) -> None:
            if kv_cache.numel() == 0:
                return
            def _do():
                kv_cache_u8 = _as_uint8_cache(kv_cache)
                num_blocks, block_size, num_kv_heads, slot_w = kv_cache_u8.shape
                flat_cache = kv_cache_u8.view(num_blocks * block_size,
                                                num_kv_heads, slot_w)
                cmp_K, cmp_V = _get_compressors(kv_cache_u8.device)
                # (b4-r2) Single-CUDA-kernel fused FHT + L2-quantize +
                # scatter encode. fp16 SMEM butterfly + fp32 norm/codebook.
                from cda.kernels_cuda.wrappers import paged_encode_fused_cuda
                paged_encode_fused_cuda(
                    key, value, slot_mapping, flat_cache,
                    cb_K=cmp_K.codebook, cb_V=cmp_V.codebook,
                )
            _time_call("do_kv_cache_update", _do)

        def forward(
            self,
            layer: torch.nn.Module,
            query: torch.Tensor,           # (num_actual_tokens, H_q, D) fp16
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: CDAv2AttentionMetadataVLLM,
            output: torch.Tensor | None = None,
            output_scale: torch.Tensor | None = None,
            output_block_scale: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if output is None:
                raise ValueError("CDA-v2 backend requires an output buffer.")
            if output_scale is not None or output_block_scale is not None:
                raise NotImplementedError(
                    "CDA-v2 backend does not support fused output quantization.")
            if attn_metadata is None:
                return output.fill_(0)
            kv_cache = _as_uint8_cache(kv_cache)

            # P29: DCP=1 → fast path (current). DCP>1 → all-rank online-
            # softmax merge via dcp_combine. Llama-70B B=1 will need
            # DCP=2 or 4 for KV cache fitting + decode parallelism.
            if attn_metadata.dcp_world_size > 1:
                return _time_call(
                    "_forward_dcp",
                    lambda: self._forward_decode_dcp(
                        query, kv_cache, output, attn_metadata),
                )

            if attn_metadata.max_query_len > 1:
                num_decodes = getattr(attn_metadata, "num_decodes", 0)
                num_prefills = getattr(attn_metadata, "num_prefills", 0)
                num_decode_tokens = getattr(
                    attn_metadata, "num_decode_tokens", 0)
                num_prefill_tokens = getattr(
                    attn_metadata, "num_prefill_tokens", 0)
                if (num_decodes + num_prefills != attn_metadata.num_reqs
                        or num_decode_tokens + num_prefill_tokens
                        != attn_metadata.num_actual_tokens):
                    (num_decodes, num_prefills, num_decode_tokens,
                     num_prefill_tokens) = _split_counts_from_query_start(
                        attn_metadata.query_start_loc,
                        attn_metadata.num_reqs,
                        attn_metadata.num_actual_tokens,
                        attn_metadata.max_query_len,
                    )
                    attn_metadata.num_decodes = num_decodes
                    attn_metadata.num_prefills = num_prefills
                    attn_metadata.num_decode_tokens = num_decode_tokens
                    attn_metadata.num_prefill_tokens = num_prefill_tokens
                if num_decode_tokens > 0 and num_prefill_tokens > 0:
                    return _time_call(
                        "_forward_prefill_decode",
                        lambda: self._forward_mixed_prefill_decode(
                            query, key, value, output, attn_metadata,
                            kv_cache),
                    )
                return _time_call(
                    "_forward_prefill",
                    lambda: self._forward_prefill(
                        query, key, value, output, attn_metadata, kv_cache),
                )
            return _time_call(
                "_forward_decode",
                lambda: self._forward_decode(
                    query, kv_cache, output, attn_metadata),
            )

        # Prefill is intentionally single-route: P34 full-compressed HMMA.
        # Fallback kernels are disabled so regressions surface as explicit
        # errors instead of silently changing the measured implementation.

        def _forward_mixed_prefill_decode(
            self, query, key, value, output, attn_metadata, kv_cache,
        ):
            """Chunked-prefill mixed batch: decode first, prefill second.

            vLLM reorders such batches as decode/short-extend requests
            followed by long prefill chunks. Keep both paths fully
            compressed: L=1 rows use `_decode_hmma.cu`; L>1 rows use
            `_multiq_paged_full_hmma.cu`.
            """
            nd = attn_metadata.num_decodes
            np = attn_metadata.num_prefills
            nd_tok = attn_metadata.num_decode_tokens
            np_tok = attn_metadata.num_prefill_tokens
            if nd_tok <= 0 or np_tok <= 0:
                raise ValueError("mixed prefill/decode requires both phases")
            if key is None or value is None:
                raise ValueError("mixed prefill/decode requires key/value")

            decode_out = output[:nd_tok]
            prefill_out = output[nd_tok:nd_tok + np_tok]

            dec_seq = attn_metadata.seq_lens[:nd]
            dec_md = CDAv2AttentionMetadata(
                num_actual_tokens=nd_tok,
                num_reqs=nd,
                max_query_len=1,
                max_seq_len=attn_metadata.max_seq_len,
                query_start_loc=attn_metadata.query_start_loc[:nd + 1],
                seq_lens=dec_seq,
                block_table=attn_metadata.block_table[:nd],
                slot_mapping=attn_metadata.slot_mapping[:nd_tok],
                causal=attn_metadata.causal,
                dcp_world_size=1,
                dcp_rank=0,
                num_decode_tokens=nd_tok,
                num_decodes=nd,
            )
            self._forward_decode(
                query[:nd_tok], kv_cache, decode_out, dec_md)

            pre_cu = attn_metadata.query_start_loc[nd:] - nd_tok
            pre_seq = attn_metadata.seq_lens[nd:nd + np]
            pre_md = CDAv2AttentionMetadata(
                num_actual_tokens=np_tok,
                num_reqs=np,
                max_query_len=attn_metadata.max_query_len,
                max_seq_len=attn_metadata.max_seq_len,
                query_start_loc=pre_cu,
                seq_lens=pre_seq,
                block_table=attn_metadata.block_table[nd:nd + np],
                slot_mapping=attn_metadata.slot_mapping[
                    nd_tok:nd_tok + np_tok],
                causal=attn_metadata.causal,
                dcp_world_size=1,
                dcp_rank=0,
                num_prefill_tokens=np_tok,
                num_prefills=np,
            )
            self._forward_prefill(
                query[nd_tok:nd_tok + np_tok],
                key[nd_tok:nd_tok + np_tok],
                value[nd_tok:nd_tok + np_tok],
                prefill_out,
                pre_md,
                kv_cache,
            )
            return output

        def _forward_prefill(
            self, query, key, value, output, attn_metadata, kv_cache,
        ):
            """Prefill dispatch: full-compressed P34 HMMA only.

            No fallback is allowed here. vLLM should schedule uniform
            chunked-prefill rows for this backend; if that contract is
            violated, fail loudly instead of silently switching kernels.
            """
            num_reqs = attn_metadata.num_reqs
            group = self.group_size
            first_L = int(attn_metadata.max_query_len)
            prefill_backend = os.getenv("CDA_PREFILL_BACKEND", "fa2_cda_fused").lower()
            uniform_prefill = (
                attn_metadata.num_actual_tokens == num_reqs * first_L)

            # Uniform query length check without device→host sync. Since
            # max_query_len is the per-batch maximum, T == B*max_query_len
            # implies every request has that same query length.
            if (not uniform_prefill
                    and prefill_backend not in (
                        "fa2_cda", "fa2_cda_fused", "fa2_cda_auto",
                        "fa2_cda_direct_v2",
                        "fa2_rawkv", "fa2_cda_rawkv", "fa2_cda_fused_rawkv",
                    )):
                raise NotImplementedError(
                    "CDA-v2 prefill requires uniform chunked-prefill shapes; "
                    "fallback paths are disabled.")
            if first_L <= 0:
                raise NotImplementedError(
                    "CDA-v2 prefill received an empty prefill chunk; "
                    "fallback paths are disabled.")
            if group not in (4, 8):
                raise NotImplementedError(
                    f"CDA-v2 P34 prefill supports group_size 4 or 8, got {group}.")

            return self._forward_prefill_full_hmma_v1(
                query, key, value, output, attn_metadata, kv_cache,
                # Negative sentinel: CUDA derives per-request past_len as
                # seq_lens[b] - L_chunk, matching vLLM varlen metadata.
                L_chunk=first_L, past_len=-1,
            )

        def _p34_setup(self, query, kv_cache):
            """Common setup for CUDA HMMA compressed-attention paths.

            Prefill and decode use different kernels but share the same
            compressor artifacts plus flattened paged cache. Returns
            ``(cb_K, cb_V, rotation_fp32, rotation_fp16, kv_cache_flat,
            block_size)``. ``kv_cache`` is assumed to be 4D
            ``(num_blocks, block_size, H_kv, slot_w)`` (vLLM paged layout).
            """
            num_blocks, block_size, H_kv, slot_w = kv_cache.shape
            cmp_K, cmp_V = _get_compressors(query.device)
            cb_K, cb_V, rotation_fp32, rotation_fp16 = (
                _get_hmma_artifacts(cmp_K, cmp_V, query.device))
            kv_cache_flat = kv_cache.view(num_blocks * block_size, H_kv, slot_w)
            return (cb_K, cb_V, rotation_fp32, rotation_fp16,
                     kv_cache_flat, block_size)

        def _call_p34(self, q_in, kv_cache_flat, attn_metadata, out_buf,
                       kv_offset, *,
                       cb_K, cb_V, rotation_fp32, rotation_fp16, block_size):
            """Single P34 prefill wrapper invocation.

            ``q_in`` is (B, L_new, H_q, D). ``kv_offset`` is ``past_len``
            for causal chunked prefill, so query row q sees keys
            ``[0, past_len + q]``.
            """
            from cda.kernels_cuda.wrappers import multiq_paged_full_hmma_v1
            multiq_paged_full_hmma_v1(
                q_in, kv_cache_flat,
                attn_metadata.block_table, attn_metadata.seq_lens,
                out_buf,
                cb_K=cb_K, cb_V=cb_V,
                rotation_fp32=rotation_fp32, rotation_fp16=rotation_fp16,
                group_size=self.group_size, block_size=block_size,
                scale=self.scale, max_seq_len=attn_metadata.max_seq_len,
                kv_offset=kv_offset,
            )

        def _call_p34_varlen(self, q_flat, kv_cache_flat, attn_metadata,
                              out_flat, kv_offset, *, cb_K, cb_V,
                              rotation_fp32, rotation_fp16, block_size,
                              max_query_len):
            """vLLM-varlen style P34 prefill invocation.

            ``q_flat`` and ``out_flat`` are (T, H_q, D).  Metadata stays in
            the same form as vLLM's FlashAttention varlen path:
            ``query_start_loc``, ``seq_lens`` and ``block_table``.  A
            negative ``kv_offset`` asks CUDA to derive per-request causal
            offsets from ``seq_lens[b] - max_query_len``.
            """
            from cda.kernels_cuda.wrappers import multiq_paged_full_hmma_varlen_v1
            multiq_paged_full_hmma_varlen_v1(
                q_flat, kv_cache_flat,
                attn_metadata.block_table,
                attn_metadata.query_start_loc,
                attn_metadata.seq_lens,
                out_flat,
                cb_K=cb_K, cb_V=cb_V,
                rotation_fp32=rotation_fp32, rotation_fp16=rotation_fp16,
                group_size=self.group_size, block_size=block_size,
                max_query_len=max_query_len,
                max_seq_len=attn_metadata.max_seq_len,
                scale=self.scale,
                kv_offset=kv_offset,
            )

        def _call_fa2_cda_varlen(self, q_flat, kv_cache_flat, attn_metadata,
                                  out_flat, *, cb_K, cb_V, rotation_fp32,
                                  block_size, max_query_len, direct=False,
                                  fuse_epilogue=False, direct_split_k=1,
                                  direct_v2=False):
            """FA2-csrc skeleton path with a CDA paged-cache loader bridge.

            This is the rewrite scaffold: CDA K/V are loaded from compressed
            pages into dense rotated fp16 K/V, then FA2 varlen forward handles
            online softmax and tensor-core scheduling.  It is opt-in via
            ``CDA_PREFILL_BACKEND=fa2_cda`` while the loader still materializes
            dense scratch.
            """
            if direct:
                if direct_v2:
                    from cda.kernels_cuda.fa2_cda import fa2_cda_prefill_direct_varlen_v2 as _fa2_cda
                else:
                    from cda.kernels_cuda.fa2_cda import fa2_cda_prefill_direct_varlen_v1 as _fa2_cda
            else:
                from cda.kernels_cuda.fa2_cda import fa2_cda_prefill_varlen_v1 as _fa2_cda
            kwargs = {"fuse_fa2_epilogue": True} if fuse_epilogue else {}
            if direct:
                kwargs["direct_split_k"] = direct_split_k
                kwargs["direct_inline_meta"] = (
                    os.getenv("CDA_DIRECT_INLINE_META", "1") == "1"
                )
            _fa2_cda(
                q_flat,
                kv_cache_flat,
                attn_metadata.block_table,
                attn_metadata.query_start_loc,
                attn_metadata.seq_lens,
                out_flat,
                cb_K=cb_K,
                cb_V=cb_V,
                rotation_fp32=rotation_fp32,
                group_size=self.group_size,
                block_size=block_size,
                max_query_len=max_query_len,
                max_seq_len=attn_metadata.max_seq_len,
                scale=self.scale,
                causal=True,
                **kwargs,
            )

        def _call_fa2_cda_rawkv_varlen(
            self, q_flat, key_flat, value_flat, attn_metadata, out_flat,
            *, rotation_fp32, max_query_len,
        ):
            """FA2-CDA fused path fed from fresh fp16 K/V instead of cache.

            This is for no-past/full prefill only. Chunked/extend prefill has
            a compressed prefix that is not present in ``key_flat``; those
            batches route to ``_call_hybrid_rawkv_chunked``.
            """
            from cda.kernels_cuda.fa2_cda import fa2_cda_prefill_rawkv_varlen_v1
            T = q_flat.shape[0]
            fa2_cda_prefill_rawkv_varlen_v1(
                q_flat,
                key_flat[:T],
                value_flat[:T],
                attn_metadata.query_start_loc,
                attn_metadata.seq_lens,
                out_flat,
                rotation_fp32=rotation_fp32,
                group_size=self.group_size,
                max_query_len=max_query_len,
                max_seq_len=attn_metadata.max_seq_len,
                scale=self.scale,
                causal=True,
            )

        def _call_fa2_rawkv_varlen(
            self, q_flat, key_flat, value_flat, attn_metadata, out_flat,
            *, max_query_len,
        ):
            """No-past prefill through standard FA2 over raw Q/K/V."""
            from cda.kernels_cuda.fa2_cda import fa2_raw_prefill_varlen_v1
            T = q_flat.shape[0]
            fa2_raw_prefill_varlen_v1(
                q_flat,
                key_flat[:T],
                value_flat[:T],
                attn_metadata.query_start_loc,
                out_flat,
                group_size=self.group_size,
                max_query_len=max_query_len,
                scale=self.scale,
                causal=True,
            )

        def _call_hybrid_rawkv_chunked(
            self, q_flat, key_flat, value_flat, kv_cache_flat, attn_metadata,
            out_flat, *, cb_K, cb_V, rotation_fp32, rotation_fp16, block_size,
        ):
            """Chunked prefill: raw current chunk + compressed paged prefix.

            vLLM supplies only the current chunk in ``key``/``value`` during
            chunked prefill, while the prefix is already in the paged cache.
            This path mirrors the vLLM split: FA2 handles fresh K/V for the
            current chunk, and CDA HMMA handles the compressed prefix with an
            online-softmax LSE merge.
            """
            from cda.kernels_cuda.wrappers import hybrid_attn_v1
            T = q_flat.shape[0]
            q_in = q_flat if q_flat.dtype == torch.float16 else q_flat.to(torch.float16)
            k_in = key_flat[:T]
            v_in = value_flat[:T]
            if k_in.dtype != torch.float16:
                k_in = k_in.to(torch.float16)
            if v_in.dtype != torch.float16:
                v_in = v_in.to(torch.float16)
            hybrid_attn_v1(
                q_in,
                k_in,
                v_in,
                out_flat,
                kv_cache_flat,
                attn_metadata.block_table,
                cu_seqlens_q=attn_metadata.query_start_loc,
                seq_lens=attn_metadata.seq_lens,
                cb_K=cb_K,
                cb_V=cb_V,
                rotation_fp32=rotation_fp32,
                rotation_fp16=rotation_fp16,
                scale=self.scale,
                group_size=self.group_size,
                block_size=block_size,
            )

        def _call_decode_hmma_v1(
            self, q_in, kv_cache_flat, attn_metadata, out_buf, *,
            cb_K, cb_V, rotation_fp32, rotation_fp16, block_size,
        ):
            """Single-token full-compressed decode invocation.

            Always uses ``decode_hmma_v1`` (single-Q HMMA). The 5-way
            comparison (B=1) showed multiq_paged_full_hmma_v1 only wins
            past N=64K, but at the e2e dispatch level (B>1, batched
            decode) the multi-Q kernel was empirically slower at N=128K
            B=2 — bench runs/bench_cda_vs_fa2_128k_v2.json showed TPOT
            p50 60ms / p95 329ms after the swap, vs decode_hmma_v1's
            61ms / 339ms before. multi-Q's win regime needs its own
            B-aware bench before we re-enable the N-based switch.
            """
            from cda.kernels_cuda.wrappers import decode_hmma_v1
            decode_hmma_v1(
                q_in, kv_cache_flat,
                attn_metadata.block_table, attn_metadata.seq_lens,
                out_buf,
                cb_K=cb_K, cb_V=cb_V,
                rotation_fp32=rotation_fp32,
                rotation_fp16=rotation_fp16,
                group_size=self.group_size,
                block_size=block_size,
                scale=self.scale,
                max_seq_len=attn_metadata.max_seq_len,
            )

        def _forward_prefill_full_hmma_v1(
            self, query, key, value, output, attn_metadata, kv_cache,
            *, L_chunk: int, past_len: int,
        ):
            """**P34 default prefill** — full-compressed multi-Q paged
            HMMA (CUDA). This is the L>1 prefill-only full-compressed
            path. Negative ``past_len`` means the kernel derives the
            per-request causal offset from ``seq_lens[b] - L_chunk``.
            """
            num_reqs = attn_metadata.num_reqs
            H_q = self.num_heads
            D = self.head_size
            T = int(attn_metadata.num_actual_tokens)

            cb_K, cb_V, rot_fp32, rot_fp16, kv_cache_flat, block_size = (
                _time_call(
                    "prefill_setup",
                    lambda: self._p34_setup(query, kv_cache)))

            q_flat = query[:T]
            if q_flat.dtype != torch.float16:
                q_flat = _time_call(
                    "prefill_q_cast",
                    lambda: q_flat.to(torch.float16))

            out_view = None
            if (output.dtype == torch.float16 and output.is_contiguous()
                    and output.shape[0] >= T):
                if (output.dim() == 3
                        and output.shape[1] == H_q
                        and output.shape[2] == D):
                    out_view = output[:T]
                elif (output.dim() == 2
                        and output.shape[1] == H_q * D):
                    out_view = output[:T].view(T, H_q, D)

            if out_view is None:
                out_view = torch.empty(
                    T, H_q, D, dtype=torch.float16, device=query.device)

            prefill_backend = os.getenv("CDA_PREFILL_BACKEND", "fa2_cda_fused").lower()
            uniform_prefill = (T == num_reqs * L_chunk)
            auto_direct = (
                prefill_backend == "fa2_cda_auto"
                and uniform_prefill
                and L_chunk <= int(os.getenv("CDA_DIRECT_SPLIT_MAX_Q", "64"))
                and int(attn_metadata.max_seq_len)
                >= int(os.getenv("CDA_DIRECT_SPLIT_MIN_K", "4096"))
            )
            if prefill_backend in (
                "fa2_cda", "fa2_cda_fused", "fa2_cda_direct",
                "fa2_cda_direct_v2", "fa2_cda_auto"
            ):
                _time_call(
                    "prefill_fa2_cda",
                    lambda: self._call_fa2_cda_varlen(
                        q_flat, kv_cache_flat, attn_metadata, out_view,
                        cb_K=cb_K, cb_V=cb_V, rotation_fp32=rot_fp32,
                        block_size=block_size, max_query_len=L_chunk,
                        direct=(
                            prefill_backend in (
                                "fa2_cda_direct", "fa2_cda_direct_v2"
                            )
                            or auto_direct
                        ),
                        direct_v2=(prefill_backend == "fa2_cda_direct_v2"),
                        fuse_epilogue=(
                            prefill_backend in ("fa2_cda_fused", "fa2_cda_auto")
                            and not auto_direct
                        ),
                        direct_split_k=0 if auto_direct else 1,
                    ),
                )
            elif prefill_backend in (
                "fa2_rawkv", "fa2_cda_rawkv", "fa2_cda_fused_rawkv"
            ):
                if key is None or value is None:
                    raise ValueError(
                        f"{prefill_backend} requires fresh prefill key/value tensors"
                    )
                key_flat = key[:T]
                value_flat = value[:T]
                if int(attn_metadata.max_seq_len) > int(L_chunk):
                    _time_call(
                        "prefill_hybrid_rawkv_chunked",
                        lambda: self._call_hybrid_rawkv_chunked(
                            q_flat, key_flat, value_flat, kv_cache_flat,
                            attn_metadata, out_view,
                            cb_K=cb_K, cb_V=cb_V,
                            rotation_fp32=rot_fp32,
                            rotation_fp16=rot_fp16,
                            block_size=block_size,
                        ),
                    )
                else:
                    if prefill_backend == "fa2_rawkv":
                        _time_call(
                            "prefill_fa2_rawkv",
                            lambda: self._call_fa2_rawkv_varlen(
                                q_flat, key_flat, value_flat, attn_metadata,
                                out_view, max_query_len=L_chunk,
                            ),
                        )
                    else:
                        _time_call(
                            "prefill_fa2_cda_rawkv",
                            lambda: self._call_fa2_cda_rawkv_varlen(
                                q_flat, key_flat, value_flat, attn_metadata,
                                out_view,
                                rotation_fp32=rot_fp32,
                                max_query_len=L_chunk,
                            ),
                        )
            elif prefill_backend == "hmma":
                _time_call(
                    "prefill_hmma",
                    lambda: self._call_p34_varlen(
                        q_flat, kv_cache_flat, attn_metadata, out_view,
                        past_len,
                        cb_K=cb_K, cb_V=cb_V,
                        rotation_fp32=rot_fp32, rotation_fp16=rot_fp16,
                        block_size=block_size,
                        max_query_len=L_chunk,
                    ),
                )
            else:
                raise NotImplementedError(
                    f"unknown CDA_PREFILL_BACKEND={prefill_backend!r}; "
                    "expected 'hmma', 'fa2_cda', 'fa2_cda_fused', "
                    "'fa2_cda_auto', 'fa2_cda_direct', 'fa2_cda_direct_v2', "
                    "'fa2_cda_fused_rawkv', 'fa2_rawkv', or 'fa2_cda_rawkv'.")

            if out_view.data_ptr() != output.data_ptr():
                if output.dim() == 3:
                    _time_call(
                        "prefill_output_copy",
                        lambda: output[:T].copy_(out_view.to(output.dtype)))
                else:
                    _time_call(
                        "prefill_output_copy",
                        lambda: output[:T].copy_(
                            out_view.reshape(T, H_q * D).to(output.dtype)))
            return output

        def _forward_prefill_hmma_v1(
            self, query, key, value, output, attn_metadata, kv_cache,
            *, L_chunk: int, past_len: int,
        ):
            """v1-style hybrid prefill: FA2(new chunk) + merged-HMMA(past)
            + LSE merge.

            Mirrors v1's ``_hybrid_prefill.hybrid_attn_batched`` flow
            but binds to the K4V4 multi-Q HMMA build. Causal mask is
            applied by FA2 on the new chunk; the compressed past is
            strictly before the new chunk so no past-side mask is
            needed.

            Setup-state caveats:
              * Uniform L_new across reqs (vLLM's typical chunked-
                prefill schedule); non-uniform raises and the caller
                drops back to SDPA.
              * ``group_size`` ∈ {4, 8}; multi-Q kernel is rebuilt per
                gs via ``-DFD_GS=...``.
              * Correctness not yet validated against
                ``cda.algorithm.attention_compressed`` reference.
            """
            from cda.kernels_cuda.wrappers import hybrid_attn_v1

            num_reqs = attn_metadata.num_reqs
            H_q = self.num_heads
            D = self.head_size
            cu = attn_metadata.query_start_loc
            cmp_K, cmp_V = _get_compressors(query.device)

            # Flatten paged cache: (num_blocks, block_size, H_kv, slot_w)
            #                    → (num_blocks * block_size, H_kv, slot_w).
            num_blocks, block_size, H_kv, slot_w = kv_cache.shape
            kv_cache_flat = kv_cache.view(num_blocks * block_size, H_kv, slot_w)

            cb_K, cb_V, rotation_fp32, rotation_fp16 = _get_hmma_artifacts(
                cmp_K, cmp_V, query.device)

            q_in = query if query.dtype == torch.float16 \
                else query.to(torch.float16)
            k_in = key   if key.dtype   == torch.float16 \
                else key.to(torch.float16)
            v_in = value if value.dtype == torch.float16 \
                else value.to(torch.float16)

            hybrid_attn_v1(
                q_in, k_in, v_in, output,
                kv_cache_flat,
                attn_metadata.block_table,
                cu_seqlens_q=cu,
                seq_lens=attn_metadata.seq_lens,
                cb_K=cb_K, cb_V=cb_V,
                rotation_fp32=rotation_fp32,
                rotation_fp16=rotation_fp16,
                scale=self.scale,
                group_size=self.group_size,
                block_size=block_size,
            )
            return output

        def _dequantize_past(self, kv_cache, block_table_row, past_len,
                              *, cmp_K, cmp_V):
            num_blocks, block_size, H_kv, slot_w = kv_cache.shape
            D = self.head_size
            k_bytes = D // 2
            v_bytes = D // 2
            num_blk = (past_len + block_size - 1) // block_size
            req_blocks = block_table_row[:num_blk].to(torch.long)
            slot_ids = (req_blocks.unsqueeze(1) * block_size +
                         torch.arange(block_size, device=kv_cache.device)
                         ).reshape(-1)[:past_len].to(torch.long)
            flat = kv_cache.view(num_blocks * block_size, H_kv,
                                   slot_w).index_select(0, slot_ids)
            K_idx = flat[:, :, 0:k_bytes].permute(1, 0, 2).contiguous()
            V_idx = flat[:, :, k_bytes:k_bytes + v_bytes].permute(1, 0, 2).contiguous()
            K_norm = (flat[:, :, k_bytes + v_bytes:k_bytes + v_bytes + 4]
                      .contiguous().reshape(-1).view(torch.float32)
                      .reshape(past_len, H_kv).permute(1, 0).contiguous())
            V_norm = (flat[:, :, k_bytes + v_bytes + 4:k_bytes + v_bytes + 8]
                      .contiguous().reshape(-1).view(torch.float32)
                      .reshape(past_len, H_kv).permute(1, 0).contiguous())
            K_slot = CompressedSlot(idx=K_idx.unsqueeze(0),
                                       norm=K_norm.unsqueeze(0))
            V_slot = CompressedSlot(idx=V_idx.unsqueeze(0),
                                       norm=V_norm.unsqueeze(0))
            K_past = cmp_K.decode(K_slot, dtype=torch.float16,
                                    rotated=False).squeeze(0)  # (H_kv, past, D)
            V_past = cmp_V.decode(V_slot, dtype=torch.float16,
                                    rotated=False).squeeze(0)
            return K_past, V_past

        def _forward_decode(self, query, kv_cache, output, attn_metadata):
            """Decode path — paged-aware kernel reads (kv_cache, block_table,
            seq_lens) directly. Eliminates the prior gather→contig
            intermediate (~200-800 µs at large B,N) when slot_w is
            16-aligned. Falls back to gather+contig for legacy non-aligned
            slot widths.

            **R5**: hot path inlined for default config.
            HMMA V1 / triton branches gated behind explicit env var
            check up front so the common case has no extra branching.

            **Default decode uses `_decode_hmma.cu`**, while prefill uses
            `_multiq_paged_full_hmma.cu`. This mirrors FA2 varlen-style
            dispatch: L=1 decode takes the single-query kernel, L>1
            prefill takes the multi-Q causal kernel.
            """
            shape = kv_cache.shape
            slot_w = shape[3]

            # Default decode path: full-compressed single-Q HMMA
            # (`_decode_hmma.cu`). Prefill is handled separately by
            # `_forward_prefill_full_hmma_v1`.
            if (slot_w & 15 == 0 and self.group_size in (4, 8)
                    and not _USE_HMMA_V1):
                cb_K, cb_V, rot_fp32, rot_fp16, kv_cache_flat, block_size = (
                    _time_call(
                        "decode_setup",
                        lambda: self._p34_setup(query, kv_cache)))
                q_in = query.squeeze(2) if query.dim() == 4 else query
                if q_in.dtype != torch.float16:
                    q_in = _time_call(
                        "decode_q_cast",
                        lambda: q_in.to(torch.float16))
                # Output buffer: vLLM passes 2D (num_tokens, H_q*D) or 3D
                # (num_tokens, H_q, D); both reduce to a single decode-HMMA
                # call after picking the right view of `output`.
                out_view = None
                if (output.dtype == torch.float16 and output.is_contiguous()
                        and output.shape[0] == q_in.shape[0]):
                    if (output.dim() == 2
                            and output.shape[1] == self.num_heads * self.head_size):
                        out_view = output.view(q_in.shape[0],
                                                self.num_heads, self.head_size)
                    elif (output.dim() == 3
                            and output.shape[1] == self.num_heads
                            and output.shape[2] == self.head_size):
                        out_view = output
                if out_view is not None:
                    _time_call(
                        "decode_hmma",
                        lambda: self._call_decode_hmma_v1(
                            q_in, kv_cache_flat, attn_metadata, out_view,
                            cb_K=cb_K, cb_V=cb_V,
                            rotation_fp32=rot_fp32, rotation_fp16=rot_fp16,
                            block_size=block_size,
                        ),
                    )
                    return output
                out = torch.empty_like(q_in)
                _time_call(
                    "decode_hmma",
                    lambda: self._call_decode_hmma_v1(
                        q_in, kv_cache_flat, attn_metadata, out,
                        cb_K=cb_K, cb_V=cb_V,
                        rotation_fp32=rot_fp32, rotation_fp16=rot_fp16,
                        block_size=block_size,
                    ),
                )
                if output.dim() == 3:
                    _time_call(
                        "decode_output_copy",
                        lambda: output[:].copy_(out.to(output.dtype)))
                else:
                    _time_call(
                        "decode_output_copy",
                        lambda: output[:].copy_(out.reshape(
                            q_in.shape[0], self.num_heads * self.head_size
                        ).to(output.dtype)))
                return output

            # Slow path (env var override or non-aligned slot width).
            num_blocks, block_size, H_kv, slot_w = shape
            block_table = attn_metadata.block_table
            seq_lens = attn_metadata.seq_lens
            B = block_table.size(0)
            cmp_K, cmp_V = _get_compressors(query.device)

            if query.dim() == 4:
                query = query.squeeze(2)
            elif query.dim() == 3 and query.size(0) != B:
                pass

            if slot_w % 16 == 0:
                kv_cache_flat = kv_cache.view(
                    num_blocks * block_size, H_kv, slot_w)
                q_in = query if query.dim() == 3 else query.squeeze(2)
                if q_in.dtype != torch.float16:
                    q_in = q_in.to(torch.float16)
                out = torch.empty_like(q_in)
                cb_K, cb_V, rotation_fp32, rotation_fp16 = (
                    _get_hmma_artifacts(cmp_K, cmp_V, query.device))
                self._call_decode_hmma_v1(
                    q_in, kv_cache_flat, attn_metadata, out,
                    cb_K=cb_K, cb_V=cb_V,
                    rotation_fp32=rotation_fp32,
                    rotation_fp16=rotation_fp16,
                    block_size=block_size,
                )
            else:
                raise NotImplementedError(
                    "CDA-v2 decode requires 16-byte aligned compressed slots; "
                    "fallback paths are disabled.")

            if output.dim() == 3:
                output[:].copy_(out.to(output.dtype))
            else:
                output[:].copy_(out.reshape(B, self.num_heads * self.head_size)
                                  .to(output.dtype))
            return output

        def _forward_decode_dcp(self, query, kv_cache, output, attn_metadata):
            """DCP decode is disabled until it has a CUDA HMMA implementation."""
            raise NotImplementedError(
                "CDA-v2 DCP decode needs a CUDA HMMA partial kernel; "
                "fallback paths are disabled.")

    return (CDAv2AttentionBackend, CDAv2AttentionImpl,
            CDAv2AttentionMetadataBuilder, CDAv2AttentionMetadataVLLM)


def get_backend_classes():
    """Lazy entry point — returns the four vLLM-shaped classes."""
    return _build_backend_classes()


def register_backend(slot: str = "CDA") -> None:
    """Register the cda-v2 backend in vLLM's AttentionBackendEnum.

    Call this BEFORE constructing ``LLM`` to enable
    ``attention_backend="CDA"`` (or whatever ``slot`` you pass) in your
    model_args. Idempotent.

    Args:
        slot: which AttentionBackendEnum member to override. ``"CDA"``
            (default) replaces vLLM's vendored cda-v1 entry; ``"CUSTOM"``
            uses the placeholder slot. Both work with
            ``LLM(attention_backend=slot)``.

    The runtime ``CDA_V2_ENABLE_MEMORY_SAVING=1`` env var (set BEFORE
    importing this module) shrinks vLLM's slot allocation to 144 B
    instead of the FP16-aligned 512 B — yields the full ~3.56× KV
    memory saving in vLLM's available-tokens math.
    """
    # Eagerly construct + cache the classes via the eager entry module so
    # vLLM's class-path resolver can reach them by module attribute.
    from cda.vllm_integration import _backend_eager  # noqa: F401

    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum, register_backend as _vllm_register,
    )

    try:
        target = AttentionBackendEnum[slot]
    except (KeyError, ValueError) as e:
        raise ValueError(
            f"slot={slot!r} is not a known AttentionBackendEnum member") from e

    _vllm_register(
        target,
        "cda.vllm_integration._backend_eager.CDAv2AttentionBackend",
    )


__all__ = [
    "CDAv2AttentionMetadata",
    "enable_cda_memory_saving",
    "get_backend_classes",
    "register_backend",
    "HEAD_DIM",
]
