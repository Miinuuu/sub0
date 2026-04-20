"""Quantized anchor storage for KVCOMM.

The core quantizer follows TurboQuant's data-oblivious recipe: random orthogonal
rotation followed by Lloyd-Max scalar quantization on the resulting Beta-like
coordinates. On top of that, the quantized symbols are bit-packed and can be
optionally DEFLATE-compressed, which applies the "quantize first, then entropy
code for storage" principle used in KV-cache transform coding work.

References:
  - "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
    (arXiv:2504.19874)
  - "KV Cache Transform Coding for Compact Storage in LLM Inference"
    (arXiv:2511.01815)
"""
from __future__ import annotations

import threading
import zlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Pre-computed Lloyd-Max codebooks for Beta((d-1)/2, (d-1)/2) with d = 128
# ---------------------------------------------------------------------------

def _lloyd_max_codebook(
    alpha: float,
    beta_param: float,
    num_levels: int,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Lloyd-Max optimal quantizer for a Beta(alpha, beta_param) distribution.

    Returns:
        boundaries: array of shape (num_levels + 1,) with boundaries[0]=0, boundaries[-1]=1
        centroids:  array of shape (num_levels,) reconstruction values
    """
    dist = sp_stats.beta(alpha, beta_param)

    # Initial uniform partition
    boundaries = np.linspace(0.0, 1.0, num_levels + 1)
    centroids = np.zeros(num_levels)

    for _ in range(max_iter):
        # Update centroids: conditional mean of each bin
        for j in range(num_levels):
            lo, hi = boundaries[j], boundaries[j + 1]
            if hi - lo < 1e-15:
                centroids[j] = (lo + hi) / 2.0
                continue
            # E[X | lo < X < hi] = integral(x * pdf(x), lo, hi) / P(lo < X < hi)
            prob = dist.cdf(hi) - dist.cdf(lo)
            if prob < 1e-15:
                centroids[j] = (lo + hi) / 2.0
                continue
            moment = sp_stats.beta.expect(
                lambda x: x, args=(alpha, beta_param), loc=0, scale=1,
                lb=lo, ub=hi, conditional=False,
            )
            centroids[j] = float(moment) / prob

        # Update boundaries: midpoints of adjacent centroids
        old_boundaries = boundaries.copy()
        for j in range(1, num_levels):
            boundaries[j] = (centroids[j - 1] + centroids[j]) / 2.0

        if np.max(np.abs(boundaries - old_boundaries)) < tol:
            break

    return boundaries, centroids


@lru_cache(maxsize=None)
def _build_codebooks(
    dim: int = 128,
    bit_widths: Tuple[int, ...] = (2, 3, 4, 8),
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Build Lloyd-Max codebooks for Beta((d-1)/2, (d-1)/2) at several bit-widths."""
    alpha = (dim - 1) / 2.0
    codebooks: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for bw in bit_widths:
        num_levels = 1 << bw
        boundaries, centroids = _lloyd_max_codebook(alpha, alpha, num_levels)
        codebooks[bw] = (boundaries, centroids)
    return codebooks


def _pack_indices(indices: torch.Tensor, bit_width: int) -> bytes:
    """Pack uint8 quantization indices into a dense byte stream."""
    flat = indices.reshape(-1).to(device="cpu", dtype=torch.uint8).numpy()
    if flat.size == 0:
        return b""
    if bit_width == 8:
        return flat.tobytes()

    total_bits = int(flat.size) * bit_width
    packed = np.zeros((total_bits + 7) // 8, dtype=np.uint8)
    base_positions = np.arange(flat.size, dtype=np.uint64) * bit_width

    for bit in range(bit_width):
        bit_values = (flat >> bit) & 1
        positions = base_positions + bit
        byte_positions = positions >> 3
        bit_positions = positions & 7
        packed[byte_positions] |= bit_values.astype(np.uint8) << bit_positions.astype(np.uint8)

    return packed.tobytes()


def _unpack_indices(payload: bytes, value_count: int, bit_width: int) -> torch.Tensor:
    """Unpack a dense byte stream into uint8 quantization indices."""
    if value_count == 0:
        return torch.empty((0,), dtype=torch.uint8)

    raw = np.frombuffer(payload, dtype=np.uint8)
    if bit_width == 8:
        return torch.from_numpy(raw.copy())

    flat = np.zeros(value_count, dtype=np.uint8)
    base_positions = np.arange(value_count, dtype=np.uint64) * bit_width

    for bit in range(bit_width):
        positions = base_positions + bit
        byte_positions = positions >> 3
        bit_positions = positions & 7
        bit_values = ((raw[byte_positions] >> bit_positions.astype(np.uint8)) & 1).astype(np.uint8)
        flat |= bit_values << bit

    return torch.from_numpy(flat.copy())


def _gpu_pack_4bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack pairs of 4-bit indices into single uint8 values on GPU.

    Input:  (N, dim) uint8 with values in [0, 15]
    Output: (N, dim//2) uint8 — two values per byte
    """
    assert indices.shape[-1] % 2 == 0, f"dim must be even for 4-bit packing, got {indices.shape[-1]}"
    hi = indices[..., 0::2]  # even positions → high nibble
    lo = indices[..., 1::2]  # odd positions  → low nibble
    return (hi << 4) | lo


def _gpu_unpack_4bit(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 bytes into pairs of 4-bit indices on GPU.

    Input:  (N, dim//2) uint8
    Output: (N, dim) uint8
    """
    hi = packed >> 4
    lo = packed & 0x0F
    return torch.stack([hi, lo], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def _gpu_pack_2bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack four 2-bit indices into single uint8 values on GPU.

    Input:  (N, dim) uint8 with values in [0, 3]
    Output: (N, dim//4) uint8 — four values per byte
    """
    d = indices.shape[-1]
    assert d % 4 == 0, f"dim must be divisible by 4 for 2-bit packing, got {d}"
    return (indices[..., 0::4] << 6) | (indices[..., 1::4] << 4) | (indices[..., 2::4] << 2) | indices[..., 3::4]


def _gpu_unpack_2bit(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 bytes into four 2-bit indices on GPU.

    Input:  (N, dim//4) uint8
    Output: (N, dim) uint8
    """
    b3 = (packed >> 6) & 0x03
    b2 = (packed >> 4) & 0x03
    b1 = (packed >> 2) & 0x03
    b0 = packed & 0x03
    return torch.stack([b3, b2, b1, b0], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 4)


def _maybe_deflate(payload: bytes, enabled: bool, level: int) -> bytes:
    if enabled and payload:
        return zlib.compress(payload, level=level)
    return payload


def _maybe_inflate(payload: bytes, enabled: bool) -> bytes:
    if enabled and payload:
        return zlib.decompress(payload)
    return payload


# ---------------------------------------------------------------------------
# Compressed tensor container
# ---------------------------------------------------------------------------

@dataclass
class CompressedTensor:
    """Stores a compact representation of a float tensor.

    Supports two storage modes:
    - **gpu**: indices (uint8) and norms (float16) stay on the original GPU device.
      No bit-packing or DEFLATE; fastest dequantize at the cost of slightly more
      memory (1 byte per coordinate instead of bit_width/8).
    - **cpu** (legacy): indices are bit-packed into a ``bytes`` payload and
      optionally DEFLATE-compressed, norms on CPU.  Maximises compression but
      dequantize is slow due to CPU→GPU transfers.

    Fields:
        indices:  uint8 GPU tensor of quantization bin indices (gpu mode)
        payload:  packed quantization indices, optionally DEFLATE-compressed (cpu mode)
        norms:    float16 per-vector L2 norms
        shape:    original tensor shape
        dtype:    original tensor dtype
        device:   original tensor device
        bit_width: number of bits per coordinate
        deflated: whether payload is DEFLATE-compressed
        gpu_resident: True if indices/norms live on GPU as tensors
    """
    indices: Optional[torch.Tensor]   # uint8, GPU  (gpu mode)
    payload: Optional[bytes]          # packed bytes (cpu mode)
    norms: torch.Tensor               # float16
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device
    bit_width: int
    deflated: bool = False
    gpu_resident: bool = True

    def nbytes(self) -> int:
        """Approximate storage footprint in bytes."""
        if self.gpu_resident and self.indices is not None:
            return self.indices.numel() + self.norms.numel() * self.norms.element_size()
        payload_size = len(self.payload) if self.payload else 0
        return payload_size + (self.norms.numel() * self.norms.element_size())


@dataclass
class PCACompressedTensor:
    """Stores a PCA-projected and quantized float tensor.

    Supports two storage modes:
    - **gpu**: quantized coefficients stay on GPU as uint8 tensor (fast dequantize).
    - **cpu** (legacy): coefficients are bit-packed into bytes, optionally DEFLATE-compressed.

    Fields:
        indices:      uint8 GPU tensor of quantized PCA coefficients (gpu mode)
        payload:      packed bytes (cpu mode)
        scale_shift:  quant_limit value needed for dequantization
    """
    indices: Optional[torch.Tensor] = None   # uint8, GPU  (gpu mode)
    payload: Optional[bytes] = None          # packed bytes (cpu mode)
    shape: torch.Size = None
    dtype: torch.dtype = None
    device: torch.device = None
    rank: int = 0
    bit_width: int = 8
    deflated: bool = False
    gpu_resident: bool = False

    def nbytes(self) -> int:
        if self.gpu_resident and self.indices is not None:
            return self.indices.numel() * self.indices.element_size()
        return len(self.payload) if self.payload else 0


# ---------------------------------------------------------------------------
# TurboQuant Compressor
# ---------------------------------------------------------------------------

class TurboQuantCompressor:
    """Data-oblivious vector quantizer using random orthogonal rotation + Lloyd-Max.

    Usage::

        compressor = TurboQuantCompressor(dim=128, bit_width=4)
        compressed = compressor.quantize(delta_tensor)   # any shape (..., dim)
        restored   = compressor.dequantize(compressed)   # same shape as original
    """

    def __init__(
        self,
        dim: int = 128,
        bit_width: int = 4,
        seed: int = 42,
        *,
        deflate: bool = False,
        deflate_level: int = 1,
        half_rotation: bool = False,
        norm_fp32: bool = False,
    ):
        self.dim = dim
        self.bit_width = bit_width
        self.deflate = deflate
        self.deflate_level = deflate_level
        self.half_rotation = half_rotation
        self.norm_fp32 = norm_fp32

        # Deterministic random orthogonal matrix via QR decomposition
        rng = np.random.RandomState(seed)
        H = rng.randn(dim, dim).astype(np.float32)
        Q, _ = np.linalg.qr(H)
        orth_err = np.linalg.norm(Q @ Q.T - np.eye(dim))
        assert orth_err < 1e-5, f"Rotation orthogonality check failed: {orth_err}"
        self._rotation_np = Q
        # Lazy-init torch tensors on first use (to match device)
        self._rotation: Optional[torch.Tensor] = None
        self._rotation_t: Optional[torch.Tensor] = None

        # Build Lloyd-Max codebook for this bit-width
        codebooks = _build_codebooks(dim, bit_widths=(bit_width,))
        boundaries, centroids = codebooks[bit_width]
        self._boundaries_np = boundaries.astype(np.float32)
        self._centroids_np = centroids.astype(np.float32)
        # Lazy-init torch tensors
        self._boundaries: Optional[torch.Tensor] = None
        self._centroids: Optional[torch.Tensor] = None
        self._device_lock = threading.Lock()
        self._current_device: Optional[torch.device] = None

    def _ensure_tensors(self, device: torch.device) -> None:
        """Move rotation matrix and codebook to *device* on first call (thread-safe)."""
        if self._current_device == device:
            return
        with self._device_lock:
            if self._current_device == device:
                return
            rot_dtype = torch.float16 if (self.half_rotation and device.type == "cuda") else torch.float32
            rotation = torch.from_numpy(self._rotation_np).to(device=device, dtype=rot_dtype)
            rotation_t = rotation.t().contiguous()
            boundaries = torch.from_numpy(self._boundaries_np).to(device=device, dtype=torch.float32)
            centroids = torch.from_numpy(self._centroids_np).to(device=device, dtype=torch.float32)
            # Assign all at once so other threads never see partial state
            self._rotation = rotation
            self._rotation_t = rotation_t
            self._boundaries = boundaries
            self._centroids = centroids
            self._current_device = device

    # ---- public API --------------------------------------------------------

    def quantize(self, tensor: torch.Tensor) -> CompressedTensor:
        """Quantize *tensor* whose last dimension equals ``self.dim``.

        Steps:
        1. Flatten to (-1, dim), compute per-vector L2 norms.
        2. Normalize to unit vectors.
        3. Apply random orthogonal rotation.
        4. Map coordinates from [-1, 1] to [0, 1] (Beta distribution support).
        5. Scalar-quantize each coordinate using Lloyd-Max boundaries.

        When ``self.deflate`` is False (default), indices and norms stay on GPU
        as tensors for fast dequantization.  When True, indices are bit-packed
        and DEFLATE-compressed for maximum storage savings (slow dequantize).
        """
        orig_shape = tensor.shape
        orig_dtype = tensor.dtype
        orig_device = tensor.device
        assert tensor.shape[-1] == self.dim, (
            f"Last dim {tensor.shape[-1]} != compressor dim {self.dim}"
        )
        self._ensure_tensors(orig_device)

        compute_dtype = torch.float16 if (self.half_rotation and orig_device.type == "cuda") else torch.float32
        flat = tensor.reshape(-1, self.dim).to(compute_dtype)  # (N, dim)
        _n = flat.shape[0]
        _chunk = min(_n, 1_000_000)

        # Per-vector norms (chunked to avoid OOM at large N)
        norms = torch.empty(_n, 1, dtype=torch.float32, device=orig_device)
        for _i in range(0, _n, _chunk):
            norms[_i:_i+_chunk] = flat[_i:_i+_chunk].float().norm(2, dim=-1, keepdim=True)
        norms.clamp_(min=1e-12)

        # Normalize + rotate + map in chunks (avoids N×dim fp32 temporaries)
        rot_t = self._rotation_t.to(compute_dtype)
        mapped = torch.empty_like(flat)
        for _i in range(0, _n, _chunk):
            _s = slice(_i, min(_i + _chunk, _n))
            _unit = flat[_s] / norms[_s].to(compute_dtype)
            _rot = _unit @ rot_t
            mapped[_s] = ((_rot + 1.0) * 0.5).clamp(0.0, 1.0)
        del flat  # free original flat tensor

        # Scalar quantize: find bin index via searchsorted (chunked to avoid int64 OOM)
        _bounds = self._boundaries[1:-1]
        indices = torch.empty(_n, self.dim, dtype=torch.uint8, device=orig_device)
        for _i in range(0, _n, _chunk):
            _s = slice(_i, min(_i + _chunk, _n))
            indices[_s] = torch.searchsorted(_bounds, mapped[_s]).to(torch.uint8)
        del mapped

        if self.deflate:
            # CPU-offloaded mode: bit-pack + DEFLATE for max compression
            payload = _pack_indices(indices, self.bit_width)
            payload = _maybe_deflate(payload, True, self.deflate_level)
            return CompressedTensor(
                indices=None,
                payload=payload,
                norms=norms.squeeze(-1).to(device="cpu", dtype=torch.float16),
                shape=orig_shape,
                dtype=orig_dtype,
                device=orig_device,
                bit_width=self.bit_width,
                deflated=True,
                gpu_resident=False,
            )

        # GPU-resident mode with bit packing on GPU
        if self.bit_width == 4:
            packed = _gpu_pack_4bit(indices)
        elif self.bit_width == 2:
            packed = _gpu_pack_2bit(indices)
        else:
            packed = indices

        return CompressedTensor(
            indices=packed,
            payload=None,
            norms=norms.squeeze(-1).to(torch.float32 if self.norm_fp32 else torch.float16),
            shape=orig_shape,
            dtype=orig_dtype,
            device=orig_device,
            bit_width=self.bit_width,
            deflated=False,
            gpu_resident=True,
        )

    def dequantize(self, compressed: CompressedTensor) -> torch.Tensor:
        """Reconstruct the original tensor from its compressed representation."""
        device = compressed.device
        self._ensure_tensors(device)

        if compressed.gpu_resident and compressed.indices is not None:
            # ---- Fast GPU path: no CPU transfers ----
            packed = compressed.indices
            if compressed.bit_width == 4 and packed.shape[-1] == self.dim // 2:
                indices = _gpu_unpack_4bit(packed)
            elif compressed.bit_width == 2 and packed.shape[-1] == self.dim // 4:
                indices = _gpu_unpack_2bit(packed)
            else:
                indices = packed
            indices = indices.long()
            norms = compressed.norms.float().unsqueeze(-1)  # (N, 1), already on device
        else:
            # ---- Legacy CPU-offloaded path ----
            payload = _maybe_inflate(compressed.payload, compressed.deflated)
            value_count = int(np.prod(compressed.shape))
            indices = _unpack_indices(payload, value_count, compressed.bit_width).reshape(-1, self.dim)
            indices = indices.to(device=device, dtype=torch.long)
            norms = compressed.norms.to(device=device, dtype=torch.float32).unsqueeze(-1)

        # Codebook lookup → values in [0, 1]
        compute_dtype = self._rotation.dtype  # fp16 if half_rotation, else fp32
        reconstructed = self._centroids[indices].to(compute_dtype)  # (N, dim)

        # Unmap from [0, 1] to [-1, 1]
        reconstructed = reconstructed * 2.0 - 1.0

        # Inverse rotation
        recovered = reconstructed @ self._rotation  # (N, dim)

        # Restore norms
        recovered = recovered.float() * norms

        # Reshape and cast back
        return recovered.reshape(compressed.shape).to(compressed.dtype)

    def dequantize_pre_rotation(
        self, compressed: CompressedTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Codebook lookup + unmap WITHOUT inverse rotation.

        Returns the reconstructed unit-direction in rotated space and the
        per-vector norms separately.  Callers can do weighted interpolation
        in rotated space and apply :meth:`apply_inverse_rotation` once.

        Returns:
            unmapped: ``(N, dim)`` tensor in ``[-1, 1]`` (rotated space).
            norms:    ``(N, 1)`` per-vector L2 norms.
        """
        device = compressed.device
        self._ensure_tensors(device)

        if compressed.gpu_resident and compressed.indices is not None:
            packed = compressed.indices
            if compressed.bit_width == 4 and packed.shape[-1] == self.dim // 2:
                indices = _gpu_unpack_4bit(packed)
            elif compressed.bit_width == 2 and packed.shape[-1] == self.dim // 4:
                indices = _gpu_unpack_2bit(packed)
            else:
                indices = packed
            indices = indices.long()
            norms = compressed.norms.float().unsqueeze(-1)
        else:
            payload = _maybe_inflate(compressed.payload, compressed.deflated)
            value_count = int(np.prod(compressed.shape))
            indices = _unpack_indices(payload, value_count, compressed.bit_width).reshape(-1, self.dim)
            indices = indices.to(device=device, dtype=torch.long)
            norms = compressed.norms.to(device=device, dtype=torch.float32).unsqueeze(-1)

        compute_dtype = self._rotation.dtype
        reconstructed = self._centroids[indices].to(compute_dtype)
        unmapped = reconstructed * 2.0 - 1.0  # (N, dim) in [-1, 1], rotated space
        return unmapped, norms

    def apply_inverse_rotation(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply inverse rotation to a tensor in rotated space.

        Args:
            tensor: ``(..., dim)`` tensor — typically the norm-weighted sum
                    of pre-rotation outputs from multiple anchors.

        Returns:
            ``(..., dim)`` tensor in the original (un-rotated) space.
        """
        self._ensure_tensors(tensor.device)
        orig_shape = tensor.shape
        flat = tensor.reshape(-1, self.dim)
        recovered = flat.to(self._rotation.dtype) @ self._rotation
        return recovered.float().reshape(orig_shape)

    def for_layer(self, layer_idx: int) -> "TurboQuantCompressor":
        """Return the compressor for a given layer (self for uniform bit-width)."""
        return self


class HadamardQuantCompressor(TurboQuantCompressor):
    """TurboQuant with random rotation replaced by normalized Hadamard matrix.

    Mathematically equivalent quality, but Hadamard uses only ±1 multiplications
    (additions/subtractions), making it suitable for FPGA/on-device deployment
    where DSP multipliers are scarce.

    Requires dim to be a power of 2 (e.g., 128).
    """

    def __init__(self, dim: int = 128, bit_width: int = 4, seed: int = 42, **kwargs):
        super().__init__(dim=dim, bit_width=bit_width, seed=seed, **kwargs)
        from scipy.linalg import hadamard
        H = hadamard(dim).astype(np.float32) / np.sqrt(dim)
        self._rotation_np = H
        # Reset cached torch tensors so _ensure_tensors rebuilds them
        self._rotation = None
        self._rotation_t = None
        self._current_device = None


def calibrate_codebook(compressor, kv_cache, skip_sinks=4, max_samples=8192):
    """Calibrate codebook from real KV data (data-dependent Lloyd-Max).

    Replaces the theoretical Beta-distribution codebook with one optimized
    for actual KV activation distributions. HW cost: 0 (same LUT structure).

    Args:
        compressor: TurboQuantCompressor or HadamardQuantCompressor
        kv_cache: DynamicCache from prefill
        skip_sinks: skip first N tokens
        max_samples: max vectors for calibration

    Returns:
        compressor with updated centroids/boundaries (modified in-place)
    """
    dim = compressor.dim
    bit_width = compressor.bit_width
    num_levels = 1 << bit_width

    # Collect rotated coordinates from all layers
    all_mapped = []
    compressor._ensure_tensors(kv_cache.key_cache[0].device)

    for li in range(len(kv_cache)):
        k = kv_cache.key_cache[li]  # (B, H, S, D)
        B, H, S, D = k.shape
        if skip_sinks > 0 and S > skip_sinks:
            k = k[:, :, skip_sinks:, :]
        flat = k.reshape(-1, dim).float()

        # Normalize + rotate (same as quantize pipeline)
        norms = flat.norm(2, dim=-1, keepdim=True).clamp(min=1e-12)
        unit = flat / norms
        compute_dtype = compressor._rotation_t.dtype
        rotated = unit.to(compute_dtype) @ compressor._rotation_t
        mapped = ((rotated.float() + 1.0) * 0.5).clamp(0.0, 1.0)

        # Subsample if too many
        if mapped.shape[0] > max_samples // len(kv_cache):
            idx = torch.randperm(mapped.shape[0])[:max_samples // len(kv_cache)]
            mapped = mapped[idx]
        all_mapped.append(mapped.cpu())

    all_mapped = torch.cat(all_mapped, dim=0).numpy().flatten()  # all coordinates

    # Run Lloyd-Max on empirical distribution (histogram-based)
    # Sort data, initialize uniform boundaries, iterate
    data = np.sort(all_mapped)
    data = data[data > 0.01]  # remove near-zero outliers
    data = data[data < 0.99]

    boundaries = np.linspace(data.min(), data.max(), num_levels + 1)
    centroids = np.zeros(num_levels)

    for _ in range(100):
        # Update centroids: mean of data in each bin
        for j in range(num_levels):
            lo, hi = boundaries[j], boundaries[j + 1]
            mask = (data >= lo) & (data < hi)
            if mask.sum() > 0:
                centroids[j] = data[mask].mean()
            else:
                centroids[j] = (lo + hi) / 2.0

        # Update boundaries
        old_b = boundaries.copy()
        for j in range(1, num_levels):
            boundaries[j] = (centroids[j - 1] + centroids[j]) / 2.0

        if np.max(np.abs(boundaries - old_b)) < 1e-8:
            break

    # Update compressor
    compressor._boundaries_np = boundaries.astype(np.float32)
    compressor._centroids_np = centroids.astype(np.float32)
    compressor._boundaries = None
    compressor._centroids = None
    compressor._current_device = None  # force re-upload to GPU

    return compressor


@dataclass
class TurboProdCompressed:
    """Container for TurboQuant_prod compressed output.

    Stores MSE indices (b-1 bits), QJL sign bits (1 bit), residual norm,
    and original vector norms (for unit sphere ↔ original scale conversion).
    Total bit-width = b per coordinate (same as TurboQuant_mse at b bits).
    """
    mse_compressed: CompressedTensor   # TurboQuant_mse at (b-1) bits
    qjl_signs: torch.Tensor            # {-1, +1}^d packed as int8, shape (N, dim)
    residual_norms: torch.Tensor       # float16, shape (N,) — ||r||₂ on unit sphere
    vec_norms: torch.Tensor            # float16, shape (N,) — original ||x||₂
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device
    bit_width: int                     # total b = (b-1) MSE + 1 QJL


class TurboQuantProdCompressor:
    """TurboQuant_prod: unbiased inner-product quantizer (Algorithm 2 in paper).

    Combines TurboQuant_mse at (b-1) bits with 1-bit QJL on the residual,
    producing an unbiased inner product estimator at b bits total per coordinate.

    Usage::

        comp = TurboQuantProdCompressor(dim=128, bit_width=4)
        compressed = comp.quantize(tensor)
        restored = comp.dequantize(compressed)
        # Inner product: <y, restored> is UNBIASED estimator of <y, original>
    """

    def __init__(
        self,
        dim: int = 128,
        bit_width: int = 4,
        seed: int = 42,
        *,
        half_rotation: bool = False,
    ):
        self.dim = dim
        self.bit_width = bit_width
        # MSE compressor at (b-1) bits
        self._mse_compressor = TurboQuantCompressor(
            dim=dim, bit_width=max(bit_width - 1, 1), seed=seed,
            half_rotation=half_rotation,
        )
        # QJL random projection matrix S (Gaussian i.i.d., NOT normalized)
        # Paper: S_{ij} ~ N(0, 1). The 1/d normalization is in the dequant formula.
        rng = np.random.RandomState(seed + 1)  # different seed from rotation
        self._S_np = rng.randn(dim, dim).astype(np.float32)
        self._S: Optional[torch.Tensor] = None
        self._S_t: Optional[torch.Tensor] = None
        self._device_lock = threading.Lock()
        self._current_device: Optional[torch.device] = None

    def _ensure_tensors(self, device: torch.device) -> None:
        if self._current_device == device:
            return
        with self._device_lock:
            if self._current_device == device:
                return
            self._S = torch.from_numpy(self._S_np).to(device=device, dtype=torch.float32)
            self._S_t = self._S.t().contiguous()
            self._current_device = device

    def quantize(self, tensor: torch.Tensor) -> TurboProdCompressed:
        """Quantize with unbiased inner-product guarantee.

        Steps (Algorithm 2):
        1. Normalize to unit sphere, store norms
        2. Quantize with TQ_mse at (b-1) bits
        3. Compute residual r = x - DeQuant_mse(idx)
        4. QJL: qjl = sign(S · r), store ||r||
        """
        orig_shape = tensor.shape
        orig_dtype = tensor.dtype
        orig_device = tensor.device
        self._ensure_tensors(orig_device)

        flat = tensor.reshape(-1, self.dim).float()

        # Separate norms — paper operates on unit sphere
        vec_norms = flat.norm(2, dim=-1, keepdim=True).clamp(min=1e-12)  # (N, 1)
        unit = flat / vec_norms  # (N, dim), unit sphere

        # Step 1-2: MSE quantize at (b-1) bits on original tensor
        # (TQ internally does norm separation, so we pass original)
        mse_compressed = self._mse_compressor.quantize(tensor)

        # Step 3: Compute residual ON UNIT SPHERE
        # DeQuant_mse restores to original scale, so we normalize back to unit
        mse_restored = self._mse_compressor.dequantize(mse_compressed).reshape(-1, self.dim).float()
        mse_unit = mse_restored / vec_norms  # back to unit sphere
        residual = unit - mse_unit  # (N, dim), residual on unit sphere

        # Step 4: QJL on residual (paper: qjl = sign(S · r))
        residual_norms = residual.norm(2, dim=-1)  # (N,), ||r||₂
        projected = residual @ self._S_t  # (N, dim)
        qjl_signs = projected.sign().to(torch.int8)  # (N, dim)
        qjl_signs[qjl_signs == 0] = 1

        return TurboProdCompressed(
            mse_compressed=mse_compressed,
            qjl_signs=qjl_signs,
            residual_norms=residual_norms.to(torch.float16),
            vec_norms=vec_norms.squeeze(-1).to(torch.float16),
            shape=orig_shape,
            dtype=orig_dtype,
            device=orig_device,
            bit_width=self.bit_width,
        )

    def dequantize(self, compressed: TurboProdCompressed) -> torch.Tensor:
        """Reconstruct with unbiased inner-product property.

        Steps (Algorithm 2):
        1. x̃_mse = DeQuant_mse(idx)
        2. x̃_qjl = (√(π/2) / d) · γ · S^T · qjl
        3. return x̃_mse + x̃_qjl
        """
        self._ensure_tensors(compressed.device)

        # Step 1: MSE dequantize (returns original-scale vectors)
        x_mse = self._mse_compressor.dequantize(compressed.mse_compressed)
        x_mse_flat = x_mse.reshape(-1, self.dim).float()

        # Step 2: QJL dequantize on unit sphere
        # Paper: x̃_qjl = (√(π/2) / d) · γ · S^T · qjl
        gamma = compressed.residual_norms.float().unsqueeze(-1)  # (N, 1)
        qjl = compressed.qjl_signs.float()  # (N, dim)
        scale = np.sqrt(np.pi / 2) / self.dim
        x_qjl_unit = scale * gamma * (qjl @ self._S)  # (N, dim), unit sphere scale

        # Step 3: Scale QJL correction from unit sphere to original scale
        vec_norms = compressed.vec_norms.float().unsqueeze(-1)  # (N, 1)
        x_qjl = x_qjl_unit * vec_norms  # scale by original ||x||₂

        result = x_mse_flat + x_qjl
        return result.reshape(compressed.shape).to(compressed.dtype)


@dataclass
class LayerAdaptiveCompressed:
    """Container for layer-adaptive compressed tensors."""
    layers: List[CompressedTensor]  # one CompressedTensor per layer
    shape: torch.Size               # original tensor shape
    dtype: torch.dtype
    device: torch.device
    layer_bit_widths: List[int]


class LayerAdaptiveCompressor:
    """Wraps per-layer TurboQuantCompressors with different bit-widths.

    Usage::

        schedule = [2,2,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,2,2]
        compressor = LayerAdaptiveCompressor(dim=128, layer_bit_schedule=schedule)
        compressed = compressor.quantize(delta_tensor)   # shape (layers, ...)
        restored   = compressor.dequantize(compressed)
    """

    def __init__(
        self,
        dim: int = 128,
        layer_bit_schedule: List[int] = None,
        default_bit_width: int = 4,
        seed: int = 42,
        *,
        half_rotation: bool = False,
        compressor_cls=None,
    ):
        self.dim = dim
        self.default_bit_width = default_bit_width
        self._compressor_cls = compressor_cls or TurboQuantCompressor
        self._half_rotation = half_rotation
        self._seed = seed

        if layer_bit_schedule is None:
            self._schedule = None
            self._compressors = {}
        else:
            self._schedule = list(layer_bit_schedule)
            # Create one compressor per unique bit-width (shared across layers)
            self._compressors = {}
            for bw in set(self._schedule):
                if bw > 0:
                    self._compressors[bw] = self._compressor_cls(
                        dim=dim, bit_width=bw, seed=seed, half_rotation=half_rotation,
                    )

    def _get_compressor(self, layer_idx: int) -> Optional[TurboQuantCompressor]:
        if self._schedule is None:
            bw = self.default_bit_width
        elif layer_idx < len(self._schedule):
            bw = self._schedule[layer_idx]
        else:
            bw = self.default_bit_width
        if bw == 0:
            return None
        if bw not in self._compressors:
            self._compressors[bw] = self._compressor_cls(
                dim=self.dim, bit_width=bw, seed=self._seed,
                half_rotation=self._half_rotation,
            )
        return self._compressors[bw]

    def for_layer(self, layer_idx: int) -> Optional[TurboQuantCompressor]:
        """Return the compressor for a given layer (per-layer bit-width)."""
        return self._get_compressor(layer_idx)

    def quantize(self, tensor: torch.Tensor) -> LayerAdaptiveCompressed:
        """Quantize tensor with per-layer bit-widths. First dim must be layers."""
        num_layers = tensor.shape[0]
        orig_shape = tensor.shape
        orig_dtype = tensor.dtype
        device = tensor.device

        compressed_layers = []
        bit_widths = []
        for l in range(num_layers):
            comp = self._get_compressor(l)
            if comp is not None:
                compressed_layers.append(comp.quantize(tensor[l]))
                bit_widths.append(comp.bit_width)
            else:
                # 0-bit: store zeros (layer dropped)
                compressed_layers.append(None)
                bit_widths.append(0)

        return LayerAdaptiveCompressed(
            layers=compressed_layers,
            shape=orig_shape,
            dtype=orig_dtype,
            device=device,
            layer_bit_widths=bit_widths,
        )

    def dequantize(self, compressed: LayerAdaptiveCompressed) -> torch.Tensor:
        """Reconstruct tensor from per-layer compressed data."""
        result_layers = []
        for l, (ct, bw) in enumerate(zip(compressed.layers, compressed.layer_bit_widths)):
            if ct is not None and bw > 0:
                comp = self._get_compressor(l)
                result_layers.append(comp.dequantize(ct))
            else:
                # 0-bit layer: reconstruct as zeros
                layer_shape = list(compressed.shape[1:])
                result_layers.append(
                    torch.zeros(layer_shape, dtype=compressed.dtype, device=compressed.device)
                )
        return torch.stack(result_layers)


def compute_norm_adaptive_schedule(
    kv_cache,
    total_bit_budget: int,
    num_layers: int,
    allowed_bits: Tuple[int, ...] = (2, 3, 4),
    min_bits: int = 2,
) -> list:
    """Compute per-layer bit allocation based on KV norm magnitudes.

    High-norm layers are more sensitive to quantization error and get more bits.
    Uses greedy allocation: start all layers at min_bits, upgrade highest-norm
    layers first until budget is exhausted.

    Args:
        kv_cache: HuggingFace DynamicCache from prefill (has .key_cache list)
        total_bit_budget: sum of all layer bit-widths (e.g. avg_bits * num_layers)
        num_layers: number of transformer layers
        allowed_bits: sorted tuple of allowed bit-widths
        min_bits: minimum bits per layer

    Returns:
        List[int] of length num_layers, each in allowed_bits
    """
    allowed_sorted = sorted(allowed_bits)
    assert min_bits in allowed_sorted, f"min_bits={min_bits} not in allowed_bits"

    # Compute per-layer importance from K norms (K norms dominate attention score quality)
    importance = []
    for li in range(min(num_layers, len(kv_cache))):
        k = kv_cache.key_cache[li]  # (B, H, S, D)
        k_norm = k.float().norm(dim=-1).mean().item()
        importance.append(k_norm)
    # Pad if model has more layers than cache entries
    while len(importance) < num_layers:
        importance.append(importance[-1] if importance else 1.0)

    # Start all layers at min_bits
    schedule = [min_bits] * num_layers
    remaining = total_bit_budget - sum(schedule)

    # Greedy: upgrade highest-importance layers first
    priority = sorted(range(num_layers), key=lambda l: -importance[l])
    for li in priority:
        if remaining <= 0:
            break
        for candidate in sorted(allowed_sorted, reverse=True):
            upgrade = candidate - schedule[li]
            if upgrade > 0 and upgrade <= remaining:
                schedule[li] = candidate
                remaining -= upgrade
                break

    return schedule


def _dp_bit_allocation(
    coeff: torch.Tensor,
    bit_budget: int,
    allowed_bits: Tuple[int, ...] = (0, 2, 4, 8),
) -> np.ndarray:
    """KVTC-inspired DP optimal bit allocation across PCA components.

    Given calibration PCA coefficients, allocates bits per-component to minimize
    Frobenius reconstruction error under a total bit budget.

    Args:
        coeff: (N, rank) PCA coefficients from calibration data
        bit_budget: total bits allowed per sample (across all components)
        allowed_bits: set of bit-widths to consider (0 = drop component)

    Returns:
        per_component_bits: array of shape (rank,) with bit-width per component
    """
    rank = coeff.shape[1]
    coeff_np = coeff.detach().cpu().float().numpy()

    # Pre-compute quantization error for each component at each bit-width
    # error[j, b] = sum of squared quantization error for component j at b bits
    component_errors = {}
    for j in range(rank):
        col = coeff_np[:, j]
        zero_error = float(np.sum(col ** 2))  # error if component dropped (0 bits)
        component_errors[(j, 0)] = zero_error

        for bw in allowed_bits:
            if bw == 0:
                continue
            qlimit = max(1, (1 << (bw - 1)) - 1)
            scale = max(np.max(np.abs(col)), 1e-6) / qlimit
            quantized = np.clip(np.round(col / scale), -qlimit, qlimit)
            restored = quantized * scale
            qerror = float(np.sum((col - restored) ** 2))
            component_errors[(j, bw)] = qerror

    # DP: best_error[j][b] = min total error using first j components with b bits
    INF = float("inf")
    best_error = [[INF] * (bit_budget + 1) for _ in range(rank + 1)]
    best_choice = [[0] * (bit_budget + 1) for _ in range(rank + 1)]

    # Base case: 0 components considered
    total_zero_error = sum(component_errors[(j, 0)] for j in range(rank))
    for b in range(bit_budget + 1):
        best_error[0][b] = total_zero_error

    for j in range(1, rank + 1):
        comp_idx = j - 1
        zero_err_j = component_errors[(comp_idx, 0)]
        for b in range(bit_budget + 1):
            # Option 1: keep 0 bits for this component (inherit previous best)
            best_error[j][b] = best_error[j - 1][b]
            best_choice[j][b] = 0

            # Option 2: allocate some bits
            for bw in allowed_bits:
                if bw == 0:
                    continue
                if bw > b:
                    continue
                # Error change: remove zero-bit error, add quantized error
                new_error = best_error[j - 1][b - bw] - zero_err_j + component_errors[(comp_idx, bw)]
                if new_error < best_error[j][b]:
                    best_error[j][b] = new_error
                    best_choice[j][b] = bw

    # Backtrace
    per_component_bits = np.zeros(rank, dtype=np.int32)
    remaining = bit_budget
    for j in range(rank, 0, -1):
        bw = best_choice[j][remaining]
        per_component_bits[j - 1] = bw
        remaining -= bw

    return per_component_bits


class PCAQuantizedCompressor:
    """One-time calibrated PCA compressor for anchor embeddings.

    This is a lightweight approximation of KVTC's transform-coding flow:
    1. fit a PCA basis once on early anchor activations,
    2. project later anchor embeddings into the low-rank space,
    3. quantize PCA coefficients with adaptive or fixed bit-width,
    4. store on GPU (default) or bit-pack + DEFLATE for CPU offload.

    When ``adaptive_bits`` is True, DP-based bit allocation assigns more bits
    to high-variance components and zero bits to low-variance ones (KVTC-style).
    When ``deflate`` is False (default), quantized coefficients stay on GPU
    as uint8 tensors with optional 4-bit/2-bit packing for fast dequantization.
    """

    def __init__(
        self,
        dim: int = 128,
        rank: int = 64,
        bit_width: int = 8,
        *,
        max_fit_samples: int = 8192,
        niter: int = 2,
        deflate: bool = False,
        deflate_level: int = 1,
        adaptive_bits: bool = False,
    ):
        self.dim = dim
        self.rank = rank
        self.bit_width = bit_width
        self.max_fit_samples = max_fit_samples
        self.niter = niter
        self.deflate = deflate
        self.deflate_level = deflate_level
        self.adaptive_bits = adaptive_bits

        self._fitted_rank = 0
        self._mean_np: Optional[np.ndarray] = None
        self._basis_np: Optional[np.ndarray] = None
        self._scale_np: Optional[np.ndarray] = None
        self._per_component_bits: Optional[np.ndarray] = None  # DP allocation

        self._mean: Optional[torch.Tensor] = None
        self._basis: Optional[torch.Tensor] = None
        self._basis_t: Optional[torch.Tensor] = None
        self._scale: Optional[torch.Tensor] = None
        self._device_lock = threading.Lock()
        self._current_device: Optional[torch.device] = None

    @property
    def is_fitted(self) -> bool:
        return self._basis_np is not None and self._mean_np is not None and self._scale_np is not None

    def _ensure_tensors(self, device: torch.device) -> None:
        """Move PCA basis and scale to *device* on first call (thread-safe)."""
        if not self.is_fitted:
            raise RuntimeError("PCAQuantizedCompressor must be fitted before use.")
        if self._current_device == device:
            return
        with self._device_lock:
            if self._current_device == device:
                return
            mean = torch.from_numpy(self._mean_np).to(device=device, dtype=torch.float32)
            basis = torch.from_numpy(self._basis_np).to(device=device, dtype=torch.float32)
            basis_t = basis.t().contiguous()
            scale = torch.from_numpy(self._scale_np).to(device=device, dtype=torch.float32)
            self._mean = mean
            self._basis = basis
            self._basis_t = basis_t
            self._scale = scale
            self._current_device = device

    def fit(self, tensor: torch.Tensor) -> int:
        if self.is_fitted:
            return self._fitted_rank

        flat = tensor.reshape(-1, self.dim).float()
        if flat.shape[0] == 0:
            raise ValueError("Cannot fit PCA compressor on an empty tensor.")
        if flat.shape[0] > self.max_fit_samples:
            idx = torch.linspace(0, flat.shape[0] - 1, self.max_fit_samples, device=flat.device).long()
            flat = flat.index_select(0, idx)

        mean = flat.mean(dim=0)
        centered = flat - mean
        fitted_rank = max(1, min(self.rank, self.dim, centered.shape[0]))
        _u, _s, v = torch.pca_lowrank(centered, q=fitted_rank, center=False, niter=self.niter)
        basis = v[:, :fitted_rank].contiguous()
        coeff = centered @ basis

        quant_limit = max(1, (1 << (self.bit_width - 1)) - 1)
        scale = coeff.abs().amax(dim=0).clamp(min=1e-6) / float(quant_limit)

        self._fitted_rank = fitted_rank
        self._mean_np = mean.detach().cpu().numpy().astype(np.float32)
        self._basis_np = basis.detach().cpu().numpy().astype(np.float32)
        self._scale_np = scale.detach().cpu().numpy().astype(np.float32)

        # DP adaptive bit allocation
        if self.adaptive_bits:
            bit_budget = fitted_rank * self.bit_width  # same total budget as fixed
            self._per_component_bits = _dp_bit_allocation(coeff, bit_budget)
            # Recompute per-component scales based on assigned bits
            new_scale = np.zeros(fitted_rank, dtype=np.float32)
            coeff_np = coeff.detach().cpu().numpy()
            for j in range(fitted_rank):
                bw = int(self._per_component_bits[j])
                if bw > 0:
                    ql = max(1, (1 << (bw - 1)) - 1)
                    new_scale[j] = max(float(np.max(np.abs(coeff_np[:, j]))), 1e-6) / ql
                else:
                    new_scale[j] = 1.0  # unused, will be zeroed
            self._scale_np = new_scale

        return self._fitted_rank

    def quantize(self, tensor: torch.Tensor) -> PCACompressedTensor:
        if not self.is_fitted:
            self.fit(tensor)
        device = tensor.device
        self._ensure_tensors(device)

        flat = tensor.reshape(-1, self.dim).float()
        coeff = (flat - self._mean) @ self._basis  # (N, rank)

        if self.adaptive_bits and self._per_component_bits is not None:
            # Adaptive: quantize each component with its assigned bit-width
            N = coeff.shape[0]
            shifted = torch.zeros(N, self._fitted_rank, dtype=torch.uint8, device=device)
            max_bw = int(self._per_component_bits.max()) if self._per_component_bits.max() > 0 else self.bit_width
            for bw in set(int(b) for b in self._per_component_bits):
                if bw == 0:
                    continue
                mask = torch.from_numpy(self._per_component_bits == bw).to(device)
                ql = max(1, (1 << (bw - 1)) - 1)
                cols = coeff[:, mask]
                sc = self._scale[mask]
                q = torch.round(cols / sc).clamp(-ql, ql).to(torch.int16)
                shifted[:, mask] = (q + ql).to(torch.uint8)
        else:
            # Fixed bit-width (original behavior)
            quant_limit = max(1, (1 << (self.bit_width - 1)) - 1)
            quantized = torch.round(coeff / self._scale).clamp(-quant_limit, quant_limit).to(torch.int16)
            shifted = (quantized + quant_limit).to(torch.uint8)
            max_bw = self.bit_width

        if self.deflate:
            payload = _pack_indices(shifted, max_bw)
            payload = _maybe_deflate(payload, True, self.deflate_level)
            return PCACompressedTensor(
                indices=None,
                payload=payload,
                shape=tensor.shape,
                dtype=tensor.dtype,
                device=tensor.device,
                rank=self._fitted_rank,
                bit_width=max_bw,
                deflated=True,
                gpu_resident=False,
            )

        # GPU-resident mode: store as uint8 (no sub-byte packing for adaptive)
        return PCACompressedTensor(
            indices=shifted,
            payload=None,
            shape=tensor.shape,
            dtype=tensor.dtype,
            device=tensor.device,
            rank=self._fitted_rank,
            bit_width=max_bw,
            deflated=False,
            gpu_resident=True,
        )

    def dequantize(self, compressed: PCACompressedTensor) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("PCAQuantizedCompressor must be fitted before dequantization.")
        if compressed.rank != self._fitted_rank:
            raise ValueError(
                f"Compressed rank {compressed.rank} does not match fitted rank {self._fitted_rank}."
            )

        device = compressed.device
        self._ensure_tensors(device)

        coeff_count = (int(np.prod(compressed.shape)) // self.dim) * compressed.rank

        if compressed.gpu_resident and compressed.indices is not None:
            # ---- Fast GPU path ----
            shifted = compressed.indices.reshape(-1, compressed.rank).float()
        else:
            # ---- Legacy CPU path ----
            payload = _maybe_inflate(compressed.payload, compressed.deflated)
            shifted = _unpack_indices(payload, coeff_count, compressed.bit_width).reshape(-1, compressed.rank)
            shifted = shifted.to(device=device, dtype=torch.float32)

        if self.adaptive_bits and self._per_component_bits is not None:
            # Adaptive dequantize: per-component bit-width
            coeff = torch.zeros_like(shifted)
            for bw in set(int(b) for b in self._per_component_bits):
                if bw == 0:
                    continue
                mask = torch.from_numpy(self._per_component_bits == bw).to(device)
                ql = max(1, (1 << (bw - 1)) - 1)
                coeff[:, mask] = (shifted[:, mask] - ql) * self._scale[mask]
        else:
            quant_limit = max(1, (1 << (compressed.bit_width - 1)) - 1)
            coeff = (shifted - quant_limit) * self._scale

        recovered = coeff @ self._basis_t + self._mean
        return recovered.reshape(compressed.shape).to(compressed.dtype)

    def project_to_coefficients(self, tensor: torch.Tensor) -> torch.Tensor:
        """Project raw tensor to PCA coefficient space (no quantization).

        Returns (N, rank) float coefficients for distance comparison.
        """
        if not self.is_fitted:
            raise RuntimeError("PCAQuantizedCompressor must be fitted before projection.")
        device = tensor.device
        self._ensure_tensors(device)
        flat = tensor.reshape(-1, self.dim).float()
        return (flat - self._mean) @ self._basis  # (N, rank)

    def extract_coefficients(self, compressed: PCACompressedTensor) -> torch.Tensor:
        """Extract dequantized PCA coefficients without full reconstruction.

        Returns (N, rank) float coefficients — stops before V^T + mean step.
        """
        if not self.is_fitted:
            raise RuntimeError("PCAQuantizedCompressor must be fitted.")

        device = compressed.device
        self._ensure_tensors(device)
        coeff_count = (int(np.prod(compressed.shape)) // self.dim) * compressed.rank

        if compressed.gpu_resident and compressed.indices is not None:
            shifted = compressed.indices.reshape(-1, compressed.rank).float()
        else:
            payload = _maybe_inflate(compressed.payload, compressed.deflated)
            shifted = _unpack_indices(payload, coeff_count, compressed.bit_width).reshape(-1, compressed.rank)
            shifted = shifted.to(device=device, dtype=torch.float32)

        if self.adaptive_bits and self._per_component_bits is not None:
            coeff = torch.zeros_like(shifted)
            for bw in set(int(b) for b in self._per_component_bits):
                if bw == 0:
                    continue
                mask = torch.from_numpy(self._per_component_bits == bw).to(device)
                ql = max(1, (1 << (bw - 1)) - 1)
                coeff[:, mask] = (shifted[:, mask] - ql) * self._scale[mask]
        else:
            quant_limit = max(1, (1 << (compressed.bit_width - 1)) - 1)
            coeff = (shifted - quant_limit) * self._scale

        return coeff  # (N, rank) — no V^T + mean
