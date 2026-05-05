"""Compressed-domain KV encode/decode primitives.

Algorithm (Path B, cda-v1 family):

    encode:
        x_fp16  : (..., D)
        x_rot   = x @ H                       # H: Hadamard rotation (D, D)
        norm    = ||x_rot||₂                  # per-token fp32 scale
        x_unit  = x_rot / norm                # in [-1, 1] approximately
        idx     = argmin_c |x_unit - cb[c]|   # 4-bit: 16 levels
        packed  = pack_4bit(idx)              # 2 dims / byte

    decode:
        x_rot   = cb[idx] * norm              # element-wise
        x       = x_rot @ H^T                 # un-rotate (skipped when
                                                attention runs in rotated coords)

K and V share the encode/decode skeleton; codebooks are independent so
each can be calibrated on its own component distribution. For K4V4 both
codebooks have 16 levels (4-bit) and are packed 2 dims/byte.

GPU-optimization notes:
* Vectorized via torch.argmin / torch.gather; no Python loops.
* Codebooks live on the same device as the data; convert once at construction.
* Pack/unpack uses bitwise ops on contiguous uint8 tensors — torch.compile
  friendly.
* The Hadamard matrix is materialized as a (D, D) fp32 tensor; for D=128
  this is 64 KB (one-time cost). Real kernels apply it via butterfly,
  but the reference uses dense matmul for simplicity + speed on GPU.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

# -----------------------------------------------------------------------------
# Hadamard rotation
# -----------------------------------------------------------------------------


def hadamard_matrix(d: int, *, dtype: torch.dtype = torch.float32,
                     device: torch.device | str | None = None) -> Tensor:
    """Sylvester Hadamard matrix of order ``d``, normalized so ``H @ H.T = I``.

    ``d`` must be a power of two (the Sylvester construction). For non-power
    sizes this raises; pad to power-of-two D before calling.
    """
    if d <= 0 or (d & (d - 1)):
        raise ValueError(f"d must be a positive power of two, got {d}")
    H = torch.tensor([[1.0]], dtype=dtype, device=device)
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], dim=1),
                        torch.cat([H, -H], dim=1)], dim=0)
    H = H / (d ** 0.5)
    return H


def half_rotate_pair(d: int, *, dtype: torch.dtype = torch.float32,
                      device: torch.device | str | None = None) -> Tensor:
    """The cda-v1 "half rotation" pattern: a Hadamard matrix used once on
    each of (Q, K, V) at the boundary, with the inverse applied to the
    attention output. For Sylvester Hadamard normalized to orthonormal,
    ``H == H^T`` so a single tensor suffices for both forward and inverse.
    """
    return hadamard_matrix(d, dtype=dtype, device=device)


# -----------------------------------------------------------------------------
# Codebook construction
# -----------------------------------------------------------------------------


def uniform_centroids(num_levels: int, *, dtype: torch.dtype = torch.float32,
                       device: torch.device | str | None = None) -> Tensor:
    """Uniform centroids over [-1, 1] for ``num_levels`` bins.

    Matches cda-v1's ``make_lloyd_centroids_uniform``. For 16 levels:
        cb = [-15/16, -13/16, ..., 13/16, 15/16]
    """
    edges = torch.linspace(-1.0, 1.0, num_levels + 1, dtype=dtype, device=device)
    return ((edges[:-1] + edges[1:]) * 0.5).contiguous()


def lloyd_max_centroids_beta(
    num_levels: int, d: int, *,
    n_samples: int = 100_000,
    n_iters: int = 50,
    seed: int = 0xCDA,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Lloyd-Max centroids for the post-Hadamard unit-norm distribution.

    For ``x`` uniformly distributed on the (d-1)-sphere, each component
    follows Beta((d-1)/2, (d-1)/2) on [-1, 1]. This is the empirical
    distribution of K/V slots after Hadamard rotation + unit normalization
    in cda. We sample from it and run Lloyd-Max iteration to produce
    near-MSE-optimal centroids. Equivalent to v1's
    ``core.compression._build_codebooks`` (which iterates the Beta CDF
    analytically); both PPL paths use this distribution. The
    ``uniform_centroids`` mode here matches v1's
    ``make_lloyd_centroids_uniform`` test helper — useful as a sanity
    baseline, not for accuracy work.

    Cached by (num_levels, d, seed) at the call site; this is a few-second
    CPU op that is deterministic given the seed.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(n_samples, d, dtype=dtype, device=device, generator=g)
    x_unit = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return lloyd_max_centroids(num_levels, x_unit[:, 0], n_iters=n_iters)


def lloyd_max_centroids(
    num_levels: int,
    samples: Tensor,
    *,
    n_iters: int = 50,
    init: Optional[Tensor] = None,
    tol: float = 1e-7,
) -> Tensor:
    """Iterative Lloyd-Max centroids on a sample distribution.

    Args:
        num_levels: codebook size (e.g. 16 for 4-bit).
        samples: 1-D tensor of empirical samples from the post-Hadamard
            unit-norm component distribution. ~10⁵ samples is plenty.
        n_iters: max EM-style passes.
        init: optional initialization (default: uniform_centroids).
        tol: stop when max centroid shift < tol.

    Returns:
        Sorted centroids (num_levels,) on the same device/dtype as samples.
    """
    samples = samples.flatten()
    if init is None:
        init = uniform_centroids(num_levels, dtype=samples.dtype,
                                  device=samples.device)
    cb = init.clone()
    for _ in range(n_iters):
        # E-step: assign each sample to nearest centroid.
        d2 = (samples.unsqueeze(-1) - cb.unsqueeze(0)).pow_(2)
        idx = d2.argmin(dim=-1)
        # M-step: per-level mean.
        new_cb = torch.empty_like(cb)
        for c in range(num_levels):
            mask = idx == c
            new_cb[c] = samples[mask].mean() if mask.any() else cb[c]
        shift = (new_cb - cb).abs().max().item()
        cb = new_cb
        if shift < tol:
            break
    return cb.sort().values.contiguous()


# -----------------------------------------------------------------------------
# 4-bit pack / unpack
# -----------------------------------------------------------------------------


def pack_4bit(idx: Tensor) -> Tensor:
    """Pack 4-bit indices into uint8 (2 dims / byte, hi<<4 | lo).

    Args:
        idx: (..., D) uint8 with values in [0, 15]. D must be even.

    Returns:
        (..., D // 2) uint8.
    """
    if idx.dtype != torch.uint8:
        idx = idx.to(torch.uint8)
    last = idx.size(-1)
    if last % 2 != 0:
        raise ValueError(f"4-bit pack requires even last dim, got {last}")
    hi = idx[..., 0::2]
    lo = idx[..., 1::2]
    return ((hi << 4) | lo).contiguous()


def unpack_4bit(packed: Tensor, *, d: int) -> Tensor:
    """Inverse of pack_4bit. ``d`` is the (unpacked) trailing dim length."""
    if packed.size(-1) * 2 != d:
        raise ValueError(f"packed last dim {packed.size(-1)} * 2 != d={d}")
    hi = (packed >> 4) & 0xF
    lo = packed & 0xF
    return torch.stack([hi, lo], dim=-1).reshape(*packed.shape[:-1], d).contiguous()


# -----------------------------------------------------------------------------
# Compressor — high-level API
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CompressedSlot:
    """Container for the encoded form of one or more KV slots.

    Shapes (encode of (..., D) input):
        idx   : (..., D // 2) uint8     — 4-bit packed, 2 dims/byte
        norm  : (...,)         fp32     — per-token (or per-row) norm
    """
    idx: Tensor
    norm: Tensor


class Compressor:
    """Stateless encode/decode for one role (K or V).

    Holds:
        rotation : (D, D) fp32          — Hadamard, applied at encode + decode
        codebook : (num_levels,) fp32   — codebook entries
        num_levels : int                — 16 for 4-bit, 4 for 2-bit, ...
        bits      : int                 — 4 (K4) or 2 (V2) or 4 (V4)

    The compressor is **GPU-resident** once :meth:`to` has been called;
    encoding then runs entirely on device.
    """

    def __init__(self, dim: int, *, num_levels: int = 16,
                 codebook: Optional[Tensor] = None,
                 rotation: Optional[Tensor] = None,
                 codebook_mode: str = "lloyd_beta",
                 device: torch.device | str = "cpu"):
        """Construct a compressor for one role (K or V).

        Args:
            dim: hidden size D (e.g. 128 for Llama-3.1).
            num_levels: 16 for 4-bit, 4 for 2-bit.
            codebook: optional pre-computed (num_levels,) tensor; bypasses
                ``codebook_mode``.
            rotation: optional (D, D) rotation tensor; default Hadamard.
            codebook_mode: how to derive default centroids when no
                ``codebook`` is supplied:
                  * ``"lloyd_beta"`` (default) — Lloyd-Max optimal for
                    Beta((D-1)/2, (D-1)/2) (theoretical post-Hadamard
                    distribution); ~0.98 cos roundtrip at 4-bit / D=128.
                    Data-oblivious, deterministic given the seed.
                  * ``"uniform"`` — uniform spacing in [-1, 1]; matches
                    v1's ``make_lloyd_centroids_uniform`` test helper
                    (NOT v1's PPL/vLLM default — that uses the same
                    Beta-prior Lloyd-Max as ``lloyd_beta``). Only 6/16
                    bins are effectively used because the post-Hadamard
                    unit-norm distribution concentrates near 0; ~0.92 cos
                    at 4-bit / D=128. Ablation only.
            device: where the rotation/codebook live.
        """
        if num_levels not in (4, 16):
            raise ValueError(f"num_levels must be 4 or 16 (got {num_levels})")
        self.dim = dim
        self.num_levels = num_levels
        self.bits = 4 if num_levels == 16 else 2
        self.dims_per_byte = 8 // self.bits
        self.bytes_per_row = dim // self.dims_per_byte
        self.codebook_mode = codebook_mode
        if rotation is None:
            rotation = hadamard_matrix(dim, device=device)
        else:
            rotation = rotation.to(device)
        if codebook is None:
            if codebook_mode == "uniform":
                codebook = uniform_centroids(num_levels, device=device)
            elif codebook_mode == "lloyd_beta":
                codebook = lloyd_max_centroids_beta(num_levels, dim,
                                                     device=device)
            else:
                raise ValueError(f"unknown codebook_mode: {codebook_mode}")
        else:
            codebook = codebook.to(device)
        self.rotation = rotation.contiguous()
        self.codebook = codebook.contiguous()

    # -- helpers --------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self.rotation.device

    def to(self, device: torch.device | str) -> "Compressor":
        return Compressor(self.dim, num_levels=self.num_levels,
                           codebook=self.codebook, rotation=self.rotation,
                           codebook_mode=self.codebook_mode, device=device)

    # -- encode / decode ------------------------------------------------------

    def rotate(self, x: Tensor) -> Tensor:
        """Apply the Hadamard rotation to x along its last dim."""
        # cast to fp32 for numerical stability, then back to original dtype
        out = torch.matmul(x.float(), self.rotation)
        return out.to(x.dtype)

    def unrotate(self, x: Tensor) -> Tensor:
        """Inverse Hadamard (== rotation^T == rotation for symmetric H)."""
        out = torch.matmul(x.float(), self.rotation.t())
        return out.to(x.dtype)

    def quantize_unit(self, x_unit: Tensor) -> Tensor:
        """Bucketize x_unit ∈ [-1, 1]^D against the codebook.

        Returns indices (..., D) uint8 in [0, num_levels). Uses
        ``torch.bucketize`` so memory cost is O(numel(x)) instead of
        O(numel(x) × num_levels) (the naive ``argmin |x - cb|`` form).
        """
        cb = self.codebook
        # bucketize → index of first cb entry > x  ∈ [0, num_levels].
        upper = torch.bucketize(x_unit.contiguous(), cb)
        upper.clamp_(0, self.num_levels - 1)
        lower = (upper - 1).clamp_(min=0)
        d_upper = (cb[upper] - x_unit).abs_()
        d_lower = (x_unit - cb[lower]).abs_()
        idx = torch.where(d_upper < d_lower, upper, lower)
        return idx.to(torch.uint8)

    def encode(self, x: Tensor) -> CompressedSlot:
        """Hadamard rotate → unit-normalize → bucketize → pack.

        Args:
            x: (..., D) fp16 / fp32 input. Last dim is D.

        Returns:
            CompressedSlot with idx (..., D // dims_per_byte) uint8 and
            norm (...,) fp32.
        """
        if x.size(-1) != self.dim:
            raise ValueError(f"x last dim {x.size(-1)} != dim={self.dim}")
        x_rot = self.rotate(x).float()
        norm = x_rot.norm(dim=-1).clamp_min(1e-12)
        x_unit = x_rot / norm.unsqueeze(-1)
        idx = self.quantize_unit(x_unit.clamp_(-1.0, 1.0))
        if self.bits == 4:
            packed = pack_4bit(idx)
        else:  # bits == 2
            packed = pack_2bit(idx)
        return CompressedSlot(idx=packed, norm=norm.contiguous())

    def decode(self, slot: CompressedSlot, *, dtype: torch.dtype = torch.float16,
                rotated: bool = False) -> Tensor:
        """Inverse of :meth:`encode`.

        Args:
            slot: encoded form (idx packed, norm fp32).
            dtype: output dtype (fp16 by default for attention).
            rotated: if True, return values in the rotated frame (skip the
                final inverse Hadamard). Set to True when the consumer
                operates on rotated K/V (e.g. compressed-domain attention).

        Returns:
            (..., D) fp16/fp32 reconstruction.
        """
        if self.bits == 4:
            idx = unpack_4bit(slot.idx, d=self.dim)
        else:
            idx = unpack_2bit(slot.idx, d=self.dim)
        # Lookup: (..., D) <- codebook[idx]
        cb = self.codebook
        x_unit = cb[idx.long()]
        x_rot = x_unit * slot.norm.unsqueeze(-1)
        if rotated:
            return x_rot.to(dtype)
        return self.unrotate(x_rot).to(dtype)

    # -- convenience ----------------------------------------------------------

    def compression_ratio(self) -> float:
        """Compression ratio vs fp16 storage of the same K/V slot."""
        # idx: D × bits / 8 bytes  +  norm: 4 bytes
        # fp16:  D × 2 bytes
        encoded = self.dim * self.bits / 8 + 4
        baseline = self.dim * 2
        return baseline / encoded


# 2-bit pack / unpack for ablation (V2 mode).
def pack_2bit(idx: Tensor) -> Tensor:
    """Pack 2-bit indices into uint8 (4 dims / byte, MSB first)."""
    if idx.dtype != torch.uint8:
        idx = idx.to(torch.uint8)
    last = idx.size(-1)
    if last % 4 != 0:
        raise ValueError(f"2-bit pack requires last dim divisible by 4, got {last}")
    a = idx[..., 0::4] << 6
    b = idx[..., 1::4] << 4
    c = idx[..., 2::4] << 2
    d = idx[..., 3::4]
    return (a | b | c | d).contiguous()


def unpack_2bit(packed: Tensor, *, d: int) -> Tensor:
    """Inverse of pack_2bit."""
    if packed.size(-1) * 4 != d:
        raise ValueError(f"packed last dim {packed.size(-1)} * 4 != d={d}")
    a = (packed >> 6) & 0x3
    b = (packed >> 4) & 0x3
    c = (packed >> 2) & 0x3
    e = packed & 0x3
    return torch.stack([a, b, c, e], dim=-1).reshape(*packed.shape[:-1], d).contiguous()
