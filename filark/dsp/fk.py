# Copyright (c) 2026 Jintao Li & Weichang Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""Frequency-wavenumber transforms and velocity-domain filtering operators."""

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.fft

from .base import as_float2d, require_positive
from .chunk import apply_channel_chunks, estimate_chunk_channels

BackendType = Literal["numpy", "scipy", "torch", "jax", "cupy"]

__all__ = [
    "BackendType",
    "get_fk_axes",
    "fk_transform",
    "apply_velocity_filter",
    "fk_fan_filter",
    "fk_fan_filter_chunked",
    "directional_wedge_denoise",
    "directional_wedge_denoise_chunked",
]


def _get_torch():
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "backend='torch' requires PyTorch.\n"
            "Install it with one of:\n"
            "  pip install torch\n"
            "  pip install 'filark[gpu]'\n"
            "  pip install 'filark[all]'"
        ) from exc
    return torch


def _get_jax_numpy():
    try:
        import jax.numpy as jnp  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "backend='jax' requires JAX.\n"
            "Install a compatible JAX build, for example:\n"
            "  pip install jax"
        ) from exc
    return jnp


def _get_cupy():
    try:
        import cupy as cp  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "backend='cupy' requires CuPy.\n"
            "Install a CuPy build that matches your CUDA runtime, for example:\n"
            "  pip install cupy-cuda12x"
        ) from exc
    return cp


def get_fk_axes(
    n_t: int,
    n_x: int,
    dt: float,
    dx: float,
    centered: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return frequency and wavenumber axes for an [nt, nx] array."""
    n_t = int(n_t)
    n_x = int(n_x)
    dt = require_positive("dt", dt)
    dx = require_positive("dx", dx)
    if n_t < 1 or n_x < 1:
        raise ValueError(f"n_t/n_x must be >= 1, got n_t={n_t}, n_x={n_x}")

    f = np.fft.fftfreq(n_t, d=dt)
    k = np.fft.fftfreq(n_x, d=dx)
    if centered:
        f = np.fft.fftshift(f)
        k = np.fft.fftshift(k)
    return f, k


def _fft2_backend(data, backend: BackendType):
    if backend == "numpy":
        return np.fft.fft2(data)
    if backend == "scipy":
        return scipy.fft.fft2(data, axes=(0, 1))
    if backend == "torch":
        torch = _get_torch()
        x = data if isinstance(data, torch.Tensor) else torch.as_tensor(data)
        return torch.fft.fft2(x)
    if backend == "jax":
        jnp = _get_jax_numpy()
        return jnp.fft.fft2(data)
    if backend == "cupy":
        cp = _get_cupy()
        x = data if isinstance(data, cp.ndarray) else cp.asarray(data)
        return cp.fft.fft2(x)
    raise ValueError(f"Unknown backend: {backend}")


def _fftshift_backend(spec, backend: BackendType):
    if backend == "numpy":
        return np.fft.fftshift(spec)
    if backend == "scipy":
        return scipy.fft.fftshift(spec)
    if backend == "torch":
        torch = _get_torch()
        return torch.fft.fftshift(spec)
    if backend == "jax":
        jnp = _get_jax_numpy()
        return jnp.fft.fftshift(spec)
    if backend == "cupy":
        cp = _get_cupy()
        return cp.fft.fftshift(spec)
    raise ValueError(f"Unknown backend: {backend}")


def _power_backend(spec, backend: BackendType):
    if backend in ("numpy", "scipy"):
        return np.abs(spec) ** 2
    if backend == "torch":
        return spec.real * spec.real + spec.imag * spec.imag
    if backend == "jax":
        return spec.real * spec.real + spec.imag * spec.imag
    if backend == "cupy":
        return spec.real * spec.real + spec.imag * spec.imag
    raise ValueError(f"Unknown backend: {backend}")


def fk_transform(
    data,
    *,
    dt: float = 1.0,
    dx: float = 1.0,
    backend: BackendType = "scipy",
    return_power: bool = False,
    shift: bool = True,
):
    """
    Compute 2D f-k transform for data shaped [nt, nx].

    Returns
    -------
    spectrum, f_axis, k_axis
        `spectrum` follows backend type (numpy/scipy/torch/cupy/jax).
        `f_axis`/`k_axis` are always numpy arrays.
    """
    x = as_float2d(data, dtype=np.float32)
    nt, nx = x.shape
    dt = require_positive("dt", dt)
    dx = require_positive("dx", dx)

    spec = _fft2_backend(x, backend=backend)
    if shift:
        spec = _fftshift_backend(spec, backend=backend)
    if return_power:
        spec = _power_backend(spec, backend=backend)

    f_axis, k_axis = get_fk_axes(nt, nx, dt=dt, dx=dx, centered=shift)
    return spec, f_axis, k_axis


def apply_velocity_filter(
    spectrum: np.ndarray,
    f_axis: np.ndarray,
    k_axis: np.ndarray,
    *,
    min_vel: float | None = None,
    max_vel: float | None = None,
    taper_width: float = 0.0,
    fmin_abs: float | None = None,
    fmax_abs: float | None = None,
    kmin_abs: float | None = None,
    kmax_abs: float | None = None,
) -> np.ndarray:
    """
    Apply a fan-like pass mask in f-k domain.

    Parameters use absolute values for bounds in Hz and cycles/m.
    """
    spec = np.asarray(spectrum)
    f = np.asarray(f_axis, dtype=np.float64).ravel()
    k = np.asarray(k_axis, dtype=np.float64).ravel()
    if spec.ndim != 2:
        raise ValueError(f"spectrum must be 2D, got ndim={spec.ndim}")
    if spec.shape != (f.size, k.size):
        raise ValueError(
            f"spectrum shape mismatch: expected {(f.size, k.size)}, got {spec.shape}"
        )

    tw = max(0.0, float(taper_width))
    F, K = np.meshgrid(f, k, indexing="ij")
    vel = np.abs(F) / (np.abs(K) + 1e-12)
    mask = np.ones(spec.shape, dtype=np.float32)

    if min_vel is not None:
        vmin = require_positive("min_vel", min_vel)
        if tw <= 0:
            mask[vel < vmin] = 0.0
        else:
            below = vel < vmin
            zone = (vel >= vmin) & (vel < (vmin + tw))
            mask[below] = 0.0
            if np.any(zone):
                alpha = (vel[zone] - vmin) / max(tw, 1e-12)
                mask[zone] = 0.5 * (1.0 - np.cos(np.pi * alpha))

    if max_vel is not None:
        vmax = require_positive("max_vel", max_vel)
        if min_vel is not None and vmax <= float(min_vel):
            raise ValueError(f"max_vel must be > min_vel, got {max_vel} <= {min_vel}")
        if tw <= 0:
            mask[vel > vmax] = 0.0
        else:
            above = vel > vmax
            zone = (vel > (vmax - tw)) & (vel <= vmax)
            mask[above] = 0.0
            if np.any(zone):
                alpha = (vel[zone] - (vmax - tw)) / max(tw, 1e-12)
                mask[zone] *= 0.5 * (1.0 + np.cos(np.pi * alpha))

    abs_f = np.abs(F)
    abs_k = np.abs(K)

    if fmin_abs is not None:
        mask[abs_f < max(0.0, float(fmin_abs))] = 0.0
    if fmax_abs is not None:
        vmax_f = max(0.0, float(fmax_abs))
        if fmin_abs is not None and vmax_f <= float(fmin_abs):
            raise ValueError(f"fmax_abs must be > fmin_abs, got {fmax_abs} <= {fmin_abs}")
        mask[abs_f > vmax_f] = 0.0
    if kmin_abs is not None:
        mask[abs_k < max(0.0, float(kmin_abs))] = 0.0
    if kmax_abs is not None:
        vmax_k = max(0.0, float(kmax_abs))
        if kmin_abs is not None and vmax_k <= float(kmin_abs):
            raise ValueError(f"kmax_abs must be > kmin_abs, got {kmax_abs} <= {kmin_abs}")
        mask[abs_k > vmax_k] = 0.0

    return spec * mask


def fk_fan_filter(
    data,
    *,
    dt: float = 1.0,
    dx: float = 1.0,
    vmin: float | None = None,
    vmax: float | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
    taper_width: float = 0.0,
    kmin_abs: float | None = None,
    kmax_abs: float | None = None,
) -> np.ndarray:
    """Apply f-k fan filtering in the time-channel domain."""
    x = as_float2d(data, dtype=np.float32)
    nt, nx = x.shape
    if nt < 2 or nx < 2:
        return x.copy()

    dt = require_positive("dt", dt)
    dx = require_positive("dx", dx)

    spec = scipy.fft.fft2(x, axes=(0, 1))
    f_axis, k_axis = get_fk_axes(nt, nx, dt=dt, dx=dx, centered=False)
    filt = apply_velocity_filter(
        spec,
        f_axis,
        k_axis,
        min_vel=vmin,
        max_vel=vmax,
        taper_width=taper_width,
        fmin_abs=fmin,
        fmax_abs=fmax,
        kmin_abs=kmin_abs,
        kmax_abs=kmax_abs,
    )
    y = scipy.fft.ifft2(filt, axes=(0, 1)).real
    return np.asarray(y, dtype=np.float32)


def fk_fan_filter_chunked(
    data,
    *,
    dt: float = 1.0,
    dx: float = 1.0,
    vmin: float | None = None,
    vmax: float | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
    taper_width: float = 0.0,
    kmin_abs: float | None = None,
    kmax_abs: float | None = None,
    chunk_channels: int | None = None,
    overlap_channels: int = 64,
    target_chunk_mb: float = 128.0,
) -> np.ndarray:
    """
    Chunked f-k fan filtering.

    Notes
    -----
    f-k filtering couples channels in k-domain. Chunked execution is approximate;
    increase `overlap_channels` to reduce seams.
    """
    x = as_float2d(data, dtype=np.float32)
    nt, _ = x.shape
    chunk = (
        estimate_chunk_channels(nt, target_chunk_mb=target_chunk_mb)
        if chunk_channels is None
        else max(1, int(chunk_channels))
    )
    ov = max(0, int(overlap_channels))

    def _op(block: np.ndarray) -> np.ndarray:
        return fk_fan_filter(
            block,
            dt=dt,
            dx=dx,
            vmin=vmin,
            vmax=vmax,
            fmin=fmin,
            fmax=fmax,
            taper_width=taper_width,
            kmin_abs=kmin_abs,
            kmax_abs=kmax_abs,
        )

    return apply_channel_chunks(
        x,
        _op,
        chunk_channels=chunk,
        overlap_channels=ov,
        layout="nt_nc",
        dtype=np.float32,
    )


def directional_wedge_denoise(
    data,
    *,
    n_angles: int = 16,
    threshold: float = 3.0,
    wedge_sigma: float = 0.12,
    soft_threshold: bool = False,
) -> np.ndarray:
    """
    Directional denoising in 2-D Fourier space using angle wedges.
    """
    x = as_float2d(data, dtype=np.float32)
    nt, nc = x.shape
    if nt < 2 or nc < 2:
        return x.copy()

    n_angles = max(2, int(n_angles))
    thr = max(0.0, float(threshold))
    ws = require_positive("wedge_sigma", wedge_sigma)

    spec = np.fft.fftshift(np.fft.fft2(x))
    fy = np.linspace(-1.0, 1.0, nt, dtype=np.float64)[:, None]
    kx = np.linspace(-1.0, 1.0, nc, dtype=np.float64)[None, :]
    angle_map = np.arctan2(fy, kx)

    out = np.zeros_like(spec, dtype=np.complex128)
    weight = np.zeros(spec.shape, dtype=np.float64)
    angles = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_angles, dtype=np.float64)
    for a in angles:
        mask = np.exp(-((angle_map - a) ** 2) / (ws * ws))
        wedge = spec * mask
        sigma = float(np.median(np.abs(wedge))) / 0.6745
        cutoff = thr * sigma
        if cutoff > 0:
            if soft_threshold:
                amp = np.abs(wedge)
                wedge *= np.maximum(0.0, amp - cutoff) / (amp + 1e-12)
            else:
                wedge[np.abs(wedge) < cutoff] = 0.0
        out += wedge
        weight += mask

    # Normalize by total mask weight so that threshold=0 gives identity
    out /= (weight + 1e-12)

    y = np.fft.ifft2(np.fft.ifftshift(out)).real
    return np.asarray(y, dtype=np.float32)


def directional_wedge_denoise_chunked(
    data,
    *,
    n_angles: int = 16,
    threshold: float = 3.0,
    wedge_sigma: float = 0.12,
    soft_threshold: bool = False,
    chunk_channels: int | None = None,
    overlap_channels: int = 64,
    target_chunk_mb: float = 128.0,
) -> np.ndarray:
    """
    Chunked directional wedge denoising.

    Directional filtering is coupled across channels, so this is approximate.
    """
    x = as_float2d(data, dtype=np.float32)
    nt, _ = x.shape
    chunk = (
        estimate_chunk_channels(nt, target_chunk_mb=target_chunk_mb)
        if chunk_channels is None
        else max(1, int(chunk_channels))
    )
    ov = max(0, int(overlap_channels))

    def _op(block: np.ndarray) -> np.ndarray:
        return directional_wedge_denoise(
            block,
            n_angles=n_angles,
            threshold=threshold,
            wedge_sigma=wedge_sigma,
            soft_threshold=soft_threshold,
        )

    return apply_channel_chunks(
        x,
        _op,
        chunk_channels=chunk,
        overlap_channels=ov,
        layout="nt_nc",
        dtype=np.float32,
    )
