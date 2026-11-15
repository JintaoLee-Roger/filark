# Copyright (c) 2026 Jintao Li & Weichang Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""Spatial-domain smoothing and normalization operators along channels."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d

from .base import Layout, as_float2d, from_nt_nc, require_positive, to_nt_nc

ArrayLike2D = np.ndarray

__all__ = [
    "demean_space",
    "detrend_space",
    "zscore_space",
    "running_mean_space",
    "gaussian_smooth_space",
    "median_smooth_space",
    "equalize_channel_gain",
    "spatial_gradient",
    "spatial_laplacian",
    "decimate_channels",
    "resample_channels",
]


def demean_space(data: ArrayLike2D) -> np.ndarray:
    """Remove spatial mean per time sample (axis=1)."""
    x = as_float2d(data)
    mu = np.mean(x, axis=1, keepdims=True)
    return (x - mu).astype(np.float32, copy=False)


def detrend_space(data: ArrayLike2D, *, kind: Literal["linear", "constant"] = "linear") -> np.ndarray:
    """Detrend along channel axis for each time sample."""
    x = as_float2d(data, dtype=np.float64)
    y = signal.detrend(x, axis=1, type=kind)
    return y.astype(np.float32, copy=False)


def zscore_space(
    data: ArrayLike2D,
    *,
    robust: bool = False,
    eps: float = 1e-8,
) -> np.ndarray:
    """Per-time-sample normalization along channel axis."""
    x = as_float2d(data, dtype=np.float64)
    eps = require_positive("eps", eps)
    if robust:
        med = np.median(x, axis=1, keepdims=True)
        mad = np.median(np.abs(x - med), axis=1, keepdims=True)
        scale = 1.4826 * mad
        y = (x - med) / np.maximum(scale, eps)
    else:
        mu = np.mean(x, axis=1, keepdims=True)
        sd = np.std(x, axis=1, keepdims=True)
        y = (x - mu) / np.maximum(sd, eps)
    return y.astype(np.float32, copy=False)


def running_mean_space(data: ArrayLike2D, *, half_window: int = 4, mode: str = "reflect") -> np.ndarray:
    """Spatial moving average (axis=1)."""
    x = as_float2d(data)
    hw = int(half_window)
    if hw < 1:
        raise ValueError(f"half_window must be >= 1, got {half_window}")
    size = 2 * hw + 1
    y = uniform_filter1d(x, size=size, axis=1, mode=mode)
    return y.astype(np.float32, copy=False)


def gaussian_smooth_space(data: ArrayLike2D, *, sigma: float = 1.5, mode: str = "reflect") -> np.ndarray:
    """Spatial Gaussian smoothing (axis=1)."""
    x = as_float2d(data)
    sigma = require_positive("sigma", sigma)
    y = gaussian_filter1d(x, sigma=sigma, axis=1, mode=mode)
    return y.astype(np.float32, copy=False)


def median_smooth_space(data: ArrayLike2D, *, kernel_size: int = 5) -> np.ndarray:
    """Spatial median smoothing (axis=1), preserves events better than mean."""
    x = as_float2d(data)
    k = int(kernel_size)
    if k < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
    if k % 2 == 0:
        k += 1
    y = median_filter(x, size=(1, k), mode="reflect")
    return y.astype(np.float32, copy=False)


def equalize_channel_gain(
    data: ArrayLike2D,
    *,
    method: Literal["rms", "std", "mad"] = "rms",
    eps: float = 1e-8,
    clip_min: float | None = 0.2,
    clip_max: float | None = 5.0,
) -> np.ndarray:
    """
    Equalize per-channel gain over time.
    Useful for suppressing hot/dead channels before visualization.
    """
    x = as_float2d(data, dtype=np.float64)
    eps = require_positive("eps", eps)

    if method == "rms":
        gain = np.sqrt(np.mean(x * x, axis=0, keepdims=True))
    elif method == "std":
        gain = np.std(x, axis=0, keepdims=True)
    elif method == "mad":
        med = np.median(x, axis=0, keepdims=True)
        gain = 1.4826 * np.median(np.abs(x - med), axis=0, keepdims=True)
    else:
        raise ValueError(f"Unknown method: {method}")

    ref = np.median(gain)
    scale = ref / np.maximum(gain, eps)
    if clip_min is not None or clip_max is not None:
        lo = -np.inf if clip_min is None else float(clip_min)
        hi = np.inf if clip_max is None else float(clip_max)
        scale = np.clip(scale, lo, hi)
    y = x * scale
    return y.astype(np.float32, copy=False)


def spatial_gradient(data: ArrayLike2D, *, dx: float = 1.0, edge_order: int = 1) -> np.ndarray:
    """First derivative along channel axis."""
    x = as_float2d(data, dtype=np.float64)
    dx = require_positive("dx", dx)
    g = np.gradient(x, dx, axis=1, edge_order=int(edge_order))
    return np.asarray(g, dtype=np.float32)


def spatial_laplacian(data: ArrayLike2D, *, dx: float = 1.0) -> np.ndarray:
    """Second derivative along channel axis."""
    x = as_float2d(data, dtype=np.float64)
    dx = require_positive("dx", dx)
    d1 = np.gradient(x, dx, axis=1)
    d2 = np.gradient(d1, dx, axis=1)
    return d2.astype(np.float32, copy=False)


def decimate_channels(data: ArrayLike2D, *, factor: int = 2, offset: int = 0) -> np.ndarray:
    """Simple channel decimation by stride slicing."""
    x = as_float2d(data)
    fac = int(factor)
    off = int(offset)
    if fac < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if off < 0 or off >= fac:
        raise ValueError(f"offset must satisfy 0 <= offset < factor, got offset={offset}, factor={factor}")
    return x[:, off::fac].astype(np.float32, copy=False)


def resample_channels(
    data: ArrayLike2D,
    *,
    out_channels: int,
    kind: Literal["linear", "nearest", "cubic"] = "linear",
) -> np.ndarray:
    """Resample channel axis to target channel count."""
    x = as_float2d(data, dtype=np.float64)
    nc = x.shape[1]
    out_channels = int(out_channels)
    if out_channels < 1:
        raise ValueError(f"out_channels must be >= 1, got {out_channels}")
    if out_channels == nc:
        return x.astype(np.float32, copy=False)

    xin = np.linspace(0.0, 1.0, nc, dtype=np.float64)
    xout = np.linspace(0.0, 1.0, out_channels, dtype=np.float64)

    if kind == "nearest":
        idx = np.round(xout * (nc - 1)).astype(int).clip(0, nc - 1)
        y = x[:, idx]
    elif kind == "linear":
        # Vectorised linear interpolation without deprecated interp1d
        idx_f = xout * (nc - 1)
        idx_lo = np.floor(idx_f).astype(int).clip(0, nc - 2)
        idx_hi = idx_lo + 1
        alpha = idx_f - idx_lo  # [out_channels]
        y = x[:, idx_lo] * (1.0 - alpha) + x[:, idx_hi] * alpha
    elif kind == "cubic":
        from scipy.interpolate import make_interp_spline
        spl = make_interp_spline(xin, x, k=3, axis=1)
        y = spl(xout)
    else:
        raise ValueError(f"Unknown kind: {kind!r}. Choose 'linear', 'nearest', or 'cubic'.")

    return np.asarray(y, dtype=np.float32)
