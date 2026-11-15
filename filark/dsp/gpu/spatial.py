# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
# Licensed under the MIT License.
"""GPU spatial-domain operators along channel axis."""

from __future__ import annotations

import math

import numpy as np

from ..base import Layout, from_nt_nc, require_positive, to_nt_nc
from ._utils import get_device, require_torch, to_numpy, to_tensor

__all__ = [
    "demean_space_gpu",
    "detrend_space_gpu",
    "zscore_space_gpu",
    "running_mean_space_gpu",
    "gaussian_smooth_space_gpu",
    "median_smooth_space_gpu",
    "equalize_channel_gain_gpu",
    "spatial_gradient_gpu",
    "spatial_laplacian_gpu",
    "resample_channels_gpu",
]


def demean_space_gpu(
    data,
    *,
    device: str = "auto",
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """Subtract per-time-sample mean along channel axis (GPU)."""
    torch = require_torch()
    x_np = to_nt_nc(data, layout=layout, dtype=np.float32)
    dev = get_device(device)
    x = to_tensor(x_np, dev)
    y = x - x.mean(dim=1, keepdim=True)
    return from_nt_nc(to_numpy(y).astype(np.float32), layout=layout)


def detrend_space_gpu(
    data,
    *,
    device: str = "auto",
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """Linear detrend along channel axis per time sample (GPU).

    Fit y = a + b*k for each row simultaneously using closed-form least squares.
    """
    torch = require_torch()
    x_np = to_nt_nc(data, layout=layout, dtype=np.float32)
    nt, nc = x_np.shape
    dev = get_device(device)
    x = to_tensor(x_np, dev)  # [nt, nc]

    k = torch.arange(nc, dtype=torch.float32, device=dev)  # [nc]
    # Closed-form: b = (n*sum(k*y) - sum(k)*sum(y)) / (n*sum(k^2) - sum(k)^2)
    n = float(nc)
    sum_k = k.sum()
    sum_k2 = (k * k).sum()
    denom = n * sum_k2 - sum_k * sum_k

    sum_y = x.sum(dim=1)           # [nt]
    sum_ky = (x * k).sum(dim=1)    # [nt]
    b = (n * sum_ky - sum_k * sum_y) / denom          # [nt]
    a = (sum_y - b * sum_k) / n                        # [nt]

    # trend = a[:, None] + b[:, None] * k[None, :]
    trend = a.unsqueeze(1) + b.unsqueeze(1) * k.unsqueeze(0)  # [nt, nc]
    y = x - trend
    return from_nt_nc(to_numpy(y).astype(np.float32), layout=layout)


def zscore_space_gpu(
    data,
    *,
    eps: float = 1e-8,
    device: str = "auto",
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """(x - mean) / (std + eps) along channel axis per time sample (GPU)."""
    torch = require_torch()
    x_np = to_nt_nc(data, layout=layout, dtype=np.float32)
    dev = get_device(device)
    x = to_tensor(x_np, dev)
    mu = x.mean(dim=1, keepdim=True)
    sd = x.std(dim=1, keepdim=True)
    y = (x - mu) / (sd + float(eps))
    return from_nt_nc(to_numpy(y).astype(np.float32), layout=layout)


def running_mean_space_gpu(
    data,
    *,
    half_window: int = 4,
    device: str = "auto",
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """Uniform 1D conv along channel axis (GPU).

    Uses reflect padding and torch.nn.functional.conv1d.
    """
    torch = require_torch()
    import torch.nn.functional as F

    x_np = to_nt_nc(data, layout=layout, dtype=np.float32)
    nt, nc = x_np.shape
    hw = int(half_window)
    if hw < 1:
        raise ValueError(f"half_window must be >= 1, got {half_window}")
    size = 2 * hw + 1
    dev = get_device(device)
    x = to_tensor(x_np, dev)  # [nt, nc]

    # shape for conv1d: [nt, 1, nc]
    x3 = x.unsqueeze(1)
    # reflect pad along last dim
    x3_pad = F.pad(x3, (hw, hw), mode="reflect")
    kernel = torch.ones(1, 1, size, dtype=torch.float32, device=dev) / size
    y3 = F.conv1d(x3_pad, kernel, padding=0)
    y = y3.squeeze(1)  # [nt, nc]
    return from_nt_nc(to_numpy(y).astype(np.float32), layout=layout)


def gaussian_smooth_space_gpu(
    data,
    *,
    sigma: float = 1.5,
    device: str = "auto",
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """1D Gaussian conv along channel axis (GPU)."""
    torch = require_torch()
    import torch.nn.functional as F

    x_np = to_nt_nc(data, layout=layout, dtype=np.float32)
    nt, nc = x_np.shape
    sigma = float(require_positive("sigma", sigma))
    radius = math.ceil(3.0 * sigma)
    dev = get_device(device)
    x = to_tensor(x_np, dev)  # [nt, nc]

    k = torch.arange(-radius, radius + 1, dtype=torch.float32, device=dev)
    kernel = torch.exp(-0.5 * (k / sigma) ** 2)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1)  # [1, 1, ksize]

    x3 = x.unsqueeze(1)  # [nt, 1, nc]
    x3_pad = F.pad(x3, (radius, radius), mode="reflect")
    y3 = F.conv1d(x3_pad, kernel, padding=0)
    y = y3.squeeze(1)  # [nt, nc]
    return from_nt_nc(to_numpy(y).astype(np.float32), layout=layout)


def median_smooth_space_gpu(
    data,
    *,
    kernel_size: int = 5,
    device: str = "auto",
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """1D median filter along channel axis (GPU) using unfold."""
    torch = require_torch()
    import torch.nn.functional as F

    x_np = to_nt_nc(data, layout=layout, dtype=np.float32)
    nt, nc = x_np.shape
    k = int(kernel_size)
    if k < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
    if k % 2 == 0:
        k += 1
    pad = k // 2
    dev = get_device(device)
    x = to_tensor(x_np, dev)  # [nt, nc]

    # Pad channel axis with reflect
    # F.pad on last dim: (pad_left, pad_right)
    x_pad = F.pad(x, (pad, pad), mode="reflect")  # [nt, nc+2*pad]
    # unfold along channel dim
    # x_pad: [nt, nc+2*pad] → unfold(dim=1, size=k, step=1) → [nt, nc, k]
    unfolded = x_pad.unfold(dimension=1, size=k, step=1)  # [nt, nc, k]
    y = unfolded.median(dim=-1).values  # [nt, nc]
    return from_nt_nc(to_numpy(y).astype(np.float32), layout=layout)


def equalize_channel_gain_gpu(
    data,
    *,
    method: str = "rms",
    eps: float = 1e-8,
    clip_min: float = 0.2,
    clip_max: float = 5.0,
    device: str = "auto",
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """Per-channel gain equalization (GPU).

    Compute per-channel amplitude (rms/std/mad along time axis),
    clip gain to [clip_min*median_gain, clip_max*median_gain], divide.
    """
    torch = require_torch()
    x_np = to_nt_nc(data, layout=layout, dtype=np.float32)
    dev = get_device(device)
    x = to_tensor(x_np, dev)  # [nt, nc]
    eps_v = float(eps)

    if method == "rms":
        gain = torch.sqrt((x * x).mean(dim=0, keepdim=True))  # [1, nc]
    elif method == "std":
        gain = x.std(dim=0, keepdim=True)  # [1, nc]
    elif method == "mad":
        med_ch = x.median(dim=0).values  # [nc]
        gain = (x - med_ch.unsqueeze(0)).abs().median(dim=0).values  # [nc]
        gain = (1.4826 * gain).unsqueeze(0)  # [1, nc]
    else:
        raise ValueError(f"Unknown method: {method!r}. Choose 'rms', 'std', or 'mad'.")

    median_gain = gain.median()
    scale = median_gain / (gain + eps_v)
    lo = float(clip_min)
    hi = float(clip_max)
    scale = scale.clamp(lo, hi)
    y = x * scale
    return from_nt_nc(to_numpy(y).astype(np.float32), layout=layout)


def spatial_gradient_gpu(
    data,
    *,
    dx: float = 1.0,
    edge_order: int = 1,
    device: str = "auto",
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """Spatial gradient along channel axis (GPU)."""
    torch = require_torch()
    x_np = to_nt_nc(data, layout=layout, dtype=np.float32)
    dx = float(require_positive("dx", dx))
    dev = get_device(device)
    x = to_tensor(x_np, dev)  # [nt, nc]
    nt, nc = x.shape

    # Use torch.gradient if available (PyTorch >= 1.11)
    if hasattr(torch, "gradient"):
        g = torch.gradient(x, spacing=(dx,), dim=1)[0]
    else:
        # Manual central differences with edge handling
        g = torch.zeros_like(x)
        # Interior: central differences
        if nc > 2:
            g[:, 1:-1] = (x[:, 2:] - x[:, :-2]) / (2.0 * dx)
        # Edges: forward/backward differences (edge_order=1) or second-order
        if edge_order == 1:
            if nc > 1:
                g[:, 0] = (x[:, 1] - x[:, 0]) / dx
                g[:, -1] = (x[:, -1] - x[:, -2]) / dx
            else:
                g[:, 0] = 0.0
        else:
            # edge_order=2: use three-point one-sided formulas
            if nc >= 3:
                g[:, 0] = (-3 * x[:, 0] + 4 * x[:, 1] - x[:, 2]) / (2.0 * dx)
                g[:, -1] = (x[:, -3] - 4 * x[:, -2] + 3 * x[:, -1]) / (2.0 * dx)
            elif nc == 2:
                g[:, 0] = (x[:, 1] - x[:, 0]) / dx
                g[:, -1] = (x[:, -1] - x[:, -2]) / dx
            else:
                g[:, 0] = 0.0

    return from_nt_nc(to_numpy(g).astype(np.float32), layout=layout)


def spatial_laplacian_gpu(
    data,
    *,
    dx: float = 1.0,
    device: str = "auto",
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """Second finite difference along channel axis (GPU).

    Interior: (x[:,2:] - 2*x[:,1:-1] + x[:,:-2]) / dx^2.
    Edges padded by replicating the first/last interior value.
    """
    torch = require_torch()
    x_np = to_nt_nc(data, layout=layout, dtype=np.float32)
    dx = float(require_positive("dx", dx))
    dev = get_device(device)
    x = to_tensor(x_np, dev)  # [nt, nc]
    nt, nc = x.shape

    dx2 = dx * dx
    lap = torch.zeros_like(x)
    if nc >= 3:
        interior = (x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]) / dx2
        lap[:, 1:-1] = interior
        # Pad edges by repeating nearest interior value
        lap[:, 0] = interior[:, 0]
        lap[:, -1] = interior[:, -1]
    elif nc == 2:
        v = (x[:, 1] - x[:, 0]) / dx
        lap[:, 0] = v
        lap[:, 1] = v
    # nc == 1: zeros

    return from_nt_nc(to_numpy(lap).astype(np.float32), layout=layout)


def resample_channels_gpu(
    data,
    *,
    out_channels: int,
    kind: str = "linear",
    device: str = "auto",
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """Resample along channel axis to out_channels (GPU).

    Uses torch.nn.functional.interpolate.
    kind='linear' → mode='linear'
    kind='nearest' → mode='nearest'
    kind='cubic' → mode='linear' (bicubic not available for 1D)
    """
    torch = require_torch()
    import torch.nn.functional as F

    x_np = to_nt_nc(data, layout=layout, dtype=np.float32)
    nt, nc = x_np.shape
    out_channels = int(out_channels)
    if out_channels < 1:
        raise ValueError(f"out_channels must be >= 1, got {out_channels}")
    if out_channels == nc:
        return from_nt_nc(x_np.copy(), layout=layout)

    dev = get_device(device)
    x = to_tensor(x_np, dev)  # [nt, nc]

    # interpolate expects [N, C, L] → use [nt, 1, nc]
    x3 = x.unsqueeze(1)  # [nt, 1, nc]

    if kind == "nearest":
        y3 = F.interpolate(x3, size=out_channels, mode="nearest")
    elif kind == "linear":
        y3 = F.interpolate(x3, size=out_channels, mode="linear", align_corners=True)
    elif kind == "cubic":
        # conv1d does not have bicubic for 1D; fall back to linear
        y3 = F.interpolate(x3, size=out_channels, mode="linear", align_corners=True)
    else:
        raise ValueError(f"Unknown kind: {kind!r}. Choose 'linear', 'nearest', or 'cubic'.")

    y = y3.squeeze(1)  # [nt, out_channels]
    return from_nt_nc(to_numpy(y).astype(np.float32), layout=layout)
