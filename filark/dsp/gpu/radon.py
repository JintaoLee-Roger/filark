# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
# Licensed under the MIT License.
"""GPU Radon / tau-p transforms."""

from __future__ import annotations

import numpy as np

from ..base import Layout, from_nt_nc, require_positive, to_nt_nc
from ._utils import get_device, require_torch, to_numpy, to_tensor

__all__ = [
    "linear_radon_forward_gpu",
    "linear_radon_inverse_gpu",
    "parabolic_radon_forward_gpu",
    "parabolic_radon_inverse_gpu",
    "linear_radon_sparse_gpu",
    "parabolic_radon_sparse_gpu",
]


def _centered_x_gpu(nc: int, dx: float, dev):
    """Channel coordinates centered at zero."""
    torch = require_torch()
    return (torch.arange(nc, dtype=torch.float32, device=dev) - 0.5 * (nc - 1)) * float(dx)


def _radon_forward_gpu(x, shifts_fn, p_values, *, dt, dx, device, layout):
    """Generic forward Radon transform (GPU) for arbitrary shift function.

    shifts_fn(pv, x_centered) → per-channel shifts in sample units [nc]
    """
    torch = require_torch()
    import torch.nn.functional as F

    x_np = to_nt_nc(x, layout=layout, dtype=np.float32)
    nt, nc = x_np.shape
    p = np.asarray(p_values, dtype=np.float64).ravel()
    if p.size == 0:
        raise ValueError("p_values must not be empty")
    dt = float(require_positive("dt", dt))
    dx = float(require_positive("dx", dx))

    dev = get_device(device)
    xd = to_tensor(x_np, dev)  # [nt, nc]

    x_centered = _centered_x_gpu(nc, dx, dev)  # [nc]

    # Normalise channel coordinate to [-1, 1] for grid_sample
    nc_norm = torch.linspace(-1.0, 1.0, nc, device=dev, dtype=torch.float32)  # [nc]
    # time indices: 0..nt-1
    t_idx = torch.arange(nt, dtype=torch.float32, device=dev)  # [nt]

    out = torch.zeros(p.size, nt, dtype=torch.float32, device=dev)

    # input for grid_sample: [1, 1, nt, nc]
    inp = xd.unsqueeze(0).unsqueeze(0)  # [1, 1, nt, nc]

    for ip, pv in enumerate(p):
        # per-channel shift in fractional samples
        shifts = shifts_fn(float(pv), x_centered)  # [nc]

        # For each (it, ic): sample at (t_idx[it] + shifts[ic])
        # grid_sample expects grid in [-1,1]: map time index to [-1,1]
        # H-coord (time): (t_shifted / (nt-1)) * 2 - 1
        # W-coord (channel): nc_norm[ic]

        t_shifted = t_idx.unsqueeze(1) + shifts.unsqueeze(0)  # [nt, nc]
        t_norm = (t_shifted / max(nt - 1, 1)) * 2.0 - 1.0     # [nt, nc]

        # grid: [1, nt, nc, 2]  (W, H) = (nc_norm, t_norm)
        grid = torch.stack(
            [nc_norm.unsqueeze(0).expand(nt, -1), t_norm], dim=-1
        ).unsqueeze(0)  # [1, nt, nc, 2]

        # sample: [1, 1, nt, nc]
        sampled = F.grid_sample(inp, grid, mode="bilinear", padding_mode="zeros",
                                align_corners=True)  # [1, 1, nt, nc]
        # sum over channel dim
        out[ip] = sampled.squeeze(0).squeeze(0).sum(dim=1)  # [nt]

    return to_numpy(out).astype(np.float32)


def _radon_inverse_gpu(R_np, shifts_fn, p_values, *, nc, dt, dx, device, layout):
    """Generic inverse (adjoint/back-projection) Radon transform (GPU)."""
    torch = require_torch()
    import torch.nn.functional as F

    R = np.asarray(R_np, dtype=np.float32)
    if R.ndim != 2:
        raise ValueError(f"Radon domain array must be 2D, got ndim={R.ndim}")
    p = np.asarray(p_values, dtype=np.float64).ravel()
    if p.size != R.shape[0]:
        raise ValueError(f"p_values length mismatch: {p.size} != {R.shape[0]}")
    nt = R.shape[1]
    nc = int(nc)
    dt = float(require_positive("dt", dt))
    dx = float(require_positive("dx", dx))

    dev = get_device(device)
    Rd = to_tensor(R, dev)  # [np, nt]

    x_centered = _centered_x_gpu(nc, dx, dev)  # [nc]

    # time indices: 0..nt-1
    t_idx = torch.arange(nt, dtype=torch.float32, device=dev)  # [nt]
    # W-coord for grid_sample along channel (single column): 0 → normalised to 0
    w_zero = torch.zeros(nt, nc, dtype=torch.float32, device=dev)  # [nt, nc]

    out = torch.zeros(nt, nc, dtype=torch.float32, device=dev)

    for ip, pv in enumerate(p):
        shifts = shifts_fn(float(pv), x_centered)  # [nc]
        # inverse: sample trace at t_idx - shifts[ic]
        t_unshift = t_idx.unsqueeze(1) - shifts.unsqueeze(0)  # [nt, nc]
        t_norm = (t_unshift / max(nt - 1, 1)) * 2.0 - 1.0    # [nt, nc]

        # trace_in: [1, 1, nt, 1]
        trace_in = Rd[ip].view(1, 1, nt, 1)

        # grid: [1, nt, nc, 2], W=0 (single column), H=t_norm
        grid = torch.stack([w_zero, t_norm], dim=-1).unsqueeze(0)  # [1, nt, nc, 2]

        sampled = F.grid_sample(trace_in, grid, mode="bilinear", padding_mode="zeros",
                                align_corners=True)  # [1, 1, nt, nc]
        out += sampled.squeeze(0).squeeze(0)  # [nt, nc]

    out = out / max(1, p.size)
    result = to_numpy(out).astype(np.float32)
    return from_nt_nc(result, layout=layout)


# ─── Linear Radon ──────────────────────────────────────────────────────────────

def linear_radon_forward_gpu(
    data,
    p_values,
    *,
    dt: float = 1.0,
    dx: float = 1.0,
    device: str = "auto",
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """Linear Radon (tau-p) forward transform (GPU).

    Returns shape (len(p_values), nt).
    """
    def _shifts(pv, xc):
        return (pv * xc) / dt

    return _radon_forward_gpu(data, _shifts, p_values, dt=dt, dx=dx,
                               device=device, layout=layout)


def linear_radon_inverse_gpu(
    tp,
    p_values,
    *,
    nc: int,
    dt: float = 1.0,
    dx: float = 1.0,
    device: str = "auto",
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """Adjoint/back-projection for linear Radon (GPU).

    Returns shape (nt, nc).
    """
    def _shifts(pv, xc):
        return (pv * xc) / dt

    return _radon_inverse_gpu(tp, _shifts, p_values, nc=nc, dt=dt, dx=dx,
                               device=device, layout=layout)


# ─── Parabolic Radon ───────────────────────────────────────────────────────────

def parabolic_radon_forward_gpu(
    data,
    q_values,
    *,
    dt: float = 1.0,
    dx: float = 1.0,
    device: str = "auto",
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """Parabolic Radon forward transform (GPU).

    Returns shape (len(q_values), nt).
    """
    def _shifts(qv, xc):
        return (qv * xc * xc) / dt

    return _radon_forward_gpu(data, _shifts, q_values, dt=dt, dx=dx,
                               device=device, layout=layout)


def parabolic_radon_inverse_gpu(
    rp,
    q_values,
    *,
    nc: int,
    dt: float = 1.0,
    dx: float = 1.0,
    device: str = "auto",
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """Adjoint/back-projection for parabolic Radon (GPU).

    Returns shape (nt, nc).
    """
    def _shifts(qv, xc):
        return (qv * xc * xc) / dt

    return _radon_inverse_gpu(rp, _shifts, q_values, nc=nc, dt=dt, dx=dx,
                               device=device, layout=layout)


# ─── Sparse Radon denoising ────────────────────────────────────────────────────

def linear_radon_sparse_gpu(
    data,
    *,
    pmin: float = -0.003,
    pmax: float = 0.003,
    np_slowness: int = 121,
    threshold_factor: float = 3.0,
    dt: float = 1.0,
    dx: float = 1.0,
    device: str = "auto",
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """Sparse linear Radon denoising via hard-threshold (GPU).

    Forward transform → hard threshold → inverse transform.
    """
    x_np = to_nt_nc(data, layout=layout, dtype=np.float32)
    p = np.linspace(float(pmin), float(pmax), int(np_slowness), dtype=np.float64)
    R = linear_radon_forward_gpu(x_np, p, dt=dt, dx=dx, device=device, layout="nt_nc")
    thr = float(threshold_factor) * float(np.median(np.abs(R)))
    Rf = np.where(np.abs(R) >= thr, R, 0.0)
    return linear_radon_inverse_gpu(Rf, p, nc=x_np.shape[1], dt=dt, dx=dx,
                                     device=device, layout=layout)


def parabolic_radon_sparse_gpu(
    data,
    *,
    qmin: float = -0.002,
    qmax: float = 0.002,
    nq: int = 81,
    threshold_factor: float = 3.0,
    dt: float = 1.0,
    dx: float = 1.0,
    device: str = "auto",
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """Sparse parabolic Radon denoising via hard-threshold (GPU).

    Forward transform → hard threshold → inverse transform.
    """
    x_np = to_nt_nc(data, layout=layout, dtype=np.float32)
    q = np.linspace(float(qmin), float(qmax), int(nq), dtype=np.float64)
    R = parabolic_radon_forward_gpu(x_np, q, dt=dt, dx=dx, device=device, layout="nt_nc")
    thr = float(threshold_factor) * float(np.median(np.abs(R)))
    Rf = np.where(np.abs(R) >= thr, R, 0.0)
    return parabolic_radon_inverse_gpu(Rf, q, nc=x_np.shape[1], dt=dt, dx=dx,
                                        device=device, layout=layout)
