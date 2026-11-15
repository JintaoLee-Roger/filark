# Copyright (c) 2026 Jintao Li & Weichang Li. 
# Zhejiang University (ZJU).
# 
# Licensed under the MIT License.

"""Tau-p / Radon-style transforms and sparse inversion utilities."""

from __future__ import annotations

from typing import Literal

import numpy as np

from .base import from_nt_nc, require_positive, to_nt_nc
from .chunk import apply_channel_chunks, estimate_chunk_channels

Layout = Literal["nt_nc", "nc_nt"]

__all__ = [
    "linear_radon_forward",
    "linear_radon_inverse",
    "linear_radon_sparse",
    "linear_radon_sparse_chunked",
    "tp_transform",
    "parabolic_radon_forward",
    "parabolic_radon_inverse",
    "parabolic_radon_sparse",
    "parabolic_radon_sparse_chunked",
]


def _centered_x(nc: int, dx: float) -> np.ndarray:
    return (np.arange(nc, dtype=np.float64) - 0.5 * (nc - 1)) * float(dx)


def _interp_sum_shifts(data_nt_nc: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Sum channels after per-channel temporal shifts (in samples)."""
    nt, nc = data_nt_nc.shape
    t = np.arange(nt, dtype=np.float64)
    out = np.zeros(nt, dtype=np.float64)
    for ic in range(nc):
        out += np.interp(t + shifts[ic], t, data_nt_nc[:, ic], left=0.0, right=0.0)
    return out


def linear_radon_forward(
    data,
    p_values: np.ndarray,
    *,
    dt: float = 1.0,
    dx: float = 1.0,
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """
    Linear Radon (tau-p) forward transform.
    p has unit s/m if dt in s and dx in m.
    """
    x = to_nt_nc(data, layout=layout, dtype=np.float32)
    nt, nc = x.shape
    p = np.asarray(p_values, dtype=np.float64).ravel()
    if p.size == 0:
        raise ValueError("p_values must not be empty")
    dt = require_positive("dt", dt)
    dx = require_positive("dx", dx)

    xs = _centered_x(nc, dx)
    tp = np.zeros((p.size, nt), dtype=np.float64)
    for ip, pv in enumerate(p):
        shifts = (pv * xs) / dt
        tp[ip, :] = _interp_sum_shifts(x, shifts)
    return tp.astype(np.float32, copy=False)


def linear_radon_inverse(
    tp: np.ndarray,
    p_values: np.ndarray,
    *,
    nc: int,
    dt: float = 1.0,
    dx: float = 1.0,
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """Adjoint/back-projection for linear Radon."""
    R = np.asarray(tp, dtype=np.float64)
    if R.ndim != 2:
        raise ValueError(f"tp must be 2D, got ndim={R.ndim}")
    p = np.asarray(p_values, dtype=np.float64).ravel()
    if p.size != R.shape[0]:
        raise ValueError(f"p_values length mismatch: {p.size} != {R.shape[0]}")
    nt = R.shape[1]
    dt = require_positive("dt", dt)
    dx = require_positive("dx", dx)
    if int(nc) < 1:
        raise ValueError(f"nc must be >= 1, got {nc}")

    xs = _centered_x(int(nc), dx)
    t = np.arange(nt, dtype=np.float64)
    out = np.zeros((nt, int(nc)), dtype=np.float64)
    for ip, pv in enumerate(p):
        trace = R[ip]
        for ic in range(int(nc)):
            shift = (pv * xs[ic]) / dt
            out[:, ic] += np.interp(t - shift, t, trace, left=0.0, right=0.0)
    out /= max(1, p.size)
    return from_nt_nc(out.astype(np.float32, copy=False), layout=layout)


def linear_radon_sparse(
    data,
    *,
    pmin: float = -0.003,
    pmax: float = 0.003,
    np_slowness: int = 121,
    threshold_factor: float = 3.0,
    dt: float = 1.0,
    dx: float = 1.0,
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """Simple sparse tau-p denoising via hard-threshold in Radon domain."""
    x = to_nt_nc(data, layout=layout, dtype=np.float32)
    p = np.linspace(float(pmin), float(pmax), int(np_slowness), dtype=np.float64)
    R = linear_radon_forward(x, p, dt=dt, dx=dx, layout="nt_nc")
    thr = float(threshold_factor) * float(np.median(np.abs(R)))
    Rf = np.where(np.abs(R) >= thr, R, 0.0)
    return linear_radon_inverse(Rf, p, nc=x.shape[1], dt=dt, dx=dx, layout=layout)


def linear_radon_sparse_chunked(
    data,
    *,
    pmin: float = -0.003,
    pmax: float = 0.003,
    np_slowness: int = 121,
    threshold_factor: float = 3.0,
    dt: float = 1.0,
    dx: float = 1.0,
    chunk_channels: int | None = None,
    overlap_channels: int = 64,
    target_chunk_mb: float = 128.0,
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """
    Chunked sparse linear Radon denoising.

    The Radon transform is channel-coupled, so chunked execution is approximate.
    """
    x = to_nt_nc(data, layout=layout, dtype=np.float32)
    nt, _ = x.shape
    chunk = (
        estimate_chunk_channels(nt, target_chunk_mb=target_chunk_mb)
        if chunk_channels is None
        else max(1, int(chunk_channels))
    )
    ov = max(0, int(overlap_channels))

    def _op(block: np.ndarray) -> np.ndarray:
        return linear_radon_sparse(
            block,
            pmin=pmin,
            pmax=pmax,
            np_slowness=np_slowness,
            threshold_factor=threshold_factor,
            dt=dt,
            dx=dx,
            layout="nt_nc",
        )

    y = apply_channel_chunks(
        x,
        _op,
        chunk_channels=chunk,
        overlap_channels=ov,
        layout="nt_nc",
        dtype=np.float32,
    )
    return from_nt_nc(y, layout=layout)


def tp_transform(
    data,
    p_values: np.ndarray,
    *,
    dt: float = 1.0,
    dx: float = 1.0,
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """Alias of linear_radon_forward for naming convenience."""
    return linear_radon_forward(data, p_values, dt=dt, dx=dx, layout=layout)


def parabolic_radon_forward(
    data,
    q_values: np.ndarray,
    *,
    dt: float = 1.0,
    dx: float = 1.0,
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """Parabolic Radon forward transform."""
    x = to_nt_nc(data, layout=layout, dtype=np.float32)
    nt, nc = x.shape
    q = np.asarray(q_values, dtype=np.float64).ravel()
    if q.size == 0:
        raise ValueError("q_values must not be empty")
    dt = require_positive("dt", dt)
    dx = require_positive("dx", dx)

    xs = _centered_x(nc, dx)
    rp = np.zeros((q.size, nt), dtype=np.float64)
    for iq, qv in enumerate(q):
        shifts = (qv * (xs ** 2)) / dt
        rp[iq, :] = _interp_sum_shifts(x, shifts)
    return rp.astype(np.float32, copy=False)


def parabolic_radon_inverse(
    rp: np.ndarray,
    q_values: np.ndarray,
    *,
    nc: int,
    dt: float = 1.0,
    dx: float = 1.0,
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """Adjoint/back-projection for parabolic Radon."""
    R = np.asarray(rp, dtype=np.float64)
    if R.ndim != 2:
        raise ValueError(f"rp must be 2D, got ndim={R.ndim}")
    q = np.asarray(q_values, dtype=np.float64).ravel()
    if q.size != R.shape[0]:
        raise ValueError(f"q_values length mismatch: {q.size} != {R.shape[0]}")
    nt = R.shape[1]
    if int(nc) < 1:
        raise ValueError(f"nc must be >= 1, got {nc}")
    dt = require_positive("dt", dt)
    dx = require_positive("dx", dx)

    xs = _centered_x(int(nc), dx)
    t = np.arange(nt, dtype=np.float64)
    out = np.zeros((nt, int(nc)), dtype=np.float64)
    for iq, qv in enumerate(q):
        trace = R[iq]
        for ic in range(int(nc)):
            shift = (qv * (xs[ic] ** 2)) / dt
            out[:, ic] += np.interp(t - shift, t, trace, left=0.0, right=0.0)
    out /= max(1, q.size)
    return from_nt_nc(out.astype(np.float32, copy=False), layout=layout)


def parabolic_radon_sparse(
    data,
    *,
    qmin: float = -0.002,
    qmax: float = 0.002,
    nq: int = 81,
    threshold_factor: float = 3.0,
    dt: float = 1.0,
    dx: float = 1.0,
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """Simple sparse parabolic Radon denoising via hard-threshold."""
    x = to_nt_nc(data, layout=layout, dtype=np.float32)
    q = np.linspace(float(qmin), float(qmax), int(nq), dtype=np.float64)
    R = parabolic_radon_forward(x, q, dt=dt, dx=dx, layout="nt_nc")
    thr = float(threshold_factor) * float(np.median(np.abs(R)))
    Rf = np.where(np.abs(R) >= thr, R, 0.0)
    return parabolic_radon_inverse(Rf, q, nc=x.shape[1], dt=dt, dx=dx, layout=layout)


def parabolic_radon_sparse_chunked(
    data,
    *,
    qmin: float = -0.002,
    qmax: float = 0.002,
    nq: int = 81,
    threshold_factor: float = 3.0,
    dt: float = 1.0,
    dx: float = 1.0,
    chunk_channels: int | None = None,
    overlap_channels: int = 64,
    target_chunk_mb: float = 128.0,
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """
    Chunked sparse parabolic Radon denoising.

    The Radon transform is channel-coupled, so chunked execution is approximate.
    """
    x = to_nt_nc(data, layout=layout, dtype=np.float32)
    nt, _ = x.shape
    chunk = (
        estimate_chunk_channels(nt, target_chunk_mb=target_chunk_mb)
        if chunk_channels is None
        else max(1, int(chunk_channels))
    )
    ov = max(0, int(overlap_channels))

    def _op(block: np.ndarray) -> np.ndarray:
        return parabolic_radon_sparse(
            block,
            qmin=qmin,
            qmax=qmax,
            nq=nq,
            threshold_factor=threshold_factor,
            dt=dt,
            dx=dx,
            layout="nt_nc",
        )

    y = apply_channel_chunks(
        x,
        _op,
        chunk_channels=chunk,
        overlap_channels=ov,
        layout="nt_nc",
        dtype=np.float32,
    )
    return from_nt_nc(y, layout=layout)
