# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""GPU f-k domain operators."""

from __future__ import annotations

import numpy as np

from ..base import Layout, from_nt_nc, require_positive, to_nt_nc
from ._utils import get_device, require_torch, to_numpy, to_tensor

__all__ = ["fk_fan_filter_gpu"]


def fk_fan_filter_gpu(
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
    device: str = "auto",
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """GPU f-k fan filter via 2-D FFT. Equivalent to `fk_fan_filter`."""
    torch = require_torch()
    x_np = to_nt_nc(data, layout=layout, dtype=np.float32)
    nt, nx = x_np.shape
    if nt < 2 or nx < 2:
        return x_np.copy()

    dt_v = require_positive("dt", dt)
    dx_v = require_positive("dx", dx)

    dev = get_device(device)
    x_t = to_tensor(x_np, dev)
    # The input is real and the mask depends only on |f| and |k|, so the half
    # spectrum is enough. Everything is released as soon as it has been used.
    X = torch.fft.rfft2(x_t)
    del x_t

    # Build mask entirely on GPU — avoids large CPU numpy arrays and
    # a second host-to-device transfer (the original bottleneck for large data).
    pi = torch.tensor(torch.pi, device=dev, dtype=torch.float32)
    tw = max(0.0, float(taper_width))

    abs_f = torch.fft.fftfreq(nt, d=dt_v, device=dev, dtype=torch.float32).abs()
    abs_k = torch.fft.rfftfreq(nx, d=dx_v, device=dev, dtype=torch.float32)

    mask = torch.ones(X.shape, device=dev, dtype=torch.float32)

    def _apply_vel_mask_gpu(m, v, v_lo, v_hi):
        if v_lo is not None:
            v_lo = float(v_lo)
            if tw > 0:
                m[v < v_lo] = 0.0
                zone = (v >= v_lo) & (v < v_lo + tw)
                if zone.any():
                    alpha = (v[zone] - v_lo) / max(tw, 1e-12)
                    m[zone] = 0.5 * (1.0 - torch.cos(pi * alpha))
            else:
                m[v < v_lo] = 0.0
        if v_hi is not None:
            v_hi = float(v_hi)
            if tw > 0:
                m[v > v_hi] = 0.0
                zone = (v > v_hi - tw) & (v <= v_hi)
                if zone.any():
                    alpha = (v[zone] - (v_hi - tw)) / max(tw, 1e-12)
                    m[zone] *= 0.5 * (1.0 + torch.cos(pi * alpha))
            else:
                m[v > v_hi] = 0.0
        return m

    if vmin is not None or vmax is not None:
        vel = abs_f[:, None] / (abs_k[None, :] + 1e-12)
        mask = _apply_vel_mask_gpu(mask, vel, vmin, vmax)
        del vel

    if fmin is not None:
        mask[abs_f < max(0.0, float(fmin)), :] = 0.0
    if fmax is not None:
        mask[abs_f > max(0.0, float(fmax)), :] = 0.0
    if kmin_abs is not None:
        mask[:, abs_k < max(0.0, float(kmin_abs))] = 0.0
    if kmax_abs is not None:
        mask[:, abs_k > max(0.0, float(kmax_abs))] = 0.0

    X *= mask
    del mask
    y_np = to_numpy(torch.fft.irfft2(X, s=(nt, nx))).astype(np.float32, copy=False)
    return from_nt_nc(y_np, layout=layout)
