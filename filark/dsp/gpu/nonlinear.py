# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""GPU nonlinear / diffusion operators."""

from __future__ import annotations

import numpy as np

from ..base import Layout, from_nt_nc, require_positive, to_nt_nc
from ._utils import get_device, require_torch, to_numpy, to_tensor

__all__ = ["anisotropic_diffusion_2d_gpu"]


def anisotropic_diffusion_2d_gpu(
    data,
    *,
    n_iter: int = 20,
    kappa: float = 20.0,
    gamma: float = 0.1,
    device: str = "auto",
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """GPU Perona-Malik anisotropic diffusion.

    Equivalent to `anisotropic_diffusion_2d` / `structure_tensor_filter` but
    runs entirely on GPU using finite-difference gradient tensors.

    gamma must be <= 0.25 for numerical stability.
    """
    torch = require_torch()
    x_np = to_nt_nc(data, layout=layout, dtype=np.float32)
    nit = int(n_iter)
    kap = require_positive("kappa", kappa)
    gam = float(gamma)
    if nit < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter}")
    if not np.isfinite(gam) or gam <= 0.0:
        raise ValueError(f"gamma must be finite and > 0, got {gamma}")
    if gam > 0.25:
        raise ValueError(f"gamma should be <= 0.25 for stability, got {gamma}")

    dev = get_device(device)
    y = to_tensor(x_np, dev).double()   # float64 for numerical stability

    for _ in range(nit):
        gy = torch.zeros_like(y)
        gy[1:-1] = 0.5 * (y[2:] - y[:-2])
        gy[0] = y[1] - y[0]
        gy[-1] = y[-1] - y[-2]

        gx = torch.zeros_like(y)
        gx[:, 1:-1] = 0.5 * (y[:, 2:] - y[:, :-2])
        gx[:, 0] = y[:, 1] - y[:, 0]
        gx[:, -1] = y[:, -1] - y[:, -2]

        mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
        c = torch.exp(-((mag / kap) ** 2))

        cgx, cgy = c * gx, c * gy
        div = torch.zeros_like(y)
        div[1:-1] += 0.5 * (cgy[2:] - cgy[:-2])
        div[0] += cgy[1] - cgy[0]
        div[-1] += cgy[-1] - cgy[-2]
        div[:, 1:-1] += 0.5 * (cgx[:, 2:] - cgx[:, :-2])
        div[:, 0] += cgx[:, 1] - cgx[:, 0]
        div[:, -1] += cgx[:, -1] - cgx[:, -2]

        y = y + gam * div

    return from_nt_nc(to_numpy(y.float()).astype(np.float32), layout=layout)
