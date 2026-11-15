# Copyright (c) 2026 Jintao Li & Weichang Li. 
# Zhejiang University (ZJU).
# 
# Licensed under the MIT License.

"""Nonlinear diffusion-style operators that preserve coherent structures."""

from __future__ import annotations

from typing import Literal

import numpy as np

from .base import from_nt_nc, require_positive, to_nt_nc
from .chunk import apply_tiled_overlap

Layout = Literal["nt_nc", "nc_nt"]

__all__ = [
    "structure_tensor_filter",
    "structure_tensor_filter_chunked",
    "anisotropic_diffusion_2d",
]


def structure_tensor_filter(
    data,
    *,
    n_iter: int = 20,
    kappa: float = 20.0,
    gamma: float = 0.1,
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """
    Perona-Malik anisotropic diffusion along both time and channel axes.

    The diffusion coefficient c = exp(-(|∇u|/kappa)^2) is large in smooth
    regions and near-zero at edges, preserving coherent events while
    suppressing incoherent noise.  gamma controls the step size and must
    be <= 0.25 for numerical stability.

    Note: despite the historical alias `structure_tensor_filter`, this
    implements scalar Perona-Malik diffusion, not eigenvalue-based
    structure-tensor filtering.
    """
    x = to_nt_nc(data, layout=layout, dtype=np.float32)
    nit = int(n_iter)
    kap = require_positive("kappa", kappa)
    gam = float(gamma)
    if nit < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter}")
    if not np.isfinite(gam) or gam <= 0.0:
        raise ValueError(f"gamma must be finite and > 0, got {gamma}")
    if gam > 0.25:
        raise ValueError(f"gamma should be <= 0.25 for stability, got {gamma}")

    y = np.asarray(x, dtype=np.float64)
    for _ in range(nit):
        gx, gy = np.gradient(y)
        mag = np.sqrt(gx * gx + gy * gy + 1e-12)
        c = np.exp(-((mag / kap) ** 2))
        div = np.gradient(c * gx, axis=0) + np.gradient(c * gy, axis=1)
        y += gam * div

    return from_nt_nc(y.astype(np.float32, copy=False), layout=layout)


def structure_tensor_filter_chunked(
    data,
    *,
    n_iter: int = 20,
    kappa: float = 20.0,
    gamma: float = 0.1,
    tile_t: int = 1024,
    tile_c: int = 512,
    halo_t: int | None = None,
    halo_c: int | None = None,
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """
    Chunked structure-tensor diffusion using 2-D tiles + halo.

    This is approximate at tile boundaries; use larger halos to reduce seams.
    """
    x = to_nt_nc(data, layout=layout, dtype=np.float32)
    ht = max(0, int(halo_t) if halo_t is not None else max(8, int(np.ceil(np.sqrt(max(1, n_iter))))))
    hc = max(0, int(halo_c) if halo_c is not None else max(16, int(np.ceil(np.sqrt(max(1, n_iter))) * 2)))

    def _op(tile: np.ndarray) -> np.ndarray:
        return structure_tensor_filter(
            tile,
            n_iter=n_iter,
            kappa=kappa,
            gamma=gamma,
            layout="nt_nc",
        )

    y = apply_tiled_overlap(
        x,
        _op,
        tile_t=max(1, int(tile_t)),
        tile_c=max(1, int(tile_c)),
        halo_t=ht,
        halo_c=hc,
        layout="nt_nc",
        dtype=np.float32,
    )
    return from_nt_nc(y, layout=layout)


def anisotropic_diffusion_2d(
    data,
    *,
    n_iter: int = 20,
    kappa: float = 20.0,
    gamma: float = 0.1,
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """Alias of `structure_tensor_filter` for naming convenience."""
    return structure_tensor_filter(
        data,
        n_iter=n_iter,
        kappa=kappa,
        gamma=gamma,
        layout=layout,
    )
