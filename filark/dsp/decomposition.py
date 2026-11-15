# Copyright (c) 2026 Jintao Li & Weichang Li. 
# Zhejiang University (ZJU).
# 
# Licensed under the MIT License.

"""Patch- and window-based matrix decomposition operators for DAS denoising."""

from __future__ import annotations

from typing import Literal

import numpy as np

from .base import from_nt_nc, require_odd, to_nt_nc
from .chunk import apply_tiled_overlap

Layout = Literal["nt_nc", "nc_nt"]

__all__ = [
    "truncated_svd_reconstruct",
    "local_svd_denoise",
    "local_svd_denoise_chunked",
]


def _window_starts(length: int, window: int, step: int) -> list[int]:
    length = int(length)
    window = max(1, int(window))
    step = max(1, int(step))
    starts = list(range(0, max(1, length - window + 1), step))
    if not starts:
        starts = [0]
    last = max(0, length - window)
    if starts[-1] != last:
        starts.append(last)
    return starts


def truncated_svd_reconstruct(data, *, rank: int, layout: Layout = "nt_nc") -> np.ndarray:
    """Global low-rank reconstruction by truncated SVD."""
    x = to_nt_nc(data, layout=layout, dtype=np.float32)
    r = int(rank)
    if r < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    s[r:] = 0.0
    rec = (u * s[None, :]) @ vt
    return from_nt_nc(rec.astype(np.float32, copy=False), layout=layout)


def local_svd_denoise(
    data,
    *,
    win_t: int = 64,
    win_c: int = 32,
    rank: int = 3,
    step_t: int | None = None,
    step_c: int | None = None,
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """
    Local low-rank denoising with overlap-add averaging.
    """
    x = to_nt_nc(data, layout=layout, dtype=np.float32)
    nt, nc = x.shape

    wt = require_odd("win_t", int(win_t))
    wc = require_odd("win_c", int(win_c))
    r = int(rank)
    if r < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")

    st = max(1, int(step_t) if step_t is not None else wt // 2)
    sc = max(1, int(step_c) if step_c is not None else wc // 2)

    out = np.zeros((nt, nc), dtype=np.float64)
    cnt = np.zeros((nt, nc), dtype=np.float64)

    t_starts = _window_starts(nt, wt, st)
    c_starts = _window_starts(nc, wc, sc)

    for t0 in t_starts:
        t1 = t0 + wt
        for c0 in c_starts:
            c1 = c0 + wc
            patch = x[t0:t1, c0:c1]
            u, s, vt = np.linalg.svd(patch, full_matrices=False)
            s[r:] = 0.0
            rec = (u * s[None, :]) @ vt
            out[t0:t1, c0:c1] += rec
            cnt[t0:t1, c0:c1] += 1.0

    y = out / np.maximum(cnt, 1e-12)
    return from_nt_nc(y.astype(np.float32, copy=False), layout=layout)


def local_svd_denoise_chunked(
    data,
    *,
    win_t: int = 64,
    win_c: int = 32,
    rank: int = 3,
    tile_t: int = 512,
    tile_c: int = 256,
    halo_t: int | None = None,
    halo_c: int | None = None,
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """
    Chunked local-SVD denoising.

    Uses (tile + halo) context and writes only non-overlap core.
    """
    x = to_nt_nc(data, layout=layout, dtype=np.float32)
    nt, nc = x.shape
    tt = max(1, int(tile_t))
    tc = max(1, int(tile_c))
    ht = int(halo_t) if halo_t is not None else max(1, int(win_t) // 2)
    hc = int(halo_c) if halo_c is not None else max(1, int(win_c) // 2)
    ht = max(0, ht)
    hc = max(0, hc)

    def _op(tile: np.ndarray) -> np.ndarray:
        return local_svd_denoise(
            tile,
            win_t=win_t,
            win_c=win_c,
            rank=rank,
            layout="nt_nc",
        )

    y = apply_tiled_overlap(
        x,
        _op,
        tile_t=tt,
        tile_c=tc,
        halo_t=ht,
        halo_c=hc,
        layout="nt_nc",
        dtype=np.float32,
    )

    return from_nt_nc(y, layout=layout)
