# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""GPU decomposition operators."""

from __future__ import annotations

import numpy as np

from ..base import Layout, from_nt_nc, require_odd, to_nt_nc
from ._utils import get_device, require_torch, to_numpy

__all__ = ["local_svd_denoise_gpu"]


def local_svd_denoise_gpu(
    data,
    *,
    win_t: int = 64,
    win_c: int = 32,
    rank: int = 3,
    step_t: int | None = None,
    step_c: int | None = None,
    device: str = "auto",
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """Local SVD denoising with GPU-batched torch.linalg.svd.

    Full-size patches are stacked and SVD-ed in a single batched call.
    Edge patches (smaller than win_t x win_c) are handled individually.
    Overlap-add averaging is performed on CPU.
    """
    from ..decomposition import _window_starts

    torch = require_torch()
    x_np = to_nt_nc(data, layout=layout, dtype=np.float32)
    nt, nc = x_np.shape

    wt = require_odd("win_t", int(win_t))
    wc = require_odd("win_c", int(win_c))
    r = int(rank)
    if r < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    st = max(1, int(step_t) if step_t is not None else wt // 2)
    sc = max(1, int(step_c) if step_c is not None else wc // 2)

    t_starts = _window_starts(nt, wt, st)
    c_starts = _window_starts(nc, wc, sc)

    full_patches, full_positions = [], []
    edge_patches, edge_positions = [], []

    for t0 in t_starts:
        t1 = min(nt, t0 + wt)
        for c0 in c_starts:
            c1 = min(nc, c0 + wc)
            patch = x_np[t0:t1, c0:c1]
            if patch.shape == (wt, wc):
                full_patches.append(patch)
                full_positions.append((t0, c0))
            else:
                edge_patches.append(patch)
                edge_positions.append((t0, c0, t1, c1))

    out = np.zeros((nt, nc), dtype=np.float64)
    cnt = np.zeros((nt, nc), dtype=np.float64)
    dev = get_device(device)

    if full_patches:
        P = torch.tensor(np.stack(full_patches), device=dev)   # [n_full, wt, wc]
        U, S, Vh = torch.linalg.svd(P, full_matrices=False)
        S[:, r:] = 0.0
        rec = (U * S.unsqueeze(-2)) @ Vh                       # [n_full, wt, wc]
        rec_np = to_numpy(rec)
        for i, (t0, c0) in enumerate(full_positions):
            out[t0: t0 + wt, c0: c0 + wc] += rec_np[i]
            cnt[t0: t0 + wt, c0: c0 + wc] += 1.0

    for patch, (t0, c0, t1, c1) in zip(edge_patches, edge_positions):
        dh, dw = t1 - t0, c1 - c0
        P = torch.tensor(patch[None], device=dev)              # [1, dh, dw]
        U, S, Vh = torch.linalg.svd(P, full_matrices=False)
        S[0, r:] = 0.0
        rec = to_numpy(((U * S.unsqueeze(-2)) @ Vh)[0])       # [dh, dw]
        out[t0:t1, c0:c1] += rec[:dh, :dw]
        cnt[t0:t1, c0:c1] += 1.0

    y = out / np.maximum(cnt, 1e-12)
    return from_nt_nc(y.astype(np.float32), layout=layout)
