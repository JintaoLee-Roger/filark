# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""GPU cross-correlation operators."""

from __future__ import annotations

import numpy as np

from ..base import Layout, to_nt_nc
from ._utils import get_device, require_torch, to_numpy, to_tensor

__all__ = ["xcorr_gather_gpu", "xcorr_all_pairs_gpu"]


def xcorr_gather_gpu(
    data,
    *,
    ref_channel: int = 0,
    max_lag: int | None = None,
    normalize: bool = True,
    device: str = "auto",
    layout: Layout = "nt_nc",
) -> tuple[np.ndarray, np.ndarray]:
    """GPU virtual-source cross-correlation gather.

    Drop-in GPU replacement for :func:`xcorr.xcorr_gather`.

    Returns
    -------
    corr : np.ndarray, shape [2*max_lag+1, nc]
    lags : np.ndarray, shape [2*max_lag+1]
    """
    torch = require_torch()
    x_np = to_nt_nc(data, layout=layout, dtype=np.float32)
    nt, nc = x_np.shape
    ref = int(ref_channel)
    if not (0 <= ref < nc):
        raise ValueError(f"ref_channel={ref} out of range [0, {nc})")

    n_fft = int(2 ** np.ceil(np.log2(2 * nt - 1)))
    dev = get_device(device)
    x_t = to_tensor(x_np, dev)

    X = torch.fft.rfft(x_t, n=n_fft, dim=0)           # [nfreq, nc]
    B = X[:, ref]                                       # [nfreq]
    cc_freq = X * torch.conj(B).unsqueeze(1)           # [nfreq, nc]
    cc_full = torch.fft.irfft(cc_freq, n=n_fft, dim=0) # [n_fft, nc]

    pos = cc_full[:nt, :]
    neg = cc_full[n_fft - nt + 1:, :]
    cc = torch.cat([torch.flip(neg, dims=[0]), pos], dim=0)  # [2*nt-1, nc]

    ml = min(int(max_lag), nt - 1) if max_lag is not None else nt - 1
    center = nt - 1
    cc = cc[center - ml: center + ml + 1, :]           # [n_lag, nc]

    if normalize:
        norm_ref = float(torch.sqrt(torch.sum(x_t[:, ref] ** 2)).item()) + 1e-12
        norm_ch = torch.sqrt(torch.sum(x_t ** 2, dim=0)) + 1e-12
        cc = cc / (norm_ref * norm_ch.unsqueeze(0))

    corr = to_numpy(cc).astype(np.float32)
    lags = np.arange(-ml, ml + 1)
    return corr, lags


def xcorr_all_pairs_gpu(
    data,
    *,
    max_lag: int | None = None,
    normalize: bool = True,
    device: str = "auto",
    layout: Layout = "nt_nc",
    batch_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """GPU all-pairs cross-correlation.

    FFTs are computed once on GPU; pairwise products are batched to bound
    VRAM usage.

    Returns
    -------
    corr : np.ndarray, shape [n_pairs, 2*max_lag+1]
    lags : np.ndarray, shape [2*max_lag+1]
    pairs : list of (i, j) with i < j
    """
    torch = require_torch()
    x_np = to_nt_nc(data, layout=layout, dtype=np.float32)
    nt, nc = x_np.shape

    n_fft = int(2 ** np.ceil(np.log2(2 * nt - 1)))
    dev = get_device(device)
    x_t = to_tensor(x_np, dev)
    X = torch.fft.rfft(x_t, n=n_fft, dim=0)           # [nfreq, nc]
    norms = torch.sqrt(torch.sum(x_t ** 2, dim=0)) + 1e-12

    pairs = [(i, j) for i in range(nc) for j in range(i + 1, nc)]
    n_pairs = len(pairs)
    ml = min(int(max_lag), nt - 1) if max_lag is not None else nt - 1
    n_lag = 2 * ml + 1
    center = nt - 1
    lags = np.arange(-ml, ml + 1)
    corr = np.zeros((n_pairs, n_lag), dtype=np.float32)

    for start in range(0, n_pairs, batch_size):
        end = min(start + batch_size, n_pairs)
        batch = pairs[start:end]
        i_idx = torch.tensor([p[0] for p in batch], device=dev)
        j_idx = torch.tensor([p[1] for p in batch], device=dev)

        Ai = X[:, i_idx]
        Bj = X[:, j_idx]
        cc_full = torch.fft.irfft(Ai * torch.conj(Bj), n=n_fft, dim=0)  # [n_fft, b]

        pos = cc_full[:nt, :]
        neg = cc_full[n_fft - nt + 1:, :]
        cc = torch.cat([torch.flip(neg, dims=[0]), pos], dim=0)
        cc = cc[center - ml: center + ml + 1, :].T     # [b, n_lag]

        if normalize:
            ni = norms[i_idx]
            nj = norms[j_idx]
            cc = cc / (ni * nj).unsqueeze(1)

        corr[start:end] = to_numpy(cc).astype(np.float32)

    return corr, lags, pairs
