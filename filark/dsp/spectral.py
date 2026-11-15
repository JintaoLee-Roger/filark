# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""
Spectral analysis operators for DAS data.

All functions accept 2-D arrays in either "nt_nc" [time, channel] or
"nc_nt" [channel, time] layout (default: "nt_nc").
"""

from __future__ import annotations
from typing import Literal
import numpy as np
from .base import to_nt_nc, require_positive

Layout = Literal["nt_nc", "nc_nt"]

__all__ = [
    "stft_gather",
    "psd_estimate",
    "rms_profile",
    "spectral_centroid",
]


def stft_gather(
    data,
    *,
    fs: float,
    nperseg: int = 256,
    noverlap: int | None = None,
    window: str = "hann",
    layout: Layout = "nt_nc",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Short-Time Fourier Transform of all channels.

    Parameters
    ----------
    data : array-like, shape (nt, nc) or (nc, nt)
    fs : float   Sampling frequency in Hz.
    nperseg : int   FFT window length (samples).
    noverlap : int | None   Overlap (default: nperseg // 2).
    window : str   Window name (passed to scipy.signal.stft).
    layout : "nt_nc" | "nc_nt"

    Returns
    -------
    f : ndarray, shape (nf,)   Frequency axis (Hz), one-sided.
    t : ndarray, shape (n_t,)  Time axis (s), centred on each segment.
    S : ndarray, shape (nc, nf, n_t)  Complex STFT coefficients.
    """
    # Use scipy.signal.stft, loop over channels
    from scipy.signal import stft as _stft

    x = to_nt_nc(data, layout=layout, dtype=np.float32)  # (nt, nc)
    nt, nc = x.shape
    fs = require_positive("fs", float(fs))
    nperseg = int(nperseg)
    if noverlap is None:
        noverlap = nperseg // 2

    # compute for first channel to get axes
    f, t, Z0 = _stft(x[:, 0], fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
    nf, n_t = Z0.shape
    S = np.empty((nc, nf, n_t), dtype=np.complex64)
    S[0] = Z0.astype(np.complex64)
    for ic in range(1, nc):
        _, _, Zi = _stft(x[:, ic], fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
        S[ic] = Zi.astype(np.complex64)

    return np.asarray(f, dtype=np.float32), np.asarray(t, dtype=np.float32), S


def psd_estimate(
    data,
    *,
    fs: float,
    nperseg: int = 256,
    noverlap: int | None = None,
    window: str = "hann",
    layout: Layout = "nt_nc",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Power spectral density estimate (Welch's method) for all channels.

    Parameters
    ----------
    data : array-like, shape (nt, nc) or (nc, nt)
    fs : float
    nperseg : int
    noverlap : int | None   (default: nperseg // 2)
    window : str
    layout : "nt_nc" | "nc_nt"

    Returns
    -------
    f   : ndarray, shape (nf,)       Frequency axis (Hz).
    psd : ndarray, shape (nc, nf)    PSD (V²/Hz), one channel per row.
    """
    from scipy.signal import welch as _welch

    x = to_nt_nc(data, layout=layout, dtype=np.float32)
    nt, nc = x.shape
    fs = require_positive("fs", float(fs))
    if noverlap is None:
        noverlap = nperseg // 2

    f, P0 = _welch(x[:, 0], fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
    nf = len(f)
    psd = np.empty((nc, nf), dtype=np.float32)
    psd[0] = P0.astype(np.float32)
    for ic in range(1, nc):
        _, Pi = _welch(x[:, ic], fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
        psd[ic] = Pi.astype(np.float32)

    return np.asarray(f, dtype=np.float32), psd


def rms_profile(
    data,
    *,
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """
    Root-mean-square amplitude per channel.

    Parameters
    ----------
    data : array-like, shape (nt, nc) or (nc, nt)
    layout : "nt_nc" | "nc_nt"

    Returns
    -------
    rms : ndarray, shape (nc,)
    """
    x = to_nt_nc(data, layout=layout, dtype=np.float32)
    return np.sqrt(np.mean(x ** 2, axis=0)).astype(np.float32)


def spectral_centroid(
    f: np.ndarray,
    psd: np.ndarray,
) -> np.ndarray:
    """
    Spectral centroid (mean frequency) per channel.

    Parameters
    ----------
    f   : ndarray, shape (nf,)
    psd : ndarray, shape (nc, nf)   Power spectral density.

    Returns
    -------
    centroid : ndarray, shape (nc,)   Weighted mean frequency (Hz).
    """
    psd = np.asarray(psd, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    total = psd.sum(axis=1, keepdims=True)
    total = np.where(total == 0, 1.0, total)
    return (psd @ f / total[:, 0]).astype(np.float32)
