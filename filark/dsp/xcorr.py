# Copyright (c) 2026 Jintao Li & Weichang Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""
Cross-correlation operators for DAS / ambient-noise seismology.

All functions use FFT-based correlation (O(N log N)) and accept data in
either "nt_nc" or "nc_nt" layout.

Key functions
-------------
xcorr_traces        Cross-correlate two 1-D traces → (corr, lags).
xcorr_gather        All channels × reference channel → (corr [n_lag, nc], lags).
xcorr_all_pairs     Every unique channel pair → (corr [n_pairs, n_lag], lags, pairs).
phase_xcorr         Phase cross-correlation (Schimmel & Paulssen 1997).
ambient_noise_xcorr Optional one-bit / whitening pre-processing + xcorr_gather.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import signal as sp_signal

from .base import Layout, as_float2d, from_nt_nc, require_positive, to_nt_nc

ArrayLike1D = np.ndarray
ArrayLike2D = np.ndarray

__all__ = [
    "xcorr_traces",
    "xcorr_gather",
    "xcorr_all_pairs",
    "phase_xcorr",
    "ambient_noise_xcorr",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _next_fast_len(n: int) -> int:
    """Return next FFT-efficient length >= n."""
    # scipy.fft.next_fast_len is faster than power-of-2 for numpy rfft
    try:
        from scipy.fft import next_fast_len
        return next_fast_len(n)
    except ImportError:
        return int(2 ** np.ceil(np.log2(max(1, n))))


def _xcorr_freq(a_fft: np.ndarray, b_fft: np.ndarray, n_fft: int, nt: int) -> np.ndarray:
    """
    Cross-correlation from pre-computed RFFTs.

    Returns centered array of shape [..., 2*nt-1] (lags -nt+1 … nt-1).
    Leading dimensions from a_fft / b_fft are broadcast-friendly.
    """
    cc_raw = np.fft.irfft(a_fft * np.conj(b_fft), n=n_fft, axis=-1)  # [..., n_fft]
    pos = cc_raw[..., :nt]                         # lags 0 … nt-1
    neg = cc_raw[..., n_fft - nt + 1:]             # lags -(nt-1) … -1
    return np.concatenate([neg[..., ::-1], pos], axis=-1)  # [..., 2*nt-1]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def xcorr_traces(
    a,
    b,
    *,
    max_lag: int | None = None,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cross-correlation of two 1-D traces (FFT-based).

    Parameters
    ----------
    a, b : array-like, shape [nt]
        Input traces of equal length.
    max_lag : int, optional
        Maximum lag in samples. ``None`` → nt - 1.
    normalize : bool
        Normalise by sqrt(energy_a * energy_b) so peak ∈ [-1, 1].

    Returns
    -------
    corr : np.ndarray, shape [2*max_lag+1]
    lags : np.ndarray, shape [2*max_lag+1]
        Lag indices in samples (negative = a leads b).
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    nt = len(a)
    if len(b) != nt:
        raise ValueError(f"Traces must have same length, got {nt} and {len(b)}")

    n_fft = _next_fast_len(2 * nt - 1)
    A = np.fft.rfft(a, n=n_fft)
    B = np.fft.rfft(b, n=n_fft)
    cc = _xcorr_freq(A, B, n_fft, nt)     # [2*nt-1]
    all_lags = np.arange(-(nt - 1), nt)

    if normalize:
        norm = np.sqrt(np.dot(a, a) * np.dot(b, b))
        cc = cc / (norm + 1e-12)

    if max_lag is not None:
        ml = min(int(max_lag), nt - 1)
        if ml < 0:
            raise ValueError(f"max_lag must be >= 0, got {max_lag}")
        center = nt - 1
        cc = cc[center - ml: center + ml + 1]
        all_lags = all_lags[center - ml: center + ml + 1]

    return cc.astype(np.float32), all_lags


def xcorr_gather(
    data,
    *,
    ref_channel: int = 0,
    max_lag: int | None = None,
    normalize: bool = True,
    layout: Layout = "nt_nc",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cross-correlate every channel with a reference channel (virtual-source gather).

    All nc channels are correlated simultaneously using a single FFT pass.

    Parameters
    ----------
    data : array-like, shape [nt, nc] (or [nc, nt] with layout="nc_nt")
    ref_channel : int
        Index of the reference (virtual source) channel.
    max_lag : int, optional
    normalize : bool

    Returns
    -------
    corr : np.ndarray, shape [2*max_lag+1, nc]
    lags : np.ndarray, shape [2*max_lag+1]
    """
    x = to_nt_nc(data, layout=layout, dtype=np.float64)
    nt, nc = x.shape
    ref = int(ref_channel)
    if not (0 <= ref < nc):
        raise ValueError(f"ref_channel={ref} out of range [0, {nc})")

    n_fft = _next_fast_len(2 * nt - 1)
    X = np.fft.rfft(x, n=n_fft, axis=0)          # [n_fft//2+1, nc]
    B = X[:, ref]                                  # [n_fft//2+1]

    cc = _xcorr_freq(X.T, B, n_fft, nt)           # [nc, 2*nt-1]
    cc = cc.T                                      # [2*nt-1, nc]
    all_lags = np.arange(-(nt - 1), nt)

    if normalize:
        norm_ref = np.sqrt(np.sum(x[:, ref] ** 2)) + 1e-12
        norm_ch = np.sqrt(np.sum(x ** 2, axis=0)) + 1e-12   # [nc]
        cc = cc / (norm_ref * norm_ch[None, :])

    if max_lag is not None:
        ml = min(int(max_lag), nt - 1)
        center = nt - 1
        cc = cc[center - ml: center + ml + 1, :]
        all_lags = all_lags[center - ml: center + ml + 1]

    return cc.astype(np.float32), all_lags


def xcorr_all_pairs(
    data,
    *,
    max_lag: int | None = None,
    normalize: bool = True,
    layout: Layout = "nt_nc",
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """
    All-pairs cross-correlation (upper-triangle pairs only).

    FFTs are computed once; pairwise products are batched to keep memory bounded.

    Parameters
    ----------
    data : array-like, shape [nt, nc]
    max_lag : int, optional
    normalize : bool
    batch_size : int
        Number of pairs per batch (tune to fit in RAM).

    Returns
    -------
    corr : np.ndarray, shape [n_pairs, 2*max_lag+1]
    lags : np.ndarray, shape [2*max_lag+1]
    pairs : list of (i, j) tuples, i < j
    """
    x = to_nt_nc(data, layout=layout, dtype=np.float64)
    nt, nc = x.shape

    n_fft = _next_fast_len(2 * nt - 1)
    X = np.fft.rfft(x, n=n_fft, axis=0)           # [n_fft//2+1, nc]
    norms = np.sqrt(np.sum(x ** 2, axis=0)) + 1e-12  # [nc]

    pairs = [(i, j) for i in range(nc) for j in range(i + 1, nc)]
    n_pairs = len(pairs)
    ml = min(int(max_lag), nt - 1) if max_lag is not None else nt - 1
    n_lag = 2 * ml + 1
    center = nt - 1

    corr = np.zeros((n_pairs, n_lag), dtype=np.float32)
    lags = np.arange(-ml, ml + 1)

    for start in range(0, n_pairs, batch_size):
        end = min(start + batch_size, n_pairs)
        batch = pairs[start:end]
        i_idx = [p[0] for p in batch]
        j_idx = [p[1] for p in batch]

        Ai = X[:, i_idx]                           # [nfreq, b]
        Bj = X[:, j_idx]                           # [nfreq, b]
        cc = _xcorr_freq(Ai.T, Bj.T, n_fft, nt)   # [b, 2*nt-1]
        cc = cc[:, center - ml: center + ml + 1]   # [b, n_lag]

        if normalize:
            ni = norms[np.array(i_idx)]             # [b]
            nj = norms[np.array(j_idx)]             # [b]
            cc = cc / (ni * nj)[:, None]

        corr[start:end] = cc.astype(np.float32)

    return corr, lags, pairs


def phase_xcorr(
    a,
    b,
    *,
    max_lag: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Phase cross-correlation (Schimmel & Paulssen, 1997).

    Correlates the instantaneous-phase analytic signals (normalised to unit
    amplitude), making it robust to amplitude imbalances and clipping.

    Returns
    -------
    corr : np.ndarray, shape [2*max_lag+1]
    lags : np.ndarray, shape [2*max_lag+1]
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    nt = len(a)
    if len(b) != nt:
        raise ValueError(f"Traces must have same length, got {nt} and {len(b)}")

    # Analytic signals normalised to unit amplitude
    ha = sp_signal.hilbert(a)
    hb = sp_signal.hilbert(b)
    ha = ha / (np.abs(ha) + 1e-12)
    hb = hb / (np.abs(hb) + 1e-12)

    n_fft = _next_fast_len(2 * nt - 1)
    A = np.fft.rfft(ha.real, n=n_fft)   # PCC uses real part of unit-amp analytic
    B = np.fft.rfft(hb.real, n=n_fft)
    cc = _xcorr_freq(A, B, n_fft, nt)   # [2*nt-1]
    # Scale to [-1, 1]
    cc = cc / (nt + 1e-12)
    all_lags = np.arange(-(nt - 1), nt)

    if max_lag is not None:
        ml = min(int(max_lag), nt - 1)
        center = nt - 1
        cc = cc[center - ml: center + ml + 1]
        all_lags = all_lags[center - ml: center + ml + 1]

    return cc.astype(np.float32), all_lags


def ambient_noise_xcorr(
    data,
    *,
    ref_channel: int = 0,
    max_lag: int | None = None,
    normalize: bool = True,
    one_bit: bool = False,
    whiten: bool = False,
    fs: float | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
    layout: Layout = "nt_nc",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full ambient-noise cross-correlation workflow.

    Optional pre-processing steps applied in order:
    1. Bandpass filter (if fs + fmin/fmax provided and whiten=False)
    2. One-bit normalisation (if one_bit=True)
    3. Spectral whitening (if whiten=True; requires fs)

    Then: xcorr_gather with the selected reference channel.

    Returns
    -------
    corr : np.ndarray, shape [2*max_lag+1, nc]
    lags : np.ndarray, shape [2*max_lag+1]
    """
    # Deferred imports to avoid circular dependency at module load time
    from .temporal import (
        iir_bandpass_time,
        iir_highpass_time,
        iir_lowpass_time,
        one_bit_normalize,
        temporal_whiten,
    )

    x = to_nt_nc(data, layout=layout, dtype=np.float32)

    # Optional bandpass pre-filter (only when not whitening)
    if (fmin is not None or fmax is not None) and not whiten:
        if fs is None:
            raise ValueError("fs must be provided when fmin/fmax are set")
        if fmin is not None and fmax is not None:
            x = iir_bandpass_time(x, fs=fs, fmin=fmin, fmax=fmax)
        elif fmin is None:
            x = iir_lowpass_time(x, fs=fs, fmax=fmax)          # type: ignore[arg-type]
        else:
            x = iir_highpass_time(x, fs=fs, fmin=fmin)         # type: ignore[arg-type]

    if one_bit:
        x = one_bit_normalize(x)

    if whiten:
        if fs is None:
            raise ValueError("fs must be provided when whiten=True")
        x = temporal_whiten(x, fs=fs, fmin=fmin, fmax=fmax)

    return xcorr_gather(x, ref_channel=ref_channel, max_lag=max_lag, normalize=normalize, layout="nt_nc")
