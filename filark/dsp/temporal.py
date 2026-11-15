# Copyright (c) 2026 Jintao Li & Weichang Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""Temporal DSP operators for DAS data.

Most functions in this module assume the canonical research layout
``nt_nc`` (time x channels).  Higher layers are expected to normalize layouts
before calling into ``dsp`` so the numerical code can stay simple.

The temporal operators are intentionally grouped by semantics:

- basic local operators such as demean, detrend, AGC
- offline filters that can treat one block as self-contained and may use
  zero-phase filtering
- realtime/stateful IIR filters that thread state across successive chunks
- chunked wrappers that preserve the same operator semantics on large arrays

The distinction between offline and realtime IIR is important.  The offline
path optimizes for analysis quality on a self-contained block, while the
realtime path optimizes for causal continuity across a moving stream.
"""

from __future__ import annotations

from typing import Literal
import warnings

import numpy as np
from scipy import fft as sp_fft
from scipy import signal
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, median_filter, uniform_filter1d

try:
    from scipy.signal._sosfilt import _sosfilt as _sosfilt_kernel
except ImportError:
    _sosfilt_kernel = None

from .base import Layout, as_float2d, require_positive
from .chunk import make_chunked

ArrayLike2D = np.ndarray

# Working-block sizes. The small block keeps one or two channels of a long record
# inside the CPU's L2 cache; the envelope block leaves room for FFT vectorization.
_BLOCK_BYTES = 2 * 1024 * 1024
_ENVELOPE_BLOCK_BYTES = 256 * 1024 * 1024

FilterKind = Literal["bandpass", "lowpass", "highpass"]

__all__ = [
    "demean_time",
    "detrend_time",
    "zscore_time",
    "running_mean_time",
    "gaussian_smooth_time",
    "median_smooth_time",
    "median_smooth_time_chunked",
    "iir_filter_time",
    "iir_bandpass_time",
    "iir_lowpass_time",
    "iir_highpass_time",
    "iir_filter_time_chunked",
    "iir_bandpass_time_chunked",
    "iir_lowpass_time_chunked",
    "iir_highpass_time_chunked",
    "bandstop_time",
    "bandstop_time_chunked",
    "notch_time",
    "notch_time_chunked",
    # realtime (stateful, causal) IIR
    "build_iir_sos",
    "iir_filter_time_rt",
    "iir_bandpass_time_rt",
    "iir_lowpass_time_rt",
    "iir_highpass_time_rt",
    "fir_filter_time",
    "fir_bandpass_time",
    "fir_lowpass_time",
    "fir_highpass_time",
    "fir_filter_time_chunked",
    "fir_bandpass_time_chunked",
    "fir_lowpass_time_chunked",
    "fir_highpass_time_chunked",
    "fft_bandpass_time",
    "fft_bandpass_time_chunked",
    "fft_lowpass_time",
    "fft_lowpass_time_chunked",
    "fft_highpass_time",
    "fft_highpass_time_chunked",
    "butterworth_time_filter",
    "bandpass_time",
    "lowpass_time",
    "highpass_time",
    "agc_time",
    "agc_time_chunked",
    "temporal_whiten",
    "one_bit_normalize",
    "clip_amplitude",
    "analytic_envelope",
]


def _validate_filter_band(
    *,
    fs: float,
    kind: FilterKind,
    fmin: float | None,
    fmax: float | None,
) -> tuple[float, float, float]:
    """Normalize and validate filter-band arguments against Nyquist limits."""
    fs = require_positive("fs", fs)
    nyq = 0.5 * fs
    lo = None if fmin is None else float(fmin)
    hi = None if fmax is None else float(fmax)

    if kind == "bandpass":
        if lo is None or hi is None:
            raise ValueError("bandpass requires both fmin and fmax")
        if lo <= 0 or hi <= 0 or hi <= lo:
            raise ValueError(f"Invalid bandpass range: fmin={lo}, fmax={hi}")
        if hi > nyq:
            raise ValueError(f"fmax must be <= Nyquist ({nyq:g}), got {hi}")
    elif kind == "lowpass":
        if hi is None:
            raise ValueError("lowpass requires fmax")
        if hi <= 0 or hi > nyq:
            raise ValueError(f"fmax must be in (0, Nyquist={nyq:g}], got {hi}")
        lo = 0.0
    elif kind == "highpass":
        if lo is None:
            raise ValueError("highpass requires fmin")
        if lo <= 0 or lo >= nyq:
            raise ValueError(f"fmin must be in (0, Nyquist={nyq:g}), got {lo}")
        hi = nyq
    else:
        raise ValueError(f"Unknown filter kind: {kind}")

    return fs, lo if lo is not None else 0.0, hi if hi is not None else nyq


# ---------------------------------------------------------------------------
# Basic time-domain operators
# ---------------------------------------------------------------------------

def demean_time(data: ArrayLike2D) -> np.ndarray:
    """Remove mean of each channel along time axis (axis=0)."""
    x = as_float2d(data, dtype=np.float32)
    mu = np.mean(x, axis=0, keepdims=True)
    return (x - mu).astype(np.float32, copy=False)


def detrend_time(data: ArrayLike2D, *, kind: Literal["linear", "constant"] = "linear") -> np.ndarray:
    """Detrend each channel along time axis.

    Uses the closed-form least-squares fit on small time blocks, so the extra
    memory is the float32 output plus one float64 block. Accumulators and block
    arithmetic stay in double precision.
    """
    x = as_float2d(data, dtype=None)
    if kind not in ("linear", "constant"):
        raise ValueError(f"kind must be 'linear' or 'constant', got {kind!r}")
    nt, nc = x.shape
    if nt < 2 or nc == 0:
        return np.asarray(signal.detrend(x.astype(np.float64), axis=0, type=kind), dtype=np.float32)

    rows = min(nt, max(1, _BLOCK_BYTES // (nc * 8)))
    buf = np.empty((rows, nc), dtype=np.float64)
    t = np.arange(nt, dtype=np.float64) - 0.5 * (nt - 1)
    mean = np.zeros(nc, dtype=np.float64)
    slope = np.zeros(nc, dtype=np.float64)
    for i0 in range(0, nt, rows):
        b = buf[:min(rows, nt - i0)]
        np.copyto(b, x[i0:i0 + rows])
        mean += b.sum(axis=0)
        if kind == "linear":
            slope += t[i0:i0 + rows] @ b
    mean /= nt
    slope /= t @ t

    out = np.empty((nt, nc), dtype=np.float32)
    for i0 in range(0, nt, rows):
        b = buf[:min(rows, nt - i0)]
        np.copyto(b, x[i0:i0 + rows])
        b -= mean
        if kind == "linear":
            b -= t[i0:i0 + rows, None] * slope
        out[i0:i0 + rows] = b
    return out


def zscore_time(
    data: ArrayLike2D,
    *,
    robust: bool = False,
    eps: float = 1e-8,
) -> np.ndarray:
    """Per-channel normalization along time axis."""
    x = as_float2d(data, dtype=np.float64)
    eps = require_positive("eps", eps)
    if robust:
        med = np.median(x, axis=0, keepdims=True)
        mad = np.median(np.abs(x - med), axis=0, keepdims=True)
        scale = 1.4826 * mad
        y = (x - med) / np.maximum(scale, eps)
    else:
        mu = np.mean(x, axis=0, keepdims=True)
        sd = np.std(x, axis=0, keepdims=True)
        y = (x - mu) / np.maximum(sd, eps)
    return y.astype(np.float32, copy=False)


def running_mean_time(data: ArrayLike2D, *, half_window: int = 8, mode: str = "reflect") -> np.ndarray:
    """Temporal moving average for each channel."""
    x = as_float2d(data, dtype=np.float32)
    hw = int(half_window)
    if hw < 1:
        raise ValueError(f"half_window must be >= 1, got {half_window}")
    size = 2 * hw + 1
    y = uniform_filter1d(x, size=size, axis=0, mode=mode)
    return y.astype(np.float32, copy=False)


def gaussian_smooth_time(data: ArrayLike2D, *, sigma: float = 2.0, mode: str = "reflect") -> np.ndarray:
    """Temporal Gaussian smoothing for each channel."""
    x = as_float2d(data, dtype=np.float32)
    sigma = require_positive("sigma", sigma)
    y = gaussian_filter1d(x, sigma=sigma, axis=0, mode=mode)
    return y.astype(np.float32, copy=False)


def median_smooth_time(data: ArrayLike2D, *, kernel_size: int = 5) -> np.ndarray:
    """Temporal median smoothing for each channel (robust to impulse noise)."""
    x = as_float2d(data, dtype=np.float32)
    k = int(kernel_size)
    if k < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
    if k % 2 == 0:
        k += 1
    y = median_filter(x, size=(k, 1), mode="reflect")
    return y.astype(np.float32, copy=False)


def agc_time(
    data: ArrayLike2D,
    *,
    half_window: int = 50,
    method: Literal["rms", "max", "mean"] = "rms",
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Automatic Gain Control (AGC): time-varying per-channel amplitude normalization.

    Divides each sample by a local amplitude estimate in a sliding window.
    Useful for equalizing temporal amplitude variations and enhancing weak arrivals.

    Parameters
    ----------
    half_window : int
        Half-window size in samples; full window = 2*half_window + 1.
    method : "rms" | "max" | "mean"
        Local amplitude estimator.
    eps : float
        Regularisation to avoid division by near-zero.
    """
    x = as_float2d(data, dtype=np.float64)
    hw = int(half_window)
    if hw < 1:
        raise ValueError(f"half_window must be >= 1, got {half_window}")
    eps = require_positive("eps", eps)
    size = 2 * hw + 1

    if method == "rms":
        gain = np.sqrt(uniform_filter1d(x ** 2, size=size, axis=0, mode="reflect"))
    elif method == "max":
        gain = maximum_filter1d(np.abs(x), size=size, axis=0, mode="reflect")
    elif method == "mean":
        gain = uniform_filter1d(np.abs(x), size=size, axis=0, mode="reflect")
    else:
        raise ValueError(f"Unknown method: {method!r}. Choose 'rms', 'max', or 'mean'.")

    y = x / np.maximum(gain, eps)
    return y.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# IIR filtering
# ---------------------------------------------------------------------------

def _sosfilt_inplace(sos: np.ndarray, block: np.ndarray, zi: np.ndarray) -> None:
    """Filter a C-contiguous float64 ``[channels, time]`` block in place.

    ``zi`` has shape ``[channels, sections, 2]`` and is updated in place. SciPy's
    compiled kernel lets one working buffer be reused; the public function, which
    must allocate a copy on every call, is the fallback if the kernel moves.
    """
    if _sosfilt_kernel is not None:
        _sosfilt_kernel(sos, block, zi)
        return
    y, zf = signal.sosfilt(sos, block, axis=-1, zi=np.moveaxis(zi, 1, 0))
    block[...] = y
    zi[...] = np.moveaxis(zf, 0, 1)


def iir_filter_time(
    data: ArrayLike2D,
    *,
    fs: float,
    kind: FilterKind,
    fmin: float | None = None,
    fmax: float | None = None,
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    """IIR Butterworth filtering along the time axis.

    ``zero_phase=True`` is the preferred research/offline path because it
    avoids phase distortion when the whole block is available.  For truly
    streaming scenarios use the realtime helpers below instead of repeatedly
    calling this function on adjacent chunks.
    """
    x = as_float2d(data, dtype=None)
    order = int(order)
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")

    fs, lo, hi = _validate_filter_band(fs=fs, kind=kind, fmin=fmin, fmax=fmax)
    nyq = 0.5 * fs

    if kind == "bandpass":
        wn = [lo / nyq, hi / nyq]
        btype = "bandpass"
    elif kind == "lowpass":
        wn = hi / nyq
        btype = "lowpass"
    else:
        wn = lo / nyq
        btype = "highpass"

    sos = signal.butter(order, wn, btype=btype, output="sos")

    # Channels are filtered independently, so process small blocks in reusable
    # float64 buffers and write each block straight into the float32 output.
    # Peak extra memory is the output plus the buffers, and values are
    # bit-identical to scipy.signal.sosfilt / sosfiltfilt in double precision.
    nt, nc = x.shape
    n_sections = sos.shape[0]
    # sosfiltfilt's default odd padding length.
    n_taps = 2 * n_sections + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())
    edge = 3 * int(n_taps)
    if zero_phase and nt <= edge:
        warnings.warn(
            "sosfiltfilt failed due to short sequence; fallback to causal sosfilt.",
            RuntimeWarning,
        )
        zero_phase = False

    # Fortran order keeps each channel contiguous, so writing a block is a plain
    # sequential copy; a C-order output would make it a large-stride write.
    out = np.empty((nt, nc), dtype=np.float32, order="F")
    length = nt + 2 * edge if zero_phase else nt
    step = min(max(nc, 1), max(1, _BLOCK_BYTES // max(1, length * 8)))
    work = np.empty((step, length), dtype=np.float64)
    state = np.empty((step, n_sections, 2), dtype=np.float64)

    if not zero_phase:
        for c0 in range(0, nc, step):
            m = min(step, nc - c0)
            b, z = work[:m], state[:m]
            np.copyto(b, x[:, c0:c0 + m].T)
            z[...] = 0.0
            _sosfilt_inplace(sos, b, z)
            out[:, c0:c0 + m] = b.T
        return out

    # Same steps as scipy.signal.sosfiltfilt, on reusable buffers.
    reverse = np.empty_like(work)
    zi = signal.sosfilt_zi(sos)[None]
    for c0 in range(0, nc, step):
        m = min(step, nc - c0)
        b, r, z = work[:m], reverse[:m], state[:m]
        np.copyto(b[:, edge:edge + nt], x[:, c0:c0 + m].T)
        first, last = b[:, edge].copy()[:, None], b[:, edge + nt - 1].copy()[:, None]
        b[:, :edge] = 2.0 * first - b[:, 2 * edge:edge:-1]
        b[:, edge + nt:] = 2.0 * last - b[:, edge + nt - 2:nt - 2:-1]
        z[...] = zi * b[:, 0, None, None]
        _sosfilt_inplace(sos, b, z)
        z[...] = zi * b[:, -1, None, None]
        np.copyto(r, b[:, ::-1])
        _sosfilt_inplace(sos, r, z)
        out[:, c0:c0 + m] = r[:, edge + nt - 1:edge - 1:-1].T
    return out


def iir_bandpass_time(
    data: ArrayLike2D,
    *,
    fs: float,
    fmin: float,
    fmax: float,
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    return iir_filter_time(data, fs=fs, kind="bandpass", fmin=fmin, fmax=fmax, order=order, zero_phase=zero_phase)


def iir_lowpass_time(
    data: ArrayLike2D,
    *,
    fs: float,
    fmax: float,
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    return iir_filter_time(data, fs=fs, kind="lowpass", fmax=fmax, order=order, zero_phase=zero_phase)


def iir_highpass_time(
    data: ArrayLike2D,
    *,
    fs: float,
    fmin: float,
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    return iir_filter_time(data, fs=fs, kind="highpass", fmin=fmin, order=order, zero_phase=zero_phase)


# ---------------------------------------------------------------------------
# Realtime (stateful, causal) IIR filtering
# ---------------------------------------------------------------------------

def build_iir_sos(
    *,
    fs: float,
    kind: FilterKind,
    fmin: float | None = None,
    fmax: float | None = None,
    order: int = 4,
) -> np.ndarray:
    """Build a Butterworth SOS matrix for realtime IIR filtering.

    Pre-computes filter coefficients once so they can be reused across
    successive chunk calls without recomputing on every invocation.  This is
    the intended entry point for realtime/stateful IIR pipelines.

    Parameters
    ----------
    fs : float
        Sampling rate (Hz).
    kind : "bandpass" | "lowpass" | "highpass"
        Filter type.
    fmin, fmax : float or None
        Cutoff frequencies (Hz).
    order : int
        Butterworth filter order.

    Returns
    -------
    sos : np.ndarray, shape (n_sections, 6)
        Second-order sections matrix passable to ``iir_filter_time_rt``.
    """
    order = int(order)
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    fs_v, lo, hi = _validate_filter_band(fs=fs, kind=kind, fmin=fmin, fmax=fmax)
    nyq = 0.5 * fs_v
    if kind == "bandpass":
        wn = [lo / nyq, hi / nyq]
        btype = "bandpass"
    elif kind == "lowpass":
        wn = hi / nyq
        btype = "lowpass"
    else:
        wn = lo / nyq
        btype = "highpass"
    return signal.butter(order, wn, btype=btype, output="sos")


def iir_filter_time_rt(
    data: ArrayLike2D,
    sos: np.ndarray,
    *,
    zi: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal stateful IIR filter with state threading for streaming data.

    Unlike ``iir_filter_time`` (which uses ``sosfiltfilt`` / ``sosfilt`` on a
    self-contained block and starts fresh each call), this function carries
    filter state across successive calls, enabling seamless processing of
    continuous sensor streams with no transient glitches at chunk boundaries.

    Parameters
    ----------
    data : array-like, shape (nt, nc)
        Input block in ``nt_nc`` layout (time × channels).
    sos : np.ndarray, shape (n_sections, 6)
        SOS matrix from ``build_iir_sos()`` or
        ``scipy.signal.butter(..., output='sos')``.
        Pre-build once and reuse across chunks to avoid repeated coefficient
        computation.
    zi : np.ndarray or None, shape (n_sections, 2, nc)
        Initial filter state from the previous call (same convention as
        ``scipy.signal.sosfilt`` with ``axis=0``).
        ``None`` initialises from zero (beginning of stream).

    Returns
    -------
    y : np.ndarray, shape (nt, nc), float32
        Filtered output block.
    zi_out : np.ndarray, shape (n_sections, 2, nc), float64
        Updated filter state — pass this as ``zi`` to the next call.

    Examples
    --------
    >>> sos = build_iir_sos(fs=500, kind="bandpass", fmin=5, fmax=200)
    >>> zi = None
    >>> for chunk in sensor_stream:
    ...     filtered, zi = iir_filter_time_rt(chunk, sos, zi=zi)
    """
    # Realtime filtering keeps state per-channel, so the channel count is part
    # of the state shape contract.  Any channel-count change must reset state.
    # Float64 state below keeps the recurrence in double precision even when
    # the input and caller-provided SOS coefficients are float32.
    x = as_float2d(data, dtype=None)
    if x.dtype != np.float32:
        x = np.asarray(x, dtype=np.float64)
    nc = x.shape[1]
    n_sec = int(sos.shape[0])
    # scipy sosfilt with axis=0 on x=(nt, nc) expects zi shape (n_sec, 2, nc)
    if zi is None:
        zi = np.zeros((n_sec, 2, nc), dtype=np.float64)
    else:
        zi = np.asarray(zi, dtype=np.float64)
        if zi.shape != (n_sec, 2, nc):
            raise ValueError(
                f"zi shape mismatch: expected ({n_sec}, 2, {nc}), got {zi.shape}"
            )
    y, zi_out = signal.sosfilt(sos, x, axis=0, zi=zi)
    return y.astype(np.float32, copy=False), zi_out


def iir_bandpass_time_rt(
    data: ArrayLike2D,
    *,
    fs: float,
    fmin: float,
    fmax: float,
    order: int = 4,
    zi: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal bandpass IIR filter with state threading.

    Convenience wrapper that builds the SOS on every call.  For repeated
    calls on successive chunks, pre-build the SOS once with
    ``build_iir_sos()`` and call ``iir_filter_time_rt()`` directly.
    """
    sos = build_iir_sos(fs=fs, kind="bandpass", fmin=fmin, fmax=fmax, order=order)
    return iir_filter_time_rt(data, sos, zi=zi)


def iir_lowpass_time_rt(
    data: ArrayLike2D,
    *,
    fs: float,
    fmax: float,
    order: int = 4,
    zi: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal lowpass IIR filter with state threading."""
    sos = build_iir_sos(fs=fs, kind="lowpass", fmax=fmax, order=order)
    return iir_filter_time_rt(data, sos, zi=zi)


def iir_highpass_time_rt(
    data: ArrayLike2D,
    *,
    fs: float,
    fmin: float,
    order: int = 4,
    zi: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal highpass IIR filter with state threading."""
    sos = build_iir_sos(fs=fs, kind="highpass", fmin=fmin, order=order)
    return iir_filter_time_rt(data, sos, zi=zi)


def bandstop_time(
    data: ArrayLike2D,
    *,
    fs: float,
    fmin: float,
    fmax: float,
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    """IIR Butterworth bandstop filter along time axis.

    Useful for removing a specific frequency band, e.g. power-line harmonics.
    """
    x = as_float2d(data, dtype=np.float64)
    fs_v = require_positive("fs", fs)
    nyq = 0.5 * fs_v
    lo, hi = float(fmin), float(fmax)
    if lo <= 0 or hi <= lo:
        raise ValueError(f"Invalid bandstop range: fmin={lo}, fmax={hi}")
    if hi >= nyq:
        raise ValueError(f"fmax must be < Nyquist ({nyq:g}), got {hi}")

    wn = [lo / nyq, hi / nyq]
    sos = signal.butter(order, wn, btype="bandstop", output="sos")

    if zero_phase:
        try:
            y = signal.sosfiltfilt(sos, x, axis=0)
        except ValueError:
            warnings.warn("sosfiltfilt failed; fallback to causal sosfilt.", RuntimeWarning)
            y = signal.sosfilt(sos, x, axis=0)
    else:
        y = signal.sosfilt(sos, x, axis=0)
    return y.astype(np.float32, copy=False)


def notch_time(
    data: ArrayLike2D,
    *,
    fs: float,
    freq: float,
    bandwidth: float = 2.0,
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    """Narrow bandstop (notch) filter at a specific frequency.

    Convenience wrapper around `bandstop_time`.  A common use case is removing
    50 Hz or 60 Hz power-line interference: ``notch_time(data, fs=1000, freq=50)``.

    Parameters
    ----------
    freq : float
        Centre frequency to reject (Hz).
    bandwidth : float
        Full bandwidth to reject (Hz); default 2 Hz.
    """
    hw = 0.5 * float(bandwidth)
    return bandstop_time(
        data,
        fs=fs,
        fmin=max(1e-6, float(freq) - hw),
        fmax=float(freq) + hw,
        order=order,
        zero_phase=zero_phase,
    )


# ---------------------------------------------------------------------------
# FIR filtering
# ---------------------------------------------------------------------------

def fir_filter_time(
    data: ArrayLike2D,
    *,
    fs: float,
    kind: FilterKind,
    fmin: float | None = None,
    fmax: float | None = None,
    numtaps: int = 257,
    window: str = "hann",
    zero_phase: bool = True,
) -> np.ndarray:
    """FIR filtering along time axis."""
    x = as_float2d(data, dtype=np.float64)
    numtaps = int(numtaps)
    if numtaps < 3:
        raise ValueError(f"numtaps must be >= 3, got {numtaps}")
    if numtaps % 2 == 0:
        numtaps += 1

    fs, lo, hi = _validate_filter_band(fs=fs, kind=kind, fmin=fmin, fmax=fmax)

    if kind == "bandpass":
        b = signal.firwin(numtaps, [lo, hi], pass_zero=False, fs=fs, window=window)
    elif kind == "lowpass":
        b = signal.firwin(numtaps, hi, pass_zero=True, fs=fs, window=window)
    else:
        b = signal.firwin(numtaps, lo, pass_zero=False, fs=fs, window=window)

    if zero_phase:
        try:
            y = signal.filtfilt(b, [1.0], x, axis=0)
        except ValueError:
            warnings.warn(
                "filtfilt failed due to short sequence; fallback to causal lfilter.",
                RuntimeWarning,
            )
            y = signal.lfilter(b, [1.0], x, axis=0)
    else:
        y = signal.lfilter(b, [1.0], x, axis=0)
    return y.astype(np.float32, copy=False)


def fir_bandpass_time(
    data: ArrayLike2D, *, fs: float, fmin: float, fmax: float,
    numtaps: int = 257, window: str = "hann", zero_phase: bool = True,
) -> np.ndarray:
    return fir_filter_time(data, fs=fs, kind="bandpass", fmin=fmin, fmax=fmax,
                           numtaps=numtaps, window=window, zero_phase=zero_phase)


def fir_lowpass_time(
    data: ArrayLike2D, *, fs: float, fmax: float,
    numtaps: int = 257, window: str = "hann", zero_phase: bool = True,
) -> np.ndarray:
    return fir_filter_time(data, fs=fs, kind="lowpass", fmax=fmax,
                           numtaps=numtaps, window=window, zero_phase=zero_phase)


def fir_highpass_time(
    data: ArrayLike2D, *, fs: float, fmin: float,
    numtaps: int = 257, window: str = "hann", zero_phase: bool = True,
) -> np.ndarray:
    return fir_filter_time(data, fs=fs, kind="highpass", fmin=fmin,
                           numtaps=numtaps, window=window, zero_phase=zero_phase)


# ---------------------------------------------------------------------------
# FFT-domain filtering
# ---------------------------------------------------------------------------

def fft_bandpass_time(
    data: ArrayLike2D,
    *,
    fs: float,
    fmin: float,
    fmax: float,
    taper_hz: float = 0.0,
) -> np.ndarray:
    """Frequency-domain bandpass with optional cosine taper at band edges."""
    x = as_float2d(data, dtype=np.float64)
    fs, lo, hi = _validate_filter_band(fs=fs, kind="bandpass", fmin=fmin, fmax=fmax)
    taper = max(0.0, float(taper_hz))

    nt = x.shape[0]
    X = np.fft.rfft(x, axis=0)
    freqs = np.fft.rfftfreq(nt, d=1.0 / fs)

    mask = np.zeros_like(freqs, dtype=np.float64)
    if taper <= 0:
        mask[(freqs >= lo) & (freqs <= hi)] = 1.0
    else:
        lo1 = max(0.0, lo - taper)
        hi1 = min(0.5 * fs, hi + taper)
        mask[(freqs >= lo) & (freqs <= hi)] = 1.0

        left = (freqs >= lo1) & (freqs < lo)
        if np.any(left):
            a = (freqs[left] - lo1) / max(lo - lo1, 1e-12)
            mask[left] = 0.5 * (1.0 - np.cos(np.pi * a))

        right = (freqs > hi) & (freqs <= hi1)
        if np.any(right):
            a = (freqs[right] - hi) / max(hi1 - hi, 1e-12)
            mask[right] = 0.5 * (1.0 + np.cos(np.pi * a))

    X *= mask[:, None]
    y = np.fft.irfft(X, n=nt, axis=0)
    return y.astype(np.float32, copy=False)


def fft_lowpass_time(
    data: ArrayLike2D,
    *,
    fs: float,
    fmax: float,
    taper_hz: float = 0.0,
) -> np.ndarray:
    """Frequency-domain lowpass filter with optional cosine taper."""
    x = as_float2d(data, dtype=np.float64)
    fs_v = require_positive("fs", fs)
    hi = float(fmax)
    nyq = 0.5 * fs_v
    if hi <= 0 or hi > nyq:
        raise ValueError(f"fmax must be in (0, Nyquist={nyq:g}], got {hi}")
    taper = max(0.0, float(taper_hz))

    nt = x.shape[0]
    X = np.fft.rfft(x, axis=0)
    freqs = np.fft.rfftfreq(nt, d=1.0 / fs_v)

    mask = np.zeros_like(freqs, dtype=np.float64)
    mask[freqs <= hi] = 1.0

    if taper > 0:
        right = (freqs > hi) & (freqs <= hi + taper)
        if np.any(right):
            a = (freqs[right] - hi) / max(taper, 1e-12)
            mask[right] = 0.5 * (1.0 + np.cos(np.pi * a))

    X *= mask[:, None]
    y = np.fft.irfft(X, n=nt, axis=0)
    return y.astype(np.float32, copy=False)


def fft_highpass_time(
    data: ArrayLike2D,
    *,
    fs: float,
    fmin: float,
    taper_hz: float = 0.0,
) -> np.ndarray:
    """Frequency-domain highpass filter with optional cosine taper."""
    x = as_float2d(data, dtype=np.float64)
    fs_v = require_positive("fs", fs)
    lo = float(fmin)
    nyq = 0.5 * fs_v
    if lo <= 0 or lo >= nyq:
        raise ValueError(f"fmin must be in (0, Nyquist={nyq:g}), got {lo}")
    taper = max(0.0, float(taper_hz))

    nt = x.shape[0]
    X = np.fft.rfft(x, axis=0)
    freqs = np.fft.rfftfreq(nt, d=1.0 / fs_v)

    mask = np.zeros_like(freqs, dtype=np.float64)
    mask[freqs >= lo] = 1.0

    if taper > 0:
        lo1 = max(0.0, lo - taper)
        left = (freqs >= lo1) & (freqs < lo)
        if np.any(left):
            a = (freqs[left] - lo1) / max(lo - lo1, 1e-12)
            mask[left] = 0.5 * (1.0 - np.cos(np.pi * a))

    X *= mask[:, None]
    y = np.fft.irfft(X, n=nt, axis=0)
    return y.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Chunked variants — generated via make_chunked()
# ---------------------------------------------------------------------------

# IIR
iir_filter_time_chunked = make_chunked(iir_filter_time)
iir_bandpass_time_chunked = make_chunked(iir_bandpass_time)
iir_lowpass_time_chunked = make_chunked(iir_lowpass_time)
iir_highpass_time_chunked = make_chunked(iir_highpass_time)
bandstop_time_chunked = make_chunked(bandstop_time)
notch_time_chunked = make_chunked(notch_time)

# FIR
fir_filter_time_chunked = make_chunked(fir_filter_time)
fir_bandpass_time_chunked = make_chunked(fir_bandpass_time)
fir_lowpass_time_chunked = make_chunked(fir_lowpass_time)
fir_highpass_time_chunked = make_chunked(fir_highpass_time)

# FFT-domain
fft_bandpass_time_chunked = make_chunked(fft_bandpass_time)
fft_lowpass_time_chunked = make_chunked(fft_lowpass_time)
fft_highpass_time_chunked = make_chunked(fft_highpass_time)

# Smoothing / AGC
median_smooth_time_chunked = make_chunked(median_smooth_time)
agc_time_chunked = make_chunked(agc_time)


# ---------------------------------------------------------------------------
# Compatibility aliases
# ---------------------------------------------------------------------------

def butterworth_time_filter(
    data: ArrayLike2D,
    *,
    fs: float,
    fmin: float | None = None,
    fmax: float | None = None,
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    if fmin is None and fmax is None:
        raise ValueError("At least one of fmin/fmax must be provided")
    if fmin is not None and fmax is not None:
        kind: FilterKind = "bandpass"
    elif fmin is None:
        kind = "lowpass"
    else:
        kind = "highpass"
    return iir_filter_time(data, fs=fs, kind=kind, fmin=fmin, fmax=fmax, order=order, zero_phase=zero_phase)


def bandpass_time(data, *, fs, fmin, fmax, order=4, zero_phase=True):
    return iir_bandpass_time(data, fs=fs, fmin=fmin, fmax=fmax, order=order, zero_phase=zero_phase)


def lowpass_time(data, *, fs, fmax, order=4, zero_phase=True):
    return iir_lowpass_time(data, fs=fs, fmax=fmax, order=order, zero_phase=zero_phase)


def highpass_time(data, *, fs, fmin, order=4, zero_phase=True):
    return iir_highpass_time(data, fs=fs, fmin=fmin, order=order, zero_phase=zero_phase)


# ---------------------------------------------------------------------------
# Amplitude / spectral operators
# ---------------------------------------------------------------------------

def temporal_whiten(
    data: ArrayLike2D,
    *,
    fs: float,
    fmin: float | None = None,
    fmax: float | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Frequency-domain temporal whitening per channel."""
    x = as_float2d(data, dtype=np.float64)
    fs = require_positive("fs", fs)
    eps = require_positive("eps", eps)

    nt = x.shape[0]
    X = np.fft.rfft(x, axis=0)
    freqs = np.fft.rfftfreq(nt, d=1.0 / fs)

    amp = np.abs(X)
    Xw = X / np.maximum(amp, eps)

    if fmin is not None or fmax is not None:
        lo = 0.0 if fmin is None else float(fmin)
        hi = 0.5 * fs if fmax is None else float(fmax)
        if lo < 0 or hi <= lo:
            raise ValueError(f"Invalid whitening band: fmin={lo}, fmax={hi}")
        mask = (freqs >= lo) & (freqs <= hi)
        Xw[~mask, :] = 0.0

    y = np.fft.irfft(Xw, n=nt, axis=0)
    return y.astype(np.float32, copy=False)


def one_bit_normalize(data: ArrayLike2D) -> np.ndarray:
    x = as_float2d(data, dtype=np.float32)
    return np.sign(x).astype(np.float32, copy=False)


def clip_amplitude(data: ArrayLike2D, *, threshold: float) -> np.ndarray:
    x = as_float2d(data, dtype=np.float32)
    th = require_positive("threshold", threshold)
    return np.clip(x, -th, th).astype(np.float32, copy=False)


def analytic_envelope(data: ArrayLike2D) -> np.ndarray:
    """Magnitude of the analytic signal along time.

    Single-precision input is computed in single precision, anything else in
    double precision. Works on channel blocks in reusable buffers, so the extra
    memory is the float32 output plus the buffers.
    """
    x = as_float2d(data, dtype=None)
    nt, nc = x.shape
    if nt == 0:
        raise ValueError("N must be positive.")
    single = x.dtype in (np.float16, np.float32)
    complex_type, real_type = (np.complex64, np.float32) if single else (np.complex128, np.float64)
    width = np.dtype(complex_type).itemsize
    # FFT libraries vectorize across channels, so keep several in each block.
    step = min(max(nc, 1), 128, max(8, _ENVELOPE_BLOCK_BYTES // (nt * width)))
    work = np.empty((step, nt), dtype=complex_type)
    magnitude = np.empty((step, nt), dtype=real_type)
    n_half = nt // 2 + 1
    doubled_stop = (nt + 1) // 2
    out = np.empty((nt, nc), dtype=np.float32, order="F")
    for c0 in range(0, nc, step):
        m = min(step, nc - c0)
        b, g = work[:m], magnitude[:m]
        b[:, :n_half] = sp_fft.rfft(x[:, c0:c0 + m].T.astype(real_type, copy=False), axis=-1)
        b[:, n_half:] = 0.0
        b[:, 1:doubled_stop] *= 2.0
        np.abs(sp_fft.ifft(b, axis=-1, overwrite_x=True), out=g)
        out[:, c0:c0 + m] = g.T
    return out
