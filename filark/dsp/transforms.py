# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""Class-based DSP transforms.

The function modules in ``filark.dsp`` are the numerical ground truth; this
module wraps them as parameter-carrying objects so they can be composed,
serialized in configs, and embedded into streaming or chunked pipelines.

Two transform families coexist here:

- stateless transforms, which behave like pure functions on one block
- stateful realtime transforms, which must preserve internal state between
  calls to represent causal streaming behavior

`Compose` deliberately understands the difference because the viewer and the
streaming layer need to snapshot/reset state when seeking, left-scrolling, or
rebuilding a buffer.

Usage
-----
>>> import filark.dsp.transforms as T
>>> pipe = T.Compose([
...     T.DeMeanTime(),
...     T.IIRBandpassTime(fs=500, fmin=5, fmax=200),
...     T.AGCTime(half_window=50),
... ])
>>> out = pipe(data)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from filark._exceptions import print_exception
from . import temporal, spatial, fk, decomposition, nonlinear

__all__ = [
    # composition
    "Compose",
    # temporal – basic
    "DeMeanTime",
    "DetrendTime",
    "ZScoreTime",
    "RunningMeanTime",
    "GaussianSmoothTime",
    "MedianSmoothTime",
    "AGCTime",
    # temporal – IIR (offline / zero-phase)
    "IIRFilterTime",
    "IIRBandpassTime",
    "IIRLowpassTime",
    "IIRHighpassTime",
    "BandstopTime",
    "NotchTime",
    # temporal – realtime IIR (causal, stateful)
    "RealtimeIIRFilter",
    "RealtimeIIRBandpassFilter",
    "RealtimeIIRLowpassFilter",
    "RealtimeIIRHighpassFilter",
    # temporal – realtime FIR (causal, stateful)
    "RealtimeFIRFilter",
    "RealtimeFIRBandpassFilter",
    "RealtimeFIRLowpassFilter",
    "RealtimeFIRHighpassFilter",
    # temporal – realtime sliding-window (causal, stateful)
    "RealtimeAGC",
    "RealtimeRunningMean",
    "RealtimeGaussianSmooth",
    "RealtimeMedianSmooth",
    # temporal – FIR
    "FIRFilterTime",
    "FIRBandpassTime",
    "FIRLowpassTime",
    "FIRHighpassTime",
    # temporal – FFT domain
    "FFTBandpassTime",
    "FFTLowpassTime",
    "FFTHighpassTime",
    # temporal – amplitude
    "TemporalWhiten",
    "OneBitNormalize",
    "ClipAmplitude",
    "AnalyticEnvelope",
    # spatial – realtime (stateful)
    "RealtimeEqualizeChannelGain",
    # spectral – realtime (stateful)
    "RealtimeSTFT",
    # spatial
    "DeMeanSpace",
    "DetrendSpace",
    "ZScoreSpace",
    "RunningMeanSpace",
    "GaussianSmoothSpace",
    "MedianSmoothSpace",
    "EqualizeChannelGain",
    "SpatialGradient",
    "SpatialLaplacian",
    "ResampleChannels",
    # f-k
    "FKFanFilter",
    "DirectionalWedgeDenoise",
    # decomposition
    "TruncatedSVDReconstruct",
    "LocalSVDDenoise",
    # nonlinear
    "AnisotropicDiffusion2D",
]


class Compose:
    """Apply a list of transforms in sequence.

    Parameters
    ----------
    transforms : list
        Ordered list of callables (transforms or arbitrary functions).
    """

    def __init__(self, transforms: list):
        self.transforms = list(transforms)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            data = t(data)
        return data

    def is_stateful(self) -> bool:
        """Return True when any child transform carries cross-call state."""
        for t in self.transforms:
            fn = getattr(t, "is_stateful", None)
            if callable(fn):
                try:
                    if bool(fn()):
                        return True
                    continue
                except Exception:
                    print_exception(f"Compose child {t!r} failed in is_stateful().", prefix="DSP")
            if callable(getattr(t, "reset", None)):
                return True
        return False

    def reset(self) -> None:
        for t in self.transforms:
            reset = getattr(t, "reset", None)
            if callable(reset):
                reset()

    def get_state(self):
        """Capture per-transform state for replay/checkpoint use."""
        if not self.is_stateful():
            return None
        states: list[Any] = []
        for t in self.transforms:
            if callable(getattr(t, "reset", None)):
                get_state = getattr(t, "get_state", None)
                if not callable(get_state):
                    raise ValueError(f"{t!r} is stateful but does not implement get_state()")
                states.append(get_state())
            else:
                states.append(None)
        return tuple(states)

    def set_state(self, state) -> None:
        """Restore a previously captured transform state tuple."""
        if state is None:
            self.reset()
            return

        states = list(state)
        if len(states) != len(self.transforms):
            raise ValueError("Compose state length mismatch")

        for t, child_state in zip(self.transforms, states):
            reset = getattr(t, "reset", None)
            if not callable(reset):
                continue
            if child_state is None:
                reset()
                continue
            set_state = getattr(t, "set_state", None)
            if not callable(set_state):
                raise ValueError(f"{t!r} is stateful but does not implement set_state()")
            set_state(child_state)

    def warmup_samples(self, visible_w: int | None = None) -> int:
        """Estimate left-history needed before a visible window is trustworthy."""
        hint = int(max(0, visible_w or 0))
        widths: list[int] = []
        for t in self.transforms:
            fn = getattr(t, "warmup_samples", None)
            if callable(fn):
                try:
                    widths.append(int(max(0, fn(hint))))
                    continue
                except TypeError:
                    widths.append(int(max(0, fn())))
                    continue
                except Exception:
                    print_exception(f"Compose child {t!r} failed in warmup_samples().", prefix="DSP")
            if callable(getattr(t, "reset", None)):
                widths.append(hint)
        return max(widths, default=0)

    def __repr__(self) -> str:
        parts = ",\n    ".join(repr(t) for t in self.transforms)
        return f"Compose([\n    {parts}\n])"


# ---------------------------------------------------------------------------
# Temporal – basic
# ---------------------------------------------------------------------------

class DeMeanTime:
    """Remove per-channel temporal mean."""

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return temporal.demean_time(data)

    def __repr__(self) -> str:
        return "DeMeanTime()"


class DetrendTime:
    """Per-channel temporal detrend.

    Parameters
    ----------
    kind : "linear" | "constant"
    """

    def __init__(self, *, kind: str = "linear"):
        self.kind = kind

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return temporal.detrend_time(data, kind=self.kind)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        return f"DetrendTime(kind={self.kind!r})"


class ZScoreTime:
    """Per-channel z-score normalization along time axis.

    Parameters
    ----------
    robust : bool
        Use median / MAD instead of mean / std.
    eps : float
        Regularisation constant.
    """

    def __init__(self, *, robust: bool = False, eps: float = 1e-8):
        self.robust = robust
        self.eps = eps

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return temporal.zscore_time(data, robust=self.robust, eps=self.eps)

    def __repr__(self) -> str:
        return f"ZScoreTime(robust={self.robust}, eps={self.eps})"


class RunningMeanTime:
    """Temporal moving average per channel.

    Parameters
    ----------
    half_window : int
    mode : str
        scipy padding mode.
    """

    def __init__(self, *, half_window: int = 8, mode: str = "reflect"):
        self.half_window = half_window
        self.mode = mode

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return temporal.running_mean_time(data, half_window=self.half_window, mode=self.mode)

    def __repr__(self) -> str:
        return f"RunningMeanTime(half_window={self.half_window})"


class GaussianSmoothTime:
    """Temporal Gaussian smoothing per channel."""

    def __init__(self, *, sigma: float = 2.0, mode: str = "reflect"):
        self.sigma = sigma
        self.mode = mode

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return temporal.gaussian_smooth_time(data, sigma=self.sigma, mode=self.mode)

    def __repr__(self) -> str:
        return f"GaussianSmoothTime(sigma={self.sigma})"


class MedianSmoothTime:
    """Temporal median smoothing per channel."""

    def __init__(self, *, kernel_size: int = 5):
        self.kernel_size = kernel_size

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return temporal.median_smooth_time(data, kernel_size=self.kernel_size)

    def chunked(self, data: np.ndarray, *, chunk_channels=None,
                target_chunk_mb: float = 128.0, layout: str = "nt_nc") -> np.ndarray:
        return temporal.median_smooth_time_chunked(
            data, kernel_size=self.kernel_size,
            chunk_channels=chunk_channels, target_chunk_mb=target_chunk_mb, layout=layout,
        )

    def __repr__(self) -> str:
        return f"MedianSmoothTime(kernel_size={self.kernel_size})"


class AGCTime:
    """Automatic Gain Control: time-varying per-channel amplitude normalization.

    Parameters
    ----------
    half_window : int
    method : "rms" | "max" | "mean"
    eps : float
    """

    def __init__(self, *, half_window: int = 50, method: str = "rms", eps: float = 1e-8):
        self.half_window = half_window
        self.method = method
        self.eps = eps

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return temporal.agc_time(data, half_window=self.half_window, method=self.method, eps=self.eps)  # type: ignore[arg-type]

    def chunked(self, data: np.ndarray, *, chunk_channels=None,
                target_chunk_mb: float = 128.0, layout: str = "nt_nc") -> np.ndarray:
        return temporal.agc_time_chunked(
            data, half_window=self.half_window, method=self.method, eps=self.eps,
            chunk_channels=chunk_channels, target_chunk_mb=target_chunk_mb, layout=layout,
        )

    def __repr__(self) -> str:
        return f"AGCTime(half_window={self.half_window}, method={self.method!r})"


# ---------------------------------------------------------------------------
# Temporal – IIR filtering
# ---------------------------------------------------------------------------

class IIRFilterTime:
    """Butterworth IIR filter along time axis."""

    def __init__(self, *, fs: float, kind: str, fmin: float | None = None,
                 fmax: float | None = None, order: int = 4, zero_phase: bool = True):
        self.fs = fs
        self.kind = kind
        self.fmin = fmin
        self.fmax = fmax
        self.order = order
        self.zero_phase = zero_phase

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return temporal.iir_filter_time(
            data, fs=self.fs, kind=self.kind,  # type: ignore[arg-type]
            fmin=self.fmin, fmax=self.fmax, order=self.order, zero_phase=self.zero_phase,
        )

    def chunked(self, data: np.ndarray, *, chunk_channels=None,
                target_chunk_mb: float = 128.0, layout: str = "nt_nc") -> np.ndarray:
        return temporal.iir_filter_time_chunked(
            data, fs=self.fs, kind=self.kind,
            fmin=self.fmin, fmax=self.fmax, order=self.order, zero_phase=self.zero_phase,
            chunk_channels=chunk_channels, target_chunk_mb=target_chunk_mb, layout=layout,
        )

    def __repr__(self) -> str:
        return (f"IIRFilterTime(fs={self.fs}, kind={self.kind!r}, "
                f"fmin={self.fmin}, fmax={self.fmax}, order={self.order})")


class IIRBandpassTime:
    """IIR Butterworth bandpass filter."""

    def __init__(self, *, fs: float, fmin: float, fmax: float,
                 order: int = 4, zero_phase: bool = True):
        self.fs = fs
        self.fmin = fmin
        self.fmax = fmax
        self.order = order
        self.zero_phase = zero_phase

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return temporal.iir_bandpass_time(
            data, fs=self.fs, fmin=self.fmin, fmax=self.fmax,
            order=self.order, zero_phase=self.zero_phase,
        )

    def chunked(self, data: np.ndarray, *, chunk_channels=None,
                target_chunk_mb: float = 128.0, layout: str = "nt_nc") -> np.ndarray:
        return temporal.iir_bandpass_time_chunked(
            data, fs=self.fs, fmin=self.fmin, fmax=self.fmax,
            order=self.order, zero_phase=self.zero_phase,
            chunk_channels=chunk_channels, target_chunk_mb=target_chunk_mb, layout=layout,
        )

    def __repr__(self) -> str:
        return f"IIRBandpassTime(fs={self.fs}, fmin={self.fmin}, fmax={self.fmax})"


class IIRLowpassTime:
    """IIR Butterworth lowpass filter."""

    def __init__(self, *, fs: float, fmax: float, order: int = 4, zero_phase: bool = True):
        self.fs = fs
        self.fmax = fmax
        self.order = order
        self.zero_phase = zero_phase

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return temporal.iir_lowpass_time(
            data, fs=self.fs, fmax=self.fmax, order=self.order, zero_phase=self.zero_phase,
        )

    def chunked(self, data: np.ndarray, *, chunk_channels=None,
                target_chunk_mb: float = 128.0, layout: str = "nt_nc") -> np.ndarray:
        return temporal.iir_lowpass_time_chunked(
            data, fs=self.fs, fmax=self.fmax, order=self.order, zero_phase=self.zero_phase,
            chunk_channels=chunk_channels, target_chunk_mb=target_chunk_mb, layout=layout,
        )

    def __repr__(self) -> str:
        return f"IIRLowpassTime(fs={self.fs}, fmax={self.fmax})"


class IIRHighpassTime:
    """IIR Butterworth highpass filter."""

    def __init__(self, *, fs: float, fmin: float, order: int = 4, zero_phase: bool = True):
        self.fs = fs
        self.fmin = fmin
        self.order = order
        self.zero_phase = zero_phase

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return temporal.iir_highpass_time(
            data, fs=self.fs, fmin=self.fmin, order=self.order, zero_phase=self.zero_phase,
        )

    def chunked(self, data: np.ndarray, *, chunk_channels=None,
                target_chunk_mb: float = 128.0, layout: str = "nt_nc") -> np.ndarray:
        return temporal.iir_highpass_time_chunked(
            data, fs=self.fs, fmin=self.fmin, order=self.order, zero_phase=self.zero_phase,
            chunk_channels=chunk_channels, target_chunk_mb=target_chunk_mb, layout=layout,
        )

    def __repr__(self) -> str:
        return f"IIRHighpassTime(fs={self.fs}, fmin={self.fmin})"


class BandstopTime:
    """IIR Butterworth bandstop filter."""

    def __init__(self, *, fs: float, fmin: float, fmax: float,
                 order: int = 4, zero_phase: bool = True):
        self.fs = fs
        self.fmin = fmin
        self.fmax = fmax
        self.order = order
        self.zero_phase = zero_phase

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return temporal.bandstop_time(
            data, fs=self.fs, fmin=self.fmin, fmax=self.fmax,
            order=self.order, zero_phase=self.zero_phase,
        )

    def chunked(self, data: np.ndarray, *, chunk_channels=None,
                target_chunk_mb: float = 128.0, layout: str = "nt_nc") -> np.ndarray:
        return temporal.bandstop_time_chunked(
            data, fs=self.fs, fmin=self.fmin, fmax=self.fmax,
            order=self.order, zero_phase=self.zero_phase,
            chunk_channels=chunk_channels, target_chunk_mb=target_chunk_mb, layout=layout,
        )

    def __repr__(self) -> str:
        return f"BandstopTime(fs={self.fs}, fmin={self.fmin}, fmax={self.fmax})"


class NotchTime:
    """Narrow bandstop (notch) filter at a specific frequency."""

    def __init__(self, *, fs: float, freq: float, bandwidth: float = 2.0,
                 order: int = 4, zero_phase: bool = True):
        self.fs = fs
        self.freq = freq
        self.bandwidth = bandwidth
        self.order = order
        self.zero_phase = zero_phase

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return temporal.notch_time(
            data, fs=self.fs, freq=self.freq, bandwidth=self.bandwidth,
            order=self.order, zero_phase=self.zero_phase,
        )

    def chunked(self, data: np.ndarray, *, chunk_channels=None,
                target_chunk_mb: float = 128.0, layout: str = "nt_nc") -> np.ndarray:
        return temporal.notch_time_chunked(
            data, fs=self.fs, freq=self.freq, bandwidth=self.bandwidth,
            order=self.order, zero_phase=self.zero_phase,
            chunk_channels=chunk_channels, target_chunk_mb=target_chunk_mb, layout=layout,
        )

    def __repr__(self) -> str:
        return f"NotchTime(fs={self.fs}, freq={self.freq}, bandwidth={self.bandwidth})"


# ---------------------------------------------------------------------------
# Temporal – FIR filtering
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Temporal – realtime (stateful, causal) IIR
# ---------------------------------------------------------------------------

class RealtimeIIRFilter:
    """Stateful causal Butterworth IIR filter for streaming / sensor data.

    Unlike the offline :class:`IIRFilterTime` (which uses ``sosfiltfilt`` on
    a self-contained block), this class threads filter state across successive
    :meth:`push` calls so chunk boundaries are seamless.

    Usage
    -----
    >>> filt = RealtimeIIRFilter(fs=500, kind="bandpass", fmin=5, fmax=200)
    >>> for chunk in sensor_stream:           # chunk: [nt, nc]
    ...     filtered = filt.push(chunk)

    The SOS matrix is built once at construction.  Call :meth:`reset` to clear
    state (e.g. after a seek or parameter change).

    Parameters
    ----------
    fs : float
        Sampling rate (Hz).
    kind : "bandpass" | "lowpass" | "highpass"
    fmin, fmax : float or None
        Cutoff frequencies (Hz).
    order : int
        Butterworth order.
    device : str
        ``"cpu"`` or a torch device string (``"cuda"``, ``"mps"``, ``"auto"``).
        On non-CPU devices, uses the GPU DF-II-T backend.
    """

    def __init__(
        self,
        *,
        fs: float,
        kind: str,
        fmin: float | None = None,
        fmax: float | None = None,
        order: int = 4,
        device: str = "cpu",
    ):
        from .temporal import build_iir_sos
        self.sos: np.ndarray = build_iir_sos(
            fs=fs, kind=kind, fmin=fmin, fmax=fmax, order=order  # type: ignore[arg-type]
        )
        self.fs = fs
        self.kind = kind
        self.fmin = fmin
        self.fmax = fmax
        self.order = order
        self.device = device
        self._zi: np.ndarray | None = None

    def is_stateful(self) -> bool:
        return True

    def push(self, chunk: np.ndarray) -> np.ndarray:
        """Filter ``chunk`` and advance state.  Input shape: ``(nt, nc)``."""
        if self._zi is not None:
            nc = int(np.asarray(chunk).shape[1])
            expected = (int(self.sos.shape[0]), 2, nc)
            if self._zi.shape != expected:
                self._zi = None
        if self.device == "cpu":
            from .temporal import iir_filter_time_rt
            y, self._zi = iir_filter_time_rt(chunk, self.sos, zi=self._zi)
        else:
            from .gpu.temporal import iir_filter_time_rt_gpu
            y, self._zi = iir_filter_time_rt_gpu(
                chunk, self.sos, zi_np=self._zi, device=self.device
            )
        return y

    def reset(self) -> None:
        """Clear filter state (call after seek or parameter change)."""
        self._zi = None

    def get_state(self) -> np.ndarray | None:
        """Return a copy of SOS state for scheduler checkpoints/replay."""
        if self._zi is None:
            return None
        return np.array(self._zi, copy=True)

    def set_state(self, state: np.ndarray | None) -> None:
        if state is None:
            self._zi = None
            return
        self._zi = np.asarray(state, dtype=np.float64).copy()

    def apply_once(self, data: np.ndarray) -> np.ndarray:
        """Stateless single-block causal filter."""
        self.reset()
        return self.push(data)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Stateful callable alias for streaming-style use."""
        return self.push(data)

    def __repr__(self) -> str:
        return (
            f"RealtimeIIRFilter(fs={self.fs}, kind={self.kind!r}, "
            f"fmin={self.fmin}, fmax={self.fmax}, order={self.order})"
        )


class RealtimeIIRBandpassFilter(RealtimeIIRFilter):
    """Stateful causal bandpass IIR filter."""

    def __init__(self, *, fs: float, fmin: float, fmax: float,
                 order: int = 4, device: str = "cpu"):
        super().__init__(fs=fs, kind="bandpass", fmin=fmin, fmax=fmax,
                         order=order, device=device)

    def __repr__(self) -> str:
        return (f"RealtimeIIRBandpassFilter(fs={self.fs}, "
                f"fmin={self.fmin}, fmax={self.fmax}, order={self.order})")


class RealtimeIIRLowpassFilter(RealtimeIIRFilter):
    """Stateful causal lowpass IIR filter."""

    def __init__(self, *, fs: float, fmax: float, order: int = 4, device: str = "cpu"):
        super().__init__(fs=fs, kind="lowpass", fmax=fmax, order=order, device=device)

    def __repr__(self) -> str:
        return (f"RealtimeIIRLowpassFilter(fs={self.fs}, "
                f"fmax={self.fmax}, order={self.order})")


class RealtimeIIRHighpassFilter(RealtimeIIRFilter):
    """Stateful causal highpass IIR filter."""

    def __init__(self, *, fs: float, fmin: float, order: int = 4, device: str = "cpu"):
        super().__init__(fs=fs, kind="highpass", fmin=fmin, order=order, device=device)

    def __repr__(self) -> str:
        return (f"RealtimeIIRHighpassFilter(fs={self.fs}, "
                f"fmin={self.fmin}, order={self.order})")


# ---------------------------------------------------------------------------
# Temporal – realtime (stateful, causal) FIR
# ---------------------------------------------------------------------------

class RealtimeFIRFilter:
    """Stateful causal FIR filter for streaming / sensor data.

    Uses ``scipy.signal.lfilter`` with explicit state threading so chunk
    boundaries are seamless.  Unlike the offline :class:`FIRFilterTime` (which
    calls ``filtfilt`` on a self-contained block), this class is strictly
    causal: each output sample depends only on the current and past inputs.

    The filter coefficients are built once at construction.  Call
    :meth:`reset` to clear state after a seek or parameter change.

    Parameters
    ----------
    fs : float
        Sampling rate (Hz).
    kind : "bandpass" | "lowpass" | "highpass"
    fmin, fmax : float or None
        Cutoff frequencies (Hz).
    numtaps : int
        Number of FIR taps (odd; incremented by 1 if even).
    window : str
        Window function for ``firwin`` (default ``"hann"``).
    """

    def __init__(
        self,
        *,
        fs: float,
        kind: str,
        fmin: float | None = None,
        fmax: float | None = None,
        numtaps: int = 257,
        window: str = "hann",
    ):
        from scipy import signal as _signal
        from .temporal import _validate_filter_band

        numtaps = int(numtaps)
        if numtaps % 2 == 0:
            numtaps += 1

        fs_v, lo, hi = _validate_filter_band(fs=fs, kind=kind, fmin=fmin, fmax=fmax)  # type: ignore[arg-type]
        if kind == "bandpass":
            b = _signal.firwin(numtaps, [lo, hi], pass_zero=False, fs=fs_v, window=window)
        elif kind == "lowpass":
            b = _signal.firwin(numtaps, hi, pass_zero=True, fs=fs_v, window=window)
        else:
            b = _signal.firwin(numtaps, lo, pass_zero=False, fs=fs_v, window=window)

        self.b: np.ndarray = b.astype(np.float64)
        self.fs = fs
        self.kind = kind
        self.fmin = fmin
        self.fmax = fmax
        self.numtaps = numtaps
        self.window = window
        self._zi: np.ndarray | None = None  # shape: (numtaps-1, nc)

    def is_stateful(self) -> bool:
        return True

    def push(self, chunk: np.ndarray) -> np.ndarray:
        """Filter ``chunk`` and advance state.  Input shape: ``(nt, nc)``."""
        from scipy.signal import lfilter
        from .base import as_float2d

        x = as_float2d(chunk, dtype=np.float64)
        nc = x.shape[1]
        zi_len = len(self.b) - 1

        if self._zi is None or self._zi.shape != (zi_len, nc):
            self._zi = np.zeros((zi_len, nc), dtype=np.float64)

        y, self._zi = lfilter(self.b, [1.0], x, axis=0, zi=self._zi)
        return y.astype(np.float32)

    def reset(self) -> None:
        """Clear filter state."""
        self._zi = None

    def get_state(self) -> np.ndarray | None:
        if self._zi is None:
            return None
        return self._zi.copy()

    def set_state(self, state: np.ndarray | None) -> None:
        if state is None:
            self._zi = None
        else:
            self._zi = np.asarray(state, dtype=np.float64).copy()

    def apply_once(self, data: np.ndarray) -> np.ndarray:
        """Stateless single-block causal filter (resets state first)."""
        self.reset()
        return self.push(data)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return self.push(data)

    def __repr__(self) -> str:
        return (f"RealtimeFIRFilter(fs={self.fs}, kind={self.kind!r}, "
                f"fmin={self.fmin}, fmax={self.fmax}, numtaps={self.numtaps})")


class RealtimeFIRBandpassFilter(RealtimeFIRFilter):
    """Stateful causal FIR bandpass filter."""

    def __init__(self, *, fs: float, fmin: float, fmax: float,
                 numtaps: int = 257, window: str = "hann"):
        super().__init__(fs=fs, kind="bandpass", fmin=fmin, fmax=fmax,
                         numtaps=numtaps, window=window)

    def __repr__(self) -> str:
        return (f"RealtimeFIRBandpassFilter(fs={self.fs}, "
                f"fmin={self.fmin}, fmax={self.fmax}, numtaps={self.numtaps})")


class RealtimeFIRLowpassFilter(RealtimeFIRFilter):
    """Stateful causal FIR lowpass filter."""

    def __init__(self, *, fs: float, fmax: float,
                 numtaps: int = 257, window: str = "hann"):
        super().__init__(fs=fs, kind="lowpass", fmax=fmax,
                         numtaps=numtaps, window=window)

    def __repr__(self) -> str:
        return (f"RealtimeFIRLowpassFilter(fs={self.fs}, "
                f"fmax={self.fmax}, numtaps={self.numtaps})")


class RealtimeFIRHighpassFilter(RealtimeFIRFilter):
    """Stateful causal FIR highpass filter."""

    def __init__(self, *, fs: float, fmin: float,
                 numtaps: int = 257, window: str = "hann"):
        super().__init__(fs=fs, kind="highpass", fmin=fmin,
                         numtaps=numtaps, window=window)

    def __repr__(self) -> str:
        return (f"RealtimeFIRHighpassFilter(fs={self.fs}, "
                f"fmin={self.fmin}, numtaps={self.numtaps})")


# ---------------------------------------------------------------------------
# Temporal – realtime sliding-window operators (stateful, causal)
#
# Each class maintains a left-boundary history buffer of shape (H, nc) so
# that chunk boundaries are seamless.  The right boundary uses "nearest"
# extrapolation (last sample repeated), which is correct for all but the
# last H samples of each chunk; those samples are corrected automatically
# by the next chunk's left boundary.
# ---------------------------------------------------------------------------

class _RealtimeSlidingBase:
    """Internal mixin: history-buffer management for sliding-window operators."""

    _history_size: int = 0         # set by subclass
    _history: np.ndarray | None    # (H, nc)

    def _init_history(self) -> None:
        self._history = None

    def _get_padded(self, x: np.ndarray) -> tuple[np.ndarray, int]:
        """Return (padded_array, H) ready for filtering.

        Prepends history (or zeros on first call) and appends H "nearest"
        samples so the centred filter output aligns with the input chunk.
        """
        H = self._history_size
        nc = x.shape[1]
        if self._history is None or self._history.shape[1] != nc:
            left = np.zeros((H, nc), dtype=x.dtype)
        else:
            left = self._history.astype(x.dtype)
        right = np.repeat(x[-1:], H, axis=0)
        return np.concatenate([left, x, right], axis=0), H

    def _update_history(self, x: np.ndarray) -> None:
        H = self._history_size
        nc = x.shape[1]
        if self._history is None or self._history.shape[1] != nc:
            buf = x
        else:
            buf = np.concatenate([self._history, x], axis=0)
        self._history = buf[-H:].copy()

    def is_stateful(self) -> bool:
        return True

    def reset(self) -> None:
        self._history = None

    def get_state(self) -> np.ndarray | None:
        if self._history is None:
            return None
        return self._history.copy()

    def set_state(self, state: np.ndarray | None) -> None:
        if state is None:
            self._history = None
        else:
            self._history = np.asarray(state, dtype=np.float32).copy()

    def apply_once(self, data: np.ndarray) -> np.ndarray:
        self.reset()
        return self.push(data)  # type: ignore[attr-defined]

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return self.push(data)  # type: ignore[attr-defined]


class RealtimeAGC(_RealtimeSlidingBase):
    """Stateful causal Automatic Gain Control for streaming data.

    Each chunk is normalised by a local amplitude estimate computed over a
    sliding window.  The left boundary uses buffered history from the
    previous chunk so the gain function is continuous across chunk edges.

    Parameters
    ----------
    half_window : int
        Half-window size in samples; full window = 2*half_window + 1.
    method : "rms" | "max" | "mean"
        Local amplitude estimator.
    eps : float
        Regularisation floor to avoid division by zero.
    """

    def __init__(self, *, half_window: int = 50,
                 method: str = "rms", eps: float = 1e-8):
        from .temporal import require_positive as _rp  # type: ignore[attr-defined]
        half_window = int(half_window)
        if half_window < 1:
            raise ValueError(f"half_window must be >= 1, got {half_window}")
        self.half_window = half_window
        self.method = method
        self.eps = float(eps)
        self._history_size = half_window
        self._init_history()

    def push(self, chunk: np.ndarray) -> np.ndarray:
        from scipy.ndimage import uniform_filter1d, maximum_filter1d
        from .base import as_float2d

        x = as_float2d(chunk, dtype=np.float64)
        nt = x.shape[0]
        H = self._history_size
        padded, _ = self._get_padded(x)
        size = 2 * H + 1

        if self.method == "rms":
            gain = np.sqrt(uniform_filter1d(padded ** 2, size=size, axis=0, mode="nearest"))
        elif self.method == "max":
            gain = maximum_filter1d(np.abs(padded), size=size, axis=0, mode="nearest")
        elif self.method == "mean":
            gain = uniform_filter1d(np.abs(padded), size=size, axis=0, mode="nearest")
        else:
            raise ValueError(f"Unknown method: {self.method!r}")

        gain_chunk = gain[H: H + nt]
        y = x / np.maximum(gain_chunk, self.eps)
        self._update_history(x.astype(np.float32))
        return y.astype(np.float32)

    def __repr__(self) -> str:
        return (f"RealtimeAGC(half_window={self.half_window}, "
                f"method={self.method!r}, eps={self.eps})")


class RealtimeRunningMean(_RealtimeSlidingBase):
    """Stateful causal temporal running mean for streaming data.

    Parameters
    ----------
    half_window : int
        Half-window in samples; full window = 2*half_window + 1.
    """

    def __init__(self, *, half_window: int = 8):
        half_window = int(half_window)
        if half_window < 1:
            raise ValueError(f"half_window must be >= 1, got {half_window}")
        self.half_window = half_window
        self._history_size = half_window
        self._init_history()

    def push(self, chunk: np.ndarray) -> np.ndarray:
        from scipy.ndimage import uniform_filter1d
        from .base import as_float2d

        x = as_float2d(chunk, dtype=np.float32)
        nt = x.shape[0]
        H = self._history_size
        padded, _ = self._get_padded(x)
        y = uniform_filter1d(padded, size=2 * H + 1, axis=0, mode="nearest")
        self._update_history(x)
        return y[H: H + nt].astype(np.float32)

    def __repr__(self) -> str:
        return f"RealtimeRunningMean(half_window={self.half_window})"


class RealtimeGaussianSmooth(_RealtimeSlidingBase):
    """Stateful causal temporal Gaussian smoothing for streaming data.

    The history buffer size is ``ceil(4 * sigma) + 1`` samples, covering
    roughly ±4σ of the Gaussian kernel.

    Parameters
    ----------
    sigma : float
        Standard deviation in samples.
    """

    def __init__(self, *, sigma: float = 2.0):
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self.sigma = float(sigma)
        self._history_size = int(np.ceil(4.0 * self.sigma)) + 1
        self._init_history()

    def push(self, chunk: np.ndarray) -> np.ndarray:
        from scipy.ndimage import gaussian_filter1d
        from .base import as_float2d

        x = as_float2d(chunk, dtype=np.float32)
        nt = x.shape[0]
        H = self._history_size
        padded, _ = self._get_padded(x)
        y = gaussian_filter1d(padded, sigma=self.sigma, axis=0, mode="nearest")
        self._update_history(x)
        return y[H: H + nt].astype(np.float32)

    def __repr__(self) -> str:
        return f"RealtimeGaussianSmooth(sigma={self.sigma})"


class RealtimeMedianSmooth(_RealtimeSlidingBase):
    """Stateful causal temporal median smoothing for streaming data.

    Robust to impulsive noise.  The history buffer size equals
    ``kernel_size // 2`` samples.

    Parameters
    ----------
    kernel_size : int
        Median kernel length (odd; incremented by 1 if even).
    """

    def __init__(self, *, kernel_size: int = 5):
        k = int(kernel_size)
        if k < 1:
            raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
        if k % 2 == 0:
            k += 1
        self.kernel_size = k
        self._history_size = k // 2
        self._init_history()

    def push(self, chunk: np.ndarray) -> np.ndarray:
        from scipy.ndimage import median_filter
        from .base import as_float2d

        x = as_float2d(chunk, dtype=np.float32)
        nt = x.shape[0]
        H = self._history_size
        padded, _ = self._get_padded(x)
        y = median_filter(padded, size=(self.kernel_size, 1), mode="nearest")
        self._update_history(x)
        return y[H: H + nt].astype(np.float32)

    def __repr__(self) -> str:
        return f"RealtimeMedianSmooth(kernel_size={self.kernel_size})"


# ---------------------------------------------------------------------------
# Spatial – realtime (stateful)
# ---------------------------------------------------------------------------

class RealtimeEqualizeChannelGain:
    """Streaming per-channel gain equalizer with exponential moving average.

    Each chunk, a per-channel amplitude estimate is computed (RMS, STD, or
    MAD).  The running estimate is updated via EMA so the scale factors adapt
    gradually rather than jumping on every chunk:

        _gain = alpha * chunk_gain + (1 - alpha) * _gain

    The channel scale applied to the data is::

        scale = median(_gain) / max(_gain, eps)

    optionally clamped to ``[clip_min, clip_max]``.

    Parameters
    ----------
    method : {"rms", "std", "mad"}
        How to estimate per-channel amplitude.
    alpha : float
        EMA smoothing factor in (0, 1].  Closer to 1 → faster adaptation
        (close to block-wise).  Default 0.3.
    eps : float
        Floor for the gain denominator to prevent division by zero.
    clip_min, clip_max : float | None
        Optional bounds on the scale factor applied to each channel.
    """

    def __init__(
        self,
        *,
        method: str = "rms",
        alpha: float = 0.3,
        eps: float = 1e-8,
        clip_min: float | None = 0.2,
        clip_max: float | None = 5.0,
    ):
        if method not in ("rms", "std", "mad"):
            raise ValueError(f"method must be 'rms', 'std', or 'mad', got {method!r}")
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.method = method
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.clip_min = None if clip_min is None else float(clip_min)
        self.clip_max = None if clip_max is None else float(clip_max)
        self._gain: np.ndarray | None = None  # shape (nc,)

    # ---- stateful interface ----

    def is_stateful(self) -> bool:
        return True

    def reset(self) -> None:
        self._gain = None

    def get_state(self):
        return {"gain": None if self._gain is None else self._gain.copy()}

    def set_state(self, state) -> None:
        g = state.get("gain")
        self._gain = None if g is None else np.asarray(g, dtype=np.float64).copy()

    # ---- processing ----

    def _estimate_gain(self, x: np.ndarray) -> np.ndarray:
        """Return per-channel amplitude estimate, shape (nc,)."""
        if self.method == "rms":
            return np.sqrt(np.mean(x * x, axis=0))
        if self.method == "std":
            return np.std(x, axis=0)
        # mad
        med = np.median(x, axis=0)
        return 1.4826 * np.median(np.abs(x - med), axis=0)

    def push(self, chunk: np.ndarray) -> np.ndarray:
        """Process one chunk and return gain-equalized output."""
        from .base import as_float2d
        x = as_float2d(chunk, dtype=np.float64)  # (nt, nc)

        chunk_gain = self._estimate_gain(x)

        if self._gain is None or self._gain.shape != chunk_gain.shape:
            self._gain = chunk_gain.copy()
        else:
            self._gain = self.alpha * chunk_gain + (1.0 - self.alpha) * self._gain

        ref = float(np.median(self._gain))
        scale = ref / np.maximum(self._gain, self.eps)  # shape (nc,)
        if self.clip_min is not None or self.clip_max is not None:
            lo = -np.inf if self.clip_min is None else self.clip_min
            hi = np.inf if self.clip_max is None else self.clip_max
            scale = np.clip(scale, lo, hi)

        y = x * scale[np.newaxis, :]  # broadcast over time
        return y.astype(np.float32, copy=False)

    def apply_once(self, data: np.ndarray) -> np.ndarray:
        """Stateless application (no EMA update)."""
        from .base import as_float2d
        x = as_float2d(data, dtype=np.float64)
        gain = self._estimate_gain(x)
        ref = float(np.median(gain))
        scale = ref / np.maximum(gain, self.eps)
        if self.clip_min is not None or self.clip_max is not None:
            lo = -np.inf if self.clip_min is None else self.clip_min
            hi = np.inf if self.clip_max is None else self.clip_max
            scale = np.clip(scale, lo, hi)
        y = x * scale[np.newaxis, :]
        return y.astype(np.float32, copy=False)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return self.push(data)

    def __repr__(self) -> str:
        return (
            f"RealtimeEqualizeChannelGain("
            f"method={self.method!r}, alpha={self.alpha}, "
            f"clip_min={self.clip_min}, clip_max={self.clip_max})"
        )


# ---------------------------------------------------------------------------
# Spectral – realtime (stateful)
# ---------------------------------------------------------------------------

class RealtimeSTFT:
    """Streaming STFT with correct chunk-boundary handling.

    Maintains a ring buffer of the last ``noverlap`` samples so that STFT
    frames that span a chunk boundary are computed correctly.

    The result time axis is adjusted so that *t = 0* corresponds to the first
    sample of the **new** chunk (not the historical buffer).

    .. note::
        The return value is a tuple ``(f, t, S)`` — not a plain array — so
        this transform should **not** be placed inside a ``Compose`` pipeline.
        Use it directly as a streaming feature extractor.

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    nperseg : int
        STFT window length (samples).  Must be ≥ 2.
    hop_length : int | None
        Hop size between successive frames.  Default: ``nperseg // 4``.
    window : str
        Window type passed to ``scipy.signal.get_window``.  Default: ``"hann"``.
    """

    def __init__(
        self,
        *,
        fs: float,
        nperseg: int = 256,
        hop_length: int | None = None,
        window: str = "hann",
    ):
        if nperseg < 2:
            raise ValueError(f"nperseg must be ≥ 2, got {nperseg}")
        self.fs = float(fs)
        self.nperseg = int(nperseg)
        self.hop_length = int(hop_length) if hop_length is not None else max(1, self.nperseg // 4)
        self.window = window

        # noverlap = frames need this many history samples for boundary frames
        self._noverlap = self.nperseg - self.hop_length
        self._buffer: np.ndarray | None = None  # shape (noverlap, nc)

    # ---- stateful interface ----

    def is_stateful(self) -> bool:
        return True

    def reset(self) -> None:
        self._buffer = None

    def get_state(self):
        return {"buffer": None if self._buffer is None else self._buffer.copy()}

    def set_state(self, state) -> None:
        b = state.get("buffer")
        self._buffer = None if b is None else np.asarray(b, dtype=np.float32).copy()

    # ---- processing ----

    def push(self, chunk: np.ndarray):
        """Compute STFT for *chunk*, prepending history samples.

        Parameters
        ----------
        chunk : array (nt, nc) or (nc, nt)

        Returns
        -------
        f : ndarray, shape (nf,)
            Frequency axis in Hz.
        t : ndarray, shape (n_frames,)
            Time axis in seconds, where t=0 is the first sample of *chunk*.
        S : ndarray, shape (nf, n_frames, nc)
            Complex STFT coefficients.
        """
        from scipy.signal import stft as _stft
        from .base import as_float2d

        x = as_float2d(chunk, dtype=np.float32)  # (nt, nc)
        nt, nc = x.shape

        # Prepend history buffer
        if self._buffer is None or self._buffer.shape[1] != nc:
            # No history yet — pad with zeros so first chunk still works
            hist = np.zeros((self._noverlap, nc), dtype=np.float32)
        else:
            hist = self._buffer

        combined = np.concatenate([hist, x], axis=0)  # (noverlap + nt, nc)

        # Compute STFT per channel, then stack
        # scipy.signal.stft works on 1-D or 2-D (n_signals, n_samples) but
        # its output time axis aligns with combined.  We'll use it per-channel
        # and vectorize over nc via a single call treating channels as signals.
        f, t_raw, S_all = _stft(
            combined.T,            # (nc, noverlap+nt)
            fs=self.fs,
            window=self.window,
            nperseg=self.nperseg,
            noverlap=self.nperseg - self.hop_length,
            padded=False,
            axis=-1,
        )
        # S_all shape: (nc, nf, n_frames)

        # Adjust time axis so t=0 is the start of the new chunk
        t = t_raw - float(self._noverlap) / self.fs

        # Rearrange to (nf, n_frames, nc)
        S = np.transpose(S_all, (1, 2, 0))

        # Update buffer with last noverlap samples of combined
        self._buffer = combined[-self._noverlap:].copy() if self._noverlap > 0 else np.zeros((0, nc), dtype=np.float32)

        return f, t, S

    def __call__(self, chunk: np.ndarray):
        return self.push(chunk)

    def __repr__(self) -> str:
        return (
            f"RealtimeSTFT(fs={self.fs}, nperseg={self.nperseg}, "
            f"hop_length={self.hop_length}, window={self.window!r})"
        )


class FIRFilterTime:
    """FIR filter along time axis (firwin)."""

    def __init__(self, *, fs: float, kind: str, fmin: float | None = None,
                 fmax: float | None = None, numtaps: int = 257,
                 window: str = "hann", zero_phase: bool = True):
        self.fs = fs
        self.kind = kind
        self.fmin = fmin
        self.fmax = fmax
        self.numtaps = numtaps
        self.window = window
        self.zero_phase = zero_phase

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return temporal.fir_filter_time(
            data, fs=self.fs, kind=self.kind,  # type: ignore[arg-type]
            fmin=self.fmin, fmax=self.fmax,
            numtaps=self.numtaps, window=self.window, zero_phase=self.zero_phase,
        )

    def chunked(self, data: np.ndarray, *, chunk_channels=None,
                target_chunk_mb: float = 128.0, layout: str = "nt_nc") -> np.ndarray:
        return temporal.fir_filter_time_chunked(
            data, fs=self.fs, kind=self.kind,
            fmin=self.fmin, fmax=self.fmax,
            numtaps=self.numtaps, window=self.window, zero_phase=self.zero_phase,
            chunk_channels=chunk_channels, target_chunk_mb=target_chunk_mb, layout=layout,
        )

    def __repr__(self) -> str:
        return (f"FIRFilterTime(fs={self.fs}, kind={self.kind!r}, "
                f"fmin={self.fmin}, fmax={self.fmax}, numtaps={self.numtaps})")


class FIRBandpassTime:
    """FIR bandpass filter."""

    def __init__(self, *, fs: float, fmin: float, fmax: float,
                 numtaps: int = 257, window: str = "hann", zero_phase: bool = True):
        self.fs = fs
        self.fmin = fmin
        self.fmax = fmax
        self.numtaps = numtaps
        self.window = window
        self.zero_phase = zero_phase

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return temporal.fir_bandpass_time(
            data, fs=self.fs, fmin=self.fmin, fmax=self.fmax,
            numtaps=self.numtaps, window=self.window, zero_phase=self.zero_phase,
        )

    def chunked(self, data: np.ndarray, *, chunk_channels=None,
                target_chunk_mb: float = 128.0, layout: str = "nt_nc") -> np.ndarray:
        return temporal.fir_bandpass_time_chunked(
            data, fs=self.fs, fmin=self.fmin, fmax=self.fmax,
            numtaps=self.numtaps, window=self.window, zero_phase=self.zero_phase,
            chunk_channels=chunk_channels, target_chunk_mb=target_chunk_mb, layout=layout,
        )

    def __repr__(self) -> str:
        return f"FIRBandpassTime(fs={self.fs}, fmin={self.fmin}, fmax={self.fmax})"


class FIRLowpassTime:
    """FIR lowpass filter."""

    def __init__(self, *, fs: float, fmax: float,
                 numtaps: int = 257, window: str = "hann", zero_phase: bool = True):
        self.fs = fs
        self.fmax = fmax
        self.numtaps = numtaps
        self.window = window
        self.zero_phase = zero_phase

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return temporal.fir_lowpass_time(
            data, fs=self.fs, fmax=self.fmax,
            numtaps=self.numtaps, window=self.window, zero_phase=self.zero_phase,
        )

    def chunked(self, data: np.ndarray, *, chunk_channels=None,
                target_chunk_mb: float = 128.0, layout: str = "nt_nc") -> np.ndarray:
        return temporal.fir_lowpass_time_chunked(
            data, fs=self.fs, fmax=self.fmax,
            numtaps=self.numtaps, window=self.window, zero_phase=self.zero_phase,
            chunk_channels=chunk_channels, target_chunk_mb=target_chunk_mb, layout=layout,
        )

    def __repr__(self) -> str:
        return f"FIRLowpassTime(fs={self.fs}, fmax={self.fmax})"


class FIRHighpassTime:
    """FIR highpass filter."""

    def __init__(self, *, fs: float, fmin: float,
                 numtaps: int = 257, window: str = "hann", zero_phase: bool = True):
        self.fs = fs
        self.fmin = fmin
        self.numtaps = numtaps
        self.window = window
        self.zero_phase = zero_phase

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return temporal.fir_highpass_time(
            data, fs=self.fs, fmin=self.fmin,
            numtaps=self.numtaps, window=self.window, zero_phase=self.zero_phase,
        )

    def chunked(self, data: np.ndarray, *, chunk_channels=None,
                target_chunk_mb: float = 128.0, layout: str = "nt_nc") -> np.ndarray:
        return temporal.fir_highpass_time_chunked(
            data, fs=self.fs, fmin=self.fmin,
            numtaps=self.numtaps, window=self.window, zero_phase=self.zero_phase,
            chunk_channels=chunk_channels, target_chunk_mb=target_chunk_mb, layout=layout,
        )

    def __repr__(self) -> str:
        return f"FIRHighpassTime(fs={self.fs}, fmin={self.fmin})"


# ---------------------------------------------------------------------------
# Temporal – FFT-domain filtering
# ---------------------------------------------------------------------------

class FFTBandpassTime:
    """Frequency-domain bandpass with optional cosine taper."""

    def __init__(self, *, fs: float, fmin: float, fmax: float, taper_hz: float = 0.0):
        self.fs = fs
        self.fmin = fmin
        self.fmax = fmax
        self.taper_hz = taper_hz

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return temporal.fft_bandpass_time(
            data, fs=self.fs, fmin=self.fmin, fmax=self.fmax, taper_hz=self.taper_hz,
        )

    def chunked(self, data: np.ndarray, *, chunk_channels=None,
                target_chunk_mb: float = 128.0, layout: str = "nt_nc") -> np.ndarray:
        return temporal.fft_bandpass_time_chunked(
            data, fs=self.fs, fmin=self.fmin, fmax=self.fmax, taper_hz=self.taper_hz,
            chunk_channels=chunk_channels, target_chunk_mb=target_chunk_mb, layout=layout,
        )

    def __repr__(self) -> str:
        return f"FFTBandpassTime(fs={self.fs}, fmin={self.fmin}, fmax={self.fmax})"


class FFTLowpassTime:
    """Frequency-domain lowpass filter with optional cosine taper."""

    def __init__(self, *, fs: float, fmax: float, taper_hz: float = 0.0):
        self.fs = fs
        self.fmax = fmax
        self.taper_hz = taper_hz

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return temporal.fft_lowpass_time(data, fs=self.fs, fmax=self.fmax, taper_hz=self.taper_hz)

    def chunked(self, data: np.ndarray, *, chunk_channels=None,
                target_chunk_mb: float = 128.0, layout: str = "nt_nc") -> np.ndarray:
        return temporal.fft_lowpass_time_chunked(
            data, fs=self.fs, fmax=self.fmax, taper_hz=self.taper_hz,
            chunk_channels=chunk_channels, target_chunk_mb=target_chunk_mb, layout=layout,
        )

    def __repr__(self) -> str:
        return f"FFTLowpassTime(fs={self.fs}, fmax={self.fmax})"


class FFTHighpassTime:
    """Frequency-domain highpass filter with optional cosine taper."""

    def __init__(self, *, fs: float, fmin: float, taper_hz: float = 0.0):
        self.fs = fs
        self.fmin = fmin
        self.taper_hz = taper_hz

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return temporal.fft_highpass_time(data, fs=self.fs, fmin=self.fmin, taper_hz=self.taper_hz)

    def chunked(self, data: np.ndarray, *, chunk_channels=None,
                target_chunk_mb: float = 128.0, layout: str = "nt_nc") -> np.ndarray:
        return temporal.fft_highpass_time_chunked(
            data, fs=self.fs, fmin=self.fmin, taper_hz=self.taper_hz,
            chunk_channels=chunk_channels, target_chunk_mb=target_chunk_mb, layout=layout,
        )

    def __repr__(self) -> str:
        return f"FFTHighpassTime(fs={self.fs}, fmin={self.fmin})"


# ---------------------------------------------------------------------------
# Temporal – amplitude / spectral
# ---------------------------------------------------------------------------

class TemporalWhiten:
    """Frequency-domain temporal whitening per channel."""

    def __init__(self, *, fs: float, fmin: float | None = None,
                 fmax: float | None = None, eps: float = 1e-8):
        self.fs = fs
        self.fmin = fmin
        self.fmax = fmax
        self.eps = eps

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return temporal.temporal_whiten(data, fs=self.fs, fmin=self.fmin, fmax=self.fmax, eps=self.eps)

    def __repr__(self) -> str:
        return f"TemporalWhiten(fs={self.fs}, fmin={self.fmin}, fmax={self.fmax})"


class OneBitNormalize:
    """One-bit (sign) normalization."""

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return temporal.one_bit_normalize(data)

    def __repr__(self) -> str:
        return "OneBitNormalize()"


class ClipAmplitude:
    """Clip amplitude to ±threshold."""

    def __init__(self, *, threshold: float):
        self.threshold = threshold

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return temporal.clip_amplitude(data, threshold=self.threshold)

    def __repr__(self) -> str:
        return f"ClipAmplitude(threshold={self.threshold})"


class AnalyticEnvelope:
    """Instantaneous amplitude envelope via Hilbert transform."""

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return temporal.analytic_envelope(data)

    def __repr__(self) -> str:
        return "AnalyticEnvelope()"


# ---------------------------------------------------------------------------
# Spatial
# ---------------------------------------------------------------------------

class DeMeanSpace:
    """Remove per-time-sample spatial mean across channels."""

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return spatial.demean_space(data)

    def __repr__(self) -> str:
        return "DeMeanSpace()"


class DetrendSpace:
    """Per-time-sample spatial detrend across channels."""

    def __init__(self, *, kind: str = "linear"):
        self.kind = kind

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return spatial.detrend_space(data, kind=self.kind)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        return f"DetrendSpace(kind={self.kind!r})"


class ZScoreSpace:
    """Per-time-sample z-score normalization across channels."""

    def __init__(self, *, robust: bool = False, eps: float = 1e-8):
        self.robust = robust
        self.eps = eps

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return spatial.zscore_space(data, robust=self.robust, eps=self.eps)

    def __repr__(self) -> str:
        return f"ZScoreSpace(robust={self.robust})"


class RunningMeanSpace:
    """Spatial moving average across channels."""

    def __init__(self, *, half_window: int = 4, mode: str = "reflect"):
        self.half_window = half_window
        self.mode = mode

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return spatial.running_mean_space(data, half_window=self.half_window, mode=self.mode)

    def __repr__(self) -> str:
        return f"RunningMeanSpace(half_window={self.half_window})"


class GaussianSmoothSpace:
    """Spatial Gaussian smoothing across channels."""

    def __init__(self, *, sigma: float = 1.0, mode: str = "reflect"):
        self.sigma = sigma
        self.mode = mode

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return spatial.gaussian_smooth_space(data, sigma=self.sigma, mode=self.mode)

    def __repr__(self) -> str:
        return f"GaussianSmoothSpace(sigma={self.sigma})"


class MedianSmoothSpace:
    """Spatial median smoothing across channels."""

    def __init__(self, *, kernel_size: int = 5):
        self.kernel_size = kernel_size

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return spatial.median_smooth_space(data, kernel_size=self.kernel_size)

    def __repr__(self) -> str:
        return f"MedianSmoothSpace(kernel_size={self.kernel_size})"


class EqualizeChannelGain:
    """Scale each channel to unit RMS."""

    def __init__(self, *, eps: float = 1e-8):
        self.eps = eps

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return spatial.equalize_channel_gain(data, eps=self.eps)

    def __repr__(self) -> str:
        return "EqualizeChannelGain()"


class SpatialGradient:
    """Finite-difference spatial gradient."""

    def __init__(self, *, dx: float = 1.0, edge_order: int = 1):
        self.dx = dx
        self.edge_order = edge_order

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return spatial.spatial_gradient(data, dx=self.dx, edge_order=self.edge_order)

    def __repr__(self) -> str:
        return f"SpatialGradient(dx={self.dx}, edge_order={self.edge_order})"


class SpatialLaplacian:
    """Spatial Laplacian (2nd derivative across channels)."""

    def __init__(self, *, dx: float = 1.0):
        self.dx = dx

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return spatial.spatial_laplacian(data, dx=self.dx)

    def __repr__(self) -> str:
        return f"SpatialLaplacian(dx={self.dx})"


class ResampleChannels:
    """Resample channel axis to a target channel count.

    Parameters
    ----------
    out_channels : int
        Target number of output channels.
    kind : "linear" | "nearest" | "cubic"
        Interpolation method.
    """

    def __init__(self, *, out_channels: int, kind: str = "linear"):
        self.out_channels = out_channels
        self.kind = kind

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return spatial.resample_channels(data, out_channels=self.out_channels, kind=self.kind)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        return f"ResampleChannels(out_channels={self.out_channels}, kind={self.kind!r})"


# ---------------------------------------------------------------------------
# F-K domain
# ---------------------------------------------------------------------------

class FKFanFilter:
    """f-k fan filter (velocity / frequency / wavenumber masking)."""

    def __init__(
        self,
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
    ):
        self.dt = dt
        self.dx = dx
        self.vmin = vmin
        self.vmax = vmax
        self.fmin = fmin
        self.fmax = fmax
        self.taper_width = taper_width
        self.kmin_abs = kmin_abs
        self.kmax_abs = kmax_abs

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return fk.fk_fan_filter(
            data,
            dt=self.dt, dx=self.dx,
            vmin=self.vmin, vmax=self.vmax,
            fmin=self.fmin, fmax=self.fmax,
            taper_width=self.taper_width,
            kmin_abs=self.kmin_abs, kmax_abs=self.kmax_abs,
        )

    def __repr__(self) -> str:
        return (f"FKFanFilter(vmin={self.vmin}, vmax={self.vmax}, "
                f"fmin={self.fmin}, fmax={self.fmax})")


class DirectionalWedgeDenoise:
    """Directional denoising in 2-D Fourier space using angle wedges."""

    def __init__(self, *, n_angles: int = 16, threshold: float = 3.0,
                 wedge_sigma: float = 0.12, soft_threshold: bool = False):
        self.n_angles = n_angles
        self.threshold = threshold
        self.wedge_sigma = wedge_sigma
        self.soft_threshold = soft_threshold

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return fk.directional_wedge_denoise(
            data, n_angles=self.n_angles, threshold=self.threshold,
            wedge_sigma=self.wedge_sigma, soft_threshold=self.soft_threshold,
        )

    def __repr__(self) -> str:
        return f"DirectionalWedgeDenoise(n_angles={self.n_angles}, threshold={self.threshold})"


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------

class TruncatedSVDReconstruct:
    """Global low-rank reconstruction by truncated SVD."""

    def __init__(self, *, rank: int):
        self.rank = rank

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return decomposition.truncated_svd_reconstruct(data, rank=self.rank)

    def __repr__(self) -> str:
        return f"TruncatedSVDReconstruct(rank={self.rank})"


class LocalSVDDenoise:
    """Local patch SVD denoising."""

    def __init__(self, *, win_t: int = 64, win_c: int = 32, rank: int = 3,
                 step_t: int | None = None, step_c: int | None = None):
        self.win_t = win_t
        self.win_c = win_c
        self.rank = rank
        self.step_t = step_t
        self.step_c = step_c

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return decomposition.local_svd_denoise(
            data, win_t=self.win_t, win_c=self.win_c, rank=self.rank,
            step_t=self.step_t, step_c=self.step_c,
        )

    def chunked(self, data: np.ndarray, *, tile_t: int = 512, tile_c: int = 256,
                halo_t: int | None = None, halo_c: int | None = None,
                layout: str = "nt_nc") -> np.ndarray:
        """Tiled chunked variant. halo defaults to win_t//2 and win_c//2."""
        return decomposition.local_svd_denoise_chunked(
            data, win_t=self.win_t, win_c=self.win_c, rank=self.rank,
            tile_t=tile_t, tile_c=tile_c, halo_t=halo_t, halo_c=halo_c, layout=layout,
        )

    def __repr__(self) -> str:
        return (f"LocalSVDDenoise(win_t={self.win_t}, win_c={self.win_c}, rank={self.rank})")


# ---------------------------------------------------------------------------
# Nonlinear
# ---------------------------------------------------------------------------

class AnisotropicDiffusion2D:
    """Perona-Malik anisotropic diffusion."""

    def __init__(self, *, n_iter: int = 20, kappa: float = 20.0, gamma: float = 0.1):
        self.n_iter = n_iter
        self.kappa = kappa
        self.gamma = gamma

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return nonlinear.anisotropic_diffusion_2d(
            data, n_iter=self.n_iter, kappa=self.kappa, gamma=self.gamma,
        )

    def chunked(self, data: np.ndarray, *, tile_t: int = 1024, tile_c: int = 512,
                halo_t: int | None = None, halo_c: int | None = None,
                layout: str = "nt_nc") -> np.ndarray:
        """Tiled chunked variant. halo defaults are computed from n_iter."""
        return nonlinear.structure_tensor_filter_chunked(
            data, n_iter=self.n_iter, kappa=self.kappa, gamma=self.gamma,
            tile_t=tile_t, tile_c=tile_c, halo_t=halo_t, halo_c=halo_c, layout=layout,
        )

    def __repr__(self) -> str:
        return f"AnisotropicDiffusion2D(n_iter={self.n_iter}, kappa={self.kappa})"
