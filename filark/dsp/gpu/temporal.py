# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""GPU temporal filters — FFT-domain and FIR convolution."""

from __future__ import annotations

import numpy as np

from ..base import Layout, from_nt_nc, require_positive, to_nt_nc
from ._utils import build_fft_mask_1d, fft_filter_gpu, get_device, require_torch, to_numpy, to_tensor

__all__ = [
    "fft_bandpass_time_gpu",
    "fft_lowpass_time_gpu",
    "fft_highpass_time_gpu",
    "fir_filter_time_gpu",
    "iir_filter_time_rt_gpu",
    "iir_bandpass_time_rt_gpu",
    "iir_lowpass_time_rt_gpu",
    "iir_highpass_time_rt_gpu",
]


def fft_bandpass_time_gpu(
    data,
    *,
    fs: float,
    fmin: float,
    fmax: float,
    taper_hz: float = 0.0,
    device: str = "auto",
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """GPU FFT bandpass filter. Drop-in replacement for `fft_bandpass_time`."""
    x_np = to_nt_nc(data, layout=layout, dtype=np.float32)
    nt = x_np.shape[0]
    fs_v = require_positive("fs", fs)
    freqs = np.fft.rfftfreq(nt, d=1.0 / fs_v).astype(np.float32)
    mask = build_fft_mask_1d(freqs, fs=fs_v, kind="bandpass", fmin=fmin, fmax=fmax, taper_hz=taper_hz)
    dev = get_device(device)
    y = fft_filter_gpu(x_np, mask_np=mask, dev=dev)
    return from_nt_nc(y, layout=layout)


def fft_lowpass_time_gpu(
    data,
    *,
    fs: float,
    fmax: float,
    taper_hz: float = 0.0,
    device: str = "auto",
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """GPU FFT lowpass filter."""
    x_np = to_nt_nc(data, layout=layout, dtype=np.float32)
    nt = x_np.shape[0]
    fs_v = require_positive("fs", fs)
    freqs = np.fft.rfftfreq(nt, d=1.0 / fs_v).astype(np.float32)
    mask = build_fft_mask_1d(freqs, fs=fs_v, kind="lowpass", fmin=None, fmax=fmax, taper_hz=taper_hz)
    dev = get_device(device)
    y = fft_filter_gpu(x_np, mask_np=mask, dev=dev)
    return from_nt_nc(y, layout=layout)


def fft_highpass_time_gpu(
    data,
    *,
    fs: float,
    fmin: float,
    taper_hz: float = 0.0,
    device: str = "auto",
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """GPU FFT highpass filter."""
    x_np = to_nt_nc(data, layout=layout, dtype=np.float32)
    nt = x_np.shape[0]
    fs_v = require_positive("fs", fs)
    freqs = np.fft.rfftfreq(nt, d=1.0 / fs_v).astype(np.float32)
    mask = build_fft_mask_1d(freqs, fs=fs_v, kind="highpass", fmin=fmin, fmax=None, taper_hz=taper_hz)
    dev = get_device(device)
    y = fft_filter_gpu(x_np, mask_np=mask, dev=dev)
    return from_nt_nc(y, layout=layout)


def iir_filter_time_rt_gpu(
    data,
    sos_np: np.ndarray,
    *,
    zi_np: np.ndarray | None = None,
    device: str = "auto",
    layout: Layout = "nt_nc",
) -> tuple[np.ndarray, np.ndarray]:
    """GPU causal stateful IIR filter using the Direct-Form II Transposed (DF-II-T) algorithm.

    Vectorizes over channels simultaneously while stepping through time sequentially
    (IIR recurrence is inherently sequential in time).  Every time step launches
    several small kernels, so for long blocks the launch overhead dominates and
    the CPU implementation is usually faster: 93,000 samples x 6,455 channels
    took about 36 s on an RTX 5090, against about 4 s on one CPU core.

    Parameters
    ----------
    data : array-like, shape (nt, nc) in ``nt_nc`` layout
        Input data block.
    sos_np : np.ndarray, shape (n_sections, 6)
        SOS coefficient matrix from ``build_iir_sos()`` or scipy butter.
    zi_np : np.ndarray or None, shape (n_sections, 2, nc)
        Initial filter state (same convention as scipy sosfilt with axis=0).
        ``None`` = start from zero.
    device : str
        Target device: ``"auto"`` (CUDA > MPS > CPU), ``"cuda"``, ``"mps"``,
        or ``"cpu"``.
    layout : "nt_nc" | "nc_nt" | "auto"
        Data layout convention.

    Returns
    -------
    y : np.ndarray, shape (nt, nc), float32
        Filtered output.
    zi_out : np.ndarray, shape (n_sections, 2, nc), float32
        Updated filter state for the next chunk.
    """
    torch = require_torch()
    x_np = to_nt_nc(data, layout=layout, dtype=np.float32)
    nt, nc = x_np.shape
    n_sec = int(sos_np.shape[0])

    dev = get_device(device)
    # SOS layout: [b0, b1, b2, 1, a1, a2] per row
    sos = torch.tensor(sos_np.astype(np.float32), device=dev)  # [n_sec, 6]
    x = to_tensor(x_np, dev)                                   # [nt, nc]

    # zi shape convention matches scipy: (n_sec, 2, nc) where axis=0
    # zi[:, 0, :] = z1 (first delay), zi[:, 1, :] = z2 (second delay)
    if zi_np is None:
        zi = torch.zeros((n_sec, 2, nc), dtype=torch.float32, device=dev)
    else:
        zi = torch.tensor(zi_np.astype(np.float32), device=dev)  # [n_sec, 2, nc]

    # DF-II-T recurrence, vectorized over nc channels simultaneously.
    # Sequential over time (IIR recurrence is inherently causal).
    # Inner section loop is small (2–8) so Python overhead is negligible.
    y = torch.empty_like(x)
    for t in range(nt):
        inp = x[t]  # [nc]
        for s in range(n_sec):
            b0, b1, b2 = sos[s, 0], sos[s, 1], sos[s, 2]
            a1, a2 = sos[s, 4], sos[s, 5]
            yt = b0 * inp + zi[s, 0]        # [nc]
            new_z0 = b1 * inp - a1 * yt + zi[s, 1]
            new_z1 = b2 * inp - a2 * yt
            zi[s, 0] = new_z0
            zi[s, 1] = new_z1
            inp = yt
        y[t] = inp

    zi_out = to_numpy(zi).astype(np.float32)
    y_np = to_numpy(y).astype(np.float32)
    return from_nt_nc(y_np, layout=layout), zi_out


def iir_bandpass_time_rt_gpu(
    data,
    *,
    fs: float,
    fmin: float,
    fmax: float,
    order: int = 4,
    zi_np: np.ndarray | None = None,
    device: str = "auto",
    layout: Layout = "nt_nc",
) -> tuple[np.ndarray, np.ndarray]:
    """GPU causal bandpass IIR filter with state threading."""
    from ..temporal import build_iir_sos
    sos = build_iir_sos(fs=fs, kind="bandpass", fmin=fmin, fmax=fmax, order=order)
    return iir_filter_time_rt_gpu(data, sos, zi_np=zi_np, device=device, layout=layout)


def iir_lowpass_time_rt_gpu(
    data,
    *,
    fs: float,
    fmax: float,
    order: int = 4,
    zi_np: np.ndarray | None = None,
    device: str = "auto",
    layout: Layout = "nt_nc",
) -> tuple[np.ndarray, np.ndarray]:
    """GPU causal lowpass IIR filter with state threading."""
    from ..temporal import build_iir_sos
    sos = build_iir_sos(fs=fs, kind="lowpass", fmax=fmax, order=order)
    return iir_filter_time_rt_gpu(data, sos, zi_np=zi_np, device=device, layout=layout)


def iir_highpass_time_rt_gpu(
    data,
    *,
    fs: float,
    fmin: float,
    order: int = 4,
    zi_np: np.ndarray | None = None,
    device: str = "auto",
    layout: Layout = "nt_nc",
) -> tuple[np.ndarray, np.ndarray]:
    """GPU causal highpass IIR filter with state threading."""
    from ..temporal import build_iir_sos
    sos = build_iir_sos(fs=fs, kind="highpass", fmin=fmin, order=order)
    return iir_filter_time_rt_gpu(data, sos, zi_np=zi_np, device=device, layout=layout)


def fir_filter_time_gpu(
    data,
    *,
    fs: float,
    kind: str,
    fmin: float | None = None,
    fmax: float | None = None,
    numtaps: int = 257,
    window: str = "hann",
    zero_phase: bool = True,
    device: str = "auto",
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """FIR filter via grouped 1-D convolution on GPU.

    For symmetric (firwin) kernels with ``zero_phase=True``, padding=numtaps//2
    gives exact zero-phase output without forward-backward passes.
    """
    from scipy import signal as sp_signal

    torch = require_torch()
    import torch.nn.functional as F

    x_np = to_nt_nc(data, layout=layout, dtype=np.float32)
    nt, nc = x_np.shape

    numtaps = int(numtaps)
    if numtaps < 3:
        raise ValueError(f"numtaps must be >= 3, got {numtaps}")
    if numtaps % 2 == 0:
        numtaps += 1
    fs_v = require_positive("fs", fs)

    if kind == "bandpass":
        b = sp_signal.firwin(numtaps, [float(fmin), float(fmax)], pass_zero=False, fs=fs_v, window=window)
    elif kind == "lowpass":
        b = sp_signal.firwin(numtaps, float(fmax), pass_zero=True, fs=fs_v, window=window)
    elif kind == "highpass":
        b = sp_signal.firwin(numtaps, float(fmin), pass_zero=False, fs=fs_v, window=window)
    else:
        raise ValueError(f"Unknown filter kind: {kind!r}")

    dev = get_device(device)
    kernel = torch.tensor(b, dtype=torch.float32, device=dev).view(1, 1, -1)
    kernel = kernel.expand(nc, 1, -1).contiguous()          # [nc, 1, numtaps]
    x_t = to_tensor(x_np, dev).T.unsqueeze(0)              # [1, nc, nt]

    if zero_phase:
        pad = numtaps // 2
        y_t = F.conv1d(x_t, kernel, groups=nc, padding=pad)
    else:
        x_t_padded = F.pad(x_t, (numtaps - 1, 0))
        y_t = F.conv1d(x_t_padded, kernel, groups=nc)

    y_np = to_numpy(y_t.squeeze(0).T).astype(np.float32)   # [nt, nc]
    return from_nt_nc(y_np, layout=layout)
