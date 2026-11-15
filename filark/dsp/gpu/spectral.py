# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
# Licensed under the MIT License.
"""GPU spectral analysis operators."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from ..base import Layout, to_nt_nc
from ._utils import get_device, require_torch, to_numpy, to_tensor

__all__ = [
    "stft_gather_gpu",
    "psd_estimate_gpu",
    "rms_profile_gpu",
    "spectral_centroid_gpu",
]


def _build_window_gpu(window: str, nperseg: int, dev):
    """Build a window tensor of shape [nperseg]."""
    torch = require_torch()
    if window == "hann":
        win = torch.hann_window(nperseg, device=dev, dtype=torch.float32)
    elif window == "hamming":
        win = torch.hamming_window(nperseg, device=dev, dtype=torch.float32)
    elif window == "blackman":
        win = torch.blackman_window(nperseg, device=dev, dtype=torch.float32)
    elif window == "bartlett":
        win = torch.bartlett_window(nperseg, device=dev, dtype=torch.float32)
    else:
        # Default: rectangular
        win = torch.ones(nperseg, device=dev, dtype=torch.float32)
    return win


def stft_gather_gpu(
    data,
    *,
    fs: float,
    nperseg: int = 256,
    noverlap: int | None = None,
    window: str = "hann",
    device: str = "auto",
    layout: Layout = "nt_nc",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Short-time Fourier transform of a DAS gather (GPU).

    Returns
    -------
    f : np.ndarray, shape (nf,)
        Frequency axis in Hz. nf = nperseg // 2 + 1.
    t : np.ndarray, shape (n_frames,)
        Time axis in seconds (centre of each frame).
    S : np.ndarray, shape (nc, nf, n_frames), complex64
        STFT output.
    """
    torch = require_torch()
    x_np = to_nt_nc(data, layout=layout, dtype=np.float32)
    nt, nc = x_np.shape
    fs = float(fs)
    nperseg = int(nperseg)
    if noverlap is None:
        noverlap = nperseg // 2
    noverlap = int(noverlap)
    hop = nperseg - noverlap
    if hop < 1:
        raise ValueError(f"noverlap must be < nperseg, got noverlap={noverlap}, nperseg={nperseg}")

    dev = get_device(device)
    x = to_tensor(x_np, dev)  # [nt, nc]
    win = _build_window_gpu(window, nperseg, dev)  # [nperseg]

    # x.T: [nc, nt] → unfold along time dimension
    xT = x.t()  # [nc, nt]
    # frames: [nc, n_frames, nperseg]
    frames = xT.unfold(dimension=1, size=nperseg, step=hop)
    n_frames = frames.shape[1]

    # Apply window
    frames = frames * win  # broadcast [nperseg] over [nc, n_frames, nperseg]

    # FFT along last dim
    S_raw = torch.fft.rfft(frames, n=nperseg, dim=-1)  # [nc, n_frames, nf]
    S = S_raw.permute(0, 2, 1)  # [nc, nf, n_frames]

    nf = nperseg // 2 + 1
    # Frequency axis
    f_np = to_numpy(torch.fft.rfftfreq(nperseg, d=1.0 / fs, device=dev)).astype(np.float32)

    # Time axis: centre of each frame
    frame_starts = torch.arange(n_frames, dtype=torch.float32, device=dev) * hop
    t_np = to_numpy((frame_starts + nperseg // 2) / fs).astype(np.float32)

    # Convert S to complex64 numpy
    S_np = to_numpy(torch.view_as_real(S))  # [nc, nf, n_frames, 2]
    S_complex = S_np[..., 0] + 1j * S_np[..., 1]  # [nc, nf, n_frames]
    S_complex = S_complex.astype(np.complex64)

    return f_np, t_np, S_complex


def psd_estimate_gpu(
    data,
    *,
    fs: float,
    nperseg: int = 256,
    noverlap: int | None = None,
    window: str = "hann",
    device: str = "auto",
    layout: Layout = "nt_nc",
) -> Tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density estimate (GPU).

    Returns
    -------
    f : np.ndarray, shape (nf,)
    psd : np.ndarray, shape (nc, nf)
    """
    torch = require_torch()
    x_np = to_nt_nc(data, layout=layout, dtype=np.float32)
    nt, nc = x_np.shape
    fs = float(fs)
    nperseg = int(nperseg)
    if noverlap is None:
        noverlap = nperseg // 2
    noverlap = int(noverlap)
    hop = nperseg - noverlap

    dev = get_device(device)
    x = to_tensor(x_np, dev)  # [nt, nc]
    win = _build_window_gpu(window, nperseg, dev)  # [nperseg]

    xT = x.t()  # [nc, nt]
    frames = xT.unfold(dimension=1, size=nperseg, step=hop)  # [nc, n_frames, nperseg]
    frames = frames * win

    S_raw = torch.fft.rfft(frames, n=nperseg, dim=-1)  # [nc, n_frames, nf]
    power = S_raw.abs() ** 2  # [nc, n_frames, nf]

    # Welch: average over frames, scale by 1/(fs * sum(win^2))
    scale = 1.0 / (fs * (win ** 2).sum().item())
    psd = power.mean(dim=1) * scale  # [nc, nf]

    f_np = to_numpy(torch.fft.rfftfreq(nperseg, d=1.0 / fs, device=dev)).astype(np.float32)
    psd_np = to_numpy(psd).astype(np.float32)

    return f_np, psd_np


def rms_profile_gpu(
    data,
    *,
    device: str = "auto",
    layout: Layout = "nt_nc",
) -> np.ndarray:
    """Per-channel RMS profile (GPU).

    Returns
    -------
    rms : np.ndarray, shape (nc,)
    """
    torch = require_torch()
    x_np = to_nt_nc(data, layout=layout, dtype=np.float32)
    dev = get_device(device)
    x = to_tensor(x_np, dev)  # [nt, nc]
    rms = torch.sqrt((x ** 2).mean(dim=0))  # [nc]
    return to_numpy(rms).astype(np.float32)


def spectral_centroid_gpu(
    f,
    psd,
    *,
    device: str = "auto",
) -> np.ndarray:
    """Weighted spectral centroid per channel (GPU).

    Parameters
    ----------
    f : array-like, shape (nf,)
    psd : array-like, shape (nc, nf)

    Returns
    -------
    centroid : np.ndarray, shape (nc,)
    """
    torch = require_torch()
    dev = get_device(device)
    f_np = np.asarray(f, dtype=np.float32)
    psd_np = np.asarray(psd, dtype=np.float32)

    f_t = to_tensor(f_np, dev)    # [nf]
    psd_t = to_tensor(psd_np, dev)  # [nc, nf]

    # weighted mean: (psd @ f) / clamp(psd.sum(dim=1), min=1e-12)
    numerator = psd_t @ f_t                                # [nc]
    denominator = psd_t.sum(dim=1).clamp(min=1e-12)       # [nc]
    centroid = numerator / denominator                     # [nc]

    return to_numpy(centroid).astype(np.float32)
