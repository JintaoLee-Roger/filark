# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""
filark.explore.plot — diagnostic plot functions for DAS data.

All functions are pure (no Qt dependency) and return a
``matplotlib.figure.Figure``.  Data layout is ``(nt, nc)`` by default.

Quick start (Jupyter)
---------------------
>>> from filark.explore.plot import plot_fk, plot_psd
>>> fig = plot_fk(data, fs=1000, dx=5)
>>> fig          # inline render in notebook
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _robust_clim(arr: np.ndarray, plo: float = 2.0, phi: float = 98.0):
    a = np.asarray(arr, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return -1.0, 1.0
    lo, hi = float(np.percentile(a, plo)), float(np.percentile(a, phi))
    return (lo, lo + 1e-12) if hi <= lo else (lo, hi)


def _sym_clim(arr: np.ndarray, phi: float = 98.0) -> tuple[float, float]:
    a = np.asarray(arr, dtype=float).ravel()
    a = a[np.isfinite(a)]
    vabs = float(np.percentile(np.abs(a), phi)) if a.size else 1.0
    return -vabs, vabs


def _mad_normalize(x: np.ndarray) -> np.ndarray:
    """Per-channel MAD normalization; input (nc, nt)."""
    m = np.median(x, axis=1, keepdims=True)
    s = 1.4826 * np.median(np.abs(x - m), axis=1, keepdims=True) + 1e-12
    return (x - m) / s


def _to_db(power: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    return 10.0 * np.log10(np.asarray(power, dtype=float) + eps)


# ---------------------------------------------------------------------------
# F-K spectrum
# ---------------------------------------------------------------------------

def plot_fk(
    data: np.ndarray,
    *,
    fs: float,
    dx: float,
    normalize: bool = True,
    db: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
    velocities: list[float] | None = None,
    figsize: tuple[float, float] = (8, 5),
) -> Figure:
    """
    F-K power spectrum of a (nt, nc) DAS gather.

    Parameters
    ----------
    data : ndarray, shape (nt, nc)
    fs : float
        Sampling rate (Hz).
    dx : float
        Channel spacing (m).
    normalize : bool
        Apply per-channel MAD normalization before FFT.
    db : bool
        Display power in dB (10·log10).
    vmin, vmax : float, optional
        Color limits.  Auto-computed (2–98 percentile) if not given.
    cmap : str
        Matplotlib colormap name.
    velocities : list[float], optional
        Draw velocity lines (m/s) as f = v·|k| overlays.
    figsize : (w, h)
    """
    from filark.dsp import fk as _fk

    data = np.asarray(data, dtype=np.float32)
    nt, nc = data.shape
    dt = 1.0 / float(fs)
    dx = float(dx)

    src = _mad_normalize(data.T) if normalize else data.T   # (nc, nt)
    spectrum, f_axis, k_axis = _fk.fk_transform(
        data=src, dt=dt, dx=dx,
        backend="scipy", return_power=True, shift=True,
    )
    spectrum = np.asarray(spectrum)
    f_axis   = np.asarray(f_axis)
    k_axis   = np.asarray(k_axis)

    # keep positive frequencies only
    pos = f_axis >= 0
    f_plot = f_axis[pos]
    pwr    = spectrum[pos, :]
    img    = _to_db(pwr) if db else pwr

    if vmin is None or vmax is None:
        _lo, _hi = _robust_clim(img)
        vmin = _lo if vmin is None else vmin
        vmax = _hi if vmax is None else vmax

    fig = Figure(figsize=figsize, constrained_layout=True)
    ax  = fig.add_subplot(111)

    extent = [float(k_axis.min()), float(k_axis.max()),
              float(f_plot.min()), float(f_plot.max())]
    im = ax.imshow(
        img, origin="lower", aspect="auto",
        extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("Power (dB)" if db else "Power")

    ax.set_xlabel("Wavenumber k (1/m)")
    ax.set_ylabel("Frequency f (Hz)")
    ax.set_title("F-K spectrum")

    if velocities:
        f_max = float(f_plot.max())
        for v in velocities:
            k_v = f_max / float(v)
            ax.plot([ k_v,  0, -k_v], [f_max, 0, f_max],
                    lw=0.8, ls="--", label=f"{v:.0f} m/s")
        ax.legend(fontsize=7)

    return fig


# ---------------------------------------------------------------------------
# STFT spectrogram
# ---------------------------------------------------------------------------

def plot_stft(
    data: np.ndarray,
    *,
    fs: float,
    channel: int | None = None,
    nperseg: int = 256,
    noverlap: int | None = None,
    db: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
    figsize: tuple[float, float] = (9, 4),
) -> Figure:
    """
    STFT spectrogram of a single channel (or channel-averaged trace).

    Parameters
    ----------
    data : ndarray, shape (nt, nc)
    fs : float
    channel : int, optional
        Channel index to plot.  If None, all channels are averaged.
    nperseg : int
        STFT window length.
    noverlap : int, optional
        Overlap samples; defaults to nperseg // 2.
    """
    from scipy.signal import stft as _stft

    data = np.asarray(data, dtype=np.float32)
    nt, nc = data.shape
    noverlap = noverlap if noverlap is not None else nperseg // 2

    if channel is None:
        trace = data.mean(axis=1)
        ch_label = "all channels (mean)"
    else:
        trace = data[:, int(channel)]
        ch_label = f"channel {channel}"

    f, t, Z = _stft(trace, fs=float(fs), nperseg=nperseg, noverlap=noverlap, window="hann")
    power = np.abs(Z) ** 2
    img   = _to_db(power) if db else power

    if vmin is None or vmax is None:
        _lo, _hi = _robust_clim(img)
        vmin = _lo if vmin is None else vmin
        vmax = _hi if vmax is None else vmax

    fig = Figure(figsize=figsize, constrained_layout=True)
    ax  = fig.add_subplot(111)

    im = ax.pcolormesh(t, f, img, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("Power (dB)" if db else "Power")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"STFT — {ch_label}")

    return fig


# ---------------------------------------------------------------------------
# Power spectral density
# ---------------------------------------------------------------------------

def plot_psd(
    data: np.ndarray,
    *,
    fs: float,
    nperseg: int = 256,
    noverlap: int | None = None,
    dx: float = 1.0,
    db: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
    figsize: tuple[float, float] = (9, 6),
) -> Figure:
    """
    2-panel PSD: per-channel image (top) + mean PSD line (bottom).

    Parameters
    ----------
    data : ndarray, shape (nt, nc)
    fs : float
    nperseg : int
    noverlap : int, optional
    dx : float
        Channel spacing (m) — used for y-axis label.
    """
    from scipy.signal import welch

    data = np.asarray(data, dtype=np.float32)
    nt, nc = data.shape
    noverlap = noverlap if noverlap is not None else nperseg // 2
    nperseg  = min(nperseg, nt)

    f_ref, _ = welch(data[:, 0], fs=float(fs), nperseg=nperseg, noverlap=noverlap)
    psds = np.stack([
        welch(data[:, ic], fs=float(fs), nperseg=nperseg, noverlap=noverlap)[1]
        for ic in range(nc)
    ])                       # (nc, nf)

    img  = _to_db(psds) if db else psds
    mean = img.mean(axis=0)

    if vmin is None or vmax is None:
        _lo, _hi = _robust_clim(img)
        vmin = _lo if vmin is None else vmin
        vmax = _hi if vmax is None else vmax

    ch_axis = np.arange(nc) * float(dx)

    fig = Figure(figsize=figsize, constrained_layout=True)
    ax_img  = fig.add_subplot(2, 1, 1)
    ax_line = fig.add_subplot(2, 1, 2, sharex=ax_img)

    im = ax_img.pcolormesh(f_ref, ch_axis, img,
                           cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    cb = fig.colorbar(im, ax=ax_img, shrink=0.85)
    cb.set_label("PSD (dB)" if db else "PSD")
    ax_img.set_ylabel("Channel offset (m)" if dx != 1.0 else "Channel")
    ax_img.set_title("PSD per channel")

    ax_line.plot(f_ref, mean, lw=1.5, color="steelblue")
    ax_line.set_xlabel("Frequency (Hz)")
    ax_line.set_ylabel("Mean PSD (dB)" if db else "Mean PSD")
    ax_line.grid(True, ls="--", alpha=0.4)

    return fig


# ---------------------------------------------------------------------------
# Cross-correlation gather
# ---------------------------------------------------------------------------

def plot_xcorr(
    data: np.ndarray,
    *,
    fs: float,
    dx: float = 1.0,
    ref_channel: int = 0,
    max_lag: float | None = None,
    normalize: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "seismic",
    figsize: tuple[float, float] = (8, 5),
) -> Figure:
    """
    Cross-correlation gather: each channel vs. a reference.

    Parameters
    ----------
    data : ndarray, shape (nt, nc)
    fs : float
    dx : float
        Channel spacing for y-axis.
    ref_channel : int
    max_lag : float, optional
        Maximum lag to display (seconds).  Defaults to nt/(2*fs).
    normalize : bool
        Normalize correlations to [−1, 1].
    """
    from filark.dsp.xcorr import xcorr_gather

    data = np.asarray(data, dtype=np.float32)
    nt, nc = data.shape
    fs = float(fs)
    max_lag_samples = int(max_lag * fs) if max_lag is not None else nt // 2

    corr, lags = xcorr_gather(
        data, ref_channel=int(ref_channel),
        max_lag=max_lag_samples, normalize=normalize,
        layout="nt_nc",
    )
    # corr: (n_lag, nc), lags: (n_lag,)
    lag_s    = lags / fs
    ch_axis  = np.arange(nc) * float(dx)

    if vmin is None or vmax is None:
        _lo, _hi = _robust_clim(corr)
        vmin = _lo if vmin is None else vmin
        vmax = _hi if vmax is None else vmax

    fig = Figure(figsize=figsize, constrained_layout=True)
    ax  = fig.add_subplot(111)

    im = ax.pcolormesh(lag_s, ch_axis, corr.T, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    ax.axvline(0, color="k", lw=0.6, ls="--")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("Normalized xcorr" if normalize else "xcorr")
    ax.set_xlabel("Lag (s)")
    ax.set_ylabel("Channel offset (m)" if dx != 1.0 else "Channel")
    ax.set_title(f"xcorr gather — ref ch {ref_channel}")

    return fig


# ---------------------------------------------------------------------------
# τ-p / linear Radon transform
# ---------------------------------------------------------------------------

def plot_tp(
    data: np.ndarray,
    *,
    fs: float,
    dx: float,
    p_min: float | None = None,
    p_max: float | None = None,
    n_p: int = 100,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "seismic",
    figsize: tuple[float, float] = (7, 6),
) -> Figure:
    """
    Linear τ-p (Radon) transform.

    Parameters
    ----------
    data : ndarray, shape (nt, nc)
    fs : float
    dx : float
        Channel spacing (m).
    p_min, p_max : float, optional
        Slowness range (s/m).  Auto-estimated if not given.
    n_p : int
        Number of slowness samples.
    """
    from filark.dsp.tp import linear_radon_forward

    data = np.asarray(data, dtype=np.float32)
    nt, nc = data.shape
    dt = 1.0 / float(fs)
    dx = float(dx)

    if p_min is None or p_max is None:
        _p_abs = min(1.0 / max(100.0, 1e-3),
                     1.0 / (dx * float(fs) * 0.5 + 1e-12))
        p_min = -_p_abs if p_min is None else p_min
        p_max =  _p_abs if p_max is None else p_max

    p_values = np.linspace(float(p_min), float(p_max), int(n_p))
    tp_data  = linear_radon_forward(data, p_values=p_values, dt=dt, dx=dx, layout="nt_nc")
    # tp_data: (n_p, nt) → transpose to (nt, n_p) for display
    tp_plot = tp_data.T
    t_axis  = np.arange(nt) * dt

    if vmin is None or vmax is None:
        _lo, _hi = _robust_clim(tp_plot)
        vmin = _lo if vmin is None else vmin
        vmax = _hi if vmax is None else vmax

    fig = Figure(figsize=figsize, constrained_layout=True)
    ax  = fig.add_subplot(111)

    im = ax.pcolormesh(
        p_values * 1e3, t_axis, tp_plot,
        cmap=cmap, vmin=vmin, vmax=vmax, shading="auto",
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("Amplitude")
    ax.set_xlabel("p (ms/m)")
    ax.set_ylabel("τ (s)")
    ax.set_title("Linear τ-p transform")

    return fig


# ---------------------------------------------------------------------------
# RMS energy profile
# ---------------------------------------------------------------------------

def plot_rms(
    data: np.ndarray,
    *,
    dx: float = 1.0,
    win_samples: int | None = None,
    figsize: tuple[float, float] = (9, 4),
) -> Figure:
    """
    RMS energy: channel profile bar chart and (optionally) windowed heatmap.

    Parameters
    ----------
    data : ndarray, shape (nt, nc)
    dx : float
        Channel spacing for axis labelling.
    win_samples : int, optional
        If given, also plot a windowed RMS heatmap (time × channel).
        Window stride = win_samples // 2.
    """
    data = np.asarray(data, dtype=np.float32)
    nt, nc = data.shape
    ch_axis = np.arange(nc) * float(dx)
    rms = np.sqrt(np.mean(data ** 2, axis=0))   # (nc,)

    if win_samples is None:
        fig = Figure(figsize=figsize, constrained_layout=True)
        ax  = fig.add_subplot(111)
        ax.barh(ch_axis, rms, height=float(dx) * 0.9 if dx != 1.0 else 0.8, alpha=0.8)
        ax.invert_yaxis()
        ax.set_xlabel("RMS amplitude")
        ax.set_ylabel("Channel offset (m)" if dx != 1.0 else "Channel")
        ax.set_title("RMS energy profile")
    else:
        win  = int(win_samples)
        stride = max(1, win // 2)
        n_wins = max(1, (nt - win) // stride + 1)
        rms_map = np.zeros((n_wins, nc), dtype=np.float32)
        for i in range(n_wins):
            t0 = i * stride
            seg = data[t0: t0 + win, :]
            rms_map[i] = np.sqrt(np.mean(seg ** 2, axis=0))
        t_axis = np.arange(n_wins) * stride / max(1.0, float(nt)) * nt  # sample indices

        vmin, vmax = _robust_clim(rms_map)

        fig = Figure(figsize=(figsize[0], figsize[1] * 1.6), constrained_layout=True)
        ax_bar  = fig.add_subplot(2, 1, 1)
        ax_heat = fig.add_subplot(2, 1, 2)

        ax_bar.barh(ch_axis, rms, height=float(dx) * 0.9 if dx != 1.0 else 0.8, alpha=0.8)
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel("RMS amplitude")
        ax_bar.set_ylabel("Channel" )
        ax_bar.set_title("RMS profile")

        im = ax_heat.pcolormesh(ch_axis, t_axis, rms_map,
                                cmap="hot", vmin=vmin, vmax=vmax, shading="auto")
        fig.colorbar(im, ax=ax_heat, shrink=0.85).set_label("RMS")
        ax_heat.set_xlabel("Channel offset (m)" if dx != 1.0 else "Channel")
        ax_heat.set_ylabel("Sample")
        ax_heat.set_title("Windowed RMS heatmap")

    return fig


# ---------------------------------------------------------------------------
# SVD singular-value decomposition
# ---------------------------------------------------------------------------

def plot_svd(
    data: np.ndarray,
    *,
    fs: float = 1.0,
    dx: float = 1.0,
    n_show: int = 20,
    n_reconstruct: int = 5,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "seismic",
    figsize: tuple[float, float] = (11, 5),
) -> Figure:
    """
    SVD: singular-value bar chart + low-rank reconstruction.

    Parameters
    ----------
    data : ndarray, shape (nt, nc)
    fs : float
    dx : float
    n_show : int
        Number of singular values to display in the bar chart.
    n_reconstruct : int
        Number of components used for the reconstruction image.
    """
    data = np.asarray(data, dtype=np.float32)
    nt, nc = data.shape
    dt = 1.0 / float(fs)

    U, s, Vt = np.linalg.svd(data, full_matrices=False)

    k        = min(int(n_reconstruct), len(s))
    n_show   = min(int(n_show), len(s))
    rec      = (U[:, :k] * s[None, :k]) @ Vt[:k, :]

    t_axis  = np.arange(nt) * dt
    ch_axis = np.arange(nc) * float(dx)
    idx     = np.arange(n_show)
    energy  = np.cumsum(s[:n_show] ** 2) / np.sum(s ** 2) * 100.0

    if vmin is None or vmax is None:
        _vabs   = float(np.percentile(np.abs(rec), 98))
        vmin = -_vabs if vmin is None else vmin
        vmax =  _vabs if vmax is None else vmax

    fig = Figure(figsize=figsize, constrained_layout=True)
    ax_sv  = fig.add_subplot(1, 2, 1)
    ax_rec = fig.add_subplot(1, 2, 2)

    ax_sv.bar(idx, s[:n_show], alpha=0.8, color="steelblue")
    ax_sv.axvline(k - 0.5, color="tomato", lw=1.2, ls="--",
                  label=f"k={k}")
    ax_sv.set_xlabel("Singular value index")
    ax_sv.set_ylabel("Singular value")
    ax_sv.set_title(f"Singular values (top {n_show})")

    ax_sv2 = ax_sv.twinx()
    ax_sv2.plot(idx, energy, "o-", ms=3, color="darkorange", lw=1.0)
    ax_sv2.set_ylabel("Cumulative energy (%)")
    ax_sv2.set_ylim(0, 105)
    ax_sv.legend(fontsize=8)

    im = ax_rec.pcolormesh(ch_axis, t_axis, rec,
                           cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    fig.colorbar(im, ax=ax_rec, shrink=0.85).set_label("Amplitude")
    ax_rec.set_xlabel("Channel offset (m)" if dx != 1.0 else "Channel")
    ax_rec.set_ylabel("Time (s)")
    ax_rec.set_title(f"Low-rank reconstruction (k={k})")

    return fig
