# Copyright (c) 2026 Jintao Li & Weichang Li. 
# Zhejiang University (ZJU).
# 
# Licensed under the MIT License.

"""High-level denoising entry points that dispatch to lower-level DSP kernels."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

from filark.dsp.decomposition import local_svd_denoise as _local_svd_denoise
from filark.dsp.fk import directional_wedge_denoise as _directional_wedge_denoise
from filark.dsp.fk import fk_fan_filter as _fk_fan_filter
from filark.dsp.nonlinear import structure_tensor_filter as _structure_tensor_filter
from filark.dsp.temporal import (
    fft_bandpass_time as _fft_bandpass_time,
    fir_bandpass_time as _fir_bandpass_time,
    iir_bandpass_time as _iir_bandpass_time,
    running_mean_time as _running_mean_time,
)
from filark.dsp.tp import parabolic_radon_sparse as _parabolic_radon_sparse


DENOISE_METHOD_LABELS: dict[str, str] = {
    "moving_average": "Moving Average",
    "gaussian": "Gaussian",
    "bandpass": "Bandpass (FFT)",
    "bandpass_iir": "Bandpass (IIR Butterworth)",
    "bandpass_fir": "Bandpass (FIR)",
    "fk_fan": "F-K Fan Filter",
    "directional_wedge": "Directional Wedge",
    "parabolic_radon": "Parabolic Radon Sparse",
    "local_svd": "Local SVD",
    "structure_tensor": "Structure Tensor Diffusion",
}


DEFAULT_DENOISE_PARAMS: dict[str, dict[str, Any]] = {
    "moving_average": {"half_window": 8},
    "gaussian": {"half_window": 8},
    "bandpass": {"fmin": 5.0, "fmax": 250.0, "taper_hz": 0.0},
    "bandpass_iir": {"fmin": 5.0, "fmax": 250.0, "order": 4, "zero_phase": True},
    "bandpass_fir": {"fmin": 5.0, "fmax": 250.0, "numtaps": 257, "window": "hann", "zero_phase": True},
    "fk_fan": {
        "vmin": 400.0,
        "vmax": 5000.0,
        "fmin": 2.0,
        "fmax": 300.0,
        "taper_width": 200.0,
        "kmin_abs": None,
        "kmax_abs": None,
    },
    "directional_wedge": {
        "n_angles": 16,
        "threshold": 3.0,
        "wedge_sigma": 0.12,
        "soft_threshold": False,
    },
    "parabolic_radon": {
        "qmin": -0.002,
        "qmax": 0.002,
        "nq": 81,
        "threshold_factor": 3.0,
    },
    "local_svd": {
        "win_t": 64,
        "win_c": 32,
        "rank": 3,
        "step_t": None,
        "step_c": None,
    },
    "structure_tensor": {
        "n_iter": 20,
        "kappa": 20.0,
        "gamma": 0.1,
    },
}


def clone_default_params() -> dict[str, dict[str, Any]]:
    return deepcopy(DEFAULT_DENOISE_PARAMS)


def _as_float2d(data: np.ndarray | Any) -> np.ndarray:
    x = np.asarray(data, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got ndim={x.ndim}")
    return x


def _sanitize_half_window(v: int | float) -> int:
    iv = int(v)
    if iv < 1:
        raise ValueError(f"half_window must be >= 1, got {v}")
    return iv


def _require_positive(name: str, value: float | int) -> float:
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be a finite positive number, got {value}")
    return v


def _optional_float(v: Any) -> float | None:
    if v in (None, ""):
        return None
    out = float(v)
    if not np.isfinite(out):
        raise ValueError(f"Invalid numeric value: {v}")
    return out


def _optional_int(v: Any) -> int | None:
    if v in (None, ""):
        return None
    out = int(v)
    return out


def _optional_bool(v: Any, *, default: bool) -> bool:
    if v in (None, ""):
        return bool(default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, np.integer)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"Invalid bool value: {v}")


def moving_average_denoise(data: np.ndarray, *, half_window: int = 8) -> np.ndarray:
    x = _as_float2d(data)
    hw = _sanitize_half_window(half_window)
    return _running_mean_time(x, half_window=hw, mode="reflect")


def gaussian_denoise(data: np.ndarray, *, half_window: int = 8) -> np.ndarray:
    x = _as_float2d(data)
    hw = _sanitize_half_window(half_window)
    sigma = max(0.5, hw / 2.0)
    # Running through band-limited FFT would not match the old behavior.
    out = gaussian_filter1d(x, sigma=float(sigma), axis=0, mode="reflect")
    return np.asarray(out, dtype=np.float32)


def bandpass_denoise(
    data: np.ndarray,
    *,
    fs: float,
    fmin: float = 5.0,
    fmax: float = 250.0,
    taper_hz: float = 0.0,
) -> np.ndarray:
    """
    Non-IIR bandpass (frequency mask), kept as the default "bandpass".
    """
    x = _as_float2d(data)
    fs = _require_positive("fs", fs)
    return _fft_bandpass_time(
        x,
        fs=fs,
        fmin=float(fmin),
        fmax=float(fmax),
        taper_hz=float(taper_hz),
    )


def bandpass_iir_denoise(
    data: np.ndarray,
    *,
    fs: float,
    fmin: float = 5.0,
    fmax: float = 250.0,
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    x = _as_float2d(data)
    fs = _require_positive("fs", fs)
    return _iir_bandpass_time(
        x,
        fs=fs,
        fmin=float(fmin),
        fmax=float(fmax),
        order=int(order),
        zero_phase=bool(zero_phase),
    )


def bandpass_fir_denoise(
    data: np.ndarray,
    *,
    fs: float,
    fmin: float = 5.0,
    fmax: float = 250.0,
    numtaps: int = 257,
    window: str = "hann",
    zero_phase: bool = True,
) -> np.ndarray:
    x = _as_float2d(data)
    fs = _require_positive("fs", fs)
    return _fir_bandpass_time(
        x,
        fs=fs,
        fmin=float(fmin),
        fmax=float(fmax),
        numtaps=int(numtaps),
        window=str(window),
        zero_phase=bool(zero_phase),
    )


def fk_fan_denoise(
    data: np.ndarray,
    *,
    fs: float,
    dx: float,
    vmin: float | None = 400.0,
    vmax: float | None = 5000.0,
    fmin: float | None = 2.0,
    fmax: float | None = 300.0,
    taper_width: float = 200.0,
    kmin_abs: float | None = None,
    kmax_abs: float | None = None,
) -> np.ndarray:
    x = _as_float2d(data)
    fs = _require_positive("fs", fs)
    dx = _require_positive("dx", dx)
    dt = 1.0 / fs
    return _fk_fan_filter(
        x,
        dt=dt,
        dx=dx,
        vmin=vmin,
        vmax=vmax,
        fmin=fmin,
        fmax=fmax,
        taper_width=float(taper_width),
        kmin_abs=kmin_abs,
        kmax_abs=kmax_abs,
    )


def directional_wedge_denoise(
    data: np.ndarray,
    *,
    n_angles: int = 16,
    threshold: float = 3.0,
    wedge_sigma: float = 0.12,
    soft_threshold: bool = False,
) -> np.ndarray:
    x = _as_float2d(data)
    return _directional_wedge_denoise(
        x,
        n_angles=int(n_angles),
        threshold=float(threshold),
        wedge_sigma=float(wedge_sigma),
        soft_threshold=bool(soft_threshold),
    )


def parabolic_radon_denoise(
    data: np.ndarray,
    *,
    qmin: float = -0.002,
    qmax: float = 0.002,
    nq: int = 81,
    threshold_factor: float = 3.0,
    fs: float = 1.0,
    dx: float = 1.0,
) -> np.ndarray:
    x = _as_float2d(data)
    fs = _require_positive("fs", fs)
    dx = _require_positive("dx", dx)
    dt = 1.0 / fs
    return _parabolic_radon_sparse(
        x,
        qmin=float(qmin),
        qmax=float(qmax),
        nq=int(nq),
        threshold_factor=float(threshold_factor),
        dt=dt,
        dx=dx,
        layout="nt_nc",
    )


def local_svd_denoise(
    data: np.ndarray,
    *,
    win_t: int = 64,
    win_c: int = 32,
    rank: int = 3,
    step_t: int | None = None,
    step_c: int | None = None,
) -> np.ndarray:
    x = _as_float2d(data)
    return _local_svd_denoise(
        x,
        win_t=int(win_t),
        win_c=int(win_c),
        rank=int(rank),
        step_t=_optional_int(step_t),
        step_c=_optional_int(step_c),
        layout="nt_nc",
    )


def structure_tensor_denoise(
    data: np.ndarray,
    *,
    n_iter: int = 20,
    kappa: float = 20.0,
    gamma: float = 0.1,
) -> np.ndarray:
    x = _as_float2d(data)
    return _structure_tensor_filter(
        x,
        n_iter=int(n_iter),
        kappa=float(kappa),
        gamma=float(gamma),
        layout="nt_nc",
    )


def apply_denoise(
    data: np.ndarray,
    *,
    method: str,
    fs: float,
    dx: float,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    cfg = dict(params or {})
    m = str(method or "").strip().lower()
    fs = float(fs)
    dx = float(dx)

    if m == "moving_average":
        return moving_average_denoise(
            data,
            half_window=_sanitize_half_window(cfg.get("half_window", 8)),
        )
    if m == "gaussian":
        return gaussian_denoise(
            data,
            half_window=_sanitize_half_window(cfg.get("half_window", 8)),
        )

    if m in ("bandpass", "bandpass_fft"):
        fmin = _optional_float(cfg.get("fmin", 5.0))
        fmax = _optional_float(cfg.get("fmax", 250.0))
        if fmin is None:
            fmin = 5.0
        if fmax is None:
            fmax = 250.0
        return bandpass_denoise(
            data,
            fs=fs,
            fmin=fmin,
            fmax=fmax,
            taper_hz=float(cfg.get("taper_hz", 0.0)),
        )

    if m == "bandpass_iir":
        return bandpass_iir_denoise(
            data,
            fs=fs,
            fmin=float(cfg.get("fmin", 5.0)),
            fmax=float(cfg.get("fmax", 250.0)),
            order=int(cfg.get("order", 4)),
            zero_phase=_optional_bool(cfg.get("zero_phase", True), default=True),
        )

    if m == "bandpass_fir":
        return bandpass_fir_denoise(
            data,
            fs=fs,
            fmin=float(cfg.get("fmin", 5.0)),
            fmax=float(cfg.get("fmax", 250.0)),
            numtaps=int(cfg.get("numtaps", 257)),
            window=str(cfg.get("window", "hann")),
            zero_phase=_optional_bool(cfg.get("zero_phase", True), default=True),
        )

    if m == "fk_fan":
        return fk_fan_denoise(
            data,
            fs=fs,
            dx=dx,
            vmin=_optional_float(cfg.get("vmin", None)),
            vmax=_optional_float(cfg.get("vmax", None)),
            fmin=_optional_float(cfg.get("fmin", cfg.get("fmin_abs", None))),
            fmax=_optional_float(cfg.get("fmax", cfg.get("fmax_abs", None))),
            taper_width=float(cfg.get("taper_width", 200.0)),
            kmin_abs=_optional_float(cfg.get("kmin_abs", None)),
            kmax_abs=_optional_float(cfg.get("kmax_abs", None)),
        )

    if m in ("directional_wedge", "directional"):
        return directional_wedge_denoise(
            data,
            n_angles=int(cfg.get("n_angles", 16)),
            threshold=float(cfg.get("threshold", 3.0)),
            wedge_sigma=float(cfg.get("wedge_sigma", 0.12)),
            soft_threshold=_optional_bool(cfg.get("soft_threshold", False), default=False),
        )

    if m in ("parabolic_radon", "radon"):
        return parabolic_radon_denoise(
            data,
            qmin=float(cfg.get("qmin", -0.002)),
            qmax=float(cfg.get("qmax", 0.002)),
            nq=int(cfg.get("nq", 81)),
            threshold_factor=float(cfg.get("threshold_factor", 3.0)),
            fs=fs,
            dx=dx,
        )

    if m in ("local_svd", "svd"):
        return local_svd_denoise(
            data,
            win_t=int(cfg.get("win_t", 64)),
            win_c=int(cfg.get("win_c", 32)),
            rank=int(cfg.get("rank", 3)),
            step_t=_optional_int(cfg.get("step_t", None)),
            step_c=_optional_int(cfg.get("step_c", None)),
        )

    if m in ("structure_tensor", "anisotropic_diffusion"):
        return structure_tensor_denoise(
            data,
            n_iter=int(cfg.get("n_iter", 20)),
            kappa=float(cfg.get("kappa", 20.0)),
            gamma=float(cfg.get("gamma", 0.1)),
        )

    raise ValueError(f"Unknown denoise method: {method}")
