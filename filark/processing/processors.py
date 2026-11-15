# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""Processor schemas and built-in processor plugins.

This module is intentionally Qt-free.  It defines the declarative processing
surface that both GUI and non-GUI code can consume:

- parameter schemas shown to users
- validation/clamping rules
- runtime capability hints such as GPU support and overlap requirements
- the actual block-level processing call

What is *not* in this module:

- large-source chunk execution
- file saving/export policy
- Qt widget creation

Those concerns live in ``pipeline`` and ``gui`` respectively.
"""

from __future__ import annotations

import math

import numpy as np

from filark.dsp.temporal import build_iir_sos, iir_filter_time_rt
from filark.processing.denoise import apply_denoise

__all__ = [
    "ProcessorPlugin",
    "MovingAverageProcessor",
    "GaussianProcessor",
    "BandpassProcessor",
    "FKFanProcessor",
    "RealtimeIIRProcessor",
    "default_processors",
]


class ProcessorPlugin:
    """Base class for named DAS processing operations.

    A processor plugin is a domain-level operation that can be presented to a
    user as a selectable method.  It describes its own UI-facing schema and
    block-level numerical behavior, while leaving chunk execution and export to
    higher layers.
    """

    label: str = "Unknown"
    method_key: str = "unknown"
    param_spec: list[dict] = []
    supports_gpu: bool = False

    def default_params(self) -> dict:
        return {entry["key"]: entry["default"] for entry in self.param_spec}

    def validate_params(self, params: dict, *, fs: float, dx: float) -> dict:  # noqa: ARG002
        """Clamp / normalize params before use. Override in subclasses."""
        return dict(params)

    def run(
        self,
        data: np.ndarray,
        *,
        fs: float,
        dx: float,
        params: dict,
        device: str = "cpu",  # noqa: ARG002
    ) -> np.ndarray:
        raise NotImplementedError

    def overlap_channels(self, params: dict) -> int:  # noqa: ARG002
        """Channel halo required for channel-coupled operations."""
        return 0

    def overlap_time_samples(self, params: dict, fs: float) -> int:  # noqa: ARG002
        """Temporal halo required for time-coupled operations."""
        return 0


class MovingAverageProcessor(ProcessorPlugin):
    label = "Moving Average"
    method_key = "moving_average"
    param_spec = [
        {
            "key": "half_window",
            "label": "Window half (samples)",
            "type": "int",
            "min": 1,
            "max": 2000,
            "default": 8,
            "tooltip": (
                "Sliding window half-width in time samples.\n"
                "Total window = 2 x half_window + 1 samples."
            ),
        },
    ]

    def run(self, data, *, fs, dx, params, device="cpu"):
        return apply_denoise(data, method=self.method_key, fs=fs, dx=dx, params=params)

    def overlap_time_samples(self, params: dict, fs: float) -> int:
        # Sliding window extends half_window samples into neighbors; without a
        # halo, boundary samples are filled by reflect-padding from the chunk's
        # own edges rather than the real adjacent data.
        return int(params.get("half_window", 8))


class GaussianProcessor(ProcessorPlugin):
    label = "Gaussian Smooth"
    method_key = "gaussian"
    param_spec = [
        {
            "key": "half_window",
            "label": "Sigma half (samples)",
            "type": "int",
            "min": 1,
            "max": 2000,
            "default": 8,
            "tooltip": "Gaussian kernel half-width.\nsigma = half_window / 2.",
        },
    ]

    def run(self, data, *, fs, dx, params, device="cpu"):
        return apply_denoise(data, method=self.method_key, fs=fs, dx=dx, params=params)

    def overlap_time_samples(self, params: dict, fs: float) -> int:
        # Gaussian kernel has 99.7 % of its energy within 3 σ; sigma = hw/2,
        # so effective reach = ceil(3 * hw / 2).
        hw = int(params.get("half_window", 8))
        return math.ceil(1.5 * hw)


class BandpassProcessor(ProcessorPlugin):
    label = "Bandpass Filter"
    method_key = "bandpass"
    supports_gpu = True
    param_spec = [
        {
            "key": "fmin",
            "label": "Low cutoff (Hz)",
            "type": "float",
            "min": 0.0,
            "max": 1e6,
            "default": 5.0,
            "step": 1.0,
        },
        {
            "key": "fmax",
            "label": "High cutoff (Hz)",
            "type": "float",
            "min": 0.0,
            "max": 1e6,
            "default": 250.0,
            "step": 1.0,
        },
    ]

    def validate_params(self, params, *, fs, dx):
        nyq = 0.5 * float(fs)
        eps = max(1e-6, nyq * 1e-6)
        fmin = float(params.get("fmin", 5.0))
        fmax = float(params.get("fmax", 250.0))
        if fmax < fmin:
            fmin, fmax = fmax, fmin
        fmin = max(0.0, min(fmin, nyq - eps))
        fmax = min(max(fmax, fmin + eps), nyq)
        return {"fmin": fmin, "fmax": fmax}

    def run(self, data, *, fs, dx, params, device="cpu"):
        if device != "cpu":
            from filark.dsp.gpu.temporal import fft_bandpass_time_gpu

            return fft_bandpass_time_gpu(
                data,
                fs=float(fs),
                fmin=float(params["fmin"]),
                fmax=float(params["fmax"]),
                device=device,
            )
        return apply_denoise(data, method=self.method_key, fs=fs, dx=dx, params=params)

    def overlap_time_samples(self, params: dict, fs: float) -> int:
        # FFT-based rectangular bandpass: sinc ringing at chunk boundaries has
        # characteristic width ≈ fs / fmin (one period of the lowest frequency).
        # Cap at 0.5 s to avoid excessive padding on low-fmin settings.
        fmin = float(params.get("fmin", 5.0))
        return min(int(float(fs) * 0.5), int(float(fs) / max(1.0, fmin)))


class FKFanProcessor(ProcessorPlugin):
    label = "F-K Fan Filter"
    method_key = "fk_fan"
    supports_gpu = True
    param_spec = [
        {
            "key": "vmin",
            "label": "Vmin (m/s)",
            "type": "float",
            "min": 0.0,
            "max": 1e7,
            "default": 400.0,
            "step": 50.0,
            "tooltip": "Min apparent velocity to keep (m/s).",
        },
        {
            "key": "vmax",
            "label": "Vmax (m/s)",
            "type": "float",
            "min": 0.0,
            "max": 1e7,
            "default": 1e6,
            "step": 100.0,
            "tooltip": "Max apparent velocity to keep (m/s). Default 1e6 = no upper limit.",
        },
        {
            "key": "fmin",
            "label": "Fmin (Hz)",
            "type": "float",
            "min": 0.0,
            "max": 1e6,
            "default": 2.0,
            "step": 1.0,
            "tooltip": "Min frequency to keep (Hz).",
        },
        {
            "key": "fmax",
            "label": "Fmax (Hz)",
            "type": "float",
            "min": 0.0,
            "max": 1e6,
            "default": 300.0,
            "step": 1.0,
            "tooltip": "Max frequency to keep (Hz).",
        },
        {
            "key": "taper_width",
            "label": "Taper width",
            "type": "float",
            "min": 0.0,
            "max": 1e6,
            "default": 200.0,
            "step": 10.0,
            "tooltip": "Cosine taper width at velocity boundaries.",
        },
    ]

    def validate_params(self, params, *, fs, dx):
        vmin = max(0.0, float(params.get("vmin", 400.0)))
        vmax = max(vmin + 1e-6, float(params.get("vmax", 1e6)))
        fmin = max(0.0, float(params.get("fmin", 2.0)))
        fmax = max(fmin + 1e-6, float(params.get("fmax", 300.0)))
        tw = max(0.0, float(params.get("taper_width", 200.0)))
        return {"vmin": vmin, "vmax": vmax, "fmin": fmin, "fmax": fmax, "taper_width": tw}

    def run(self, data, *, fs, dx, params, device="cpu"):
        if device != "cpu":
            from filark.dsp.gpu.fk import fk_fan_filter_gpu

            return fk_fan_filter_gpu(
                data,
                dt=1.0 / float(fs),
                dx=float(dx),
                vmin=float(params.get("vmin", 400.0)),
                vmax=float(params.get("vmax", 1e6)),
                fmin=float(params.get("fmin", 2.0)),
                fmax=float(params.get("fmax", 300.0)),
                taper_width=float(params.get("taper_width", 200.0)),
                device=device,
            )
        return apply_denoise(data, method=self.method_key, fs=fs, dx=dx, params=params)

    def overlap_channels(self, params: dict) -> int:
        return 64

    def overlap_time_samples(self, params: dict, fs: float) -> int:
        return max(128, int(float(fs) * 0.25))


class RealtimeIIRProcessor(ProcessorPlugin):
    """Causal Butterworth IIR filter mirroring realtime streaming behavior."""

    label = "Realtime IIR (Causal)"
    method_key = "realtime_iir"
    supports_gpu = True
    param_spec = [
        {
            "key": "kind",
            "label": "Filter type",
            "type": "choice",
            "choices": ["bandpass", "lowpass", "highpass"],
            "default": "bandpass",
            "tooltip": (
                "bandpass: pass frequencies between fmin and fmax.\n"
                "lowpass: pass frequencies below fmax.\n"
                "highpass: pass frequencies above fmin."
            ),
        },
        {
            "key": "fmin",
            "label": "Low cutoff (Hz)",
            "type": "float",
            "min": 0.0,
            "max": 1e6,
            "default": 5.0,
            "step": 1.0,
            "tooltip": "Lower cutoff frequency (Hz). Used by bandpass and highpass.",
        },
        {
            "key": "fmax",
            "label": "High cutoff (Hz)",
            "type": "float",
            "min": 0.0,
            "max": 1e6,
            "default": 250.0,
            "step": 1.0,
            "tooltip": "Upper cutoff frequency (Hz). Used by bandpass and lowpass.",
        },
        {
            "key": "order",
            "label": "Order",
            "type": "int",
            "min": 1,
            "max": 20,
            "default": 4,
            "tooltip": "Butterworth filter order. Higher order = steeper roll-off.",
        },
    ]

    def validate_params(self, params: dict, *, fs: float, dx: float) -> dict:
        nyq = 0.5 * float(fs)
        eps = max(1e-6, nyq * 1e-6)
        kind = str(params.get("kind", "bandpass"))
        fmin = float(params.get("fmin", 5.0))
        fmax = float(params.get("fmax", 250.0))
        order = max(1, int(params.get("order", 4)))
        fmin = max(eps, min(fmin, nyq - eps))
        fmax = min(max(fmax, fmin + eps), nyq - eps)
        return {"kind": kind, "fmin": fmin, "fmax": fmax, "order": order}

    def run(self, data: np.ndarray, *, fs: float, dx: float, params: dict, device: str = "cpu") -> np.ndarray:
        kind = str(params.get("kind", "bandpass"))
        fmin = float(params["fmin"]) if kind != "lowpass" else None
        fmax = float(params["fmax"]) if kind != "highpass" else None
        order = int(params.get("order", 4))

        sos = build_iir_sos(
            fs=float(fs),
            kind=kind,
            fmin=fmin,
            fmax=fmax,
            order=order,
        )
        if device != "cpu":
            from filark.dsp.gpu.temporal import iir_filter_time_rt_gpu

            y, _ = iir_filter_time_rt_gpu(data, sos, device=device)
            return y
        y, _ = iir_filter_time_rt(data, sos)
        return y

    def overlap_time_samples(self, params: dict, fs: float) -> int:
        # Causal IIR with zero initial conditions: the filter needs ≈
        # 3 × order / (π × fc) seconds to reach steady state.  We use the
        # lower cutoff (fmin for bandpass/highpass, fmax for lowpass) as fc.
        order = int(params.get("order", 4))
        kind = str(params.get("kind", "bandpass"))
        if kind == "lowpass":
            fc = float(params.get("fmax", 250.0))
        else:
            fc = float(params.get("fmin", 5.0))
        return int(math.ceil(3.0 * order * float(fs) / (math.pi * max(1.0, fc))))


def default_processors() -> list[ProcessorPlugin]:
    """Return the built-in processor list used by the ROI process view."""

    return [
        MovingAverageProcessor(),
        GaussianProcessor(),
        BandpassProcessor(),
        FKFanProcessor(),
    ]
