# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

r"""Deterministic measurement conversions for DAS-style data.

This module intentionally covers only conversions whose result is uniquely
determined once the input quantity and calibration parameters are known.
Typical examples are:

- temporal derivative / integral
- strain <-> strain rate
- strain <-> microstrain
- extension <-> strain
- optical phase <-> extension / strain

What is intentionally *not* included here:

- "equivalent geophone" or "equivalent hydrophone" mappings
- coupling- or site-dependent conversions
- model-based pressure / velocity inference

Those belong in ``filark.algorithms`` because they depend on additional
assumptions rather than deterministic bookkeeping.
"""

from __future__ import annotations

from typing import Final

import numpy as np

__all__ = [
    "canonical_measurement_name",
    "time_derivative",
    "time_integral",
    "strain_to_strain_rate",
    "strain_rate_to_strain",
    "strain_to_microstrain",
    "microstrain_to_strain",
    "strain_rate_to_microstrain_rate",
    "microstrain_rate_to_strain_rate",
    "extension_to_strain",
    "strain_to_extension",
    "extension_rate_to_strain_rate",
    "strain_rate_to_extension_rate",
    "phase_to_extension",
    "extension_to_phase",
    "phase_to_strain",
    "strain_to_phase",
]


_MEASUREMENT_ALIASES: Final[dict[str, str]] = {
    "strain": "strain",
    "epsilon": "strain",
    "strain_rate": "strain_rate",
    "strainrate": "strain_rate",
    "strain-rate": "strain_rate",
    "dstrain/dt": "strain_rate",
    "extension": "extension",
    "elongation": "extension",
    "extension_rate": "extension_rate",
    "extensionrate": "extension_rate",
    "phase": "phase",
    "optical_phase": "phase",
    "phase_shift": "phase",
}

_STRAIN_TO_MICROSTRAIN: Final[float] = 1e6


def _as_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _scalar_or_array(reference, values: np.ndarray):
    ref = np.asarray(reference)
    if ref.ndim == 0:
        return float(np.asarray(values))
    return values


def _require_positive(name: str, value: float) -> float:
    value = float(value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return value


def canonical_measurement_name(name: str) -> str:
    """Normalize common measurement aliases to a canonical quantity name."""
    key = (name or "").strip().lower()
    if not key:
        raise ValueError("measurement name cannot be empty")
    return _MEASUREMENT_ALIASES.get(key, key)


def time_derivative(values, *, fs: float, axis: int = -1, edge_order: int = 1):
    r"""Differentiate a time series along one axis.

    The derivative is computed with:

    .. math::

        \frac{dx}{dt} \approx \nabla_t x

    using ``numpy.gradient`` with sample spacing

    .. math::

        dt = \frac{1}{f_s}

    Parameters
    ----------
    values
        Scalar or array-like data.
    fs : float
        Sampling rate in Hz.
    axis : int
        Time axis.
    edge_order : int
        Edge order passed to ``numpy.gradient``.
    """
    fs = _require_positive("fs", fs)
    arr = _as_float_array(values)
    out = np.gradient(arr, 1.0 / fs, axis=axis, edge_order=int(edge_order))
    return _scalar_or_array(values, out)


def time_integral(values, *, fs: float, axis: int = -1, initial: float = 0.0):
    r"""Integrate a time series along one axis with a left Riemann sum.

    The discrete integral is:

    .. math::

        x[n] = x_0 + \sum_{k=0}^{n} y[k] \, dt

    where

    .. math::

        dt = \frac{1}{f_s}

    Parameters
    ----------
    values
        Scalar or array-like data.
    fs : float
        Sampling rate in Hz.
    axis : int
        Time axis.
    initial : float
        Integration constant at the first sample.
    """
    fs = _require_positive("fs", fs)
    arr = _as_float_array(values)
    out = np.cumsum(arr, axis=axis, dtype=np.float64) / fs
    if initial != 0.0:
        out = out + float(initial)
    return _scalar_or_array(values, out)


def strain_to_strain_rate(strain, *, fs: float, axis: int = -1, edge_order: int = 1):
    r"""Convert strain to strain rate by temporal differentiation.

    .. math::

        \dot{\varepsilon}(t) = \frac{d\varepsilon(t)}{dt}
    """
    return time_derivative(strain, fs=fs, axis=axis, edge_order=edge_order)


def strain_rate_to_strain(strain_rate, *, fs: float, axis: int = -1, initial: float = 0.0):
    r"""Convert strain rate to strain by temporal integration.

    .. math::

        \varepsilon(t) = \varepsilon_0 + \int \dot{\varepsilon}(t)\,dt

    Notes
    -----
    The integration constant is not observable from strain-rate data alone.
    ``initial`` defaults to 0, which is often suitable for display or relative
    analysis but should not be interpreted as an absolute reference level.
    """
    return time_integral(strain_rate, fs=fs, axis=axis, initial=initial)


def strain_to_microstrain(strain):
    r"""Convert dimensionless strain to microstrain.

    .. math::

        \mu\varepsilon = 10^{6} \varepsilon
    """
    arr = _as_float_array(strain) * _STRAIN_TO_MICROSTRAIN
    return _scalar_or_array(strain, arr)


def microstrain_to_strain(microstrain):
    r"""Convert microstrain to dimensionless strain.

    .. math::

        \varepsilon = 10^{-6} \mu\varepsilon
    """
    arr = _as_float_array(microstrain) / _STRAIN_TO_MICROSTRAIN
    return _scalar_or_array(microstrain, arr)


def strain_rate_to_microstrain_rate(strain_rate):
    r"""Convert strain rate to microstrain per second.

    .. math::

        \mu\varepsilon/s = 10^{6} \dot{\varepsilon}
    """
    arr = _as_float_array(strain_rate) * _STRAIN_TO_MICROSTRAIN
    return _scalar_or_array(strain_rate, arr)


def microstrain_rate_to_strain_rate(microstrain_rate):
    r"""Convert microstrain per second to strain rate.

    .. math::

        \dot{\varepsilon} = 10^{-6} (\mu\varepsilon/s)
    """
    arr = _as_float_array(microstrain_rate) / _STRAIN_TO_MICROSTRAIN
    return _scalar_or_array(microstrain_rate, arr)


def extension_to_strain(extension, *, gauge_length: float):
    r"""Convert gauge extension to strain.

    .. math::

        \varepsilon = \frac{\Delta L}{L_g}

    Parameters
    ----------
    extension
        Gauge extension in metres.
    gauge_length : float
        Gauge length :math:`L_g` in metres.
    """
    gauge_length = _require_positive("gauge_length", gauge_length)
    arr = _as_float_array(extension) / gauge_length
    return _scalar_or_array(extension, arr)


def strain_to_extension(strain, *, gauge_length: float):
    r"""Convert strain to gauge extension.

    .. math::

        \Delta L = \varepsilon L_g
    """
    gauge_length = _require_positive("gauge_length", gauge_length)
    arr = _as_float_array(strain) * gauge_length
    return _scalar_or_array(strain, arr)


def extension_rate_to_strain_rate(extension_rate, *, gauge_length: float):
    r"""Convert extension rate to strain rate.

    .. math::

        \dot{\varepsilon} = \frac{1}{L_g}\frac{d(\Delta L)}{dt}
    """
    gauge_length = _require_positive("gauge_length", gauge_length)
    arr = _as_float_array(extension_rate) / gauge_length
    return _scalar_or_array(extension_rate, arr)


def strain_rate_to_extension_rate(strain_rate, *, gauge_length: float):
    r"""Convert strain rate to extension rate.

    .. math::

        \frac{d(\Delta L)}{dt} = \dot{\varepsilon} L_g
    """
    gauge_length = _require_positive("gauge_length", gauge_length)
    arr = _as_float_array(strain_rate) * gauge_length
    return _scalar_or_array(strain_rate, arr)


def phase_to_extension(
    phase,
    *,
    wavelength: float,
    refractive_index: float = 1.4682,
    photoelastic_factor: float = 0.78,
):
    r"""Convert gauge-integrated optical phase change to extension.

    The common round-trip DAS phase model is:

    .. math::

        \Delta \phi = \frac{4 \pi n_{\mathrm{eff}} \xi}{\lambda} \Delta L

    so

    .. math::

        \Delta L = \frac{\lambda \Delta \phi}{4 \pi n_{\mathrm{eff}} \xi}

    Parameters
    ----------
    phase
        Optical phase change in radians.
    wavelength : float
        Optical wavelength in metres.
    refractive_index : float
        Effective refractive index :math:`n_{\mathrm{eff}}`.
    photoelastic_factor : float
        Effective strain-to-phase scaling factor :math:`\xi`.

    Notes
    -----
    This function assumes that the provided phase already represents the
    gauge-integrated phase response used by the DAS interrogator.
    """
    wavelength = _require_positive("wavelength", wavelength)
    refractive_index = _require_positive("refractive_index", refractive_index)
    photoelastic_factor = _require_positive("photoelastic_factor", photoelastic_factor)
    scale = wavelength / (4.0 * np.pi * refractive_index * photoelastic_factor)
    arr = _as_float_array(phase) * scale
    return _scalar_or_array(phase, arr)


def extension_to_phase(
    extension,
    *,
    wavelength: float,
    refractive_index: float = 1.4682,
    photoelastic_factor: float = 0.78,
):
    r"""Convert gauge extension to optical phase change.

    .. math::

        \Delta \phi = \frac{4 \pi n_{\mathrm{eff}} \xi}{\lambda} \Delta L
    """
    wavelength = _require_positive("wavelength", wavelength)
    refractive_index = _require_positive("refractive_index", refractive_index)
    photoelastic_factor = _require_positive("photoelastic_factor", photoelastic_factor)
    scale = 4.0 * np.pi * refractive_index * photoelastic_factor / wavelength
    arr = _as_float_array(extension) * scale
    return _scalar_or_array(extension, arr)


def phase_to_strain(
    phase,
    *,
    wavelength: float,
    gauge_length: float,
    refractive_index: float = 1.4682,
    photoelastic_factor: float = 0.78,
):
    r"""Convert optical phase change to strain.

    Combining

    .. math::

        \Delta \phi = \frac{4 \pi n_{\mathrm{eff}} \xi}{\lambda} \Delta L

    with

    .. math::

        \varepsilon = \frac{\Delta L}{L_g}

    gives

    .. math::

        \varepsilon = \frac{\lambda \Delta \phi}
        {4 \pi n_{\mathrm{eff}} \xi L_g}
    """
    extension = phase_to_extension(
        phase,
        wavelength=wavelength,
        refractive_index=refractive_index,
        photoelastic_factor=photoelastic_factor,
    )
    return extension_to_strain(extension, gauge_length=gauge_length)


def strain_to_phase(
    strain,
    *,
    wavelength: float,
    gauge_length: float,
    refractive_index: float = 1.4682,
    photoelastic_factor: float = 0.78,
):
    r"""Convert strain to optical phase change.

    .. math::

        \Delta \phi =
        \frac{4 \pi n_{\mathrm{eff}} \xi L_g}{\lambda}\varepsilon
    """
    extension = strain_to_extension(strain, gauge_length=gauge_length)
    return extension_to_phase(
        extension,
        wavelength=wavelength,
        refractive_index=refractive_index,
        photoelastic_factor=photoelastic_factor,
    )
