# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""
filark.explore — research and exploratory analysis tools.

Submodules
----------
plot
    Pure-function diagnostic plots (FK, STFT, PSD, xcorr, τ-p, RMS, SVD).
    No Qt dependency; returns ``matplotlib.figure.Figure`` objects,
    suitable for Jupyter notebooks or scripts.
"""

from .plot import (
    plot_fk,
    plot_stft,
    plot_psd,
    plot_xcorr,
    plot_tp,
    plot_rms,
    plot_svd,
)

__all__ = [
    "plot_fk",
    "plot_stft",
    "plot_psd",
    "plot_xcorr",
    "plot_tp",
    "plot_rms",
    "plot_svd",
]
