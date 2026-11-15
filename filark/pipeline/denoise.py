# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""Built-in denoising pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np

from filark.processing.denoise import apply_denoise

from .base import BasePipeline

__all__ = ["DenoisePipeline"]


def _channel_overlap_for_method(method: str, cfg: dict, default_ov: int) -> int:
    """Heuristic: channel overlap needed for each method."""
    m = str(method or "").strip().lower()
    custom = cfg.get("channel_overlap", None)
    if custom not in (None, ""):
        return max(0, int(custom))
    if m in {"fk_fan", "directional_wedge", "directional", "parabolic_radon", "radon"}:
        return max(0, default_ov)
    if m in {"local_svd", "svd"}:
        wc = max(1, int(cfg.get("win_c", 32)))
        return max(default_ov, wc // 2)
    if m in {"structure_tensor", "anisotropic_diffusion"}:
        nit = max(1, int(cfg.get("n_iter", 20)))
        return max(default_ov, int(np.ceil(np.sqrt(nit)) * 2))
    return 0  # temporal-only methods need no channel overlap


class DenoisePipeline(BasePipeline):
    """Pipeline for DAS denoising.

    Parameters
    ----------
    method : str
        Denoise method key (see ``processing.denoise.DENOISE_METHOD_LABELS``).
    fs : float
        Sampling frequency (Hz).
    dx : float
        Channel spacing (m).
    params : dict, optional
        Method-specific parameters.  Defaults from ``DEFAULT_DENOISE_PARAMS``
        are used for any key not supplied.
    fk_channel_overlap : int
        Default channel overlap for channel-coupled methods (f-k, SVD, etc.).

    Examples
    --------
    >>> pipe = DenoisePipeline(method="bandpass_iir", fs=500, fmin=5, fmax=200)
    >>> out = pipe(data)                                    # in-memory
    >>> pipe(source, save_path="out.npy", chunked=True)    # large data
    """

    def __init__(
        self,
        *,
        method: str,
        fs: float,
        dx: float = 1.0,
        params: dict[str, Any] | None = None,
        fk_channel_overlap: int = 64,
    ):
        super().__init__()
        self.method = str(method).strip().lower()
        self.fs = float(fs)
        self.dx = float(dx)
        self.params = dict(params or {})
        self.fk_channel_overlap = int(fk_channel_overlap)

    def _run(self, data: np.ndarray) -> np.ndarray:
        return apply_denoise(
            data, method=self.method, fs=self.fs, dx=self.dx, params=self.params,
        )

    def _run_stream(self, source, save_path=None, *, overlap_channels=None, **kwargs):
        """Override to inject method-specific channel overlap."""
        if overlap_channels is None:
            overlap_channels = _channel_overlap_for_method(
                self.method, self.params, self.fk_channel_overlap,
            )
        return super()._run_stream(
            source, save_path, overlap_channels=overlap_channels, **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"DenoisePipeline(method={self.method!r}, fs={self.fs}, dx={self.dx})"
        )
