# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""
Stateful DAS processing bridge.

This module exists for one narrow job: adapt raw DAS arrays or Tape-like
sources into the ``nt_nc`` layout expected by research transforms, apply a
callable transform chain, then return the requested layout.  It is not a
processor registry and it is not a pipeline executor; chunked execution is
delegated to ``filark.pipeline``.

Typical usage
-------------
>>> from filark.processing.das_processor import DASProcessor
>>> import filark.dsp.transforms as T
>>> proc = DASProcessor(
...     fs=500.0,
...     dx=1.0,
...     transforms=T.Compose([
...         T.DeMeanTime(),
...         T.IIRBandpassTime(fs=500, fmin=5, fmax=200),
...         T.AGCTime(half_window=50),
...     ]),
... )
>>> out = proc(tape)          # tape: Tape / ndarray / memmap
>>> out = proc.process_chunk(data_block, layout="nc_nt")
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from filark.dsp.base import to_nt_nc, from_nt_nc
from filark.pipeline.execution import run_nt_nc_transform_to_file

__all__ = ["DASProcessor"]


class DASProcessor:
    """
    Stateful DAS processing bridge.

    Stores sampling metadata and a transform chain. Given a raw array or a
    Tape-like source, it normalizes layout into ``nt_nc`` (time x channels),
    applies the transform chain, and converts the result back to the requested
    layout. This keeps algorithm code layout-clean while avoiding any GUI or
    file-format assumptions in the transforms themselves.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    dx : float
        Channel spacing in metres.
    transforms : callable, optional
        Any callable (e.g. ``dsp.transforms.Compose``) that maps
        ``np.ndarray [nt, nc] → np.ndarray [nt, nc]``.
        Pass ``None`` to return data as-is after layout conversion.
    layout_out : "nt_nc" | "nc_nt"
        Output layout (default matches input layout).
    """

    def __init__(
        self,
        *,
        fs: float,
        dx: float,
        transforms: Callable[[np.ndarray], np.ndarray] | None = None,
        layout_out: str = "nt_nc",
    ):
        if not (fs > 0):
            raise ValueError(f"fs must be positive, got {fs}")
        if not (dx > 0):
            raise ValueError(f"dx must be positive, got {dx}")

        self.fs = float(fs)
        self.dx = float(dx)
        self.transforms = transforms
        self.layout_out = str(layout_out)

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def _process_nt_nc(self, data_nt_nc: np.ndarray) -> np.ndarray:
        x = np.asarray(data_nt_nc, dtype=np.float32)
        if self.transforms is not None:
            x = np.asarray(self.transforms(x), dtype=np.float32)
        return x

    def process_chunk(
        self,
        data: np.ndarray,
        *,
        layout: str = "nt_nc",
    ) -> np.ndarray:
        """
        Process a single array block.

        Parameters
        ----------
        data : np.ndarray
            2-D array in *layout* format.
        layout : "nt_nc" | "nc_nt"
            Layout of *data*.

        Returns
        -------
        np.ndarray
            Processed array in ``layout_out`` format.
        """
        x = to_nt_nc(data, layout=layout, dtype=np.float32)
        x = self._process_nt_nc(x)
        return from_nt_nc(x, layout=self.layout_out)

    def __call__(self, source: Any, *, layout: str | None = None) -> np.ndarray:
        """
        Process a Tape-like source or plain array.

        If *source* has a ``dims`` attribute, that is used as the layout
        (unless *layout* is explicitly given).

        For large sources use :meth:`process_chunked` to avoid loading the
        entire dataset into memory.

        Parameters
        ----------
        source :
            Array-like with ``shape`` and optional ``dims`` attribute.
        layout : str, optional
            Override the detected layout.

        Returns
        -------
        np.ndarray
        """
        detected = str(getattr(source, "dims", "nt_nc") or "nt_nc").lower()
        if detected not in ("nt_nc", "nc_nt"):
            detected = "nt_nc"
        lay = layout if layout is not None else detected

        if isinstance(source, (np.ndarray, np.memmap)):
            data = np.asarray(source, dtype=np.float32)
        elif hasattr(source, "shape") and hasattr(source, "__getitem__"):
            data = np.asarray(source[:, :], dtype=np.float32)
        else:
            data = np.asarray(source, dtype=np.float32)
        return self.process_chunk(data, layout=lay)

    def process_chunked(
        self,
        source: Any,
        save_path: str,
        *,
        chunk_channels: int | None = None,
        target_chunk_mb: float = 128.0,
        overlap_channels: int = 0,
        progress_cb: Callable[[int, int], None] | None = None,
        cancel_cb: Callable[[], bool] | None = None,
    ) -> str:
        """
        Process *source* in channel chunks and save to a ``.npy`` file.

        Parameters
        ----------
        source :
            Tape-like source.
        save_path :
            Output file path (written as NumPy memory-map).
        chunk_channels, target_chunk_mb :
            Chunking controls.
        overlap_channels :
            Channel halo for channel-coupled transforms.
        progress_cb, cancel_cb :
            Optional progress / cancellation callbacks.

        Returns
        -------
        str
            Absolute path of the saved file.
        """
        result = run_nt_nc_transform_to_file(
            source,
            save_path,
            transform=self._process_nt_nc,
            method_key="das_processor",
            device="cpu",
            chunk_channels=chunk_channels,
            target_chunk_mb=target_chunk_mb,
            overlap_channels=overlap_channels,
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
        )
        return result.save_path

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    @property
    def dt(self) -> float:
        """Sampling interval in seconds (1 / fs)."""
        return 1.0 / self.fs

    def __repr__(self) -> str:
        return (
            f"DASProcessor(fs={self.fs}, dx={self.dx}, "
            f"transforms={self.transforms!r})"
        )
