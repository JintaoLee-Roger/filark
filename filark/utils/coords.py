# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""Spatial-axis utilities for DAS data.

All functions operate on plain Python/NumPy types; no Qt, no IO.
"""

from __future__ import annotations

from typing import Union

import numpy as np


def tape_channel_axis(tape) -> np.ndarray:
    """Return the physical position axis (in ``tape.dx_unit``) for channels.

    Channel *i* maps to ``i * dx``.
    """
    dims = getattr(tape, "dims", "nt_nc")
    nc = tape.shape[1] if dims == "nt_nc" else tape.shape[0]
    dx = float(tape.dx) if getattr(tape, "dx", None) else 1.0
    return np.arange(nc, dtype=np.float64) * dx


def sample_to_coord(
    n: Union[int, np.ndarray],
    *,
    dx: float,
    x0: float = 0.0,
) -> Union[float, np.ndarray]:
    """Convert channel index(es) to physical coordinate(s).

    Parameters
    ----------
    n : int or array-like
        Channel index(es), 0-based.
    dx : float
        Channel spacing.
    x0 : float
        Position of channel 0.
    """
    arr = np.asarray(n)
    result = float(x0) + arr * float(dx)
    return float(result) if arr.ndim == 0 else result


def coord_to_sample(
    x: float,
    *,
    dx: float,
    x0: float = 0.0,
    nc: int | None = None,
) -> int:
    """Convert a physical coordinate to the nearest channel index.

    Parameters
    ----------
    x : float
        Target position.
    dx : float
        Channel spacing.
    x0 : float
        Position of channel 0.
    nc : int, optional
        If given, clamp result to [0, nc - 1].
    """
    n = int(round((float(x) - float(x0)) / float(dx)))
    if nc is not None:
        n = max(0, min(n, int(nc) - 1))
    return n
