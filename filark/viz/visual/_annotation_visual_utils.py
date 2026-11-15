# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""Shared helpers for annotation-specific visuals."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


def coerce_points(points: Iterable[tuple[float, float]] | np.ndarray | None) -> np.ndarray:
    """Normalize point-like inputs to a ``(N, 2)`` float32 array."""
    if points is None:
        return np.zeros((0, 2), dtype=np.float32)
    arr = np.array(list(points), dtype=np.float32).reshape(-1, 2)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return arr


def normalize_bbox(bbox: dict | None) -> Optional[dict[str, float]]:
    """Return a normalized bbox with increasing ``t`` and ``c`` bounds."""
    if bbox is None:
        return None
    t0 = float(bbox.get("t0", 0.0))
    c0 = float(bbox.get("c0", 0.0))
    t1 = float(bbox.get("t1", 0.0))
    c1 = float(bbox.get("c1", 0.0))
    if t1 < t0:
        t0, t1 = t1, t0
    if c1 < c0:
        c0, c1 = c1, c0
    return {"t0": t0, "c0": c0, "t1": t1, "c1": c1}


def bbox_handle_points(bbox: dict[str, float]) -> np.ndarray:
    """Return the eight bbox edit handles used by the viewer."""
    t0, c0, t1, c1 = bbox["t0"], bbox["c0"], bbox["t1"], bbox["c1"]
    tc = 0.5 * (t0 + t1)
    cc = 0.5 * (c0 + c1)
    return np.array(
        [
            (t0, c0),
            (tc, c0),
            (t1, c0),
            (t0, cc),
            (t1, cc),
            (t0, c1),
            (tc, c1),
            (t1, c1),
        ],
        dtype=np.float32,
    )
