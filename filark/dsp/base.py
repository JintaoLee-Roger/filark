# Copyright (c) 2026 Jintao Li. 
# Zhejiang University (ZJU).
# 
# Licensed under the MIT License.

"""Layout normalization and small shared helpers used across DSP modules."""

from __future__ import annotations

from typing import Literal

import numpy as np

Layout = Literal["nt_nc", "nc_nt"]

__all__ = [
    "Layout",
    "as_float2d",
    "require_positive",
    "require_odd",
    "to_nt_nc",
    "from_nt_nc",
]


def as_float2d(data, *, dtype=np.float32) -> np.ndarray:
    x = np.asarray(data, dtype=dtype)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got ndim={x.ndim}")
    return x


def require_positive(name: str, value: float | int) -> float:
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value}")
    return v


def require_odd(name: str, value: int) -> int:
    v = int(value)
    if v < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    if v % 2 == 0:
        v += 1
    return v


def to_nt_nc(data, *, layout: Layout = "nt_nc", dtype=np.float32) -> np.ndarray:
    x = as_float2d(data, dtype=dtype)
    if layout == "nt_nc":
        return x
    if layout == "nc_nt":
        return x.T
    raise ValueError(f"Unsupported layout: {layout}")


def from_nt_nc(data_nt_nc: np.ndarray, *, layout: Layout = "nt_nc", dtype=np.float32) -> np.ndarray:
    x = as_float2d(data_nt_nc, dtype=dtype)
    if layout == "nt_nc":
        return x
    if layout == "nc_nt":
        return x.T
    raise ValueError(f"Unsupported layout: {layout}")
