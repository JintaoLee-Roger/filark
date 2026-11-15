# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""Time-axis utilities for DAS data.

All functions operate on plain Python/NumPy types
"""

from __future__ import annotations

import datetime
from typing import Optional, Union

import numpy as np


def sample_to_time(
    n: Union[int, np.ndarray],
    *,
    fs: float,
    t0: float = 0.0,
) -> Union[float, np.ndarray]:
    """Convert sample index(es) to elapsed or absolute time (seconds).

    Parameters
    ----------
    n : int or array-like
        Sample index(es), 0-based.
    fs : float
        Sampling rate in Hz.
    t0 : float
        Time of sample 0 (Unix timestamp, or 0.0 for relative time).

    Returns
    -------
    float or ndarray
        Time in seconds (same type as input).
    """
    arr = np.asarray(n)
    result = float(t0) + arr / float(fs)
    return float(result) if arr.ndim == 0 else result


def time_to_sample(
    t: float,
    *,
    fs: float,
    t0: float = 0.0,
    nt: Optional[int] = None,
) -> int:
    """Convert a time value to the nearest sample index.

    Parameters
    ----------
    t : float
        Target time (same reference as ``t0``).
    fs : float
        Sampling rate in Hz.
    t0 : float
        Time of sample 0.
    nt : int, optional
        If given, clamp result to [0, nt - 1].

    Returns
    -------
    int
    """
    n = int(round((float(t) - float(t0)) * float(fs)))
    if nt is not None:
        n = max(0, min(n, int(nt) - 1))
    return n


def tape_sample_to_unix(tape, index):
    """Absolute time (Unix seconds) of sample ``index`` of ``tape``.

    Uses ``tape.sample_to_time`` when available (e.g. a ``TapeFileSet`` whose
    timeline contains gap separators); otherwise ``start_time + index / fs``.
    Returns ``None`` when the tape has no ``start_time``.
    """
    if tape is None or getattr(tape, "start_time", None) is None:
        return None
    fn = getattr(tape, "sample_to_time", None)
    if callable(fn):
        return fn(index)
    fs = float(getattr(tape, "fs", None) or 1.0) or 1.0
    idx = np.asarray(index, dtype=np.float64)
    out = float(tape.start_time) + idx / fs
    return out if out.ndim else float(out)


def tape_unix_to_sample(tape, t, nt: Optional[int] = None) -> Optional[int]:
    """Inverse of :func:`tape_sample_to_unix` for a scalar time (clamped to ``nt``)."""
    if tape is None or getattr(tape, "start_time", None) is None:
        return None
    fn = getattr(tape, "time_to_sample", None)
    if callable(fn):
        n = int(fn(float(t)))
    else:
        fs = float(getattr(tape, "fs", None) or 1.0) or 1.0
        n = int(round((float(t) - float(tape.start_time)) * fs))
    if nt is not None:
        n = max(0, min(n, int(nt) - 1))
    return n


def tape_time_axis(tape) -> np.ndarray:
    """Return the time axis (in seconds) for a Tape.

    If ``tape.start_time`` is set, returns absolute Unix timestamps (gap-aware
    for sources that provide ``sample_to_time``).  Otherwise returns elapsed
    seconds from 0.
    """
    dims = getattr(tape, "dims", "nt_nc")
    nt = tape.shape[0] if dims == "nt_nc" else tape.shape[1]
    idx = np.arange(nt, dtype=np.float64)
    abs_t = tape_sample_to_unix(tape, idx)
    if abs_t is not None:
        return np.asarray(abs_t, dtype=np.float64)
    fs = float(tape.fs) if getattr(tape, "fs", None) else 1.0
    return idx / fs


def format_timestamp(
    t: float,
    *,
    fmt: str = "%Y-%m-%d %H:%M:%S.%f",
    utc: bool = True,
) -> str:
    """Format a Unix timestamp as a human-readable string.

    Parameters
    ----------
    t : float
        Unix timestamp (seconds since epoch).
    fmt : str
        ``strftime`` format string.
    utc : bool
        If True, use UTC; otherwise use local time.
    """
    tz = datetime.timezone.utc if utc else None
    dt = datetime.datetime.fromtimestamp(float(t), tz=tz)
    return dt.strftime(fmt)


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a compact human-readable string.

    Examples
    --------
    >>> format_duration(3661.5)
    '1h 01m 01.500s'
    >>> format_duration(72.3)
    '1m 12.300s'
    >>> format_duration(0.042)
    '0.042s'
    """
    s = abs(float(seconds))
    sign = "-" if seconds < 0 else ""
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    if h > 0:
        return f"{sign}{h}h {m:02d}m {sec:06.3f}s"
    if m > 0:
        return f"{sign}{m}m {sec:06.3f}s"
    return f"{sign}{sec:.3f}s"


def parse_timestamp(s: str) -> float:
    """Parse a datetime string to a Unix timestamp (UTC assumed if no tz).

    Accepts ISO-8601 strings like ``"2025-07-07T11:59:18"`` or
    ``"2025-07-07 11:59:18.123456"``.
    """
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            dt = datetime.datetime.strptime(s.strip(), fmt)
            dt = dt.replace(tzinfo=datetime.timezone.utc)
            return dt.timestamp()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: {s!r}")
