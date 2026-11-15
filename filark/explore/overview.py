# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""Bounded full-read overview of a DAS time range (no Qt dependency).

The streaming viewer reads only what is visible and, when zoomed out along
time, decimates columns; that keeps its cost constant but makes a zoomed-out
view an aliased subsample.  An *overview* answers a different question
("where is the energy in this minute/file?") and therefore reads every sample
of a bounded range:

1. the range is read in time chunks of bounded size (all requested channels);
2. an optional causal IIR filter runs at the **original** sampling rate, with
   state carried across chunks and reset at every data-segment start (gaps);
3. every output pixel aggregates all samples it covers (signed mean, i.e. a
   bin-averaged amplitude image, RMS or max-abs),
   along time and, when there are more channels than rows, along channels.

Gap separators of a :class:`~filark.io.fileset.TapeFileSet` produce ``NaN``
columns so that missing data is never rendered as a quiet signal.

Example
-------
>>> from filark.io.fileset import TapeFileSet
>>> from filark.explore.overview import compute_overview
>>> tape = TapeFileSet("data/DS", suffixes=(".h5",))
>>> tb, cb = default_bins(tape.shape[0], tape.shape[1])   # ≈ 1920 × 1080 image
>>> ov = compute_overview(tape, 0, tape.shape[0], time_bin=tb, channel_bin=cb, band=(15, 32))
>>> ov.image.shape          # (rows, cols) — channels × time
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal

from filark.dsp.temporal import build_iir_sos, iir_filter_time_rt
from filark.utils.time import tape_sample_to_unix

__all__ = [
    "OverviewResult",
    "compute_overview",
    "default_bins",
    "estimate_overview_bytes",
    "file_range_at",
]

STATS = ("mean", "rms", "maxabs")


@dataclass
class OverviewResult:
    """Aggregated overview image and its coordinates.

    ``image[r, j]`` aggregates channels ``row_edges[r]:row_edges[r+1]`` and
    logical samples ``col_edges[j]:col_edges[j+1]``.  Columns without data
    (gap separators) are ``NaN``.
    """

    image: np.ndarray
    col_edges: np.ndarray
    row_edges: np.ndarray
    start: int
    stop: int
    fs: float
    stat: str
    band: Optional[Tuple[Optional[float], Optional[float]]]
    filter_kind: Optional[str]
    start_time: Optional[float]
    gaps: List[Tuple[int, int, float]] = field(default_factory=list)

    @property
    def c0(self) -> int:
        return int(self.row_edges[0])

    @property
    def c1(self) -> int:
        return int(self.row_edges[-1])


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _layout(tape) -> str:
    dims = str(getattr(tape, "dims", "nt_nc") or "nt_nc")
    return dims if dims in ("nt_nc", "nc_nt") else "nt_nc"


def _shape_nt_nc(tape) -> Tuple[int, int]:
    a, b = int(tape.shape[0]), int(tape.shape[1])
    return (a, b) if _layout(tape) == "nt_nc" else (b, a)


def _read_nt_nc(tape, t0: int, t1: int, c0: int, c1: int) -> np.ndarray:
    if _layout(tape) == "nt_nc":
        blk = tape[t0:t1, c0:c1]
    else:
        blk = np.asarray(tape[c0:c1, t0:t1]).T
    return np.asarray(blk)


def _segments_in(tape, start: int, stop: int) -> List[Tuple[int, int]]:
    segs = getattr(tape, "segments", None) or [(0, _shape_nt_nc(tape)[0])]
    out = []
    for a, b in segs:
        a, b = max(int(a), start), min(int(b), stop)
        if b > a:
            out.append((a, b))
    return out


def estimate_overview_bytes(tape, start: int, stop: int, c0: int = 0, c1: Optional[int] = None) -> int:
    """Bytes that :func:`compute_overview` will read for this range."""
    nt, nc = _shape_nt_nc(tape)
    c1 = nc if c1 is None else int(c1)
    n = sum(b - a for a, b in _segments_in(tape, int(start), int(stop)))
    itemsize = np.dtype(getattr(tape, "dtype", np.float32)).itemsize
    return int(n) * max(0, c1 - int(c0)) * int(itemsize)


def file_range_at(tape, sample: int) -> Tuple[int, int]:
    """Logical sample range of the file (or contiguous piece) containing ``sample``.

    For a plain tape this is the whole record.
    """
    nt, _ = _shape_nt_nc(tape)
    starts = getattr(tape, "_piece_start", None)
    files = getattr(tape, "_piece_file", None)
    if starts is None or files is None or len(starts) < 2:
        return 0, nt
    s = int(np.clip(sample, 0, max(0, nt - 1)))
    k = int(np.searchsorted(starts, s, side="right") - 1)
    k = int(np.clip(k, 0, len(files) - 1))
    if int(files[k]) < 0:  # inside a gap separator: take the following file
        k = min(k + 1, len(files) - 1)
    return int(starts[k]), int(starts[k + 1])


def _edges(start: int, stop: int, size: int) -> np.ndarray:
    """Bin edges of fixed ``size``; the last bin holds the remainder."""
    return np.append(np.arange(start, stop, max(1, int(size)), dtype=np.int64), np.int64(stop))


def default_bins(n_samples: int, n_channels: int, cols: int = 1920, rows: int = 1080) -> Tuple[int, int]:
    """Smallest ``(time_bin, channel_bin)`` giving at most ``cols × rows`` bins."""
    tb = max(1, -(-int(n_samples) // max(1, int(cols))))
    cb = max(1, -(-int(n_channels) // max(1, int(rows))))
    return tb, cb


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------


def compute_overview(
    tape,
    start: int,
    stop: int,
    *,
    time_bin: int = 1,
    channel_bin: int = 1,
    c0: int = 0,
    c1: Optional[int] = None,
    stat: str = "rms",
    band: Optional[Sequence[Optional[float]]] = None,
    filter_kind: Optional[str] = None,
    order: int = 4,
    target_chunk_mb: float = 128.0,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> OverviewResult:
    """Aggregate every sample of ``[start, stop)`` into ``channel_bin × time_bin`` bins.

    Parameters
    ----------
    tape :
        Any Tape (``dims`` ``"nt_nc"`` or ``"nc_nt"``); a ``TapeFileSet``'s
        ``segments`` are honoured.
    start, stop :
        Logical sample range along time.
    time_bin, channel_bin :
        Bin size in samples and in channels; ``1, 1`` keeps every sample (the
        last bin of each axis holds the remainder).  See :func:`default_bins`.
    c0, c1 :
        Channel range (default: all channels).
    stat :
        ``"mean"`` (signed bin average: the ordinary amplitude image, as
        low-passed by the bin size), ``"rms"`` (energy) or ``"maxabs"``
        (peak envelope).
    band :
        ``(fmin, fmax)`` in Hz for an optional causal Butterworth filter
        applied at the original sampling rate.  ``filter_kind`` defaults to
        ``"bandpass"``, ``"lowpass"`` (``fmin`` is None) or ``"highpass"``
        (``fmax`` is None).
    target_chunk_mb :
        Memory budget of one float64 time chunk.
    progress_cb, cancel_cb :
        ``progress_cb(done_samples, total_samples)``; ``cancel_cb()`` returning
        True raises ``RuntimeError``.
    """
    stat = str(stat).lower()
    if stat not in STATS:
        raise ValueError(f"stat must be one of {STATS}, got {stat!r}")

    nt, nc = _shape_nt_nc(tape)
    start = int(np.clip(start, 0, nt))
    stop = int(np.clip(stop, 0, nt))
    if stop <= start:
        raise ValueError(f"empty range [{start}, {stop})")
    c0 = int(np.clip(c0, 0, nc))
    c1 = nc if c1 is None else int(np.clip(c1, 0, nc))
    if c1 <= c0:
        raise ValueError(f"empty channel range [{c0}, {c1})")

    fs = float(getattr(tape, "fs", None) or 1.0)
    col_edges = _edges(start, stop, time_bin)
    row_edges = _edges(c0, c1, channel_bin)
    n_cols, n_rows = col_edges.size - 1, row_edges.size - 1
    row_starts = (row_edges[:-1] - c0).astype(np.int64)
    row_counts = np.diff(row_edges).astype(np.float64)

    sos = None
    kind = None
    if band is not None:
        fmin, fmax = (list(band) + [None, None])[:2]
        kind = filter_kind or ("bandpass" if fmin and fmax else ("lowpass" if fmax else "highpass"))
        sos = build_iir_sos(fs=fs, kind=kind, fmin=fmin, fmax=fmax, order=int(order))

    acc = np.zeros((n_rows, n_cols), dtype=np.float64)
    cnt = np.zeros(n_cols, dtype=np.float64)

    segments = _segments_in(tape, start, stop)
    total = sum(b - a for a, b in segments)
    done = 0
    chunk_t = max(1, int(target_chunk_mb * 1024 * 1024 // (8 * (c1 - c0))))

    for s0, s1 in segments:
        zi = None
        for t0 in range(s0, s1, chunk_t):
            if cancel_cb is not None and bool(cancel_cb()):
                raise RuntimeError("Overview cancelled.")
            t1 = min(s1, t0 + chunk_t)
            x = _read_nt_nc(tape, t0, t1, c0, c1).astype(np.float64, copy=False)

            if sos is not None:
                if zi is None:
                    # Steady-state start for the segment's first sample, so a DC
                    # offset does not produce a long start-up transient.
                    zi = signal.sosfilt_zi(sos)[:, :, None] * x[0][None, None, :]
                x, zi = iir_filter_time_rt(x, sos, zi=zi)

            # channel aggregation -> [t, rows]
            if stat == "mean":
                v = np.add.reduceat(x, row_starts, axis=1) / row_counts[None, :]
            elif stat == "rms":
                v = np.add.reduceat(x * x, row_starts, axis=1) / row_counts[None, :]
            else:
                v = np.maximum.reduceat(np.abs(x), row_starts, axis=1)

            # time aggregation into the columns this chunk touches
            cols = np.searchsorted(col_edges, np.arange(t0, t1), side="right") - 1
            cols = np.clip(cols, 0, n_cols - 1)
            brk = np.flatnonzero(np.diff(cols)) + 1
            run_starts = np.concatenate(([0], brk))
            run_cols = cols[run_starts]
            run_len = np.diff(np.concatenate((run_starts, [cols.size]))).astype(np.float64)
            if stat in ("mean", "rms"):
                acc[:, run_cols] += np.add.reduceat(v, run_starts, axis=0).T
            else:
                acc[:, run_cols] = np.maximum(acc[:, run_cols], np.maximum.reduceat(v, run_starts, axis=0).T)
            cnt[run_cols] += run_len

            done += t1 - t0
            if progress_cb is not None:
                progress_cb(done, total)

    with np.errstate(invalid="ignore", divide="ignore"):
        if stat == "mean":
            img = acc / cnt[None, :]
        elif stat == "rms":
            img = np.sqrt(acc / cnt[None, :])
        else:
            img = acc.copy()
    img[:, cnt == 0] = np.nan

    gaps = []
    for g in getattr(tape, "gaps", None) or []:
        if g.stop > g.start and g.stop > start and g.start < stop:
            gaps.append((int(g.start), int(g.stop), float(g.seconds)))

    st = tape_sample_to_unix(tape, start)
    return OverviewResult(
        image=img.astype(np.float32),
        col_edges=col_edges,
        row_edges=row_edges,
        start=start,
        stop=stop,
        fs=fs,
        stat=stat,
        band=None if band is None else tuple(band),  # type: ignore[arg-type]
        filter_kind=kind,
        start_time=None if st is None else float(st),
        gaps=gaps,
    )
