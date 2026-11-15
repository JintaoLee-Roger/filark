# Copyright (c) 2026 Jintao Li. 
# Zhejiang University (ZJU).
# 
# Licensed under the MIT License.

"""Virtual tape that presents many files as one continuous time axis.

Timestamp gaps
--------------
When every file carries a ``start_time``, consecutive files are checked for
temporal continuity.  The expected start of file ``i+1`` is
``start_time[i] + nt[i] / fs``; deviations within ``gap_tolerance`` seconds
(timestamp jitter) are treated as contiguous.  Larger deviations are handled
according to ``gap_policy``:

``"fill"`` (default)
    Files are ordered by ``start_time``.  A positive deviation (missing data)
    is represented by a short *separator* of ``gap_samples`` fill samples
    (default 100, or the true gap length if shorter) instead of the full
    missing duration, so long outages do not inflate the logical timeline or
    pipeline outputs.  Reads inside a separator return ``fill_value``.  A
    negative deviation (overlap) is resolved by dropping the duplicated leading
    samples of the later file.  Because the index no longer maps linearly to
    absolute time across gaps, use :meth:`TapeFileSet.sample_to_time` and
    :meth:`TapeFileSet.time_to_sample` for conversions.
``"raise"``
    Any gap or overlap raises :class:`ValueError`.
``"ignore"``
    Legacy behaviour: files are concatenated in the given order and
    timestamps are not inspected.

The detected discontinuities are exposed through :attr:`TapeFileSet.gaps`
and :attr:`TapeFileSet.segments` (logical sample ranges) so that stateful
consumers can reset their state at segment boundaries.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .tapeio import H5Tape, NpyTape, BinTape
from .protocols import Tape
from filark._exceptions import print_exception

GAP_POLICIES = ("fill", "raise", "ignore")


@dataclass(frozen=True)
class TimeGap:
    """A discontinuity between two consecutive files.

    ``start``/``stop`` are logical sample indices of the separator inserted in
    the stitched time axis (``stop - start`` samples, zero for an overlap).
    ``seconds`` is the true signed deviation from the expected start time
    (the missing duration for a gap, negative for an overlap).
    """

    start: int
    stop: int
    seconds: float
    before_file: int
    after_file: int
    kind: str  # "gap" | "overlap"

    @property
    def n_samples(self) -> int:
        return int(self.stop - self.start)


@dataclass
class BinFolderOpts:
    nt: int
    nc: int
    dtype: np.dtype
    dims: str              # "nt_nc" / "nc_nt"
    fs: float = 1.0
    dx: float = 1.0
    dx_unit: str = "m"
    gauge_length: Optional[float] = None
    order: str = "C"


class TapeFileSet(Tape):
    def __init__(
        self,
        file_paths: str | Sequence[str | Path] | Sequence[Tape],
        *,
        # npy NEED dims
        npy_dims: Optional[str] = None,
        npy_fs: float = 1.0,
        npy_dx: float = 1.0,
        npy_dx_unit: str = "m",
        npy_gauge_length: Optional[float] = None,
        # bin NEED completed params
        bin_opts: Optional[BinFolderOpts] = None,
        h5_kwargs: Optional[dict] = None,
        suffixes: Sequence[str] = (".h5", ".hdf5", ".npy", ".dat", ".bin"),
        # timestamp continuity
        gap_policy: str = "fill",
        gap_tolerance: float = 0.01,
        gap_samples: int = 100,
        fill_value: float = 0.0,
        # open-handle budget (files whose tapes support ``release()``)
        max_open_files: int = 6,
    ):
        gap_policy = str(gap_policy).lower()
        if gap_policy not in GAP_POLICIES:
            raise ValueError(f"gap_policy must be one of {GAP_POLICIES}, got {gap_policy!r}")
        if not np.isfinite(gap_tolerance) or gap_tolerance < 0:
            raise ValueError("gap_tolerance must be a non-negative number of seconds")
        self._gap_policy = gap_policy
        self._gap_tolerance = float(gap_tolerance)
        if int(gap_samples) < 1:
            raise ValueError("gap_samples must be >= 1")
        self._gap_samples = int(gap_samples)
        self._fill_value = fill_value
        self._h5_kwargs = h5_kwargs or {}
        self._files: List[Tape] = []
        if int(max_open_files) < 1:
            raise ValueError("max_open_files must be >= 1")
        self._max_open_files = int(max_open_files)
        # LRU of child indices whose handles are open; guarded by _io_lock so a
        # handle is never released while another thread is reading from it.
        self._open_lru: "OrderedDict[int, None]" = OrderedDict()
        self._io_lock = threading.RLock()

        # Support direct concatenation from already-opened Tape objects.
        if self._is_tape_sequence(file_paths):
            self._files = list(file_paths)  # type: ignore[arg-type]
            self._validate_and_init()
            return

        paths = self._normalize_paths(file_paths, suffixes=suffixes)
        if not paths:
            raise FileNotFoundError(f"No files found for {file_paths}")

        for p in paths:
            suf = p.suffix.lower()
            if suf in (".h5", ".hdf5"):
                tape = H5Tape(str(p), **self._h5_kwargs)
                tape.release()  # metadata is cached; keep the fd count bounded
                self._files.append(tape)
            elif suf == ".npy":
                if npy_dims not in ("nt_nc", "nc_nt"):
                    raise ValueError("npy_dims must be provided for NPY folders (nt_nc / nc_nt).")
                self._files.append(NpyTape(
                    str(p),
                    dims=npy_dims,
                    fs=npy_fs,
                    dx=npy_dx,
                    dx_unit=npy_dx_unit,
                    gauge_length=npy_gauge_length,
                ))
            elif suf in (".dat", ".bin"):
                if bin_opts is None:
                    raise ValueError("bin_opts must be provided for BIN/DAT folders.")
                self._files.append(BinTape(
                    path=str(p),
                    nt=bin_opts.nt,
                    nc=bin_opts.nc,
                    dtype=bin_opts.dtype,
                    dims=bin_opts.dims,
                    fs=bin_opts.fs,
                    dx=bin_opts.dx,
                    dx_unit=bin_opts.dx_unit,
                    gauge_length=bin_opts.gauge_length,
                    order=bin_opts.order,
                ))
            else:
                raise ValueError(f"Unsupported suffix: {p}")

        self._validate_and_init()

    @staticmethod
    def _is_tape_sequence(v: object) -> bool:
        if isinstance(v, (str, Path)):
            return False
        if not isinstance(v, Sequence):
            return False
        return all(isinstance(x, Tape) for x in v)

    # --------------------
    # path helpers
    # --------------------
    def _normalize_paths(self, file_paths: str | Sequence[str | Path], *, suffixes: Sequence[str]) -> List[Path]:
        if isinstance(file_paths, (str, Path)):
            root = Path(file_paths)
            if root.is_dir():
                out: List[Path] = []
                for suf in suffixes:
                    out.extend(sorted(root.glob(f"*{suf}")))
                out = [p for p in out if not p.name.startswith(".")]
                return out
            else:
                return [root]
        else:
            return [Path(p) for p in file_paths]

    # --------------------
    # init + validation
    # --------------------
    def _validate_and_init(self):
        if not self._files:
            raise ValueError("Empty fileset.")

        dims0 = getattr(self._files[0], "dims", None)
        if dims0 not in ("nt_nc", "nc_nt"):
            raise ValueError(f"Invalid dims for first tape: {dims0}")

        def semantic_nc(t: Tape) -> int:
            return t.shape[1] if t.dims == "nt_nc" else t.shape[0]

        def semantic_nt(t: Tape) -> int:
            return t.shape[0] if t.dims == "nt_nc" else t.shape[1]

        nc0 = semantic_nc(self._files[0])
        dt0 = self._files[0].dtype
        self._dims = dims0
        self._dtype = dt0
        self._ndim = 2

        nts = []
        for i, t in enumerate(self._files):
            if t.ndim != 2:
                raise ValueError(f"Only 2D tapes are supported, file index {i} has ndim={t.ndim}")
            if t.dims != self._dims:
                raise ValueError(f"dims mismatch at index {i}: {t.dims} vs {self._dims}")
            if semantic_nc(t) != nc0:
                raise ValueError(f"nc mismatch at index {i}: {semantic_nc(t)} vs {nc0}")
            if t.dtype != dt0:
                raise ValueError(f"dtype mismatch at index {i}: {t.dtype} vs {dt0}")
            nts.append(semantic_nt(t))

        fs0 = self._child_fs(self._files[0])
        for i, t in enumerate(self._files[1:], start=1):
            fsi = self._child_fs(t)
            if fs0 is not None and fsi is not None and not np.isclose(fsi, fs0, rtol=1e-6, atol=0.0):
                raise ValueError(f"fs mismatch at index {i}: {fsi} vs {fs0}")

        self._nts = nts
        self._nc = int(nc0)
        self._build_layout()
        self._release_all_handles()

    # --------------------
    # open-handle budget
    # --------------------
    @property
    def max_open_files(self) -> int:
        return self._max_open_files

    def _release_all_handles(self) -> None:
        """Metadata is cached by the tapes; reads reopen on demand."""
        with self._io_lock:
            for t in self._files:
                release = getattr(t, "release", None)
                if callable(release):
                    release()
            self._open_lru.clear()

    def _touch(self, fidx: int) -> None:
        """Mark child ``fidx`` as in use; release the least recently used beyond budget."""
        lru = self._open_lru
        if fidx in lru:
            lru.move_to_end(fidx)
            return
        lru[fidx] = None
        while len(lru) > self._max_open_files:
            old, _ = lru.popitem(last=False)
            release = getattr(self._files[old], "release", None)
            if callable(release):
                release()

    # --------------------
    # timeline layout
    # --------------------
    @staticmethod
    def _child_fs(t: Tape) -> Optional[float]:
        fs = getattr(t, "fs", None)
        if fs is None:
            return None
        fs = float(fs)
        return fs if np.isfinite(fs) and fs > 0 else None

    def _timestamps_usable(self) -> bool:
        if self._gap_policy == "ignore" or len(self._files) < 2:
            return False
        if self._child_fs(self._files[0]) is None:
            return False
        return all(getattr(t, "start_time", None) is not None for t in self._files)

    def _build_layout(self) -> None:
        """Build the piece table that maps logical samples to files or gaps.

        Each piece covers ``[piece_start[k], piece_start[k+1])`` logical
        samples and is either ``(file index, offset into that file)`` or a
        virtual gap (file index ``-1``).
        """
        order = list(range(len(self._files)))
        gaps: List[TimeGap] = []
        pieces: List[Tuple[int, int, int]] = []  # (file_idx, file_offset, length)
        piece_t0: List[float] = []  # absolute time of each piece's first sample (NaN = unknown)

        if self._timestamps_usable():
            fs = float(self._child_fs(self._files[0]))  # type: ignore[arg-type]
            starts = [float(getattr(t, "start_time")) for t in self._files]
            order = sorted(order, key=lambda i: starts[i])
            tol = max(self._gap_tolerance, 0.5 / fs)

            cursor = 0
            prev_idx = order[0]
            prev_end_time = starts[prev_idx] + self._nts[prev_idx] / fs
            pieces.append((prev_idx, 0, self._nts[prev_idx]))
            piece_t0.append(starts[prev_idx])
            cursor += self._nts[prev_idx]

            for idx in order[1:]:
                dev = starts[idx] - prev_end_time  # seconds; >0 gap, <0 overlap
                offset = 0
                if abs(dev) > tol:
                    kind = "gap" if dev > 0 else "overlap"
                    if self._gap_policy == "raise":
                        raise ValueError(
                            f"Timestamp {kind} of {dev:+.6f} s between file {prev_idx} "
                            f"and file {idx} (tolerance {tol:.6f} s)."
                        )
                    if dev > 0:
                        n_gap = max(1, min(int(round(dev * fs)), self._gap_samples))
                        pieces.append((-1, 0, n_gap))
                        piece_t0.append(prev_end_time)
                        gaps.append(TimeGap(cursor, cursor + n_gap, float(dev), prev_idx, idx, "gap"))
                        cursor += n_gap
                    else:
                        offset = min(int(round(-dev * fs)), self._nts[idx])
                        gaps.append(TimeGap(cursor, cursor, float(dev), prev_idx, idx, "overlap"))
                length = self._nts[idx] - offset
                if length > 0:
                    pieces.append((idx, offset, length))
                    piece_t0.append(starts[idx] + offset / fs)
                    cursor += length
                file_end = starts[idx] + self._nts[idx] / fs
                if file_end > prev_end_time:
                    prev_end_time = file_end
                prev_idx = idx
        else:
            pieces = [(i, 0, n) for i, n in enumerate(self._nts)]
            piece_t0 = [float("nan")] * len(pieces)

        self._order = order
        self._gaps = gaps
        self._piece_file = np.array([p[0] for p in pieces], dtype=np.int64)
        self._piece_off = np.array([p[1] for p in pieces], dtype=np.int64)
        self._piece_start = np.cumsum([0] + [p[2] for p in pieces]).astype(np.int64)
        self._piece_t0 = np.asarray(piece_t0, dtype=np.float64)
        self._nt_total = int(self._piece_start[-1])

        # Contiguous data segments.  Both gaps and (trimmed) overlaps are
        # treated as boundaries: continuity across them cannot be verified.
        boundaries = {g.start for g in gaps} | {g.stop for g in gaps}
        segs: List[Tuple[int, int]] = []
        for k in range(len(pieces)):
            if self._piece_file[k] < 0:
                continue
            a, b = int(self._piece_start[k]), int(self._piece_start[k + 1])
            if segs and segs[-1][1] == a and a not in boundaries:
                segs[-1] = (segs[-1][0], b)
            else:
                segs.append((a, b))
        self._segments = segs

    @property
    def gap_policy(self) -> str:
        return self._gap_policy

    @property
    def gaps(self) -> List[TimeGap]:
        """Detected discontinuities (gaps and overlaps) in logical samples."""
        return list(self._gaps)

    @property
    def segments(self) -> List[Tuple[int, int]]:
        """Contiguous data ranges ``[start, stop)`` in logical samples.

        Stateful processing must not carry state from one segment to the
        next.  Without discontinuities this is a single ``(0, nt)`` range.
        """
        return list(self._segments)

    # --------------------
    # sample <-> absolute time
    # --------------------
    def _piece_time_bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Absolute (t_start, t_end) of every piece, or None without timestamps."""
        t0 = getattr(self, "_piece_t0", None)
        if t0 is None or t0.size == 0 or not np.all(np.isfinite(t0)):
            return None
        t_end = np.empty_like(t0)
        t_end[:-1] = t0[1:]
        fs = float(self._child_fs(self._files[0]) or 1.0)
        t_end[-1] = t0[-1] + (self._piece_start[-1] - self._piece_start[-2]) / fs
        for k in range(t0.size - 1):
            if self._piece_file[k] >= 0:  # data pieces end exactly after their samples
                t_end[k] = t0[k] + (self._piece_start[k + 1] - self._piece_start[k]) / fs
        return t0, t_end

    def sample_to_time(self, index):
        """Absolute time (Unix seconds) of logical sample ``index`` (array-like ok).

        Inside data the mapping is ``piece_start_time + offset / fs``; inside a
        gap separator time is interpolated across the true missing interval.
        Without timestamps this falls back to ``start_time + index / fs``.
        """
        idx = np.asarray(index, dtype=np.float64)
        fs = float(self._child_fs(self._files[0]) or 1.0)
        bounds = self._piece_time_bounds()
        if bounds is None:
            st = self.start_time
            out = (0.0 if st is None else float(st)) + idx / fs
            return out if out.ndim else float(out)
        t0, t1 = bounds
        k = np.clip(np.searchsorted(self._piece_start, idx, side="right") - 1, 0, t0.size - 1)
        a = self._piece_start[k].astype(np.float64)
        n = (self._piece_start[k + 1] - self._piece_start[k]).astype(np.float64)
        is_gap = self._piece_file[k] < 0
        frac = np.where(n > 0, (idx - a) / np.maximum(n, 1.0), 0.0)
        out = np.where(is_gap, t0[k] + frac * (t1[k] - t0[k]), t0[k] + (idx - a) / fs)
        return out if out.ndim else float(out)

    def time_to_sample(self, t, *, clamp: bool = True):
        """Inverse of :meth:`sample_to_time`; times inside a gap map into its separator."""
        tt = np.asarray(t, dtype=np.float64)
        fs = float(self._child_fs(self._files[0]) or 1.0)
        bounds = self._piece_time_bounds()
        if bounds is None:
            st = self.start_time
            out = (tt - (0.0 if st is None else float(st))) * fs
        else:
            t0, t1 = bounds
            k = np.clip(np.searchsorted(t0, tt, side="right") - 1, 0, t0.size - 1)
            a = self._piece_start[k].astype(np.float64)
            n = (self._piece_start[k + 1] - self._piece_start[k]).astype(np.float64)
            is_gap = self._piece_file[k] < 0
            span = np.maximum(t1[k] - t0[k], 1e-12)
            out = np.where(is_gap, a + (tt - t0[k]) / span * n, a + (tt - t0[k]) * fs)
        out = np.rint(out)
        if clamp:
            out = np.clip(out, 0, max(0, self._nt_total - 1))
        out = out.astype(np.int64)
        return out if out.ndim else int(out)

    @property
    def gap_ranges(self) -> List[Tuple[int, int]]:
        """Logical sample ranges ``[start, stop)`` occupied by virtual gaps."""
        return [(g.start, g.stop) for g in self._gaps if g.stop > g.start]

    # --------------------
    # Tape interface
    # --------------------
    @property
    def shape(self):
        if self._dims == "nt_nc":
            return (self._nt_total, self._nc)
        else:
            return (self._nc, self._nt_total)

    @property
    def ndim(self):
        return self._ndim

    @property
    def dtype(self):
        return self._dtype

    @property
    def fs(self):
        return self._files[0].fs if self._files else None

    @fs.setter
    def fs(self, value: float):
        self._set_child_metadata("fs", "_fs", float(value))
        self._build_layout()  # gap sizes depend on fs

    @property
    def dx(self):
        return self._files[0].dx if self._files else None

    @dx.setter
    def dx(self, value: float):
        self._set_child_metadata("dx", "_dx", float(value))

    @property
    def dx_unit(self):
        return self._files[0].dx_unit if self._files else None

    @dx_unit.setter
    def dx_unit(self, value: str):
        self._set_child_metadata("dx_unit", "_dx_unit", str(value))

    @property
    def gauge_length(self):
        return getattr(self._files[0], "gauge_length", None) if self._files else None

    @gauge_length.setter
    def gauge_length(self, value: Optional[float]):
        gauge_length = None if value is None else float(value)
        self._set_child_metadata("gauge_length", "_gauge_length", gauge_length)

    def _set_child_metadata(self, public_name: str, private_name: str, value) -> None:
        for t in self._files:
            try:
                setattr(t, public_name, value)
            except Exception:
                if hasattr(t, private_name):
                    setattr(t, private_name, value)

    @property
    def dims(self):
        return self._dims

    @dims.setter
    def dims(self, value: str):
        new_dims = str(value)
        if new_dims not in ("nt_nc", "nc_nt"):
            raise ValueError(f"Invalid dims: {new_dims}")

        layout_attrs = (
            "_dims", "_dtype", "_ndim", "_nts", "_nc", "_order", "_gaps",
            "_piece_file", "_piece_off", "_piece_start", "_piece_t0", "_nt_total", "_segments",
        )
        old_state = {k: getattr(self, k) for k in layout_attrs if hasattr(self, k)}
        old_child_dims = [getattr(t, "dims", None) for t in self._files]

        try:
            for t in self._files:
                setattr(t, "dims", new_dims)
            self._validate_and_init()
        except Exception:
            for t, child_dims in zip(self._files, old_child_dims):
                if child_dims is not None:
                    try:
                        setattr(t, "dims", child_dims)
                    except Exception:
                        if hasattr(t, "_dims"):
                            t._dims = child_dims
            for k, v in old_state.items():
                setattr(self, k, v)
            raise

    @property
    def start_time(self):
        """Start time of logical sample 0 (earliest file), or None if unknown."""
        if not self._files:
            return None
        first = self._order[0] if getattr(self, "_order", None) else 0
        return getattr(self._files[first], "start_time", None)

    def __len__(self):
        return self._nt_total

    # --------------------
    # indexing
    # --------------------
    def __getitem__(self, key: Any) -> np.ndarray:
        if isinstance(key, tuple):
            k0 = key[0]
            k1 = key[1] if len(key) > 1 else slice(None)
        else:
            k0 = key
            k1 = slice(None)

        if self._dims == "nt_nc":
            tkey, okey = k0, k1
        else:
            tkey, okey = k1, k0

        # Case 1: scalar time index
        if isinstance(tkey, (int, np.integer)):
            i = int(tkey)
            if i < 0:
                i += self._nt_total
            if i < 0 or i >= self._nt_total:
                raise IndexError(i)
            out = self._read_range(i, i + 1, 1, okey)
            return np.take(out, 0, axis=self._result_time_axis(out, okey))

        # Case 2: time slice
        if isinstance(tkey, slice):
            start, stop, step = tkey.indices(self._nt_total)
            if step < 0:
                idx = list(range(start, stop, step))
                if not idx:
                    return self._empty(okey)
                parts = [self._read_range(i, i + 1, 1, okey) for i in idx]
                return np.concatenate(parts, axis=self._result_time_axis(parts[0], okey))
            if start >= stop:
                return self._empty(okey)
            return self._read_range(start, stop, step, okey)

        raise TypeError(f"Unsupported time-axis index type: {type(tkey)}")

    # --------------------
    # piece-wise reading
    # --------------------
    def _child_read(self, fidx: int, r0: int, r1: int, step: int, okey: Any) -> np.ndarray:
        t = self._files[fidx]
        with self._io_lock:
            self._touch(fidx)
            if self._dims == "nt_nc":
                return t[r0:r1:step, okey]
            return t[okey, r0:r1:step]

    def _empty(self, okey: Any) -> np.ndarray:
        return self._child_read(self._order[0] if self._order else 0, 0, 0, 1, okey)

    def _result_time_axis(self, template: np.ndarray, okey: Any) -> int:
        # nt_nc: time is axis 0.  nc_nt: time is the last axis of the result
        # (axis 0 when the channel key is a scalar).
        if self._dims == "nt_nc":
            return 0
        return template.ndim - 1

    def _read_range(self, start: int, stop: int, step: int, okey: Any) -> np.ndarray:
        """Read logical samples ``start:stop:step`` (step >= 1)."""
        p0 = int(np.searchsorted(self._piece_start, start, side="right") - 1)
        p1 = int(np.searchsorted(self._piece_start, stop - 1, side="right") - 1)

        chunks: List[np.ndarray] = []
        template: Optional[np.ndarray] = None
        for k in range(p0, p1 + 1):
            a = int(self._piece_start[k])
            b = int(self._piece_start[k + 1])
            s0 = max(start, a)
            s1 = min(stop, b)
            if s0 >= s1:
                continue
            # first sample on the global step grid inside this piece
            first = s0 if s0 == start else start + ((s0 - start + step - 1) // step) * step
            if first >= s1:
                continue
            n = len(range(first, s1, step))
            fidx = int(self._piece_file[k])
            if fidx >= 0:
                off = int(self._piece_off[k])
                chunk = self._child_read(fidx, off + first - a, off + s1 - a, step, okey)
                if template is None:
                    template = chunk
                chunks.append(chunk)
            else:
                chunks.append(("gap", n))  # type: ignore[arg-type]

        if template is None:
            template = self._empty(okey)
        axis = self._result_time_axis(template, okey)
        out_parts: List[np.ndarray] = []
        for c in chunks:
            if isinstance(c, tuple):
                shape = list(template.shape)
                shape[axis] = c[1]
                out_parts.append(np.full(shape, self._fill_value, dtype=template.dtype))
            else:
                out_parts.append(c)
        if not out_parts:
            return template
        if len(out_parts) == 1:
            return out_parts[0]
        return np.concatenate(out_parts, axis=axis)

    def is_gap_index(self, idx: int) -> bool:
        """Return True when logical sample ``idx`` lies inside a virtual gap."""
        k = int(np.searchsorted(self._piece_start, int(idx), side="right") - 1)
        return 0 <= k < len(self._piece_file) and int(self._piece_file[k]) < 0

    # --------------------
    # lifecycle
    # --------------------
    @property
    def files(self) -> List[Tape]:
        return self._files

    def close(self):
        with self._io_lock:
            self._open_lru.clear()
        for t in self._files:
            try:
                t.close()
            except Exception:
                print_exception("Failed to close a child tape in TapeFileSet.", prefix="TapeFileSet")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
