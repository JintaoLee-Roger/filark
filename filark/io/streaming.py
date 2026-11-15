# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""Streaming source abstraction that slices raw data into viewer-ready blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np

from .protocols import Tape
from filark._exceptions import print_exception
from filark.utils.perf import elapsed_ms, log_perf, perf_debug_enabled, perf_now

YMode = Literal["local", "fit_all"]
Layout = Literal["auto", "nc_nt", "nt_nc"]
DownsampleMode = Literal["drop", "hybrid", "peak", "linear"]


def _resize_1d_linear(arr: np.ndarray, out_len: int, axis: int) -> np.ndarray:
    if out_len <= 0:
        raise ValueError("out_len must be > 0")
    in_len = arr.shape[axis]
    if in_len == out_len:
        return arr

    x = np.moveaxis(arr, axis, -1)
    in_len = x.shape[-1]
    if in_len <= 1:
        y = np.repeat(x, out_len, axis=-1)
        return np.moveaxis(y, -1, axis)

    src_pos = np.linspace(0.0, in_len - 1, out_len, dtype=np.float32)
    left = np.floor(src_pos).astype(np.int64)
    right = np.clip(left + 1, 0, in_len - 1)
    t = (src_pos - left).astype(np.float32)

    xl = np.take(x, left, axis=-1)
    xr = np.take(x, right, axis=-1)
    y = xl * (1.0 - t) + xr * t
    return np.moveaxis(y, -1, axis)


def _resize_2d_separable_linear(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    tmp = _resize_1d_linear(img, out_w, axis=1)
    out = _resize_1d_linear(tmp, out_h, axis=0)
    return out


def _resize_1d_nearest(arr: np.ndarray, out_len: int, axis: int) -> np.ndarray:
    if out_len <= 0:
        raise ValueError("out_len must be > 0")
    in_len = arr.shape[axis]
    if in_len == out_len:
        return arr
    if in_len <= 0:
        raise ValueError("in_len must be > 0")
    if in_len == 1:
        return np.repeat(arr, out_len, axis=axis)
    idx = np.clip(np.rint(np.linspace(0.0, in_len - 1, out_len, dtype=np.float32)).astype(np.int64), 0, in_len - 1)
    return np.take(arr, idx, axis=axis)


def _sampled_len(length: int, step: int) -> int:
    length = int(length)
    step = int(max(1, step))
    if length <= 0:
        return 0
    return 1 + (length - 1) // step


def _drop_sample_indices(in_len: int, out_len: int) -> np.ndarray:
    if out_len <= 0:
        raise ValueError("out_len must be > 0")
    if in_len <= 0:
        raise ValueError("in_len must be > 0")
    if out_len >= in_len:
        if out_len == in_len:
            return np.arange(in_len, dtype=np.int64)
        return np.clip(
            np.rint(np.linspace(0.0, in_len - 1, out_len, dtype=np.float32)).astype(np.int64),
            0,
            in_len - 1,
        )
    return (np.arange(out_len, dtype=np.int64) * in_len) // out_len


def _drop_axis_index_map(src_len: int, out_len: int, *, presample_step: int = 1) -> np.ndarray:
    presample_step = int(max(1, presample_step))
    sampled_len = _sampled_len(src_len, presample_step)
    if out_len <= sampled_len:
        return _drop_sample_indices(sampled_len, out_len).astype(np.float32) * float(presample_step)
    return np.clip(np.rint(np.linspace(0.0, src_len - 1, out_len, dtype=np.float32)), 0, max(0, src_len - 1))


def _bin_edges(in_len: int, out_len: int) -> np.ndarray:
    if out_len <= 0:
        raise ValueError("out_len must be > 0")
    if in_len <= 0:
        raise ValueError("in_len must be > 0")
    if out_len > in_len:
        raise ValueError("binning requires out_len <= in_len")
    edges = np.linspace(0, in_len, out_len + 1, dtype=np.int64)
    edges[0] = 0
    edges[-1] = in_len
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1
    edges[-1] = in_len
    return edges


def _bin_centers(in_len: int, out_len: int) -> np.ndarray:
    if out_len == in_len:
        return np.arange(in_len, dtype=np.float32)
    if out_len > in_len:
        return np.linspace(0.0, in_len - 1, out_len, dtype=np.float32)
    edges = _bin_edges(in_len, out_len).astype(np.float32)
    return 0.5 * (edges[:-1] + np.maximum(edges[:-1], edges[1:] - 1.0))


def _downsample_1d_peak(arr: np.ndarray, out_len: int, axis: int) -> np.ndarray:
    if out_len <= 0:
        raise ValueError("out_len must be > 0")
    in_len = arr.shape[axis]
    if in_len == out_len:
        return arr
    if out_len > in_len:
        return _resize_1d_linear(arr.astype(np.float32, copy=False), out_len=out_len, axis=axis)

    if in_len % out_len == 0:
        fac = in_len // out_len
        x = np.moveaxis(arr, axis, -1)
        x = x.reshape(*x.shape[:-1], out_len, fac)
        seg_max = np.max(x, axis=-1)
        seg_min = np.min(x, axis=-1)
        y = np.where(np.abs(seg_min) > np.abs(seg_max), seg_min, seg_max)
        return np.moveaxis(y, -1, axis)

    starts = _bin_edges(in_len, out_len)[:-1]
    seg_max = np.maximum.reduceat(arr, starts, axis=axis)
    seg_min = np.minimum.reduceat(arr, starts, axis=axis)
    return np.where(np.abs(seg_min) > np.abs(seg_max), seg_min, seg_max)


def _downsample_2d_peak(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    out = img
    if out.shape[1] != out_w:
        out = _downsample_1d_peak(out, out_w, axis=1)
    if out.shape[0] != out_h:
        out = _downsample_1d_peak(out, out_h, axis=0)
    return out


def _downsample_1d_drop(arr: np.ndarray, out_len: int, axis: int) -> np.ndarray:
    if out_len <= 0:
        raise ValueError("out_len must be > 0")
    in_len = arr.shape[axis]
    if in_len == out_len:
        return arr
    if out_len > in_len:
        return _resize_1d_nearest(arr, out_len, axis=axis)

    if in_len % out_len == 0:
        step = in_len // out_len
        slicer = [slice(None)] * arr.ndim
        slicer[axis] = slice(0, step * out_len, step)
        out = arr[tuple(slicer)]
        if out.shape[axis] == out_len:
            return out

    return np.take(arr, _drop_sample_indices(in_len, out_len), axis=axis)


def _downsample_2d_drop(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    out = img
    if out.shape[1] != out_w:
        out = _downsample_1d_drop(out, out_w, axis=1)
    if out.shape[0] != out_h:
        out = _downsample_1d_drop(out, out_h, axis=0)
    return out


def _choose_presample_steps(
    src_h: int,
    src_w: int,
    out_h: int,
    out_w: int,
    max_samples: int,
) -> Tuple[int, int]:
    if max_samples <= 0:
        return 1, 1
    if src_h <= 0 or src_w <= 0:
        return 1, 1
    if src_h * src_w <= max_samples:
        return 1, 1

    max_step_y = max(1, int(src_h // max(1, out_h)))
    max_step_x = max(1, int(src_w // max(1, out_w)))
    step_y = 1
    step_x = 1

    def sampled_area() -> int:
        return _sampled_len(src_h, step_y) * _sampled_len(src_w, step_x)

    while sampled_area() > max_samples:
        can_x = step_x < max_step_x
        can_y = step_y < max_step_y
        if not can_x and not can_y:
            break

        rem_x = (float(src_w) / max(1.0, float(out_w))) / float(step_x) if can_x else 0.0
        rem_y = (float(src_h) / max(1.0, float(out_h))) / float(step_y) if can_y else 0.0

        if rem_x >= rem_y and can_x:
            step_x = min(max_step_x, 2 if step_x == 1 else step_x * 2)
        elif can_y:
            step_y = min(max_step_y, 2 if step_y == 1 else step_y * 2)
        elif can_x:
            step_x = min(max_step_x, 2 if step_x == 1 else step_x * 2)
        else:
            break

    return int(step_y), int(step_x)


@dataclass
class StreamingSource:
    """
    Streaming data source with explicit view↔raw coordinate mapping.

    Coordinate system (layout-independent logical axes):
      - X: time / sample (width)
      - Y: channel (height)

    Scaling model (scale >= 1.0):
      view_x = raw_x / scale_x
      view_y = raw_y / scale_y

    IO strategy
    -----------
    For row-major backends (H5, numpy, memmap) the cost model is:

    - Stepping along the **row axis** skips entire rows at the backend level,
      reducing IO proportionally to the step.
    - Stepping along the **column axis** reads the same disk data regardless
      (full rows are fetched by the chunk/page cache), so backend-level column
      striding only shrinks the output buffer — it does not save IO.

    To avoid wasting IO on column-axis stepping we always read full column
    ranges and drop in numpy.  ``_col_drop_read`` is the shared primitive that
    implements this pattern for both layouts; ``_nt_nc_col_drop_read`` and
    ``_nc_nt_col_drop_read`` are thin layout-specific wrappers.

    Layout mapping:
      nc_nt  →  rows = channels (Y),  cols = time (X)
                → Y-stepping is row-skip (efficient); X-stepping needs col-drop.
      nt_nc  →  rows = time (X),  cols = channels (Y)
                → X-stepping is row-skip (efficient); Y-stepping needs col-drop.
    """

    data: Union[np.ndarray, np.memmap, Tape]

    buffer_w: Optional[int] = None
    buffer_h: Optional[int] = None

    scale_x: float = 1.0
    scale_y: float = 1.0
    y_mode: YMode = "local"

    layout: Layout = "auto"
    dims: Layout = "auto"

    downsample_mode: DownsampleMode = "drop"
    max_read_samples: int = 32_000_000

    preprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None
    postprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None

    return_none_on_oob: bool = True

    def __post_init__(self):
        if self.data.ndim != 2:
            raise ValueError("StreamingSource expects 2D data")
        if self.scale_x < 1.0 or self.scale_y < 1.0:
            raise ValueError("scale_x/scale_y are compression factors and must be >= 1.0")
        if self.downsample_mode not in ("drop", "hybrid", "peak", "linear"):
            raise ValueError(f"unknown downsample_mode: {self.downsample_mode}")

        if self.dims != "auto":
            self.layout = self.dims
        self._layout = self._infer_layout(self.data.shape, self.layout)
        self._base_h, self._base_w = self._logical_shape_from_data_shape(self.data.shape, self._layout)
        self.layout = self._layout
        self.dims = self._layout
        self._fit_all_drop_selector_cache: dict[int, int] = {}  # out_h -> step

        # Lightweight per-read timing (always on).
        # _bench_last_filter_ms: time spent in postprocess (display filter).
        # Read by scheduler after each source.read() call.
        self._bench_last_filter_ms: float = 0.0

    @staticmethod
    def _infer_layout(shape: Tuple[int, int], layout: Layout) -> Layout:
        if layout != "auto":
            return layout
        a, b = int(shape[0]), int(shape[1])
        return "nc_nt" if b >= a else "nt_nc"

    @staticmethod
    def _logical_shape_from_data_shape(shape: Tuple[int, int], layout: Layout) -> Tuple[int, int]:
        if layout == "nc_nt":
            nc, nt = int(shape[0]), int(shape[1])
            return nc, nt
        if layout == "nt_nc":
            nt, nc = int(shape[0]), int(shape[1])
            return nc, nt
        raise ValueError(f"unknown layout: {layout}")

    # ------------------------------------------------------------------
    # IO primitives
    # ------------------------------------------------------------------

    def _col_drop_read(
        self,
        *,
        row_start: int,
        row_stop: int,
        row_step: int,
        col_start: int,
        col_stop: int,
        col_step: int,
    ) -> np.ndarray:
        """Row-major IO primitive: row-skip at backend + column-drop in numpy.

        Row stepping (``row_step``) is applied at the backend level — it skips
        entire rows and genuinely reduces disk IO for both H5 and numpy/memmap.

        Column stepping (``col_step``) does *not* reduce disk IO for H5 with
        full-width row chunks because the entire row is read from disk regardless
        of which columns are selected.  We therefore always read the full column
        range contiguously and then drop columns with a numpy stride, which is
        both cache-friendly and avoids H5py compound-selection overhead.

        To cap peak memory the time/channel axis is processed in chunks whose
        intermediate buffer stays within ``max_read_samples`` elements.  When
        the full range fits in one chunk this degenerates to a single contiguous
        read (zero extra allocation for numpy/memmap backends).

        Returns shape ``(n_rows_out, n_cols_out)``.
        """
        row_step = int(max(1, row_step))
        col_step = int(max(1, col_step))
        n_raw_cols = col_stop - col_start

        # How many output rows fit in one chunk given the memory budget.
        chunk_rows = max(1, self.max_read_samples // max(1, n_raw_cols))

        n_rows_out = _sampled_len(row_stop - row_start, row_step)
        n_cols_out = _sampled_len(n_raw_cols, col_step)

        if n_rows_out <= chunk_rows:
            # Fast path: single backend read.
            raw = self.data[row_start:row_stop:row_step, col_start:col_stop]
            if col_step > 1:
                raw = raw[:, ::col_step]
            return raw

        # Multi-chunk path: pre-allocate output, fill chunk by chunk.
        out = np.empty((n_rows_out, n_cols_out), dtype=self.data.dtype)
        r = 0
        i = row_start
        while i < row_stop:
            i_end = min(i + chunk_rows * row_step, row_stop)
            chunk = self.data[i:i_end:row_step, col_start:col_stop]
            if col_step > 1:
                chunk = chunk[:, ::col_step]
            n = chunk.shape[0]
            out[r:r + n] = chunk
            r += n
            i = i_end
        return out

    def _nt_nc_col_drop_read(
        self,
        *,
        t0: int,
        t1: int,
        step_t: int,
        c0: int,
        c1: int,
        step_c: int,
    ) -> np.ndarray:
        """nt_nc layout wrapper around ``_col_drop_read``.

        Storage rows = time samples.  Time-axis stepping (``step_t``) is the
        efficient row-skip direction.  Channel-axis stepping (``step_c``) is
        the column-drop direction.

        Returns ``(nc_out, nt_out)`` — the canonical logical (H, W) shape.
        """
        return self._col_drop_read(
            row_start=t0, row_stop=t1, row_step=step_t,
            col_start=c0, col_stop=c1, col_step=step_c,
        ).T  # (nt_out, nc_out) → transpose → (nc_out, nt_out)

    def _nc_nt_col_drop_read(
        self,
        *,
        c0: int,
        c1: int,
        step_c: int,
        t0: int,
        t1: int,
        step_t: int,
    ) -> np.ndarray:
        """nc_nt layout wrapper around ``_col_drop_read``.

        Storage rows = channels.  Channel-axis stepping (``step_c``) is the
        efficient row-skip direction.  Time-axis stepping (``step_t``) is the
        column-drop direction.

        Returns ``(nc_out, nt_out)`` — the canonical logical (H, W) shape.
        """
        return self._col_drop_read(
            row_start=c0, row_stop=c1, row_step=step_c,
            col_start=t0, col_stop=t1, col_step=step_t,
        )  # already (nc_out, nt_out)

    def _slice_raw(
        self,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        *,
        step_y: int = 1,
        step_x: int = 1,
    ) -> np.ndarray:
        """Read a raw sub-block and return shape ``(nc_out, nt_out)``.

        Applies the IO strategy appropriate for each layout:

        nc_nt (channel rows, time cols):
            - Y (channel) stepping → row-skip, efficient at backend level.
            - X (time) stepping → col-drop via ``_nc_nt_col_drop_read``.

        nt_nc (time rows, channel cols):
            - X (time) stepping → row-skip, efficient at backend level.
            - Y (channel) stepping → col-drop via ``_nt_nc_col_drop_read``.
        """
        step_y = int(max(1, step_y))
        step_x = int(max(1, step_x))

        if self._layout == "nc_nt":
            if step_x <= 1:
                # No col-drop needed; backend handles row stepping efficiently.
                return self.data[y0:y1:step_y, x0:x1]
            return self._nc_nt_col_drop_read(
                c0=y0, c1=y1, step_c=step_y,
                t0=x0, t1=x1, step_t=step_x,
            )

        else:  # nt_nc
            if step_y <= 1:
                # No col-drop needed; backend handles row stepping efficiently.
                return self.data[x0:x1:step_x, y0:y1].T
            return self._nt_nc_col_drop_read(
                t0=x0, t1=x1, step_t=step_x,
                c0=y0, c1=y1, step_c=step_y,
            )

    def _fit_all_drop_row_step(self, out_h: int) -> int:
        """Return the channel stride for a fit_all drop read.

        Always returns a plain integer step so that both nc_nt and nt_nc
        layouts use a single stepped hyperslab at the backend level instead of
        a fancy (point-selection) index.  The caller trims the result to
        exactly out_h rows with _downsample_1d_drop if needed.
        """
        out_h = int(max(1, out_h))
        cached = self._fit_all_drop_selector_cache.get(out_h)
        if cached is not None:
            return cached
        step = 1 if out_h >= self._base_h else max(1, self._base_h // out_h)
        self._fit_all_drop_selector_cache[out_h] = step
        return step

    def _slice_fit_all_drop_rows(
        self,
        *,
        x0: int,
        x1: int,
        out_h: int,
        step_x: int = 1,
    ) -> np.ndarray:
        """Read all channels sub-sampled to ``out_h`` rows, return ``(out_h, width)``.

        Layout-aware IO strategy mirrors ``_slice_raw``:

        nc_nt (channel rows):
            Channel stepping (``step_ch``) is the efficient row-skip direction.
            Time stepping (``step_x``) is col-drop → ``_nc_nt_col_drop_read``.

        nt_nc (time rows):
            Time stepping (``step_x``) is the efficient row-skip direction.
            Channel stepping (``step_ch``) is col-drop → ``_nt_nc_col_drop_read``.
        """
        step_ch = self._fit_all_drop_row_step(out_h)

        if self._layout == "nc_nt":
            if step_x <= 1:
                raw = self.data[::step_ch, x0:x1]
            else:
                raw = self._nc_nt_col_drop_read(
                    c0=0, c1=self._base_h, step_c=step_ch,
                    t0=x0, t1=x1, step_t=step_x,
                )
        else:  # nt_nc
            if step_ch <= 1:
                raw = self.data[x0:x1:step_x, :].T
            else:
                raw = self._nt_nc_col_drop_read(
                    t0=x0, t1=x1, step_t=step_x,
                    c0=0, c1=self._base_h, step_c=step_ch,
                )

        if raw.shape[0] != out_h:
            raw = _downsample_1d_drop(raw, out_h, axis=0)
        return raw

    # ------------------------------------------------------------------
    # Properties and configuration
    # ------------------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, int]:
        return (self._base_h, self._base_w)

    @property
    def dtype(self):
        return self.data.dtype

    def set_buffer_size(self, buffer_h: int, buffer_w: int):
        if buffer_w <= 0 or buffer_h <= 0:
            raise ValueError("buffer_w/buffer_h must be positive")
        self.buffer_w = int(buffer_w)
        self.buffer_h = int(buffer_h)

    def reset_processing_state(
        self,
        *,
        reset_preprocess: bool = True,
        reset_postprocess: bool = True,
    ) -> None:
        if reset_preprocess:
            reset = getattr(self.preprocess, "reset", None)
            if callable(reset):
                reset()
        if reset_postprocess:
            reset = getattr(self.postprocess, "reset", None)
            if callable(reset):
                reset()

    # ------------------------------------------------------------------
    # Discontinuity-aware processing
    # ------------------------------------------------------------------

    def _segment_arrays(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Contiguous data segments of the underlying source (raw time samples).

        Returns ``None`` when the source is a single continuous segment, so the
        common case keeps the original single-call processing path.
        """
        segs = getattr(self.data, "segments", None)
        if not segs or len(segs) <= 1:
            if not segs:
                return None
            a, b = segs[0]
            if int(a) == 0 and int(b) >= self._base_w:
                return None
        arr = np.asarray(segs, dtype=np.float64).reshape(-1, 2)
        return arr[:, 0], arr[:, 1]

    @staticmethod
    def _column_labels(pos: np.ndarray, starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
        """Segment index per column position, ``-1`` inside a gap."""
        idx = np.searchsorted(starts, pos, side="right") - 1
        valid = idx >= 0
        inside = np.zeros(pos.shape, dtype=bool)
        inside[valid] = pos[valid] < stops[idx[valid]]
        return np.where(inside, idx, -1)

    def _apply_processing(
        self,
        proc: Callable[[np.ndarray], np.ndarray],
        block: np.ndarray,
        col_pos: np.ndarray,
        col_step: float,
        pad_value: float,
    ) -> np.ndarray:
        """Apply ``proc`` without carrying information across discontinuities.

        ``col_pos`` holds the raw time position of every column of ``block``.
        Stateful processors are run segment by segment and reset at every
        segment start; stateless processors run once, after which gap columns
        are blanked so filter tails do not leak into missing-data regions.
        """
        seg = self._segment_arrays()
        if seg is None or block.ndim != 2 or block.shape[1] == 0:
            return proc(block)
        starts, stops = seg
        labels = self._column_labels(np.asarray(col_pos, dtype=np.float64), starts, stops)
        prev = self._column_labels(np.asarray([col_pos[0] - col_step], dtype=np.float64), starts, stops)[0]

        # Fast path: entirely inside one segment that also contains the previous column.
        if labels[0] >= 0 and np.all(labels == labels[0]) and prev == labels[0]:
            return proc(block)

        change = np.flatnonzero(np.diff(labels)) + 1
        bounds = np.concatenate(([0], change, [labels.size]))

        if not self._proc_is_stateful(proc):
            out = proc(block)
            gap_cols = labels < 0
            if np.any(gap_cols):
                out = np.array(out, copy=True)
                out[:, gap_cols] = pad_value
            return out

        reset = getattr(proc, "reset", None)
        parts: list[np.ndarray] = []
        out_dtype = None
        pending_gaps: list[int] = []
        for i in range(bounds.size - 1):
            a, b = int(bounds[i]), int(bounds[i + 1])
            lab = int(labels[a])
            if lab < 0:
                parts.append(None)  # type: ignore[arg-type]
                pending_gaps.append(len(parts) - 1)
                continue
            if (i > 0 or prev != lab) and callable(reset):
                reset()
            y = proc(block[:, a:b])
            out_dtype = y.dtype if out_dtype is None else out_dtype
            parts.append(y)
        if out_dtype is None:
            out_dtype = block.dtype
            if callable(reset):
                reset()
        for j in pending_gaps:
            a, b = int(bounds[j]), int(bounds[j + 1])
            parts[j] = np.full((block.shape[0], b - a), pad_value, dtype=out_dtype)
        return np.concatenate(parts, axis=1)

    @staticmethod
    def _proc_is_stateful(proc) -> bool:
        if proc is None:
            return False
        fn = getattr(proc, "is_stateful", None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                print_exception(f"StreamingSource processor {proc!r} failed in is_stateful().", prefix="Streaming")
        return callable(getattr(proc, "reset", None))

    def has_stateful_processing(self) -> bool:
        for proc in (self.preprocess, self.postprocess):
            if self._proc_is_stateful(proc):
                return True
        return False

    def can_snapshot_processing_state(self) -> bool:
        saw_stateful = False
        for proc in (self.preprocess, self.postprocess):
            if not self._proc_is_stateful(proc):
                continue
            saw_stateful = True
            if not callable(getattr(proc, "get_state", None)):
                return False
            if not callable(getattr(proc, "set_state", None)):
                return False
        return saw_stateful

    def capture_processing_state(self) -> Optional[Tuple[object, object]]:
        if not self.can_snapshot_processing_state():
            return None

        states: list[object] = []
        for proc in (self.preprocess, self.postprocess):
            if not self._proc_is_stateful(proc):
                states.append(None)
                continue
            states.append(getattr(proc, "get_state")())
        return tuple(states)  # type: ignore[return-value]

    def restore_processing_state(self, snapshot: Optional[Tuple[object, object]]) -> None:
        if snapshot is None:
            self.reset_processing_state()
            return

        states = list(snapshot)
        if len(states) != 2:
            raise ValueError("processing snapshot must have two entries")

        for proc, state in zip((self.preprocess, self.postprocess), states):
            if proc is None:
                continue
            set_state = getattr(proc, "set_state", None)
            reset = getattr(proc, "reset", None)
            if not self._proc_is_stateful(proc):
                continue
            if state is None:
                if callable(reset):
                    reset()
                continue
            if not callable(set_state):
                raise ValueError("processor does not support state restore")
            set_state(state)

    def processing_warmup_width(self, visible_w: Optional[int] = None) -> int:
        hint = int(max(0, visible_w or 0))
        widths: list[int] = []
        for proc in (self.preprocess, self.postprocess):
            fn = getattr(proc, "warmup_samples", None)
            if callable(fn):
                try:
                    widths.append(int(max(0, fn(hint))))
                    continue
                except TypeError:
                    widths.append(int(max(0, fn())))
                    continue
                except Exception:
                    print_exception(f"StreamingSource processor {proc!r} failed in warmup_samples().", prefix="Streaming")
            reset = getattr(proc, "reset", None)
            if callable(reset):
                widths.append(hint)
        return max(widths, default=0)

    def fit_scale_y(self, buffer_h: Optional[int] = None) -> float:
        bh = int(buffer_h if buffer_h is not None else self.buffer_h)
        if bh <= 0:
            raise ValueError("buffer_h must be positive")
        return max(1.0, float(self._base_h) / float(bh))

    def set_scale(
        self,
        scale_x: Optional[float] = None,
        scale_y: Optional[float] = None,
        y_mode: Optional[YMode] = None,
        downsample_mode: Optional[DownsampleMode] = None,
    ):
        if scale_x is not None:
            if scale_x < 1.0:
                raise ValueError("scale_x is compression factor, must be >= 1.0")
            self.scale_x = float(scale_x)
        if scale_y is not None:
            if scale_y < 1.0:
                raise ValueError("scale_y is compression factor, must be >= 1.0")
            self.scale_y = float(scale_y)
        if y_mode is not None:
            self.y_mode = y_mode
        if downsample_mode is not None:
            if downsample_mode not in ("drop", "hybrid", "peak", "linear"):
                raise ValueError(f"unknown downsample_mode: {downsample_mode}")
            self.downsample_mode = downsample_mode

    # ------------------------------------------------------------------
    # Downsampling helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ceil_int(x: float) -> int:
        return int(np.ceil(x))

    def _clip0(self, v0: int, src_len: int, base_len: int) -> int:
        if src_len >= base_len:
            return 0
        return max(0, min(int(v0), base_len - src_len))

    @staticmethod
    def _as_int_scale(s: float, tol: float = 1e-6) -> Optional[int]:
        r = int(round(s))
        if r >= 1 and abs(s - r) <= tol:
            return r
        return None

    def _linear_downsample(
        self,
        raw: np.ndarray,
        *,
        bw: int,
        bh: int,
        sx_i: Optional[int],
        sy_i: Optional[int],
        y_mode: YMode,
        pad_value: float,
    ) -> np.ndarray:
        if y_mode != "fit_all" and (sx_i is not None or sy_i is not None):
            out = raw

            if sx_i is not None:
                out = out[:, ::sx_i]
                if out.shape[1] != bw:
                    out = out[:, :bw] if out.shape[1] > bw else np.pad(
                        out,
                        ((0, 0), (0, bw - out.shape[1])),
                        mode="constant",
                        constant_values=pad_value,
                    )
            else:
                out = _resize_1d_linear(out.astype(np.float32, copy=False), out_len=bw, axis=1)

            if sy_i is not None:
                out = out[::sy_i, :]
                if out.shape[0] != bh:
                    out = out[:bh, :] if out.shape[0] > bh else np.pad(
                        out,
                        ((0, bh - out.shape[0]), (0, 0)),
                        mode="constant",
                        constant_values=pad_value,
                    )
            else:
                out = _resize_1d_linear(out.astype(np.float32, copy=False), out_len=bh, axis=0)
            return out

        return _resize_2d_separable_linear(raw.astype(np.float32, copy=False), out_h=bh, out_w=bw)

    def _peak_downsample(
        self,
        raw: np.ndarray,
        *,
        bw: int,
        bh: int,
    ) -> np.ndarray:
        return _downsample_2d_peak(raw.astype(np.float32, copy=False), out_h=bh, out_w=bw)

    def _drop_downsample(
        self,
        raw: np.ndarray,
        *,
        bw: int,
        bh: int,
    ) -> np.ndarray:
        return _downsample_2d_drop(raw, out_h=bh, out_w=bw)

    # ------------------------------------------------------------------
    # Raw (no-scale) read — for ROI and processing use cases
    # ------------------------------------------------------------------

    def _read_raw_block(
        self,
        c0: int,
        t0: int,
        bh: int,
        bw: int,
        *,
        pad_value: float,
        out_dtype: Optional[np.dtype],
        return_none_on_oob: bool,
    ) -> Optional[np.ndarray]:
        """Read an exact ``bh × bw`` block in raw (channel, time) coordinates.

        No scale mapping, no downsampling.  ``c0`` and ``t0`` are raw channel
        and time indices respectively.  ``preprocess`` and ``postprocess`` are
        still applied if set.

        Out-of-bounds coordinates are handled by padding with ``pad_value`` (or
        returning ``None`` when ``return_none_on_oob`` is True and the origin
        itself is out of bounds).
        """
        if return_none_on_oob and (c0 >= self._base_h or t0 >= self._base_w):
            return None

        c1 = min(c0 + bh, self._base_h)
        t1 = min(t0 + bw, self._base_w)
        out = self._slice_raw(c0, c1, t0, t1, step_y=1, step_x=1)

        if out.shape[0] == 0 or out.shape[1] == 0:
            return None if return_none_on_oob else np.full((bh, bw), pad_value, dtype=self.data.dtype)

        pad_h = bh - out.shape[0]
        pad_w = bw - out.shape[1]
        if pad_h > 0 or pad_w > 0:
            out = np.pad(
                out,
                ((0, max(0, pad_h)), (0, max(0, pad_w))),
                mode="constant",
                constant_values=pad_value,
            )

        col_pos = t0 + np.arange(out.shape[1], dtype=np.float64)
        if self.preprocess is not None:
            out = self._apply_processing(self.preprocess, out, col_pos, 1.0, pad_value)

        if self.postprocess is not None:
            t_start = perf_now()
            out = self._apply_processing(self.postprocess, out, col_pos, 1.0, pad_value)
            self._bench_last_filter_ms = elapsed_ms(t_start)
        else:
            self._bench_last_filter_ms = 0.0

        if out_dtype is not None:
            out = out.astype(out_dtype, copy=False)
        return out

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    def read(
        self,
        y0: int,
        x0: int,
        buffer_h: Optional[int] = None,
        buffer_w: Optional[int] = None,
        *,
        scale_y: Optional[float] = None,
        scale_x: Optional[float] = None,
        y_mode: Optional[YMode] = None,
        pad_value: float = 0.0,
        out_dtype: Optional[np.dtype] = None,
        return_none_on_oob: Optional[bool] = None,
        raw: bool = False,
        debug_context: Optional[dict] = None,
    ) -> Optional[np.ndarray]:
        """Read a buffer-sized block from the source.

        Parameters
        ----------
        y0, x0 :
            View-space origin (channel, time).  When ``raw=True`` these are
            raw coordinates (channel index, time sample index).
        buffer_h, buffer_w :
            Output buffer dimensions.  Defaults to the values set by
            ``set_buffer_size``.
        scale_y, scale_x :
            Override the instance-level compression factors for this call.
            Ignored when ``raw=True``.
        y_mode :
            ``"local"`` (default) or ``"fit_all"`` (all channels visible).
            Ignored when ``raw=True``.
        raw : bool, default False
            When True, bypass scale mapping and downsampling entirely.
            ``y0`` / ``x0`` are treated as raw (channel, time) coordinates and
            the returned array contains exactly ``buffer_h × buffer_w`` raw
            samples (zero-padded at boundaries).  Useful for ROI processing
            where exact sample values are required.

            ``preprocess`` and ``postprocess`` are still applied when set.
        """
        perf_on = perf_debug_enabled()
        total_start = perf_now() if perf_on else 0.0

        if (buffer_h is None and self.buffer_h is None) or (buffer_w is None and self.buffer_w is None):
            raise ValueError("buffer size not init")

        bw = int(buffer_w if buffer_w is not None else self.buffer_w)
        bh = int(buffer_h if buffer_h is not None else self.buffer_h)
        none_on_oob = self.return_none_on_oob if return_none_on_oob is None else bool(return_none_on_oob)

        if bw <= 0 or bh <= 0:
            raise ValueError("buffer size must be positive")

        # ------------------------------------------------------------------
        # Raw mode: direct block read, no scale/downsampling.
        # ------------------------------------------------------------------
        if raw:
            return self._read_raw_block(
                int(y0), int(x0), bh, bw,
                pad_value=pad_value,
                out_dtype=out_dtype,
                return_none_on_oob=none_on_oob,
            )

        # ------------------------------------------------------------------
        # Normal (scaled / downsampled) mode.
        # ------------------------------------------------------------------
        slice_ms = 0.0
        preprocess_ms = 0.0
        downsample_ms = 0.0
        postprocess_ms = 0.0

        sx = float(scale_x if scale_x is not None else self.scale_x)
        sy = float(scale_y if scale_y is not None else self.scale_y)
        ym = y_mode if y_mode is not None else self.y_mode

        if sx < 1.0 or sy < 1.0:
            raise ValueError("scale_x/scale_y are compression factors and must be >= 1.0")

        sx_i = self._as_int_scale(sx)
        sy_i = self._as_int_scale(sy)

        raw_x0 = int(np.floor(x0 * sx))
        raw_y0 = int(np.floor(y0 * sy))

        if sx_i is not None:
            src_w = max(1, bw * sx_i)
        else:
            src_w = max(1, self._ceil_int(bw * sx))

        if ym == "fit_all":
            src_h = self._base_h
            raw_y0_eff = 0
        else:
            if sy_i is not None:
                src_h = max(1, bh * sy_i)
            else:
                src_h = max(1, self._ceil_int(bh * sy))
            raw_y0_eff = self._clip0(raw_y0, src_h, self._base_h)

        raw_x0_eff = self._clip0(raw_x0, src_w, self._base_w)

        if none_on_oob and (raw_x0 >= self._base_w or raw_y0 >= self._base_h):
            if perf_on:
                fields = dict(debug_context or {})
                fields.update(
                    result="oob",
                    x0=int(x0),
                    y0=int(y0),
                    bw=int(bw),
                    bh=int(bh),
                    sx=float(sx),
                    sy=float(sy),
                    y_mode=str(ym),
                    downsample=str(self.downsample_mode),
                    ms=elapsed_ms(total_start),
                )
                log_perf("stream.read", **fields)
            return None

        step_y = 1
        step_x = 1
        if self.downsample_mode == "hybrid":
            step_y, step_x = _choose_presample_steps(
                src_h=src_h,
                src_w=src_w,
                out_h=bh,
                out_w=bw,
                max_samples=int(self.max_read_samples),
            )
        elif self.downsample_mode == "drop":
            step_x = max(1, src_w // max(1, bw))
            if ym != "fit_all":
                step_y = max(1, src_h // max(1, bh))

        x1 = min(self._base_w, raw_x0_eff + src_w)
        use_fit_all_drop_row_select = (
            self.downsample_mode == "drop"
            and ym == "fit_all"
        )
        if use_fit_all_drop_row_select:
            slice_start = perf_now() if perf_on else 0.0
            raw_block = self._slice_fit_all_drop_rows(
                x0=raw_x0_eff,
                x1=x1,
                out_h=bh,
                step_x=step_x,
            )
            if perf_on:
                slice_ms += elapsed_ms(slice_start)
            exp_h = bh
        else:
            y1 = min(self._base_h, raw_y0_eff + src_h)
            slice_start = perf_now() if perf_on else 0.0
            raw_block = self._slice_raw(raw_y0_eff, y1, raw_x0_eff, x1, step_y=step_y, step_x=step_x)
            if perf_on:
                slice_ms += elapsed_ms(slice_start)
            exp_h = _sampled_len(src_h, step_y)

        exp_w = _sampled_len(src_w, step_x)
        if raw_block.shape[0] != exp_h or raw_block.shape[1] != exp_w:
            pad_start = perf_now() if perf_on else 0.0
            if none_on_oob and (raw_block.shape[0] == 0 or raw_block.shape[1] == 0):
                if perf_on:
                    fields = dict(debug_context or {})
                    fields.update(
                        result="empty",
                        x0=int(x0),
                        y0=int(y0),
                        bw=int(bw),
                        bh=int(bh),
                        sx=float(sx),
                        sy=float(sy),
                        y_mode=str(ym),
                        downsample=str(self.downsample_mode),
                        slice_ms=float(slice_ms),
                        ms=elapsed_ms(total_start),
                    )
                    log_perf("stream.read", **fields)
                return None
            pad_h = exp_h - raw_block.shape[0]
            pad_w = exp_w - raw_block.shape[1]
            raw_block = np.pad(
                raw_block,
                pad_width=((0, max(0, pad_h)), (0, max(0, pad_w))),
                mode="constant",
                constant_values=pad_value,
            )
            if perf_on:
                slice_ms += elapsed_ms(pad_start)

        if self.preprocess is not None:
            proc_start = perf_now() if perf_on else 0.0
            pre_pos = raw_x0_eff + np.arange(raw_block.shape[1], dtype=np.float64) * step_x
            raw_block = self._apply_processing(self.preprocess, raw_block, pre_pos, float(step_x), pad_value)
            if perf_on:
                preprocess_ms = elapsed_ms(proc_start)

        downsample_start = perf_now() if perf_on else 0.0
        if self.downsample_mode == "linear":
            out = self._linear_downsample(
                raw_block,
                bw=bw,
                bh=bh,
                sx_i=sx_i,
                sy_i=sy_i,
                y_mode=ym,
                pad_value=pad_value,
            )
        elif self.downsample_mode in ("peak", "hybrid"):
            out = self._peak_downsample(raw_block, bw=bw, bh=bh)
        else:
            out = self._drop_downsample(raw_block, bw=bw, bh=bh)
        if perf_on:
            downsample_ms = elapsed_ms(downsample_start)

        if self.postprocess is not None:
            proc_start = perf_now()
            col_step = float(src_w) / float(max(1, out.shape[1]))
            post_pos = raw_x0_eff + np.floor((np.arange(out.shape[1], dtype=np.float64) + 0.5) * col_step)
            out = self._apply_processing(self.postprocess, out, post_pos, col_step, pad_value)
            postprocess_ms = elapsed_ms(proc_start)
            self._bench_last_filter_ms = postprocess_ms
        else:
            self._bench_last_filter_ms = 0.0
            postprocess_ms = 0.0

        if out_dtype is not None:
            out = out.astype(out_dtype, copy=False)

        if perf_on:
            fields = dict(debug_context or {})
            fields.update(
                result="ok",
                x0=int(x0),
                y0=int(y0),
                raw_x0=int(raw_x0_eff),
                raw_y0=int(raw_y0_eff),
                src_w=int(src_w),
                src_h=int(src_h),
                step_x=int(step_x),
                step_y=int(step_y),
                bw=int(bw),
                bh=int(bh),
                out_shape=(int(out.shape[0]), int(out.shape[1])),
                sx=float(sx),
                sy=float(sy),
                y_mode=str(ym),
                downsample=str(self.downsample_mode),
                slice_ms=float(slice_ms),
                preprocess_ms=float(preprocess_ms),
                downsample_ms=float(downsample_ms),
                postprocess_ms=float(postprocess_ms),
                ms=elapsed_ms(total_start),
            )
            log_perf("stream.read", **fields)

        return out

    def read_with_history(
        self,
        y0: int,
        x0: int,
        buffer_h: Optional[int] = None,
        buffer_w: Optional[int] = None,
        *,
        history_w: Optional[int] = None,
        scale_y: Optional[float] = None,
        scale_x: Optional[float] = None,
        y_mode: Optional[YMode] = None,
        pad_value: float = 0.0,
        out_dtype: Optional[np.dtype] = None,
        return_none_on_oob: Optional[bool] = None,
        debug_context: Optional[dict] = None,
    ) -> Optional[np.ndarray]:
        perf_on = perf_debug_enabled()
        total_start = perf_now() if perf_on else 0.0
        bw = int(buffer_w if buffer_w is not None else self.buffer_w)
        if bw <= 0:
            raise ValueError("buffer width must be positive")

        history = int(
            max(
                0,
                history_w if history_w is not None else self.processing_warmup_width(bw),
            )
        )
        if history <= 0 or not self.has_stateful_processing():
            self.reset_processing_state()
            out = self.read(
                y0,
                x0,
                buffer_h,
                buffer_w,
                scale_y=scale_y,
                scale_x=scale_x,
                y_mode=y_mode,
                pad_value=pad_value,
                out_dtype=out_dtype,
                return_none_on_oob=return_none_on_oob,
                debug_context=dict(debug_context or {}, history_w=0, history_mode="disabled"),
            )
            if perf_on:
                fields = dict(debug_context or {})
                fields.update(history_w=0, history_mode="disabled", ms=elapsed_ms(total_start))
                log_perf("stream.read_with_history", **fields)
            return out

        x0_i = int(x0)
        hist_x0 = max(0, x0_i - history)
        hist_cols = bw + (x0_i - hist_x0)

        self.reset_processing_state()
        out = self.read(
            y0,
            hist_x0,
            buffer_h,
            hist_cols,
            scale_y=scale_y,
            scale_x=scale_x,
            y_mode=y_mode,
            pad_value=pad_value,
            out_dtype=None,
            return_none_on_oob=return_none_on_oob,
            debug_context=dict(
                debug_context or {},
                history_w=int(history),
                history_mode="warmup",
                hist_x0=int(hist_x0),
                hist_cols=int(hist_cols),
            ),
        )
        if out is None:
            if perf_on:
                fields = dict(debug_context or {})
                fields.update(
                    history_w=int(history),
                    history_mode="warmup",
                    hist_x0=int(hist_x0),
                    hist_cols=int(hist_cols),
                    result="none",
                    ms=elapsed_ms(total_start),
                )
                log_perf("stream.read_with_history", **fields)
            return None

        offset = x0_i - hist_x0
        out = out[:, offset:offset + bw]
        if out.shape[1] != bw:
            out = np.pad(
                out,
                pad_width=((0, 0), (0, max(0, bw - out.shape[1]))),
                mode="constant",
                constant_values=pad_value,
            )
        if out_dtype is not None:
            out = out.astype(out_dtype, copy=False)
        if perf_on:
            fields = dict(debug_context or {})
            fields.update(
                history_w=int(history),
                history_mode="warmup",
                hist_x0=int(hist_x0),
                hist_cols=int(hist_cols),
                result="ok",
                out_shape=(int(out.shape[0]), int(out.shape[1])),
                ms=elapsed_ms(total_start),
            )
            log_perf("stream.read_with_history", **fields)
        return out

    def index_map(
        self,
        y0: int,
        x0: int,
        *,
        buffer_w: Optional[int] = None,
        buffer_h: Optional[int] = None,
        scale_x: Optional[float] = None,
        scale_y: Optional[float] = None,
        y_mode: Optional[YMode] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        bw = int(buffer_w if buffer_w is not None else self.buffer_w)
        bh = int(buffer_h if buffer_h is not None else self.buffer_h)
        sx = float(scale_x if scale_x is not None else self.scale_x)
        sy = float(scale_y if scale_y is not None else self.scale_y)
        ym = y_mode if y_mode is not None else self.y_mode

        sx_i = self._as_int_scale(sx)
        sy_i = self._as_int_scale(sy)

        if sx_i is not None:
            src_w = max(1, bw * sx_i)
        else:
            src_w = max(1, self._ceil_int(bw * sx))

        raw_x0 = float(np.floor(x0 * sx))
        raw_x0_eff = float(self._clip0(int(raw_x0), src_w, self._base_w))

        if ym == "fit_all":
            src_h = self._base_h
            raw_y0_eff = 0.0
        else:
            if sy_i is not None:
                src_h = max(1, bh * sy_i)
            else:
                src_h = max(1, self._ceil_int(bh * sy))
            raw_y0 = float(np.floor(y0 * sy))
            raw_y0_eff = float(self._clip0(int(raw_y0), src_h, self._base_h))

        step_y = 1
        step_x = 1
        if self.downsample_mode == "hybrid":
            step_y, step_x = _choose_presample_steps(
                src_h=src_h,
                src_w=src_w,
                out_h=bh,
                out_w=bw,
                max_samples=int(self.max_read_samples),
            )
        elif self.downsample_mode == "drop":
            step_y = max(1, src_h // max(1, bh))
            step_x = max(1, src_w // max(1, bw))

        if self.downsample_mode == "linear" and ym != "fit_all" and sx_i is not None:
            xs = raw_x0_eff + (np.arange(bw, dtype=np.float32) * sx_i)
        elif self.downsample_mode == "drop":
            xs = raw_x0_eff + _drop_axis_index_map(src_w, bw, presample_step=step_x)
        else:
            xs = raw_x0_eff + _bin_centers(src_w, bw)

        if self.downsample_mode == "linear" and ym != "fit_all" and sy_i is not None:
            ys = raw_y0_eff + (np.arange(bh, dtype=np.float32) * sy_i)
        elif self.downsample_mode == "drop":
            ys = raw_y0_eff + _drop_axis_index_map(src_h, bh, presample_step=step_y)
        else:
            ys = raw_y0_eff + _bin_centers(src_h, bh)

        return ys, xs
