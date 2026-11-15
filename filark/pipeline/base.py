# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""
BasePipeline — unified interface for in-memory and large-data processing.

Sub-classes implement ``_run(data)`` for single-pass logic; the base class
provides ``_run_stream`` for chunked execution over large sources, and
``__call__`` as the single entry point for both paths.

Chunking modes in ``_run_stream``
----------------------------------
Channel mode (default):
    Iterates over channel slices — full time range per chunk.
    Good for medium-sized data.

Block mode (``block_t`` is set):
    Iterates over 2-D blocks, **time as the outer loop** — matches the
    natural on-disk layout of DAS data (data is stored / arrives as time
    segments, not channel columns).  Good for large data.

    Pass ``block_t=0`` to auto-estimate block dimensions from
    ``target_chunk_mb``.

Iteration geometry (slice maths) is delegated to ``dsp.chunk`` iterators
(``iter_channel_slices``, ``iter_tiles_2d``) so there is no duplication
between the pipeline layer and the DSP layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

# Reuse slice iterators from dsp.chunk — no duplication of geometry maths.
from filark.dsp.chunk import (
    _temporal_taper_weights,
    estimate_block_size,
    estimate_chunk_channels,
    iter_channel_slices,
    iter_tiles_2d,
)

__all__ = ["BasePipeline"]


# ---------------------------------------------------------------------------
# Internal IO helpers
# ---------------------------------------------------------------------------

def _norm_layout(source: Any) -> str:
    dims = str(getattr(source, "dims", "nt_nc") or "nt_nc").lower()
    return dims if dims in ("nt_nc", "nc_nt") else "nt_nc"


def _semantic_shape(source: Any, layout: str) -> tuple[int, int]:
    a, b = int(source.shape[0]), int(source.shape[1])
    return (a, b) if layout == "nt_nc" else (b, a)


def _source_segments(source: Any, nt: int) -> list[tuple[int, int]]:
    """Contiguous data ranges of ``source`` along time (see ``TapeFileSet.segments``).

    Sources without discontinuity metadata are a single ``(0, nt)`` segment.
    """
    segs = getattr(source, "segments", None)
    if not segs:
        return [(0, int(nt))]
    out = [(max(0, int(a)), min(int(nt), int(b))) for a, b in segs]
    return [(a, b) for a, b in out if b > a]


def _read_channel_block(
    source: Any, layout: str, r0: int, r1: int, t0: int | None = None, t1: int | None = None,
) -> np.ndarray:
    """Read a channel-range slice over ``[t0, t1)`` (full time by default) → [nt, r1-r0] float32."""
    ts = slice(t0, t1)
    blk = (
        np.asarray(source[ts, r0:r1], dtype=np.float32)
        if layout == "nt_nc"
        else np.asarray(source[r0:r1, ts], dtype=np.float32).T
    )
    if blk.ndim != 2:
        raise ValueError(f"Expected 2D block, got ndim={blk.ndim}")
    return blk


def _write_channel_block(
    out_arr: np.ndarray, layout: str, c0: int, c1: int, block_nt_nc: np.ndarray,
    t0: int | None = None, t1: int | None = None,
) -> None:
    ts = slice(t0, t1)
    if layout == "nt_nc":
        out_arr[ts, c0:c1] = block_nt_nc
    else:
        out_arr[c0:c1, ts] = block_nt_nc.T


def _read_tile(
    source: Any, layout: str, rt0: int, rt1: int, rc0: int, rc1: int
) -> np.ndarray:
    """Read a 2-D tile (time × channel) → [rt, rc] float32."""
    if layout == "nt_nc":
        blk = np.asarray(source[rt0:rt1, rc0:rc1], dtype=np.float32)
    else:
        blk = np.asarray(source[rc0:rc1, rt0:rt1], dtype=np.float32).T
    if blk.ndim != 2:
        raise ValueError(f"Expected 2D tile, got ndim={blk.ndim}")
    return blk


def _write_tile(
    out_arr: np.ndarray,
    layout: str,
    t0: int,
    t1: int,
    c0: int,
    c1: int,
    core_nt_nc: np.ndarray,
) -> None:
    if layout == "nt_nc":
        out_arr[t0:t1, c0:c1] = core_nt_nc
    else:
        out_arr[c0:c1, t0:t1] = core_nt_nc.T


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BasePipeline:
    """Base class for all DAS pipelines.

    Sub-classes must implement :meth:`_run`.  The base provides chunked
    execution via :meth:`_run_stream`, and routes through :meth:`__call__`.

    Parameters
    ----------
    algorithm : callable, optional
        Core processing callable ``(np.ndarray) -> np.ndarray``.
        Used by the default ``_run`` implementation.
    processor : object, optional
        Object with optional ``preprocess`` / ``postprocess`` methods.
    """

    def __init__(self, algorithm=None, processor=None):
        self.algorithm = algorithm
        self.processor = processor

    # ------------------------------------------------------------------
    # Override points
    # ------------------------------------------------------------------

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        if self.processor is not None and hasattr(self.processor, "preprocess"):
            return self.processor.preprocess(data)
        return data

    def postprocess(self, data: np.ndarray) -> np.ndarray:
        if self.processor is not None and hasattr(self.processor, "postprocess"):
            return self.processor.postprocess(data)
        return data

    def _run(self, data: np.ndarray) -> np.ndarray:
        """Single-pass: preprocess → algorithm → postprocess."""
        data = self.preprocess(data)
        if self.algorithm is not None:
            data = self.algorithm(data)
        return self.postprocess(data)

    # ------------------------------------------------------------------
    # Chunked execution engine
    # ------------------------------------------------------------------

    def _run_stream(
        self,
        source: Any,
        save_path: str | Path | None = None,
        *,
        chunked: bool = True,
        # --- channel mode ---
        chunk_channels: int | None = None,
        overlap_channels: int = 0,
        # --- block mode (activated when block_t is not None) ---
        block_t: int | None = None,
        block_c: int | None = None,
        overlap_t: int = 0,
        overlap_c: int | None = None,
        taper_t: bool = False,
        # --- common ---
        target_chunk_mb: float = 128.0,
        progress_cb: Callable[[int, int], None] | None = None,
        cancel_cb: Callable[[], bool] | None = None,
    ) -> np.ndarray | str:
        """Chunked execution: read block → _run → write block.

        Parameters
        ----------
        source:
            Array-like with ``shape`` (and optional ``dims``) attributes.
        save_path:
            If given, write output to a ``.npy`` memory-mapped file and
            return its path.  If ``None``, accumulate in memory.
        chunk_channels:
            [channel mode] Channels per chunk.  Auto-estimated when ``None``.
        overlap_channels:
            [channel mode] Read halo on each channel side.
        block_t:
            [block mode] Time samples per block.  Activates block mode.
            Pass ``0`` to auto-estimate from ``target_chunk_mb``.
        block_c:
            [block mode] Channels per block.  Defaults to full width.
        overlap_t:
            [block mode] Read halo on each temporal side.
        overlap_c:
            [block mode] Read halo on each channel side.
            Falls back to ``overlap_channels`` when ``None``.
        taper_t:
            [block mode] Apply cosine taper to temporal halo before ``_run``.
            Reduces spectral leakage for frequency-domain operators (e.g. f-k).
        target_chunk_mb:
            Memory budget for auto-estimation (default 128 MiB).
        progress_cb:
            ``progress_cb(done, total)`` after each chunk.
            Channel mode: done/total in channels.
            Block mode: done/total in time samples (updated per time-row).
        cancel_cb:
            Called before each chunk; raises ``RuntimeError`` if ``True``.
        """
        layout = _norm_layout(source)
        nt, nc = _semantic_shape(source, layout)
        if nt <= 0 or nc <= 0:
            raise ValueError(f"Invalid source shape: {source.shape}")

        # Prepare output array
        out_shape = tuple(int(v) for v in source.shape[:2])
        if save_path is not None:
            out_arr: np.ndarray = np.lib.format.open_memmap(
                str(save_path), mode="w+", dtype=np.float32, shape=out_shape,
            )
        else:
            out_arr = np.zeros(out_shape, dtype=np.float32)

        # Discontinuous sources (e.g. TapeFileSet with timestamp gaps) are
        # processed segment by segment so no chunk, halo or filter state spans
        # a gap.  Gap regions keep the zero fill of the output array.
        segments = _source_segments(source, nt)

        if block_t is not None:
            self._run_stream_block(
                source, out_arr, layout, nt, nc, segments=segments,
                block_t=block_t,
                block_c=block_c,
                overlap_t=overlap_t,
                overlap_c=int(overlap_c) if overlap_c is not None else int(overlap_channels),
                taper_t=taper_t,
                target_chunk_mb=target_chunk_mb,
                progress_cb=progress_cb,
                cancel_cb=cancel_cb,
            )
        else:
            self._run_stream_channel(
                source, out_arr, layout, nt, nc, segments=segments,
                chunk_channels=chunk_channels,
                overlap_channels=overlap_channels,
                target_chunk_mb=target_chunk_mb,
                progress_cb=progress_cb,
                cancel_cb=cancel_cb,
            )

        if save_path is not None:
            if hasattr(out_arr, "flush"):
                out_arr.flush()
            return str(save_path)
        return out_arr

    def _run_stream_channel(
        self,
        source, out_arr, layout, nt, nc, *,
        chunk_channels, overlap_channels, target_chunk_mb, progress_cb, cancel_cb,
        segments=None,
    ):
        """Channel-mode inner loop (full time per segment, chunked channels)."""
        segments = segments or [(0, nt)]
        seg_nt = max(b - a for a, b in segments)
        chunk = (
            estimate_chunk_channels(seg_nt, target_chunk_mb=target_chunk_mb)
            if chunk_channels is None
            else max(1, int(chunk_channels))
        )
        ov = max(0, int(overlap_channels))
        whole = len(segments) == 1 and segments[0] == (0, nt)

        for c0, c1, r0, r1 in iter_channel_slices(nc, chunk_channels=chunk, overlap_channels=ov):
            for t0, t1 in segments:
                if cancel_cb is not None and bool(cancel_cb()):
                    raise RuntimeError("Pipeline run cancelled.")

                ta, tb = (None, None) if whole else (t0, t1)
                blk = _read_channel_block(source, layout, r0, r1, ta, tb)
                out_blk = np.asarray(self._run(blk), dtype=np.float32)
                if out_blk.shape != blk.shape:
                    raise ValueError(f"_run output shape mismatch: {out_blk.shape} != {blk.shape}")

                core = out_blk[:, (c0 - r0):(c1 - r0)]
                _write_channel_block(out_arr, layout, c0, c1, core, ta, tb)

            if progress_cb is not None:
                progress_cb(c1, nc)

    def _run_stream_block(
        self,
        source, out_arr, layout, nt, nc, *,
        block_t, block_c, overlap_t, overlap_c, taper_t=False,
        target_chunk_mb, progress_cb, cancel_cb, segments=None,
    ):
        """Block-mode inner loop (time outer, channel inner), per data segment."""
        for s0, s1 in (segments or [(0, nt)]):
            self._run_stream_block_segment(
                source, out_arr, layout, nt, nc, s0, s1,
                block_t=block_t, block_c=block_c, overlap_t=overlap_t,
                overlap_c=overlap_c, taper_t=taper_t,
                target_chunk_mb=target_chunk_mb,
                progress_cb=progress_cb, cancel_cb=cancel_cb,
            )

    def _run_stream_block_segment(
        self,
        source, out_arr, layout, nt, nc, s0, s1, *,
        block_t, block_c, overlap_t, overlap_c, taper_t,
        target_chunk_mb, progress_cb, cancel_cb,
    ):
        """Tile one contiguous segment ``[s0, s1)``; halos never leave it."""
        seg_nt = int(s1 - s0)
        bt_raw = int(block_t)
        if bt_raw < 1:
            bt, bc_auto = estimate_block_size(
                seg_nt, nc, target_block_mb=target_chunk_mb, dtype=np.float32
            )
        else:
            bt, bc_auto = bt_raw, nc
        bc = max(1, int(block_c)) if block_c is not None else bc_auto
        ot = max(0, int(overlap_t))
        oc = max(0, int(overlap_c))

        for t0, t1, c0, c1, rt0, rt1, rc0, rc1 in iter_tiles_2d(
            seg_nt, nc, tile_t=bt, tile_c=bc, halo_t=ot, halo_c=oc
        ):
            t0, t1, rt0, rt1 = t0 + s0, t1 + s0, rt0 + s0, rt1 + s0
            if cancel_cb is not None and bool(cancel_cb()):
                raise RuntimeError("Pipeline run cancelled.")

            blk = _read_tile(source, layout, rt0, rt1, rc0, rc1)
            if taper_t and ot > 0:
                hs = t0 - rt0
                he = rt1 - t1
                if hs > 0 or he > 0:
                    blk = blk * _temporal_taper_weights(rt1 - rt0, hs, he)
            out_blk = np.asarray(self._run(blk), dtype=np.float32)
            if out_blk.shape != blk.shape:
                raise ValueError(f"_run output shape mismatch: {out_blk.shape} != {blk.shape}")

            core = out_blk[(t0 - rt0):(t1 - rt0), (c0 - rc0):(c1 - rc0)]
            _write_tile(out_arr, layout, t0, t1, c0, c1, core)

            # Report progress per completed time-row (when channel loop wraps)
            if c1 == nc and progress_cb is not None:
                progress_cb(t1, nt)

    # ------------------------------------------------------------------
    # Unified entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        source: Any,
        save_path: str | Path | None = None,
        *,
        chunked: bool = False,
        **kwargs,
    ) -> np.ndarray | str:
        """Process data.

        Parameters
        ----------
        source:
            Input data.  ``np.ndarray`` for in-memory, or any array-like
            with ``shape`` for large/file-backed data.
        save_path:
            Output file path.  Only used when ``chunked=True``.
        chunked:
            ``False`` (default): single-pass via ``_run``.
            ``True``: chunked execution via ``_run_stream``.
        **kwargs:
            Forwarded to ``_run_stream`` when ``chunked=True``.
        """
        if chunked:
            return self._run_stream(source, save_path, **kwargs)
        return self._run(source)
