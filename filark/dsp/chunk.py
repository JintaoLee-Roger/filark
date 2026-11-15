# Copyright (c) 2026 Jintao Li. 
# Zhejiang University (ZJU).
# 
# Licensed under the MIT License.

"""Chunking utilities for large nt_nc arrays and overlap-aware operators."""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator
from typing import Literal

import numpy as np

from .base import from_nt_nc, to_nt_nc

Layout = Literal["nt_nc", "nc_nt"]

__all__ = [
    "estimate_chunk_channels",
    "estimate_block_size",
    "iter_channel_slices",
    "iter_tiles_2d",
    "apply_channel_chunks",
    "apply_tiled_overlap",
    "apply_block_chunks",
    "make_chunked",
]


def _temporal_taper_weights(blk_t: int, halo_start: int, halo_end: int) -> np.ndarray:
    """Cosine taper weights for a block of ``blk_t`` time samples.

    Returns a column vector of shape ``(blk_t, 1)`` ready to broadcast over
    a ``[nt, nc]`` block.  The halo regions ramp smoothly from 0 (far edge)
    to 1 (adjacent to core).  Data-boundary edges (``halo_start/end == 0``)
    are left at 1.0 so no signal is lost at the true recording endpoints.

    This is the standard windowed overlap-save technique for reducing spectral
    leakage when a global frequency-domain operator (e.g. f-k filter) is
    applied to a sub-block of a longer recording.
    """
    w = np.ones(blk_t, dtype=np.float32)
    if halo_start > 0:
        j = np.arange(halo_start, dtype=np.float32)
        w[:halo_start] = 0.5 * (1.0 - np.cos(np.pi * j / halo_start))
        # j=0 → w=0 (far edge); j→halo_start → w→1 (adjacent to core)
    if halo_end > 0:
        j = np.arange(halo_end, dtype=np.float32)
        ramp_up = 0.5 * (1.0 - np.cos(np.pi * j / halo_end))
        w[-halo_end:] = ramp_up[::-1]
        # w[-1]=0 (far edge); w[-halo_end]→1 (adjacent to core)
    return w[:, np.newaxis]  # (blk_t, 1)


def estimate_chunk_channels(
    nt: int,
    *,
    target_chunk_mb: float = 128.0,
    dtype=np.float32,
    min_chunk_channels: int = 1,
) -> int:
    """Estimate a channel-chunk size for arrays shaped [nt, nc]."""
    nt = int(nt)
    if nt < 1:
        raise ValueError(f"nt must be >= 1, got {nt}")
    target_bytes = max(8 * 1024 * 1024, int(float(target_chunk_mb) * 1024 * 1024))
    bytes_per_channel = max(1, nt * int(np.dtype(dtype).itemsize))
    return max(int(min_chunk_channels), target_bytes // bytes_per_channel)


def estimate_block_size(
    nt: int,
    nc: int,
    *,
    target_block_mb: float = 128.0,
    dtype=np.float32,
    min_block_t: int = 1,
    min_block_c: int = 1,
) -> tuple[int, int]:
    """Estimate 2-D block dimensions ``(block_t, block_c)`` for block chunking.

    Produces roughly square blocks (in samples) that fit within the memory
    budget.  Both dimensions are capped at the actual array size.

    Returns
    -------
    (block_t, block_c)
    """
    nt = int(nt)
    nc = int(nc)
    target_bytes = max(8 * 1024 * 1024, int(float(target_block_mb) * 1024 * 1024))
    item_size = int(np.dtype(dtype).itemsize)
    total_elems = max(1, target_bytes // item_size)
    side = max(1, int(total_elems ** 0.5))
    block_t = max(int(min_block_t), min(nt, side))
    block_c = max(int(min_block_c), min(nc, total_elems // block_t))
    return block_t, block_c


def iter_channel_slices(
    nc: int,
    *,
    chunk_channels: int,
    overlap_channels: int = 0,
) -> Iterator[tuple[int, int, int, int]]:
    """
    Yield channel chunk slices as (c0, c1, r0, r1), where:
    - [c0:c1] is the write core
    - [r0:r1] is the read region (core + overlap halo)
    """
    nc = int(nc)
    chunk = int(chunk_channels)
    ov = int(overlap_channels)
    if nc < 1:
        raise ValueError(f"nc must be >= 1, got {nc}")
    if chunk < 1:
        raise ValueError(f"chunk_channels must be >= 1, got {chunk_channels}")
    if ov < 0:
        raise ValueError(f"overlap_channels must be >= 0, got {overlap_channels}")

    for c0 in range(0, nc, chunk):
        c1 = min(nc, c0 + chunk)
        r0 = max(0, c0 - ov)
        r1 = min(nc, c1 + ov)
        yield c0, c1, r0, r1


def iter_tiles_2d(
    nt: int,
    nc: int,
    *,
    tile_t: int,
    tile_c: int,
    halo_t: int = 0,
    halo_c: int = 0,
) -> Iterator[tuple[int, int, int, int, int, int, int, int]]:
    """
    Yield 2-D tile geometry:
    (t0, t1, c0, c1, rt0, rt1, rc0, rc1)
    where [t0:t1, c0:c1] is write core and [rt0:rt1, rc0:rc1] is read region.
    """
    nt = int(nt)
    nc = int(nc)
    tt = int(tile_t)
    tc = int(tile_c)
    ht = int(halo_t)
    hc = int(halo_c)
    if nt < 1 or nc < 1:
        raise ValueError(f"nt/nc must be >= 1, got nt={nt}, nc={nc}")
    if tt < 1 or tc < 1:
        raise ValueError(f"tile_t/tile_c must be >= 1, got tile_t={tile_t}, tile_c={tile_c}")
    if ht < 0 or hc < 0:
        raise ValueError(f"halo_t/halo_c must be >= 0, got halo_t={halo_t}, halo_c={halo_c}")

    for t0 in range(0, nt, tt):
        t1 = min(nt, t0 + tt)
        rt0 = max(0, t0 - ht)
        rt1 = min(nt, t1 + ht)
        for c0 in range(0, nc, tc):
            c1 = min(nc, c0 + tc)
            rc0 = max(0, c0 - hc)
            rc1 = min(nc, c1 + hc)
            yield t0, t1, c0, c1, rt0, rt1, rc0, rc1


def apply_channel_chunks(
    data,
    op: Callable[[np.ndarray], np.ndarray],
    *,
    chunk_channels: int,
    overlap_channels: int = 0,
    layout: Layout = "nt_nc",
    dtype=np.float32,
    progress_cb: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """
    Apply `op` to channel chunks.

    Parameters
    ----------
    data:
        Input 2-D array in `layout`.
    op:
        Callable receiving an `nt_nc` block with shape [nt, r_channels]
        and returning the same shape.
    chunk_channels:
        Core write width for each chunk.
    overlap_channels:
        Read halo on both channel sides.
    """
    x = to_nt_nc(data, layout=layout, dtype=dtype)
    nt, nc = x.shape
    out = np.empty((nt, nc), dtype=dtype)

    done = 0
    for c0, c1, r0, r1 in iter_channel_slices(
        nc,
        chunk_channels=int(chunk_channels),
        overlap_channels=int(overlap_channels),
    ):
        blk = x[:, r0:r1]
        den = np.asarray(op(blk), dtype=dtype)
        if den.shape != blk.shape:
            raise ValueError(f"Chunk op output shape mismatch: {den.shape} != {blk.shape}")
        out[:, c0:c1] = den[:, (c0 - r0):(c1 - r0)]
        done = c1
        if progress_cb is not None:
            progress_cb(done, nc)

    return from_nt_nc(out, layout=layout, dtype=dtype)


def apply_block_chunks(
    data,
    op: Callable[[np.ndarray], np.ndarray],
    *,
    block_t: int,
    block_c: int,
    overlap_t: int = 0,
    overlap_c: int = 0,
    taper_t: bool = False,
    layout: Layout = "nt_nc",
    dtype=np.float32,
    progress_cb: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """
    Apply ``op`` to 2-D blocks with time as the outer loop.

    Parameters
    ----------
    data:
        Input 2-D array in ``layout``.
    op:
        Callable receiving an ``nt_nc`` block with shape [rt, rc]
        (core + halos) and returning the same shape.
    block_t:
        Core write height (time samples) per block.
    block_c:
        Core write width (channels) per block.
    overlap_t:
        Read halo on both temporal sides.
    overlap_c:
        Read halo on both channel sides.
    taper_t:
        When ``True`` and ``overlap_t > 0``, apply a cosine window to the
        temporal halo before passing the block to ``op``.  Reduces spectral
        leakage for global frequency-domain operators (e.g. f-k filter).
        Data-boundary edges are never tapered.

    Notes
    -----
    Iterates time first (outer), channels second (inner).  This matches
    the natural streaming order of DAS data (data arrives time-segment by
    time-segment) and keeps hot working memory to ``(block_t + 2*overlap_t)
    × (block_c + 2*overlap_c)`` samples at a time.
    """
    x = to_nt_nc(data, layout=layout, dtype=dtype)
    nt, nc = x.shape
    out = np.empty((nt, nc), dtype=dtype)

    done_t = 0
    for t0, t1, c0, c1, rt0, rt1, rc0, rc1 in iter_tiles_2d(
        nt,
        nc,
        tile_t=int(block_t),
        tile_c=int(block_c),
        halo_t=int(overlap_t),
        halo_c=int(overlap_c),
    ):
        blk = x[rt0:rt1, rc0:rc1]
        if taper_t and overlap_t > 0:
            hs = t0 - rt0   # actual start halo (0 at data boundary → no taper)
            he = rt1 - t1   # actual end halo   (0 at data boundary → no taper)
            if hs > 0 or he > 0:
                blk = blk * _temporal_taper_weights(rt1 - rt0, hs, he)
        res = np.asarray(op(blk), dtype=dtype)
        if res.shape != blk.shape:
            raise ValueError(f"Block op output shape mismatch: {res.shape} != {blk.shape}")
        out[t0:t1, c0:c1] = res[(t0 - rt0):(t1 - rt0), (c0 - rc0):(c1 - rc0)]
        if c1 == nc:
            done_t = t1
            if progress_cb is not None:
                progress_cb(done_t, nt)

    return from_nt_nc(out, layout=layout, dtype=dtype)


def make_chunked(fn, *, default_overlap: int = 0, get_overlap=None):
    """
    Factory that produces a chunked variant of *fn*.

    **Channel mode** (default): full time range, chunks along channels only.

    **Block mode**: 2-D blocks with time as the outer loop — activated by
    passing ``block_t`` (and optionally ``block_c``) at call time.  This
    matches the natural streaming order of DAS data and limits peak memory to
    one block at a time.

    Extra keyword args added to the wrapper
    ----------------------------------------
    chunk_channels : int or None
        [channel mode] Channels per chunk.  Auto-estimated from
        *target_chunk_mb* when ``None``.
    target_chunk_mb : float
        Target chunk size in MiB for auto-estimation (default 128).
    overlap_channels : int or None
        [channel mode] Read halo on each channel side.  Falls back to
        *default_overlap* (or ``get_overlap(kwargs)``) when ``None``.
    block_t : int or None
        [block mode] Time samples per block.  Activates block mode when set.
        Auto-estimated from *target_chunk_mb* when ``None`` in block mode —
        pass ``block_t=0`` as a sentinel to auto-estimate.
    block_c : int or None
        [block mode] Channels per block.  Defaults to full channel width when
        ``None``.
    overlap_t : int
        [block mode] Read halo on each temporal side (default 0).
    overlap_c : int or None
        [block mode] Read halo on each channel side.  Falls back to
        *default_overlap* (or ``get_overlap(kwargs)``) when ``None``.
    layout : "nt_nc" | "nc_nt"
        Input array layout (default "nt_nc").

    Parameters
    ----------
    fn:
        A function whose first positional argument is a 2D array and remaining
        arguments are keyword-only.
    default_overlap:
        Overlap used when per-call overlap is not supplied.
    get_overlap:
        Optional callable ``(kwargs) -> int`` to derive overlap from the
        wrapped function's kwargs.
    """

    @functools.wraps(fn)
    def _chunked(
        data,
        *args,
        # channel mode
        chunk_channels=None,
        target_chunk_mb: float = 128.0,
        overlap_channels=None,
        # block mode (activated when block_t is not None)
        block_t=None,
        block_c=None,
        overlap_t: int = 0,
        overlap_c=None,
        # common
        layout: Layout = "nt_nc",
        **kwargs,
    ):
        x = to_nt_nc(data, layout=layout, dtype=np.float32)
        nt, nc = x.shape

        if block_t is not None:
            # ---- Block mode: 2-D tiles, time outer loop ----
            bt_raw = int(block_t)
            if bt_raw < 1:
                bt, bc_auto = estimate_block_size(
                    nt, nc,
                    target_block_mb=float(target_chunk_mb),
                    dtype=np.float32,
                )
            else:
                bt = bt_raw
                bc_auto = nc
            bc = max(1, int(block_c)) if block_c is not None else bc_auto

            ot = max(0, int(overlap_t))
            if overlap_c is not None:
                oc = int(overlap_c)
            elif get_overlap is not None:
                oc = int(get_overlap(kwargs))
            else:
                oc = int(default_overlap)

            tt = bool(kwargs.pop("taper_t", False))

            y = apply_block_chunks(
                x,
                lambda b: fn(b, *args, **kwargs),
                block_t=bt,
                block_c=bc,
                overlap_t=ot,
                overlap_c=oc,
                taper_t=tt,
                layout="nt_nc",
                dtype=np.float32,
            )
        else:
            # ---- Channel mode (default): full time, channel chunks ----
            chunk = (
                estimate_chunk_channels(nt, target_chunk_mb=float(target_chunk_mb))
                if chunk_channels is None
                else max(1, int(chunk_channels))
            )
            if overlap_channels is not None:
                ov = int(overlap_channels)
            elif get_overlap is not None:
                ov = int(get_overlap(kwargs))
            else:
                ov = int(default_overlap)

            y = apply_channel_chunks(
                x,
                lambda b: fn(b, *args, **kwargs),
                chunk_channels=chunk,
                overlap_channels=ov,
                layout="nt_nc",
                dtype=np.float32,
            )

        return from_nt_nc(y, layout=layout)

    _chunked.__name__ = fn.__name__ + "_chunked"
    _chunked.__qualname__ = fn.__qualname__ + "_chunked"
    _chunked.__doc__ = (
        (fn.__doc__ or "")
        + "\n\nChunked variant — channel mode extra kwargs: "
        "chunk_channels, target_chunk_mb, overlap_channels, layout. "
        "Block mode (time-outer) activated by passing block_t: "
        "block_t, block_c, overlap_t, overlap_c, target_chunk_mb, layout."
    )
    return _chunked


def apply_tiled_overlap(
    data,
    op: Callable[[np.ndarray], np.ndarray],
    *,
    tile_t: int,
    tile_c: int,
    halo_t: int = 0,
    halo_c: int = 0,
    layout: Layout = "nt_nc",
    dtype=np.float32,
) -> np.ndarray:
    """
    Apply `op` to 2-D tiles with halo and write-back of non-overlap core.

    `op` receives and returns a full context tile in `nt_nc` layout.
    """
    x = to_nt_nc(data, layout=layout, dtype=dtype)
    nt, nc = x.shape
    out = np.empty((nt, nc), dtype=dtype)

    for t0, t1, c0, c1, rt0, rt1, rc0, rc1 in iter_tiles_2d(
        nt,
        nc,
        tile_t=int(tile_t),
        tile_c=int(tile_c),
        halo_t=int(halo_t),
        halo_c=int(halo_c),
    ):
        blk = x[rt0:rt1, rc0:rc1]
        den = np.asarray(op(blk), dtype=dtype)
        if den.shape != blk.shape:
            raise ValueError(f"Tile op output shape mismatch: {den.shape} != {blk.shape}")
        out[t0:t1, c0:c1] = den[(t0 - rt0):(t1 - rt0), (c0 - rc0):(c1 - rc0)]

    return from_nt_nc(out, layout=layout, dtype=dtype)
