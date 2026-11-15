"""Tests for filark.explore.overview.compute_overview against brute force."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from filark.explore.overview import compute_overview, default_bins, estimate_overview_bytes, file_range_at
from filark.io.fileset import TapeFileSet

FS = 200.0
T0 = 1_700_000_000.0


class MemTape:
    def __init__(self, x_nt_nc, start_time=None, dims="nt_nc"):
        self._x = np.asarray(x_nt_nc, dtype=np.float32)
        if dims == "nc_nt":
            self._x = np.ascontiguousarray(self._x.T)
        self.fs, self.dx, self.dx_unit, self.gauge_length = FS, 1.0, "m", None
        self.dims, self.ndim, self.start_time = dims, 2, start_time

    shape = property(lambda self: self._x.shape)
    dtype = property(lambda self: self._x.dtype)

    def __getitem__(self, k):
        return self._x[k]

    def close(self):
        pass


def _data(nt, nc, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((nt, nc))
    x[nt // 3: nt // 3 + 50, nc // 2] += 20.0  # an "event"
    return x.astype(np.float32)


def _brute(x, col_edges, row_edges, stat):
    out = np.empty((row_edges.size - 1, col_edges.size - 1))
    for r in range(row_edges.size - 1):
        for j in range(col_edges.size - 1):
            blk = x[col_edges[j]:col_edges[j + 1], row_edges[r]:row_edges[r + 1]].astype(np.float64)
            if stat == "mean":
                out[r, j] = np.mean(blk)
            elif stat == "rms":
                out[r, j] = np.sqrt(np.mean(blk ** 2))
            else:
                out[r, j] = np.max(np.abs(blk))
    return out


@pytest.mark.parametrize("stat", ["mean", "rms", "maxabs"])
@pytest.mark.parametrize("dims", ["nt_nc", "nc_nt"])
def test_matches_brute_force_with_channel_binning(stat, dims):
    x = _data(5000, 37)
    tape = MemTape(x, dims=dims)
    ov = compute_overview(tape, 300, 4700, time_bin=45, channel_bin=4, stat=stat, target_chunk_mb=0.05)
    assert ov.image.shape == (10, 98)  # 37 ch / 4 -> 10 rows; 4400 / 45 -> 98 cols (last partial)
    assert ov.col_edges[1] - ov.col_edges[0] == 45 and ov.col_edges[-1] == 4700
    ref = _brute(x, ov.col_edges, ov.row_edges, stat)  # edges are absolute indices
    np.testing.assert_allclose(ov.image, ref, rtol=1e-5)


def test_unit_bins_keep_every_sample_and_event_located():
    x = _data(400, 8)
    ov = compute_overview(MemTape(x), 0, 400, stat="maxabs")
    assert ov.image.shape == (8, 400)
    r, j = np.unravel_index(np.nanargmax(ov.image), ov.image.shape)
    assert r == 4 and 133 <= j < 183


def test_filter_runs_at_full_rate_and_resets_per_segment():
    n = 3000
    a, b = _data(n, 6, 1), _data(n, 6, 2)
    tapes = [MemTape(a, T0), MemTape(b, T0 + n / FS + 120.0)]
    fsx = TapeFileSet(tapes)
    g = fsx.gaps[0]
    sos = signal.butter(4, [10, 40], btype="bandpass", fs=FS, output="sos")

    ov = compute_overview(fsx, 0, fsx.shape[0], band=(10, 40), stat="maxabs", target_chunk_mb=0.02)

    full = np.asarray(fsx[:, :], dtype=np.float64)
    ref = np.full_like(full, np.nan)
    for s0, s1 in fsx.segments:
        seg = full[s0:s1]
        zi = signal.sosfilt_zi(sos)[:, :, None] * seg[0][None, None, :]
        ref[s0:s1], _ = signal.sosfilt(sos, seg, axis=0, zi=zi)
    np.testing.assert_allclose(ov.image.T[: g.start], np.abs(ref[: g.start]), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(ov.image.T[g.stop:], np.abs(ref[g.stop:]), rtol=1e-4, atol=1e-6)
    assert np.all(np.isnan(ov.image[:, g.start:g.stop]))
    assert ov.gaps == [(g.start, g.stop, pytest.approx(120.0))]
    assert ov.start_time == pytest.approx(T0)


def test_cancel_and_helpers():
    n = 1000
    fsx = TapeFileSet([MemTape(_data(n, 4), T0), MemTape(_data(n, 4), T0 + n / FS + 60)])
    with pytest.raises(RuntimeError):
        compute_overview(fsx, 0, fsx.shape[0], cancel_cb=lambda: True)
    g = fsx.gaps[0]
    assert file_range_at(fsx, 10) == (0, n)
    assert file_range_at(fsx, g.stop + 5) == (g.stop, g.stop + n)
    assert file_range_at(fsx, g.start + 1) == (g.stop, g.stop + n)  # inside the separator
    assert estimate_overview_bytes(fsx, 0, fsx.shape[0]) == 2 * n * 4 * 4
    assert default_bins(1920, 1080) == (1, 1)
    assert default_bins(60_000, 2500) == (32, 3)
    assert default_bins(100, 10) == (1, 1)
