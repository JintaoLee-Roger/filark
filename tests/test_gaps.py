"""Targeted tests for discontinuous (timestamp-gapped) multi-file recordings.

These cover the three layers that must agree on discontinuities:

* ``TapeFileSet`` — gap/overlap detection and the logical timeline;
* ``StreamingSource`` — stateful (IIR) processing reset at segment starts;
* ``BasePipeline`` — chunked execution that never spans a gap.

The reference ("ground truth") for every stateful case is to process each
contiguous segment independently from a zero initial state, with gap
samples left at the fill value.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from filark.io.fileset import TapeFileSet
from filark.io.stream_processing import RealtimeDisplaySOSFilter
from filark.io.streaming import StreamingSource
from filark.pipeline.base import BasePipeline

FS = 100.0
NC = 4
T0 = 1_700_000_000.0  # arbitrary UTC epoch seconds


class MemTape:
    """Minimal in-memory Tape used to build synthetic file sets."""

    def __init__(self, data_nt_nc: np.ndarray, start_time, *, fs=FS, dims="nt_nc"):
        self._x = np.asarray(data_nt_nc, dtype=np.float32)
        if dims == "nc_nt":
            self._x = np.ascontiguousarray(self._x.T)
        self.fs = float(fs)
        self.dx = 1.0
        self.dx_unit = "m"
        self.gauge_length = None
        self.dims = dims
        self.ndim = 2
        self.start_time = start_time

    @property
    def shape(self):
        return self._x.shape

    @property
    def dtype(self):
        return self._x.dtype

    def __getitem__(self, key):
        return self._x[key]

    def close(self):
        pass


def _signal(n, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    base = np.sin(2 * np.pi * 7.0 * t)[:, None] * np.arange(1, NC + 1)[None, :]
    return (base + 0.3 * rng.standard_normal((n, NC))).astype(np.float32)


def _three_files(gap_s=2.5, *, jitter_s=0.0, dims="nt_nc"):
    """File A, B contiguous (with optional jitter), then a gap, then file C."""
    n = 300
    a, b, c = _signal(n, 1), _signal(n, 2), _signal(n, 3)
    ta = T0
    tb = ta + n / FS + jitter_s
    tc = tb + n / FS + gap_s
    tapes = [MemTape(a, ta, dims=dims), MemTape(b, tb, dims=dims), MemTape(c, tc, dims=dims)]
    return tapes, (a, b, c), n


SOS = signal.butter(4, [5.0, 15.0], btype="bandpass", fs=FS, output="sos")


def _reference_filter(fileset: TapeFileSet) -> np.ndarray:
    """Per-segment causal filtering from zero state; gaps stay zero. [nt, nc]"""
    full = np.asarray(fileset[:, :], dtype=np.float64)
    out = np.zeros_like(full)
    for s0, s1 in fileset.segments:
        out[s0:s1] = signal.sosfilt(SOS, full[s0:s1], axis=0)
    return out


# ---------------------------------------------------------------------------
# TapeFileSet: detection and logical timeline
# ---------------------------------------------------------------------------


def test_jitter_within_tolerance_is_contiguous():
    tapes, (a, b, _), n = _three_files(jitter_s=0.004)  # 0.4 samples, < 10 ms
    fsx = TapeFileSet(tapes[:2])
    assert fsx.gaps == []
    assert fsx.segments == [(0, 2 * n)]
    np.testing.assert_array_equal(fsx[:, :], np.concatenate([a, b]))
    legacy = TapeFileSet(tapes[:2], gap_policy="ignore")
    np.testing.assert_array_equal(fsx[:, :], legacy[:, :])


def test_gap_is_inserted_as_fixed_separator():
    gap_s = 2.5  # 250 samples missing -> 100-sample separator
    tapes, (a, b, c), n = _three_files(gap_s=gap_s)
    fsx = TapeFileSet(tapes)
    n_gap = 100

    assert len(fsx.gaps) == 1
    g = fsx.gaps[0]
    assert (g.kind, g.start, g.stop) == ("gap", 2 * n, 2 * n + n_gap)
    assert g.seconds == pytest.approx(gap_s)
    assert fsx.segments == [(0, 2 * n), (2 * n + n_gap, 3 * n + n_gap)]
    assert fsx.shape == (3 * n + n_gap, NC)

    full = fsx[:, :]
    np.testing.assert_array_equal(full[: 2 * n], np.concatenate([a, b]))
    assert np.all(full[2 * n: 2 * n + n_gap] == 0)
    np.testing.assert_array_equal(full[2 * n + n_gap:], c)

    # Absolute time is recovered through the piecewise mapping.
    assert fsx.sample_to_time(g.stop) == pytest.approx(tapes[2].start_time)
    assert fsx.sample_to_time(2 * n - 1) == pytest.approx(tapes[1].start_time + (n - 1) / FS)
    # Slices that straddle a gap edge, scalar and strided access.
    np.testing.assert_array_equal(fsx[2 * n - 3: 2 * n + 3, 1], np.r_[b[-3:, 1], [0, 0, 0]])
    assert fsx[2 * n + 10, 2] == 0
    np.testing.assert_array_equal(fsx[g.stop, :], c[0])
    np.testing.assert_array_equal(fsx[::7, :], full[::7])
    assert fsx.is_gap_index(2 * n) and not fsx.is_gap_index(g.stop)


def test_nc_nt_layout_and_scalar_channel_key():
    tapes, (a, b, c), n = _three_files(dims="nc_nt")
    fsx = TapeFileSet(tapes)
    g = fsx.gaps[0]
    assert fsx.shape == (NC, 3 * n + g.n_samples)
    row = fsx[2, :]
    assert row.shape == (fsx.shape[1],)
    np.testing.assert_array_equal(row[g.stop:], c[:, 2])
    assert np.all(row[g.start:g.stop] == 0)


def test_overlap_is_trimmed_and_marks_a_boundary():
    n = 300
    a, b = _signal(n, 1), _signal(n, 2)
    overlap = 40
    tapes = [MemTape(a, T0), MemTape(b, T0 + (n - overlap) / FS)]
    fsx = TapeFileSet(tapes)
    assert [g.kind for g in fsx.gaps] == ["overlap"]
    assert fsx.shape == (2 * n - overlap, NC)
    np.testing.assert_array_equal(fsx[n:, :], b[overlap:])
    assert fsx.segments == [(0, n), (n, 2 * n - overlap)]


def test_files_are_ordered_by_start_time():
    tapes, (a, b, c), n = _three_files()
    fsx = TapeFileSet([tapes[2], tapes[0], tapes[1]])
    np.testing.assert_array_equal(fsx[:n, :], a)
    assert fsx.start_time == tapes[0].start_time


def test_raise_policy_and_fs_mismatch():
    tapes, _, _ = _three_files()
    with pytest.raises(ValueError, match="gap"):
        TapeFileSet(tapes, gap_policy="raise")
    bad = MemTape(_signal(10, 9), T0 + 100, fs=FS * 2)
    with pytest.raises(ValueError, match="fs mismatch"):
        TapeFileSet([tapes[0], bad])


def test_missing_timestamps_fall_back_to_concatenation():
    tapes, (a, b, c), n = _three_files()
    tapes[1].start_time = None
    fsx = TapeFileSet(tapes)
    assert fsx.gaps == [] and fsx.shape == (3 * n, NC)


# ---------------------------------------------------------------------------
# StreamingSource: stateful processing across a gap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["preprocess", "postprocess"])
def test_iir_state_resets_after_gap_single_read(which):
    tapes, _, _ = _three_files()
    fsx = TapeFileSet(tapes)
    ref = _reference_filter(fsx)  # [nt, nc]
    nt = fsx.shape[0]

    src = StreamingSource(fsx, dims="nt_nc", **{which: RealtimeDisplaySOSFilter(SOS)})
    out = src.read(0, 0, NC, nt, raw=True)  # [nc, nt]
    np.testing.assert_allclose(out.T, ref, rtol=1e-5, atol=1e-5)


def test_iir_state_resets_with_incremental_blocks():
    """Sequential blocks with carried state (viewer scrolling) match the reference,
    including blocks that start exactly at the end of a gap."""
    tapes, _, n = _three_files()
    fsx = TapeFileSet(tapes)
    ref = _reference_filter(fsx)
    nt = fsx.shape[0]
    g = fsx.gaps[0]

    proc = RealtimeDisplaySOSFilter(SOS)
    src = StreamingSource(fsx, dims="nt_nc", postprocess=proc)
    # Block edges chosen so one block starts exactly at the gap end.
    edges = sorted({0, 137, g.start - 10, g.stop, g.stop + 55, nt})
    parts = [src.read(0, x0, NC, x1 - x0, raw=True) for x0, x1 in zip(edges[:-1], edges[1:])]
    out = np.concatenate(parts, axis=1)
    np.testing.assert_allclose(out.T, ref, rtol=1e-5, atol=1e-5)


def test_no_leakage_from_pre_gap_data():
    """Output after a gap must not depend on anything recorded before it."""
    tapes, (a, b, c), n = _three_files()
    spiky = [MemTape(a * 0 + 1e3, tapes[0].start_time), MemTape(b * 0 + 1e3, tapes[1].start_time), tapes[2]]
    out1 = StreamingSource(TapeFileSet(tapes), dims="nt_nc", postprocess=RealtimeDisplaySOSFilter(SOS))
    out2 = StreamingSource(TapeFileSet(spiky), dims="nt_nc", postprocess=RealtimeDisplaySOSFilter(SOS))
    g = TapeFileSet(tapes).gaps[0]
    nt = out1.shape[1]
    y1 = out1.read(0, 0, NC, nt, raw=True)
    y2 = out2.read(0, 0, NC, nt, raw=True)
    np.testing.assert_array_equal(y1[:, g.start:], y2[:, g.start:])
    assert np.all(y1[:, g.start:g.stop] == 0)


def test_continuous_source_is_unchanged():
    """No discontinuities: identical to processing the concatenated array once."""
    tapes, (a, b, _), n = _three_files(jitter_s=0.002)
    fsx = TapeFileSet(tapes[:2])
    x = np.concatenate([a, b]).astype(np.float64)
    ref = signal.sosfilt(SOS, x, axis=0)
    src = StreamingSource(fsx, dims="nt_nc", postprocess=RealtimeDisplaySOSFilter(SOS))
    out = np.concatenate([src.read(0, x0, NC, 100, raw=True) for x0 in range(0, 2 * n, 100)], axis=1)
    np.testing.assert_allclose(out.T, ref, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Pipeline: chunked execution never spans a gap
# ---------------------------------------------------------------------------


class _FilterPipeline(BasePipeline):
    def _run(self, data):  # data [nt, nc]
        return signal.sosfilt(SOS, data.astype(np.float64), axis=0).astype(np.float32)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(chunk_channels=2),
        dict(block_t=0),  # block mode, auto block size
    ],
)
def test_pipeline_processes_segments_independently(kwargs):
    tapes, _, _ = _three_files()
    fsx = TapeFileSet(tapes)
    ref = _reference_filter(fsx)
    out = _FilterPipeline()(fsx, chunked=True, **kwargs)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)
    g = fsx.gaps[0]
    assert np.all(out[g.start:g.stop] == 0)


def test_short_gap_keeps_true_length_and_custom_separator():
    tapes, _, n = _three_files(gap_s=0.3)  # 30 samples < 100
    assert TapeFileSet(tapes).gaps[0].n_samples == 30
    tapes, _, n = _three_files(gap_s=600.0)
    g = TapeFileSet(tapes, gap_samples=7).gaps[0]
    assert g.n_samples == 7 and g.seconds == pytest.approx(600.0)


def test_sample_time_mapping_roundtrip():
    gap_s = 3600.0
    tapes, _, n = _three_files(gap_s=gap_s)
    fsx = TapeFileSet(tapes)
    g = fsx.gaps[0]
    idx = np.array([0, 5, n, 2 * n - 1, g.stop, g.stop + 17, fsx.shape[0] - 1])
    t = fsx.sample_to_time(idx)
    expect = np.array([
        T0, T0 + 5 / FS, tapes[1].start_time, tapes[1].start_time + (n - 1) / FS,
        tapes[2].start_time, tapes[2].start_time + 17 / FS, tapes[2].start_time + (n - 1) / FS,
    ])
    np.testing.assert_allclose(t, expect, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(fsx.time_to_sample(t), idx)
    # time inside the outage lands inside the separator and is monotonic
    mid = fsx.time_to_sample(tapes[1].start_time + n / FS + gap_s / 2)
    assert g.start <= mid < g.stop
    tt = fsx.sample_to_time(np.arange(fsx.shape[0]))
    assert np.all(np.diff(tt) > 0)


def test_time_helpers_fall_back_without_mapping():
    from filark.utils.time import tape_sample_to_unix, tape_unix_to_sample
    tape = MemTape(_signal(50, 0), T0)
    assert tape_sample_to_unix(tape, 10) == pytest.approx(T0 + 10 / FS)
    assert tape_unix_to_sample(tape, T0 + 0.2) == 20
    tape.start_time = None
    assert tape_sample_to_unix(tape, 10) is None
