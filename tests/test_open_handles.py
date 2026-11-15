"""TapeFileSet keeps at most ``max_open_files`` HDF5 handles open.

Each open HDF5 dataset holds its own chunk cache, so an unbounded number of
open files makes memory grow with the number of files browsed.
"""

from __future__ import annotations

import threading

import h5py
import numpy as np
import pytest

from filark.io.fileset import TapeFileSet

NT, NC, FS = 200, 8, 100.0


@pytest.fixture()
def h5_files(tmp_path):
    paths = []
    for k in range(10):
        p = tmp_path / f"f{k:02d}.h5"
        with h5py.File(p, "w") as h:
            raw = h.create_group("Acquisition").create_group("Raw[0]")
            raw.attrs["OutputDataRate"] = FS
            raw.attrs["PartStartTime"] = f"2026-01-01T00:{k:02d}:00+00:00"
            # contiguous 1-minute files with 2 s of data each -> every file is its own segment
            d = raw.create_dataset("RawData", data=np.full((NT, NC), k, dtype=np.float32), chunks=(50, NC))
            d.attrs["Dimensions"] = np.bytes_("time,distance")
        paths.append(p)
    return paths


def _n_open(fileset):
    return sum(bool(getattr(t, "is_open", False)) for t in fileset.files)


def test_no_handles_open_after_construction(h5_files):
    fsx = TapeFileSet(h5_files, gap_policy="ignore")
    assert _n_open(fsx) == 0
    assert fsx.shape == (10 * NT, NC)


def test_handle_budget_is_respected_and_data_is_correct(h5_files):
    fsx = TapeFileSet(h5_files, gap_policy="ignore", max_open_files=3)
    for k in range(10):
        block = fsx[k * NT:(k + 1) * NT, :]
        assert np.all(block == k)
        assert _n_open(fsx) <= 3
    # revisit an evicted file: reopened transparently
    assert np.all(fsx[0:5, :] == 0)
    assert _n_open(fsx) <= 3
    # a read spanning several files
    span = fsx[NT - 3:3 * NT + 3, :]
    assert span.shape == (2 * NT + 6, NC)
    assert np.all(span[:3] == 0) and np.all(span[-3:] == 3)


def test_concurrent_reads_never_hit_a_released_handle(h5_files):
    fsx = TapeFileSet(h5_files, gap_policy="ignore", max_open_files=2)
    errors = []

    def worker(seed):
        rng = np.random.default_rng(seed)
        try:
            for _ in range(200):
                k = int(rng.integers(0, 10))
                assert np.all(fsx[k * NT + 10:k * NT + 20, :] == k)
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(s,)) for s in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert _n_open(fsx) <= 2


def test_close_releases_everything(h5_files):
    fsx = TapeFileSet(h5_files, gap_policy="ignore")
    fsx[:, :]
    fsx.close()
    assert _n_open(fsx) == 0
