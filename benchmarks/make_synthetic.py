"""Write a synthetic multi-file DAS record (PRODML-style HDF5, [nt, nc] float32).

Content: band-limited noise plus sparse moving-source arrivals, so the display
and filters have something realistic to show. Optional timestamp gaps between
files (``--gap-after 3:120`` = 120 s outage after the 3rd file).

Used for smoke tests, the long-record memory test when real data are short,
and the discontinuity (gap) demonstrations.

    python benchmarks/make_synthetic.py OUT_DIR --files 12 --nc 2000 --fs 1000 --file-s 60
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import h5py
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--files", type=int, default=6)
    ap.add_argument("--nc", type=int, default=1000)
    ap.add_argument("--fs", type=float, default=500.0)
    ap.add_argument("--file-s", type=float, default=60.0, help="seconds per file")
    ap.add_argument("--dx", type=float, default=2.0)
    ap.add_argument("--gap-after", action="append", default=[], metavar="IDX:SECONDS",
                    help="insert a timestamp gap after file IDX (1-based); repeatable")
    ap.add_argument("--start", default="2026-01-01T00:00:00+00:00")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def make_block(rng, nt, nc, fs, t_abs0):
    x = rng.standard_normal((nt, nc)).astype(np.float32)
    # smooth in time a little (crude low-pass) so it is not pure white noise
    x = (x + np.roll(x, 1, axis=0) + np.roll(x, 2, axis=0)) / 3.0
    t = t_abs0 + np.arange(nt) / fs
    ch = np.arange(nc)
    # a few linear-moveout arrivals per file
    for _ in range(4):
        tc = rng.uniform(t[0], t[-1])
        c0 = rng.uniform(0, nc)
        v = rng.choice([-1, 1]) * rng.uniform(0.2, 1.0) * nc  # channels per second
        f0 = rng.uniform(5, min(40, 0.2 * fs))
        tau = (t[:, None] - tc) - (ch[None, :] - c0) / v
        x += (6.0 * np.exp(-(tau * f0) ** 2) * np.cos(2 * np.pi * f0 * tau)).astype(np.float32)
    return x


def main():
    a = parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    gaps = {}
    for g in a.gap_after:
        i, s = g.split(":")
        gaps[int(i)] = float(s)
    rng = np.random.default_rng(a.seed)
    t_start = dt.datetime.fromisoformat(a.start)
    nt = int(round(a.file_s * a.fs))
    t = t_start
    for k in range(1, a.files + 1):
        name = f"SYN_{a.nc}ch_{int(a.fs)}Hz_{t.strftime('%Y%m%dT%H%M%S')}.h5"
        x = make_block(rng, nt, a.nc, a.fs, t.timestamp())
        with h5py.File(out / name, "w") as h:
            acq = h.create_group("Acquisition")
            acq.attrs["SpatialSamplingInterval"] = a.dx
            acq.attrs["SpatialSamplingIntervalUnit"] = "m"
            raw = acq.create_group("Raw[0]")
            raw.attrs["OutputDataRate"] = a.fs
            raw.attrs["PartStartTime"] = t.isoformat()
            d = raw.create_dataset("RawData", data=x, chunks=(min(nt, 1024), min(a.nc, 1024)))
            d.attrs["Dimensions"] = np.bytes_("time,distance")
        print(f"wrote {name}  ({x.nbytes / 1024**2:.0f} MB)", flush=True)
        t = t + dt.timedelta(seconds=a.file_s + gaps.get(k, 0.0))


if __name__ == "__main__":
    main()
