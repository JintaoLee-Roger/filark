"""Discontinuous recordings: stateful (IIR) processing across timestamp gaps.

Answers SE-7 / R2-3. For every case three results are compared:

* ``reference``  each contiguous segment filtered on its own from a zero
                 state (``scipy.signal.sosfilt``, float64); gap samples = 0
* ``filark``     FiLark as used: the viewer path (``StreamingSource`` read in
                 sequential blocks, display IIR filter) and the batch path
                 (``BasePipeline`` chunked, channel mode)
* ``carried``    files concatenated without gap handling
                 (``gap_policy="ignore"``): the IIR state runs across the outage

Reported: max |filark - reference| (relative to the signal RMS), and for the
carried case the error right after the gap and how long it takes to decay
below 1 % of the signal RMS.

Part A uses synthetic data (several gap lengths, incl. one shorter than the
separator). Part B uses a real record with a gap (``--real DIR``) and draws a
figure of the data around it.

    python benchmarks/bench_gaps.py --real /path/to/record_with_gap
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy import signal

import _common as C
from filark.dsp.temporal import build_iir_sos
from filark.io.fileset import TapeFileSet
from filark.io.stream_processing import RealtimeDisplaySOSFilter
from filark.io.streaming import StreamingSource
from filark.io.tapeio import ArrayTape
from filark.pipeline.base import BasePipeline


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", default=None, help="folder of a real multi-file record containing a gap")
    ap.add_argument("--pattern", default="*.h5")
    ap.add_argument("--channels", type=int, nargs=2, default=[5200, 5456], help="channel range [c0, c1)")
    ap.add_argument("--before", type=float, default=4.0, help="seconds shown before the gap")
    ap.add_argument("--after", type=float, default=8.0, help="seconds shown after the gap")
    ap.add_argument("--band", type=float, nargs=2, default=[1.0, 50.0], help="IIR bandpass (Hz)")
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--block", type=int, default=256, help="viewer read block (samples)")
    ap.add_argument("--clim", type=float, default=None, help="colour limit for the figure (default: 99th pct)")
    ap.add_argument("--out", default="benchmarks/results")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# processing paths
# ---------------------------------------------------------------------------

def viewer_path(fileset, sos, c0, c1, t0, t1, block):
    """Sequential block reads through StreamingSource (what the viewer does). -> [nt, nc]"""
    src = StreamingSource(fileset, dims="nt_nc", postprocess=RealtimeDisplaySOSFilter(sos))
    parts = []
    for x0 in range(t0, t1, block):
        w = min(block, t1 - x0)
        parts.append(src.read(c0, x0, c1 - c0, w, raw=True))
    return np.concatenate(parts, axis=1).T.astype(np.float64)


class _SOSPipeline(BasePipeline):
    def __init__(self, sos):
        super().__init__()
        self.sos = sos

    def _run(self, data):
        return signal.sosfilt(self.sos, np.asarray(data, dtype=np.float64), axis=0).astype(np.float32)


def batch_path(fileset, sos):
    """Chunked batch execution (channel mode: each chunk spans a whole segment in
    time, so the recursive filter runs continuously within a segment). -> [nt, nc]"""
    return np.asarray(_SOSPipeline(sos)(fileset, chunked=True, chunk_channels=8), dtype=np.float64)


def reference(raw, segments, sos):
    """Per-segment filtering from a zero state; samples outside segments stay 0."""
    out = np.zeros_like(raw, dtype=np.float64)
    for a, b in segments:
        out[a:b] = signal.sosfilt(sos, raw[a:b].astype(np.float64), axis=0)
    return out


def carried_error(carried_post, ref_post, fs, sig_rms):
    """Error of the no-reset result on the post-gap segment."""
    err = np.max(np.abs(carried_post - ref_post), axis=1)  # worst channel per sample
    above = np.nonzero(err > 0.01 * sig_rms)[0]
    settle_s = (above[-1] + 1) / fs if above.size else 0.0
    return {
        "carried_max_abs_err": float(err.max()),
        "carried_max_rel_err": float(err.max() / sig_rms),
        "carried_err_first_100ms_rel": float(err[: max(1, int(0.1 * fs))].max() / sig_rms),
        "carried_settle_to_1pct_s": float(settle_s),
        "carried_settle_reached": bool(above.size == 0 or above[-1] + 1 < err.size),
    }


# ---------------------------------------------------------------------------
# Part A: synthetic
# ---------------------------------------------------------------------------

def synthetic_case(gap_s, *, fs=1000.0, nc=32, seg_s=20.0, band=(1.0, 50.0), order=4, block=256, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seg_s * fs)
    t = np.arange(n) / fs

    def seg(k):
        # strong low-frequency drift + in-band arrivals + noise, different level per file
        x = (40.0 * (k + 1)) * np.sin(2 * np.pi * 0.3 * t + k)[:, None]
        x = x + np.sin(2 * np.pi * 12.0 * t)[:, None] * rng.uniform(0.5, 2.0, nc)[None, :]
        x = x + 0.5 * rng.standard_normal((n, nc))
        return x.astype(np.float32)

    t0 = 1.7e9
    starts = [t0, t0 + seg_s + gap_s, t0 + 2 * seg_s + 2 * gap_s]
    datas = [seg(k) for k in range(3)]
    tapes = lambda: [ArrayTape(d, dims="nt_nc", fs=fs, start_time=s) for d, s in zip(datas, starts)]

    sos = build_iir_sos(fs=fs, kind="bandpass", fmin=band[0], fmax=band[1], order=order)
    fsx = TapeFileSet(tapes(), gap_policy="fill")
    nt = fsx.shape[0]
    raw = np.asarray(fsx[:, :], dtype=np.float64)
    ref = reference(raw, fsx.segments, sos)
    sig_rms = float(np.sqrt(np.mean(ref[fsx.segments[0][0]:fsx.segments[0][1]] ** 2)))

    view = viewer_path(fsx, sos, 0, nc, 0, nt, block)
    batch = batch_path(fsx, sos)
    in_seg = np.zeros(nt, bool)
    for a, b in fsx.segments:
        in_seg[a:b] = True

    res = {
        "gap_s": gap_s, "fs": fs, "n_channels": nc, "segment_s": seg_s,
        "n_gaps": len(fsx.gaps), "separator_samples": [g.n_samples for g in fsx.gaps],
        "signal_rms": sig_rms,
        "viewer_max_rel_err": float(np.abs(view - ref)[in_seg].max() / sig_rms),
        "batch_max_rel_err": float(np.abs(batch - ref)[in_seg].max() / sig_rms),
        "viewer_gap_all_zero": bool(np.all(view[~in_seg] == 0)),
        "batch_gap_all_zero": bool(np.all(batch[~in_seg] == 0)),
    }
    # carried state: no separator, IIR runs straight through
    lg = TapeFileSet(tapes(), gap_policy="ignore")
    carried = viewer_path(lg, sos, 0, nc, 0, lg.shape[0], block)
    b0, b1 = fsx.segments[1]
    res.update(carried_error(carried[n:2 * n], ref[b0:b1], fs, sig_rms))
    return res


# ---------------------------------------------------------------------------
# Part B: real record
# ---------------------------------------------------------------------------

def real_case(args, run_dir):
    files = C.list_files(args.real, args.pattern)
    full = C.open_fileset(files)
    if not full.gaps:
        raise SystemExit("no gap found in --real record")
    g = full.gaps[0]
    pair = [files[g.before_file], files[g.after_file]]
    fs = float(full.fs)
    c0, c1 = args.channels
    sos = build_iir_sos(fs=fs, kind="bandpass", fmin=args.band[0], fmax=args.band[1], order=args.order)

    fsx = TapeFileSet([str(p) for p in pair], gap_policy="fill")
    gg = fsx.gaps[0]
    nb, na = int(args.before * fs), int(args.after * fs)
    t0, t1 = gg.start - nb, gg.stop + na
    raw = np.asarray(fsx[t0:t1, c0:c1], dtype=np.float64)
    segs = [(0, nb), (nb + gg.n_samples, nb + gg.n_samples + na)]
    ref = reference(raw, segs, sos)
    view = viewer_path(fsx, sos, c0, c1, t0, t1, args.block)
    sig_rms = float(np.sqrt(np.mean(ref[segs[1][0]:] ** 2)))

    lg = TapeFileSet([str(p) for p in pair], gap_policy="ignore")
    end_a = int(lg.files[0].shape[0])
    carried = viewer_path(lg, sos, c0, c1, end_a - nb, end_a + na, args.block)

    post = slice(segs[1][0], segs[1][1])
    res = {
        "files": [str(p) for p in pair],
        "gap_s": float(g.seconds), "separator_samples": int(gg.n_samples),
        "fs": fs, "channels": [c0, c1], "band_hz": list(args.band), "order": args.order,
        "signal_rms_post_gap": sig_rms,
        "viewer_max_rel_err_pre_gap": float(np.abs(view[:nb] - ref[:nb]).max() / sig_rms),
        "viewer_max_rel_err_post_gap": float(np.abs(view[post] - ref[post]).max() / sig_rms),
        "viewer_separator_all_zero": bool(np.all(view[nb:nb + gg.n_samples] == 0)),
    }
    res.update(carried_error(carried[nb:], ref[post], fs, sig_rms))
    np.savez_compressed(run_dir / "real_gap_arrays.npz", raw=raw.astype(np.float32),
                        reference=ref.astype(np.float32), filark=view.astype(np.float32),
                        carried=carried.astype(np.float32), nb=nb, nsep=gg.n_samples, fs=fs)
    plot_real(raw, view, carried, nb, gg.n_samples, fs, g.seconds, (c0, c1), args, run_dir)
    return res


def plot_real(raw, view, carried, nb, nsep, fs, gap_s, chans, args, run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c0, c1 = chans
    # time axis: seconds relative to the end of the pre-gap segment; the separator is drawn
    # as a narrow band and the post-gap segment continues to its right (real outage = gap_s)
    tsep = nsep / fs
    t_pre = (np.arange(nb) - nb) / fs
    t_post = tsep + np.arange(raw.shape[0] - nb - nsep) / fs
    t_full = np.concatenate([t_pre, np.arange(nsep) / fs, t_post])
    n_post = raw.shape[0] - nb - nsep

    # carried result has no separator: put it on the same axis
    carried_axis = np.full_like(view, np.nan)
    carried_axis[:nb] = carried[:nb]
    carried_axis[nb + nsep:] = carried[nb:nb + n_post]

    def v99(a):
        return float(np.nanpercentile(np.abs(a[nb + nsep:]), 99))

    raw_d = raw - np.median(raw[:nb], axis=0)  # remove static offset for display only
    panels = [
        ("(a) Raw (offset removed)", raw_d, v99(raw_d)),
        ("(b) FiLark: IIR state reset at the gap", view, args.clim or v99(view)),
        ("(c) Error if the IIR state is carried across the gap (carried \u2212 reset)",
         carried_axis - view, args.clim or v99(view)),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 8.6),
                             gridspec_kw={"height_ratios": [1, 1, 1, 0.9]}, constrained_layout=True)
    extent = [t_full[0], t_full[-1], c1, c0]
    for ax, (title, img, v) in zip(axes[:3], panels):
        im = ax.imshow(img.T, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v, extent=extent,
                       interpolation="nearest")
        ax.axvspan(0, tsep, color="0.55", alpha=0.6, lw=0)
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_ylabel("Channel", fontsize=8)
        ax.set_xlabel("Time relative to the gap (s)", fontsize=8)
        ax.tick_params(labelsize=7)
        cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.025)
        cb.ax.tick_params(labelsize=6)
    for ax in axes[:3]:
        ax.set_xlim(t_full[0], t_full[-1])
    axes[0].text(tsep + 0.05, c0 + 0.04 * (c1 - c0), f"{gap_s:.0f} s outage (separator)", fontsize=7,
                 color="0.15", va="top", bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))

    # trace panel: one channel, FiLark vs reference vs carried
    err_post = np.abs(carried_axis[nb + nsep:] - view[nb + nsep:])
    k = int(np.argmax(err_post.max(axis=0)))  # channel where carrying the state hurts most
    ax = axes[3]
    ax.plot(t_full, carried_axis[:, k], color="#d95f02", lw=1.0, label="State carried (no reset)")
    ax.plot(t_full, view[:, k], color="#1b5e9e", lw=1.4, label="FiLark (reset at gap)")
    ref_k = np.full(view.shape[0], np.nan)
    ref_k[:nb] = signal.sosfilt(build_iir_sos(fs=fs, kind="bandpass", fmin=args.band[0], fmax=args.band[1],
                                              order=args.order), raw[:nb, k])
    ref_k[nb + nsep:] = signal.sosfilt(build_iir_sos(fs=fs, kind="bandpass", fmin=args.band[0],
                                                     fmax=args.band[1], order=args.order), raw[nb + nsep:, k])
    ax.plot(t_full, ref_k, color="black", lw=0.8, ls="--", label="Reference (per-segment)")
    ax.axvspan(0, tsep, color="0.55", alpha=0.6, lw=0)
    ax.set_title(f"(d) Channel {c0 + k}", loc="left", fontsize=9)
    ax.set_ylabel("Strain rate", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.set_xlim(-0.5, 2.5)
    lim = 1.1 * np.nanmax(np.abs(carried_axis[nb + nsep:nb + nsep + int(2.5 * fs), k]))
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Time relative to the gap (s), zoomed; grey band = separator", fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(run_dir / f"gap_figure.{ext}", dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    run_dir = C.make_run_dir(args.out, "gaps")
    C.write_json(run_dir / "system.json", {**C.system_info(), "args": vars(args)})

    rows = [synthetic_case(gs, band=tuple(args.band), order=args.order, block=args.block)
            for gs in (0.05, 2.0, 300.0)]
    C.write_csv(run_dir / "synthetic.csv", rows)
    for r in rows:
        print(f"[synthetic gap={r['gap_s']:7.2f}s sep={r['separator_samples']}] "
              f"viewer err={r['viewer_max_rel_err']:.1e} batch err={r['batch_max_rel_err']:.1e} "
              f"gap zero={r['viewer_gap_all_zero'] and r['batch_gap_all_zero']} | carried: "
              f"max={r['carried_max_rel_err']:.2f}xRMS settle(1%)={r['carried_settle_to_1pct_s']:.2f}s", flush=True)

    if args.real:
        res = real_case(args, run_dir)
        C.write_json(run_dir / "real.json", res)
        print("[real] " + ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                                    for k, v in res.items() if k != "files"), flush=True)
    print(f"results -> {run_dir}")


if __name__ == "__main__":
    main()
