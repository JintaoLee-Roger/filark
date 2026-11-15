"""Memory footprint of the streaming viewer vs. browsing duration and file count.

Answers SE-1 / R1-1: process memory and GPU (texture) memory while browsing a
long multi-file record, including the viewer's buffering/caching.

Two measurements:

1. ``--open-sweep``  open 1, 2, 4, ... N files as one logical record and
   record memory after opening + first full view (file-count dependence).
2. browsing run     auto-scroll through the whole record (or ``--wall-max``
   seconds) and sample memory every ``--sample-every`` seconds
   (duration dependence).

Recorded per sample: RSS, macOS phys_footprint (includes GPU allocations in
unified memory), ring-buffer texture size, bytes read from disk, data
position (seconds of record browsed, current file index).

Example (MacBook, 1 h of DS data, 20x real time):
    python benchmarks/bench_memory.py /path/to/record_folder --speed 10 --open-sweep
"""

from __future__ import annotations

import argparse
import gc

import numpy as np

import _common as C
import _viewer as V


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", help="folder (or single file) of the record")
    ap.add_argument("--pattern", default="*.h5")
    ap.add_argument("--files", type=int, default=None, help="use only the first N files")
    ap.add_argument("--speed", type=float, default=20.0, help="scroll speed, x real time")
    ap.add_argument("--wall-max", type=float, default=3 * 3600, help="max wall seconds of browsing")
    ap.add_argument("--sample-every", type=float, default=1.0)
    ap.add_argument("--no-filter", action="store_true", help="disable the display IIR filter")
    ap.add_argument("--no-fit-all", action="store_true", help="local channel window instead of fit-all")
    ap.add_argument("--async-io", action="store_true")
    ap.add_argument("--clim", type=float, nargs=2, default=[-0.1, 0.1], metavar=("VMIN", "VMAX"))
    ap.add_argument("--buffer-h", type=int, default=2048)
    ap.add_argument("--buffer-w", type=int, default=4096)
    ap.add_argument("--open-sweep", action="store_true", help="also run the file-count sweep")
    ap.add_argument("--no-browse", action="store_true", help="only run the open sweep")
    ap.add_argument("--out", default="benchmarks/results")
    ap.add_argument("--_probe-open", dest="probe_open", action="store_true", help=argparse.SUPPRESS)
    return ap.parse_args()


def mem_now() -> dict:
    return {"rss_mb": C.rss_mb(), "footprint_mb": C.phys_footprint_mb(),
            "gpu_proc_mb": C.nvidia_process_mem_mb()}


def file_index_of(fileset, t_sample: int) -> int:
    """Index of the physical file containing logical sample ``t_sample`` (-1 inside a gap)."""
    starts = np.asarray(fileset._piece_start)
    k = int(np.clip(np.searchsorted(starts, t_sample, side="right") - 1, 0, len(starts) - 2))
    return int(fileset._piece_file[k])


def probe_open(files, cfg) -> dict:
    """Run inside a fresh process: open ``files``, show the first view, report memory."""
    app = V.make_app()
    canvas = V.build_canvas(cfg)
    V.settle(app, 1.0)
    before = mem_now()
    fs_ = C.open_fileset(files)
    V.attach_source(canvas, fs_, cfg, filter_fs=float(fs_.fs))
    V.settle(app, 3.0)
    after = mem_now()
    desc = C.describe_source(fs_, files)
    return {"n_files": len(files), "record_s": desc["duration_s"], "record_gb": desc["gb_on_disk"],
            **{f"{k}_before": v for k, v in before.items()},
            **{f"{k}_after": v for k, v in after.items()},
            "texture_mb": V.texture_mb(canvas)}


def open_sweep(args, files, run_dir):
    import json
    import subprocess
    import sys
    rows = []
    counts = sorted({1, *[2**k for k in range(1, 20) if 2**k < len(files)], len(files)})
    for n in counts:
        cmd = [sys.executable, "-W", "ignore", __file__, args.data, "--pattern", args.pattern,
               "--files", str(n), "--_probe-open", "--buffer-h", str(args.buffer_h),
               "--buffer-w", str(args.buffer_w)]
        cmd += ["--clim", str(args.clim[0]), str(args.clim[1])]
        if args.no_filter:
            cmd.append("--no-filter")
        if args.no_fit_all:
            cmd.append("--no-fit-all")
        out = subprocess.run(cmd, capture_output=True, text=True)
        line = [ln for ln in out.stdout.splitlines() if ln.startswith("PROBE ")]
        if not line:
            print(out.stderr[-2000:])
            raise RuntimeError(f"probe failed for n={n}")
        r = json.loads(line[-1][6:])
        rows.append(r)
        print(f"[open-sweep] files={n:5d} record={r['record_s']/60:8.1f} min {r['record_gb']:7.2f} GB "
              f"RSS={r['rss_mb_after']:.0f} MB footprint={r['footprint_mb_after']:.0f} MB", flush=True)
    C.write_csv(run_dir / "open_sweep.csv", rows)


def browse(app, files, cfg, args, run_dir):
    m0 = mem_now()
    fileset = C.open_fileset(files)
    desc = C.describe_source(fileset, files)
    C.write_json(run_dir / "source.json", desc)
    fs = float(fileset.fs)
    canvas = V.build_canvas(cfg)
    V.attach_source(canvas, fileset, cfg, filter_fs=fs)
    V.settle(app, 2.0)
    m1 = mem_now()
    io0 = C.disk_read_bytes()

    samples = []

    def on_sample(t_wall):
        t0 = V.view_t0(canvas)
        sched = canvas.scheduler
        cache = getattr(sched, "_replay_cache", None)
        samples.append({
            "t_wall_s": t_wall,
            "data_pos_s": t0 / fs,
            "data_pos_sample": t0,
            "file_index": file_index_of(fileset, t0),
            **mem_now(),
            "texture_mb": V.texture_mb(canvas),
            "disk_read_gb": (C.disk_read_bytes() - io0) / 1024**3,
            "replay_window_blocks": len(getattr(cache, "_window_cache", ()) or ()),
            "replay_checkpoints": len(getattr(cache, "_checkpoints", ()) or ()),
        })
        if len(samples) % 30 == 1:
            s = samples[-1]
            print(f"[browse] wall={t_wall:7.0f}s data={s['data_pos_s']/60:8.2f} min "
                  f"file={s['file_index']:4d} RSS={s['rss_mb']:7.1f} MB "
                  f"footprint={s['footprint_mb']:7.1f} MB read={s['disk_read_gb']:.2f} GB", flush=True)

    metrics = V.run_autoscroll(app, canvas, scroll_fs=fs, speed=args.speed, wall_s=args.wall_max,
                               on_sample=on_sample, sample_every_s=args.sample_every)
    C.write_csv(run_dir / "browse_samples.csv", samples)

    rss = np.array([s["rss_mb"] for s in samples])
    pos_h = np.array([s["data_pos_s"] for s in samples]) / 3600
    half = len(samples) // 2
    slope = float(np.polyfit(pos_h[half:], rss[half:], 1)[0]) if len(samples) > 10 and np.ptp(pos_h[half:]) > 0 else float("nan")
    summary = {
        "rss_before_open_mb": m0["rss_mb"],
        "rss_after_first_view_mb": m1["rss_mb"],
        "footprint_after_first_view_mb": m1["footprint_mb"],
        "rss_min_mb": float(rss.min()), "rss_max_mb": float(rss.max()),
        "rss_last_mb": float(rss[-1]),
        "rss_std_mb": float(rss.std()),
        "rss_slope_second_half_mb_per_hour_of_data": slope,
        "footprint_max_mb": float(np.nanmax([s["footprint_mb"] for s in samples])),
        "texture_mb": V.texture_mb(canvas),
        "data_browsed_s": float(pos_h[-1] * 3600),
        "data_browsed_gb": float(samples[-1]["disk_read_gb"]),
        "files_traversed": int(max(s["file_index"] for s in samples) + 1),
        "record_gb": desc["gb_on_disk"],
        "speed_x": args.speed,
        "achieved_x": metrics.get("achieved_x"),
        "ended_at_data_end": metrics.get("ended_at_data_end"),
        "interrupted": metrics.get("interrupted"),
    }
    C.write_json(run_dir / "browse_summary.json", summary)
    print("[browse] summary:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in summary.items()})


def main():
    args = parse_args()
    files = C.list_files(args.data, args.pattern, args.files)
    cfg = V.ViewerConfig(
        buffer_h=args.buffer_h, buffer_w=args.buffer_w, async_io=args.async_io,
        fit_all=not args.no_fit_all, display_filter=None if args.no_filter else (1.0, 50.0, 4),
        clim=tuple(args.clim),
    )
    if args.probe_open:
        import json
        print("PROBE " + json.dumps(probe_open(files, cfg)), flush=True)
        return
    run_dir = C.make_run_dir(args.out, "memory")
    C.write_json(run_dir / "system.json", {**C.system_info([args.data]), "args": vars(args)})
    if args.open_sweep:
        open_sweep(args, files, run_dir)
    if not args.no_browse:
        browse(V.make_app(), files, cfg, args, run_dir)
    print(f"results -> {run_dir}")


if __name__ == "__main__":
    main()
