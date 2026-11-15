"""Streaming-renderer throughput (Fig. perf) with repeated trials and cache control.

Answers SE-2 / R1-3: repeated runs with variability, and the effect of the OS
page cache. Each trial runs in a fresh process so trials are independent.

For every (dataset = channel count N_c, scrolling rate f_s, IO mode) the
viewer scrolls in fit-all mode with the display IIR filter on, at 1x real
time, and the canvas' own instrumentation reports the per-component cost in
ms per second of data (fetch, filter, upload, render).

Cache conditions:
  cold  OS file cache flushed before every trial (``sudo purge`` on macOS;
        run ``sudo -v`` first so it does not prompt)
  warm  one discarded trial first, then measured trials re-read cached data

Example (the OOI channel-count series used in the paper):
    sudo -v
    python benchmarks/bench_render.py data/OOI-{2048,4096,8192,16384,32768,65536} \\
        --fs 1000 2000 4000 --modes sync async --cache cold warm --repeats 10
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import _common as C

VALUE_KEYS = ["fetch_ms_per_s", "filter_ms_per_s", "upload_ms_per_s", "render_ms_per_s",
              "io_thread_ms_per_s", "main_thread_ms_per_s", "total_ms_per_s", "headroom_ms",
              "achieved_x", "tick_hz", "trial_read_gb"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+", help="one folder per channel-count configuration")
    ap.add_argument("--pattern", default="*.h5")
    ap.add_argument("--fs", type=float, nargs="+", default=[1000.0, 2000.0, 4000.0],
                    help="scrolling rates (samples/s of data advanced per wall second at 1x)")
    ap.add_argument("--modes", nargs="+", default=["sync", "async"], choices=["sync", "async"])
    ap.add_argument("--cache", nargs="+", default=["cold", "warm"], choices=["cold", "warm"])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--wall", type=float, default=10.0, help="measured seconds per trial")
    ap.add_argument("--warmup", type=float, default=1.0, help="discarded seconds before measuring")
    ap.add_argument("--clim", type=float, nargs=2, default=[-0.1, 0.1], metavar=("VMIN", "VMAX"))
    ap.add_argument("--buffer-h", type=int, default=2048)
    ap.add_argument("--buffer-w", type=int, default=4096)
    ap.add_argument("--no-filter", action="store_true")
    ap.add_argument("--timer-hz", type=int, default=120, help="autoscroll timer / frame request rate")
    ap.add_argument("--no-vsync", action="store_true", help="disable vsync so render excludes display-refresh waits")
    ap.add_argument("--out", default="benchmarks/results")
    ap.add_argument("--_trial", dest="trial", default=None, help=argparse.SUPPRESS)
    return ap.parse_args()


def run_trial(spec: dict) -> dict:
    import _viewer as V
    files = C.list_files(spec["dataset"], spec["pattern"])
    tape = C.open_fileset(files)
    cfg = V.ViewerConfig(buffer_h=spec["buffer_h"], buffer_w=spec["buffer_w"],
                         async_io=spec["mode"] == "async", fit_all=True,
                         display_filter=None if spec["no_filter"] else (1.0, 50.0, 4),
                         clim=tuple(spec["clim"]), vsync=not spec.get("no_vsync", False),
                         timer_hz=int(spec.get("timer_hz", 120)))
    app = V.make_app()
    canvas = V.build_canvas(cfg)
    V.attach_source(canvas, tape, cfg, filter_fs=spec["fs"])
    V.settle(app, 1.5)
    io0 = C.disk_read_bytes()
    m = V.run_autoscroll(app, canvas, scroll_fs=spec["fs"], speed=spec["speed"],
                         wall_s=spec["wall"], warmup_s=spec["warmup"], stop_at_end=False)
    m["trial_read_gb"] = (C.disk_read_bytes() - io0) / 1024**3
    m["n_channels"] = int(canvas.nc)
    m["rss_mb_end"] = C.rss_mb()
    return m


def launch(spec: dict) -> dict:
    cmd = [sys.executable, "-W", "ignore", __file__, spec["dataset"], "--_trial", json.dumps(spec)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("TRIAL ")]
    if not lines:
        sys.stderr.write(out.stderr[-3000:])
        raise RuntimeError(f"trial failed: {spec}")
    return json.loads(lines[-1][6:])


def main():
    args = parse_args()
    if args.trial:
        print("TRIAL " + json.dumps(run_trial(json.loads(args.trial)), default=float), flush=True)
        return

    run_dir = C.make_run_dir(args.out, "render")
    sources = []
    for d in args.datasets:
        files = C.list_files(d, args.pattern)
        sources.append({"dataset": d, **C.describe_source(C.open_fileset(files), files)})
    C.write_json(run_dir / "system.json", {**C.system_info(args.datasets), "args": vars(args)})
    C.write_json(run_dir / "sources.json", sources)

    if "cold" in args.cache and not C.drop_os_cache():
        sys.exit("cold-cache trials need `sudo -v` first (sudo purge / drop_caches must not prompt)")

    rows = []
    trials_csv = run_dir / "trials.csv"
    for cache in args.cache:
        for mode in args.modes:
            for fs in args.fs:
                for d in args.datasets:
                    spec = dict(dataset=d, pattern=args.pattern, fs=fs, mode=mode, speed=args.speed,
                                wall=args.wall, warmup=args.warmup, buffer_h=args.buffer_h,
                                buffer_w=args.buffer_w, no_filter=args.no_filter, no_vsync=args.no_vsync, timer_hz=args.timer_hz,
                                clim=args.clim)
                    reps = list(range(args.repeats))
                    if cache == "warm":
                        reps = ["first"] + reps  # first run populates the page cache; kept, not averaged
                    for rep in reps:
                        if cache == "cold":
                            C.drop_os_cache(verbose=False)
                        m = launch(spec)
                        row = {"cache": cache if rep != "first" else "first-run", "mode": mode, "fs": fs,
                               "dataset": Path(d).name,
                               "n_channels": m.get("n_channels"), "rep": rep,
                               **{k: m.get(k) for k in VALUE_KEYS},
                               "render_ticks": m.get("render_ticks"), "ticks": m.get("ticks")}
                        rows.append(row)
                        C.write_csv(trials_csv, rows)
                        print(f"[{row['cache']:9s} {mode:5s} fs={fs:6.0f} Nc={row['n_channels']:6d} rep={rep!s:>5}] "
                              f"fetch={row['fetch_ms_per_s']:7.1f} filter={row['filter_ms_per_s']:6.1f} "
                              f"upload={row['upload_ms_per_s']:5.1f} render={row['render_ms_per_s']:5.1f} "
                              f"total={row['total_ms_per_s']:7.1f} ms/s  achieved={row['achieved_x']:.2f}x",
                              flush=True)
    summary = C.summarize(rows, ["cache", "mode", "fs", "n_channels"], VALUE_KEYS)
    C.write_csv(run_dir / "summary.csv", summary)
    print(f"results -> {run_dir}")


if __name__ == "__main__":
    main()
