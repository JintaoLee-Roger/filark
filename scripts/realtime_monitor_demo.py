#!/usr/bin/env python3
"""Realtime monitor plugin demo with throughput-scaled fake inference latency.

The fake detector sleeps in proportion to the amount of data it receives:

    sleep_seconds = data_seconds * Uniform(0.5, 0.7)

So the default model simulates roughly 900 +/- 100 ms of inference time per
1 second of DAS data, not a fixed delay per chunk. It then emits random bbox,
polyline, and mask-patch events to exercise overlay rendering.

By default this script launches the normal FiLark GUI with the fake detector
registered as a MonitorPlugin. Use Analyze > Detect Events… and click Run to start
it, just like a real plugin. Pass --auto-start to run the old self-starting
realtime demo path.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from filark.gui.app import apply_theme
from filark.gui.config import GuiConfig
from filark.gui.main_window import MainWindow
from filark.gui.plugins.monitor import MonitorPlugin
from filark.io.fileset import TapeFileSet
from filark.io.tapeio import H5Tape, NpyTape


DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "DS"


def _monitor_param_spec(defaults: dict) -> list[dict]:
    """Build param defaults from CLI options so GUI runs match script settings."""
    return [
        {
            "key": "delay_ms_per_data_s",
            "label": "Latency mean (ms / data s)",
            "type": "float",
            "min": 0.0,
            "max": 5000.0,
            "default": float(defaults.get("delay_ms_per_data_s", 900.0)),
            "step": 50.0,
        },
        {
            "key": "jitter_ms_per_data_s",
            "label": "Latency jitter (+/- ms / data s)",
            "type": "float",
            "min": 0.0,
            "max": 2000.0,
            "default": float(defaults.get("jitter_ms_per_data_s", 100.0)),
            "step": 25.0,
        },
        {
            "key": "events_per_data_s",
            "label": "Events per data second",
            "type": "float",
            "min": 0.0,
            "max": 20.0,
            "default": float(defaults.get("events_per_data_s", 1.2)),
            "step": 0.2,
        },
        {
            "key": "visible_seed_events",
            "label": "Force visible seed events",
            "type": "bool",
            "default": bool(defaults.get("visible_seed_events", True)),
        },
        {
            "key": "visible_seed_period",
            "label": "Visible seed period (chunks)",
            "type": "int",
            "min": 1,
            "max": 60,
            "default": int(defaults.get("visible_seed_period", 2)),
            "step": 1,
        },
        {
            "key": "process_initial_current",
            "label": "Realtime: process current view once",
            "type": "bool",
            "default": bool(defaults.get("process_initial_current", True)),
        },
        {
            "key": "process_reset_chunks",
            "label": "Realtime: process reset chunks",
            "type": "bool",
            "default": bool(defaults.get("process_reset_chunks", False)),
        },
        {
            "key": "display_queue_max",
            "label": "Realtime queue max",
            "type": "int",
            "min": 1,
            "max": 32,
            "default": int(defaults.get("display_queue_max", 4)),
            "step": 1,
        },
        {
            "key": "bbox_pool_size",
            "label": "BBox pool size",
            "type": "int",
            "min": 1,
            "max": 256,
            "default": int(defaults.get("bbox_pool_size", 32)),
            "step": 1,
        },
        {
            "key": "polyline_pool_size",
            "label": "Polyline pool size",
            "type": "int",
            "min": 1,
            "max": 256,
            "default": int(defaults.get("polyline_pool_size", 32)),
            "step": 1,
        },
        {
            "key": "mask_pool_size",
            "label": "Mask pool size",
            "type": "int",
            "min": 1,
            "max": 64,
            "default": int(defaults.get("mask_pool_size", 12)),
            "step": 1,
        },
        {
            "key": "mask_capacity_t",
            "label": "Mask capacity t",
            "type": "int",
            "min": 1,
            "max": 4096,
            "default": int(defaults.get("mask_capacity_t", 512)),
            "step": 32,
        },
        {
            "key": "mask_capacity_c",
            "label": "Mask capacity c",
            "type": "int",
            "min": 1,
            "max": 4096,
            "default": int(defaults.get("mask_capacity_c", 512)),
            "step": 32,
        },
    ]


class ThroughputLatencyMonitor(MonitorPlugin):
    """Fake deep-learning monitor whose latency scales with data duration."""

    label = "Demo DL Monitor (900ms/s)"
    key = "demo_dl_monitor"

    def __init__(self, *, seed: int = 20260429, verbose: bool = True, defaults: dict | None = None) -> None:
        self._seed = int(seed)
        self._verbose = bool(verbose)
        self._run_count = 0
        self.param_spec = _monitor_param_spec(dict(defaults or {}))

    def validate_params(self, params: dict, *, fs: float, dx: float, nt: int, nc: int) -> dict:
        mean = max(0.0, float(params.get("delay_ms_per_data_s", 900.0)))
        jitter = max(0.0, float(params.get("jitter_ms_per_data_s", 100.0)))
        rate = max(0.0, float(params.get("events_per_data_s", 1.2)))
        visible_seed_events = bool(params.get("visible_seed_events", True))
        visible_seed_period = max(1, int(params.get("visible_seed_period", 2)))
        return {
            "delay_ms_per_data_s": mean,
            "jitter_ms_per_data_s": min(jitter, mean) if mean > 0 else jitter,
            "events_per_data_s": rate,
            "visible_seed_events": visible_seed_events,
            "visible_seed_period": visible_seed_period,
            "process_initial_current": bool(params.get("process_initial_current", True)),
            "process_reset_chunks": bool(params.get("process_reset_chunks", False)),
            "display_queue_max": max(1, int(params.get("display_queue_max", 4))),
            "bbox_pool_size": max(1, int(params.get("bbox_pool_size", 32))),
            "polyline_pool_size": max(1, int(params.get("polyline_pool_size", 32))),
            "mask_pool_size": max(1, int(params.get("mask_pool_size", 12))),
            "mask_capacity_t": max(1, int(params.get("mask_capacity_t", 512))),
            "mask_capacity_c": max(1, int(params.get("mask_capacity_c", 512))),
        }

    def run(
        self,
        source,
        *,
        params: dict,
        progress_cb: Callable[[int, int], None] | None = None,
        cancel_cb: Callable[[], bool] | None = None,
        device: str = "cpu",
        result_cb: Callable[[list[dict]], None] | None = None,
    ) -> list[dict]:
        dims = str(getattr(source, "dims", "nt_nc") or "nt_nc")
        if dims == "nt_nc":
            nt, nc = int(source.shape[0]), int(source.shape[1])
        else:
            nc, nt = int(source.shape[0]), int(source.shape[1])
        fs = max(1e-9, float(getattr(source, "fs", 1.0) or 1.0))
        data_s = max(0.0, float(nt) / fs)

        self._run_count += 1
        seed = self._seed + self._run_count * 1009 + int(time.time_ns() & 0xFFFF)
        rng = np.random.default_rng(seed)
        jitter = float(params["jitter_ms_per_data_s"])
        mean = float(params["delay_ms_per_data_s"])
        ms_per_data_s = float(rng.uniform(max(0.0, mean - jitter), mean + jitter))
        sleep_s = data_s * ms_per_data_s / 1000.0

        if progress_cb is not None:
            progress_cb(0, 1)
        self._sleep_cancelable(sleep_s, cancel_cb)
        if cancel_cb is not None and cancel_cb():
            return []

        events = self._random_events(
            rng,
            nt=nt,
            nc=nc,
            fs=fs,
            data_s=data_s,
            rate=float(params["events_per_data_s"]),
            ms_per_data_s=ms_per_data_s,
            visible_seed_events=(
                bool(params.get("visible_seed_events", True))
                and (self._run_count % max(1, int(params.get("visible_seed_period", 2))) == 0)
            ),
        )
        if progress_cb is not None:
            progress_cb(1, 1)
        if self._verbose:
            print(
                "[demo-monitor] "
                f"chunk={nt}x{nc} data_s={data_s:.3f} "
                f"latency={ms_per_data_s:.0f}ms/s sleep={sleep_s:.3f}s "
                f"events={len(events)}",
                flush=True,
            )
        return events

    @staticmethod
    def _sleep_cancelable(seconds: float, cancel_cb) -> None:
        deadline = time.perf_counter() + max(0.0, float(seconds))
        while True:
            if cancel_cb is not None and cancel_cb():
                return
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return
            time.sleep(min(0.03, remaining))

    def _random_events(
        self,
        rng: np.random.Generator,
        *,
        nt: int,
        nc: int,
        fs: float,
        data_s: float,
        rate: float,
        ms_per_data_s: float,
        visible_seed_events: bool,
    ) -> list[dict]:
        if nt <= 2 or nc <= 1:
            return []
        events = []
        if visible_seed_events:
            events = self._guaranteed_visible_events(
                rng,
                nt=nt,
                nc=nc,
                fs=fs,
                ms_per_data_s=ms_per_data_s,
            )
        expected = max(0.0, data_s * rate)
        count = int(rng.poisson(expected))
        count = min(count, 12)
        for _ in range(count):
            r = float(rng.random())
            if r < 0.34:
                events.append(self._polyline_event(rng, nt=nt, nc=nc, fs=fs, ms_per_data_s=ms_per_data_s))
            elif r < 0.67:
                events.append(self._bbox_event(rng, nt=nt, nc=nc, fs=fs, ms_per_data_s=ms_per_data_s))
            else:
                events.append(self._mask_event(rng, nt=nt, nc=nc, fs=fs, ms_per_data_s=ms_per_data_s))
        return events

    def _guaranteed_visible_events(
        self,
        rng: np.random.Generator,
        *,
        nt: int,
        nc: int,
        fs: float,
        ms_per_data_s: float,
    ) -> list[dict]:
        """Put one of each geometry near the right edge of the processed chunk."""
        if nt <= 8 or nc <= 3:
            return []
        w = max(16, min(max(24, nt // 3), int(max(24, 0.28 * fs)), nt - 1))
        h = max(10, min(max(18, nc // 8), 96, nc))
        t0 = max(0, min(nt - w, int(0.70 * nt)))
        t1 = min(nt, t0 + w)
        c0 = max(0, min(nc - h, int(0.30 * nc)))
        c1 = min(nc, c0 + h)

        y_mid = 0.5 * (c0 + c1)
        points = [
            (float(t0), float(y_mid)),
            (float(0.5 * (t0 + t1)), float(min(nc - 1, y_mid + max(1.0, h * 0.6)))),
            (float(t1), float(max(0.0, y_mid - max(1.0, h * 0.5)))),
        ]
        mask = self._blob_mask(t1 - t0, c1 - c0, rng=rng)
        return [
            {
                "geometry_type": "bbox",
                "t0": t0,
                "t1": t1,
                "c0": c0,
                "c1": c1,
                "label": "visible-bbox",
                "color": "#FFD400",
                "width": 6.0,
                "score": 0.99,
                "meta": {"latency_ms_per_data_s": ms_per_data_s, "visible_seed": True},
            },
            {
                "geometry_type": "polyline",
                "points": points,
                "label": "visible-polyline",
                "color": "#00F5FF",
                "width": 7.0,
                "score": 0.98,
                "meta": {"latency_ms_per_data_s": ms_per_data_s, "visible_seed": True},
            },
            {
                "geometry_type": "mask",
                "t0": t0,
                "t1": t1,
                "c0": c0,
                "c1": c1,
                "mask": mask,
                "label": "visible-mask",
                "color": "#39FF14",
                "alpha": 0.52,
                "threshold": 0.04,
                "score": 0.97,
                "meta": {"latency_ms_per_data_s": ms_per_data_s, "visible_seed": True},
            },
        ]

    @staticmethod
    def _span(rng: np.random.Generator, *, nt: int, nc: int, fs: float) -> tuple[int, int, int, int]:
        min_w = max(12, int(round(0.08 * fs)))
        max_w = max(min_w + 1, min(nt, int(round(0.55 * fs))))
        width = int(rng.integers(min_w, max_w + 1))
        width = min(width, max(1, nt - 1))
        right_bias_min = max(0, int(round(0.45 * nt)))
        t0_min = min(right_bias_min, max(0, nt - width))
        t0 = int(rng.integers(t0_min, max(t0_min + 1, nt - width + 1)))
        t1 = min(nt, t0 + width)

        min_h = max(8, min(24, nc))
        max_h = max(min_h, min(nc, max(16, nc // 6)))
        height = int(rng.integers(min_h, max_h + 1))
        height = min(height, max(1, nc))
        c0 = int(rng.integers(0, max(1, nc - height + 1)))
        c1 = min(nc, c0 + height)
        return t0, t1, c0, c1

    def _bbox_event(self, rng, *, nt: int, nc: int, fs: float, ms_per_data_s: float) -> dict:
        t0, t1, c0, c1 = self._span(rng, nt=nt, nc=nc, fs=fs)
        return {
            "geometry_type": "bbox",
            "t0": t0,
            "t1": t1,
            "c0": c0,
            "c1": c1,
            "label": "demo-bbox",
            "color": "#FFB000",
            "width": 5.0,
            "score": float(rng.uniform(0.4, 0.99)),
            "meta": {"latency_ms_per_data_s": ms_per_data_s},
        }

    def _polyline_event(self, rng, *, nt: int, nc: int, fs: float, ms_per_data_s: float) -> dict:
        t0, t1, c0, c1 = self._span(rng, nt=nt, nc=nc, fs=fs)
        n_pts = int(rng.integers(3, 7))
        ts = np.linspace(t0, max(t0 + 1, t1 - 1), n_pts)
        center = 0.5 * (c0 + c1)
        amp = max(1.0, 0.45 * (c1 - c0))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        points = []
        for i, t in enumerate(ts):
            c = center + amp * np.sin(phase + i * 0.9) + float(rng.normal(0.0, 0.4))
            points.append((float(t), float(min(max(c, 0.0), nc - 1.0))))
        return {
            "geometry_type": "polyline",
            "points": points,
            "label": "demo-polyline",
            "color": "#00C2FF",
            "width": 6.0,
            "score": float(rng.uniform(0.4, 0.99)),
            "meta": {"latency_ms_per_data_s": ms_per_data_s},
        }

    def _mask_event(self, rng, *, nt: int, nc: int, fs: float, ms_per_data_s: float) -> dict:
        t0, t1, c0, c1 = self._span(rng, nt=nt, nc=nc, fs=fs)
        mask = self._blob_mask(t1 - t0, c1 - c0, rng=rng)
        return {
            "geometry_type": "mask",
            "t0": t0,
            "t1": t1,
            "c0": c0,
            "c1": c1,
            "mask": mask,
            "label": "demo-mask",
            "color": "#39FF14",
            "alpha": 0.48,
            "threshold": 0.05,
            "score": float(rng.uniform(0.4, 0.99)),
            "meta": {"latency_ms_per_data_s": ms_per_data_s},
        }

    @staticmethod
    def _blob_mask(dt: int, dc: int, *, rng: np.random.Generator) -> np.ndarray:
        dt = max(1, int(dt))
        dc = max(1, int(dc))
        tt = np.linspace(-1.0, 1.0, dt, dtype=np.float32)[:, None]
        cc = np.linspace(-1.0, 1.0, dc, dtype=np.float32)[None, :]
        angle = float(rng.uniform(-0.7, 0.7))
        ca, sa = np.cos(angle), np.sin(angle)
        x = ca * tt - sa * cc
        y = sa * tt + ca * cc
        sx = float(rng.uniform(0.28, 0.48))
        sy = float(rng.uniform(0.20, 0.40))
        core = np.exp(-0.5 * ((x / sx) ** 2 + (y / sy) ** 2))
        ripple = 0.10 * np.sin(8.0 * x + float(rng.uniform(0.0, 6.28)))
        mask = np.clip(core + ripple - 0.08, 0.0, 1.0)
        return mask.astype(np.float32, copy=False)


def load_ds_tape(path: str | Path):
    p = Path(path).expanduser()
    if p.is_dir():
        return TapeFileSet(p, suffixes=(".h5", ".hdf5"))
    if p.is_file() and p.suffix.lower() in (".h5", ".hdf5"):
        return H5Tape(str(p))
    raise FileNotFoundError(f"DS H5 path not found: {p}")


def make_demo_tape(*, duration_s: float, fs: float, nc: int, seed: int) -> NpyTape:
    rng = np.random.default_rng(int(seed))
    nt = int(round(float(duration_s) * float(fs)))
    nt = max(nt, 1024)
    nc = max(4, int(nc))

    data = rng.normal(0.0, 0.18, size=(nt, nc)).astype(np.float32)
    t = np.arange(nt, dtype=np.float32) / float(fs)
    channels = np.arange(nc, dtype=np.float32)
    data += 0.06 * np.sin(2.0 * np.pi * 7.0 * t[:, None] + channels[None, :] * 0.05).astype(np.float32)
    data += 0.04 * np.sin(2.0 * np.pi * 31.0 * t[:, None] - channels[None, :] * 0.09).astype(np.float32)

    # Add a few faint diagonal wave packets so the background is not pure noise.
    for _ in range(18):
        t0 = int(rng.integers(0, max(1, nt - int(0.5 * fs))))
        width = int(rng.integers(max(8, int(0.05 * fs)), max(9, int(0.35 * fs))))
        c_center = float(rng.uniform(0, nc - 1))
        slope = float(rng.uniform(-0.04, 0.04))
        amp = float(rng.uniform(0.4, 1.0))
        end = min(nt, t0 + width)
        local_t = np.arange(end - t0, dtype=np.float32)
        chan_center = c_center + slope * local_t
        env_t = np.exp(-((local_t - 0.5 * len(local_t)) ** 2) / (2 * (0.22 * len(local_t)) ** 2))
        for c in range(nc):
            env_c = np.exp(-((c - chan_center) ** 2) / (2 * 2.5 ** 2))
            data[t0:end, c] += (amp * env_t * env_c).astype(np.float32)

    tape = NpyTape.from_array(data, dims="nt_nc", fs=float(fs), dx=1.0, dx_unit="m")
    return tape


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Launch FiLark with a throughput-scaled MonitorPlugin demo.")
    p.add_argument("--data-path", default=str(DEFAULT_DATA_PATH), help="HDF5 file or folder used as the default demo source.")
    p.add_argument("--synthetic", action="store_true", help="Use generated synthetic data instead of --data-path.")
    p.add_argument("--duration-s", type=float, default=90.0, help="Synthetic source duration in seconds.")
    p.add_argument("--fs", type=float, default=1000.0, help="Source sample rate.")
    p.add_argument("--nc", type=int, default=96, help="Number of channels.")
    p.add_argument("--speed", type=float, default=1.0, help="Realtime autoscroll speed in x realtime.")
    p.add_argument(
        "--scope",
        choices=["current_view", "full_source", "realtime_display"],
        default="realtime_display",
        help="Initial Monitor panel scope selection.",
    )
    p.add_argument("--delay-ms-per-s", type=float, default=900.0, help="Mean inference ms per 1 data second.")
    p.add_argument("--jitter-ms-per-s", type=float, default=100.0, help="+/- jitter around mean ms per data second.")
    p.add_argument("--events-per-s", type=float, default=1.2, help="Expected random event count per data second.")
    p.add_argument(
        "--visible-seed-period",
        type=int,
        default=2,
        help="Draw guaranteed visible bbox/polyline/mask every N processed chunks.",
    )
    p.add_argument("--theme", choices=["dark", "light"], default="dark")
    p.add_argument("--seed", type=int, default=20260429)
    p.add_argument(
        "--auto-start",
        action="store_true",
        help="Run the old self-starting realtime demo path after GUI startup.",
    )
    p.add_argument(
        "--open-monitor-panel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open Analyze > Detect Events… on startup without pressing Run.",
    )
    p.add_argument("--quiet-monitor", action="store_true", help="Disable per-chunk demo monitor logs.")
    p.add_argument("--save-annotations", default="", help="Save monitor detections to annotation JSON.")
    p.add_argument("--mask-pool-size", type=int, default=12, help="Realtime mask texture slot count.")
    p.add_argument("--mask-capacity-t", type=int, default=512, help="Realtime mask slot time capacity.")
    p.add_argument("--mask-capacity-c", type=int, default=512, help="Realtime mask slot channel capacity.")
    p.add_argument(
        "--warmup-overlay-passes",
        type=int,
        default=3,
        help="Draw fixed-position realtime bbox/polyline/mask passes before autoscroll.",
    )
    p.add_argument(
        "--ui-metrics",
        action="store_true",
        help="Print periodic UI smoothness metrics. Manual plugin mode keeps this off by default.",
    )
    p.add_argument("--no-ui-metrics", action="store_true", help="Disable periodic UI smoothness metrics.")
    p.add_argument("--ui-metrics-interval-s", type=float, default=2.0, help="Seconds between UI smoothness reports.")
    p.add_argument(
        "--no-visible-smoke",
        action="store_true",
        help="Disable visible smoke helpers, including warmup and per-chunk seed bbox/polyline/mask events.",
    )
    return p


def _visible_smoke_events(bounds: dict) -> list[dict]:
    """Build fixed-position bbox/polyline/mask events inside one camera view."""
    if not bounds:
        return []
    t0 = int(bounds["t0"])
    t1 = int(bounds["t1"])
    c0 = int(bounds["c0"])
    c1 = int(bounds["c1"])
    if t1 <= t0 or c1 <= c0:
        return []

    w = max(24, int((t1 - t0) * 0.24))
    h = max(12, int((c1 - c0) * 0.24))
    left = t0 + max(1, int((t1 - t0) * 0.68))
    left = max(t0, min(left, t1 - max(1, w)))
    top = c0 + max(1, int((c1 - c0) * 0.18))
    right = min(t1, left + w)
    bottom = min(c1, top + h)

    mid_c = 0.5 * (top + bottom)
    mask_w = max(8, min(w, 192))
    mask_h = max(8, min(h, 96))
    mask_left = max(t0, min(t1 - mask_w, left + max(1, int(0.12 * w))))
    mask_top = max(c0, min(c1 - mask_h, top + max(1, int(0.18 * h))))
    smoke_rng = np.random.default_rng(20260430)
    return [
        {
            "geometry_type": "bbox",
            "t0": left,
            "t1": right,
            "c0": top,
            "c1": bottom,
            "label": "visible-smoke-bbox",
            "color": "#FFFF00",
            "width": 7.0,
            "visible": True,
        },
        {
            "geometry_type": "polyline",
            "points": [
                (float(left), float(mid_c)),
                (float(0.5 * (left + right)), float(min(c1 - 1, mid_c + max(1.0, h * 0.8)))),
                (float(right), float(max(c0, mid_c - max(1.0, h * 0.6)))),
            ],
            "label": "visible-smoke-polyline",
            "color": "#00E5FF",
            "width": 8.0,
            "visible": True,
        },
        {
            "geometry_type": "mask",
            "t0": mask_left,
            "t1": mask_left + mask_w,
            "c0": mask_top,
            "c1": mask_top + mask_h,
            "mask": ThroughputLatencyMonitor._blob_mask(mask_w, mask_h, rng=smoke_rng),
            "label": "visible-smoke-mask",
            "color": "#39FF14",
            "alpha": 0.55,
            "threshold": 0.04,
            "visible": True,
        },
    ]


def draw_visible_smoke_overlays(window: MainWindow) -> None:
    """Draw one regular bbox, polyline, and mask inside the current camera view."""
    events = _visible_smoke_events(window.disp_model.current_view_bounds())
    if not events:
        return
    for event in events:
        window.add_monitor_overlay(str(event["geometry_type"]), dict(event))

    print("[demo] drew visible smoke overlays: bbox + polyline + mask", flush=True)


def warmup_realtime_overlays(window: MainWindow, *, passes: int) -> int:
    """Exercise the realtime overlay pool by drawing repeated fixed-position events."""
    passes = max(0, int(passes))
    if passes <= 0:
        return 0
    bounds = window.disp_model.current_view_bounds()
    events = _visible_smoke_events(bounds)
    if not events:
        return 0

    apply_events = getattr(window, "_apply_realtime_display_events", None)
    add_events = getattr(window, "_realtime_add_events", None)
    if not callable(apply_events) and not callable(add_events):
        return 0

    t0 = int(bounds["t0"])
    t1 = int(bounds["t1"])
    context = {
        "realtime": True,
        "display_realtime": True,
        "scope_label": "warmup",
        "label": "warmup",
        "origin_t": 0,
        "origin_c": 0,
        "owned_t0": t0,
        "owned_t1": t1,
        "scale_x": 1.0,
        "scale_y": 1.0,
        "allow_future_events": True,
    }
    total = 0
    for _ in range(passes):
        if callable(apply_events):
            total += int(apply_events([dict(event) for event in events], context, "warmup") or 0)
        else:
            total += int(
                add_events(
                    [dict(event) for event in events],
                    label="warmup",
                    origin_t=0,
                    origin_c=0,
                    owned_t0=t0,
                    owned_t1=t1,
                    visible_t0=t0,
                    visible_t1=None,
                )
                or 0
            )
    print(
        f"[demo] warmed realtime overlays at fixed view position: passes={passes} events={total}",
        flush=True,
    )
    return total


class DemoSmoothnessMonitor:
    """Print UI heartbeat gaps and realtime monitor queue stats."""

    def __init__(self, window: MainWindow, *, report_interval_s: float = 2.0, tick_ms: int = 33) -> None:
        self._window = window
        self._report_interval_s = max(0.5, float(report_interval_s))
        self._tick_ms = max(10, int(tick_ms))
        self._timer = QTimer(window)
        self._timer.setInterval(self._tick_ms)
        self._timer.timeout.connect(self._on_tick)
        self._last_t: float | None = None
        self._last_report_t = time.perf_counter()
        self._gaps: list[float] = []
        self._last_stats = {
            "events_added": 0,
            "overlay_ms_total": 0.0,
            "overlay_calls": 0,
            "evict_ms_total": 0.0,
            "evict_calls": 0,
        }

    def start(self) -> None:
        self._last_t = time.perf_counter()
        self._last_report_t = self._last_t
        self._timer.start()

    def _on_tick(self) -> None:
        now = time.perf_counter()
        if self._last_t is not None:
            self._gaps.append((now - self._last_t) * 1000.0)
        self._last_t = now
        if now - self._last_report_t < self._report_interval_s:
            return
        self._last_report_t = now
        gaps = np.asarray(self._gaps[-512:], dtype=np.float64)
        self._gaps.clear()
        if gaps.size == 0:
            return
        stats = {}
        getter = getattr(self._window, "realtime_display_stats", None)
        if callable(getter):
            stats = dict(getter())
        overlay_ms_total = float(stats.get("overlay_ms_total", 0.0) or 0.0)
        overlay_calls = int(stats.get("overlay_calls", 0) or 0)
        events_added = int(stats.get("events_added", 0) or 0)
        evict_ms_total = float(stats.get("evict_ms_total", 0.0) or 0.0)
        evict_calls = int(stats.get("evict_calls", 0) or 0)
        overlay_ms_delta = max(0.0, overlay_ms_total - float(self._last_stats["overlay_ms_total"]))
        overlay_calls_delta = max(0, overlay_calls - int(self._last_stats["overlay_calls"]))
        events_added_delta = max(0, events_added - int(self._last_stats["events_added"]))
        evict_ms_delta = max(0.0, evict_ms_total - float(self._last_stats["evict_ms_total"]))
        evict_calls_delta = max(0, evict_calls - int(self._last_stats["evict_calls"]))
        overlay_avg_ms = overlay_ms_delta / overlay_calls_delta if overlay_calls_delta > 0 else 0.0
        evict_avg_ms = evict_ms_delta / evict_calls_delta if evict_calls_delta > 0 else 0.0
        self._last_stats = {
            "events_added": events_added,
            "overlay_ms_total": overlay_ms_total,
            "overlay_calls": overlay_calls,
            "evict_ms_total": evict_ms_total,
            "evict_calls": evict_calls,
        }
        jank_threshold = max(2.0 * self._tick_ms, self._tick_ms + 25.0)
        jank_count = int(np.sum(gaps > jank_threshold))
        print(
            "[demo-ui] "
            f"tick_ms avg={float(np.mean(gaps)):.1f} "
            f"p95={float(np.percentile(gaps, 95)):.1f} "
            f"max={float(np.max(gaps)):.1f} "
            f"jank>{jank_threshold:.0f}ms={jank_count}/{int(gaps.size)} "
            f"queue={int(stats.get('queue', 0) or 0)} "
            f"submitted={int(stats.get('submitted', 0) or 0)} "
            f"completed={int(stats.get('completed', 0) or 0)} "
            f"dropped={int(stats.get('dropped', 0) or 0)} "
            f"overlays={int(stats.get('overlays', 0) or 0)} "
            f"worker_ms={float(stats.get('last_run_ms', 0.0) or 0.0):.1f} "
            f"wait_ms={float(stats.get('last_queue_wait_ms', 0.0) or 0.0):.1f} "
            f"overlay_avg_ms={overlay_avg_ms:.2f} "
            f"overlay_calls={overlay_calls_delta} "
            f"overlay_events={events_added_delta} "
            f"overlay_max_ms={float(stats.get('overlay_ms_max', 0.0) or 0.0):.2f} "
            f"evict_avg_ms={evict_avg_ms:.2f} "
            f"evict_calls={evict_calls_delta} "
            f"evict_max_ms={float(stats.get('evict_ms_max', 0.0) or 0.0):.2f}",
            flush=True,
        )


def build_demo_monitor_defaults(args: argparse.Namespace) -> dict:
    return {
        "delay_ms_per_data_s": float(args.delay_ms_per_s),
        "jitter_ms_per_data_s": float(args.jitter_ms_per_s),
        "events_per_data_s": float(args.events_per_s),
        "visible_seed_events": not bool(args.no_visible_smoke),
        "visible_seed_period": max(1, int(args.visible_seed_period)),
        "process_initial_current": True,
        "process_reset_chunks": False,
        "display_queue_max": 4,
        "bbox_pool_size": 32,
        "polyline_pool_size": 32,
        "mask_pool_size": max(1, int(args.mask_pool_size)),
        "mask_capacity_t": max(1, int(args.mask_capacity_t)),
        "mask_capacity_c": max(1, int(args.mask_capacity_c)),
    }


def configure_manual_monitor_panel(window: MainWindow, args: argparse.Namespace) -> None:
    """Preselect the demo monitor controls, but leave Run to the user."""
    try:
        if bool(args.open_monitor_panel):
            window.handle_panel_toggle(3)
        window.alg_panel.tab_widget.setCurrentIndex(2)
        idx = window.alg_panel.cb_monitor_scope.findData(args.scope)
        if idx >= 0:
            window.alg_panel.cb_monitor_scope.setCurrentIndex(idx)
        save_path = str(args.save_annotations or "").strip()
        if save_path:
            window.alg_panel.edit_monitor_save_path.setText(save_path)
            window.alg_panel.ck_monitor_save.setChecked(True)
            window.alg_panel.set_monitor_save_status(f"Will save to {save_path}")
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.synthetic:
        tape = make_demo_tape(duration_s=args.duration_s, fs=args.fs, nc=args.nc, seed=args.seed)
        source_label = f"synthetic nt={tape.shape[0]} nc={tape.shape[1]}"
    else:
        tape = load_ds_tape(args.data_path)
        source_label = str(Path(args.data_path).expanduser())
    monitor_defaults = build_demo_monitor_defaults(args)
    monitor = ThroughputLatencyMonitor(
        seed=args.seed,
        verbose=not bool(args.quiet_monitor),
        defaults=monitor_defaults,
    )
    cfg = GuiConfig(
        source=tape,
        monitors=[monitor],
        theme=args.theme,
        async_io=True,
        auto_display_aspect=True,
        realtime_speed=float(args.speed),
        realtime_detection_mask_pool=max(1, int(args.mask_pool_size)),
        realtime_detection_mask_capacity_t=max(1, int(args.mask_capacity_t)),
        realtime_detection_mask_capacity_c=max(1, int(args.mask_capacity_c)),
    )

    app = QApplication([sys.argv[0]])
    app.setStyle("Fusion")
    apply_theme(app, cfg.theme)
    window = MainWindow(cfg=cfg)
    window.resize(1400, 900)
    window.show()
    enable_ui_metrics = bool(args.ui_metrics or args.auto_start) and not bool(args.no_ui_metrics)
    if enable_ui_metrics:
        window._demo_smoothness_monitor = DemoSmoothnessMonitor(
            window,
            report_interval_s=float(args.ui_metrics_interval_s),
        )
        window._demo_smoothness_monitor.start()

    params = dict(monitor_defaults)

    def configure_when_ready(attempt: int = 0) -> None:
        source_ready = getattr(window, "_source", None) is not None
        scheduler_ready = getattr(window.canvas, "scheduler", None) is not None
        if not source_ready or not scheduler_ready:
            if attempt < 80:
                QTimer.singleShot(100, lambda: configure_when_ready(attempt + 1))
            return
        configure_manual_monitor_panel(window, args)
        if args.auto_start:
            return
        print(
            "[demo] manual plugin mode ready "
            f"source={source_label} default_scope={args.scope} speed={args.speed}x "
            f"latency={args.delay_ms_per_s}+/-{args.jitter_ms_per_s} ms/data-s. "
            "Open Analyze > Detect Events… and click Run; press R in the canvas for realtime autoscroll.",
            flush=True,
        )

    QTimer.singleShot(500, configure_when_ready)

    if args.auto_start:
        autoscroll_started = {"value": False}

        def start_when_ready(attempt: int = 0) -> None:
            source_ready = getattr(window, "_source", None) is not None
            scheduler_ready = getattr(window.canvas, "scheduler", None) is not None
            if not source_ready or not scheduler_ready:
                if attempt < 80:
                    QTimer.singleShot(100, lambda: start_when_ready(attempt + 1))
                return
            try:
                window.alg_panel.tab_widget.setCurrentIndex(2)
                window.alg_panel.cb_monitor_scope.setCurrentIndex(
                    window.alg_panel.cb_monitor_scope.findData(args.scope)
                )
            except Exception:
                pass
            options = {"scope": args.scope}
            save_path = str(args.save_annotations or "").strip()
            if save_path:
                options["save_enabled"] = True
                options["save_path"] = save_path
            window._on_monitor_run(monitor, params, options)
            print(
                "[demo] auto-started "
                f"source={source_label} scope={args.scope} speed={args.speed}x "
                f"latency={args.delay_ms_per_s}+/-{args.jitter_ms_per_s} ms/data-s "
                "Processing the initial view before autoscroll. "
                "Press R/S in the canvas to toggle/stop autoscroll; use Monitor Cancel to stop detection.",
                flush=True,
            )
            if args.scope != "realtime_display":
                return
            QTimer.singleShot(0, wait_for_initial_pass)

        def start_autoscroll() -> None:
            if autoscroll_started["value"]:
                return
            autoscroll_started["value"] = True
            window.canvas.set_realtime_speed(float(args.speed))
            window.canvas.start_realtime_autoscroll()
            print("[demo] autoscroll started after initial monitor pass", flush=True)

        def start_after_initial_pass() -> None:
            if not args.no_visible_smoke and int(args.warmup_overlay_passes) > 0:
                warmup_realtime_overlays(window, passes=int(args.warmup_overlay_passes))
                QTimer.singleShot(120, start_autoscroll)
                return
            start_autoscroll()

        def wait_for_initial_pass(attempt: int = 0) -> None:
            stats_getter = getattr(window, "realtime_display_stats", None)
            stats = dict(stats_getter()) if callable(stats_getter) else {}
            done = int(stats.get("completed", 0) or 0) + int(stats.get("failed", 0) or 0)
            submitted = int(stats.get("submitted", 0) or 0)
            if submitted > 0 and done <= 0 and attempt < 400:
                QTimer.singleShot(50, lambda: wait_for_initial_pass(attempt + 1))
                return
            start_after_initial_pass()

        QTimer.singleShot(500, start_when_ready)

    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
