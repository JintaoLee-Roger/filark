"""Headless-ish driver for the FiLark streaming viewer (a real on-screen window).

Used by ``bench_memory.py`` and ``bench_render.py``. The canvas is the same
``StreamingCanvas`` the GUI embeds; scrolling uses its built-in realtime
auto-scroll, whose per-component timing (fetch / filter / upload / render)
is collected by the canvas itself (``_rt_compute_metrics``).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from _common import REPO_ROOT  # noqa: F401  (puts repo on sys.path)


@dataclass
class ViewerConfig:
    width: int = 1600
    height: int = 900
    buffer_h: int = 2048
    buffer_w: Optional[int] = 4096
    async_io: bool = False
    fit_all: bool = True
    # display filter: (fmin, fmax, order) designed at the scrolling rate; None = off
    display_filter: Optional[tuple[float, float, int]] = (1.0, 50.0, 4)
    timer_hz: int = 120
    vsync: bool = True
    clim: tuple[float, float] = (-0.1, 0.1)


def make_app():
    from vispy import app
    app.use_app("pyside6")
    return app


def build_canvas(cfg: ViewerConfig):
    from filark.viz.canvas import StreamingCanvas
    canvas = StreamingCanvas(
        size=(cfg.width, cfg.height),
        async_io=cfg.async_io,
        default_buffer_h=cfg.buffer_h,
        auto_buffer_resize=cfg.buffer_w is None,
        realtime_autoscroll={"timer_hz": cfg.timer_hz, "enable_stats": True},
        clim=tuple(cfg.clim),
        keys=None,
        vsync=cfg.vsync,
    )
    canvas.show()
    raise_window(canvas)
    return canvas


def raise_window(canvas) -> None:
    """Bring the window to the front: macOS does not paint occluded/background
    windows, and without draws vispy's queued texture uploads are never flushed."""
    try:
        w = canvas.native.window()
        w.raise_()
        w.activateWindow()
    except Exception:
        pass
    if sys.platform == "darwin":
        try:
            from AppKit import NSApplication
            NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        except Exception:
            pass


def attach_source(canvas, tape, cfg: ViewerConfig, *, filter_fs: float):
    from filark.io.stream_processing import RealtimeDisplaySOSFilter
    from filark.io.streaming import StreamingSource
    from filark.dsp.temporal import build_iir_sos

    post = None
    if cfg.display_filter is not None:
        fmin, fmax, order = cfg.display_filter
        fmax = min(float(fmax), 0.45 * filter_fs)
        sos = build_iir_sos(fs=filter_fs, kind="bandpass", fmin=fmin, fmax=fmax, order=int(order))
        post = RealtimeDisplaySOSFilter(sos)
    src = StreamingSource(tape, dims=getattr(tape, "dims", "auto"), postprocess=post)
    buffer_h = min(cfg.buffer_h, src.shape[0])
    canvas.set_source(src, buffer_h=buffer_h, buffer_w=cfg.buffer_w)
    if cfg.fit_all:
        canvas.fit_y()
    return src


def texture_mb(canvas) -> float:
    """Size of the ring-buffer GPU texture (r32f => 4 bytes/texel)."""
    img = getattr(canvas, "image", None)
    h = getattr(img, "_H", None)
    w = getattr(img, "_W", None)
    if not h or not w:
        return float("nan")
    return float(h) * float(w) * 4 / 1024**2


def view_t0(canvas) -> int:
    b = canvas.current_view_bounds()
    return int(b["t0"]) if b else 0


def run_autoscroll(
    app,
    canvas,
    *,
    scroll_fs: float,
    speed: float,
    wall_s: float,
    warmup_s: float = 0.0,
    on_sample: Optional[Callable[[float], None]] = None,
    sample_every_s: float = 1.0,
    stop_at_end: bool = True,
) -> dict:
    """Scroll at ``speed`` x real time (``scroll_fs`` samples/s) and return canvas metrics.

    ``warmup_s`` of scrolling is discarded (stats reset) before the measured
    ``wall_s`` window. ``on_sample(t_wall)`` is called every ``sample_every_s``.
    Returns the canvas metric dict plus ``ended_at_data_end``.
    """
    import time

    canvas.set_realtime_fs(scroll_fs, follow_source=False)
    canvas.set_realtime_speed(speed)
    canvas.set_realtime_auto_stop(None)

    state = {"phase": "warmup" if warmup_s > 0 else "measure", "t_phase": 0.0,
             "metrics": None, "ended": False, "last_t0": -1, "stalls": 0, "t_start": 0.0}

    def finish():
        state["metrics"] = canvas._rt_compute_metrics()
        canvas.stop_realtime_autoscroll(reason="bench")
        sampler.stop()
        app.quit()

    def tick(_ev=None):
        now = time.perf_counter()
        if state["phase"] == "warmup":
            if now - state["t_phase"] >= warmup_s:
                canvas.stop_realtime_autoscroll(reason="bench-warmup")
                canvas.start_realtime_autoscroll(reason="bench")  # resets stats
                state["phase"] = "measure"
                state["t_phase"] = now
            return
        if on_sample is not None:
            on_sample(now - state["t_start"])
        t0 = view_t0(canvas)
        if stop_at_end:
            if t0 == state["last_t0"]:
                state["stalls"] += 1
            else:
                state["stalls"] = 0
            state["last_t0"] = t0
            if state["stalls"] >= max(3, int(3.0 / sample_every_s)):
                state["ended"] = True
                finish()
                return
        if now - state["t_phase"] >= wall_s:
            finish()

    sampler = app.Timer(interval=sample_every_s, connect=tick, start=False)

    def start(_ev=None):
        state["t_phase"] = state["t_start"] = time.perf_counter()
        if on_sample is not None and state["phase"] == "measure":
            on_sample(0.0)
        canvas.start_realtime_autoscroll(reason="bench")
        sampler.start()

    starter = app.Timer(interval=1.0, connect=start, iterations=1, start=True)
    app.run()
    del starter
    interrupted = state["metrics"] is None  # event loop left early (e.g. window closed)
    if interrupted:
        state["metrics"] = canvas._rt_compute_metrics()
        canvas.stop_realtime_autoscroll(reason="bench-interrupted")
        sampler.stop()
    m = dict(state["metrics"] or {})
    m["interrupted"] = interrupted
    m.pop("extra", None)
    m["ended_at_data_end"] = bool(state["ended"])
    return m


def settle(app, seconds: float = 1.0) -> None:
    """Spin the event loop for a moment without quitting it (vispy's Qt
    ``app.quit()`` hides the window, after which nothing is painted)."""
    import time
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        app.process_events()
        time.sleep(0.002)


def as_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float(np.nan)
