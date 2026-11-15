# Copyright (c) 2026 Jintao Li. 
# Zhejiang University (ZJU).
# 
# Licensed under the MIT License.

"""Keyboard and realtime scrolling mixins for ``StreamingCanvas``."""

import math as _math
import time

from vispy import app

from filark._exceptions import print_exception

# Realtime autoscroll catch-up limits (see _rt_on_tick).
_RT_PAUSE_S = 0.5     # a tick gap longer than this is a pause, not lag to repay
_RT_MAX_LAG_S = 1.0   # cap on accumulated wall-time lag


# ── memory sampling helpers (optional, needs psutil / torch) ─────────────────

def _sample_rss_mb() -> float:
    """Return process RSS in MB, or nan if psutil is unavailable."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 ** 2)
    except Exception:
        return float("nan")


def _print_exception(context: str) -> None:
    print_exception(context, prefix="CanvasMixin")


class KeyboardPanMixin:
    """
    Deterministic keyboard control.

    - Left/Right: pan X via camera (camera -> scheduler X streaming)
    - Up/Down:    pan Y via scheduler (deterministic) + optional camera Y sync
    - Y / Shift+Y: scale_y ×2 / ÷2
    - X / Shift+X: scale_x ×2 / ÷2
    - A / Shift+A: display_aspect ÷1.25 / ×1.25
    - F / Shift+F: fit Y / reset Y
    """

    def _init_keyboard_pan(self):
        self.events.key_press.connect(self._on_key_press)

    def _on_key_press(self, ev):
        # Guard: do nothing until a source is loaded and canvas is initialised.
        if getattr(self, "source", None) is None:
            return
        if not getattr(self, "_did_init", True):
            return

        key_name = _event_key_name(ev)
        if key_name is None:
            return
        key = key_name.upper()
        shift, ctrl, meta, alt = _event_modifier_flags(ev)

        if key in ("BACKSPACE", "DELETE"):
            ev.handled = True
            return

        scale_factor = 2.0
        if key == "X":
            if ctrl or meta or alt:
                return
            new_sx = (self.source.scale_x / scale_factor) if shift else (self.source.scale_x * scale_factor)
            new_sx = max(1.0, float(new_sx))
            self.set_scale_x(new_sx)
            ev.handled = True
            return

        if key == "Y":
            if ctrl or meta or alt:
                return
            new_sy = (self.source.scale_y / scale_factor) if shift else (self.source.scale_y * scale_factor)
            new_sy = max(1.0, float(new_sy))
            self.set_scale_y(new_sy)
            ev.handled = True
            return

        if key == "A":
            if ctrl or meta or alt:
                return
            cur_aspect = float(getattr(self, "current_display_aspect", lambda: 1.0)())
            aspect_factor = 1.25
            new_aspect = (cur_aspect * aspect_factor) if shift else (cur_aspect / aspect_factor)
            self.set_display_aspect(new_aspect)
            ev.handled = True
            return

        if key == "F":
            if ctrl or meta or alt:
                return
            if shift:
                self.set_scale_y(1.0)
            else:
                self.fit_y()
            ev.handled = True
            return

        if key == "C":
            if shift or ctrl or meta or alt:
                return
            if hasattr(self, "toggle_cursor_pos"):
                self.toggle_cursor_pos()
            ev.handled = True
            return

        name = key_name
        if name not in ("Left", "Right", "Up", "Down", "PageUp", "PageDown"):
            return

        # Guard: camera and scheduler must exist.
        if getattr(self, "camera", None) is None or getattr(self, "scheduler", None) is None:
            return

        # Ctrl or Meta(Cmd on macOS) + Left/Right: let Qt-level shortcuts handle
        # compare-source cycling; do NOT pan.
        if (ctrl or meta) and name in ("Left", "Right"):
            ev.handled = True
            return
        if name in ("Left", "Right") and (shift or alt):
            return
        if name in ("PageUp", "PageDown") and (shift or ctrl or meta or alt):
            return
        if name in ("Up", "Down"):
            valid_alt_page = alt and not (shift or ctrl or meta)
            valid_plain_scroll = not (shift or ctrl or meta or alt)
            if not (valid_alt_page or valid_plain_scroll):
                return

        # temporarily disable coalesce for debug clarity
        was = getattr(self.scheduler, "_enable_coalesce", False)
        if was:
            self.scheduler.stop_coalesce()

        if name in ("Left", "Right"):
            step = 100
            dx = -step if name == "Left" else step
            self.camera.pan_x(dx)
        elif name in ("PageUp", "PageDown"):
            if self._wrap_y:
                self.page_channels(-1 if name == "PageUp" else +1, sync_camera=True)
        else:  # Up / Down
            if not self._wrap_y:
                pass  # Y-axis navigation disabled
            elif alt and not (shift or ctrl or meta):
                # Alt + Up/Down → page channels (Page Up/Down substitute)
                self.page_channels(-1 if name == "Up" else +1, sync_camera=True)
            elif not (shift or ctrl or meta or alt):
                self.scroll_channels(-20 if name == "Up" else +20, sync_camera=True)

        if was:
            self.scheduler.start_coalesce(hz=self.scheduler.coalesce_hz)

        ev.handled = True


class RealtimeAutoScrollMixin:
    """
    Realtime auto-scroll mixin for StreamingCanvas (simplified, 2-level stats + demo benchmark).

    Keys:
      - R : toggle realtime auto-scroll
      - S : stop
      - 1..8 : set fs (samples/sec)
      - +/= : speed up
      - -/_ : speed down
      - P : print DEBUG stats
      - Shift+P : print SHOWCASE stats (for demos / papers)
      - B : run a short benchmark (push speed up to a strong cap, report max sustained throughput)
    """

    def _init_realtime_autoscroll(
        self,
        fs=500,
        speed=1.0,
        timer_hz=120,
        max_dt=0.05,
        enable_stats=True,
        # auto-stop: stop scrolling after this many wall-seconds and print report
        # set to None to disable
        auto_stop_s=None,
        # benchmark knobs (a "very strong but reasonable" cap)
        bench_max_speed=100.0,
        bench_duration_s=2.0,
        bench_min_good_rate=0.90,   # accept as "sustained" if >= 90% of requested speed
        fs_presets=None,
    ):
        self._rt_running = False
        self._rt_fs = float(fs)
        self._rt_speed = float(speed)
        self._rt_follow_source_fs = True
        self._rt_max_dt = float(max_dt)
        self._rt_enable_stats = bool(enable_stats)

        self._rt_timer_hz = float(max(1, int(timer_hz)))
        self._rt_interval = 1.0 / self._rt_timer_hz
        self._rt_timer = app.Timer(interval=self._rt_interval, connect=self._rt_on_tick, start=False)

        self._rt_last_t = None
        self._rt_lag = 0.0  # wall seconds of stalled ticks still to repay (see _rt_on_tick)

        # stats (minimal)
        self._rt_t0 = None
        self._rt_ticks = 0
        self._rt_data_sum = 0.0          # sum of data seconds advanced (dt*speed)
        self._rt_dt_max = 0.0            # max dt (clamped)

        # optional: cpu time around pan (python-side only; not true draw)
        self._rt_pan_cpu_sum = 0.0
        self._rt_pan_cpu_max = 0.0

        # per-component breakdown (consumed from scheduler.telemetry_snapshot()).
        self._rt_fetch_sum = 0.0    # raw data read + downsample (no filter)
        self._rt_fetch_max = 0.0
        self._rt_filter_sum = 0.0   # display filter (postprocess)
        self._rt_filter_max = 0.0
        self._rt_upload_sum = 0.0   # image.push_right() — CPU→GPU texture upload
        self._rt_upload_max = 0.0
        self._rt_render_sum = 0.0   # events.draw handler — actual OpenGL draw call
        self._rt_render_ticks = 0
        self._rt_draw_mark: float | None = None  # set at draw begin, consumed at draw end
        self._rt_metric_providers: dict[str, object] = {}
        self._rt_state_listeners: list[object] = []

        # RSS sampling (at most once per second of wall time).
        # VRAM is not tracked here: on macOS there is no GPU, and when torch
        # GPU operators are used they are for DSP — unrelated to the viz pipeline.
        self._rt_mem_next_t: float = 0.0
        self._rt_rss_mb_start: float = float("nan")   # RSS when scrolling started
        self._rt_rss_mb_last: float = float("nan")    # most recent RSS sample

        # auto-stop after k wall-seconds
        self._rt_auto_stop_s = float(auto_stop_s) if auto_stop_s is not None else None
        self._rt_auto_stop_deadline: float | None = None

        # benchmark state
        self._rt_bench_on = False
        self._rt_bench_t0 = None
        self._rt_bench_prev_speed = None
        self._rt_bench_best = None  # dict: {"req_speed":..., "achieved":...}
        self._rt_bench_max_speed = float(bench_max_speed)
        self._rt_bench_duration_s = float(bench_duration_s)
        self._rt_bench_min_good_rate = float(bench_min_good_rate)
        self._rt_fs_presets = {
            str(key): float(value)
            for key, value in (fs_presets or (
                ("1", 200.0),
                ("2", 500.0),
                ("3", 1000.0),
                ("4", 1500.0),
                ("5", 2000.0),
                ("6", 3000.0),
                ("7", 5000.0),
                ("8", 10000.0),
            ))
        }

        self.events.key_press.connect(self._rt_on_key_press)

    # ---------------------------
    # controls
    # ---------------------------
    def register_realtime_metric_provider(self, name: str, provider) -> None:
        """Register an optional realtime stats provider owned by another layer."""
        self._rt_metric_providers[str(name)] = provider

    def unregister_realtime_metric_provider(self, name: str) -> None:
        self._rt_metric_providers.pop(str(name), None)

    def register_realtime_state_listener(self, listener) -> None:
        """Notify another layer when realtime autoscroll starts or stops."""
        if callable(listener) and listener not in self._rt_state_listeners:
            self._rt_state_listeners.append(listener)

    def unregister_realtime_state_listener(self, listener) -> None:
        try:
            self._rt_state_listeners.remove(listener)
        except ValueError:
            pass

    def _emit_realtime_state_changed(self, *, running: bool, reason: str) -> None:
        for listener in tuple(getattr(self, "_rt_state_listeners", []) or []):
            try:
                listener(bool(running), str(reason))
            except Exception:
                _print_exception("Realtime state listener failed.")

    def _rt_collect_metric_providers(self, *, reset: bool = False) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for name, provider in list((getattr(self, "_rt_metric_providers", {}) or {}).items()):
            if not callable(provider):
                continue
            try:
                try:
                    value = provider(reset=reset)
                except TypeError:
                    value = provider()
                if value is not None:
                    out[str(name)] = dict(value)
            except Exception:
                _print_exception(f"Realtime metric provider failed: {name}")
        return out

    def start_realtime_autoscroll(self, *, reason: str = "manual"):
        if self._rt_running:
            return
        self._rt_sync_source_fs()
        self._rt_running = True
        self._rt_last_t = None

        if self._rt_enable_stats:
            self._rt_t0 = time.perf_counter()
            self._rt_ticks = 0
            self._rt_data_sum = 0.0
            self._rt_dt_max = 0.0
            self._rt_pan_cpu_sum = 0.0
            self._rt_pan_cpu_max = 0.0
            self._rt_fetch_sum = 0.0
            self._rt_fetch_max = 0.0
            self._rt_filter_sum = 0.0
            self._rt_filter_max = 0.0
            self._rt_upload_sum = 0.0
            self._rt_upload_max = 0.0
            self._rt_render_sum = 0.0
            self._rt_render_ticks = 0
            self._rt_draw_mark = None
            self._rt_mem_next_t = 0.0
            rss0 = _sample_rss_mb()
            self._rt_rss_mb_start = rss0
            self._rt_rss_mb_last = rss0
            self._rt_collect_metric_providers(reset=True)

        # set auto-stop deadline
        if self._rt_auto_stop_s is not None:
            self._rt_auto_stop_deadline = time.perf_counter() + self._rt_auto_stop_s
        else:
            self._rt_auto_stop_deadline = None

        self._rt_timer.start()
        self.update()
        # memory snapshot when scrolling starts
        mon = getattr(self, "_mem_monitor", None)
        if mon is not None:
            mon.snapshot("scroll start")
        self._emit_realtime_state_changed(running=True, reason=reason)

    def stop_realtime_autoscroll(self, *, reason: str = "manual"):
        if not self._rt_running:
            return
        self._rt_running = False
        self._rt_timer.stop()
        self._rt_last_t = None
        self.update()
        # memory snapshot when scrolling stops
        mon = getattr(self, "_mem_monitor", None)
        if mon is not None:
            mon.snapshot("scroll stop")
        self._emit_realtime_state_changed(running=False, reason=reason)

    def _rt_sync_source_fs(self):
        if not getattr(self, "_rt_follow_source_fs", True):
            return
        src = getattr(self, "source", None)
        if src is None:
            return
        tape = getattr(src, "data", None)
        if tape is None:
            tape = src
        fs = getattr(tape, "fs", None)
        if fs is None:
            return
        try:
            fs = float(fs)
        except Exception:
            _print_exception("Failed to parse source sampling rate for realtime autoscroll sync.")
            return
        if fs > 0:
            self._rt_fs = fs

    def set_realtime_fs(self, fs, *, follow_source: bool = False):
        self._rt_fs = float(fs)
        self._rt_follow_source_fs = bool(follow_source)

    def set_realtime_speed(self, speed):
        self._rt_speed = float(speed)

    def set_realtime_auto_stop(self, seconds):
        """Set auto-stop duration in wall-seconds (None to disable)."""
        self._rt_auto_stop_s = float(seconds) if seconds is not None else None

    # ---------------------------
    # metrics
    # ---------------------------
    def _rt_compute_metrics(self):
        if not self._rt_enable_stats or self._rt_t0 is None:
            return None

        wall_elapsed = max(1e-12, time.perf_counter() - self._rt_t0)
        ticks = max(1, self._rt_ticks)

        tick_hz = self._rt_ticks / wall_elapsed

        # achieved data throughput in "x realtime" units (because data_sum already includes speed)
        achieved_x = self._rt_data_sum / wall_elapsed

        # for the *requested* speed, the target throughput is speed_x = self._rt_speed
        # so "tracking ratio" tells you if you are keeping up with requested playback speed
        tracking = achieved_x / max(1e-12, self._rt_speed)

        # user-facing: "10 seconds data takes how many seconds wall"
        wall_per_10s = (10.0 / achieved_x) if achieved_x > 1e-12 else float("inf")

        # user-facing: visual overhead relative to ideal realtime (speed=1)
        # overhead% = (1/achieved_x - 1) * 100
        overhead_rt = (max(0.0, (1.0 / max(1e-12, achieved_x)) - 1.0)) * 100.0

        pan_cpu_avg = self._rt_pan_cpu_sum / ticks
        pan_cpu_max = self._rt_pan_cpu_max

        # per-component breakdown (ms per tick; divide by data_s_per_tick for ms/s)
        data_s_per_tick = self._rt_data_sum / max(1, self._rt_ticks)  # s of data per tick
        fetch_avg_ms  = self._rt_fetch_sum  / ticks
        fetch_max_ms  = self._rt_fetch_max
        filter_avg_ms = self._rt_filter_sum / ticks
        filter_max_ms = self._rt_filter_max
        upload_avg_ms = self._rt_upload_sum / ticks
        upload_max_ms = self._rt_upload_max
        # all draws in the window (a tick may trigger 0..n draws), like the other components
        render_avg_ms = self._rt_render_sum / ticks
        # overhead per second of data (the paper metric)
        s = max(1e-12, data_s_per_tick)
        fetch_ms_per_s  = fetch_avg_ms  / s
        filter_ms_per_s = filter_avg_ms / s
        upload_ms_per_s = upload_avg_ms / s
        render_ms_per_s = render_avg_ms / s

        sched = getattr(self, "scheduler", None)
        async_io = bool(getattr(sched, "_async_io", False))
        if async_io:
            # IO thread and main thread run in parallel.
            # Bottleneck = max(IO thread, main thread).
            io_thread_ms_per_s   = (fetch_avg_ms + filter_avg_ms) / s
            main_thread_ms_per_s = (upload_avg_ms + render_avg_ms) / s
            total_ms_per_s       = max(io_thread_ms_per_s, main_thread_ms_per_s)
        else:
            io_thread_ms_per_s   = (fetch_avg_ms + filter_avg_ms) / s
            main_thread_ms_per_s = (upload_avg_ms + render_avg_ms) / s
            total_ms_per_s       = (fetch_avg_ms + filter_avg_ms + upload_avg_ms + render_avg_ms) / s
        headroom_ms = max(0.0, 1000.0 - total_ms_per_s)

        extra = self._rt_collect_metric_providers(reset=False)
        data_s = max(1e-12, self._rt_data_sum)
        extra_main_ms_per_s = 0.0
        for values in extra.values():
            total_ms = float(values.get("total_ms", 0.0) or 0.0)
            ms_per_s = total_ms / data_s
            values["ms_per_s"] = ms_per_s
            if str(values.get("thread", "main")) == "main":
                extra_main_ms_per_s += ms_per_s
        main_thread_with_extra_ms_per_s = main_thread_ms_per_s + extra_main_ms_per_s
        if async_io:
            total_with_extra_ms_per_s = max(io_thread_ms_per_s, main_thread_with_extra_ms_per_s)
        else:
            total_with_extra_ms_per_s = total_ms_per_s + extra_main_ms_per_s
        headroom_with_extra_ms = max(0.0, 1000.0 - total_with_extra_ms_per_s)

        rss_delta = (self._rt_rss_mb_last - self._rt_rss_mb_start
                     if not _math.isnan(self._rt_rss_mb_last) and not _math.isnan(self._rt_rss_mb_start)
                     else float("nan"))

        return dict(
            wall_elapsed=wall_elapsed,
            ticks=ticks,
            tick_hz=tick_hz,
            dt_max=self._rt_dt_max,
            fs=self._rt_fs,
            speed=self._rt_speed,
            timer_hz=self._rt_timer_hz,
            achieved_x=achieved_x,
            tracking=tracking,
            wall_per_10s=wall_per_10s,
            overhead_rt=overhead_rt,
            pan_cpu_avg=pan_cpu_avg,
            pan_cpu_max=pan_cpu_max,
            async_io=async_io,
            # breakdown
            fetch_avg_ms=fetch_avg_ms,
            fetch_max_ms=fetch_max_ms,
            filter_avg_ms=filter_avg_ms,
            filter_max_ms=filter_max_ms,
            upload_avg_ms=upload_avg_ms,
            upload_max_ms=upload_max_ms,
            render_avg_ms=render_avg_ms,
            render_ticks=self._rt_render_ticks,
            fetch_ms_per_s=fetch_ms_per_s,
            filter_ms_per_s=filter_ms_per_s,
            upload_ms_per_s=upload_ms_per_s,
            render_ms_per_s=render_ms_per_s,
            io_thread_ms_per_s=io_thread_ms_per_s,
            main_thread_ms_per_s=main_thread_ms_per_s,
            total_ms_per_s=total_ms_per_s,
            headroom_ms=headroom_ms,
            extra=extra,
            extra_main_ms_per_s=extra_main_ms_per_s,
            main_thread_with_extra_ms_per_s=main_thread_with_extra_ms_per_s,
            total_with_extra_ms_per_s=total_with_extra_ms_per_s,
            headroom_with_extra_ms=headroom_with_extra_ms,
            # RSS only (VRAM omitted: macOS has no GPU; torch VRAM is DSP, not viz)
            rss_mb=self._rt_rss_mb_last,
            rss_delta_mb=rss_delta,
        )

    # ---------------------------
    # printing (two modes)
    # ---------------------------
    @staticmethod
    def _rt_annotation_metrics(m: dict) -> dict | None:
        values = (m.get("extra") or {}).get("annotation")
        if not values:
            return None
        return values

    def _rt_annotation_debug_line(self, m: dict) -> str | None:
        ann = self._rt_annotation_metrics(m)
        if ann is None:
            return None
        calls = int(ann.get("calls", 0) or 0)
        active_total = int(ann.get("active_total", 0) or 0)
        active_bbox = int(ann.get("active_bbox", 0) or 0)
        active_polyline = int(ann.get("active_polyline", 0) or 0)
        active_mask = int(ann.get("active_mask", 0) or 0)
        return (
            "[ANN-BDN] "
            f"refresh={float(ann.get('ms_per_s', 0.0) or 0.0):.2f}ms/s "
            f"avg={float(ann.get('avg_ms', 0.0) or 0.0):.2f}ms "
            f"last={float(ann.get('last_ms', 0.0) or 0.0):.2f}ms "
            f"max={float(ann.get('max_ms', 0.0) or 0.0):.2f}ms "
            f"calls={calls} "
            f"apply/skip={int(ann.get('applied', 0) or 0)}/{int(ann.get('skipped', 0) or 0)} "
            f"active={active_total}(bbox={active_bbox}, polyline={active_polyline}, mask={active_mask}) "
            f"pool={int(ann.get('pool_active_total', 0) or 0)}/{int(ann.get('pool_slots_total', 0) or 0)} "
            f"virtual={int(ann.get('virtual_count', 0) or 0)} "
            f"effective+ann={float(m.get('total_with_extra_ms_per_s', m['total_ms_per_s'])):.1f}ms/s "
            f"headroom+ann={float(m.get('headroom_with_extra_ms', m['headroom_ms'])):.0f}ms"
        )

    def _rt_annotation_showcase_line(self, m: dict) -> str:
        ann = self._rt_annotation_metrics(m)
        if ann is None:
            return ""
        return (
            f"  Annotation refresh: {float(ann.get('ms_per_s', 0.0) or 0.0):>7.2f} ms/s"
            f"  (avg {float(ann.get('avg_ms', 0.0) or 0.0):.2f} ms / "
            f"max {float(ann.get('max_ms', 0.0) or 0.0):.2f} ms per refresh,"
            f" calls={int(ann.get('calls', 0) or 0)}, "
            f"apply/skip={int(ann.get('applied', 0) or 0)}/"
            f"{int(ann.get('skipped', 0) or 0)}, "
            f"active={int(ann.get('active_total', 0) or 0)}, "
            f"mask={int(ann.get('active_mask', 0) or 0)}/"
            f"{int(ann.get('pool_slots_mask', 0) or 0)})\n"
        )

    def print_realtime_debug(self):
        m = self._rt_compute_metrics()
        if not m:
            print("[RT] stats disabled or not started yet.")
            return
        print(
            "[RT-DBG] "
            f"run={self._rt_running} fs={m['fs']:.0f} speed(req)={m['speed']:.3g} target_timer={m['timer_hz']:.0f}Hz | "
            f"tick={m['tick_hz']:.1f}Hz dt_max={m['dt_max']*1000:.2f}ms | "
            f"achieved={m['achieved_x']:.3f}x tracking={m['tracking']*100:.1f}% | "
            f"10s_data→{m['wall_per_10s']:.2f}s | "
            f"pan_cpu={m['pan_cpu_avg']*1000:.3f}/{m['pan_cpu_max']*1000:.3f}ms(avg/max)"
        )
        rss_str = f"{m['rss_mb']:.0f}MB(Δ{m['rss_delta_mb']:+.0f})" if not _math.isnan(m['rss_mb']) else "n/a(install psutil)"
        print(
            "[RT-BDN] "
            f"per-1s-data: "
            f"fetch={m['fetch_ms_per_s']:.1f}ms  "
            f"filter={m['filter_ms_per_s']:.1f}ms  "
            f"upload={m['upload_ms_per_s']:.1f}ms  "
            f"render={m['render_ms_per_s']:.1f}ms  "
            f"total={m['total_ms_per_s']:.1f}ms  "
            f"headroom={m['headroom_ms']:.0f}ms  "
            f"| RSS={rss_str}"
        )
        ann_line = self._rt_annotation_debug_line(m)
        if ann_line:
            print(ann_line)

    def print_realtime_showcase(self):
        m = self._rt_compute_metrics()
        if not m:
            print("[RT] stats disabled or not started yet.")
            return

        # Strong, direct, user-facing wording:
        # - Achieved throughput (x realtime)
        # - "10s data -> ?s" readability
        # - Overhead for speed=1 case (still meaningful as a headline)
        print(
            "[REAL-TIME] "
            f"{m['fs']:.0f} Hz | "
            f"throughput={m['achieved_x']:.2f}× realtime | "
            f"10s data → {m['wall_per_10s']:.2f}s | "
            f"realtime overhead ≈ {m['overhead_rt']:.2f}%"
        )
        def _fmt_rss(v):
            return f"{v:.1f} MB" if not _math.isnan(v) else "n/a  (pip install psutil)"
        def _fmt_delta(v):
            return f"{v:+.1f} MB" if not _math.isnan(v) else "n/a"
        if m['async_io']:
            print(
                "[TIME-BDN] overhead per 1 s of data  [async IO: IO thread ∥ main thread]\n"
                f"  Fetch (IO+dsamp)  : {m['fetch_ms_per_s']:>7.2f} ms/s"
                f"  (avg {m['fetch_avg_ms']:.2f} ms / max {m['fetch_max_ms']:.2f} ms per tick)\n"
                f"  Display filter    : {m['filter_ms_per_s']:>7.2f} ms/s"
                f"  (avg {m['filter_avg_ms']:.2f} ms / max {m['filter_max_ms']:.2f} ms per tick)\n"
                f"  ── IO thread      : {m['io_thread_ms_per_s']:>7.2f} ms/s  (fetch + filter)\n"
                f"  GPU tex upload    : {m['upload_ms_per_s']:>7.2f} ms/s"
                f"  (avg {m['upload_avg_ms']:.2f} ms / max {m['upload_max_ms']:.2f} ms per tick)\n"
                f"  Render (draw call): {m['render_ms_per_s']:>7.2f} ms/s"
                f"  (avg {m['render_avg_ms']:.2f} ms per draw, n={m['render_ticks']})\n"
                f"{self._rt_annotation_showcase_line(m)}"
                f"  ── Main thread    : {m['main_thread_ms_per_s']:>7.2f} ms/s  (upload + render)\n"
                f"  ── Main + extras  : {m['main_thread_with_extra_ms_per_s']:>7.2f} ms/s  (registered UI metrics included)\n"
                f"  ──────────────────────────────────────────────────────────────\n"
                f"  Effective (bottleneck = max): {m['total_with_extra_ms_per_s']:>7.2f} ms/s\n"
                f"  Algorithm headroom: {m['headroom_with_extra_ms']:>7.1f} ms/s"
                f"  ({m['headroom_with_extra_ms']/10:.0f}% of 1-s budget)\n"
                f"  ──────────────────────────────────────────────────────────────\n"
                f"  RSS (process RAM) : {_fmt_rss(m['rss_mb'])}"
                f"  (Δ since scroll start: {_fmt_delta(m['rss_delta_mb'])})\n"
                f"  [Shift+M for full lifecycle memory history]"
            )
        else:
            print(
                "[TIME-BDN] overhead per 1 s of data:\n"
                f"  Fetch (IO+dsamp)  : {m['fetch_ms_per_s']:>7.2f} ms/s"
                f"  (avg {m['fetch_avg_ms']:.2f} ms / max {m['fetch_max_ms']:.2f} ms per tick)\n"
                f"  Display filter    : {m['filter_ms_per_s']:>7.2f} ms/s"
                f"  (avg {m['filter_avg_ms']:.2f} ms / max {m['filter_max_ms']:.2f} ms per tick)\n"
                f"  GPU tex upload    : {m['upload_ms_per_s']:>7.2f} ms/s"
                f"  (avg {m['upload_avg_ms']:.2f} ms / max {m['upload_max_ms']:.2f} ms per tick)\n"
                f"  Render (draw call): {m['render_ms_per_s']:>7.2f} ms/s"
                f"  (avg {m['render_avg_ms']:.2f} ms per draw, n={m['render_ticks']})\n"
                f"{self._rt_annotation_showcase_line(m)}"
                f"  ──────────────────────────────────────────────────────────────\n"
                f"  Total overhead    : {m['total_with_extra_ms_per_s']:>7.2f} ms/s\n"
                f"  Algorithm headroom: {m['headroom_with_extra_ms']:>7.1f} ms/s"
                f"  ({m['headroom_with_extra_ms']/10:.0f}% of 1-s budget)\n"
                f"  ──────────────────────────────────────────────────────────────\n"
                f"  RSS (process RAM) : {_fmt_rss(m['rss_mb'])}"
                f"  (Δ since scroll start: {_fmt_delta(m['rss_delta_mb'])})\n"
                f"  [Shift+M for full lifecycle memory history]"
            )
        # also take a named snapshot for history
        mon = getattr(self, "_mem_monitor", None)
        if mon is not None:
            mon.snapshot(f"Shift+P  speed={self._rt_speed:.1f}x")

        if self._rt_bench_best is not None:
            best = self._rt_bench_best
            print(
                "[BENCH] "
                f"max sustained ≈ {best['achieved']:.1f}× realtime "
                f"(tested up to {self._rt_bench_max_speed:.0f}×)"
            )

    # ---------------------------
    # benchmark
    # ---------------------------
    def run_realtime_benchmark(self):
        """
        Push speed up to a "strong cap" and see what throughput you can sustain.
        - keeps normal scrolling running (so it also works during a demo)
        - reports best sustained achieved_x (x realtime)
        """
        if not self._rt_running:
            # start if not running; benchmark assumes ticks are happening
            self.start_realtime_autoscroll()

        if self._rt_bench_on:
            return  # already running

        self._rt_bench_on = True
        self._rt_bench_t0 = time.perf_counter()
        self._rt_bench_prev_speed = self._rt_speed
        self._rt_bench_best = None

        # jump to a "very strong" requested speed (cap)
        self._rt_speed = float(self._rt_bench_max_speed)

        # reset stats window so benchmark is clean
        if self._rt_enable_stats:
            self._rt_t0 = time.perf_counter()
            self._rt_ticks = 0
            self._rt_data_sum = 0.0
            self._rt_dt_max = 0.0
            self._rt_pan_cpu_sum = 0.0
            self._rt_pan_cpu_max = 0.0
            self._rt_fetch_sum = 0.0
            self._rt_fetch_max = 0.0
            self._rt_filter_sum = 0.0
            self._rt_filter_max = 0.0
            self._rt_upload_sum = 0.0
            self._rt_upload_max = 0.0
            self._rt_render_sum = 0.0
            self._rt_render_ticks = 0
            self._rt_draw_mark = None
            self._rt_mem_next_t = 0.0
            rss0 = _sample_rss_mb()
            self._rt_rss_mb_start = rss0
            self._rt_rss_mb_last = rss0
            self._rt_collect_metric_providers(reset=True)

        print(f"[BENCH] running... requesting {self._rt_speed:.0f}× for {self._rt_bench_duration_s:.1f}s")

    def _rt_maybe_finish_benchmark(self):
        if not self._rt_bench_on:
            return

        if (time.perf_counter() - self._rt_bench_t0) < self._rt_bench_duration_s:
            return

        m = self._rt_compute_metrics()
        if m:
            # "sustained" means achieved is close enough to requested (tracking high)
            if m["tracking"] >= self._rt_bench_min_good_rate:
                self._rt_bench_best = dict(req_speed=self._rt_speed, achieved=m["achieved_x"])
            else:
                # If not tracking well, still record achieved as a lower-bound headline
                self._rt_bench_best = dict(req_speed=self._rt_speed, achieved=m["achieved_x"])

        # restore previous speed and end benchmark
        self._rt_speed = float(self._rt_bench_prev_speed)
        self._rt_bench_on = False
        self._rt_bench_t0 = None
        self._rt_bench_prev_speed = None

        print("[BENCH] done.")
        # Print showcase line including benchmark summary
        self.print_realtime_showcase()

    # ---------------------------
    # hotkeys
    # ---------------------------
    def _rt_on_key_press(self, ev):
        # Guard: only handle realtime keys when a source is loaded.
        if getattr(self, "source", None) is None:
            return

        k = _event_key_name(ev)
        if k is None:
            return
        key = str(k).upper()
        shift, ctrl, meta, alt = _event_modifier_flags(ev)
        if ctrl or meta or alt:
            return

        if key == "R" and not shift:
            if self._rt_running:
                self.stop_realtime_autoscroll(reason="key:r")
            else:
                self.start_realtime_autoscroll(reason="key:r")
            ev.handled = True
            return

        if key == "S" and not shift:
            self.stop_realtime_autoscroll(reason="key:s")
            ev.handled = True
            return

        fs_map = getattr(self, "_rt_fs_presets", {})
        if not shift and str(k) in fs_map:
            self.set_realtime_fs(fs_map[str(k)], follow_source=False)
            ev.handled = True
            return

        if str(k) in ("+", "="):
            self._rt_speed *= 1.25
            ev.handled = True
            return

        if str(k) in ("-", "_"):
            self._rt_speed /= 1.25
            ev.handled = True
            return

        if key == "B" and not shift:
            self.run_realtime_benchmark()
            ev.handled = True
            return

        if key == "P":
            if shift:
                self.print_realtime_showcase()
            else:
                self.print_realtime_debug()
            ev.handled = True
            return

    # ---------------------------
    # tick
    # ---------------------------
    def _rt_on_tick(self, ev):
        if not self._rt_running:
            return
        if getattr(self, "source", None) is None or getattr(self, "camera", None) is None:
            self.stop_realtime_autoscroll(reason="source-unavailable")
            return

        now = time.perf_counter()
        if self._rt_last_t is None:
            self._rt_last_t = now
            self._rt_lag = 0.0
            return

        dt_raw = now - self._rt_last_t
        self._rt_last_t = now

        dt = dt_raw
        if dt < 0:
            dt = 0.0
        # A tick longer than max_dt advances at most max_dt (no large visual jump);
        # the excess is kept as lag and repaid over later ticks (at most max_dt
        # extra per tick), so short stalls do not slow playback. Gaps longer than
        # _RT_PAUSE_S are treated as a pause, and lag is capped at _RT_MAX_LAG_S so
        # a sustained overload degrades to slower playback instead of runaway catch-up.
        lag = self._rt_lag
        if dt > self._rt_max_dt:
            if dt <= _RT_PAUSE_S:
                lag = min(_RT_MAX_LAG_S, lag + dt - self._rt_max_dt)
            dt = self._rt_max_dt
        if lag > 0.0:
            repay = min(lag, self._rt_max_dt)
            dt += repay
            lag -= repay
        self._rt_lag = lag

        # data time advanced (seconds)
        data_dt = dt * self._rt_speed

        sx = float(getattr(self.source, "scale_x", 1.0))
        if sx <= 0:
            sx = 1.0

        dx = (self._rt_fs * data_dt) / sx

        t0 = time.perf_counter()
        self.camera.pan_x(dx)
        t1 = time.perf_counter()

        if self._rt_enable_stats and self._rt_t0 is not None:
            self._rt_ticks += 1
            self._rt_data_sum += data_dt
            self._rt_dt_max = max(self._rt_dt_max, dt)

            pan_cpu = (t1 - t0)
            self._rt_pan_cpu_sum += pan_cpu
            self._rt_pan_cpu_max = max(self._rt_pan_cpu_max, pan_cpu)

            # collect per-component breakdown from scheduler
            sched = getattr(self, "scheduler", None)
            if sched is not None:
                snap_fn = getattr(sched, "telemetry_snapshot", None)
                if callable(snap_fn):
                    sample = snap_fn(reset=True)
                else:
                    sample = {}
                fetch_ms = float(sample.get("fetch_ms", 0.0))
                filter_ms = float(sample.get("filter_ms", 0.0))
                up_ms = float(sample.get("upload_ms", 0.0))
                self._rt_fetch_sum   += fetch_ms
                self._rt_fetch_max    = max(self._rt_fetch_max, fetch_ms)
                self._rt_filter_sum  += filter_ms
                self._rt_filter_max   = max(self._rt_filter_max, filter_ms)
                self._rt_upload_sum  += up_ms
                self._rt_upload_max   = max(self._rt_upload_max, up_ms)

            # RSS sampling: at most once per second of wall time
            now_w = time.perf_counter()
            if now_w >= self._rt_mem_next_t:
                self._rt_mem_next_t = now_w + 1.0
                self._rt_rss_mb_last = _sample_rss_mb()

        # render timing is bracketed inside the draw event (canvas._rt_draw_begin/_end)
        self.update()

        # auto-stop after k wall-seconds
        if (self._rt_auto_stop_deadline is not None
                and time.perf_counter() >= self._rt_auto_stop_deadline):
            self._rt_auto_stop_deadline = None
            self.stop_realtime_autoscroll(reason="auto-stop")
            self.print_realtime_showcase()
            return

        # benchmark check (does nothing unless B was pressed)
        self._rt_maybe_finish_benchmark()


def _event_key_name(ev):
    """Return a stable key name from a VisPy key event or ``None``."""
    key = getattr(ev, "key", None)
    if key is None:
        return None
    if isinstance(key, str):
        return key
    name = getattr(key, "name", None)
    if name is not None:
        return str(name)
    text = str(key)
    return text if text else None


def _event_modifier_flags(ev) -> tuple[bool, bool, bool, bool]:
    """Return ``(shift, ctrl, meta, alt)`` from a VisPy key event."""
    mods_raw = getattr(ev, "modifiers", None)
    if not mods_raw:
        return False, False, False, False

    mods: set[str] = set()
    for modifier in mods_raw:
        name = getattr(modifier, "name", None)
        if name is not None:
            mods.add(str(name).upper())
        mods.add(str(modifier).upper())

    shift = "SHIFT" in mods
    ctrl = ("CONTROL" in mods) or ("CTRL" in mods)
    meta = "META" in mods
    alt = ("ALT" in mods) or ("OPTION" in mods)
    return shift, ctrl, meta, alt
