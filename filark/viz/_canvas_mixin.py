# Copyright (c) 2026 Jintao Li. 
# Zhejiang University (ZJU).
# 
# Licensed under the MIT License.

from vispy import app
import time


class KeyboardPanMixin:
    """
    Deterministic keyboard control.

    - Left/Right: pan X via camera (camera -> scheduler X streaming)
    - Up/Down:    pan Y via scheduler (deterministic) + optional camera Y sync
    - Y / Shift+Y: scale_y
    - X / Shift+X: scale_x
    """

    def _init_keyboard_pan(self):
        self.events.key_press.connect(self._on_key_press)

    def _on_key_press(self, ev):
        key = ev.key.upper() if isinstance(ev.key, str) else ev.key

        shift = False
        if hasattr(ev, "modifiers") and ev.modifiers:
            shift = ("Shift" in ev.modifiers) or ("SHIFT" in ev.modifiers)

        # scale
        step_scale = 1
        if key == "X":
            new_sx = (self.source.scale_x - step_scale) if shift else (self.source.scale_x + step_scale)
            new_sx = max(1.0, float(new_sx))
            self.set_scale_x(new_sx)
            ev.handled = True
            return

        if key == "Y":
            new_sy = (self.source.scale_y - step_scale) if shift else (self.source.scale_y + step_scale)
            new_sy = max(1.0, float(new_sy))
            self.set_scale_y(new_sy)
            ev.handled = True
            return

        name = ev.key.name
        if name not in ("Left", "Right", "Up", "Down", "PageUp", "PageDown"):
            return

        # temporarily disable coalesce for debug clarity
        was = getattr(self.scheduler, "_enable_coalesce", False)
        if was:
            self.scheduler.stop_coalesce()

        # step sizes
        if name in ("Left", "Right"):
            step = 100
            dx = -step if name == "Left" else step
            # only X via camera
            self.camera._pan_x(dx)  # uses camera internal method
            # camera.view_changed() is invoked by rect setter, so scheduler sees it
        else:
            if not self._wrap_y:
                # Y disabled
                pass
            else:
                if name == "Up":
                    self.scroll_channels(-20, sync_camera=True)
                elif name == "Down":
                    self.scroll_channels(+20, sync_camera=True)
                elif name == "PageUp":
                    self.page_channels(-1, sync_camera=True)
                elif name == "PageDown":
                    self.page_channels(+1, sync_camera=True)

        if was:
            self.scheduler.start_coalesce(hz=self.scheduler._coalesce_hz)

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
        # benchmark knobs (a "very strong but reasonable" cap)
        bench_max_speed=100.0,
        bench_duration_s=2.0,
        bench_min_good_rate=0.90,   # accept as "sustained" if >= 90% of requested speed
    ):
        self._rt_running = False
        self._rt_fs = float(fs)
        self._rt_speed = float(speed)
        self._rt_max_dt = float(max_dt)
        self._rt_enable_stats = bool(enable_stats)

        self._rt_timer_hz = float(max(1, int(timer_hz)))
        self._rt_interval = 1.0 / self._rt_timer_hz
        self._rt_timer = app.Timer(interval=self._rt_interval, connect=self._rt_on_tick, start=False)

        self._rt_last_t = None

        # stats (minimal)
        self._rt_t0 = None
        self._rt_ticks = 0
        self._rt_data_sum = 0.0          # sum of data seconds advanced (dt*speed)
        self._rt_dt_max = 0.0            # max dt (clamped)

        # optional: cpu time around pan (python-side only; not true draw)
        self._rt_pan_cpu_sum = 0.0
        self._rt_pan_cpu_max = 0.0

        # benchmark state
        self._rt_bench_on = False
        self._rt_bench_t0 = None
        self._rt_bench_prev_speed = None
        self._rt_bench_best = None  # dict: {"req_speed":..., "achieved":...}
        self._rt_bench_max_speed = float(bench_max_speed)
        self._rt_bench_duration_s = float(bench_duration_s)
        self._rt_bench_min_good_rate = float(bench_min_good_rate)

        self.events.key_press.connect(self._rt_on_key_press)

    # ---------------------------
    # controls
    # ---------------------------
    def start_realtime_autoscroll(self):
        if self._rt_running:
            return
        self._rt_running = True
        self._rt_last_t = None

        if self._rt_enable_stats:
            self._rt_t0 = time.perf_counter()
            self._rt_ticks = 0
            self._rt_data_sum = 0.0
            self._rt_dt_max = 0.0
            self._rt_pan_cpu_sum = 0.0
            self._rt_pan_cpu_max = 0.0

        self._rt_timer.start()
        self.update()

    def stop_realtime_autoscroll(self):
        if not self._rt_running:
            return
        self._rt_running = False
        self._rt_timer.stop()
        self._rt_last_t = None
        self.update()

    def set_realtime_fs(self, fs):
        self._rt_fs = float(fs)

    def set_realtime_speed(self, speed):
        self._rt_speed = float(speed)

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
        )

    # ---------------------------
    # printing (two modes)
    # ---------------------------
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
        k = ev.key.name

        if k == "R":
            if self._rt_running:
                self.stop_realtime_autoscroll()
            else:
                self.start_realtime_autoscroll()
            ev.handled = True
            return

        if k == "S":
            self.stop_realtime_autoscroll()
            ev.handled = True
            return

        fs_map = {"1":200,"2":500,"3":1000,"4":1500,"5":2000,"6":3000,"7":5000,"8":10000}
        if k in fs_map:
            self.set_realtime_fs(fs_map[k])
            ev.handled = True
            return

        if k in ("+", "="):
            self._rt_speed *= 1.25
            ev.handled = True
            return

        if k in ("-", "_"):
            self._rt_speed /= 1.25
            ev.handled = True
            return

        if k == "B":
            self.run_realtime_benchmark()
            ev.handled = True
            return

        if k == "P":
            mods = getattr(ev, "modifiers", None)
            shift = True
            if mods:
                shift = ("Shift" in mods) or ("shift" in mods)
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

        now = time.perf_counter()
        if self._rt_last_t is None:
            self._rt_last_t = now
            return

        dt_raw = now - self._rt_last_t
        self._rt_last_t = now

        dt = dt_raw
        if dt < 0:
            dt = 0.0
        if dt > self._rt_max_dt:
            dt = self._rt_max_dt

        # data time advanced (seconds)
        data_dt = dt * self._rt_speed

        sx = float(getattr(self.source, "scale_x", 1.0))
        if sx <= 0:
            sx = 1.0

        dx = (self._rt_fs * data_dt) / sx

        t0 = time.perf_counter()
        if hasattr(self.camera, "_pan_x"):
            self.camera._pan_x(dx)
        else:
            self.camera._pan_world(dx, 0.0)
        t1 = time.perf_counter()

        if self._rt_enable_stats and self._rt_t0 is not None:
            self._rt_ticks += 1
            self._rt_data_sum += data_dt
            self._rt_dt_max = max(self._rt_dt_max, dt)

            pan_cpu = (t1 - t0)
            self._rt_pan_cpu_sum += pan_cpu
            self._rt_pan_cpu_max = max(self._rt_pan_cpu_max, pan_cpu)

        self.update()

        # benchmark check (does nothing unless B was pressed)
        self._rt_maybe_finish_benchmark()
