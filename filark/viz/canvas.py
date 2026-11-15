# Copyright (c) 2026 Jintao Li. 
# Zhejiang University (ZJU).
# 
# Licensed under the MIT License.

"""Top-level VisPy canvas that wires the viewer image, camera, and scheduler."""

from dataclasses import dataclass
from typing import Optional

import datetime
import sys
import time as _time
import numpy as np
from vispy import scene
from vispy.visuals.transforms import STTransform

from filark._exceptions import print_exception
from filark.io.streaming import DownsampleMode, StreamingSource
from filark.utils.perf import elapsed_ms, log_perf, next_perf_id, perf_debug_enabled, perf_now
from filark.utils.memory import MemoryMonitor
from filark.utils.time import tape_sample_to_unix
from .visual import StreamingImage, AxisWidget
from .camera import StreamingCamera
from ._aspect import (
    auto_display_aspect_for_buffer,
    clamp_display_aspect,
    normalize_display_aspect_limits,
    recommended_buffer_width,
)
from .scheduler import BufferScheduler2D, ChunkEvent2D
from ._canvas_mixin import KeyboardPanMixin, RealtimeAutoScrollMixin, _event_key_name, _event_modifier_flags


def _print_exception(context: str) -> None:
    print_exception(context, prefix="StreamingCanvas")


@dataclass
class ViewState2D:
    """State required to restore current source view consistently."""
    left: float
    bottom: float
    height: float
    t0: int
    c0: int
    scale_x: float = 1.0
    scale_y: float = 1.0
    y_mode: str = "local"
    downsample_mode: str = "drop"
    buffer_h: int = 0
    buffer_w: int = 0
    auto_buffer_resize: bool = True
    display_aspect: float = 1.0
    auto_display_aspect: bool = True


class StreamingCanvas(scene.SceneCanvas, RealtimeAutoScrollMixin, KeyboardPanMixin):
    """
    Top-level canvas orchestrator for DAS streaming visualization.

    StreamingCanvas is the ONLY owner that creates and wires together:
      - View
      - StreamingImageVisual
      - StreamingCamera
      - BufferScheduler2D

    Responsibilities:
      - Construct and connect camera ↔ scheduler
      - Expose deterministic public APIs for navigation and scaling
      - Own keyboard / realtime auto-scroll behaviors
      - Handle delayed initialization after first draw / resize

    Parameters (summary)
    --------------------
    source : array-like or StreamingSource
        Data source for streaming.
    buffer_h, buffer_w : int or None
        Streaming buffer size.
    wrap_y : bool
        Enable channel (Y) streaming.
    x_align, x_margin : int
        X streaming alignment and safety margin.
    """

    def __init__(
        self,
        source=None,
        size=(1200, 600),
        buffer_h=None,
        buffer_w=None,
        display_aspect=None,
        display_aspect_limits=None,
        auto_display_aspect=True,
        world_origin=(0.0, 0.0),
        world_scale=(1.0, 1.0),
        axis_units=("", ""),
        axis_labels=None,
        invert_y=False,
        title="",
        wrap_y=False,
        cmap="seismic",
        clim=(-5.0, 5.0),
        interpolation="nearest",
        keep_cpu=True,
        x_align=256,
        x_margin=512,
        enable_coalesce=False,
        coalesce_hz=60,
        async_io=False,
        auto_buffer_resize=True,
        default_buffer_h=2048,
        realtime_autoscroll=None,
        bgcolor="black",
        **kwargs,
    ):
        # Keep vispy default key handling for non-GUI use, but allow caller override
        # (e.g. keys=None in PySide GUI to disable Esc closing behavior).
        canvas_keys = kwargs.pop("keys", "interactive")
        super().__init__(keys=canvas_keys, size=size, bgcolor=bgcolor, title="FiLark Viewer", **kwargs)
        self.unfreeze()

        # ---- store config (no data) ----
        self._wrap_y = bool(wrap_y)
        self._world_origin = world_origin
        self._world_scale = world_scale
        self._axis_units = axis_units
        self._invert_y = bool(invert_y)
        self._title = title
        self._cmap = cmap
        self._clim = clim
        self._interpolation = interpolation
        self._keep_cpu = bool(keep_cpu)

        self._x_align = int(x_align)
        self._x_margin = int(x_margin)
        self._enable_coalesce = bool(enable_coalesce)
        self._coalesce_hz = int(coalesce_hz)
        self._async_io = bool(async_io)
        self._auto_buffer_resize = bool(auto_buffer_resize)
        self._default_buffer_h = int(max(1, default_buffer_h or 2048))
        self._buffer_resize_inflight = False
        self._auto_display_aspect = bool(auto_display_aspect)
        self._display_aspect_limits = normalize_display_aspect_limits(display_aspect_limits)
        self._display_aspect = clamp_display_aspect(
            1.0 if display_aspect is None else display_aspect,
            limits=self._display_aspect_limits,
        )
        self._effective_display_aspect = float(self._display_aspect)
        self._state_listeners = []
        self._chunk_listeners = []
        self._perf_pending_draws: list[dict] = []

        default_axis_labels = ["Time", "Channels"]
        if axis_units[0]:
            default_axis_labels[0] = f"Time ({axis_units[0]})"
        if axis_units[1]:
            default_axis_labels[1] = f"Distance ({axis_units[1]})"
        if axis_labels is None:
            axis_label_overrides = (None, None)
        else:
            if len(axis_labels) != 2:
                raise ValueError("axis_labels must contain exactly two labels")
            axis_label_overrides = tuple(None if label is None else str(label) for label in axis_labels)
        self._axis_label_overrides = axis_label_overrides
        self._axis_labels = [
            default if override is None else override
            for default, override in zip(default_axis_labels, self._axis_label_overrides)
        ]

        # Coordinate display mode (persists across source switches)
        self._x_mode: str = "sample"    # "sample" | "time" | "utc"
        self._y_mode: str = "channel"   # "channel" | "distance"
        self._start_time: float | None = None  # Unix timestamp of first sample
        # Reference sample for UTC axis (left-edge of current viewport).
        # Updated on every camera transform_changed event in UTC mode.
        self._utc_ref_sample: int = 0

        # ---- layout ----
        self.grid = self.central_widget.add_grid(margin=10)
        self.grid.spacing = 0
        border_color = "white" if bgcolor == "black" else "black"
        self.view = self.grid.add_view(row=1, col=1, border_color=border_color)

        # ---- image: create as "empty" placeholder ----
        # init with shape (1, 1)
        self.image = StreamingImage(
            shape=(1, 1),
            cmap=self._cmap,
            clim=self._clim,
            gamma=1.0,
            interpolation=self._interpolation,
            keep_cpu=self._keep_cpu,
            base=None,
            wrap_y=self._wrap_y,
            parent=self.view.scene,
        )
        self.image.order = 0
        self.image.visible = False
        self.image.transform = STTransform(translate=(0.0, 0.0))

        # ---- World layers: committed overlays below, selection/preview UI above ------
        self.world_layer = scene.Node(parent=self.view.scene)
        self.world_layer.order = 10
        self.world_layer.transform = STTransform(scale=(1.0, 1.0, 1.0))
        self.world_data_layer = scene.Node(parent=self.world_layer)
        self.world_data_layer.order = 20
        self.world_ui_layer = scene.Node(parent=self.world_layer)
        self.world_ui_layer.order = 30

        # ---- Camera ----
        self.camera = StreamingCamera(
            ymin=0, ymax=1,
            xmin=0, xmax=1,
            y_home_height=1.0,
            wrap_y=self._wrap_y,
            interactive=True,
            display_aspect=self._effective_display_aspect,
            display_aspect_limits=self._display_aspect_limits,
            flip=(False, self._invert_y, False),
        )
        self.view.camera = self.camera
        self.camera.events.transform_change.connect(self._on_view_changed)

        # ---- scheduler initial ----
        self.scheduler = None
        self.source = None
        self.nc = self.nt = 0
        self.buffer_h = self.buffer_w = 0

        # ---- axis ----
        self._init_axis()

        if hasattr(self, "_init_keyboard_pan"):
            self._init_keyboard_pan()
        if hasattr(self, "_init_realtime_autoscroll"):
            self._init_realtime_autoscroll(**dict(realtime_autoscroll or {}))

        # Memory monitor: tracks RSS + VRAM at key lifecycle events.
        # Press M at any time to print current usage; Shift+M for full history.
        self._mem_monitor = MemoryMonitor(print_on_snapshot=False)
        self._mem_monitor.snapshot("canvas init")
        self.events.key_press.connect(self._on_mem_key)

        # delayed init: waiting source
        self._did_init = False
        self.events.resize.connect(self._on_canvas_resize)
        self.events.draw.connect(self._try_init_once)
        self.events.draw.connect(self._on_draw_perf_event)

        # Cursor position HUD (toggled with C key)
        self._cursor_pos_visible = False
        self.events.mouse_move.connect(self._on_mouse_move_cursor)
        self._update_cursor_label_pos()

        # Channel markers: list of (source_channel_index, Line visual)
        self._channel_markers: list = []
        self._gap_overlays: list = []

        self.freeze()

        if source is not None:
            self.set_source(source, buffer_h=buffer_h, buffer_w=buffer_w)

    def add_state_listener(self, fn) -> None:
        if fn not in self._state_listeners:
            self._state_listeners.append(fn)

    def remove_state_listener(self, fn) -> bool:
        try:
            self._state_listeners.remove(fn)
            return True
        except ValueError:
            return False

    def _emit_state_changed(self) -> None:
        if not self._state_listeners:
            return
        state = self.current_scale_state()
        for fn in tuple(self._state_listeners):
            try:
                fn(state)
            except Exception:
                _print_exception("Failed to notify a canvas state listener.")

    def initialize_if_ready(self) -> None:
        """Run delayed initialization if the canvas has enough size information."""
        self._try_init_once()

    def _close_scheduler(self) -> None:
        scheduler = self.scheduler
        if scheduler is None:
            return
        try:
            if self.camera is not None:
                self.camera.remove_view_listener(scheduler.on_camera_changed)
        except Exception:
            _print_exception("Failed to remove scheduler camera listener.")
        try:
            scheduler.close()
        finally:
            self.scheduler = None

    def mutate_source(self, fn):
        if self.source is None:
            return None
        scheduler = self.scheduler
        if scheduler is not None and hasattr(scheduler, "mutate_source"):
            return scheduler.mutate_source(lambda _src: fn(self.source))
        return fn(self.source)

    def _overlay_visual_count(self) -> int:
        node = getattr(self, "world_data_layer", None)
        if node is None:
            return 0
        return int(len(getattr(node, "children", []) or []))

    def _mark_next_draw(self, reason: str, **fields) -> None:
        if not perf_debug_enabled():
            return
        payload = {
            "trace": next_perf_id("draw"),
            "reason": str(reason),
            "start": perf_now(),
            "overlay_count": self._overlay_visual_count(),
        }
        payload.update(fields)
        self._perf_pending_draws.append(payload)

    def on_draw(self, event):
        # Realtime "render" = main-thread time to issue the scene's draw commands
        # (SceneCanvas.on_draw). The following GL flush / buffer swap is excluded:
        # there the driver paces frames to the display and its wait expands into
        # any idle time, so it is not a fixed per-frame cost. Queue wait before
        # the draw event is excluded as well.
        timing = getattr(self, "_rt_enable_stats", False) and getattr(self, "_rt_t0", None) is not None
        t0 = _time.perf_counter() if timing else 0.0
        super().on_draw(event)
        if timing:
            self._rt_render_sum += (_time.perf_counter() - t0) * 1000.0
            self._rt_render_ticks += 1

    def _on_draw_perf_event(self, event=None) -> None:
        if not perf_debug_enabled():
            self._perf_pending_draws.clear()
            return
        if not self._perf_pending_draws:
            return
        pending = self._perf_pending_draws[:]
        self._perf_pending_draws.clear()
        for item in pending:
            start = float(item.pop("start", perf_now()))
            item["overlay_count"] = int(self._overlay_visual_count())
            item["ms"] = elapsed_ms(start)
            log_perf("canvas.draw", **item)

    def _viewport_pixel_size(self) -> tuple[float, float]:
        viewbox = getattr(self.camera, "_viewbox", None)
        rect = getattr(viewbox, "rect", None)
        if rect is not None and float(rect.width) > 1.0 and float(rect.height) > 1.0:
            return float(rect.width), float(rect.height)
        size = getattr(self, "size", None) or (1, 1)
        return float(max(1, size[0])), float(max(1, size[1]))

    def _resolve_display_aspect(self, buffer_h: Optional[int] = None) -> float:
        if not self._auto_display_aspect:
            return float(self._display_aspect)
        if self.source is None and (buffer_h is None and self.buffer_h <= 0):
            return float(self._display_aspect)
        bh = int(buffer_h if buffer_h is not None else max(1, self.buffer_h))
        _, viewport_h = self._viewport_pixel_size()
        return auto_display_aspect_for_buffer(bh, viewport_h, limits=self._display_aspect_limits)

    def _sync_display_aspect(self, *, buffer_h: Optional[int] = None, reapply: bool = True) -> bool:
        eff = self._resolve_display_aspect(buffer_h=buffer_h)
        if abs(float(eff) - float(self._effective_display_aspect)) <= 1e-9:
            return False
        self._effective_display_aspect = float(eff)
        if getattr(self, "camera", None) is not None:
            self.camera.set_display_aspect(self._effective_display_aspect, reapply=reapply)
        return True

    def current_display_aspect(self) -> float:
        return float(self._effective_display_aspect)

    def current_display_aspect_limits(self) -> tuple[float, float]:
        return tuple(self._display_aspect_limits)

    def _reapply_camera_rect(self) -> None:
        if self.camera is None:
            return
        rect = self.camera.rect
        self.camera.apply_rect(
            left=float(rect.left),
            bottom=float(rect.bottom),
            height=float(rect.height),
            notify=True,
        )
        self.update()

    def _apply_display_aspect_change(self, *, preserve_view: bool = True) -> None:
        changed = self._sync_display_aspect(reapply=False)
        if self.source is None:
            if changed:
                self._reapply_camera_rect()
                self._emit_state_changed()
            return

        if not self._buffer_resize_inflight:
            new_bw = self._recommended_buffer_w(self.buffer_h)
            need_resize = (new_bw != self.buffer_w) if self._auto_buffer_resize else (new_bw > self.buffer_w)
            if need_resize:
                self.set_buffer_shape(buffer_h=self.buffer_h, buffer_w=new_bw, preserve_view=preserve_view)
                return

        if changed:
            self._reapply_camera_rect()
            self._emit_state_changed()

    def set_display_aspect(self, display_aspect: float) -> None:
        was_auto = self._auto_display_aspect
        prev_effective = float(self.current_display_aspect())
        new_aspect = clamp_display_aspect(display_aspect, limits=self._display_aspect_limits)
        unchanged = (not self._auto_display_aspect) and (abs(new_aspect - self._display_aspect) <= 1e-9)
        self._display_aspect = new_aspect
        self._auto_display_aspect = False
        if unchanged:
            return
        self._apply_display_aspect_change(preserve_view=True)
        if was_auto and abs(prev_effective - float(self.current_display_aspect())) <= 1e-9:
            self._emit_state_changed()

    def set_display_aspect_limits(self, display_aspect_limits: tuple[float, float] | None) -> None:
        new_limits = normalize_display_aspect_limits(display_aspect_limits)
        if new_limits == self._display_aspect_limits:
            return
        prev_effective = float(self.current_display_aspect())
        self._display_aspect_limits = new_limits
        self._display_aspect = clamp_display_aspect(self._display_aspect, limits=self._display_aspect_limits)
        if getattr(self, "camera", None) is not None:
            self.camera.set_display_aspect_limits(self._display_aspect_limits, reapply=False)
        self._apply_display_aspect_change(preserve_view=True)
        if abs(prev_effective - float(self.current_display_aspect())) <= 1e-9:
            self._emit_state_changed()

    def set_auto_display_aspect(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._auto_display_aspect:
            return
        prev_effective = float(self.current_display_aspect())
        if not enabled:
            self._display_aspect = float(self.current_display_aspect())
        self._auto_display_aspect = enabled
        self._apply_display_aspect_change(preserve_view=True)
        if abs(prev_effective - float(self.current_display_aspect())) <= 1e-9:
            self._emit_state_changed()


    def set_source(
        self,
        source,
        *,
        buffer_h=None,
        buffer_w=None,
        reset_view=True,
    ):
        perf_on = perf_debug_enabled()
        total_start = perf_now() if perf_on else 0.0
        self.unfreeze()
        self.camera.interactive = True
        if getattr(self, "xaxis", None) is not None:
            self.xaxis.visible = True
        if getattr(self, "yaxis", None) is not None:
            self.yaxis.visible = True
        prev_view: ViewState2D | None = None
        if not reset_view:
            prev_view = self.capture_view_state()
        self._close_scheduler()

        # 1) wrap source
        wrap_start = perf_now() if perf_on else 0.0
        if not isinstance(source, StreamingSource):
            dims = "auto"
            if hasattr(source, "dims"):
                dims = source.dims
            source = StreamingSource(source, dims=dims)
        wrap_ms = elapsed_ms(wrap_start) if perf_on else 0.0
        self.source = source
        self.nc, self.nt = self.source.shape

        # 2) decide buffer sizes
        buffer_start = perf_now() if perf_on else 0.0
        if buffer_h is None:
            if (not reset_view) and self.buffer_h > 0:
                buffer_h = min(self.nc, self.buffer_h)
            else:
                buffer_h = min(self.nc, self._default_buffer_h)
        self.buffer_h = int(buffer_h)
        self._sync_display_aspect(buffer_h=self.buffer_h, reapply=False)

        if buffer_w is None:
            if (not reset_view) and (not self._auto_buffer_resize) and self.buffer_w > 0:
                buffer_w = self.buffer_w
            else:
                buffer_w = self._recommended_buffer_w(self.buffer_h)
        self.buffer_w = int(buffer_w)
        buffer_ms = elapsed_ms(buffer_start) if perf_on else 0.0

        # 3) tell source buffer size
        source_cfg_start = perf_now() if perf_on else 0.0
        self.source.set_buffer_size(self.buffer_h, self.buffer_w)
        source_cfg_ms = elapsed_ms(source_cfg_start) if perf_on else 0.0

        # 4) resize image texture
        resize_start = perf_now() if perf_on else 0.0
        self.image.resize((self.buffer_h, self.buffer_w), fill=0.0)
        self.image.visible = True
        resize_ms = elapsed_ms(resize_start) if perf_on else 0.0

        # 5) update camera bounds from current source scale
        bounds_start = perf_now() if perf_on else 0.0
        self._sync_camera_bounds_from_source()
        bounds_ms = elapsed_ms(bounds_start) if perf_on else 0.0

        # 6) rebuild scheduler
        scheduler_start = perf_now() if perf_on else 0.0
        self.scheduler = BufferScheduler2D(
            source=self.source,
            image=self.image,
            camera=self.camera,
            buffer_h=self.buffer_h,
            buffer_w=self.buffer_w,
            x_align=self._x_align,
            x_margin=self._x_margin,
            y_align=1,
            wrap_y=self._wrap_y,
            enable_coalesce=self._enable_coalesce,
            coalesce_hz=self._coalesce_hz,
            async_io=self._async_io,
            request_draw=self.update,
        )
        for fn in tuple(self._chunk_listeners):
            self.scheduler.add_chunk_listener(fn)
        self.camera.add_view_listener(self.scheduler.on_camera_changed)
        scheduler_ms = elapsed_ms(scheduler_start) if perf_on else 0.0

        # 7) reset pan window / wrap bottom
        view_setup_start = perf_now() if perf_on else 0.0
        self.scheduler.begin_camera_event_suppression()
        try:
            self.camera.set_y_pan_window(0, min(self.buffer_h, self.nc))
            if self._wrap_y and reset_view:
                self.camera.set_y_bottom(0)
        finally:
            self.scheduler.end_camera_event_suppression()
        self._sync_display_aspect(buffer_h=self.buffer_h, reapply=False)

        # 8) axis scale + coordinate mode
        self.xaxis.update_scale(float(self.source.scale_x))
        self.yaxis.update_scale(float(self.source.scale_y))
        self._update_world_layer_transform()
        tape = getattr(self.source, "data", None)
        if tape is None:
            tape = self.source
        st = getattr(tape, "start_time", None)
        self._start_time = float(st) if st is not None else None
        self._apply_axis_mode()
        if hasattr(self, "_rt_sync_source_fs"):
            self._rt_sync_source_fs()

        # 9/10) init policy: either reset view, or preserve previous camera window
        if reset_view:
            self._did_init = False
            self._try_init_once()
        else:
            if prev_view is None:
                self._did_init = False
                self._try_init_once()
            else:
                self.apply_view_state(prev_view)
        view_setup_ms = elapsed_ms(view_setup_start) if perf_on else 0.0

        self.freeze()
        self._update_channel_marker_positions()
        self._update_gap_overlays()
        self._mark_next_draw(
            "set_source",
            reset_view=int(reset_view),
            buffer_h=int(self.buffer_h),
            buffer_w=int(self.buffer_w),
            nt=int(self.nt),
            nc=int(self.nc),
        )
        self.update()
        self._emit_state_changed()
        # Memory snapshot after source is loaded (always, regardless of perf mode)
        self._mem_monitor.snapshot(f"set_source  nc={self.nc} nt={self.nt}")
        if perf_on:
            log_perf(
                "canvas.set_source",
                reset_view=int(reset_view),
                nt=int(self.nt),
                nc=int(self.nc),
                buffer_h=int(self.buffer_h),
                buffer_w=int(self.buffer_w),
                wrap_ms=float(wrap_ms),
                buffer_ms=float(buffer_ms),
                source_cfg_ms=float(source_cfg_ms),
                resize_ms=float(resize_ms),
                bounds_ms=float(bounds_ms),
                scheduler_ms=float(scheduler_ms),
                view_setup_ms=float(view_setup_ms),
                ms=elapsed_ms(total_start),
            )

    def capture_view_state(self) -> ViewState2D | None:
        if self.scheduler is None or self.camera is None:
            return None
        try:
            r = self.camera.rect
            return ViewState2D(
                left=float(r.left),
                bottom=float(r.bottom),
                height=float(r.height),
                t0=int(self.scheduler.t0),
                c0=int(self.scheduler.c0),
                scale_x=float(getattr(self.source, "scale_x", 1.0)),
                scale_y=float(getattr(self.source, "scale_y", 1.0)),
                y_mode=str(getattr(self.source, "y_mode", "local")),
                downsample_mode=str(getattr(self.source, "downsample_mode", "drop")),
                buffer_h=int(self.buffer_h),
                buffer_w=int(self.buffer_w),
                auto_buffer_resize=bool(self._auto_buffer_resize),
                display_aspect=float(self.current_display_aspect()),
                auto_display_aspect=bool(self._auto_display_aspect),
            )
        except Exception:
            _print_exception("Failed to capture current canvas view state.")
            return None

    def apply_view_state(self, state: ViewState2D) -> None:
        if self.scheduler is None or self.camera is None:
            return

        sx = max(1.0, float(state.scale_x))
        sy = max(1.0, float(state.scale_y))
        ym = "fit_all" if str(state.y_mode) == "fit_all" else "local"
        dm = str(getattr(state, "downsample_mode", "drop"))
        self._auto_buffer_resize = bool(getattr(state, "auto_buffer_resize", self._auto_buffer_resize))
        self._display_aspect = clamp_display_aspect(
            getattr(state, "display_aspect", self._display_aspect),
            limits=self._display_aspect_limits,
        )
        self._auto_display_aspect = bool(getattr(state, "auto_display_aspect", self._auto_display_aspect))
        self._sync_display_aspect(buffer_h=self.buffer_h, reapply=False)

        try:
            self.mutate_source(lambda src: src.set_scale(scale_x=sx, scale_y=sy, y_mode=ym, downsample_mode=dm))
        except Exception:
            _print_exception("Failed to restore full view state scale; falling back to scale_x/scale_y only.")
            self.mutate_source(lambda src: src.set_scale(scale_x=sx, scale_y=sy))

        self.xaxis.update_scale(float(self.source.scale_x))
        self.yaxis.update_scale(float(self.source.scale_y))
        self._sync_camera_bounds_from_source()
        self._update_world_layer_transform()

        c0 = self._clamp_c0_view(int(state.c0))

        camera_rect = (float(state.left), float(state.bottom), float(state.height))
        t0 = self._buffer_t0_for_camera_rect(camera_rect)
        self._did_init = True
        self.scheduler.reset_full(
            t0,
            c0,
            reason="apply_view_state",
            camera_rect=camera_rect,
            y_pan_window=(c0, c0 + self.buffer_h),
        )

    def _sync_camera_bounds_from_source(self) -> None:
        if self.source is None:
            return

        sx = max(1e-6, float(getattr(self.source, "scale_x", 1.0)))
        sy = max(1e-6, float(getattr(self.source, "scale_y", 1.0)))
        view_nt = max(1.0, float(self.nt) / sx)
        view_nc = max(1.0, float(self.nc) / sy)

        self.camera.configure_bounds(
            xmin=0.0,
            xmax=view_nt,
            ymin=0.0,
            ymax=view_nc,
            y_home_height=float(min(self.buffer_h, view_nc)),
        )

    def _clamp_t0_view(self, t0: int) -> int:
        if self.source is None:
            return int(t0)
        sx = max(1e-6, float(getattr(self.source, "scale_x", 1.0)))
        view_nt = max(1, int(np.ceil(float(self.nt) / sx)))
        if view_nt <= self.buffer_w:
            return 0
        return int(max(0, min(int(t0), view_nt - self.buffer_w)))

    def _buffer_t0_for_camera_rect(self, rect=None) -> int:
        if self.scheduler is None:
            return 0
        if rect is None:
            rect = self.camera.rect
        sx = max(1e-6, float(getattr(self.source, "scale_x", 1.0)))
        view_nt = int(np.ceil(float(self.nt) / sx))
        if isinstance(rect, (tuple, list)):
            left = float(rect[0])
            height = float(rect[2])
            view_width = max(1.0, float(self.camera.width_from_height(height)))
            if view_width >= float(view_nt):
                left = 0.0
            else:
                left = max(0.0, min(left, float(view_nt) - view_width))
        else:
            left = float(rect.left)
            view_width = float(rect.width) if float(rect.width) > 1.0 else float(max(1, self.buffer_w))
        target_t0 = self.scheduler.align_start(
            left + 0.5 * view_width - 0.5 * float(self.buffer_w),
            self.scheduler.x_align,
            0,
            max(0, view_nt - self.buffer_w),
        )
        return self._clamp_t0_view(int(target_t0))

    def _clamp_c0_view(self, c0: int) -> int:
        if self.source is None:
            return int(c0)
        if not self._wrap_y:
            return 0
        if str(getattr(self.source, "y_mode", "local")) == "fit_all":
            return 0

        sy = max(1e-6, float(getattr(self.source, "scale_y", 1.0)))
        view_nc = max(1, int(np.ceil(float(self.nc) / sy)))
        if view_nc <= self.buffer_h:
            return 0
        return int(max(0, min(int(c0), view_nc - self.buffer_h)))



    # -------------------------
    # Axis
    # -------------------------
    def _axis_label_or_default(self, index: int, default: str) -> str:
        override = self._axis_label_overrides[int(index)]
        return str(default) if override is None else str(override)

    def _axis_cursor_name(self, index: int, default: str) -> str:
        label = self._axis_label_or_default(index, default)
        return str(label).split("(", 1)[0].strip() or str(default)

    def _get_win_dpr(self) -> float:
        """Return DPR on Windows only — macOS/Linux handle HiDPI in VisPy itself."""
        if sys.platform != "win32":
            return 1.0
        try:
            from PySide6.QtGui import QGuiApplication
            screen = QGuiApplication.primaryScreen()
            if screen is not None:
                return max(1.0, float(screen.devicePixelRatio()))
        except Exception:
            pass
        try:
            return max(1.0, float(self.native.devicePixelRatio()))
        except Exception:
            pass
        return 1.0

    def _init_axis(self):
        dpr = self._get_win_dpr()
        font_size = round(12.0 / dpr, 1)

        text_color_title = "white" if self.bgcolor.rgba[0] < 0.5 else "black"
        self.title_label = scene.Label(self._title, color=text_color_title, font_size=font_size)
        self.title_label.height_max = 20
        self.grid.add_widget(self.title_label, row=0, col=0, col_span=2)

        r, g, b = self.bgcolor.rgba[:3]
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = "black" if luminance > 0.5 else "white"
        tick_color = (0.3, 0.3, 0.3) if luminance > 0.5 else (0.7, 0.7, 0.7)
        axis_color = (0, 0, 0) if luminance > 0.5 else (1, 1, 1)

        dz = self._world_scale[1]
        self.yaxis = AxisWidget(
            orientation="left",
            transform=lambda y: f"{float(y) * dz:g}",
            unit=self._axis_units[1],
            axis_label=self._axis_labels[1],
            axis_font_size=font_size,
            axis_label_margin=50,
            tick_label_margin=5,
            text_color=text_color,
            axis_color=axis_color,
            tick_color=tick_color,
        )
        self.yaxis.width_max = 80
        self.grid.add_widget(self.yaxis, row=1, col=0)

        fs = self._world_scale[0]
        self.xaxis = AxisWidget(
            orientation="bottom",
            transform=lambda t: f"{float(t) / fs:g}",
            unit=self._axis_units[0],
            axis_label=self._axis_labels[0],
            axis_font_size=font_size,
            axis_label_margin=50,
            tick_label_margin=20,
            text_color=text_color,
            axis_color=axis_color,
            tick_color=tick_color,
        )
        self.xaxis.height_max = 60
        self.grid.add_widget(self.xaxis, row=2, col=1)

        self.xaxis.link_view(self.view)
        self.yaxis.link_view(self.view)

        # Cursor position HUD: floating text, right-aligned at the same level as
        # the x-axis title ("Time (samples)"), i.e. bottom-right of the canvas.
        self.cursor_label = scene.Text(
            "",
            color="yellow",
            font_size=font_size-4,
            anchor_x="right",
            anchor_y="center",
            parent=self.scene,
        )
        self.cursor_label.visible = False

    # -------------------------
    # delayed init
    # -------------------------
    def _try_init_once(self, event=None):
        if self._did_init:
            return
        if self.source is None or self.scheduler is None:
            return

        ok = self.camera.initialize_view(x0=0)
        if not ok:
            self.update()
            return

        self._did_init = True

        # trigger one scheduler sync and draw
        self.camera.view_changed()
        self.update()

    def _update_cursor_label_pos(self) -> None:
        """Pin cursor HUD to the right side of the x-axis label row."""
        w, h = self.size
        # xaxis height_max=60; place at vertical center of that row
        self.cursor_label.pos = (w - 8, h - 30)

    def _on_canvas_resize(self, event=None):
        self._try_init_once(event)
        self._update_cursor_label_pos()
        aspect_changed = self._sync_display_aspect(reapply=False)
        if self.source is None or self._buffer_resize_inflight:
            if aspect_changed:
                self._reapply_camera_rect()
            return
        if not self._auto_buffer_resize:
            if aspect_changed:
                new_bw = self._recommended_buffer_w(self.buffer_h)
                if new_bw > self.buffer_w:
                    self.set_buffer_shape(buffer_h=self.buffer_h, buffer_w=new_bw, preserve_view=True)
                else:
                    self._reapply_camera_rect()
                    self._emit_state_changed()
            return
        new_bw = self._recommended_buffer_w(self.buffer_h)
        if new_bw != self.buffer_w:
            self.set_buffer_shape(buffer_h=self.buffer_h, buffer_w=new_bw, preserve_view=True)
        elif aspect_changed:
            self._reapply_camera_rect()
            self._emit_state_changed()

    def refresh_buffer(self):
        """
        Re-read and redraw the current scheduler window.
        Useful after changing preprocess/postprocess callbacks.
        """
        if self.scheduler is None:
            return
        perf_on = perf_debug_enabled()
        total_start = perf_now() if perf_on else 0.0
        self.scheduler.reset_full(self.scheduler.t0, self.scheduler.c0, reason="refresh_buffer")
        self._mark_next_draw(
            "refresh_buffer",
            t0=int(self.scheduler.t0),
            c0=int(self.scheduler.c0),
            buffer_h=int(self.buffer_h),
            buffer_w=int(self.buffer_w),
        )
        self.update()
        if perf_on:
            log_perf(
                "canvas.refresh_buffer",
                t0=int(self.scheduler.t0),
                c0=int(self.scheduler.c0),
                buffer_h=int(self.buffer_h),
                buffer_w=int(self.buffer_w),
                ms=elapsed_ms(total_start),
            )

    def _recommended_buffer_w(self, buffer_h: int) -> int:
        return recommended_buffer_width(
            buffer_h=int(buffer_h),
            viewport_size=self._viewport_pixel_size(),
            x_margin=self._x_margin,
            x_align=self._x_align,
            display_aspect=self._resolve_display_aspect(buffer_h=int(buffer_h)),
            display_aspect_limits=self._display_aspect_limits,
        )

    def set_buffer_shape(
        self,
        *,
        buffer_h: Optional[int] = None,
        buffer_w: Optional[int] = None,
        auto_resize: Optional[bool] = None,
        preserve_view: bool = True,
    ) -> None:
        if auto_resize is not None:
            self._auto_buffer_resize = bool(auto_resize)
        if self.source is None:
            return

        new_bh = int(buffer_h if buffer_h is not None else self.buffer_h)
        new_bh = max(1, min(self.nc, new_bh))
        if buffer_w is not None:
            new_bw = int(buffer_w)
        elif self._auto_buffer_resize:
            new_bw = self._recommended_buffer_w(new_bh)
        else:
            new_bw = int(self.buffer_w)
        new_bw = max(1, new_bw)

        if new_bh == self.buffer_h and new_bw == self.buffer_w:
            return

        self._buffer_resize_inflight = True
        try:
            self.set_source(self.source, buffer_h=new_bh, buffer_w=new_bw, reset_view=not preserve_view)
        finally:
            self._buffer_resize_inflight = False

    def set_buffer_h(self, buffer_h: int) -> None:
        self.set_buffer_shape(buffer_h=buffer_h, preserve_view=True)

    def set_buffer_w(self, buffer_w: int) -> None:
        self.set_buffer_shape(buffer_w=buffer_w, preserve_view=True)

    def set_auto_buffer_resize(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._auto_buffer_resize:
            return
        self.set_buffer_shape(auto_resize=enabled, buffer_w=None if enabled else self.buffer_w, preserve_view=True)

    def current_scale_state(self) -> dict[str, float | str | bool | int]:
        if self.source is None:
            return {
                "scale_x": 1.0,
                "scale_y": 1.0,
                "y_mode": "local",
                "downsample_mode": "drop",
                "buffer_h": int(self.buffer_h or self._default_buffer_h),
                "buffer_w": 0,
                "auto_buffer_resize": bool(self._auto_buffer_resize),
                "display_aspect": float(self.current_display_aspect()),
                "auto_display_aspect": bool(self._auto_display_aspect),
                "display_aspect_min": float(self._display_aspect_limits[0]),
                "display_aspect_max": float(self._display_aspect_limits[1]),
            }
        return {
            "scale_x": float(getattr(self.source, "scale_x", 1.0)),
            "scale_y": float(getattr(self.source, "scale_y", 1.0)),
            "y_mode": str(getattr(self.source, "y_mode", "local")),
            "downsample_mode": str(getattr(self.source, "downsample_mode", "drop")),
            "buffer_h": int(self.buffer_h),
            "buffer_w": int(self.buffer_w),
            "auto_buffer_resize": bool(self._auto_buffer_resize),
            "display_aspect": float(self.current_display_aspect()),
            "auto_display_aspect": bool(self._auto_display_aspect),
            "display_aspect_min": float(self._display_aspect_limits[0]),
            "display_aspect_max": float(self._display_aspect_limits[1]),
        }

    def current_view_bounds(self) -> dict[str, int | float | str] | None:
        """Return current visible source bounds in semantic sample/channel indices."""
        if self.source is None or self.camera is None:
            return None
        try:
            rect = self.camera.rect
            sx = max(1e-6, float(getattr(self.source, "scale_x", 1.0)))
            sy = max(1e-6, float(getattr(self.source, "scale_y", 1.0)))
            y_mode = str(getattr(self.source, "y_mode", "local"))
            rect_width = float(rect.width) if float(rect.width) > 1.0 else float(max(1, self.buffer_w))
            rect_height = float(rect.height) if float(rect.height) > 1.0 else float(max(1, self.buffer_h))

            t0 = max(0, int(np.floor(float(rect.left) * sx)))
            t1 = min(int(self.nt), int(np.ceil(float(rect.left + rect_width) * sx)))
            if t1 <= t0:
                t1 = min(int(self.nt), t0 + 1)

            if y_mode == "fit_all":
                c0 = 0
                c1 = int(self.nc)
            else:
                c0 = max(0, int(np.floor(float(rect.bottom) * sy)))
                c1 = min(int(self.nc), int(np.ceil(float(rect.bottom + rect_height) * sy)))
                if c1 <= c0:
                    c1 = min(int(self.nc), c0 + 1)

            return {
                "t0": int(t0),
                "t1": int(t1),
                "c0": int(c0),
                "c1": int(c1),
                "scale_x": float(sx),
                "scale_y": float(sy),
                "y_mode": y_mode,
            }
        except Exception:
            _print_exception("Failed to compute current canvas view bounds.")
            return None

    def add_chunk_listener(self, fn) -> None:
        """Receive processed display chunks as ``ChunkEvent2D`` objects."""
        if fn not in self._chunk_listeners:
            self._chunk_listeners.append(fn)
        if self.scheduler is not None:
            self.scheduler.add_chunk_listener(fn)

    def remove_chunk_listener(self, fn) -> bool:
        removed = False
        try:
            self._chunk_listeners.remove(fn)
            removed = True
        except ValueError:
            pass
        if self.scheduler is not None:
            removed = self.scheduler.remove_chunk_listener(fn) or removed
        return removed

    def current_display_chunk_event(self) -> ChunkEvent2D | None:
        """Return a full-buffer event for the currently displayed logical buffer."""
        if self.scheduler is None or self.source is None:
            return None
        snap = getattr(self.image, "snapshot_logical", None)
        if not callable(snap):
            return None
        block = snap()
        if block is None:
            return None
        sx = float(getattr(self.source, "scale_x", 1.0)) or 1.0
        sy = float(getattr(self.source, "scale_y", 1.0)) or 1.0
        h = int(block.shape[0])
        w = int(block.shape[1])
        view_t0 = int(self.scheduler.t0)
        view_c0 = int(self.scheduler.c0)
        return ChunkEvent2D(
            kind="current",
            block=np.asarray(block),
            t0=int(view_t0 * sx),
            t1=int((view_t0 + w) * sx),
            c0=int(view_c0 * sy),
            c1=int((view_c0 + h) * sy),
            scale_x=float(sx),
            scale_y=float(sy),
            view_t0=view_t0,
            view_t1=view_t0 + w,
            view_c0=view_c0,
            view_c1=view_c0 + h,
        )

    # -------------------------
    # deterministic Y APIs (GUI/keyboard)
    # -------------------------
    def show_channel_range(self, c0_raw: int, c1_raw: int) -> None:
        """
        Navigate to show exactly raw channels ``c0_raw`` to ``c1_raw``.

        Sets buffer_h = count so the range exactly fills the buffer (scale_y=1),
        keeping buffer_w unchanged to avoid disturbing the time axis.
        """
        if self.source is None or not self._wrap_y:
            return
        nc = self.source.shape[0]
        c0_raw = int(max(0, min(c0_raw, nc - 1)))
        c1_raw = int(max(c0_raw + 1, min(c1_raw, nc)))
        count = c1_raw - c0_raw

        # Resize buffer_h to exactly cover the range; keep buffer_w fixed.
        if count != self.buffer_h:
            self.set_buffer_shape(buffer_h=count, buffer_w=self.buffer_w, preserve_view=True)

        # buffer_h == count, so scale_y = 1 always fits without triggering fit_all.
        cur_sy = float(getattr(self.source, "scale_y", 1.0))
        if abs(1.0 - cur_sy) > 1e-9:
            self.set_scale_y(1.0)

        # Navigate and zoom camera to show exactly the requested range.
        target_c0 = self._clamp_c0_view(c0_raw)
        rect = self.camera.rect
        camera_rect = (float(rect.left), float(target_c0), float(count))
        target_t0 = self._buffer_t0_for_camera_rect(camera_rect)
        self.scheduler.reset_full(
            target_t0,
            target_c0,
            reason="show_channel_range",
            camera_rect=camera_rect,
            y_pan_window=(target_c0, target_c0 + self.buffer_h),
        )
        self.update()
        self._emit_state_changed()

    def set_channel_start(self, c0, *, hard_reset=False, sync_camera=True):
        """
        Move channel window deterministically.
        """
        target_c0 = self._clamp_c0_view(int(c0))
        if hard_reset:
            camera_rect = None
            y_pan_window = None
            if sync_camera:
                rect = self.camera.rect
                camera_rect = (float(rect.left), float(target_c0), float(rect.height))
                y_pan_window = (target_c0, target_c0 + self.buffer_h)
            target_t0 = self._buffer_t0_for_camera_rect(camera_rect)
            self.scheduler.reset_full(
                target_t0,
                target_c0,
                reason="set_channel_start",
                camera_rect=camera_rect,
                y_pan_window=y_pan_window,
            )
            self.update()
            return

        self.scheduler.set_channel_start(target_c0, hard_reset=False)
        if sync_camera:
            # Update mouse-pan window to current buffer window
            self.camera.set_y_pan_window(self.scheduler.c0, self.scheduler.c0 + self.buffer_h)

            # keep camera bottom consistent with buffer origin (optional but recommended)
            if self._wrap_y:
                self.camera.set_y_bottom(self.scheduler.c0)

        self.update()

    # -------------------------
    # Channel markers
    # -------------------------
    def set_channel_markers(self, channels: list[int], colors=None) -> None:
        """Draw horizontal lines at fixed channel positions.

        Lines are children of world_ui_layer (order=30, above the image) in
        source coordinates (y = source_channel_index, x = [0, nt]).
        """
        for _ch, _vis in self._channel_markers:
            _vis.parent = None
        self._channel_markers.clear()
        if not channels:
            self.update()
            return
        nt = max(1, int(self.nt)) if self.nt else 1
        color_list = list(colors) if colors else []
        for i, ch in enumerate(channels):
            color = color_list[i] if i < len(color_list) else "yellow"
            y = float(ch)
            pos = np.array([[0.0, y], [float(nt), y]], dtype=np.float32)
            line = scene.visuals.Line(
                pos=pos, color=color, method="agg", width=1.5,
                parent=self.world_ui_layer,
            )
            self._channel_markers.append((int(ch), line))
        self.update()

    # -------------------------
    # Timestamp-gap overlays
    # -------------------------
    @staticmethod
    def _format_gap_duration(seconds: float) -> str:
        seconds = float(seconds)
        if seconds >= 3600:
            return f"{seconds / 3600:.2f} h"
        if seconds >= 60:
            return f"{seconds / 60:.1f} min"
        if seconds >= 1:
            return f"{seconds:.2f} s"
        return f"{seconds * 1e3:.1f} ms"

    def _update_gap_overlays(self) -> None:
        """Shade virtual gaps (missing data) of the current source.

        Each gap is a short fill separator in the timeline; the band, edge
        lines and the true missing duration in the label make it explicit so
        it cannot be mistaken for a quiet recording.  Coordinates are raw
        samples / channels in ``world_ui_layer``.
        """
        for vis in self._gap_overlays:
            vis.parent = None
        self._gap_overlays.clear()
        if self.source is None:
            return
        gaps = getattr(self.source.data, "gaps", None) or []
        fs = float(getattr(self.source.data, "fs", 0.0) or 0.0)
        nc = float(max(1, int(self.nc or 1)))
        for g in gaps:
            g0, g1 = float(g.start), float(g.stop)
            if g1 <= g0:
                continue
            rect = scene.visuals.Rectangle(
                center=((g0 + g1) / 2.0, nc / 2.0),
                width=g1 - g0,
                height=nc,
                color=(0.45, 0.45, 0.50, 0.70),
                border_color=None,
                parent=self.world_ui_layer,
            )
            rect.set_gl_state("translucent", depth_test=False, cull_face=False)
            rect.order = 5
            self._gap_overlays.append(rect)
            for x in (g0, g1):
                edge = scene.visuals.Line(
                    pos=np.array([[x, 0.0], [x, nc]], dtype=np.float32),
                    color=(1.0, 0.55, 0.0, 1.0),
                    width=2.0,
                    method="gl",
                    parent=self.world_ui_layer,
                )
                edge.set_gl_state("translucent", depth_test=False)
                edge.order = 6
                self._gap_overlays.append(edge)
            dur = float(getattr(g, "seconds", 0.0)) or ((g1 - g0) / fs if fs > 0 else None)
            label = "NO DATA" + (f" ({self._format_gap_duration(dur)})" if dur else "")
            # Duration label just right of the separator, below the top edge.
            text = scene.visuals.Text(
                "  " + label,
                pos=(g1, nc * 0.92),
                color=(1.0, 0.55, 0.0, 1.0),
                font_size=12,
                bold=True,
                anchor_x="left",
                anchor_y="center",
                parent=self.world_ui_layer,
            )
            text.set_gl_state("translucent", depth_test=False)
            text.order = 7
            self._gap_overlays.append(text)
        self.update()

    def _update_channel_marker_positions(self) -> None:
        """Update x extent after nt changes (new source loaded)."""
        if not self._channel_markers:
            return
        nt = max(1, int(self.nt)) if self.nt else 1
        for _ch, line in self._channel_markers:
            y = float(_ch)
            line.set_data(pos=np.array([[0.0, y], [float(nt), y]], dtype=np.float32))
        self.update()

    def scroll_channels(self, delta, *, hard_reset=False, sync_camera=True):
        self.set_channel_start(
            self.scheduler.c0 + int(delta),
            hard_reset=hard_reset,
            sync_camera=sync_camera,
        )

    def page_channels(
        self,
        direction,
        overlap=0.2,
        *,
        hard_reset=False,
        sync_camera=True,
    ):
        step = int(round(self.buffer_h * (1.0 - float(overlap))))
        step = max(1, step)
        self.scroll_channels(
            step if direction > 0 else -step,
            hard_reset=hard_reset,
            sync_camera=sync_camera,
        )

    def goto_sample(self, sample: int) -> None:
        """
        Jump to a specific sample position (0-indexed, in source coordinates).

        The channel window is preserved; only the time axis is moved.
        """
        if self.scheduler is None or self.source is None:
            return
        perf_on = perf_debug_enabled()
        total_start = perf_now() if perf_on else 0.0
        sx = max(1e-6, float(getattr(self.source, "scale_x", 1.0)))
        view_t0 = self._clamp_t0_view(int(sample / sx))
        r = self.camera.rect
        camera_rect = (float(view_t0), float(r.bottom), float(r.height))
        buffer_t0 = self._buffer_t0_for_camera_rect(camera_rect)
        self.scheduler.reset_full(
            buffer_t0,
            self.scheduler.c0,
            reason="goto_sample",
            camera_rect=camera_rect,
        )
        self._mark_next_draw(
            "goto_sample",
            sample=int(sample),
            view_t0=int(view_t0),
            buffer_t0=int(buffer_t0),
            c0=int(self.scheduler.c0),
            buffer_h=int(self.buffer_h),
            buffer_w=int(self.buffer_w),
        )
        self.update()
        if perf_on:
            log_perf(
                "canvas.goto_sample",
                sample=int(sample),
                view_t0=int(view_t0),
                buffer_t0=int(buffer_t0),
                c0=int(self.scheduler.c0),
                buffer_h=int(self.buffer_h),
                buffer_w=int(self.buffer_w),
                ms=elapsed_ms(total_start),
            )

    def fit_y(self) -> None:
        if self.source is None:
            return
        self.set_scale_y(self.source.fit_scale_y(self.buffer_h))

    def set_downsample_mode(self, mode: DownsampleMode) -> None:
        if self.source is None:
            return
        mode = str(mode)
        if mode not in ("drop", "hybrid", "peak", "linear"):
            raise ValueError(f"unknown downsample mode: {mode}")
        if str(getattr(self.source, "downsample_mode", "drop")) == mode:
            return
        self.mutate_source(lambda src: src.set_scale(downsample_mode=mode))
        self.refresh_buffer()
        self._emit_state_changed()

    # -------------------------
    # scaling APIs (GUI/keyboard)
    # -------------------------
    def set_scale_x(self, scale_x: float):
        """
        Change x compression factor in source.
        We anchor at current VIEW LEFT in "scaled view coords".
        """
        if scale_x is None:
            return

        old_sx = float(self.source.scale_x)
        new_sx = float(scale_x)
        if new_sx < 1.0:
            new_sx = 1.0
        if new_sx == old_sx:
            return

        rect = self.camera.rect

        # anchor at view_left
        view_left_old = float(rect.left)
        view_left_new = view_left_old * old_sx / new_sx

        self.mutate_source(lambda src: src.set_scale(scale_x=new_sx, scale_y=None))
        self._sync_camera_bounds_from_source()

        # Defer the camera move until the reset block is uploaded.  Moving the
        # camera first can expose an empty area while async IO is still reading.
        camera_rect = (float(view_left_new), float(rect.bottom), float(rect.height))
        target_t0 = self._buffer_t0_for_camera_rect(camera_rect)
        self.scheduler.reset_full(
            target_t0,
            self.scheduler.c0,
            reason="scale_x",
            camera_rect=camera_rect,
        )

        # axis
        self.xaxis.update_scale(new_sx)

        self._update_world_layer_transform()
        self.update()
        self._emit_state_changed()

    def set_scale_y(self, scale_y: float):
        """
        Change y compression factor in source.
        Supports fit_all switch when raw_h >= total channels.
        """
        if scale_y is None:
            return

        old_sy = float(self.source.scale_y)
        new_sy = float(scale_y)
        if new_sy < 1.0:
            new_sy = 1.0
        if new_sy == old_sy:
            return

        rect = self.camera.rect
        H = self.source.shape[0]
        bh = int(self.source.buffer_h or rect.height)

        raw_h_local = int(np.ceil(bh * new_sy))

        # fit_all mode
        if raw_h_local >= H:
            new_sy = self.source.fit_scale_y(bh)
            self.mutate_source(lambda src: src.set_scale(
                scale_x=None,
                scale_y=new_sy,
                y_mode="fit_all",
            ))
            target_c0 = 0
            self._sync_camera_bounds_from_source()

            self.yaxis.update_scale(new_sy)

            span_y = float(max(1.0, H / max(1e-6, new_sy)))
            camera_rect = (float(rect.left), 0.0, float(rect.height))
            target_t0 = self._buffer_t0_for_camera_rect(camera_rect)
            self.scheduler.reset_full(
                target_t0,
                target_c0,
                reason="scale_y_fit_all",
                camera_rect=camera_rect,
                y_pan_window=(0.0, span_y),
            )
            self._update_world_layer_transform()
            self._update_channel_marker_positions()
            self.update()
            self._emit_state_changed()
            return

        # local mode
        self.mutate_source(lambda src: src.set_scale(scale_x=None, scale_y=new_sy, y_mode="local"))

        # bottom-anchored scaling
        view_bot_old = float(rect.bottom)
        view_bot_new = view_bot_old * (old_sy / new_sy)

        c_min = 0.0
        c_max = (H - raw_h_local) / new_sy
        c_new = float(np.clip(view_bot_new, c_min, c_max))

        target_c0 = self._clamp_c0_view(int(c_new))
        self._sync_camera_bounds_from_source()

        self.yaxis.update_scale(new_sy)

        target_bottom = float(target_c0) if self._wrap_y else float(rect.bottom)
        camera_rect = (float(rect.left), target_bottom, float(rect.height))
        target_t0 = self._buffer_t0_for_camera_rect(camera_rect)
        self.scheduler.reset_full(
            target_t0,
            target_c0,
            reason="scale_y",
            camera_rect=camera_rect,
            y_pan_window=(float(target_c0), float(target_c0 + self.buffer_h)),
        )
        self._update_world_layer_transform()
        self._update_channel_marker_positions()
        self.update()
        self._emit_state_changed()


    def _update_world_layer_transform(self):
        sx = self.source.scale_x
        sy = self.source.scale_y
        
        sx = sx if sx > 1e-6 else 1.0
        sy = sy if sy > 1e-6 else 1.0

        # physic coords to pixel coords
        self.world_layer.transform.scale = (1.0/sx, 1.0/sy, 1.0)
        # self.world_layer.transform.translate = (tx, ty)


    # -------------------------
    # Axis coordinate mode
    # -------------------------
    def source_meta(self) -> tuple[float, float, str]:
        """Return (fs, dx, dx_unit) from the current source's underlying Tape."""
        if self.source is None:
            return 1.0, 1.0, "m"
        tape = self.source.data
        fs = float(getattr(tape, "fs", None) or 1.0) or 1.0
        dx = float(getattr(tape, "dx", None) or 1.0) or 1.0
        dx_unit = str(getattr(tape, "dx_unit", None) or "m")
        return fs, dx, dx_unit

    def toggle_cursor_pos(self) -> None:
        """Toggle display of mouse cursor coordinates (bound to C key)."""
        self._cursor_pos_visible = not self._cursor_pos_visible
        self.cursor_label.visible = self._cursor_pos_visible
        if not self._cursor_pos_visible:
            self.cursor_label.text = ""
        self.update()

    def _on_mouse_move_cursor(self, event) -> None:
        """Update cursor HUD text with the world coordinates under the mouse."""
        if not self._cursor_pos_visible or self.source is None:
            return
        try:
            # Convert canvas pixel → world (data) coordinates via world_layer transform.
            # world_layer.node_transform(scene) maps FROM world_layer TO canvas.scene,
            # so its inverse maps canvas pixel pos → world (sample/channel) space.
            tr = self.world_layer.node_transform(self.scene)
            point = tr.inverse.map(
                np.array([event.pos[0], event.pos[1], 0], dtype=np.float64)
            )
            wx, wy = float(point[0]), float(point[1])

            fs, dx, dx_unit = self.source_meta()

            if self._x_mode == "utc" and self._start_time is not None:
                t_sec = self._sample_to_unix(wx)
                dt = datetime.datetime.fromtimestamp(t_sec, tz=datetime.timezone.utc)
                ms = dt.microsecond // 1000
                x_str = dt.strftime(f"%H:%M:%S.{ms:03d}")
            elif self._x_mode == "time":
                x_str = f"{self._axis_cursor_name(0, 't')}={wx / fs:.4g}s"
            else:
                x_str = f"{self._axis_cursor_name(0, 't')}={int(wx)}"

            if self._y_mode == "distance":
                y_str = f"{self._axis_cursor_name(1, 'ch')}={wy * dx:.4g}{dx_unit}"
            else:
                y_str = f"{self._axis_cursor_name(1, 'ch')}={int(wy)}"

            self.cursor_label.text = f"{x_str}  {y_str}"
        except Exception:
            pass

    def _on_view_changed(self, event=None) -> None:
        """Camera transform-changed hook: refresh UTC axis label when viewport moves."""
        if self._x_mode == "utc" and self._start_time is not None and self.source is not None:
            self._update_utc_axis_label()

    def _sample_to_unix(self, sample: float) -> float:
        """Absolute time of a raw sample, honouring gap separators in the source."""
        tape = self.source.data if self.source is not None else None
        t = tape_sample_to_unix(tape, sample)
        if t is None:
            fs, _, _ = self.source_meta()
            return float(self._start_time or 0.0) + float(sample) / fs
        return float(t)

    def _update_utc_axis_label(self) -> None:
        """Recompute the UTC tick transform and axis label from the current view-left.

        The axis label shows the absolute UTC wall-clock time at the left edge of
        the visible window.  Tick values show elapsed seconds *relative to that
        reference*, so they always stay in a compact range (e.g. 0–60 s) regardless
        of how far the user has scrolled into the recording.
        """
        if self._start_time is None or self.source is None:
            return
        try:
            sx = float(getattr(self.source, "scale_x", 1.0))
            ref_sample = max(0, int(float(self.camera.rect.left) * sx))
        except Exception:
            _print_exception("Failed to compute UTC axis reference sample; falling back to 0.")
            ref_sample = 0

        self._utc_ref_sample = ref_sample
        fs, _, _ = self.source_meta()

        t_ref = self._sample_to_unix(ref_sample)
        dt_ref = datetime.datetime.fromtimestamp(t_ref, tz=datetime.timezone.utc)
        ms = dt_ref.microsecond // 1000
        x_label = dt_ref.strftime(f"%Y-%m-%d %H:%M:%S.{ms:03d} UTC")

        x_tr = lambda t, _r=float(ref_sample), _fs=fs: f"{(float(t) - _r) / _fs:.4g}"
        self.xaxis.update_transform(x_tr, unit=" s", label=self._axis_label_or_default(0, x_label))

    def _apply_axis_mode(self) -> None:
        """Re-compute axis tick transforms from _x_mode / _y_mode + current source."""
        if self.source is None:
            return
        fs, dx, dx_unit = self.source_meta()

        if self._x_mode == "utc" and self._start_time is not None:
            # Dynamic label: delegate to _update_utc_axis_label so the reference
            # point is immediately set to the current view-left (not fixed at t=0).
            self._update_utc_axis_label()
        elif self._x_mode == "time":
            x_tr = lambda t, _fs=fs: f"{float(t) / _fs:.4g}"
            self.xaxis.update_transform(x_tr, unit=" s", label=self._axis_label_or_default(0, "Time (s)"))
        else:
            x_tr = lambda t: f"{int(float(t))}"
            self.xaxis.update_transform(x_tr, unit="", label=self._axis_label_or_default(0, "Time (samples)"))

        if self._y_mode == "distance":
            y_tr = lambda y, _dx=dx: f"{float(y) * _dx:.4g}"
            y_unit = f" {dx_unit}"
            y_label = f"Distance ({dx_unit})"
        else:
            y_tr = lambda y: f"{int(float(y))}"
            y_unit = ""
            y_label = "Channel"

        self.yaxis.update_transform(y_tr, unit=y_unit, label=self._axis_label_or_default(1, y_label))

    def set_axis_mode(self, *, x_mode: str = "sample", y_mode: str = "channel") -> None:
        """Switch axis coordinate display mode.

        Parameters
        ----------
        x_mode : "sample" | "time" | "utc"
            "sample" — raw sample index; "time" — elapsed seconds;
            "utc" — absolute wall-clock (HH:MM:SS ticks, date in axis label).
        y_mode : "channel" | "distance"
            Show raw channel index or physical distance (uses tape.dx / dx_unit).
        """
        self._x_mode = x_mode
        self._y_mode = y_mode
        self._apply_axis_mode()
        self.update()

    def clear_source(self):
        self.unfreeze()

        self._close_scheduler()
        self.source = None
        self.nc = self.nt = 0
        self.buffer_h = self.buffer_w = 0
        self._did_init = False

        self.camera.interactive = False
        if getattr(self, "xaxis", None) is not None:
            self.xaxis.visible = False
        if getattr(self, "yaxis", None) is not None:
            self.yaxis.visible = False

        self.image.resize((1, 1))
        self.image.visible = False

        if getattr(self, "world_data_layer", None) is not None:
            self._detach_all_children(self.world_data_layer)
        if getattr(self, "world_ui_layer", None) is not None:
            self._detach_all_children(self.world_ui_layer)
        if getattr(self, "world_layer", None) is not None:
            try:
                self.world_layer.transform = STTransform(scale=(1.0, 1.0, 1.0), translate=(0.0, 0.0, 0.0))
            except Exception:
                _print_exception("Failed to reset world-layer transform while clearing the canvas.")

        self.freeze()
        self.update()


    def _detach_all_children(self, node: scene.Node):
        """Detach all children nodes from a scene.Node safely."""
        # copy list to avoid modifying while iterating
        for ch in list(getattr(node, "children", [])):
            try:
                if hasattr(ch, "close") and callable(ch.close):
                    ch.close()
            except Exception:
                _print_exception("Failed to close a canvas child node during detach.")
            try:
                ch.parent = None
            except Exception:
                _print_exception("Failed to detach a canvas child node via parent=None; trying _parent fallback.")
                try:
                    ch._parent = None
                except Exception:
                    _print_exception("Failed to detach a canvas child node via _parent fallback.")

    # ── memory monitor key handler ────────────────────────────────────────────

    def _on_mem_key(self, ev) -> None:
        """M — print current RSS;  Shift+M — print full lifecycle snapshot history."""
        k = _event_key_name(ev)
        if k is None or str(k).upper() != "M":
            return

        shift, ctrl, meta, alt = _event_modifier_flags(ev)
        if ctrl or meta or alt:
            return

        if shift:
            # Take a snapshot now then print the full history
            self._mem_monitor.snapshot("M key (history requested)")
            self._mem_monitor.print_report()
        else:
            # Quick one-liner to stderr
            label = "scrolling" if getattr(self, "_rt_running", False) else "browsing"
            if self.source is not None:
                label += f"  nc={self.nc} nt={self.nt}"
            self._mem_monitor.print_current(label)

        ev.handled = True
