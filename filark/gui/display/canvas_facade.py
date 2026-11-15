"""Canvas facade, source processing, and routing helpers for DisplayModel."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Dict, Optional

import numpy as np

from filark._exceptions import print_exception
from filark.dsp.temporal import build_iir_sos
from filark.io.stream_processing import (
    RealtimeDisplaySOSFilter,
    SourceProcessingController,
    ensure_streaming_source,
)
from filark.io.protocols import TransformableTape
from filark.utils.perf import elapsed_ms, log_perf, perf_debug_enabled, perf_now
from filark.viz._aspect import DEFAULT_MAX_DISPLAY_ASPECT, DEFAULT_MIN_DISPLAY_ASPECT


def _print_exception(context: str) -> None:
    print_exception(context, prefix="DisplayModel")


class DisplayCanvasFacadeMixin:
    """Own source/display filter controls and canvas navigation wrappers."""

    def _init_canvas_facade_state(self) -> None:
        self._source = None
        self._clim = None
        self._cmap = None
        self._interp = None
        self._display_filter_params: Optional[Dict[str, Any]] = None
        self._display_filter_status: Dict[str, Any] = {"state": "off"}
        self._processing = SourceProcessingController()

    def _init_event_router(self) -> None:
        self._handlers = {}
        self._handler_order = []
        self._active_handler = None

    def register_handler(self, name: str, handler, *, priority: int = 100) -> None:
        self._handlers[name] = (priority, handler)
        self._handler_order = [name for name, _ in sorted(self._handlers.items(), key=lambda item: item[1][0])]

    def set_active_handler(self, name: Optional[str]) -> None:
        self._active_handler = name

    def _dispatch(self, method_name: str, ev) -> None:
        if getattr(ev, "handled", False):
            return

        if self._active_handler is not None:
            _, handler = self._handlers.get(self._active_handler, (None, None))
            if handler is not None:
                func = getattr(handler, method_name, None)
                if func is not None:
                    func(ev)
                    if getattr(ev, "handled", False):
                        return

        for name in self._handler_order:
            if name == self._active_handler:
                continue
            _, handler = self._handlers[name]
            func = getattr(handler, method_name, None)
            if func is None:
                continue
            func(ev)
            if getattr(ev, "handled", False):
                return

    def canvas_pos_to_image(self, pos) -> tuple[float, float]:
        transform = self._canvas.scene.node_transform(self._canvas.image)
        world = transform.map(np.array(pos, dtype=np.float32)[:2])
        return float(world[0]), float(world[1])

    def image_pixel_to_world(self, px: float, py: float) -> tuple[float, float]:
        transform = self._canvas.image.node_transform(self._canvas.view.scene)
        world = transform.map((px, py, 0))[:2]
        return float(world[0]), float(world[1])

    def canvas_pos_to_layer(self, pos) -> tuple[float, float]:
        transform = self._canvas.world_layer.node_transform(self._canvas.scene)
        point = transform.inverse.map((pos[0], pos[1], 0))
        return float(point[0]), float(point[1])

    def canvas_pos_to_world(self, pos) -> tuple[float, float]:
        return self.canvas_pos_to_layer(pos)

    @property
    def _world_overlay_parent(self):
        return getattr(self._canvas, "world_data_layer", self._canvas.world_layer)

    @property
    def _world_ui_parent(self):
        return getattr(self._canvas, "world_ui_layer", self._canvas.world_layer)

    def _update_rect(self, rect, p0, p1) -> None:
        x0, y0 = p0
        x1, y1 = p1
        rect.center = (0.5 * (x0 + x1), 0.5 * (y0 + y1))
        rect.width = max(abs(x1 - x0), 1e-6)
        rect.height = max(abs(y1 - y0), 1e-6)
        self._canvas.update()

    def set_display_params(self, *, clim=None, cmap=None, interpolation=None) -> None:
        if clim is not None:
            self.set_clim(*clim)
        if cmap is not None:
            self.set_cmap(cmap)
        if interpolation is not None:
            self.set_interpolation(interpolation)

    def set_clim(self, vmin, vmax) -> None:
        self._clim = (vmin, vmax)
        self._canvas.image.clim = [vmin, vmax]

    def set_cmap(self, cmap) -> None:
        self._cmap = cmap
        self._canvas.image.cmap = cmap

    def set_interpolation(self, interp) -> None:
        self._interp = interp
        self._canvas.image.interpolation = interp

    def set_source(self, source) -> None:
        """Bind a new source and keep transient visuals alive across resets."""
        self._source = source
        self._canvas.set_source(ensure_streaming_source(source))
        self._ensure_preview_visuals()
        if self._canvas.source is not None:
            self._canvas.mutate_source(lambda _src: self._processing.bind_source(self._canvas.source))
        self._reapply_display_filter()
        self._emit_source_transform_changed()

    def set_stream_processing(self, preprocess=None, postprocess=None) -> None:
        if self._canvas.source is not None:
            self._canvas.mutate_source(
                lambda _src: self._processing.set_stream_processing(preprocess, postprocess)
            )
            self._canvas.refresh_buffer()
        else:
            self._processing.set_stream_processing(preprocess, postprocess)

    def set_display_filter(
        self,
        kind: str,
        fmin: Optional[float],
        fmax: Optional[float],
        order: int,
    ) -> None:
        self._display_filter_params = {"kind": kind, "fmin": fmin, "fmax": fmax, "order": order}
        self._reapply_display_filter()

    def clear_display_filter(self) -> None:
        self._display_filter_params = None
        if self._canvas.source is not None:
            self._canvas.mutate_source(lambda _src: self._processing.set_display_postprocess(None))
            self._canvas.refresh_buffer()
        self._set_display_filter_status({"state": "off"})

    def display_filter_params(self) -> Optional[Dict[str, Any]]:
        """Parameters of the user's display filter (None when disabled)."""
        return None if self._display_filter_params is None else dict(self._display_filter_params)

    def display_filter_status(self) -> Dict[str, Any]:
        """``{"state": "off" | "active" | "suspended", "fs": ..., "message": ...}``."""
        return dict(self._display_filter_status)

    def _set_display_filter_status(self, status: Dict[str, Any]) -> None:
        self._display_filter_status = dict(status)
        sig = getattr(self, "display_filter_status_changed", None)
        if sig is not None:
            try:
                sig.emit(dict(status))
            except Exception:
                _print_exception("display_filter_status_changed emit failed")
        else:
            self._processing.set_display_postprocess(None)

    def current_display_filter_fs(self, *, scale_x: Optional[float] = None) -> float:
        fs, _, _ = self._canvas.source_meta()
        if fs <= 0:
            return 0.0
        sx = 1.0
        if scale_x is not None:
            sx = max(1.0, float(scale_x))
        elif self._canvas.source is not None:
            sx = max(1.0, float(getattr(self._canvas.source, "scale_x", 1.0)))
        return float(fs) / sx

    def _reapply_display_filter(self, *, refresh: bool = True, scale_x: Optional[float] = None) -> None:
        if self._display_filter_params is None:
            return
        fs = self.current_display_filter_fs(scale_x=scale_x)
        if fs <= 0:
            return

        params = self._display_filter_params
        try:
            sos = build_iir_sos(
                fs=fs,
                kind=params["kind"],  # type: ignore[arg-type]
                fmin=params["fmin"],
                fmax=params["fmax"],
                order=int(params["order"]),
            )
        except Exception as exc:
            # The display filter runs on the time-decimated buffer (fs / scale_x).
            # When the requested band no longer fits below that Nyquist, never
            # keep the previous filter (designed for another rate): suspend it,
            # keep the parameters, and re-enable automatically on zoom-in.
            self._canvas.mutate_source(lambda _src: self._processing.set_display_postprocess(None))
            if refresh and self._canvas.source is not None:
                self._canvas.refresh_buffer()
            self._set_display_filter_status({
                "state": "suspended",
                "fs": float(fs),
                "message": (
                    f"Suspended: display rate at this zoom is {fs:.4g} Hz "
                    f"(Nyquist {0.5 * fs:.4g} Hz), below the filter band. Zoom in to re-enable."
                ),
                "error": f"{type(exc).__name__}: {exc}",
            })
            self._emit_scale_state_changed()
            return
        self._canvas.mutate_source(
            lambda _src: self._processing.set_display_postprocess(
                RealtimeDisplaySOSFilter(np.array(sos, dtype=np.float64))
            )
        )
        if refresh and self._canvas.source is not None:
            self._canvas.refresh_buffer()
        self._set_display_filter_status({"state": "active", "fs": float(fs), "message": ""})
        self._emit_scale_state_changed()

    def current_source(self):
        return self._source

    @staticmethod
    def _normalize_transform_options(raw_options) -> list[dict[str, str]]:
        options: list[dict[str, str]] = []
        for item in raw_options or ():
            if isinstance(item, Mapping):
                key = item.get("key", item.get("value"))
                if key is None:
                    continue
                label = item.get("label", key)
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                key, label = item[0], item[1]
            else:
                key = label = item
            options.append({"key": str(key), "label": str(label)})
        return options

    @staticmethod
    def _transform_options_for(target) -> list[dict[str, str]]:
        raw_options = getattr(target, "transform_options", None)
        if callable(raw_options):
            raw_options = raw_options()
        return DisplayCanvasFacadeMixin._normalize_transform_options(raw_options)

    @staticmethod
    def _active_transform_for(target, options: list[dict[str, str]] | None = None) -> str | None:
        active = getattr(target, "transform", None)
        if callable(active):
            active = active()
        if active is not None:
            return str(active)
        opts = options if options is not None else DisplayCanvasFacadeMixin._transform_options_for(target)
        return opts[0]["key"] if opts else None

    @staticmethod
    def _transform_revision_for(target) -> int:
        rev = getattr(target, "transform_revision", 0)
        if callable(rev):
            rev = rev()
        try:
            return int(rev)
        except Exception:
            return 0

    @staticmethod
    def _supports_transform(target) -> bool:
        if target is None:
            return False
        return isinstance(target, TransformableTape)

    def _source_transform_target(self):
        source = self._source
        candidates = [
            source,
            getattr(source, "data", None),
            getattr(getattr(self._canvas, "source", None), "data", None),
        ]
        for candidate in candidates:
            if self._supports_transform(candidate):
                return candidate
        return None

    def source_transform_state(self) -> dict[str, object]:
        target = self._source_transform_target()
        if target is None:
            return {
                "enabled": False,
                "options": [],
                "active": None,
                "label": "",
                "revision": 0,
            }
        options = self._transform_options_for(target)
        active = self._active_transform_for(target, options)
        label = ""
        for option in options:
            if option["key"] == active:
                label = option["label"]
                break
        if not label and active is not None:
            label = str(active)
        return {
            "enabled": bool(options),
            "options": options,
            "active": active,
            "label": label,
            "revision": self._transform_revision_for(target),
        }

    def _emit_source_transform_changed(self) -> None:
        signal = getattr(self, "source_transform_changed", None)
        if signal is not None:
            signal.emit(self.source_transform_state())

    def _mutate_source_transform(self, fn) -> dict[str, object]:
        target = self._source_transform_target()
        if target is None:
            state = self.source_transform_state()
            self._emit_source_transform_changed()
            return state

        before = (
            self._active_transform_for(target),
            self._transform_revision_for(target),
        )

        try:
            if self._canvas.source is not None:
                self._canvas.mutate_source(lambda _src: fn(target))
            else:
                fn(target)
        except Exception as exc:
            _print_exception(f"Failed to switch source transform: {type(exc).__name__}: {exc}")

        after = (
            self._active_transform_for(target),
            self._transform_revision_for(target),
        )
        if after != before and self._canvas.source is not None:
            self._canvas.refresh_buffer()
        self._emit_source_transform_changed()
        return self.source_transform_state()

    def set_source_transform(self, key: str) -> dict[str, object]:
        def apply(target) -> None:
            setter = getattr(target, "set_transform", None)
            if callable(setter):
                setter(str(key))

        return self._mutate_source_transform(apply)

    def cycle_source_transform(self, step: int = 1) -> dict[str, object]:
        def apply(target) -> None:
            cycler = getattr(target, "cycle_transform", None)
            if callable(cycler):
                cycler(int(step))
                return
            setter = getattr(target, "set_transform", None)
            if not callable(setter):
                return
            options = self._transform_options_for(target)
            if not options:
                return
            active = self._active_transform_for(target, options)
            keys = [option["key"] for option in options]
            idx = keys.index(active) if active in keys else 0
            setter(keys[(idx + int(step)) % len(keys)])

        return self._mutate_source_transform(apply)

    def reset_source_transform(self) -> dict[str, object]:
        target = self._source_transform_target()
        if target is None:
            return self.source_transform_state()
        options = self._transform_options_for(target)
        if not options:
            return self.source_transform_state()
        key = next((option["key"] for option in options if option["key"] == "raw"), options[0]["key"])
        return self.set_source_transform(key)

    def add_mouse_press_interceptor(self, fn: Callable[[Any], bool]) -> None:
        if fn not in self._mouse_press_interceptors:
            self._mouse_press_interceptors.append(fn)

    def remove_mouse_press_interceptor(self, fn: Callable[[Any], bool]) -> bool:
        try:
            self._mouse_press_interceptors.remove(fn)
            return True
        except ValueError:
            return False

    def _run_mouse_press_interceptors(self, ev) -> bool:
        for fn in tuple(self._mouse_press_interceptors):
            try:
                if bool(fn(ev)):
                    ev.handled = True
                    return True
            except Exception:
                _print_exception("mouse press interceptor failed")
        return bool(getattr(ev, "handled", False))

    def goto_sample(self, sample: int, *, center: bool = False) -> None:
        perf_on = perf_debug_enabled()
        total_start = perf_now() if perf_on else 0.0
        try:
            target = int(sample)
            if center and self._canvas.source is not None and self._canvas.camera is not None:
                sx = max(1e-6, float(getattr(self._canvas.source, "scale_x", 1.0)))
                target = max(0, int(sample - (self._canvas.camera.rect.width / 2) * sx))
            self._canvas.goto_sample(target)
            if perf_on:
                log_perf(
                    "display_model.goto_sample",
                    sample=int(sample),
                    target=int(target),
                    center=int(center),
                    overlay_count=int(self.overlay_count()),
                    ms=elapsed_ms(total_start),
                )
        except Exception:
            _print_exception("goto_sample failed")

    def set_axis_mode(self, x_mode: str, y_mode: str) -> None:
        try:
            self._canvas.set_axis_mode(x_mode=x_mode, y_mode=y_mode)
        except Exception:
            _print_exception("set_axis_mode failed")

    def set_channel_markers(self, channels: list[int], colors: list[str] | None = None) -> None:
        try:
            self._canvas.set_channel_markers(channels, colors)
        except Exception:
            _print_exception("set_channel_markers failed")

    def current_scale_state(self) -> dict:
        try:
            return dict(self._canvas.current_scale_state())
        except Exception:
            _print_exception("current_scale_state failed")
            return {
                "scale_x": 1.0,
                "scale_y": 1.0,
                "y_mode": "local",
                "downsample_mode": "drop",
                "buffer_h": self._default_buffer_h,
                "buffer_w": 0,
                "auto_buffer_resize": True,
                "display_aspect": 1.0,
                "auto_display_aspect": True,
                "display_aspect_min": DEFAULT_MIN_DISPLAY_ASPECT,
                "display_aspect_max": DEFAULT_MAX_DISPLAY_ASPECT,
            }

    def current_view_bounds(self) -> dict | None:
        try:
            return self._canvas.current_view_bounds()
        except Exception:
            _print_exception("current_view_bounds failed")
            return None

    def add_chunk_listener(self, fn) -> None:
        self._canvas.add_chunk_listener(fn)

    def remove_chunk_listener(self, fn) -> bool:
        return self._canvas.remove_chunk_listener(fn)

    def current_display_chunk_event(self):
        return self._canvas.current_display_chunk_event()

    def set_scale_x(self, scale_x: float) -> None:
        new_scale_x = max(1.0, float(scale_x))
        if self._display_filter_params is not None:
            self._reapply_display_filter(refresh=False, scale_x=new_scale_x)
        self._canvas.set_scale_x(new_scale_x)

    def set_scale_y(self, scale_y: float) -> None:
        self._canvas.set_scale_y(scale_y)

    def set_channel_start(self, c0: int) -> None:
        self._canvas.set_channel_start(int(c0))

    def show_channel_range(self, c0: int, c1: int) -> None:
        self._canvas.show_channel_range(int(c0), int(c1))

    def fit_y(self) -> None:
        self._canvas.fit_y()

    def set_downsample_mode(self, mode: str) -> None:
        self._canvas.set_downsample_mode(mode)

    def set_buffer_h(self, buffer_h: int) -> None:
        self._canvas.set_buffer_h(buffer_h)

    def set_buffer_w(self, buffer_w: int) -> None:
        self._canvas.set_buffer_w(buffer_w)

    def set_auto_buffer_resize(self, enabled: bool) -> None:
        self._canvas.set_auto_buffer_resize(enabled)

    def set_display_aspect(self, display_aspect: float) -> None:
        self._canvas.set_display_aspect(display_aspect)

    def set_display_aspect_limits(self, display_aspect_limits: tuple[float, float] | None) -> None:
        self._canvas.set_display_aspect_limits(display_aspect_limits)

    def set_auto_display_aspect(self, enabled: bool) -> None:
        self._canvas.set_auto_display_aspect(enabled)

    def switch_source(self, source) -> None:
        self._source = source
        self._canvas.set_source(ensure_streaming_source(source), reset_view=False)
        if self._canvas.source is not None:
            self._canvas.mutate_source(lambda _src: self._processing.bind_source(self._canvas.source))
        self._reapply_display_filter()
        self._ensure_preview_visuals()
        self._emit_scale_state_changed()
        self._emit_source_transform_changed()

    def _emit_scale_state_changed(self) -> None:
        try:
            self.scale_state_changed.emit(self.current_scale_state())
        except Exception:
            _print_exception("scale_state_changed signal emission failed")

    def _on_canvas_state_changed(self, _state: dict | None = None) -> None:
        refresh_virtual = getattr(self, "_refresh_annotation_virtual_overlays", None)
        if callable(refresh_virtual):
            refresh_virtual()
        self._emit_scale_state_changed()
