# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""Declarative configuration surface for the Qt application."""

from dataclasses import dataclass, field
from typing import Any, Sequence

from filark.io.loaders import Loader, default_loaders


def _default_realtime_fs_presets() -> tuple[tuple[str, float], ...]:
    return (
        ("1", 200.0),
        ("2", 500.0),
        ("3", 1000.0),
        ("4", 1500.0),
        ("5", 2000.0),
        ("6", 3000.0),
        ("7", 5000.0),
        ("8", 10000.0),
    )


@dataclass
class GuiConfig:
    """
    Declarative configuration for Filark GUI.

    This object describes *what the GUI is capable of*,
    not how it is implemented.
    """

    # initial data source (expert / programmatic mode)
    source: Any | None = None

    # GUI-discoverable capabilities
    loaders: Sequence[Loader] = field(default_factory=tuple)
    builtins: Sequence[Loader] = field(default_factory=default_loaders)
    # Streaming display pipeline.
    # stream_preprocess runs before view downsampling; stream_postprocess runs after.
    stream_preprocess: Any | None = None
    stream_postprocess: Any | None = None

    # Called after a source is loaded, before it is displayed.
    # Signature: (tape) -> tape | None
    # Use this to inspect the tape and attach preprocessing (tape.preprocess = ...).
    # Returning a new tape replaces the source; returning None uses the original.
    on_source_loaded: Any | None = None

    # Processor plugins for the ROI denoising view.
    # None → use default_processors() (the 4 built-in methods).
    processors: Sequence[Any] | None = None

    # Monitor plugins for the Algorithms > Monitor section.
    # None → no monitors available (placeholder UI shown).
    monitors: Sequence[Any] | None = None

    # Analysis plugins for the Algorithms > Analysis section.
    # None → use built-in analyses.
    analyses: Sequence[Any] | None = None

    # appearance
    theme: str = "dark"  # dark or light
    display_aspect: float | None = None
    display_aspect_limits: tuple[float, float] | None = None
    auto_display_aspect: bool = True
    x_role: str | None = None
    y_role: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    x_unit: str | None = None
    y_unit: str | None = None
    invert_y: bool | str = "auto"
    async_io: bool = True

    # GUI layout defaults
    window_size: tuple[int, int] = (1200, 800)
    navbar_width: int = 50
    drawer_width: int = 360
    drawer_animation_ms: int = 250

    # Top-level tool windows.
    analysis_widget_size: tuple[int, int] = (1400, 920)
    analysis_figure_window_size: tuple[int, int] = (900, 600)
    roi_window_size: tuple[int, int] = (1100, 750)

    # Window shortcuts.
    shortcut_collapse_ui: str = "Esc"
    shortcut_goto: str = "Ctrl+G"
    shortcut_overview: str = "Ctrl+Shift+O"
    shortcut_compare_prev: str = "Ctrl+Left"
    shortcut_compare_next: str = "Ctrl+Right"
    shortcut_tape_transform_next: str = "Alt+T"
    shortcut_tape_transform_prev: str = "Alt+Shift+T"
    shortcut_tape_transform_raw: str = "Alt+0"

    # Annotation defaults and canvas interaction style.
    annotation_default_mode: str = "bbox"
    annotation_default_label: str = "event"
    annotation_default_enabled: bool = True
    annotation_hold_ctrl: bool = True
    annotation_confirm_each: bool = False
    annotation_preview_color: str = "#FFD54F"
    annotation_preview_width: float = 3.0
    annotation_preview_cursor_width: float = 2.0
    annotation_bbox_width: float = 2.0
    annotation_polyline_width: float = 3.0
    annotation_marker_size: int = 9
    annotation_handle_size: int = 8
    annotation_hit_slop_t_fraction: float = 0.015
    annotation_hit_slop_t_min: float = 4.0
    annotation_hit_slop_c_fraction: float = 0.03
    annotation_hit_slop_c_min: float = 1.5
    annotation_mask_brush_radius: int = 16  # Screen pixels.
    annotation_mask_brush_mode: str = "paint"
    annotation_render_bbox_pool: int = 64
    annotation_render_polyline_pool: int = 64
    annotation_render_mask_pool: int = 12
    annotation_render_mask_capacity_t: int = 512
    annotation_render_mask_capacity_c: int = 512
    annotation_render_margin_fraction: float = 0.35

    # Analysis overlay style.
    analysis_channel_marker_color: str = "#00BCD4"

    # Streaming canvas defaults
    default_buffer_h: int = 2048

    # Realtime autoscroll defaults.
    realtime_fs: float = 500.0
    realtime_speed: float = 1.0
    realtime_timer_hz: int = 120
    realtime_max_dt: float = 0.05
    realtime_enable_stats: bool = True
    realtime_auto_stop_s: float | None = None
    realtime_bench_max_speed: float = 100.0
    realtime_bench_duration_s: float = 2.0
    realtime_bench_min_good_rate: float = 0.90
    realtime_fs_presets: Sequence[tuple[str, float]] = field(default_factory=_default_realtime_fs_presets)

    # Display-driven monitor defaults.
    realtime_display_queue_max: int = 4
    realtime_detection_bbox_pool: int = 32
    realtime_detection_polyline_pool: int = 32
    realtime_detection_mask_pool: int = 12
    realtime_detection_mask_capacity_t: int = 512
    realtime_detection_mask_capacity_c: int = 512

    def loader_list(self) -> Sequence[Loader]:
        return tuple(self.loaders) + tuple(self.builtins)

    def axis_metadata(self):
        from filark.gui.axis_metadata import build_axis_metadata

        return build_axis_metadata(
            x_role=self.x_role,
            y_role=self.y_role,
            x_label=self.x_label,
            y_label=self.y_label,
            x_unit=self.x_unit,
            y_unit=self.y_unit,
            invert_y=self.invert_y,
        )

    def build_stream_preprocess(self, *, fs: float, dx: float):
        from filark.io.stream_processing import build_stream_preprocess
        return build_stream_preprocess(self.stream_preprocess, fs=float(fs), dx=float(dx))

    def build_stream_postprocess(self, *, fs: float, dx: float):
        from filark.io.stream_processing import build_stream_postprocess
        return build_stream_postprocess(self.stream_postprocess, fs=float(fs), dx=float(dx))

    def processor_list(self) -> list:
        """Return the effective processor list (built-ins if not overridden)."""
        if self.processors is None:
            from filark.processing.processors import default_processors
            return default_processors()
        return list(self.processors)

    def monitor_list(self) -> list:
        """Return the effective monitor list (empty if not configured)."""
        if self.monitors is None:
            return []
        return list(self.monitors)

    def analysis_list(self) -> list:
        """Return the effective analysis list (built-ins plus custom analyses)."""
        from filark.gui.builtin_analyses import default_analyses

        builtins = list(default_analyses())
        custom = list(self.analyses) if self.analyses is not None else []
        out: list = []
        seen: set[object] = set()
        for plugin in builtins + custom:
            key = getattr(plugin, "key", id(plugin))
            if key in seen:
                continue
            seen.add(key)
            out.append(plugin)
        return out

    def realtime_autoscroll_options(self) -> dict[str, Any]:
        """Return canvas options for realtime autoscroll."""
        return {
            "fs": float(self.realtime_fs),
            "speed": float(self.realtime_speed),
            "timer_hz": int(self.realtime_timer_hz),
            "max_dt": float(self.realtime_max_dt),
            "enable_stats": bool(self.realtime_enable_stats),
            "auto_stop_s": self.realtime_auto_stop_s,
            "bench_max_speed": float(self.realtime_bench_max_speed),
            "bench_duration_s": float(self.realtime_bench_duration_s),
            "bench_min_good_rate": float(self.realtime_bench_min_good_rate),
            "fs_presets": tuple(self.realtime_fs_presets),
        }

    def annotation_options(self) -> dict[str, Any]:
        """Return annotation interaction defaults for DisplayModel and panel UI."""
        return {
            "mode": str(self.annotation_default_mode),
            "label": str(self.annotation_default_label),
            "enabled": bool(self.annotation_default_enabled),
            "hold_ctrl": bool(self.annotation_hold_ctrl),
            "confirm_each": bool(self.annotation_confirm_each),
            "preview_color": str(self.annotation_preview_color),
            "preview_width": float(self.annotation_preview_width),
            "preview_cursor_width": float(self.annotation_preview_cursor_width),
            "bbox_width": float(self.annotation_bbox_width),
            "polyline_width": float(self.annotation_polyline_width),
            "marker_size": int(self.annotation_marker_size),
            "handle_size": int(self.annotation_handle_size),
            "hit_slop_t_fraction": float(self.annotation_hit_slop_t_fraction),
            "hit_slop_t_min": float(self.annotation_hit_slop_t_min),
            "hit_slop_c_fraction": float(self.annotation_hit_slop_c_fraction),
            "hit_slop_c_min": float(self.annotation_hit_slop_c_min),
            "mask_brush_radius": int(self.annotation_mask_brush_radius),
            "mask_brush_mode": str(self.annotation_mask_brush_mode),
            "render_bbox_pool": int(self.annotation_render_bbox_pool),
            "render_polyline_pool": int(self.annotation_render_polyline_pool),
            "render_mask_pool": int(self.annotation_render_mask_pool),
            "render_mask_capacity_t": int(self.annotation_render_mask_capacity_t),
            "render_mask_capacity_c": int(self.annotation_render_mask_capacity_c),
            "render_margin_fraction": float(self.annotation_render_margin_fraction),
        }

    def merge(
        self,
        *,
        source: Any | None = None,
        theme: str | None = None,
        display_aspect: float | None = None,
        display_aspect_limits: tuple[float, float] | None = None,
        auto_display_aspect: bool | None = None,
        x_role: str | None = None,
        y_role: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        x_unit: str | None = None,
        y_unit: str | None = None,
        invert_y: bool | str | None = None,
        async_io: bool | None = None,
        default_buffer_h: int | None = None,
    ) -> "GuiConfig":
        """
        Merge shortcut parameters into this config.
        """
        if source is not None:
            self.source = source
        if theme is not None:
            self.theme = theme
        if display_aspect is not None:
            self.display_aspect = float(display_aspect)
        if display_aspect_limits is not None:
            self.display_aspect_limits = tuple(display_aspect_limits)
        if auto_display_aspect is not None:
            self.auto_display_aspect = bool(auto_display_aspect)
        if x_role is not None:
            self.x_role = str(x_role)
        if y_role is not None:
            self.y_role = str(y_role)
        if x_label is not None:
            self.x_label = str(x_label)
        if y_label is not None:
            self.y_label = str(y_label)
        if x_unit is not None:
            self.x_unit = str(x_unit)
        if y_unit is not None:
            self.y_unit = str(y_unit)
        if invert_y is not None:
            self.invert_y = invert_y if isinstance(invert_y, str) else bool(invert_y)
        if async_io is not None:
            self.async_io = bool(async_io)
        if default_buffer_h is not None:
            self.default_buffer_h = int(default_buffer_h)
        return self
