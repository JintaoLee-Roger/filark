"""GUI-facing interaction model that binds Qt panels to the streaming canvas."""

from __future__ import annotations

from PySide6.QtCore import QObject, QTimer, Signal

from filark.viz import StreamingCanvas
from .annotation_interactor import AnnotationInteractionMixin
from .canvas_facade import DisplayCanvasFacadeMixin
from .overlays import OverlayRegistryMixin


class DisplayModel(
    QObject,
    DisplayCanvasFacadeMixin,
    OverlayRegistryMixin,
    AnnotationInteractionMixin,
):
    """Canvas interaction model for drawing, editing, and viewing annotations."""

    bbox_finished = Signal(dict)
    polyline_finished = Signal(dict)
    mask_finished = Signal(dict)
    annotation_selected = Signal(object)
    annotation_geometry_edited = Signal(int, dict)
    roi_committed = Signal(dict)
    scale_state_changed = Signal(dict)
    source_transform_changed = Signal(dict)
    display_filter_status_changed = Signal(dict)

    def __init__(
        self,
        canvas: StreamingCanvas,
        *,
        default_buffer_h: int = 2048,
        annotation_options: dict | None = None,
    ):
        super().__init__()
        self._canvas = canvas
        self._default_buffer_h = int(max(1, default_buffer_h or 2048))
        self._annotation_options = dict(annotation_options or {})
        self._init_canvas_facade_state()
        self._init_overlay_state()
        self._init_event_router()
        self._init_annotation_interactor()

        self._bind_canvas_events()
        self._canvas.add_state_listener(self._on_canvas_state_changed)
        self._annotation_view_refresh_pending = False
        self._annotation_view_refresh_timer = QTimer(self)
        self._annotation_view_refresh_timer.setSingleShot(True)
        self._annotation_view_refresh_timer.timeout.connect(self._flush_annotation_view_refresh)
        self._canvas.camera.add_view_listener(self._on_camera_view_changed)
        register_metrics = getattr(self._canvas, "register_realtime_metric_provider", None)
        if callable(register_metrics):
            register_metrics("annotation", self.realtime_annotation_stats)
        self._canvas.camera.interactive = True
        self._refresh_active_handler()

    def clear(self) -> None:
        self._source = None
        self.clear_overlays()
        self._canvas.clear_source()
        self._ensure_preview_visuals()
        self._emit_scale_state_changed()
        self._emit_source_transform_changed()

    def _on_camera_view_changed(self, _camera=None) -> None:
        if self._annotation_view_refresh_pending:
            return
        self._annotation_view_refresh_pending = True
        self._annotation_view_refresh_timer.start(0)

    def _flush_annotation_view_refresh(self) -> None:
        self._annotation_view_refresh_pending = False
        self._refresh_annotation_virtual_overlays()
