"""Window layout and drawer assembly for the main GUI window."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QHBoxLayout, QScrollArea, QVBoxLayout, QWidget

from filark.viz.canvas import StreamingCanvas

from ..display import DisplayModel
from ..modules import (
    AnnotationPanel,
    DetectEventsWindow,
    LoadPanel,
    ROIPanel,
    SourcePropertiesDialog,
    VisualsPanel,
)
from ..widgets.navbar import NavBar
from ..widgets.sliding_drawer import SlidingDrawer
from ..widgets.source_switcher import SourceBar


# Drawer tab indices (left navbar).  The drawer only holds things that act on
# the picture currently on the canvas; everything else lives in the menu bar.
TAB_VISUALS = 0
TAB_ANNOTATION = 1
TAB_ROI = 2


class LayoutControllerMixin:
    """Own main widget layout, drawer panels, and drawer geometry."""

    def _build_main_layout(self) -> None:
        cfg = self.cfg

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.navbar_width = int(cfg.navbar_width)
        self.tabs = [
            ("Visuals", "🎨"),
            ("Annotation", "🏷️"),
            ("ROI", "🔲"),
        ]
        self.navbar = NavBar(self.tabs)
        self.navbar.setFixedWidth(self.navbar_width)

        self.axis_metadata = cfg.axis_metadata()
        bgcolor = "white" if cfg.theme == "light" else "black"
        self.canvas = StreamingCanvas(
            bgcolor=bgcolor,
            wrap_y=True,
            keys=None,
            display_aspect=cfg.display_aspect,
            display_aspect_limits=cfg.display_aspect_limits,
            auto_display_aspect=cfg.auto_display_aspect,
            axis_labels=(self.axis_metadata.x.label, self.axis_metadata.y.label),
            invert_y=self.axis_metadata.invert_y,
            async_io=cfg.async_io,
            default_buffer_h=cfg.default_buffer_h,
            realtime_autoscroll=cfg.realtime_autoscroll_options(),
        )
        self.disp_model = DisplayModel(
            self.canvas,
            default_buffer_h=cfg.default_buffer_h,
            annotation_options=cfg.annotation_options(),
        )
        register_rt_state = getattr(self.canvas, "register_realtime_state_listener", None)
        if callable(register_rt_state):
            register_rt_state(self._on_canvas_realtime_autoscroll_state_changed)
        self.disp_model.add_mouse_press_interceptor(self._on_analysis_pick_mouse_press)
        self.disp_model.source_transform_changed.connect(self._on_source_transform_changed)

        self._canvas_container = QWidget()
        canvas_vbox = QVBoxLayout(self._canvas_container)
        canvas_vbox.setContentsMargins(0, 0, 0, 0)
        canvas_vbox.setSpacing(0)

        self.source_bar = SourceBar()
        self.source_bar.source_activated.connect(self._on_source_activated)
        canvas_vbox.addWidget(self.source_bar)
        canvas_vbox.addWidget(self.canvas.native, 1)

        self.main_layout.addWidget(self.navbar)
        self.main_layout.addWidget(self._canvas_container, 1)
        self.canvas.native.installEventFilter(self)

        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
            app.aboutToQuit.connect(self._shutdown_background_workers)

        self.drawer = SlidingDrawer(
            self.central_widget,
            width=int(cfg.drawer_width),
            animation_ms=int(cfg.drawer_animation_ms),
        )
        self.is_drawer_open = False
        self._current_tab_idx = -1

        self.init_drawer_modules()
        self.navbar.idx_changed.connect(self.handle_panel_toggle)

    def init_drawer_modules(self):
        # Source loading + metadata form; shown via File ▸ Source Properties…
        self.load_panel = LoadPanel(loaders=self.cfg.loader_list())
        self.load_panel.set_load_guard(lambda: self._source_switch_allowed("switching the source"))
        self.load_panel.set_dialog_parent(self)
        self.load_panel.loaded.connect(self._on_loaded)
        self.load_panel.compare_requested.connect(self._on_compare_requested)
        self.source_props_dialog = SourcePropertiesDialog(self.load_panel, parent=self)

        self.viz_panel = VisualsPanel(self.disp_model)
        self.viz_scroll = QScrollArea()
        self.viz_scroll.setWidgetResizable(True)
        self.viz_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.viz_scroll.setFrameShape(QScrollArea.NoFrame)
        self.viz_scroll.setWidget(self.viz_panel)
        self.drawer.add_module(self.viz_scroll)

        self.ann_panel = AnnotationPanel(
            default_mode=str(self.cfg.annotation_default_mode),
            default_label=str(self.cfg.annotation_default_label),
            confirm_each=bool(self.cfg.annotation_confirm_each),
            mask_brush_radius=int(self.cfg.annotation_mask_brush_radius),
            mask_brush_mode=str(self.cfg.annotation_mask_brush_mode),
            bbox_width=float(self.cfg.annotation_bbox_width),
            polyline_width=float(self.cfg.annotation_polyline_width),
        )
        self.drawer.add_module(self.ann_panel)

        self.ann_panel.intent_changed.connect(self._on_annot_intent_changed)
        self._on_annot_intent_changed(self.ann_panel.get_intent())
        self.disp_model.bbox_finished.connect(lambda geom: self.ann_panel.commit_geometry("bbox", geom))
        self.disp_model.polyline_finished.connect(lambda geom: self.ann_panel.commit_geometry("polyline", geom))
        self.disp_model.mask_finished.connect(lambda geom: self.ann_panel.commit_geometry("mask", geom))
        self.disp_model.annotation_selected.connect(self.ann_panel.select_uid)
        self.disp_model.annotation_geometry_edited.connect(self.ann_panel.apply_geometry_patch)
        self.ann_panel.overlay_add.connect(self._on_ann_overlay_add)
        self.ann_panel.overlay_add_many.connect(self._on_ann_overlay_add_many)
        self.ann_panel.overlay_update.connect(self._on_ann_overlay_update)
        self.ann_panel.overlay_delete.connect(self._on_ann_overlay_delete)
        self.ann_panel.overlay_clear.connect(self._on_ann_overlay_clear)
        self.ann_panel.polyline_continue_requested.connect(self.disp_model.continue_selected_polyline)
        self.ann_panel.selection_changed.connect(self._on_annot_selection_changed)

        self.roi_panel = ROIPanel()
        self.drawer.add_module(self.roi_panel)
        self.roi_panel.roi_mode_toggled.connect(self._on_roi_mode)
        self.roi_panel.roi_changed.connect(self.disp_model.set_roi_region)
        self.disp_model.roi_committed.connect(self._on_roi_committed)

        self._roi_windows: list = []
        self._overview_windows: list = []

        # Analyze ▸ Detect Events…
        self.detect_window = DetectEventsWindow(monitors=self.cfg.monitor_list(), parent=self)
        self.detect_window.monitor_run_requested.connect(self._on_monitor_run)
        self.detect_window.monitor_cancel_requested.connect(self._on_monitor_cancel)
        self.detect_window.reset_monitor_feedback()

        axis = getattr(self, "axis_metadata", None)
        if axis is not None:
            if hasattr(self.viz_panel, "set_axis_metadata"):
                self.viz_panel.set_axis_metadata(axis)
            if hasattr(self.ann_panel, "set_axis_metadata"):
                self.ann_panel.set_axis_metadata(axis)
            self.roi_panel.set_axis_metadata(axis)

    def handle_panel_toggle(self, idx):
        self._current_tab_idx = int(idx)
        if idx >= 0:
            self._last_open_tab_idx = idx

        if idx == -1:
            self.is_drawer_open = False
            self.drawer.toggle(False, self.navbar_width, self.central_widget.height())
            self.canvas.native.setFocus()
            self._sync_annotation_gate()
            return

        self.is_drawer_open = True
        self.drawer.set_content_widget(idx)
        self.drawer.toggle(True, self.navbar_width, self.central_widget.height())

        self._sync_annotation_gate()
        if idx != TAB_ROI:
            self.roi_panel.set_roi_mode(False)

    def _resize_drawer(self) -> None:
        self.drawer.resize(self.drawer.target_width, self.central_widget.height())
        if self.is_drawer_open:
            self.drawer.move(self.navbar_width, 0)
        else:
            self.drawer.move(-self.drawer.target_width, 0)
