# filark/gui/main_window.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
import numpy as np

from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout

from .widgets.navbar import NavBar
from .widgets.sliding_drawer import SlidingDrawer
from .modules import VisualsPanel, LoadPanel, AnnotationPanel, AnalysisPanel, AlgorithmsPanel
from .display_model import DisplayModel
from .roi import ROIWindow

from filark.viz.canvas import StreamingCanvas
from filark.io.protocols import Tape
from filark.io.tapeio import NpyTape

def fast_clim(source, *, cap=256, k=3.0):
    a, b = source.shape[:2]
    if a == 0 or b == 0:
        return 0.0, 1.0

    s = min(cap, a, b)
    coords = [
        (0, 0),
        # (max(0, (a - s)//2), max(0, (b - s)//2)),
        # (max(0, a - s),      max(0, b - s)),
    ]

    blocks = [np.asarray(source[i:i+s, j:j+s]) for i, j in coords]
    vals = np.concatenate([blk.ravel() for blk in blocks if blk.size], axis=0)

    if vals.size == 0:
        return 0.0, 1.0

    ma = float(vals.mean())
    st = float(vals.std()) or 1.0
    return ma - k * st, ma + k * st


class LoadedInitMixin:
    """
    Mixin for MainWindow to handle loaded signal during init.
    """
    viz_panel: VisualsPanel
    disp_model: DisplayModel

    def _on_loaded(self, source: Tape):
        vmin, vmax = fast_clim(source)
        self._source = source
        self.disp_model.set_source(source)
        self.viz_panel.set_clim(vmin, vmax)

    def _on_clear(self):
        self.source = None 
        self.ann_panel._on_clear_clicked()
        self.ana_panel.reset()
        self.viz_panel.reset()
        self.disp_model.clear()


class AnnotationMixin:
    def _on_annot_intent_changed(self, intent: dict):
        mode = intent.get("mode", "bbox")
        label = intent.get("preset_label", "event")
        self.disp_model.set_mode(mode)
        self.disp_model.set_label(label)

    def _on_annot_selection_changed(self, uid):
        # optional selection highlight
        pass

    # ---- overlay forwarders ----
    def _on_ann_overlay_add(self, uid: int, anno_type: str, data: dict):
        if anno_type == "bbox":
            self.disp_model.add_bbox_overlay(uid, data)
        elif anno_type == "polyline":
            self.disp_model.add_polyline_overlay(uid, data)

    def _on_ann_overlay_update(self, uid: int, patch: dict):
        self.disp_model.update_overlay(uid, patch)

    def _on_ann_overlay_delete(self, uid: int):
        self.disp_model.remove_overlay(uid)

    def _on_ann_overlay_clear(self):
        self.disp_model.clear_overlays()


class MainWindow(QMainWindow, LoadedInitMixin, AnnotationMixin):
    def __init__(self, theme: str = "dark", source: Optional[Tape] = None):
        super().__init__()
        self.setWindowTitle("FiLark Viz")
        self.resize(1200, 800)

        # source
        self._source = None

        # =========================================
        # 1. Core layout
        # =========================================
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Left: NavBar
        self.navbar_width = 50
        self.tabs = [
            ("Load", "📂"),
            ("Visuals", "🎨"),
            ("Annotation", "🏷️"),
            ("Analysis", "📈"),
            ("Algorithms", "🔧"),
        ]
        self.navbar = NavBar(self.tabs)
        self.navbar.setFixedWidth(self.navbar_width)

        # Right: Canvas
        bgcolor = "white" if theme == "light" else "black"
        self.canvas = StreamingCanvas(bgcolor=bgcolor, wrap_y=True)
        self.disp_model = DisplayModel(self.canvas)

        self.main_layout.addWidget(self.navbar)
        self.main_layout.addWidget(self.canvas.native, 1)

        # =========================================
        # 2. Overlay Drawer
        # =========================================
        self.drawer = SlidingDrawer(self.central_widget, width=280)

        # state
        self.is_drawer_open = False
        self._current_tab_idx = -1

        # init modules
        self.init_drawer_modules()

        # signals
        self.navbar.idx_changed.connect(self.handle_panel_toggle)

        if source is not None:
            self.set_source(source)

    def set_source(self, source: Tape):
        # close old
        old = getattr(self, "_source", None)
        if old is not None and old is not source:
            try:
                old.close()
            except Exception:
                pass

        if isinstance(source, (str, os.PathLike)):
            path = Path(str(source))

            if path.is_file():
                ok = self.load_panel._load_file(path)
                if ok:
                    self.load_panel._sync_info_from_source()
                    self.load_panel.loaded.emit(self.load_panel._current_source)
                    return
            elif path.is_dir():
                ok = self.load_panel._load_folder(path)
                if ok:
                    self.load_panel._sync_info_from_source()
                    self.load_panel.loaded.emit(self.load_panel._current_source)
                    return
            else:
                raise ValueError(f"unkown source: {source}")
        elif isinstance(source, (np.ndarray, np.memmap)):
            source = NpyTape(
                source,
                dims="nt_nc",
                fs=1,
                dx=1,
                dx_unit="m",
            )
        elif isinstance(source, (Tape)):
            pass
        else:
            raise ValueError(f"unkown source: {source}")

        self._on_loaded(source)
        self.load_panel.set_current_source(source)


    def init_drawer_modules(self):
        # 1) Load
        self.load_panel = LoadPanel()
        self.drawer.add_module(self.load_panel)
        self.load_panel.loaded.connect(self._on_loaded)
        self.load_panel.btn_close.clicked.connect(self._on_clear)

        # 2) Visuals
        self.viz_panel = VisualsPanel(self.disp_model)
        self.drawer.add_module(self.viz_panel)

        # 3) Annotation
        self.ann_panel = AnnotationPanel()
        self.drawer.add_module(self.ann_panel)

        # Panel intent -> DisplayModel
        self.ann_panel.intent_changed.connect(self._on_annot_intent_changed)

        # DisplayModel draw finished -> Panel commit
        self.disp_model.bbox_finished.connect(lambda geom: self.ann_panel.commit_geometry("bbox", geom))
        self.disp_model.polyline_finished.connect(lambda geom: self.ann_panel.commit_geometry("polyline", geom))

        # Panel overlay commands -> DisplayModel
        self.ann_panel.overlay_add.connect(self._on_ann_overlay_add)
        self.ann_panel.overlay_update.connect(self._on_ann_overlay_update)
        self.ann_panel.overlay_delete.connect(self._on_ann_overlay_delete)
        self.ann_panel.overlay_clear.connect(self._on_ann_overlay_clear)

        # optional selection
        self.ann_panel.selection_changed.connect(self._on_annot_selection_changed)

        # 4) Analysis
        self.ana_panel = AnalysisPanel()
        self.drawer.add_module(self.ana_panel)

        # ROI mode
        self.ana_panel.roi_mode_toggled.connect(self._on_roi_mode)
        self.ana_panel.roi_changed.connect(self.disp_model._set_roi_region)
        self.disp_model.roi_committed.connect(self._on_roi_committed)

        # run
        self._roi_windows = []
        self.ana_panel.run_requested.connect(self._on_roi_annalysis)

        # 5) Algorithms
        self.alg_panel = AlgorithmsPanel()
        self.drawer.add_module(self.alg_panel)

    # -------------------------
    # ROI hooks
    # -------------------------
    def _on_roi_mode(self, enabled: bool):
        enabled = bool(enabled)

        # suspend annotation panel accepting new ones
        self.ann_panel.suspend(enabled)

        # disable annotation drawing logic
        self.disp_model.set_annotation_enabled(not enabled)

        # enable ROI mode (locks camera)
        self.disp_model.set_analysis_roi_mode(enabled)

        # critical: ROI enabled => gate off regardless of current tab
        if enabled:
            self.disp_model.set_annotation_gate(False)
        else:
            # restore gate based on current tab (only Annotation tab opens gate)
            self.disp_model.set_annotation_gate(self._current_tab_idx == 2)

    def _on_roi_committed(self, roi_dict: dict):
        self.ana_panel.set_roi(**roi_dict, commit=True)

    def _on_roi_annalysis(self, mode, roi):
        roi_data = self._extract_roi_data(roi)
        if roi_data['data'].size == 0:
            return

        win = ROIWindow(roi_data=roi_data, view_type=mode)
        win.resize(1100, 750)
        win.show()

        self._roi_windows.append(win)
        win.destroyed.connect(lambda *_: self._roi_windows.remove(win) if win in self._roi_windows else None)

    def _extract_roi_data(self, roi):
        t0, t1, c0, c1 = roi.t0, roi.t1, roi.c0, roi.c1
        if self._source.dims == 'nt_nc':
            data = self._source[t0:t1, c0:c1]
        else:
            data = self._source[c0:c1, t0:t1].T

        return {
            "data": data,
            "fs": self._source.fs,
            "dx": self._source.dx,
            "roi": roi,
        }

    # -------------------------
    # Drawer toggle
    # -------------------------
    def handle_panel_toggle(self, idx):
        self._current_tab_idx = int(idx)

        if idx == -1:
            # close
            self.is_drawer_open = False
            self.drawer.toggle(False, self.navbar_width, self.central_widget.height())
            self.canvas.native.setFocus()

            # gate off when drawer closed
            self.disp_model.set_annotation_gate(False)
            return

        # open/switch
        self.is_drawer_open = True
        self.drawer.set_content_widget(idx)
        self.drawer.toggle(True, self.navbar_width, self.central_widget.height())

        # gate only on Annotation tab (idx == 2), and only when ROI is not enabled
        # (DisplayModel will ignore if ROI is enabled anyway, but we keep it explicit)
        self.disp_model.set_annotation_gate(idx == 2)
        if idx != 3:
            self.ana_panel.set_roi_mode(False)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.drawer.resize(self.drawer.target_width, self.central_widget.height())
        if self.is_drawer_open:
            self.drawer.move(self.navbar_width, 0)
        else:
            self.drawer.move(-self.drawer.target_width, 0)
