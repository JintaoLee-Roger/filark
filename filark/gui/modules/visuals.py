# filark/gui/modules/visuals.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from PySide6.QtCore import Qt, QSignalBlocker
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout, QLabel,
    QDoubleSpinBox, QComboBox, QSpinBox, QCheckBox, QSlider
)
from filark.gui.widgets.typography import PanelTitle
from filark.gui.display_model import DisplayModel

@dataclass
class VisualState:
    vmin: float = 0.0
    vmax: float = 1.0
    cmap: str = "seismic"
    interpolation: str = "nearest"
    height: int = 1024               # number of channels shown (if not full)
    scale_x: int = 1                # time axis compression (>=1)
    scale_y: int = 1                # channel axis compression (>=1)
    full_channels: bool = False
    scroll_enabled: bool = False
    scroll_speed: float = 1.0       # arbitrary unit; you map in canvas


class VisualsPanel(QWidget):
    """
    Visual controls panel for StreamingCanvas.

    Assumptions:
    - Canvas consumes a "source" shaped (nt, nc)
    - Panel sets view params: clim/cmap/interpolation/height/scale/scroll
    """

    def __init__(self, disp_model: DisplayModel, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.disp_model = disp_model
        self.state = VisualState()

        self._build_ui()
        self._wire()
        self._apply_all_to_canvas()  # apply defaults

    # ---------------- UI ----------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = PanelTitle("Visuals")
        root.addWidget(title)

        # --- Contrast / clim ---
        g_clim = QGroupBox("clim")
        f1 = QFormLayout(g_clim)

        self.sp_vmin = QDoubleSpinBox()
        self.sp_vmin.setRange(-1e30, 1e30)
        self.sp_vmin.setDecimals(4)
        self.sp_vmin.setSingleStep(0.1)
        self.sp_vmin.setValue(-1)

        self.sp_vmax = QDoubleSpinBox()
        self.sp_vmax.setRange(-1e30, 1e30)
        self.sp_vmax.setDecimals(4)
        self.sp_vmax.setSingleStep(0.1)
        self.sp_vmax.setValue(1)

        f1.addRow("vmin:", self.sp_vmin)
        f1.addRow("vmax:", self.sp_vmax)
        root.addWidget(g_clim)

        # --- Rendering ---
        g_r = QGroupBox("Rendering")
        f2 = QFormLayout(g_r)

        self.cb_cmap = QComboBox()
        # 先给一套常用的；你也可以从 matplotlib/vispy 里动态拉列表
        self.cb_cmap.addItems([
            "seismic", "gray", "viridis", "RdBu_r", "RdBu", "magma", "inferno", "plasma",
            "turbo", "cividis", "coolwarm"
        ])

        self.cb_interp = QComboBox()
        self.cb_interp.addItems(["nearest", "linear"])

        f2.addRow("cmap:", self.cb_cmap)
        f2.addRow("interpolation:", self.cb_interp)
        root.addWidget(g_r)

        # --- View geometry ---
        g_view = QGroupBox("View")
        f3 = QFormLayout(g_view)

        # Full channels toggle
        self.ck_full = QCheckBox("Full channels")
        self.ck_full.setChecked(False)
        f3.addRow("", self.ck_full)

        # Height (channels shown)
        self.sp_height = QSpinBox()
        self.sp_height.setRange(1, 10_000_000)
        self.sp_height.setValue(self.state.height)
        f3.addRow("height:", self.sp_height)

        # Scale x/y (>=1)
        row_scale = QWidget()
        h_scale = QHBoxLayout(row_scale)
        h_scale.setContentsMargins(0, 0, 0, 0)
        h_scale.setSpacing(8)

        self.sp_scale_x = QSpinBox()
        self.sp_scale_x.setRange(1, 1_000_000)
        self.sp_scale_x.setValue(self.state.scale_x)

        self.sp_scale_y = QSpinBox()
        self.sp_scale_y.setRange(1, 1_000_000)
        self.sp_scale_y.setValue(self.state.scale_y)

        h_scale.addWidget(QLabel("x"))
        h_scale.addWidget(self.sp_scale_x, 1)
        h_scale.addSpacing(8)
        h_scale.addWidget(QLabel("y"))
        h_scale.addWidget(self.sp_scale_y, 1)

        f3.addRow("scale:", row_scale)
        root.addWidget(g_view)

        # --- Scrolling ---
        g_scroll = QGroupBox("Scrolling")
        f4 = QFormLayout(g_scroll)

        self.ck_scroll = QCheckBox("Enable scrolling")
        self.ck_scroll.setChecked(False)
        f4.addRow("", self.ck_scroll)

        # Speed slider (0.1 .. 10.0)
        self.sl_speed = QSlider(Qt.Horizontal)
        self.sl_speed.setRange(1, 100)   # map to 0.1..10.0
        self.sl_speed.setValue(10)       # 1.0
        self.lb_speed = QLabel("1.0")
        row_speed = QWidget()
        h_speed = QHBoxLayout(row_speed)
        h_speed.setContentsMargins(0, 0, 0, 0)
        h_speed.setSpacing(8)
        h_speed.addWidget(self.sl_speed, 1)
        h_speed.addWidget(self.lb_speed)
        f4.addRow("speed:", row_speed)

        root.addWidget(g_scroll)

        root.addStretch(1)

        # initial disable for full channels
        self._update_enabled_states()

    def _wire(self):
        # clim
        self.sp_vmin.valueChanged.connect(self._on_clim_changed)
        self.sp_vmax.valueChanged.connect(self._on_clim_changed)

        # rendering
        self.cb_cmap.currentTextChanged.connect(self._on_cmap_changed)
        self.cb_interp.currentTextChanged.connect(self._on_interp_changed)

        # view
        # self.ck_full.toggled.connect(self._on_full_toggled)
        # self.sp_height.valueChanged.connect(self._on_height_changed)
        # self.sp_scale_x.valueChanged.connect(self._on_scale_changed)
        # self.sp_scale_y.valueChanged.connect(self._on_scale_changed)

        # scroll
        # self.ck_scroll.toggled.connect(self._on_scroll_toggled)
        # self.sl_speed.valueChanged.connect(self._on_scroll_speed_changed)

    # ---------------- External API ----------------
    def set_clim(self, vmin: float, vmax: float):
        """
        Called by LoadPanel after auto-estimating vmin/vmax.
        No auto-scale button; this is the main entry.
        """
        if vmax == vmin:
            vmax = vmin + 0.01

        self.state.vmin = float(vmin)
        self.state.vmax = float(vmax)

        # update UI without triggering feedback loops
        with QSignalBlocker(self.sp_vmin), QSignalBlocker(self.sp_vmax):
            self.sp_vmin.setValue(self.state.vmin)
            self.sp_vmax.setValue(self.state.vmax)

        self._apply_clim_to_canvas()

    def set_source_shape_hint(self, nt: int, nc: int):
        """
        Optional: if you want the panel to adjust defaults based on data size.
        Not required.
        """
        # e.g. set height default to min(nc, 512) when not full
        if nc > 0:
            self.state.height = min(nc, 512)
            with QSignalBlocker(self.sp_height):
                self.sp_height.setValue(self.state.height)

    # ---------------- Handlers ----------------
    def _on_clim_changed(self):
        vmin = float(self.sp_vmin.value())
        vmax = float(self.sp_vmax.value())

        # keep invariant vmin < vmax (soft)
        if vmax == vmin:
            vmax = vmin + 1e-6
            with QSignalBlocker(self.sp_vmax):
                self.sp_vmax.setValue(vmax)

        # if user swapped them, auto swap (user-friendly)
        if vmax < vmin:
            vmin, vmax = vmax, vmin
            with QSignalBlocker(self.sp_vmin), QSignalBlocker(self.sp_vmax):
                self.sp_vmin.setValue(vmin)
                self.sp_vmax.setValue(vmax)

        self.state.vmin, self.state.vmax = vmin, vmax
        self._apply_clim_to_canvas()

    def _on_cmap_changed(self, name: str):
        self.state.cmap = name
        self._apply_cmap_to_canvas()

    def _on_interp_changed(self, name: str):
        self.state.interpolation = name
        self._apply_interp_to_canvas()

    def _on_full_toggled(self, on: bool):
        self.state.full_channels = bool(on)
        self._update_enabled_states()
        self._apply_view_to_canvas()

    def _on_height_changed(self, v: int):
        self.state.height = int(v)
        if not self.state.full_channels:
            self._apply_view_to_canvas()

    def _on_scale_changed(self):
        self.state.scale_x = int(self.sp_scale_x.value())
        self.state.scale_y = int(self.sp_scale_y.value())
        self._apply_view_to_canvas()

    def _on_scroll_toggled(self, on: bool):
        self.state.scroll_enabled = bool(on)
        self._apply_scroll_to_canvas()

    def _on_scroll_speed_changed(self, v: int):
        # map 1..100 -> 0.1..10.0
        speed = round(v / 10.0, 1)
        self.state.scroll_speed = float(speed)
        self.lb_speed.setText(f"{speed:.1f}")
        if self.state.scroll_enabled:
            self._apply_scroll_to_canvas()

    def _update_enabled_states(self):
        # When full channels is on, height is irrelevant; scale_y could be allowed or not.
        # 这里我默认：full 时禁用 height，但 scale_y 仍可用（压缩通道显示更常见）。
        self.sp_height.setEnabled(not self.state.full_channels)

    # ---------------- Canvas apply (adapter) ----------------
    def _apply_all_to_canvas(self):
        self._apply_clim_to_canvas()
        self._apply_cmap_to_canvas()
        self._apply_interp_to_canvas()
        # self._apply_view_to_canvas()
        # self._apply_scroll_to_canvas()

    def _apply_clim_to_canvas(self):
        vmin, vmax = self.state.vmin, self.state.vmax
        # Try common APIs
        self.disp_model.set_clim(vmin, vmax)


    def _apply_cmap_to_canvas(self):
        cmap = self.state.cmap

        self.disp_model.set_cmap(cmap)


    def _apply_interp_to_canvas(self):
        interp = self.state.interpolation
        self.disp_model.set_interpolation(interp)

    def _apply_view_to_canvas(self):
        full = self.state.full_channels
        height = self.state.height
        sx, sy = self.state.scale_x, self.state.scale_y
        self.disp_model.set_scale(sx, sy)


    def _apply_scroll_to_canvas(self):
        on = self.state.scroll_enabled
        sp = self.state.scroll_speed

        self.disp_model.set_scroll(on, sp)

