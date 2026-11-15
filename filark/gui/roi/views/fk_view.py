"""Matplotlib-backed f-k spectrum view for ROI analysis."""

from __future__ import annotations

import re

import numpy as np

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QDoubleSpinBox, QPushButton,
    QCheckBox, QGroupBox, QScrollArea, QSizePolicy, QLineEdit
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

from filark.dsp import fk as fk_ops
from filark._exceptions import print_exception


def _robust_normalize_traces(data: np.ndarray, *, time_first: bool = False) -> np.ndarray:
    """Apply per-trace robust normalization using median and MAD."""
    x = data.T if time_first else np.asarray(data)
    median = np.median(x, axis=1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(x - median), axis=1, keepdims=True) + 1e-12
    out = (x - median) / mad
    return out.T if time_first else out


class FKView(QWidget):
    """ROI f-k viewer with a separate control panel for display and selection."""

    fk_selection_changed = Signal(tuple)

    def __init__(self, roi_data, parent=None):
        super().__init__(parent)
        self.roi_data = roi_data

        self._show_db = True

        data = np.asarray(roi_data.get("data"))
        fs = float(roi_data.get("fs") or 1.0)
        dt = 1 / fs
        dx = float(roi_data.get("dx") or 1.0)

        spectrum, f_axis, k_axis = fk_ops.fk_transform(
            data=_robust_normalize_traces(data, time_first=True),
            dt=dt,
            dx=dx,
            backend="scipy",
            return_power=True,
            shift=True,
        )

        spectrum = np.asarray(spectrum)
        f_axis = np.asarray(f_axis)
        k_axis = np.asarray(k_axis)

        pos = f_axis >= 0
        self.f_axis = f_axis[pos]
        self.k_axis = k_axis

        self.fk_raw = spectrum[pos, :]
        self.fk = self._to_display(self.fk_raw)

        self.k_extent = (float(self.k_axis.min()), float(self.k_axis.max()))
        self.f_extent = (float(self.f_axis.min()), float(self.f_axis.max()))

        self._vel_lines = []
        self._vel_values = [500.0, 1500.0, 3000.0]
        self._vel_lines_visible = False

        self._selector: RectangleSelector | None = None
        self._selection = None

        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self._im = None
        self._cbar = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self._plot_initial()

    def create_controls(self, parent=None) -> QWidget:
        """Build the dockable control panel used by ``ROIWindow``."""
        root = QWidget(parent)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea(root)
        scroll.setWidgetResizable(True)
        root_layout.addWidget(scroll)

        panel = QWidget()
        scroll.setWidget(panel)

        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(10)

        g0 = QGroupBox("Spectrum Display")
        f0 = QFormLayout(g0)

        self.db_cb = QCheckBox("Show dB (log scale)")
        self.db_cb.setChecked(True)
        self.db_cb.toggled.connect(self.set_show_db)
        f0.addRow(self.db_cb)

        vbox.addWidget(g0)

        g1 = QGroupBox("Appearance")
        f1 = QFormLayout(g1)

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["cividis", "viridis", "magma", "plasma", "gray"])
        self.cmap_combo.setCurrentText("cividis")
        self.cmap_combo.currentTextChanged.connect(self.set_cmap)
        f1.addRow("cmap", self.cmap_combo)

        self.vmin_spin = self._spin()
        self.vmax_spin = self._spin()
        vmin, vmax = self._robust_minmax(self.fk)
        self.vmin_spin.setValue(vmin)
        self.vmax_spin.setValue(vmax)
        f1.addRow("vmin", self.vmin_spin)
        f1.addRow("vmax", self.vmax_spin)

        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        btn_apply = QPushButton("Apply")
        btn_auto = QPushButton("Auto")
        btn_apply.clicked.connect(lambda: self.set_clim(self.vmin_spin.value(), self.vmax_spin.value()))
        btn_auto.clicked.connect(self.autoscale_clim)
        h.addWidget(btn_apply)
        h.addWidget(btn_auto)
        h.addStretch(1)
        f1.addRow("clim", row)

        vbox.addWidget(g1)

        g2 = QGroupBox("View Range (axis)")
        f2 = QFormLayout(g2)

        self.kmin_spin = self._spin()
        self.kmax_spin = self._spin()
        self.fmin_spin = self._spin()
        self.fmax_spin = self._spin()

        k0, k1 = self.k_extent
        f0_, f1_ = self.f_extent
        self.kmin_spin.setValue(k0)
        self.kmax_spin.setValue(k1)
        self.fmin_spin.setValue(f0_)
        self.fmax_spin.setValue(f1_)

        f2.addRow("k min", self.kmin_spin)
        f2.addRow("k max", self.kmax_spin)
        f2.addRow("f min", self.fmin_spin)
        f2.addRow("f max", self.fmax_spin)

        row2 = QWidget()
        h2 = QHBoxLayout(row2)
        h2.setContentsMargins(0, 0, 0, 0)
        btn_apply_lim = QPushButton("Apply")
        btn_reset_lim = QPushButton("Reset")
        btn_apply_lim.clicked.connect(self._apply_limits_from_ui)
        btn_reset_lim.clicked.connect(self.reset_limits)
        h2.addWidget(btn_apply_lim)
        h2.addWidget(btn_reset_lim)
        h2.addStretch(1)
        f2.addRow("limits", row2)

        vbox.addWidget(g2)

        g3 = QGroupBox("Selection (fk-domain)")
        f3 = QFormLayout(g3)

        self.sel_cb = QCheckBox("Enable rectangle selection")
        self.sel_cb.toggled.connect(self.enable_fk_selector)
        f3.addRow(self.sel_cb)

        vbox.addWidget(g3)

        g4 = QGroupBox("Velocity Lines")
        f4 = QFormLayout(g4)

        self.vel_cb = QCheckBox("Show velocity lines")
        self.vel_cb.toggled.connect(self.set_velocity_lines_visible)
        f4.addRow(self.vel_cb)

        self.vel_edit = QLineEdit()
        self.vel_edit.setPlaceholderText("e.g. 500,1500,3000")
        self.vel_edit.setText(",".join(str(int(v)) if float(v).is_integer() else str(v) for v in self._vel_values))
        f4.addRow("velocities", self.vel_edit)

        btn_apply_vel = QPushButton("Apply velocities")
        btn_apply_vel.clicked.connect(self._apply_velocities_from_ui)
        f4.addRow(btn_apply_vel)

        vbox.addWidget(g4)

        vbox.addStretch(1)

        self.set_clim(self.vmin_spin.value(), self.vmax_spin.value())
        self._update_colorbar_label()
        return root

    def _plot_initial(self):
        self.ax.clear()
        self._im = self.ax.imshow(
            self.fk,
            origin="lower",
            aspect="auto",
            extent=[self.k_extent[0], self.k_extent[1], self.f_extent[0], self.f_extent[1]],
            cmap="cividis",
        )
        self.ax.set_title("f-k")
        self.ax.set_xlabel("k")
        self.ax.set_ylabel("f (Hz)")

        self._cbar = self.figure.colorbar(self._im, ax=self.ax, shrink=0.85)
        self._update_colorbar_label()

        self.ax.set_xlim(self.k_extent)
        self.ax.set_ylim(self.f_extent)

        self.canvas.draw_idle()

    def set_show_db(self, enabled: bool):
        enabled = bool(enabled)
        if enabled == self._show_db:
            return
        self._show_db = enabled

        self.fk = self._to_display(self.fk_raw)

        if self._im is not None:
            self._im.set_data(self.fk)

            vmin, vmax = self._robust_minmax(self.fk)
            self._im.set_clim(vmin, vmax)

            if hasattr(self, "vmin_spin"):
                self.vmin_spin.blockSignals(True)
                self.vmax_spin.blockSignals(True)
                self.vmin_spin.setValue(vmin)
                self.vmax_spin.setValue(vmax)
                self.vmin_spin.blockSignals(False)
                self.vmax_spin.blockSignals(False)

        self._update_colorbar_label()
        self.canvas.draw_idle()

    def _to_display(self, power_spectrum: np.ndarray) -> np.ndarray:
        """Convert raw power to the currently selected display space."""
        p = np.asarray(power_spectrum)
        if self._show_db:
            eps = 1e-12 * float(np.max(p)) + 1e-12
            return 10.0 * np.log10(p + eps)
        return p

    def _update_colorbar_label(self):
        if self._cbar is None:
            return
        self._cbar.set_label("Power (dB)" if self._show_db else "Power")

    def set_cmap(self, cmap: str):
        if self._im is None:
            return
        self._im.set_cmap(cmap)
        self.canvas.draw_idle()

    def set_clim(self, vmin: float, vmax: float):
        if self._im is None:
            return
        vmin, vmax = float(vmin), float(vmax)
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        self._im.set_clim(vmin, vmax)
        self.canvas.draw_idle()

    def autoscale_clim(self):
        if self._im is None:
            return
        vmin, vmax = self._robust_minmax(self.fk)
        self._im.set_clim(vmin, vmax)
        if hasattr(self, "vmin_spin"):
            self.vmin_spin.setValue(vmin)
            self.vmax_spin.setValue(vmax)
        self.canvas.draw_idle()

    def _apply_limits_from_ui(self):
        kmin = float(self.kmin_spin.value())
        kmax = float(self.kmax_spin.value())
        fmin = float(self.fmin_spin.value())
        fmax = float(self.fmax_spin.value())
        self.set_limits(kmin, kmax, fmin, fmax)

    def set_limits(self, kmin: float, kmax: float, fmin: float, fmax: float):
        if kmax < kmin:
            kmin, kmax = kmax, kmin
        if fmax < fmin:
            fmin, fmax = fmax, fmin
        self.ax.set_xlim(kmin, kmax)
        self.ax.set_ylim(fmin, fmax)
        if self._vel_lines_visible:
            self._draw_velocity_lines()
        self.canvas.draw_idle()

    def reset_limits(self):
        self.ax.set_xlim(self.k_extent[0], self.k_extent[1])
        self.ax.set_ylim(self.f_extent[0], self.f_extent[1])
        if hasattr(self, "kmin_spin"):
            self.kmin_spin.setValue(self.k_extent[0])
            self.kmax_spin.setValue(self.k_extent[1])
            self.fmin_spin.setValue(self.f_extent[0])
            self.fmax_spin.setValue(self.f_extent[1])
        if self._vel_lines_visible:
            self._draw_velocity_lines()
        self.canvas.draw_idle()

    def enable_fk_selector(self, enabled: bool):
        if enabled:
            if self._selector is None:
                self._selector = RectangleSelector(
                    self.ax,
                    onselect=self._on_rect_select,
                    useblit=True,
                    button=[1],
                    interactive=True,
                    drag_from_anywhere=True,
                )
            self._selector.set_active(True)
        else:
            if self._selector is not None:
                self._selector.set_active(False)

    def _on_rect_select(self, eclick, erelease):
        if (
            eclick.xdata is None or eclick.ydata is None or
            erelease.xdata is None or erelease.ydata is None
        ):
            return

        k0, f0 = float(eclick.xdata), float(eclick.ydata)
        k1, f1 = float(erelease.xdata), float(erelease.ydata)

        kmin, kmax = (k0, k1) if k0 <= k1 else (k1, k0)
        fmin, fmax = (f0, f1) if f0 <= f1 else (f1, f0)

        self._selection = (kmin, kmax, fmin, fmax)

        if hasattr(self, "kmin_spin"):
            self.kmin_spin.setValue(kmin)
            self.kmax_spin.setValue(kmax)
            self.fmin_spin.setValue(fmin)
            self.fmax_spin.setValue(fmax)

        self.set_limits(kmin, kmax, fmin, fmax)
        self.fk_selection_changed.emit((kmin, kmax, fmin, fmax))
        self.clear_selection()

    def clear_selection(self):
        self._selection = None
        if self._selector is not None:
            try:
                self._selector.extents = (0, 0, 0, 0)
            except Exception:
                print_exception("Failed to reset FK rectangle selector extents.", prefix="FKView")
        self.canvas.draw_idle()

    def _apply_velocities_from_ui(self):
        text = self.vel_edit.text().strip() if hasattr(self, "vel_edit") else ""
        vels = self._parse_velocities(text)
        if not vels:
            return
        self.set_velocity_lines(vels)

    def set_velocity_lines(self, velocities: list[float]):
        self._vel_values = [float(v) for v in velocities if float(v) > 0]
        if self._vel_lines_visible:
            self._draw_velocity_lines()
            self.canvas.draw_idle()

    def set_velocity_lines_visible(self, visible: bool):
        self._vel_lines_visible = bool(visible)
        if self._vel_lines_visible:
            self._draw_velocity_lines()
        else:
            self._remove_velocity_lines()
        self.canvas.draw_idle()

    def _remove_velocity_lines(self):
        for ln in self._vel_lines:
            try:
                ln.remove()
            except Exception:
                print_exception("Failed to remove an FK velocity overlay line.", prefix="FKView")
        self._vel_lines = []

    def _draw_velocity_lines(self):
        self._remove_velocity_lines()

        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        ymin, ymax = min(y0, y1), max(y0, y1)

        # With k in cycles/m, constant-velocity guides satisfy f = v * |k|.
        k = np.linspace(x0, x1, 400)
        kk = np.abs(k)

        for v in self._vel_values:
            f = v * kk
            mask = (f >= ymin) & (f <= ymax)
            if not np.any(mask):
                continue
            ln, = self.ax.plot(k[mask], f[mask], linewidth=1.0, alpha=0.7)
            self._vel_lines.append(ln)

    def _spin(self) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setDecimals(6)
        s.setRange(-1e12, 1e12)
        s.setSingleStep(0.1)
        s.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        s.setMinimumWidth(120)
        return s

    def _robust_minmax(self, arr: np.ndarray, p_low=1.0, p_high=99.0):
        a = np.asarray(arr)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return 0.0, 1.0
        lo = float(np.percentile(a, p_low))
        hi = float(np.percentile(a, p_high))
        if hi == lo:
            hi = lo + 1e-6
        return lo, hi

    def _parse_velocities(self, text: str) -> list[float]:
        if not text:
            return []
        parts = [part for part in re.split(r"[\s,;]+", text.strip()) if part]
        out = []
        for p in parts:
            try:
                v = float(p)
                if v > 0:
                    out.append(v)
            except ValueError:
                continue
        dedup = []
        seen = set()
        for v in out:
            if v not in seen:
                dedup.append(v)
                seen.add(v)
        return dedup
