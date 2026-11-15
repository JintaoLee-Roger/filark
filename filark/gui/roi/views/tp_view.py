# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""TpView — linear τ-p (Radon) transform view for a ROI."""

from __future__ import annotations

import numpy as np

from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from filark._exceptions import print_exception


def _robust_clim(arr, plo=2.0, phi=98.0):
    a = np.asarray(arr)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return -1.0, 1.0
    lo, hi = float(np.percentile(a, plo)), float(np.percentile(a, phi))
    return (lo, lo + 1e-6) if hi <= lo else (lo, hi)


class TpView(QWidget):
    """
    Linear τ-p (Radon) transform view.

    Applies the linear Radon forward transform and displays the result
    as a τ-p image.  p values are in s/m (slowness).
    """

    def __init__(self, roi_data: dict, parent=None):
        super().__init__(parent)
        self.roi_data = roi_data

        data = np.asarray(roi_data["data"], dtype=np.float32)
        self._nt, self._nc = data.shape
        self._data = data
        self._fs = float(roi_data.get("fs") or 1.0)
        self._dx = float(roi_data.get("dx") or 1.0)
        self._dt = 1.0 / self._fs

        # Default p range based on dx and fs
        # p = 1/v ; v_min ~ 100 m/s → p_max = 0.01 s/m; v_max ~ 10000 m/s → p_min = 0.0001 s/m
        p_max = 1.0 / max(100.0, 1e-3)
        p_max = min(p_max, 1.0 / (self._dx * self._fs * 0.5 + 1e-12))
        self._p_min = -float(p_max)
        self._p_max = float(p_max)
        self._n_p = 100
        self._cmap = "seismic"

        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self._im = None
        self._cbar = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self._compute_and_plot()

    # ------------------------------------------------------------------

    def _compute_and_plot(self):
        from filark.dsp.tp import linear_radon_forward

        p_values = np.linspace(self._p_min, self._p_max, int(self._n_p))
        tp_data = linear_radon_forward(
            self._data,
            p_values=p_values,
            dt=self._dt,
            dx=self._dx,
            layout="nt_nc",
        )
        # tp_data: (nt, n_p)
        t_axis = np.arange(self._nt) * self._dt

        self.ax.clear()
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                print_exception("Failed to remove TpView colorbar before redraw.", prefix="TpView")

        # tp_data shape: (n_p, nt) → transpose to (nt, n_p) for pcolormesh
        tp_plot = tp_data.T  # (nt, n_p)
        vmin, vmax = _robust_clim(tp_plot)
        self._im = self.ax.pcolormesh(
            p_values * 1e3,  # X: p axis in ms/m
            t_axis,           # Y: τ axis in s
            tp_plot,          # C: (nt, n_p)
            cmap=self._cmap, vmin=vmin, vmax=vmax,
            shading="auto",
        )
        self._cbar = self.figure.colorbar(self._im, ax=self.ax, shrink=0.85)
        self._cbar.set_label("Amplitude")
        self.ax.set_xlabel("p (ms/m)")
        self.ax.set_ylabel("τ (s)")
        self.ax.set_title("Linear τ-p transform")

        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def create_controls(self, parent=None) -> QWidget:
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

        g = QGroupBox("Radon Parameters")
        f = QFormLayout(g)

        def _dspin(val, lo=-1e6, hi=1e6, dec=6):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(dec)
            s.setValue(val)
            s.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            return s

        self.sp_pmin = _dspin(self._p_min * 1e3, -1e6, 1e6, 4)
        self.sp_pmin.setSuffix(" ms/m")
        self.sp_pmax = _dspin(self._p_max * 1e3, -1e6, 1e6, 4)
        self.sp_pmax.setSuffix(" ms/m")
        self.sp_np = QSpinBox()
        self.sp_np.setRange(10, 1000)
        self.sp_np.setValue(self._n_p)

        f.addRow("p min:", self.sp_pmin)
        f.addRow("p max:", self.sp_pmax)
        f.addRow("n_p:", self.sp_np)

        self.cb_cmap = QComboBox()
        for c in ["seismic", "RdBu_r", "bwr", "viridis", "gray"]:
            self.cb_cmap.addItem(c)
        self.cb_cmap.setCurrentText(self._cmap)
        self.cb_cmap.currentTextChanged.connect(self._on_cmap)
        f.addRow("Colormap:", self.cb_cmap)

        btn = QPushButton("Compute")
        btn.clicked.connect(self._on_compute)
        f.addRow(btn)

        vbox.addWidget(g)

        g2 = QGroupBox("Color Limits")
        f2 = QFormLayout(g2)

        self.vmin_spin = _dspin(0, -1e9, 1e9, 4)
        self.vmax_spin = _dspin(1, -1e9, 1e9, 4)
        f2.addRow("vmin:", self.vmin_spin)
        f2.addRow("vmax:", self.vmax_spin)

        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        btn_c = QPushButton("Apply")
        btn_a = QPushButton("Auto")
        btn_c.clicked.connect(
            lambda: self._set_clim(self.vmin_spin.value(), self.vmax_spin.value())
        )
        btn_a.clicked.connect(self._autoscale)
        h.addWidget(btn_c); h.addWidget(btn_a)
        f2.addRow(row)

        vbox.addWidget(g2)
        vbox.addStretch(1)
        return root

    def _on_compute(self):
        self._p_min = float(self.sp_pmin.value()) * 1e-3  # ms/m → s/m
        self._p_max = float(self.sp_pmax.value()) * 1e-3
        self._n_p = int(self.sp_np.value())
        self._compute_and_plot()

    def _on_cmap(self, name: str):
        self._cmap = name
        if self._im is not None:
            self._im.set_cmap(name)
            self.canvas.draw_idle()

    def _set_clim(self, vmin: float, vmax: float):
        if self._im is None:
            return
        if vmax <= vmin:
            vmax = vmin + 1e-6
        self._im.set_clim(vmin, vmax)
        self.canvas.draw_idle()

    def _autoscale(self):
        if self._im is None:
            return
        arr = self._im.get_array()
        if arr is None:
            return
        vmin, vmax = _robust_clim(np.asarray(arr))
        self._im.set_clim(vmin, vmax)
        if hasattr(self, "vmin_spin"):
            self.vmin_spin.setValue(vmin)
            self.vmax_spin.setValue(vmax)
        self.canvas.draw_idle()
