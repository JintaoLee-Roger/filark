# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""XcorrView — cross-correlation gather (channel vs lag) for a ROI."""

from __future__ import annotations

import numpy as np

from PySide6.QtWidgets import (
    QCheckBox,
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


class XcorrView(QWidget):
    """
    Cross-correlation gather view.

    Shows the cross-correlation of every channel against a reference
    channel as a 2-D image (channel × lag).
    """

    def __init__(self, roi_data: dict, parent=None):
        super().__init__(parent)
        self.roi_data = roi_data

        data = np.asarray(roi_data["data"], dtype=np.float32)
        self._nt, self._nc = data.shape
        self._data = data
        self._fs = float(roi_data.get("fs") or 1.0)
        self._dx = float(roi_data.get("dx") or 1.0)
        self._ref_ch = 0
        self._max_lag = min(self._nt // 4, 512)
        self._normalize = True
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
        from filark.dsp.xcorr import xcorr_gather

        ref = np.clip(int(self._ref_ch), 0, self._nc - 1)
        ml = int(self._max_lag)

        corr, lags = xcorr_gather(
            self._data,
            ref_channel=ref,
            max_lag=ml,
            normalize=bool(self._normalize),
            layout="nt_nc",
        )
        # corr: (n_lag, nc), lags: (n_lag,)
        lag_s = lags / self._fs

        self.ax.clear()
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                print_exception("Failed to remove XcorrView colorbar before redraw.", prefix="XcorrView")

        ch_axis = np.arange(self._nc) * self._dx
        vmin, vmax = _robust_clim(corr)
        self._im = self.ax.pcolormesh(
            lag_s, ch_axis, corr.T,
            cmap=self._cmap, vmin=vmin, vmax=vmax,
            shading="auto",
        )
        self._cbar = self.figure.colorbar(self._im, ax=self.ax, shrink=0.85)
        self._cbar.set_label("Normalized xcorr" if self._normalize else "xcorr")

        self.ax.axvline(0, color="k", lw=0.5, ls="--", alpha=0.4)
        self.ax.set_xlabel("Lag (s)")
        self.ax.set_ylabel("Channel distance (m)" if self._dx != 1.0 else "Channel index")
        self.ax.set_title(f"xcorr gather — ref ch {ref}")

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

        g = QGroupBox("xcorr Parameters")
        f = QFormLayout(g)

        self.sp_ref = QSpinBox()
        self.sp_ref.setRange(0, max(0, self._nc - 1))
        self.sp_ref.setValue(self._ref_ch)
        f.addRow("Reference ch:", self.sp_ref)

        self.sp_max_lag = QSpinBox()
        self.sp_max_lag.setRange(1, self._nt - 1)
        self.sp_max_lag.setValue(self._max_lag)
        f.addRow("Max lag (samp):", self.sp_max_lag)

        self.ck_norm = QCheckBox("Normalize")
        self.ck_norm.setChecked(self._normalize)
        f.addRow(self.ck_norm)

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
        sp = QSizePolicy.Expanding

        self.vmin_spin = QDoubleSpinBox()
        self.vmin_spin.setRange(-1e9, 1e9)
        self.vmin_spin.setDecimals(4)
        self.vmin_spin.setSizePolicy(sp, QSizePolicy.Fixed)

        self.vmax_spin = QDoubleSpinBox()
        self.vmax_spin.setRange(-1e9, 1e9)
        self.vmax_spin.setDecimals(4)
        self.vmax_spin.setSizePolicy(sp, QSizePolicy.Fixed)

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
        self._ref_ch = int(self.sp_ref.value())
        self._max_lag = int(self.sp_max_lag.value())
        self._normalize = self.ck_norm.isChecked()
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
        data = self._im.get_array()
        if data is None:
            return
        vmin, vmax = _robust_clim(np.asarray(data))
        self._im.set_clim(vmin, vmax)
        if hasattr(self, "vmin_spin"):
            self.vmin_spin.setValue(vmin)
            self.vmax_spin.setValue(vmax)
        self.canvas.draw_idle()
