# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""PSDView — Power Spectral Density image for all channels in a ROI."""

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


def _robust_clim(arr: np.ndarray, plo: float = 2.0, phi: float = 98.0):
    a = np.asarray(arr)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0, 1.0
    lo, hi = float(np.percentile(a, plo)), float(np.percentile(a, phi))
    return (lo, lo + 1e-6) if hi <= lo else (lo, hi)


class PSDView(QWidget):
    """
    Power Spectral Density view.

    Displays the Welch PSD estimate for every channel as a 2-D image
    (channel × frequency) and optionally an average PSD line.
    """

    def __init__(self, roi_data: dict, parent=None):
        super().__init__(parent)
        self.roi_data = roi_data

        data = np.asarray(roi_data["data"], dtype=np.float32)
        self._nt, self._nc = data.shape
        self._data = data
        self._fs = float(roi_data.get("fs") or 1.0)
        self._dx = float(roi_data.get("dx") or 1.0)
        self._nperseg = min(256, self._nt)
        self._noverlap = self._nperseg // 2
        self._show_db = True
        self._cmap = "viridis"

        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)

        # Two subplots: top = image, bottom = mean PSD line
        self.ax_img, self.ax_line = self.figure.subplots(
            2, 1, gridspec_kw={"height_ratios": [3, 1]}
        )
        self._im = None
        self._cbar = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self._compute_and_plot()

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def _compute_and_plot(self):
        from scipy.signal import welch as _welch

        nperseg = max(4, min(int(self._nperseg), self._nt))
        noverlap = min(int(self._noverlap), nperseg - 1)

        # compute PSD for all channels
        f, P0 = _welch(
            self._data[:, 0], fs=self._fs,
            nperseg=nperseg, noverlap=noverlap,
        )
        psd = np.empty((self._nc, len(f)), dtype=np.float32)
        psd[0] = P0
        for ic in range(1, self._nc):
            _, Pi = _welch(
                self._data[:, ic], fs=self._fs,
                nperseg=nperseg, noverlap=noverlap,
            )
            psd[ic] = Pi

        if self._show_db:
            eps = 1e-30
            img = 10.0 * np.log10(psd + eps)
            mean_line = 10.0 * np.log10(psd.mean(axis=0) + eps)
            ylabel_img = "Channel"
            cblabel = "PSD (dB/Hz)"
            y_line = "Mean PSD (dB/Hz)"
        else:
            img = psd
            mean_line = psd.mean(axis=0)
            ylabel_img = "Channel"
            cblabel = "PSD (V²/Hz)"
            y_line = "Mean PSD (V²/Hz)"

        self._f_axis = f
        self._psd_img = img
        self._mean_line = mean_line

        # ---- image axes ----
        self.ax_img.clear()
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                print_exception("Failed to remove PSDView colorbar before redraw.", prefix="PSDView")

        vmin, vmax = _robust_clim(img)
        ch_axis = np.arange(self._nc)
        self._im = self.ax_img.pcolormesh(
            f, ch_axis, img,
            cmap=self._cmap, vmin=vmin, vmax=vmax,
            shading="auto",
        )
        self._cbar = self.figure.colorbar(self._im, ax=self.ax_img, shrink=0.85)
        self._cbar.set_label(cblabel)
        self.ax_img.set_ylabel(ylabel_img)
        self.ax_img.set_title("PSD (Welch)")
        self.ax_img.set_xlabel("Frequency (Hz)")

        # ---- mean line axes ----
        self.ax_line.clear()
        self.ax_line.plot(f, mean_line, lw=1.0)
        self.ax_line.set_xlabel("Frequency (Hz)")
        self.ax_line.set_ylabel(y_line)
        self.ax_line.set_title("Mean PSD")
        self.ax_line.set_xlim(float(f[0]), float(f[-1]))

        self.canvas.draw_idle()

        if hasattr(self, "vmin_spin"):
            self.vmin_spin.setValue(vmin)
            self.vmax_spin.setValue(vmax)

    # ------------------------------------------------------------------
    # Controls dock
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

        # ── Welch params ───────────────────────────────────────────
        g = QGroupBox("Welch Parameters")
        f = QFormLayout(g)

        self.sp_nperseg = QSpinBox()
        self.sp_nperseg.setRange(4, 8192)
        self.sp_nperseg.setValue(self._nperseg)
        f.addRow("Window (samp):", self.sp_nperseg)

        self.sp_overlap = QSpinBox()
        self.sp_overlap.setRange(0, 8191)
        self.sp_overlap.setValue(self._noverlap)
        f.addRow("Overlap (samp):", self.sp_overlap)

        self.cb_cmap = QComboBox()
        for c in ["viridis", "plasma", "magma", "inferno", "cividis", "gray"]:
            self.cb_cmap.addItem(c)
        self.cb_cmap.setCurrentText(self._cmap)
        self.cb_cmap.currentTextChanged.connect(self._on_cmap)
        f.addRow("Colormap:", self.cb_cmap)

        self.ck_db = QCheckBox("Show dB")
        self.ck_db.setChecked(True)
        self.ck_db.toggled.connect(self._on_db_toggled)
        f.addRow(self.ck_db)

        btn_apply = QPushButton("Apply")
        btn_apply.clicked.connect(self._on_apply)
        f.addRow(btn_apply)
        vbox.addWidget(g)

        # ── Color limits ───────────────────────────────────────────
        g2 = QGroupBox("Color Limits")
        f2 = QFormLayout(g2)
        sp = QSizePolicy.Expanding

        self.vmin_spin = QDoubleSpinBox()
        self.vmin_spin.setRange(-1e9, 1e9)
        self.vmin_spin.setDecimals(3)
        self.vmin_spin.setSizePolicy(sp, QSizePolicy.Fixed)

        self.vmax_spin = QDoubleSpinBox()
        self.vmax_spin.setRange(-1e9, 1e9)
        self.vmax_spin.setDecimals(3)
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

        if self._psd_img is not None:
            vmin, vmax = _robust_clim(self._psd_img)
            self.vmin_spin.setValue(vmin)
            self.vmax_spin.setValue(vmax)

        return root

    def _on_apply(self):
        self._nperseg = int(self.sp_nperseg.value())
        self._noverlap = int(self.sp_overlap.value())
        self._compute_and_plot()

    def _on_cmap(self, name: str):
        self._cmap = name
        if self._im is not None:
            self._im.set_cmap(name)
            self.canvas.draw_idle()

    def _on_db_toggled(self, v: bool):
        self._show_db = v
        self._compute_and_plot()

    def _set_clim(self, vmin: float, vmax: float):
        if self._im is None:
            return
        if vmax <= vmin:
            vmax = vmin + 1e-6
        self._im.set_clim(vmin, vmax)
        self.canvas.draw_idle()

    def _autoscale(self):
        if self._psd_img is None or self._im is None:
            return
        vmin, vmax = _robust_clim(self._psd_img)
        self._im.set_clim(vmin, vmax)
        if hasattr(self, "vmin_spin"):
            self.vmin_spin.setValue(vmin)
            self.vmax_spin.setValue(vmax)
        self.canvas.draw_idle()
