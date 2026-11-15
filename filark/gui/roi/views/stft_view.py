# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""STFTView — Short-Time Fourier Transform spectrogram for a ROI."""

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


class STFTView(QWidget):
    """
    STFT spectrogram view.

    Shows the Short-Time Fourier Transform of a selected channel (or channel
    average) as a time-frequency image.  Controls are in a separate dock
    (ROIWindow calls ``create_controls()``).
    """

    def __init__(self, roi_data: dict, parent=None):
        super().__init__(parent)
        self.roi_data = roi_data

        data = np.asarray(roi_data["data"], dtype=np.float32)  # (nt, nc)
        nt, nc = data.shape
        self._data = data
        self._nc = nc
        self._fs = float(roi_data.get("fs") or 1.0)
        self._nperseg = min(256, nt)
        self._noverlap = self._nperseg // 2
        self._ch_idx = 0       # channel index, -1 = average
        self._show_db = True

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
    # Computation
    # ------------------------------------------------------------------

    def _get_channel_data(self) -> np.ndarray:
        if self._ch_idx < 0:
            return self._data.mean(axis=1)
        return self._data[:, self._ch_idx]

    def _compute_and_plot(self):
        from scipy.signal import stft as _stft

        trace = self._get_channel_data()
        nperseg = max(4, min(int(self._nperseg), len(trace)))
        noverlap = min(int(self._noverlap), nperseg - 1)

        f, t, Z = _stft(
            trace, fs=self._fs,
            nperseg=nperseg, noverlap=noverlap,
            window="hann",
        )
        power = np.abs(Z) ** 2  # (nf, n_t)

        if self._show_db:
            eps = 1e-12 * float(np.max(power)) + 1e-12
            img = 10.0 * np.log10(power + eps)
            cblabel = "Power (dB)"
        else:
            img = power
            cblabel = "Power"

        self._f_axis = f
        self._t_axis = t
        self._img = img

        self.ax.clear()
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                print_exception("Failed to remove STFTView colorbar before redraw.", prefix="STFTView")

        vmin, vmax = _robust_clim(img)
        self._im = self.ax.pcolormesh(
            t, f, img,
            cmap=self._cmap if hasattr(self, "_cmap") else "viridis",
            vmin=vmin, vmax=vmax,
            shading="auto",
        )
        self._cbar = self.figure.colorbar(self._im, ax=self.ax, shrink=0.85)
        self._cbar.set_label(cblabel)

        ch_lbl = "avg" if self._ch_idx < 0 else str(self._ch_idx)
        self.ax.set_title(f"STFT — channel {ch_lbl}")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Frequency (Hz)")

        self.canvas.draw_idle()

        # Sync UI controls if they exist
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

        # ── Channel ────────────────────────────────────────────────
        g_ch = QGroupBox("Channel")
        f_ch = QFormLayout(g_ch)

        self.cb_channel = QComboBox()
        self.cb_channel.addItem("Average", -1)
        for i in range(self._nc):
            self.cb_channel.addItem(f"Ch {i}", i)
        self.cb_channel.currentIndexChanged.connect(self._on_channel_changed)
        f_ch.addRow("Channel:", self.cb_channel)
        vbox.addWidget(g_ch)

        # ── STFT params ────────────────────────────────────────────
        g_stft = QGroupBox("STFT Parameters")
        f_stft = QFormLayout(g_stft)

        self.sp_nperseg = QSpinBox()
        self.sp_nperseg.setRange(4, 8192)
        self.sp_nperseg.setValue(self._nperseg)
        f_stft.addRow("Window (samp):", self.sp_nperseg)

        self.sp_overlap = QSpinBox()
        self.sp_overlap.setRange(0, 8191)
        self.sp_overlap.setValue(self._noverlap)
        f_stft.addRow("Overlap (samp):", self.sp_overlap)

        self._cmap = "viridis"
        self.cb_cmap = QComboBox()
        for c in ["viridis", "plasma", "magma", "inferno", "cividis", "gray"]:
            self.cb_cmap.addItem(c, c)
        self.cb_cmap.setCurrentText(self._cmap)
        self.cb_cmap.currentTextChanged.connect(self._on_cmap_changed)
        f_stft.addRow("Colormap:", self.cb_cmap)

        self.ck_db = QCheckBox("Show dB")
        self.ck_db.setChecked(True)
        self.ck_db.toggled.connect(self._on_db_toggled)
        f_stft.addRow(self.ck_db)

        btn_apply = QPushButton("Apply")
        btn_apply.clicked.connect(self._on_apply_stft)
        f_stft.addRow(btn_apply)
        vbox.addWidget(g_stft)

        # ── Color limits ───────────────────────────────────────────
        g_clim = QGroupBox("Color Limits")
        f_clim = QFormLayout(g_clim)

        self.vmin_spin = QDoubleSpinBox()
        self.vmin_spin.setRange(-1e9, 1e9)
        self.vmin_spin.setDecimals(3)
        self.vmin_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.vmax_spin = QDoubleSpinBox()
        self.vmax_spin.setRange(-1e9, 1e9)
        self.vmax_spin.setDecimals(3)
        self.vmax_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        f_clim.addRow("vmin:", self.vmin_spin)
        f_clim.addRow("vmax:", self.vmax_spin)

        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        btn_clim = QPushButton("Apply")
        btn_auto = QPushButton("Auto")
        btn_clim.clicked.connect(
            lambda: self._apply_clim(self.vmin_spin.value(), self.vmax_spin.value())
        )
        btn_auto.clicked.connect(self._autoscale_clim)
        h.addWidget(btn_clim)
        h.addWidget(btn_auto)
        f_clim.addRow(row)

        # Init clim values
        if self._img is not None:
            vmin, vmax = _robust_clim(self._img)
            self.vmin_spin.setValue(vmin)
            self.vmax_spin.setValue(vmax)

        vbox.addWidget(g_clim)
        vbox.addStretch(1)
        return root

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_channel_changed(self, idx: int):
        self._ch_idx = self.cb_channel.currentData()
        self._compute_and_plot()

    def _on_apply_stft(self):
        self._nperseg = int(self.sp_nperseg.value())
        self._noverlap = int(self.sp_overlap.value())
        self._compute_and_plot()

    def _on_cmap_changed(self, name: str):
        self._cmap = name
        if self._im is not None:
            self._im.set_cmap(name)
            self.canvas.draw_idle()

    def _on_db_toggled(self, checked: bool):
        self._show_db = checked
        self._compute_and_plot()

    def _apply_clim(self, vmin: float, vmax: float):
        if self._im is None:
            return
        if vmax <= vmin:
            vmax = vmin + 1e-6
        self._im.set_clim(vmin, vmax)
        self.canvas.draw_idle()

    def _autoscale_clim(self):
        if self._img is None or self._im is None:
            return
        vmin, vmax = _robust_clim(self._img)
        self._im.set_clim(vmin, vmax)
        if hasattr(self, "vmin_spin"):
            self.vmin_spin.setValue(vmin)
            self.vmax_spin.setValue(vmax)
        self.canvas.draw_idle()
