# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""RMSView — per-channel RMS energy profile for a ROI."""

from __future__ import annotations

import numpy as np

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class RMSView(QWidget):
    """
    RMS (Root-Mean-Square) energy profile.

    Shows the per-channel RMS amplitude as a horizontal bar / line chart.
    Optionally shows a windowed RMS heatmap (time × channel).
    """

    def __init__(self, roi_data: dict, parent=None):
        super().__init__(parent)
        self.roi_data = roi_data

        data = np.asarray(roi_data["data"], dtype=np.float32)
        self._nt, self._nc = data.shape
        self._data = data
        self._dx = float(roi_data.get("dx") or 1.0)
        self._fs = float(roi_data.get("fs") or 1.0)
        self._mode = "profile"  # "profile" or "heatmap"
        self._n_windows = 20    # for heatmap mode

        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self._plot()

    # ------------------------------------------------------------------

    def _plot(self):
        self.ax.clear()
        data = self._data
        nt, nc = self._nt, self._nc

        if self._mode == "profile":
            rms = np.sqrt(np.mean(data ** 2, axis=0))  # (nc,)
            ch_axis = np.arange(nc) * self._dx
            self.ax.barh(ch_axis, rms, height=self._dx * 0.8, align="center")
            self.ax.set_xlabel("RMS amplitude")
            self.ax.set_ylabel("Channel distance (m)" if self._dx != 1.0 else "Channel index")
            self.ax.set_title("Per-channel RMS energy")
            self.ax.invert_yaxis()

        else:  # heatmap
            n_win = max(1, int(self._n_windows))
            win_len = max(1, nt // n_win)
            n_segs = nt // win_len
            if n_segs < 1:
                n_segs = 1
                win_len = nt

            rms_map = np.zeros((n_segs, nc), dtype=np.float32)
            for i in range(n_segs):
                seg = data[i * win_len: (i + 1) * win_len, :]
                rms_map[i, :] = np.sqrt(np.mean(seg ** 2, axis=0))

            t_axis = np.arange(n_segs) * win_len / self._fs
            ch_axis = np.arange(nc)

            im = self.ax.pcolormesh(
                ch_axis, t_axis, rms_map,
                cmap="hot", shading="auto",
            )
            self.figure.colorbar(im, ax=self.ax, shrink=0.85, label="RMS")
            self.ax.set_xlabel("Channel index")
            self.ax.set_ylabel("Time (s)")
            self.ax.set_title("RMS energy heatmap (time × channel)")

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

        g = QGroupBox("Display Mode")
        f = QFormLayout(g)

        self.cb_mode = QComboBox()
        self.cb_mode.addItem("Profile (per channel)", "profile")
        self.cb_mode.addItem("Heatmap (time × channel)", "heatmap")
        self.cb_mode.currentIndexChanged.connect(self._on_mode_changed)
        f.addRow("Mode:", self.cb_mode)

        btn = QPushButton("Refresh")
        btn.clicked.connect(self._plot)
        f.addRow(btn)

        vbox.addWidget(g)
        vbox.addStretch(1)
        return root

    def _on_mode_changed(self):
        self._mode = self.cb_mode.currentData()
        self._plot()
