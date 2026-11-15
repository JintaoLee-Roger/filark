# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""SVDView — Singular Value Decomposition analysis view for a ROI."""

from __future__ import annotations

import numpy as np

from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class SVDView(QWidget):
    """
    SVD (Singular Value Decomposition) analysis view.

    Left panel: singular value spectrum (bar chart) with cumulative energy.
    Right panel: reconstructed data using the top-k singular components.

    This helps identify:
    - How many dominant modes exist (rank of the signal)
    - What each low-rank component looks like
    """

    def __init__(self, roi_data: dict, parent=None):
        super().__init__(parent)
        self.roi_data = roi_data

        data = np.asarray(roi_data["data"], dtype=np.float32)
        self._nt, self._nc = data.shape
        self._data = data
        self._fs = float(roi_data.get("fs") or 1.0)
        self._n_show = min(20, min(self._nt, self._nc))  # singular values to show
        self._n_reconstruct = 3                           # components to reconstruct
        self._cmap = "seismic"

        # Compute SVD once
        self._u, self._s, self._vt = np.linalg.svd(data, full_matrices=False)

        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self._plot()

    # ------------------------------------------------------------------

    def _plot(self):
        self.figure.clear()
        axes = self.figure.subplots(1, 2, gridspec_kw={"width_ratios": [1, 2]})
        ax_sv, ax_rec = axes

        # ── Left: singular value bar chart ────────────────────────
        s = self._s[:int(self._n_show)]
        idx = np.arange(len(s))
        energy_frac = np.cumsum(s ** 2) / (np.sum(self._s ** 2) + 1e-30) * 100

        ax_sv.bar(idx, s, color="steelblue", alpha=0.8)
        ax_sv2 = ax_sv.twinx()
        ax_sv2.plot(idx, energy_frac, "r.-", ms=4, lw=1.0)
        ax_sv2.set_ylabel("Cumulative energy (%)", color="r")
        ax_sv2.tick_params(axis="y", colors="r")
        ax_sv2.set_ylim(0, 105)

        ax_sv.set_xlabel("Component index")
        ax_sv.set_ylabel("Singular value")
        ax_sv.set_title(f"Singular values (top {len(s)})")

        # Mark the n_reconstruct threshold
        k = int(self._n_reconstruct)
        if 0 < k <= len(s):
            ax_sv.axvline(k - 0.5, color="orange", lw=1.5, ls="--", label=f"k={k}")
            ax_sv.legend(fontsize=8)

        # ── Right: reconstructed image ─────────────────────────────
        u = self._u
        s_full = self._s
        vt = self._vt

        # Reconstruct using first k components
        k = max(1, min(int(self._n_reconstruct), len(s_full)))
        rec = (u[:, :k] * s_full[None, :k]) @ vt[:k, :]

        t_axis = np.arange(self._nt) / self._fs
        ch_axis = np.arange(self._nc)

        # Robust clim
        flat = rec.ravel()
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            vabs = 1.0
        else:
            vabs = float(np.percentile(np.abs(flat), 98)) or 1.0

        im = ax_rec.pcolormesh(
            ch_axis, t_axis, rec,
            cmap=self._cmap, vmin=-vabs, vmax=vabs,
            shading="auto",
        )
        self.figure.colorbar(im, ax=ax_rec, shrink=0.85)
        ax_rec.set_xlabel("Channel index")
        ax_rec.set_ylabel("Time (s)")
        ax_rec.set_title(f"Reconstruction (top {k} components)")

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

        g = QGroupBox("SVD Display")
        f = QFormLayout(g)

        max_sv = min(min(self._nt, self._nc), 200)

        self.sp_n_show = QSpinBox()
        self.sp_n_show.setRange(1, max_sv)
        self.sp_n_show.setValue(self._n_show)
        f.addRow("Show N singular values:", self.sp_n_show)

        self.sp_n_rec = QSpinBox()
        self.sp_n_rec.setRange(1, max_sv)
        self.sp_n_rec.setValue(self._n_reconstruct)
        f.addRow("Reconstruct (k comps):", self.sp_n_rec)

        self.cb_cmap = QComboBox()
        for c in ["seismic", "RdBu_r", "bwr", "viridis", "gray"]:
            self.cb_cmap.addItem(c)
        self.cb_cmap.setCurrentText(self._cmap)
        f.addRow("Colormap:", self.cb_cmap)

        btn = QPushButton("Refresh")
        btn.clicked.connect(self._on_refresh)
        f.addRow(btn)

        vbox.addWidget(g)
        vbox.addStretch(1)
        return root

    def _on_refresh(self):
        self._n_show = int(self.sp_n_show.value())
        self._n_reconstruct = int(self.sp_n_rec.value())
        self._cmap = self.cb_cmap.currentText()
        self._plot()
