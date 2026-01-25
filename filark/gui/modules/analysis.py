# Copyright (c) 2026 Jintao Li. 
# Zhejiang University (ZJU).
# 
# Licensed under the MIT License.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QCheckBox
)
from filark.gui.widgets.typography import PanelTitle


@dataclass(frozen=True)
class ROI:
    """ROI in index space (time x channel)."""
    t0: int = 0
    t1: int = 0
    c0: int = 0
    c1: int = 0

    def normalized(self) -> "ROI":
        t0, t1 = (self.t0, self.t1) if self.t0 <= self.t1 else (self.t1, self.t0)
        c0, c1 = (self.c0, self.c1) if self.c0 <= self.c1 else (self.c1, self.c0)
        return ROI(t0, t1, c0, c1)


class AnalysisPanel(QWidget):
    """
    Analysis panel (UI only).

    Scheme: single checkbox like ann_panel.

    - "ROI Mode" checkbox:
        ON  -> controller enables ROI overlay selection on canvas
        OFF -> controller disables ROI overlay selection (camera regains control)

    - ROI is committed automatically on mouse release (controller calls set_roi(..., commit=True)).
      Users may fine-tune ROI via spinboxes.

    - "Run" triggers analysis using current method + current ROI.
      Parameters are configured in an external Settings dialog (controller-owned).
    """

    roi_mode_toggled = Signal(bool)      # user toggles ROI mode
    roi_changed = Signal(object)         # ROI (live updates)
    roi_committed = Signal(object)       # ROI (end-of-drag or programmatic commit)

    method_changed = Signal(str)         # method id, e.g. "fk"
    settings_requested = Signal(str)     # method id

    run_requested = Signal(str, object)  # method id, ROI
    clear_results = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.roi: ROI = ROI(0, 0, 0, 0)

        self._build_ui()
        self._wire()
        self._sync_roi_fields()

    # ---------------- External API ----------------

    def set_roi(self, t0: int, t1: int, c0: int, c1: int, *, commit: bool = True):
        """Controller calls this when ROI is updated from canvas overlay."""
        self.roi = ROI(t0, t1, c0, c1).normalized()
        self._sync_roi_fields()
        self.roi_changed.emit(self.roi)
        if commit:
            self.roi_committed.emit(self.roi)

    def set_roi_mode(self, enabled: bool):
        """Controller can force-sync UI state (e.g., when switching tools)."""
        self.ck_roi_mode.blockSignals(True)
        self.ck_roi_mode.setChecked(bool(enabled))
        self.ck_roi_mode.blockSignals(False)

    # ---------------- UI ----------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        root.addWidget(PanelTitle("Analysis"))

        # -------- ROI --------
        g_roi = QGroupBox("ROI (time x channel)")
        f = QFormLayout(g_roi)

        self.sp_t0 = QSpinBox(); self.sp_t0.setRange(-2_000_000_000, 2_000_000_000)
        self.sp_t1 = QSpinBox(); self.sp_t1.setRange(-2_000_000_000, 2_000_000_000)
        self.sp_c0 = QSpinBox(); self.sp_c0.setRange(-2_000_000_000, 2_000_000_000)
        self.sp_c1 = QSpinBox(); self.sp_c1.setRange(-2_000_000_000, 2_000_000_000)

        f.addRow("t0:", self.sp_t0)
        f.addRow("t1:", self.sp_t1)
        f.addRow("c0:", self.sp_c0)
        f.addRow("c1:", self.sp_c1)

        self.ck_roi_mode = QCheckBox("Drag to select")
        self.ck_roi_mode.setToolTip("Enable ROI selection on canvas (drag to select).")
        f.addRow("", self.ck_roi_mode)

        root.addWidget(g_roi)

        # -------- Method --------
        g_type = QGroupBox("Method")
        f2 = QFormLayout(g_type)

        self.cb_type = QComboBox()
        self.cb_type.addItem("F-K spectrum", "fk")
        self.cb_type.addItem("Time-Frequency (STFT)", "stft")
        self.cb_type.addItem("Power spectrum (PSD)", "psd")
        self.cb_type.addItem("Band power / energy", "bandpower")
        self.cb_type.addItem("Coherence / semblance (simple)", "coherence")
        self.cb_type.addItem("Slowness / velocity scan (simple)", "slowness")

        self.btn_settings = QPushButton("Settings...")
        row_m = QHBoxLayout()
        row_m.addWidget(self.cb_type, 1)
        row_m.addWidget(self.btn_settings, 0)
        f2.addRow("Method:", row_m)

        root.addWidget(g_type)

        # -------- Actions --------
        g_act = QGroupBox("Run")
        h = QHBoxLayout(g_act)
        self.btn_run = QPushButton("Run")
        self.btn_clear = QPushButton("Clear")
        h.addWidget(self.btn_run)
        h.addWidget(self.btn_clear)
        h.addStretch(1)
        root.addWidget(g_act)

        self.lbl_hint = QLabel("Parameters are configured in Settings.")
        self.lbl_hint.setWordWrap(True)
        root.addWidget(self.lbl_hint)

        root.addStretch(1)

    def _wire(self):
        # ROI edits
        self.sp_t0.valueChanged.connect(self._on_roi_fields_changed)
        self.sp_t1.valueChanged.connect(self._on_roi_fields_changed)
        self.sp_c0.valueChanged.connect(self._on_roi_fields_changed)
        self.sp_c1.valueChanged.connect(self._on_roi_fields_changed)

        # ROI mode toggle
        self.ck_roi_mode.toggled.connect(self.roi_mode_toggled.emit)

        # method / settings
        self.cb_type.currentIndexChanged.connect(self._on_method_changed)
        self.btn_settings.clicked.connect(self._on_settings_clicked)

        # run/clear
        self.btn_run.clicked.connect(self._on_run_clicked)
        self.btn_clear.clicked.connect(self.clear_results.emit)

    # ---------------- Helpers ----------------

    def current_method(self) -> str:
        return str(self.cb_type.currentData() or "fk")

    def _sync_roi_fields(self):
        r = self.roi
        for sp in (self.sp_t0, self.sp_t1, self.sp_c0, self.sp_c1):
            sp.blockSignals(True)
        self.sp_t0.setValue(r.t0); self.sp_t1.setValue(r.t1)
        self.sp_c0.setValue(r.c0); self.sp_c1.setValue(r.c1)
        for sp in (self.sp_t0, self.sp_t1, self.sp_c0, self.sp_c1):
            sp.blockSignals(False)

    def _read_roi_fields(self) -> ROI:
        return ROI(
            int(self.sp_t0.value()),
            int(self.sp_t1.value()),
            int(self.sp_c0.value()),
            int(self.sp_c1.value()),
        ).normalized()

    def _on_roi_fields_changed(self):
        self.roi = self._read_roi_fields()
        self.roi_changed.emit(self.roi)

    def _on_method_changed(self):
        self.method_changed.emit(self.current_method())

    def _on_settings_clicked(self):
        self.settings_requested.emit(self.current_method())

    def _on_run_clicked(self):
        self.run_requested.emit(self.current_method(), self._read_roi_fields())
