"""ROI drawer panel: the region that the Analyze menu works on.

The panel only owns the selection itself (drag-to-select on the canvas and the
numeric bounds).  Everything that *uses* the ROI — the inspection views, the
processing window, analysis plugins — is started from the Analyze menu.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QFormLayout, QGroupBox, QLabel, QSpinBox, QVBoxLayout, QWidget

from filark.gui.widgets.typography import PanelTitle


@dataclass(frozen=True)
class ROI:
    """ROI in index space (time × channel)."""

    t0: int = 0
    t1: int = 0
    c0: int = 0
    c1: int = 0

    def normalized(self) -> "ROI":
        t0, t1 = (self.t0, self.t1) if self.t0 <= self.t1 else (self.t1, self.t0)
        c0, c1 = (self.c0, self.c1) if self.c0 <= self.c1 else (self.c1, self.c0)
        return ROI(t0, t1, c0, c1)

    def is_empty(self) -> bool:
        r = self.normalized()
        return r.t1 <= r.t0 or r.c1 <= r.c0


class ROIPanel(QWidget):
    """Drawer panel holding the shared ROI selection.

    Signals
    -------
    roi_mode_toggled(bool)
        User toggled "Drag to select".
    roi_changed(ROI)
        ROI bounds changed (from the spin boxes or from the canvas).
    """

    roi_mode_toggled = Signal(bool)
    roi_changed = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.roi = ROI(0, 0, 0, 0)
        self._build_ui()
        for sp in (self.sp_t0, self.sp_t1, self.sp_c0, self.sp_c1):
            sp.valueChanged.connect(self._on_fields_changed)
        self.ck_roi_mode.toggled.connect(self.roi_mode_toggled.emit)
        self._sync_fields()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)
        root.addWidget(PanelTitle("ROI"))

        self.g_roi = QGroupBox("Region (time x channel)")
        form = QFormLayout(self.g_roi)
        form.setVerticalSpacing(4)

        self.sp_t0 = QSpinBox()
        self.sp_t1 = QSpinBox()
        self.sp_c0 = QSpinBox()
        self.sp_c1 = QSpinBox()
        for sp in (self.sp_t0, self.sp_t1, self.sp_c0, self.sp_c1):
            sp.setRange(-2_000_000_000, 2_000_000_000)
        form.addRow("t0:", self.sp_t0)
        form.addRow("t1:", self.sp_t1)
        form.addRow("c0:", self.sp_c0)
        form.addRow("c1:", self.sp_c1)

        self.ck_roi_mode = QCheckBox("Drag to select")
        self.ck_roi_mode.setToolTip("Enable ROI drag selection on the canvas.")
        form.addRow("", self.ck_roi_mode)
        root.addWidget(self.g_roi)

        hint = QLabel("Tools in the Analyze menu work on this region.")
        hint.setWordWrap(True)
        hint.setEnabled(False)
        root.addWidget(hint)
        root.addStretch(1)

    def set_axis_metadata(self, axis_metadata) -> None:
        x_label = str(getattr(getattr(axis_metadata, "x", None), "label", "x"))
        y_label = str(getattr(getattr(axis_metadata, "y", None), "label", "y"))
        self.g_roi.setTitle(f"Region ({x_label} x {y_label})")

    # ----------------------------------------------------------- public API
    def current_roi(self) -> ROI:
        return ROI(
            int(self.sp_t0.value()),
            int(self.sp_t1.value()),
            int(self.sp_c0.value()),
            int(self.sp_c1.value()),
        ).normalized()

    def set_roi(self, t0: int, t1: int, c0: int, c1: int, *, commit: bool = True) -> None:
        """Called by the controller when the canvas ROI overlay changes."""
        self.roi = ROI(t0, t1, c0, c1).normalized()
        self._sync_fields()
        self.roi_changed.emit(self.roi)

    def set_roi_mode(self, enabled: bool) -> None:
        """Sync the checkbox; emits roi_mode_toggled when the state changes."""
        self.ck_roi_mode.setChecked(bool(enabled))

    def reset(self, *, clear_roi: bool = True, disable_roi_mode: bool = True) -> None:
        if clear_roi:
            self.roi = ROI(0, 0, 0, 0)
            self._sync_fields()
            self.roi_changed.emit(self.roi)
        if disable_roi_mode:
            self.ck_roi_mode.blockSignals(True)
            self.ck_roi_mode.setChecked(False)
            self.ck_roi_mode.blockSignals(False)
            self.roi_mode_toggled.emit(False)

    # ------------------------------------------------------------- internal
    def _on_fields_changed(self) -> None:
        self.roi = self.current_roi()
        self.roi_changed.emit(self.roi)

    def _sync_fields(self) -> None:
        r = self.roi
        spins = (self.sp_t0, self.sp_t1, self.sp_c0, self.sp_c1)
        for sp in spins:
            sp.blockSignals(True)
        self.sp_t0.setValue(r.t0)
        self.sp_t1.setValue(r.t1)
        self.sp_c0.setValue(r.c0)
        self.sp_c1.setValue(r.c1)
        for sp in spins:
            sp.blockSignals(False)
