"""Detect Events window (Analyze ▸ Detect Events…).

Runs a ``MonitorPlugin`` over the current view, the full source, or the
realtime display stream and overlays the detections on the main canvas.
The window keeps its state (detector, scope, save path, progress) between
openings; closing it only hides it, so a background run keeps reporting.
"""

from __future__ import annotations

import os
import time
from typing import Optional, Sequence

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from filark.gui.roi.views.param_dialog import ParamDialog


class DetectEventsWindow(QWidget):
    """Floating window that configures and runs event detectors.

    Signals
    -------
    monitor_run_requested(object, dict, dict)
        ``(MonitorPlugin instance, params dict, options dict)``
    monitor_cancel_requested()
    """

    monitor_run_requested = Signal(object, dict, dict)
    monitor_cancel_requested = Signal()

    _READY = "Ready. Select a scope and run the detector."

    def __init__(self, monitors: Sequence | None = None, parent: Optional[QWidget] = None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Detect Events")
        self._monitors: list = list(monitors) if monitors else []
        self._param_dialogs: dict[str, ParamDialog] = {}
        self._settings = QSettings()
        self._settings_key_save_dir = "monitor/save_dir"
        self._build_ui()
        self.resize(420, self.sizeHint().height())

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)

        if not self._monitors:
            lbl = QLabel(
                "No detectors configured.\n\n"
                "Implement a MonitorPlugin subclass and pass\n"
                "monitors=[...] to GuiConfig."
            )
            lbl.setWordWrap(True)
            lbl.setEnabled(False)
            v.addWidget(lbl)
            return

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)
        self.cb_monitor = QComboBox()
        for m in self._monitors:
            self.cb_monitor.addItem(m.label, m.key)
        form.addRow("Detector:", self.cb_monitor)

        self.cb_monitor_scope = QComboBox()
        self.cb_monitor_scope.addItem("Current view (manual)", "current_view")
        self.cb_monitor_scope.addItem("Full source (background)", "full_source")
        self.cb_monitor_scope.addItem("Realtime (from display, filtered)", "realtime_display")
        form.addRow("Scope:", self.cb_monitor_scope)
        v.addLayout(form)

        row = QHBoxLayout()
        self.btn_monitor_cfg = QPushButton("Parameters…")
        self.btn_monitor_run = QPushButton("Run")
        self.btn_monitor_run.setDefault(True)
        self.btn_monitor_cancel = QPushButton("Cancel")
        self.btn_monitor_cancel.setEnabled(False)
        row.addWidget(self.btn_monitor_cfg)
        row.addStretch(1)
        row.addWidget(self.btn_monitor_run)
        row.addWidget(self.btn_monitor_cancel)
        v.addLayout(row)

        save_group = QGroupBox("Save Results")
        save_layout = QVBoxLayout(save_group)
        self.ck_monitor_save = QCheckBox("Save detections as annotation JSON")
        self.edit_monitor_save_path = QLineEdit()
        self.edit_monitor_save_path.setPlaceholderText("annotations from detector...")
        self.btn_monitor_save_path = QPushButton("Browse…")
        save_row = QHBoxLayout()
        save_row.addWidget(self.edit_monitor_save_path, 1)
        save_row.addWidget(self.btn_monitor_save_path)
        self.lbl_monitor_save_status = QLabel("Save is off.")
        self.lbl_monitor_save_status.setWordWrap(True)
        self.lbl_monitor_save_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        save_layout.addWidget(self.ck_monitor_save)
        save_layout.addLayout(save_row)
        save_layout.addWidget(self.lbl_monitor_save_status)
        v.addWidget(save_group)

        self.lbl_monitor_status = QLabel(self._READY)
        self.lbl_monitor_status.setWordWrap(True)
        self.lbl_monitor_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(self.lbl_monitor_status)

        self.pb_monitor = QProgressBar()
        self.pb_monitor.setTextVisible(True)
        self.pb_monitor.setRange(0, 1)
        self.pb_monitor.setValue(0)
        self.pb_monitor.setFormat("Idle")
        v.addWidget(self.pb_monitor)

        self.btn_monitor_cfg.clicked.connect(self._on_configure)
        self.btn_monitor_run.clicked.connect(self._on_run)
        self.btn_monitor_cancel.clicked.connect(self.monitor_cancel_requested.emit)
        self.btn_monitor_save_path.clicked.connect(self._on_save_path)
        self.ck_monitor_save.toggled.connect(self._on_save_toggled)

    def show_and_raise(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

    # ------------------------------------------------------------- actions
    def _current_monitor(self):
        if not self._monitors or not hasattr(self, "cb_monitor"):
            return None
        idx = self.cb_monitor.currentIndex()
        if idx < 0 or idx >= len(self._monitors):
            return None
        return self._monitors[idx]

    def _param_dialog(self, plugin) -> ParamDialog:
        key = getattr(plugin, "key", id(plugin))
        if key not in self._param_dialogs:
            self._param_dialogs[key] = ParamDialog(
                param_spec=plugin.param_spec,
                title=f"{plugin.label} — Parameters",
                parent=self,
            )
        return self._param_dialogs[key]

    def _on_configure(self) -> None:
        plugin = self._current_monitor()
        if plugin is not None:
            self._param_dialog(plugin).show_and_raise()

    def _on_run(self) -> None:
        plugin = self._current_monitor()
        if plugin is None:
            return
        save_enabled = bool(self.ck_monitor_save.isChecked())
        save_path = ""
        if save_enabled:
            save_path = self._ensure_save_path(plugin)
            if not save_path:
                return
        dlg = self._param_dialog(plugin)
        params = dlg.get_params() if hasattr(dlg, "get_params") else plugin.default_params()
        options = {
            "scope": str(self.cb_monitor_scope.currentData() or "current_view"),
            "save_enabled": save_enabled,
            "save_path": save_path,
        }
        self.monitor_run_requested.emit(plugin, params, options)

    def _on_save_toggled(self, checked: bool) -> None:
        if checked:
            self.set_monitor_save_status("Save is on. Choose a path before running.")
        else:
            self.set_monitor_save_status("Save is off.")

    def _on_save_path(self) -> None:
        default_path = self._suggest_save_path(self._current_monitor())
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Detections",
            default_path,
            "Annotation JSON (*.json);;All Files (*)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        self.edit_monitor_save_path.setText(path)
        self._settings.setValue(self._settings_key_save_dir, os.path.dirname(path))
        if self.ck_monitor_save.isChecked():
            self.set_monitor_save_status(f"Will save to {path}")

    def _ensure_save_path(self, plugin) -> str:
        path = str(self.edit_monitor_save_path.text() or "").strip()
        if path:
            if not path.lower().endswith(".json"):
                path += ".json"
                self.edit_monitor_save_path.setText(path)
            self._settings.setValue(self._settings_key_save_dir, os.path.dirname(path))
            return path
        self._on_save_path()
        return str(self.edit_monitor_save_path.text() or "").strip()

    def _suggest_save_path(self, plugin) -> str:
        base_dir = str(self._settings.value(self._settings_key_save_dir, "") or "")
        if not base_dir or not os.path.isdir(base_dir):
            base_dir = os.path.expanduser("~")
        key = str(getattr(plugin, "key", "monitor") or "monitor")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(base_dir, f"{key}_annotations_{stamp}.json")

    # ------------------------------------------------- feedback (controller)
    def set_monitor_busy(self, busy: bool) -> None:
        for name in (
            "cb_monitor",
            "cb_monitor_scope",
            "btn_monitor_cfg",
            "btn_monitor_run",
            "ck_monitor_save",
            "edit_monitor_save_path",
            "btn_monitor_save_path",
        ):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(not busy)
        if hasattr(self, "btn_monitor_cancel"):
            self.btn_monitor_cancel.setEnabled(bool(busy))

    @staticmethod
    def _set_label(label: QLabel, text: str, error: bool) -> None:
        label.setText(str(text))
        label.setProperty("error", bool(error))
        label.style().unpolish(label)
        label.style().polish(label)

    def set_monitor_status(self, text: str, *, error: bool = False) -> None:
        if hasattr(self, "lbl_monitor_status"):
            self._set_label(self.lbl_monitor_status, text, error)

    def set_monitor_save_status(self, text: str, *, error: bool = False) -> None:
        if hasattr(self, "lbl_monitor_save_status"):
            self._set_label(self.lbl_monitor_save_status, text, error)

    def set_monitor_progress(self, done: int | None, total: int | None) -> None:
        if not hasattr(self, "pb_monitor"):
            return
        if done is None or total is None or int(total) <= 0:
            self.pb_monitor.setRange(0, 0)
            self.pb_monitor.setFormat("Running…")
            return
        total_i = max(1, int(total))
        done_i = max(0, min(int(done), total_i))
        self.pb_monitor.setRange(0, total_i)
        self.pb_monitor.setValue(done_i)
        self.pb_monitor.setFormat(f"{done_i}/{total_i} patches")

    def reset_monitor_feedback(self) -> None:
        self.set_monitor_busy(False)
        self.set_monitor_status(self._READY)
        if hasattr(self, "pb_monitor"):
            self.pb_monitor.setRange(0, 1)
            self.pb_monitor.setValue(0)
            self.pb_monitor.setFormat("Idle")
