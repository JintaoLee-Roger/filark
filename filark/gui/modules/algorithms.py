### TODO

# filark/gui/modules/algorithms.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout, QLabel,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QStackedWidget, QTextEdit
)

from filark.gui.widgets.typography import PanelTitle

@dataclass
class AlgoRunRequest:
    algo_id: str
    params: Dict[str, Any]


class AlgorithmsPanel(QWidget):
    """
    Placeholder algorithms panel.
    - No actual algorithm implementation.
    - Emits run/stop requests for future controller/pipeline integration.
    """

    run_requested = Signal(object)   # AlgoRunRequest
    stop_requested = Signal(str)     # algo_id
    algo_changed = Signal(str)       # algo_id

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._build_ui()
        self._wire()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = PanelTitle("Algorithms")
        root.addWidget(title)

        # ---- Select algorithm ----
        g_sel = QGroupBox("Select")
        f_sel = QFormLayout(g_sel)

        self.cb_algo = QComboBox()
        # Placeholder entries; you can add more later
        self.cb_algo.addItem("Energy detector (placeholder)", "energy")
        self.cb_algo.addItem("Coming soon...", "todo")

        f_sel.addRow("Algorithm:", self.cb_algo)
        root.addWidget(g_sel)

        # ---- Parameters area (stacked) ----
        g_params = QGroupBox("Parameters")
        v_params = QVBoxLayout(g_params)
        self.stack = QStackedWidget()
        v_params.addWidget(self.stack)
        root.addWidget(g_params)

        # Page 0: Energy detector placeholder params
        page_energy = QWidget()
        f_energy = QFormLayout(page_energy)

        self.sp_win = QSpinBox()
        self.sp_win.setRange(1, 10_000_000)
        self.sp_win.setValue(256)

        self.sp_stride = QSpinBox()
        self.sp_stride.setRange(1, 10_000_000)
        self.sp_stride.setValue(64)

        self.sp_thresh = QDoubleSpinBox()
        self.sp_thresh.setRange(0.0, 1e30)
        self.sp_thresh.setDecimals(6)
        self.sp_thresh.setValue(1.0)

        self.ck_norm = QCheckBox("Normalize energy (placeholder)")
        self.ck_norm.setChecked(True)

        f_energy.addRow("Window (samples):", self.sp_win)
        f_energy.addRow("Stride (samples):", self.sp_stride)
        f_energy.addRow("Threshold:", self.sp_thresh)
        f_energy.addRow("", self.ck_norm)

        note = QLabel("This is a placeholder panel. No computation is performed yet.")
        note.setWordWrap(True)
        # note.setStyleSheet("color: rgba(255,255,255,0.65);")
        f_energy.addRow("", note)

        self.stack.addWidget(page_energy)

        # Page 1: Coming soon
        page_todo = QWidget()
        v_todo = QVBoxLayout(page_todo)
        lbl = QLabel("More algorithms will appear here.\nUse the combo box to switch algorithms.")
        lbl.setWordWrap(True)
        # lbl.setStyleSheet("color: rgba(255,255,255,0.75);")
        v_todo.addWidget(lbl)
        v_todo.addStretch(1)
        self.stack.addWidget(page_todo)

        # ---- Actions ----
        g_act = QGroupBox("Run")
        h_act = QHBoxLayout(g_act)

        self.btn_run = QPushButton("Run")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(True)  # still placeholder

        h_act.addWidget(self.btn_run)
        h_act.addWidget(self.btn_stop)
        h_act.addStretch(1)
        root.addWidget(g_act)

        # ---- Status / Log ----
        g_log = QGroupBox("Status")
        v_log = QVBoxLayout(g_log)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFixedHeight(110)
        self.txt_log.setPlainText("Ready. (No algorithms executed in this placeholder UI.)")
        v_log.addWidget(self.txt_log)
        root.addWidget(g_log)

        root.addStretch(1)

    def _wire(self):
        self.cb_algo.currentIndexChanged.connect(self._on_algo_changed)
        self.btn_run.clicked.connect(self._on_run_clicked)
        self.btn_stop.clicked.connect(self._on_stop_clicked)

    def _on_algo_changed(self):
        algo_id = self.cb_algo.currentData()
        if algo_id == "energy":
            self.stack.setCurrentIndex(0)
        else:
            self.stack.setCurrentIndex(1)

        self.algo_changed.emit(str(algo_id))
        self._log(f"Selected algorithm: {algo_id}")

    def _collect_params(self, algo_id: str) -> Dict[str, Any]:
        # Placeholder: parameter collection for future use
        if algo_id == "energy":
            return {
                "window": int(self.sp_win.value()),
                "stride": int(self.sp_stride.value()),
                "threshold": float(self.sp_thresh.value()),
                "normalize": bool(self.ck_norm.isChecked()),
            }
        return {}

    def _on_run_clicked(self):
        algo_id = str(self.cb_algo.currentData())
        params = self._collect_params(algo_id)
        self._log(f"Run requested: {algo_id} with params={params}")
        self.run_requested.emit(AlgoRunRequest(algo_id=algo_id, params=params))

    def _on_stop_clicked(self):
        algo_id = str(self.cb_algo.currentData())
        self._log(f"Stop requested: {algo_id}")
        self.stop_requested.emit(algo_id)

    def _log(self, msg: str):
        self.txt_log.append(msg)
