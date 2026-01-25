# filark/gui/modules/load.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union, Any

import numpy as np
import h5py

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog,
    QComboBox, QGroupBox, QFormLayout, QSpinBox, QCheckBox, QGridLayout, QMessageBox
)

from filark.gui.widgets.typography import PanelTitle
from filark.io.fileset import H5DASFileSet
from filark.io.h5io import H5Source
from filark.io.protocols import Tape



class LoadPanel(QWidget):
    """
    GUI: Load file/folder -> build source -> infer/layout -> stats -> signal out.
    """
    loaded = Signal(Tape)  # (source, vmin, vmax, layout_used, meta)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._current_source: Tape = None

        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # =====================================================
        # Panel title
        # =====================================================
        title = PanelTitle("Load Data")
        root.addWidget(title)

        # =====================================================
        # Source selector
        # =====================================================
        g_source = QGroupBox("Source")
        root.addWidget(g_source)

        source_layout = QVBoxLayout(g_source)
        source_layout.setSpacing(8)

        # Path row (edit + buttons aligned)
        path_row = QVBoxLayout()
        path_row.setSpacing(6)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText(".npy / .dat / .h5")
        path_row.addWidget(self.path_edit)

        btn_row = QHBoxLayout()
        self.btn_file = QPushButton("Browse File")
        self.btn_dir = QPushButton("Browse Folder")
        btn_row.addWidget(self.btn_file)
        btn_row.addWidget(self.btn_dir)
        btn_row.addStretch(1)

        path_row.addLayout(btn_row)
        source_layout.addLayout(path_row)

        # =====================================================
        # Data information (metadata / inferred config)
        # =====================================================
        g_info = QGroupBox("Data Information")
        root.addWidget(g_info)

        grid = QGridLayout(g_info)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(1, 1)

        row = 0

        # ---- Fixed (lock auto-detected info) ----
        grid.addWidget(QLabel("Fixed:"), row, 0)
        self.fixed_check = QCheckBox("Locked")
        self.fixed_check.setChecked(True)
        grid.addWidget(self.fixed_check, row, 1)
        row += 1

        # ---- Layout ----
        grid.addWidget(QLabel("Layout:"), row, 0)
        self.layout_combo = QComboBox()
        self.layout_combo.addItems([
            "nc_nt",
            "nt_nc",
        ])
        self.layout_combo.setEnabled(False)
        grid.addWidget(self.layout_combo, row, 1)
        row += 1

        # ---- dtype ----
        grid.addWidget(QLabel("dtype:"), row, 0)
        self.lbl_dtype = QLabel("—")
        grid.addWidget(self.lbl_dtype, row, 1)
        row += 1

        # ---- dx ----
        grid.addWidget(QLabel("dx:"), row, 0)
        self.lbl_dx = QLabel("—")
        grid.addWidget(self.lbl_dx, row, 1)
        row += 1

        # ---- fs ----
        grid.addWidget(QLabel("fs:"), row, 0)
        self.lbl_fs = QLabel("—")
        grid.addWidget(self.lbl_fs, row, 1)
        row += 1

        # ---- nc ----
        grid.addWidget(QLabel("nc:"), row, 0)
        self.lbl_nc = QLabel("—")
        grid.addWidget(self.lbl_nc, row, 1)
        row += 1

        # ---- nt (editable, -1 = streaming) ----
        grid.addWidget(QLabel("nt:"), row, 0)
        self.nt_spin = QSpinBox()
        self.nt_spin.setRange(-1, 2_000_000_000)
        self.nt_spin.setSpecialValueText("Streaming")
        self.nt_spin.setEnabled(False)
        grid.addWidget(self.nt_spin, row, 1)
        row += 1

        # =====================================================
        # Actions
        # =====================================================
        act_row = QHBoxLayout()
        self.btn_close = QPushButton("Close Source")
        act_row.addWidget(self.btn_close)
        act_row.addStretch(1)

        root.addLayout(act_row)
        root.addStretch(1)

        # =====================================================
        # Logic: fixed <-> editable
        # =====================================================
        self.fixed_check.toggled.connect(self._on_fixed_toggled)


        # =====================================================
        # Signals
        # =====================================================
        self.btn_file.clicked.connect(self._pick_file)
        self.btn_dir.clicked.connect(self._pick_dir)
        # self.btn_close.clicked.connect(self._on_close_clicked)


    def _on_fixed_toggled(self, fixed: bool):
        editable = not fixed
        self.layout_combo.setEnabled(editable)
        self.nt_spin.setEnabled(editable)

    # =====================================================
    # API: called after load / auto-detect
    # =====================================================
    def set_data_info(self):
        self.layout_combo.setCurrentText(self._current_source.dims)
        self.lbl_dtype.setText(str(self._current_source.dtype))
        self.lbl_dx.setText(f"{self._current_source.dx:.3f} {self._current_source.dx_unit}")
        self.lbl_fs.setText(f"{self._current_source.fs:.2f} Hz")
        if self._current_source.dims == 'nt_nc':
            self.lbl_nc.setText(str(self._current_source.shape[1]))
            self.nt_spin.setValue(self._current_source.shape[0])
        else:
            self.lbl_nc.setText(str(self._current_source.shape[0]))
            self.nt_spin.setValue(self._current_source.shape[1])

        self.loaded.emit(
            self._current_source
        )


    def _pick_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select data file", "", "Data Files (*.npy *.dat *.h5 *.hdf5);;All Files (*)"
        )
        if path:
            self.path_edit.setText(Path(path).name)
        
        self._load_file(path)

        self.set_data_info()

    def _pick_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select folder containing H5 files")
        if path:
            self.path_edit.setText(Path(path).name)

        self._load_folder(path)

        self.set_data_info()



    def _load_file(self, path: str):

        p = Path(path)
        if not p.exists():
            QMessageBox.critical(
                None,                  
                "File Error",             
                f"File not found:\n{path}",
                QMessageBox.Ok,
            )
            return None

        if p.suffix.lower() in [".h5", ".hdf5"]:
            data = H5Source(str(p))
        else:
            QMessageBox.critical(
                None,                  
                "File Error",             
                f"Unsupported file type:\n{path}",
                QMessageBox.Ok,
            )
            return None

        self._current_source = data


    def _load_folder(self, path: str):
        p = Path(path)
        if not p.exists() or not p.is_dir():
            QMessageBox.critical(
                None,                  
                "Folder Error",             
                f"Folder not found:\n{path}",
                QMessageBox.Ok,
            )
            return None

        data = H5DASFileSet(str(p))

        self._current_source = data


    def _close_source(self):
        # 尽量关掉句柄
        try:
            if hasattr(self._current_source, "close"):
                self._current_source.close()
        except Exception:
            pass
        self._current_source = None