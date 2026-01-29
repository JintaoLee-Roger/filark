# filark.gui.modules.dialogs.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal, Optional, Tuple, List

import h5py
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QMessageBox, QDoubleSpinBox, QTreeWidget, QTableWidget, QTreeWidgetItem, QTableWidgetItem,
    QDialog, QFormLayout, QDialogButtonBox, QSpinBox, QSplitter, QHeaderView, QApplication
)


Dims = Literal["nt_nc", "nc_nt"]


# ============================================================
# Dialogs
# ============================================================
@dataclass
class NpyOpts:
    dims: Dims
    fs: float
    dx: float
    dx_unit: str


class NpyOptsDialog(QDialog):
    def __init__(self, parent: QWidget | None = None, preset: Optional[NpyOpts] = None):
        super().__init__(parent)
        self.setWindowTitle("NPY Options")
        self._opts: Optional[NpyOpts] = None

        preset = preset or NpyOpts(dims="nt_nc", fs=1.0, dx=1.0, dx_unit="m")

        form = QFormLayout(self)

        self.cb_dims = QComboBox()
        self.cb_dims.addItems(["nt_nc", "nc_nt"])
        self.cb_dims.setCurrentText(preset.dims)
        form.addRow("layout (dims):", self.cb_dims)

        self.sp_fs = QDoubleSpinBox()
        self.sp_fs.setRange(0.0, 1e12)
        self.sp_fs.setDecimals(6)
        self.sp_fs.setValue(float(preset.fs))
        form.addRow("fs (Hz):", self.sp_fs)

        self.sp_dx = QDoubleSpinBox()
        self.sp_dx.setRange(0.0, 1e12)
        self.sp_dx.setDecimals(6)
        self.sp_dx.setValue(float(preset.dx))
        form.addRow("dx:", self.sp_dx)

        self.ed_unit = QLineEdit(preset.dx_unit)
        form.addRow("dx_unit:", self.ed_unit)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._on_ok)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def _on_ok(self):
        dims = self.cb_dims.currentText()
        if dims not in ("nt_nc", "nc_nt"):
            QMessageBox.critical(self, "Invalid dims", f"Invalid dims: {dims}")
            return
        self._opts = NpyOpts(
            dims=dims,  # type: ignore
            fs=float(self.sp_fs.value()),
            dx=float(self.sp_dx.value()),
            dx_unit=(self.ed_unit.text().strip() or "m"),
        )
        self.accept()

    def result(self) -> Optional[NpyOpts]:
        return self._opts



@dataclass
class BinOpts:
    nt: int
    nc: int
    dtype: np.dtype
    dims: Dims
    fs: float
    dx: float
    dx_unit: str



# -------------------------
# default name inference
# -------------------------
# Example:
# DasPhase-5-1000-5-97-9682-20250707_121308.dat
# -> gauge=5 (unused), fs=1000, dx=5, channel range 97..9682 => nc=9586
_DEFAULT_PATTERN = r".*?-(?P<gauge>\d+(?:\.\d+)?)-(?P<fs>\d+(?:\.\d+)?)-(?P<dx>\d+(?:\.\d+)?)-(?P<c0>\d+)-(?P<c1>\d+)-(?P<ts>\d{8}_\d{6})$"


def infer_bin_name_default(stem: str, pattern: str = _DEFAULT_PATTERN) -> dict:
    rx = re.compile(pattern)
    m = rx.match(stem)
    if not m:
        raise ValueError("pattern not matched")
    fs = float(m.group("fs"))
    dx = float(m.group("dx"))
    c0 = int(m.group("c0"))
    c1 = int(m.group("c1"))
    nc = abs(c1 - c0) + 1
    return {"fs": fs, "dx": dx, "nc": nc, "dx_unit": "m"}


class BinOptsDialog(QDialog):
    """
    BIN/DAT options dialog.

    - preset: last successful options, used as defaults
    - file_path: used for name inference button
    - error_text: shown on top to aid retry loop
    """
    def __init__(
        self,
        parent: QWidget | None = None,
        preset: Optional[BinOpts] = None,
        file_path: Optional[str] = None,
        error_text: str = "",
    ):
        super().__init__(parent)
        self.setWindowTitle("BIN/DAT Options")
        self._opts: Optional[BinOpts] = None
        self._file_path = file_path

        preset = preset or BinOpts(
            nt=1000,
            nc=1000,
            dtype=np.dtype("float32"),
            dims="nt_nc",
            fs=1.0,
            dx=1.0,
            dx_unit="m",
        )

        root = QVBoxLayout(self)

        if error_text:
            lab = QLabel(error_text)
            lab.setWordWrap(True)
            root.addWidget(lab)

        # file info + infer row
        info_row = QHBoxLayout()
        self.lbl_file = QLabel("File: —" if not file_path else f"File: {Path(file_path).name}")
        info_row.addWidget(self.lbl_file, 1)

        self.btn_infer = QPushButton("Infer from filename")
        self.btn_infer.setEnabled(bool(file_path))
        info_row.addWidget(self.btn_infer)
        root.addLayout(info_row)

        # regex pattern (optional)
        pat_row = QHBoxLayout()
        pat_row.addWidget(QLabel("Infer regex:"))
        self.ed_pattern = QLineEdit(_DEFAULT_PATTERN)
        self.ed_pattern.setPlaceholderText("Python regex with groups fs/dx/c0/c1 ...")
        pat_row.addWidget(self.ed_pattern, 1)
        root.addLayout(pat_row)

        # main form
        form = QFormLayout()
        root.addLayout(form)

        self.sp_nt = QSpinBox()
        self.sp_nt.setRange(1, 2_000_000_000)
        self.sp_nt.setValue(int(preset.nt))
        form.addRow("nt:", self.sp_nt)

        self.sp_nc = QSpinBox()
        self.sp_nc.setRange(1, 2_000_000_000)
        self.sp_nc.setValue(int(preset.nc))
        form.addRow("nc:", self.sp_nc)

        self.cb_dims = QComboBox()
        self.cb_dims.addItems(["nt_nc", "nc_nt"])
        self.cb_dims.setCurrentText(preset.dims)
        form.addRow("layout (dims):", self.cb_dims)

        self.ed_dtype = QLineEdit(str(preset.dtype))
        self.ed_dtype.setPlaceholderText("e.g. float32, int16, <f4, >i2 ...")
        form.addRow("dtype:", self.ed_dtype)

        self.sp_fs = QDoubleSpinBox()
        self.sp_fs.setRange(0.0, 1e12)
        self.sp_fs.setDecimals(6)
        self.sp_fs.setValue(float(preset.fs))
        form.addRow("fs (Hz):", self.sp_fs)

        self.sp_dx = QDoubleSpinBox()
        self.sp_dx.setRange(0.0, 1e12)
        self.sp_dx.setDecimals(6)
        self.sp_dx.setValue(float(preset.dx))
        form.addRow("dx:", self.sp_dx)

        self.ed_unit = QLineEdit(preset.dx_unit)
        form.addRow("dx_unit:", self.ed_unit)

        # ok/cancel
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        root.addWidget(bb)

        # signals
        bb.accepted.connect(self._on_ok)
        bb.rejected.connect(self.reject)
        self.btn_infer.clicked.connect(self._on_infer)

    def result(self) -> Optional[BinOpts]:
        return self._opts

    # -------------------------
    # infer handler
    # -------------------------
    def _on_infer(self):
        if not self._file_path:
            QMessageBox.information(self, "Infer", "No file path provided.")
            return
        stem = Path(self._file_path).stem
        pat = self.ed_pattern.text().strip() or _DEFAULT_PATTERN

        try:
            info = infer_bin_name_default(stem, pattern=pat)
        except Exception as e:
            QMessageBox.warning(self, "Infer", f"Failed to infer from name:\n{stem}\n\n{type(e).__name__}: {e}")
            return

        # only override "safe" fields
        try:
            self.sp_nc.setValue(int(info["nc"]))
        except Exception:
            pass
        try:
            self.sp_fs.setValue(float(info["fs"]))
        except Exception:
            pass
        try:
            self.sp_dx.setValue(float(info["dx"]))
        except Exception:
            pass
        try:
            self.ed_unit.setText(str(info.get("dx_unit", "m")))
        except Exception:
            pass

        QMessageBox.information(self, "Infer", f"Inferred from filename:\n{info}")

    # -------------------------
    # ok
    # -------------------------
    def _on_ok(self):
        dims = self.cb_dims.currentText()
        if dims not in ("nt_nc", "nc_nt"):
            QMessageBox.critical(self, "Invalid dims", f"Invalid dims: {dims}")
            return

        s = self.ed_dtype.text().strip()
        try:
            dt = np.dtype(s)
        except Exception as e:
            QMessageBox.critical(self, "Invalid dtype", f"Invalid dtype: {s}\n\n{e}")
            return

        self._opts = BinOpts(
            nt=int(self.sp_nt.value()),
            nc=int(self.sp_nc.value()),
            dtype=dt,
            dims=dims,  # type: ignore
            fs=float(self.sp_fs.value()),
            dx=float(self.sp_dx.value()),
            dx_unit=(self.ed_unit.text().strip() or "m"),
        )
        self.accept()



@dataclass
class H5Opts:
    datakey: str
    fs_key: Tuple[str, str]
    dx_key: Tuple[str, str]
    dims_key: Tuple[str, str]
    dxunit_key: Tuple[str, str]


def _to_str(v) -> str:
    if isinstance(v, (bytes, np.bytes_)):
        try:
            return v.decode("utf-8", errors="replace")
        except Exception:
            return str(v)
    if isinstance(v, np.ndarray):
        if v.size <= 32:
            return np.array2string(v, threshold=64)
        return f"ndarray(shape={v.shape}, dtype={v.dtype})"
    return str(v)


class H5OptsDialog(QDialog):
    """
    H5 browser + options editor.
    Supports `preset` (e.g. last successful config) and a reset button.
    """
    def __init__(self, filepath: str, parent: QWidget | None = None,
                 preset: Optional[H5Opts] = None, error_text: str = ""):
        super().__init__(parent)
        self.setWindowTitle("H5 Browser / Options")
        self.resize(980, 640)

        self.filepath = filepath
        self._opts: Optional[H5Opts] = None
        self._file: Optional[h5py.File] = None

        self._default_preset = H5Opts(
            datakey="Acquisition/Raw[0]/RawData",
            fs_key=("Acquisition/Raw[0]", "OutputDataRate"),
            dx_key=("Acquisition", "SpatialSamplingInterval"),
            dims_key=("Acquisition/Raw[0]/RawData", "Dimensions"),
            dxunit_key=("Acquisition", "SpatialSamplingIntervalUnit"),
        )
        self._preset = preset or self._default_preset  # store for "Reset to preset"

        root = QVBoxLayout(self)

        if error_text:
            err = QLabel(error_text)
            err.setWordWrap(True)
            root.addWidget(err)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        # -------------------- Left: tree --------------------
        left = QVBoxLayout()
        left_w = QWidget()
        left_w.setLayout(left)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Path", "Type"])
        left.addWidget(self.tree, 1)

        btns = QHBoxLayout()
        self.btn_guess = QPushButton("Auto Guess")
        self.btn_copy = QPushButton("Copy Path")
        btns.addWidget(self.btn_guess)
        btns.addWidget(self.btn_copy)
        btns.addStretch(1)
        left.addLayout(btns)

        splitter.addWidget(left_w)

        # -------------------- Right: attrs + option form --------------------
        right = QVBoxLayout()
        right_w = QWidget()
        right_w.setLayout(right)

        self.lbl_meta = QLabel("—")
        self.lbl_meta.setWordWrap(True)
        right.addWidget(self.lbl_meta)

        self.attr_table = QTableWidget(0, 2)
        self.attr_table.setHorizontalHeaderLabels(["Attr", "Value"])
        self.attr_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.attr_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        right.addWidget(self.attr_table, 1)

        form = QFormLayout()

        self.ed_datakey = QLineEdit()
        form.addRow("dataset key:", self.ed_datakey)

        self.ed_fs_group = QLineEdit()
        self.ed_fs_attr = QLineEdit()
        form.addRow("fs group:", self.ed_fs_group)
        form.addRow("fs attr:", self.ed_fs_attr)

        self.ed_dx_group = QLineEdit()
        self.ed_dx_attr = QLineEdit()
        form.addRow("dx group:", self.ed_dx_group)
        form.addRow("dx attr:", self.ed_dx_attr)

        self.ed_dims_group = QLineEdit()
        self.ed_dims_attr = QLineEdit()
        form.addRow("dims group:", self.ed_dims_group)
        form.addRow("dims attr:", self.ed_dims_attr)

        self.ed_u_group = QLineEdit()
        self.ed_u_attr = QLineEdit()
        form.addRow("dx_unit group:", self.ed_u_group)
        form.addRow("dx_unit attr:", self.ed_u_attr)

        right.addLayout(form)

        # Quick fill buttons
        fill_row = QHBoxLayout()
        self.btn_use_as_datakey = QPushButton("Use selected as datakey")
        self.btn_use_as_group = QPushButton("Use selected as attr group")
        self.btn_reset = QPushButton("Reset to preset")
        fill_row.addWidget(self.btn_use_as_datakey)
        fill_row.addWidget(self.btn_use_as_group)
        fill_row.addWidget(self.btn_reset)
        fill_row.addStretch(1)
        right.addLayout(fill_row)

        splitter.addWidget(right_w)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        # Bottom OK/Cancel
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._on_ok)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

        # Signals
        self.tree.currentItemChanged.connect(self._on_tree_changed)
        self.attr_table.itemDoubleClicked.connect(self._on_attr_double_clicked)

        self.btn_copy.clicked.connect(self._copy_selected_path)
        self.btn_guess.clicked.connect(self._auto_guess)
        self.btn_use_as_datakey.clicked.connect(self._use_selected_as_datakey)
        self.btn_use_as_group.clicked.connect(self._use_selected_as_group)
        self.btn_reset.clicked.connect(self._apply_preset_to_form)

        # init form values from preset
        self._apply_preset_to_form()

        # Load file + build tree
        self._open_and_build()

    def closeEvent(self, event):
        try:
            if self._file is not None:
                self._file.close()
        except Exception:
            pass
        super().closeEvent(event)

    def result(self) -> Optional[H5Opts]:
        return self._opts

    # -------------------- preset helpers --------------------
    def _apply_preset_to_form(self):
        pr = self._preset
        self.ed_datakey.setText(pr.datakey)
        self.ed_fs_group.setText(pr.fs_key[0]); self.ed_fs_attr.setText(pr.fs_key[1])
        self.ed_dx_group.setText(pr.dx_key[0]); self.ed_dx_attr.setText(pr.dx_key[1])
        self.ed_dims_group.setText(pr.dims_key[0]); self.ed_dims_attr.setText(pr.dims_key[1])
        self.ed_u_group.setText(pr.dxunit_key[0]); self.ed_u_attr.setText(pr.dxunit_key[1])

    # -------------------- internals --------------------
    def _open_and_build(self):
        try:
            self._file = h5py.File(self.filepath, "r")
        except Exception as e:
            QMessageBox.critical(self, "H5 Error", f"Failed to open:\n{self.filepath}\n\n{e}")
            self.reject()
            return

        self.tree.clear()
        root_item = QTreeWidgetItem(["/", "group"])
        root_item.setData(0, Qt.UserRole, "/")
        self.tree.addTopLevelItem(root_item)
        root_item.setExpanded(True)

        self._populate_children(root_item, self._file["/"])
        self.tree.setCurrentItem(root_item)

    def _populate_children(self, parent_item: QTreeWidgetItem, grp: h5py.Group, depth: int = 0, max_depth: int = 8):
        if depth > max_depth:
            return
        for k in grp.keys():
            obj = grp[k]
            path = obj.name
            typ = "dataset" if isinstance(obj, h5py.Dataset) else "group"
            item = QTreeWidgetItem([k, typ])
            item.setData(0, Qt.UserRole, path)
            parent_item.addChild(item)
            if isinstance(obj, h5py.Group):
                self._populate_children(item, obj, depth + 1, max_depth)

    def _selected_path(self) -> Optional[str]:
        it = self.tree.currentItem()
        if not it:
            return None
        return it.data(0, Qt.UserRole)

    def _on_tree_changed(self, cur: QTreeWidgetItem, prev: QTreeWidgetItem):
        path = self._selected_path()
        if not path or self._file is None:
            return
        obj = self._file[path]
        if isinstance(obj, h5py.Dataset):
            self.lbl_meta.setText(f"{path}\nDataset: shape={obj.shape}, dtype={obj.dtype}")
        else:
            self.lbl_meta.setText(f"{path}\nGroup")
        self._fill_attrs(obj)

    def _fill_attrs(self, obj):
        attrs = list(obj.attrs.items())
        self.attr_table.setRowCount(len(attrs))
        for r, (k, v) in enumerate(attrs):
            self.attr_table.setItem(r, 0, QTableWidgetItem(str(k)))
            self.attr_table.setItem(r, 1, QTableWidgetItem(_to_str(v)))

    # -------------------- UX: double-click attr name to fill the focused attr box --------------------
    def _focused_attr_lineedit(self) -> Optional[QLineEdit]:
        w = self.focusWidget()
        if isinstance(w, QLineEdit):
            # only allow filling into attr/group fields, not datakey
            allowed = {
                self.ed_fs_group, self.ed_fs_attr,
                self.ed_dx_group, self.ed_dx_attr,
                self.ed_dims_group, self.ed_dims_attr,
                self.ed_u_group, self.ed_u_attr,
            }
            if w in allowed:
                return w
        return None

    def _on_attr_double_clicked(self, item: QTableWidgetItem):
        if item.column() != 0:
            return
        attr = item.text().strip()
        if not attr:
            return

        target = self._focused_attr_lineedit()
        if target is None:
            # fallback: if no focus, fill dims_attr
            target = self.ed_dims_attr
        target.setText(attr)

    def _copy_selected_path(self):
        path = self._selected_path()
        if not path:
            return
        QApplication.clipboard().setText(path)

    def _use_selected_as_datakey(self):
        path = self._selected_path()
        if not path or self._file is None:
            return
        obj = self._file[path]
        if not isinstance(obj, h5py.Dataset):
            QMessageBox.information(self, "Info", "Selected node is not a dataset.")
            return
        self.ed_datakey.setText(path)

    def _use_selected_as_group(self):
        path = self._selected_path()
        if not path:
            return
        self.ed_fs_group.setText(path)
        self.ed_dx_group.setText(path)
        self.ed_dims_group.setText(path)
        self.ed_u_group.setText(path)

    def _auto_guess(self):
        if self._file is None:
            return

        candidates: List[h5py.Dataset] = []

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim == 2:
                candidates.append(obj)

        self._file.visititems(visit)
        if not candidates:
            QMessageBox.information(self, "Guess", "No 2D dataset found.")
            return

        def score(ds: h5py.Dataset) -> int:
            n = ds.name.lower()
            s = 0
            if "rawdata" in n: s += 5
            if "/raw" in n: s += 2
            if "data" in n: s += 1
            return s

        candidates.sort(key=score, reverse=True)
        best = candidates[0]
        self.ed_datakey.setText(best.name)
        self.ed_dims_group.setText(best.name)

    def _on_ok(self):
        datakey = self.ed_datakey.text().strip()
        if not datakey:
            QMessageBox.critical(self, "Invalid", "dataset key cannot be empty")
            return

        self._opts = H5Opts(
            datakey=datakey,
            fs_key=(self.ed_fs_group.text().strip(), self.ed_fs_attr.text().strip()),
            dx_key=(self.ed_dx_group.text().strip(), self.ed_dx_attr.text().strip()),
            dims_key=(self.ed_dims_group.text().strip(), self.ed_dims_attr.text().strip()),
            dxunit_key=(self.ed_u_group.text().strip(), self.ed_u_attr.text().strip()),
        )
        self.accept()
