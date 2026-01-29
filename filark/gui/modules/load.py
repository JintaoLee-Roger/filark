# filark.gui.modules.load.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Any, Literal, Tuple
import numpy as np

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog,
    QComboBox, QGroupBox, QGridLayout, QMessageBox, QCheckBox, QDoubleSpinBox,
    QDialog
)

from filark.gui.widgets.typography import PanelTitle
from filark.io.fileset import TapeFileSet, BinFolderOpts
from filark.io.tapeio import H5Tape, NpyTape, BinTape
from filark.io.protocols import Tape

from .dialogs import H5OptsDialog, H5Opts, NpyOpts, NpyOptsDialog, BinOpts, BinOptsDialog


Dims = Literal["nt_nc", "nc_nt"]


class LoadPanel(QWidget):
    """
    GUI: Load file/folder -> build Tape -> show info -> optional override -> emit Tape.
    """
    loaded = Signal(Tape)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._current_source: Optional[Tape] = None

        # ---- remember last inputs (enhancement #1) ----
        self._last_h5_opts: Optional[H5Opts] = None
        self._last_npy_opts: Optional[NpyOpts] = None
        self._last_bin_opts: Optional[BinOpts] = None

        self._build_ui()

    # ============================================================
    # UI
    # ============================================================
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        root.addWidget(PanelTitle("Load Data"))

        # -------------------------
        # Source
        # -------------------------
        g_source = QGroupBox("Source")
        root.addWidget(g_source)
        source_layout = QVBoxLayout(g_source)
        source_layout.setSpacing(8)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText(".npy / .dat / .bin / .h5 / folder")
        source_layout.addWidget(self.path_edit)

        btn_row = QHBoxLayout()
        self.btn_file = QPushButton("Browse File")
        self.btn_dir = QPushButton("Browse Folder")
        btn_row.addWidget(self.btn_file)
        btn_row.addWidget(self.btn_dir)
        btn_row.addStretch(1)
        source_layout.addLayout(btn_row)

        # -------------------------
        # Info
        # -------------------------
        g_info = QGroupBox("Data Information")
        root.addWidget(g_info)

        grid = QGridLayout(g_info)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(1, 1)

        row = 0

        grid.addWidget(QLabel("Fixed:"), row, 0)
        self.fixed_check = QCheckBox("Locked")
        self.fixed_check.setChecked(True)
        grid.addWidget(self.fixed_check, row, 1)
        row += 1

        grid.addWidget(QLabel("Layout (dims):"), row, 0)
        self.layout_combo = QComboBox()
        self.layout_combo.addItems(["nt_nc", "nc_nt"])
        grid.addWidget(self.layout_combo, row, 1)
        row += 1

        grid.addWidget(QLabel("fs (Hz):"), row, 0)
        self.fs_spin = QDoubleSpinBox()
        self.fs_spin.setRange(0.0, 1e12)
        self.fs_spin.setDecimals(6)
        self.fs_spin.setValue(1.0)
        grid.addWidget(self.fs_spin, row, 1)
        row += 1

        grid.addWidget(QLabel("dx:"), row, 0)
        self.dx_spin = QDoubleSpinBox()
        self.dx_spin.setRange(0.0, 1e12)
        self.dx_spin.setDecimals(6)
        self.dx_spin.setValue(1.0)
        grid.addWidget(self.dx_spin, row, 1)
        row += 1

        grid.addWidget(QLabel("dx_unit:"), row, 0)
        self.dxunit_edit = QLineEdit("m")
        grid.addWidget(self.dxunit_edit, row, 1)
        row += 1

        grid.addWidget(QLabel("dtype:"), row, 0)
        self.lbl_dtype = QLabel("—")
        grid.addWidget(self.lbl_dtype, row, 1)
        row += 1

        grid.addWidget(QLabel("nc:"), row, 0)
        self.lbl_nc = QLabel("—")
        grid.addWidget(self.lbl_nc, row, 1)
        row += 1

        grid.addWidget(QLabel("nt:"), row, 0)
        self.lbl_nt = QLabel("—")
        grid.addWidget(self.lbl_nt, row, 1)
        row += 1

        # -------------------------
        # Actions
        # -------------------------
        act_row = QHBoxLayout()
        self.btn_apply = QPushButton("Apply Overrides")
        self.btn_close = QPushButton("Close Source")
        act_row.addWidget(self.btn_apply)
        act_row.addWidget(self.btn_close)
        act_row.addStretch(1)
        root.addLayout(act_row)
        root.addStretch(1)

        # -------------------------
        # Signals
        # -------------------------
        self.btn_file.clicked.connect(self._pick_file)
        self.btn_dir.clicked.connect(self._pick_dir)
        self.btn_close.clicked.connect(self._close_source)
        self.btn_apply.clicked.connect(self._apply_overrides_and_emit)
        self.fixed_check.toggled.connect(self._on_fixed_toggled)

        self._on_fixed_toggled(self.fixed_check.isChecked())

    def _on_fixed_toggled(self, fixed: bool):
        editable = not fixed
        self.layout_combo.setEnabled(editable)
        self.fs_spin.setEnabled(editable)
        self.dx_spin.setEnabled(editable)
        self.dxunit_edit.setEnabled(editable)
        self.btn_apply.setEnabled(editable)

    # ============================================================
    # UI helpers
    # ============================================================
    def _sync_info_from_source(self):
        s = self._current_source
        if s is None:
            self.layout_combo.setCurrentText("nt_nc")
            self.fs_spin.setValue(1.0)
            self.dx_spin.setValue(1.0)
            self.dxunit_edit.setText("m")
            self.lbl_dtype.setText("—")
            self.lbl_nc.setText("—")
            self.lbl_nt.setText("—")
            return

        try:
            if getattr(s, "dims", None) in ("nt_nc", "nc_nt"):
                self.layout_combo.setCurrentText(s.dims)  # type: ignore
        except Exception:
            pass

        try:
            self.fs_spin.setValue(float(s.fs))
        except Exception:
            pass

        try:
            self.dx_spin.setValue(float(s.dx))
        except Exception:
            pass

        try:
            self.dxunit_edit.setText(str(s.dx_unit))
        except Exception:
            pass

        try:
            self.lbl_dtype.setText(str(s.dtype))
        except Exception:
            self.lbl_dtype.setText("—")

        try:
            if s.dims == "nt_nc":
                nt, nc = s.shape[0], s.shape[1]
            else:
                nc, nt = s.shape[0], s.shape[1]
            self.lbl_nc.setText(str(nc))
            self.lbl_nt.setText(str(nt))
        except Exception:
            self.lbl_nc.setText("—")
            self.lbl_nt.setText("—")

    def _apply_overrides_to_source(self):
        s = self._current_source
        if s is None:
            return

        dims = self.layout_combo.currentText()
        fs = float(self.fs_spin.value())
        dx = float(self.dx_spin.value())
        dx_unit = self.dxunit_edit.text().strip() or "m"

        for k, v in [("dims", dims), ("fs", fs), ("dx", dx), ("dx_unit", dx_unit)]:
            if hasattr(s, k):
                try:
                    setattr(s, k, v)
                except Exception:
                    pass

        for k, v in [("_dims", dims), ("_fs", fs), ("_dx", dx), ("_dx_unit", dx_unit)]:
            if hasattr(s, k):
                try:
                    setattr(s, k, v)
                except Exception:
                    pass

    def _apply_overrides_and_emit(self):
        if self._current_source is None:
            return
        self._apply_overrides_to_source()
        self._sync_info_from_source()
        self.loaded.emit(self._current_source)

    # ============================================================
    # Pick / Load
    # ============================================================
    def _pick_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select data file",
            "",
            "Data Files (*.npy *.dat *.bin *.h5 *.hdf5);;All Files (*)",
        )
        if not path:
            return

        self.path_edit.setText(path)
        ok = self._load_file(path)
        if ok:
            self._sync_info_from_source()
            self.loaded.emit(self._current_source)

    def _pick_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select folder containing data files")
        if not path:
            return

        self.path_edit.setText(path)
        ok = self._load_folder(path)
        if ok:
            self._sync_info_from_source()
            self.loaded.emit(self._current_source)


    def _open_with_retry(self, *, make_dialog, try_open, title: str) -> bool:
        """
        General: dialogs -> try to open -> fail then dialogs again until successful/canceled.
        make_dialog(error_text) -> QDialog
        try_open(opts) -> (ok: bool, err: str|None, source: Tape|None)
        """
        last_err = ""
        while True:
            dlg = make_dialog(last_err)
            if dlg.exec() != QDialog.Accepted:
                return False
            opts = dlg.result()
            if opts is None:
                return False

            ok, err, src = try_open(opts)
            if ok and src is not None:
                self._current_source = src
                return True

            last_err = f"{title} open failed.\n{err or 'Unknown error'}"
            QMessageBox.warning(self, title, last_err)


    def _open_h5_with_retry(self, path: str) -> bool:
        p = Path(path)

        try:
            self._current_source = H5Tape(str(p))
            return True
        except Exception as e:
            default_err = f"Default open failed.\n{type(e).__name__}: {e}"

        def make_dialog(err_text: str):
            shown = err_text or default_err
            return H5OptsDialog(str(p), parent=self, preset=self._last_h5_opts, error_text=shown)

        def try_open(opts):
            try:
                src = H5Tape(
                    path=str(p),
                    datakey=opts.datakey,
                    fs_key=opts.fs_key,
                    dx_key=opts.dx_key,
                    dims_key=opts.dims_key,
                    dxunit_key=opts.dxunit_key,
                )
                self._last_h5_opts = opts
                return True, None, src
            except Exception as e:
                return False, f"{type(e).__name__}: {e}", None

        self._current_source = None
        return self._open_with_retry(make_dialog=make_dialog, try_open=try_open, title="H5")


    def _open_npy_with_retry(self, path: str) -> bool:
        p = Path(path)

        try:
            arr = np.load(str(p), mmap_mode="r")
            if arr.ndim != 2:
                QMessageBox.critical(self, "NPY Error", f"Only 2D npy supported, got ndim={arr.ndim}")
                return False
        except Exception as e:
            QMessageBox.critical(self, "NPY Error", f"Failed to load npy:\n{type(e).__name__}: {e}")
            return False

        def make_dialog(err_text: str):
            preset = self._last_npy_opts or NpyOpts(dims="nt_nc", fs=1.0, dx=1.0, dx_unit="m")
            return NpyOptsDialog(self, preset=preset)

        def try_open(opts):
            try:
                src = NpyTape(
                    str(p),
                    dims=opts.dims,
                    fs=float(opts.fs),
                    dx=float(opts.dx),
                    dx_unit=str(opts.dx_unit),
                )
                self._last_npy_opts = opts
                return True, None, src
            except Exception as e:
                return False, f"{type(e).__name__}: {e}", None

        self._current_source = None
        return self._open_with_retry(make_dialog=make_dialog, try_open=try_open, title="NPY")


    def _open_bin_with_retry(self, path: str) -> bool:
        p = Path(path)

        def make_dialog(err_text: str):
            preset = self._last_bin_opts or BinOpts(
                nt=1000, nc=1000, dtype=np.dtype("float32"),
                dims="nt_nc", fs=1.0, dx=1.0, dx_unit="m"
            )
            return BinOptsDialog(self, preset=preset, file_path=str(p), error_text=err_text)

        def try_open(opts):
            try:
                src = BinTape(
                    path=str(p),
                    nt=int(opts.nt),
                    nc=int(opts.nc),
                    dtype=np.dtype(opts.dtype),
                    dims=opts.dims,
                    fs=float(opts.fs),
                    dx=float(opts.dx),
                    dx_unit=str(opts.dx_unit),
                )
                self._last_bin_opts = opts
                return True, None, src
            except Exception as e:
                return False, f"{type(e).__name__}: {e}", None

        self._current_source = None
        return self._open_with_retry(make_dialog=make_dialog, try_open=try_open, title="BIN/DAT")


    def _choose_folder_kind(self, kinds: Iterable[str]) -> Optional[str]:
        """
        kinds: e.g. {".h5", ".npy", ".dat"}
        return chosen kind or None if cancelled
        """
        kinds = sorted(set(kinds))
        if not kinds:
            return None
        if len(kinds) == 1:
            return kinds[0]

        mb = QMessageBox(self)
        mb.setIcon(QMessageBox.Question)
        mb.setWindowTitle("Choose file type")
        mb.setText("Mixed file types found in folder.\nChoose which type to load:")

        btn_map = {}
        for k in kinds:
            btn_map[k] = mb.addButton(k, QMessageBox.AcceptRole)

        mb.addButton("Cancel", QMessageBox.RejectRole)

        mb.exec()

        clicked = mb.clickedButton()
        for k, b in btn_map.items():
            if clicked is b:
                return k
        return None

    def _load_file(self, path: str) -> bool:
        p = Path(path)
        if not p.exists():
            QMessageBox.critical(self, "File Error", f"File not found:\n{path}")
            return False

        self._close_source()
        suf = p.suffix.lower()

        if suf in (".h5", ".hdf5"):
            return self._open_h5_with_retry(str(p))
        if suf == ".npy":
            return self._open_npy_with_retry(str(p))
        if suf in (".bin", ".dat"):
            return self._open_bin_with_retry(str(p))

        QMessageBox.critical(self, "File Error", f"Unsupported file type:\n{path}")
        return False


    def _load_folder(self, path: str) -> bool:
        p = Path(path)
        if not p.exists() or not p.is_dir():
            QMessageBox.critical(self, "Folder Error", f"Folder not found:\n{path}")
            return False

        self._close_source()

        suffixes = [".h5", ".hdf5", ".npy", ".dat", ".bin"]
        files = []
        for suf in suffixes:
            files.extend(sorted(p.glob(f"*{suf}")))
        files = [f for f in files if f.is_file() and not f.name.startswith(".")]

        if not files:
            QMessageBox.critical(self, "Folder Error", f"No supported files found in:\n{path}")
            return False

        kinds = {f.suffix.lower() for f in files}
        if ".h5" in kinds or ".hdf5" in kinds:
            kinds.discard(".h5"); kinds.discard(".hdf5"); kinds.add(".h5")

        # Mixed -> choose
        kind = self._choose_folder_kind(kinds)
        if kind is None:
            return False

        if kind == ".h5":
            chosen_files = [f for f in files if f.suffix.lower() in (".h5", ".hdf5")]
        else:
            chosen_files = [f for f in files if f.suffix.lower() == kind]
        if not chosen_files:
            QMessageBox.critical(self, "Folder Error", f"No files for chosen type: {kind}")
            return False

        if kind == ".h5":
            try:
                self._current_source = TapeFileSet(chosen_files)
                return True
            except Exception as e:
                default_err = f"Default open failed.\n{type(e).__name__}: {e}"

            # loop: select keys -> TapeFileSet(...h5_kwargs...) -> dialogs when failed
            last_err = ""
            while True:
                shown = last_err or default_err
                dlg = H5OptsDialog(str(chosen_files[0]), parent=self, preset=self._last_h5_opts, error_text=shown)
                if dlg.exec() != QDialog.Accepted:
                    self._current_source = None
                    return False
                opts = dlg.result()
                if opts is None:
                    self._current_source = None
                    return False

                h5_kwargs = dict(
                    datakey=opts.datakey,
                    fs_key=opts.fs_key,
                    dx_key=opts.dx_key,
                    dims_key=opts.dims_key,
                    dxunit_key=opts.dxunit_key,
                )
                try:
                    self._current_source = TapeFileSet(chosen_files, h5_kwargs=h5_kwargs)
                    self._last_h5_opts = opts
                    return True
                except Exception as e2:
                    last_err = f"H5 folder open failed.\n{type(e2).__name__}: {e2}"
                    QMessageBox.warning(self, "H5 Folder", last_err)

        if kind == ".npy":
            preset = self._last_npy_opts or NpyOpts(dims="nt_nc", fs=1.0, dx=1.0, dx_unit="m")
            dlg = NpyOptsDialog(self, preset=preset)
            if dlg.exec() != QDialog.Accepted:
                self._current_source = None
                return False
            opts = dlg.result()
            if opts is None:
                self._current_source = None
                return False
            try:
                self._current_source = TapeFileSet(
                    chosen_files,
                    npy_dims=opts.dims,
                    npy_fs=float(opts.fs),
                    npy_dx=float(opts.dx),
                    npy_dx_unit=str(opts.dx_unit),
                )
                self._last_npy_opts = opts
                return True
            except Exception as e:
                QMessageBox.critical(self, "NPY Folder", f"{type(e).__name__}: {e}")
                self._current_source = None
                return False

        if kind in (".dat", ".bin"):
            preset = self._last_bin_opts or BinOpts(
                nt=1000, nc=1000, dtype=np.dtype("float32"),
                dims="nt_nc", fs=1.0, dx=1.0, dx_unit="m"
            )
            dlg = BinOptsDialog(self, preset=preset, file_path=str(chosen_files[0]), error_text="")
            if dlg.exec() != QDialog.Accepted:
                self._current_source = None
                return False
            opts = dlg.result()
            if opts is None:
                self._current_source = None
                return False

            try:
                bin_opts = BinFolderOpts(
                    nt=int(opts.nt),
                    nc=int(opts.nc),
                    dtype=np.dtype(opts.dtype),
                    dims=str(opts.dims),
                    fs=float(opts.fs),
                    dx=float(opts.dx),
                    dx_unit=str(opts.dx_unit),
                    order="C",
                )
                self._current_source = TapeFileSet(chosen_files, bin_opts=bin_opts)
                self._last_bin_opts = opts
                return True
            except Exception as e:
                QMessageBox.critical(self, "BIN/DAT Folder", f"{type(e).__name__}: {e}")
                self._current_source = None
                return False

        QMessageBox.critical(self, "Folder Error", f"Unsupported folder content type: {kind}")
        self._current_source = None
        return False


    def set_current_source(self, source: Tape | None):
        self._current_source = source
        self._sync_info_from_source()

    # ============================================================
    # Close
    # ============================================================
    def _close_source(self):
        try:
            if self._current_source is not None and hasattr(self._current_source, "close"):
                self._current_source.close()
        except Exception:
            pass
        self._current_source = None
        self._sync_info_from_source()
