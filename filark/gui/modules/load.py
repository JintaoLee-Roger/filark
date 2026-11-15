"""Load panel for opening tapes, folders, and display-ready sources."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import h5py
import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog,
    QComboBox, QGroupBox, QGridLayout, QMessageBox, QCheckBox, QDoubleSpinBox,
    QDialog, QFormLayout, QSpinBox, QDialogButtonBox,
)

from filark.io.loaders import Loader
from filark.io.protocols import Tape
from filark.io.tapeio import BinTape, H5Tape, NpyTape
from filark.io.fileset import BinFolderOpts, TapeFileSet
from filark._exceptions import print_exception

from .dialogs import (
    H5OptsDialog,
    H5Opts,
    NpyOpts,
    NpyOptsDialog,
    BinOpts,
    BinOptsDialog,
    _optional_float_text,
    _parse_optional_float,
)


class LoadPanel(QWidget):
    """Source loading logic plus the "Source Properties" form.

    Loading is driven from the File menu (``pick_file`` / ``pick_dir`` /
    ``load_path``); the widget itself is shown inside the Source Properties
    dialog and only displays and overrides metadata of the current source.
    """
    loaded = Signal(Tape)
    compare_requested = Signal(str)  # emits a selected .npy path for comparison

    def __init__(self, parent: QWidget | None = None, *, loaders: Sequence[Loader] | None = None):
        super().__init__(parent)
        self._current_source: Optional[Tape] = None
        self._loaders = list(loaders or [])
        self._loader_map = self._index_loaders(self._loaders)
        self._gui_suffixes = [".h5", ".hdf5", ".npy", ".dat", ".bin"]
        self._load_guard = None
        self._dialog_parent: QWidget | None = None

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

        # -------------------------
        # Source
        # -------------------------
        g_source = QGroupBox("Source")
        root.addWidget(g_source)
        source_layout = QVBoxLayout(g_source)
        source_layout.setSpacing(8)

        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("No source loaded (File ▸ Open File… / Open Folder…)")
        source_layout.addWidget(self.path_edit)

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

        grid.addWidget(QLabel("gauge_length:"), row, 0)
        self.gauge_edit = QLineEdit("")
        self.gauge_edit.setPlaceholderText("optional")
        grid.addWidget(self.gauge_edit, row, 1)
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

        grid.addWidget(QLabel("start time:"), row, 0)
        self.lbl_start_time = QLabel("—")
        self.lbl_start_time.setWordWrap(True)
        grid.addWidget(self.lbl_start_time, row, 1)
        row += 1

        # -------------------------
        # Actions
        # -------------------------
        action_row = QHBoxLayout()
        action_row.addStretch(1)
        self.btn_apply = QPushButton("Apply Overrides")
        action_row.addWidget(self.btn_apply)
        root.addLayout(action_row)

        root.addStretch(1)

        # -------------------------
        # Signals
        # -------------------------
        self.btn_apply.clicked.connect(self._apply_overrides_and_emit)
        self.fixed_check.toggled.connect(self._on_fixed_toggled)

        self._on_fixed_toggled(self.fixed_check.isChecked())

    def set_dialog_parent(self, widget: QWidget | None) -> None:
        """Parent for file pickers / option dialogs (the main window)."""
        self._dialog_parent = widget

    def _dp(self) -> QWidget:
        return self._dialog_parent or self

    def _on_fixed_toggled(self, fixed: bool):
        editable = not fixed
        self.layout_combo.setEnabled(editable)
        self.fs_spin.setEnabled(editable)
        self.dx_spin.setEnabled(editable)
        self.dxunit_edit.setEnabled(editable)
        self.gauge_edit.setEnabled(editable)
        self.btn_apply.setEnabled(editable)

    @staticmethod
    def _normalize_suffix(suffix: str) -> str:
        suffix = suffix.lower()
        return ".h5" if suffix == ".hdf5" else suffix

    def _index_loaders(self, loaders: Sequence[Loader]) -> dict[str, list[Loader]]:
        out: dict[str, list[Loader]] = {}
        for ld in loaders:
            for suf in getattr(ld, "suffixes", ()):
                key = self._normalize_suffix(str(suf))
                if not key.startswith("."):
                    continue
                out.setdefault(key, []).append(ld)
        return out

    def _loaders_for(self, suffix: str) -> list[Loader]:
        return list(self._loader_map.get(self._normalize_suffix(suffix), ()))

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
            self.gauge_edit.clear()
            self.lbl_dtype.setText("—")
            self.lbl_nc.setText("—")
            self.lbl_nt.setText("—")
            self.lbl_start_time.setText("—")
            return

        try:
            if getattr(s, "dims", None) in ("nt_nc", "nc_nt"):
                self.layout_combo.setCurrentText(s.dims)  # type: ignore
        except Exception:
            print_exception("Load panel failed to sync source dims into the form.", prefix="LoadPanel")

        try:
            self.fs_spin.setValue(float(s.fs))
        except Exception:
            print_exception("Load panel failed to sync source fs into the form.", prefix="LoadPanel")

        try:
            self.dx_spin.setValue(float(s.dx))
        except Exception:
            print_exception("Load panel failed to sync source dx into the form.", prefix="LoadPanel")

        try:
            self.dxunit_edit.setText(str(s.dx_unit))
        except Exception:
            print_exception("Load panel failed to sync source dx_unit into the form.", prefix="LoadPanel")

        try:
            self.gauge_edit.setText(_optional_float_text(getattr(s, "gauge_length", None)))
        except Exception:
            print_exception("Load panel failed to sync source gauge_length into the form.", prefix="LoadPanel")

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

        try:
            st = getattr(s, "start_time", None)
            if st is not None:
                dt = datetime.datetime.fromtimestamp(float(st), tz=datetime.timezone.utc)
                ms = dt.microsecond // 1000
                self.lbl_start_time.setText(dt.strftime(f"%Y-%m-%d %H:%M:%S.{ms:03d} UTC"))
            else:
                self.lbl_start_time.setText("—")
        except Exception:
            self.lbl_start_time.setText("—")

    def _apply_overrides_to_source(self) -> bool:
        s = self._current_source
        if s is None:
            return False

        dims = self.layout_combo.currentText()
        fs = float(self.fs_spin.value())
        dx = float(self.dx_spin.value())
        dx_unit = self.dxunit_edit.text().strip() or "m"
        try:
            gauge_length = _parse_optional_float(self.gauge_edit.text(), "gauge_length")
        except ValueError as exc:
            QMessageBox.warning(self._dp(), "Override failed", str(exc))
            self._sync_info_from_source()
            return False

        overrides = [
            ("dims", dims),
            ("fs", fs),
            ("dx", dx),
            ("dx_unit", dx_unit),
            ("gauge_length", gauge_length),
        ]
        old_values = {}
        for k, _ in overrides:
            if hasattr(s, k):
                old_values[k] = getattr(s, k)

        try:
            for k, v in overrides:
                if hasattr(s, k):
                    setattr(s, k, v)
        except Exception as exc:
            for k, old in old_values.items():
                try:
                    setattr(s, k, old)
                except Exception:
                    print_exception(f"Load panel failed to restore source override {k}={old!r}.", prefix="LoadPanel")
            self._sync_info_from_source()
            QMessageBox.warning(
                self._dp(),
                "Override failed",
                f"Failed to apply {k}={v!r}.\n\n{type(exc).__name__}: {exc}",
            )
            print_exception(f"Load panel failed to apply override {k}={v!r} to source.", prefix="LoadPanel")
            return False

        return True

    def _apply_overrides_and_emit(self):
        if self._current_source is None:
            return
        if not self._apply_overrides_to_source():
            return
        self._sync_info_from_source()
        self.loaded.emit(self._current_source)

    # ============================================================
    # Stage 1: non-interactive loaders
    # ============================================================
    def _try_stage1(self, path: Path) -> Tuple[Optional[Tape], Optional[str]]:
        if not path.exists():
            return None, f"Path not found: {path}"
        if path.is_dir():
            return self._try_stage1_dir(path)
        return self._try_stage1_file(path)

    def _try_stage1_file(self, path: Path) -> Tuple[Optional[Tape], Optional[str]]:
        loaders = self._loaders_for(path.suffix)
        if not loaders:
            suf = self._normalize_suffix(path.suffix) or "(none)"
            return None, f"No loader for {suf}"

        last_err = None
        for ld in loaders:
            try:
                return ld.load(path), None
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"

        return None, last_err or "Load failed"

    def _dir_kinds(self, path: Path, suffixes: Iterable[str]) -> set[str]:
        sufset = {self._normalize_suffix(s) for s in suffixes}
        kinds: set[str] = set()
        for f in path.iterdir():
            if not f.is_file() or f.name.startswith("."):
                continue
            suf = self._normalize_suffix(f.suffix)
            if suf in sufset:
                kinds.add(suf)
        return kinds

    def _try_stage1_dir(self, path: Path) -> Tuple[Optional[Tape], Optional[str]]:
        kinds = self._dir_kinds(path, self._gui_suffixes)
        if not kinds:
            return None, "No supported files"
        if len(kinds) > 1:
            return None, "Mixed file types"

        kind = next(iter(kinds))
        files_of_kind = sorted(
            f for f in path.iterdir()
            if f.is_file() and (not f.name.startswith(".")) and self._normalize_suffix(f.suffix) == kind
        )
        if not files_of_kind:
            return None, f"No files for {kind}"

        loaders = self._loaders_for(kind)
        last_err = None
        for ld in loaders:
            try:
                return ld.load(path), None
            except Exception as e:
                dir_err = f"{type(e).__name__}: {e}"

            # Fallback: user loader only handles single file.
            # Framework stitches folder by applying loader file-by-file.
            opened: list[Tape] = []
            try:
                for fp in files_of_kind:
                    opened.append(ld.load(fp))
                return TapeFileSet(opened), None
            except Exception as e:
                for t in opened:
                    try:
                        t.close()
                    except Exception:
                        print_exception("Load panel failed to close a partially opened file during folder fallback loading.", prefix="LoadPanel")
                file_err = f"{type(e).__name__}: {e}"
                last_err = f"dir-load failed ({dir_err}); file-by-file failed ({file_err})"

        return None, last_err or "Load failed"

    # ============================================================
    # Pick / Load
    # ============================================================
    def pick_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self._dp(),
            "Select data file",
            "",
            "Data Files (*.npy *.dat *.bin *.h5 *.hdf5);;All Files (*)",
        )
        if not path:
            return

        self.load_path(path)

    def pick_dir(self):
        path = QFileDialog.getExistingDirectory(self._dp(), "Select folder containing data files")
        if not path:
            return

        self.load_path(path)


    def _open_with_retry(self, *, make_dialog, try_open, title: str, start_err: str = "") -> bool:
        """
        General: dialogs -> try to open -> fail then dialogs again until successful/canceled.
        make_dialog(error_text) -> QDialog
        try_open(opts) -> (ok: bool, err: str|None, source: Tape|None)
        """
        last_err = start_err or ""
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
            QMessageBox.warning(self._dp(), title, last_err)


    def _open_h5_with_retry(self, path: str, err_text: Optional[str] = None) -> bool:
        p = Path(path)

        def make_dialog(msg: str):
            return H5OptsDialog(str(p), parent=self._dp(), preset=self._last_h5_opts, error_text=msg)

        def try_open(opts):
            try:
                kwargs = {k: v for k, v in [
                    ("fs_key",    opts.fs_key),
                    ("dx_key",    opts.dx_key),
                    ("dims_key",  opts.dims_key),
                    ("dxunit_key", opts.dxunit_key),
                    ("gauge_length_key", opts.gauge_length_key),
                ] if v is not None}
                src = H5Tape(path=str(p), datakey=opts.datakey, **kwargs)
                self._last_h5_opts = opts
                return True, None, src
            except Exception as e:
                return False, f"{type(e).__name__}: {e}", None

        self._current_source = None
        return self._open_with_retry(
            make_dialog=make_dialog,
            try_open=try_open,
            title="H5",
            start_err=err_text or "",
        )


    def _open_npy_with_retry(self, path: str) -> bool:
        p = Path(path)

        try:
            arr = np.load(str(p), mmap_mode="r")
            if arr.ndim != 2:
                QMessageBox.critical(self._dp(), "NPY Error", f"Only 2D npy supported, got ndim={arr.ndim}")
                return False
        except Exception as e:
            QMessageBox.critical(self._dp(), "NPY Error", f"Failed to load npy:\n{type(e).__name__}: {e}")
            return False

        def make_dialog(err_text: str):
            preset = self._last_npy_opts or NpyOpts(dims="nt_nc", fs=1.0, dx=1.0, dx_unit="m")
            return NpyOptsDialog(self._dp(), preset=preset)

        def try_open(opts):
            try:
                src = NpyTape(
                    str(p),
                    dims=opts.dims,
                    fs=float(opts.fs),
                    dx=float(opts.dx),
                    dx_unit=str(opts.dx_unit),
                    gauge_length=opts.gauge_length,
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
            return BinOptsDialog(self._dp(), preset=preset, file_path=str(p), error_text=err_text)

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
                    gauge_length=opts.gauge_length,
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

        mb = QMessageBox(self._dp())
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

    def _load_file(self, path: str | Path) -> bool:
        p = Path(path)
        self._close_source()

        src, err_text = self._try_stage1(p)
        if src is not None:
            self._current_source = src
            self.path_edit.setText(str(p))
            return True

        return self._load_file_gui(p, err_text)

    def _load_file_gui(self, path: Path, err_text: Optional[str]) -> bool:
        if not path.exists():
            QMessageBox.critical(self._dp(), "File Error", f"File not found:\n{path}")
            return False

        suf = self._normalize_suffix(path.suffix)

        if suf == ".h5":
            return self._open_h5_with_retry(str(path), err_text=err_text)
        if suf == ".npy":
            return self._open_npy_with_retry(str(path))
        if suf in (".bin", ".dat"):
            return self._open_bin_with_retry(str(path))

        QMessageBox.critical(self._dp(), "File Error", f"Unsupported file type:\n{path}")
        return False

    def _load_folder(self, path: str | Path) -> bool:
        p = Path(path)
        self._close_source()

        src, err_text = self._try_stage1(p)
        if src is not None:
            self._current_source = src
            self.path_edit.setText(str(p))
            return True

        return self._load_folder_gui(p, err_text)

    def _load_folder_gui(self, path: Path, err_text: Optional[str]) -> bool:
        if not path.exists() or not path.is_dir():
            QMessageBox.critical(self._dp(), "Folder Error", f"Folder not found:\n{path}")
            return False

        files = []
        for suf in self._gui_suffixes:
            files.extend(sorted(path.glob(f"*{suf}")))
        files = [f for f in files if f.is_file() and not f.name.startswith(".")]

        if not files:
            QMessageBox.critical(self._dp(), "Folder Error", f"No supported files found in:\n{path}")
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
            QMessageBox.critical(self._dp(), "Folder Error", f"No files for chosen type: {kind}")
            return False

        if kind == ".h5":
            def make_dialog(msg: str):
                return H5OptsDialog(str(chosen_files[0]), parent=self._dp(), preset=self._last_h5_opts, error_text=msg)

            def try_open(opts):
                try:
                    h5_kwargs = {k: v for k, v in [
                        ("datakey",   opts.datakey),
                        ("fs_key",    opts.fs_key),
                        ("dx_key",    opts.dx_key),
                        ("dims_key",  opts.dims_key),
                        ("dxunit_key", opts.dxunit_key),
                        ("gauge_length_key", opts.gauge_length_key),
                    ] if v is not None}
                    src = TapeFileSet(chosen_files, h5_kwargs=h5_kwargs)
                    self._last_h5_opts = opts
                    return True, None, src
                except Exception as e:
                    return False, f"{type(e).__name__}: {e}", None

            self._current_source = None
            return self._open_with_retry(
                make_dialog=make_dialog,
                try_open=try_open,
                title="H5 Folder",
                start_err=err_text or "",
            )

        if kind == ".npy":
            preset = self._last_npy_opts or NpyOpts(dims="nt_nc", fs=1.0, dx=1.0, dx_unit="m")
            dlg = NpyOptsDialog(self._dp(), preset=preset)
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
                    npy_gauge_length=opts.gauge_length,
                )
                self._last_npy_opts = opts
                return True
            except Exception as e:
                QMessageBox.critical(self._dp(), "NPY Folder", f"{type(e).__name__}: {e}")
                self._current_source = None
                return False

        if kind in (".dat", ".bin"):
            preset = self._last_bin_opts or BinOpts(
                nt=1000, nc=1000, dtype=np.dtype("float32"),
                dims="nt_nc", fs=1.0, dx=1.0, dx_unit="m"
            )
            dlg = BinOptsDialog(self._dp(), preset=preset, file_path=str(chosen_files[0]), error_text="")
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
                    gauge_length=opts.gauge_length,
                    order="C",
                )
                self._current_source = TapeFileSet(chosen_files, bin_opts=bin_opts)
                self._last_bin_opts = opts
                return True
            except Exception as e:
                QMessageBox.critical(self._dp(), "BIN/DAT Folder", f"{type(e).__name__}: {e}")
                self._current_source = None
                return False

        QMessageBox.critical(self._dp(), "Folder Error", f"Unsupported folder content type: {kind}")
        self._current_source = None
        return False


    def export_subset(self):
        s = self._current_source
        if s is None:
            QMessageBox.warning(self._dp(), "Export", "No data loaded.")
            return

        dims = getattr(s, "dims", "nt_nc")
        if dims == "nt_nc":
            nt, nc = s.shape[0], s.shape[1]
        else:
            nc, nt = s.shape[0], s.shape[1]

        dlg = _ExportDialog(nt=nt, nc=nc, parent=self._dp())
        if dlg.exec() != QDialog.Accepted:
            return
        t0, t1, c0, c1, fmt = dlg.result()

        if fmt == "npy":
            path, _ = QFileDialog.getSaveFileName(self._dp(), "Export data", "export.npy", "NumPy array (*.npy)")
        else:
            path, _ = QFileDialog.getSaveFileName(self._dp(), "Export data", "export.h5", "HDF5 file (*.h5)")
        if not path:
            return

        try:
            if dims == "nt_nc":
                arr = np.asarray(s[t0:t1, c0:c1])
            else:
                arr = np.asarray(s[c0:c1, t0:t1])

            fs      = float(getattr(s, "fs",      1.0))
            dx      = float(getattr(s, "dx",      1.0))
            dx_unit = str(getattr(s,   "dx_unit", "m"))
            gauge_length = getattr(s, "gauge_length", None)

            if fmt == "npy":
                np.save(path, arr)
            else:
                with h5py.File(path, "w") as f:
                    ds = f.create_dataset("data", data=arr, compression="gzip")
                    ds.attrs["dims"]    = dims
                    ds.attrs["fs"]      = fs
                    ds.attrs["dx"]      = dx
                    ds.attrs["dx_unit"] = dx_unit
                    if gauge_length is not None:
                        ds.attrs["gauge_length"] = float(gauge_length)
                    ds.attrs["t0"]      = t0
                    ds.attrs["t1"]      = t1
                    ds.attrs["c0"]      = c0
                    ds.attrs["c1"]      = c1

            QMessageBox.information(self._dp(), "Export", f"Saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self._dp(), "Export failed", str(exc))

    def set_current_source(self, source: Tape | None):
        self._current_source = source
        self._sync_info_from_source()
        self.path_edit.setText(self._describe_source(source) if source is not None else "")

    @staticmethod
    def _describe_source(source) -> str:
        """Best-effort path text for sources that were not opened via the File menu."""
        def one(tape) -> str | None:
            path = getattr(tape, "path", None)
            if path is None:
                path = getattr(getattr(tape, "file", None), "filename", None)  # H5Tape
            return str(Path(str(path)).resolve()) if path else None

        own = one(source)
        if own:
            return own
        paths = [p for p in (one(f) for f in (getattr(source, "files", None) or [])) if p]
        if len(paths) > 1:
            return f"{Path(paths[0]).parent}  ({len(paths)} files)"
        if paths:
            return paths[0]
        return type(source).__name__

    @property
    def current_source(self) -> Tape | None:
        return self._current_source

    def set_load_guard(self, guard) -> None:
        self._load_guard = guard

    def _can_begin_load(self) -> bool:
        if not callable(self._load_guard):
            return True
        try:
            return bool(self._load_guard())
        except Exception:
            print_exception("Load panel load guard failed.", prefix="LoadPanel")
            return False

    def load_path(self, path: str | Path, *, emit_loaded: bool = True) -> bool:
        if not self._can_begin_load():
            return False
        p = Path(path)
        if p.is_file():
            ok = self._load_file(p)
        elif p.is_dir():
            ok = self._load_folder(p)
        else:
            return False
        if not ok:
            return False
        self.set_current_source(self._current_source)
        self.path_edit.setText(str(p))
        if emit_loaded and self._current_source is not None:
            self.loaded.emit(self._current_source)
        return True

    def request_compare(self):
        if self._current_source is None:
            QMessageBox.information(self._dp(), "Compare", "Load a primary source first.")
            return

        start_dir = ""
        text = self.path_edit.text().strip()
        if text:
            p = Path(text)
            if p.is_dir():
                start_dir = str(p)
            elif p.parent.exists():
                start_dir = str(p.parent)

        path, _ = QFileDialog.getOpenFileName(
            self._dp(),
            "Select Compare File",
            start_dir,
            "NumPy Array (*.npy);;All Files (*)",
        )
        if not path:
            return

        self.compare_requested.emit(path)

    # ============================================================
    # Close
    # ============================================================
    def reset_panel(self):
        """Reset UI state after source has been closed externally (no close() call)."""
        self._current_source = None
        self.path_edit.clear()
        self._sync_info_from_source()

    def _close_source(self):
        """Close and release the current source, then reset UI. Called internally before loading a new source."""
        try:
            if self._current_source is not None and hasattr(self._current_source, "close"):
                self._current_source.close()
        except Exception:
            print_exception("Load panel failed to close the current source.", prefix="LoadPanel")
        self.reset_panel()


# ============================================================
# Export dialog
# ============================================================

class _ExportDialog(QDialog):
    """Small dialog for selecting the export range and format."""

    def __init__(self, nt: int, nc: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export subset")
        self.setFixedWidth(280)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        form = QFormLayout()
        form.setVerticalSpacing(6)
        layout.addLayout(form)

        self.sp_t0 = QSpinBox(); self.sp_t0.setRange(0, nt); self.sp_t0.setValue(0)
        self.sp_t1 = QSpinBox(); self.sp_t1.setRange(0, nt); self.sp_t1.setValue(nt)
        self.sp_c0 = QSpinBox(); self.sp_c0.setRange(0, nc); self.sp_c0.setValue(0)
        self.sp_c1 = QSpinBox(); self.sp_c1.setRange(0, nc); self.sp_c1.setValue(nc)

        t_row = QHBoxLayout(); t_row.setSpacing(4)
        t_row.addWidget(self.sp_t0, 1); t_row.addWidget(QLabel("–")); t_row.addWidget(self.sp_t1, 1)
        form.addRow("t (samples):", t_row)

        c_row = QHBoxLayout(); c_row.setSpacing(4)
        c_row.addWidget(self.sp_c0, 1); c_row.addWidget(QLabel("–")); c_row.addWidget(self.sp_c1, 1)
        form.addRow("channels:", c_row)

        self.cb_fmt = QComboBox()
        self.cb_fmt.addItem("NumPy (.npy)", "npy")
        self.cb_fmt.addItem("HDF5 (.h5)",   "h5")
        form.addRow("Format:", self.cb_fmt)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        self._result = None

    def _on_accept(self):
        t0 = int(self.sp_t0.value())
        t1 = int(self.sp_t1.value())
        c0 = int(self.sp_c0.value())
        c1 = int(self.sp_c1.value())
        if t1 <= t0:
            QMessageBox.warning(self, "Export", "t end must be greater than t start.")
            return
        if c1 <= c0:
            QMessageBox.warning(self, "Export", "ch end must be greater than ch start.")
            return
        self._result = (t0, t1, c0, c1, self.cb_fmt.currentData() or "npy")
        self.accept()

    def result(self):
        return self._result


# ============================================================
# Source Properties dialog (File ▸ Source Properties…)
# ============================================================

class SourcePropertiesDialog(QDialog):
    """Non-modal window showing / overriding metadata of the current source."""

    def __init__(self, panel: LoadPanel, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Source Properties")
        self.setModal(False)
        self.panel = panel

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 8)
        layout.addWidget(panel)
        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.hide)
        btns.setContentsMargins(12, 0, 12, 0)
        layout.addWidget(btns)
        self.resize(420, self.sizeHint().height())

    def show_and_raise(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()
