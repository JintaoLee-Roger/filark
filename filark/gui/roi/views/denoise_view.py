# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""
DenoiseView — ROI denoising playground (preview only).

Right-panel layout (top → bottom):
  Algorithm selector   — always visible, flat
  ▶ Display            — collapsible, collapsed by default
  ▶ Metrics            — collapsible, expanded by default
  [stretch]
  Apply to Full Data   — always visible, pinned to bottom
  Status label

Algorithm params live in a floating ParamDialog (one per processor, lazy).
"""

from __future__ import annotations

from importlib.util import find_spec
from typing import Sequence

import numpy as np

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from filark.dsp import fk as fk_ops
from filark.gui.widgets import CollapsibleSection
from filark.gui.roi.views.param_dialog import ParamDialog
from filark.processing.processors import ProcessorPlugin, default_processors


def _hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setFrameShadow(QFrame.Sunken)
    return f


def _available_devices() -> list[tuple[str, str]]:
    """Return (label, device_key) pairs for the device combo box.

    Only two options: CPU and GPU (auto-selects CUDA or MPS).
    GPU option is hidden if PyTorch is not installed or no GPU is found.
    """
    items: list[tuple[str, str]] = [("CPU", "cpu")]
    if find_spec("torch") is not None:
        import torch

        has_gpu = torch.cuda.is_available() or (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        if has_gpu:
            items.append(("GPU", "auto"))
    return items


# ---------------------------------------------------------------------------
# DenoiseView
# ---------------------------------------------------------------------------

class DenoiseView(QWidget):
    """
    ROI denoising playground — preview only.

    Emits ``apply_requested(method_key, params, use_chunk)`` when the user
    clicks "Apply to Full Data".  ROIWindow handles file-save + progress.

    Parameters
    ----------
    roi_data : dict
        Keys: ``data`` (2-D ndarray, nt×nc), ``fs``, ``dx``.
    processors : list[ProcessorPlugin], optional
        Processor plugins.  Defaults to ``default_processors()``.
    """

    apply_requested = Signal(str, dict, str, str)   # method_key, params, chunk_mode, device

    def __init__(
        self,
        roi_data: dict,
        processors: Sequence[ProcessorPlugin] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.roi_data = roi_data

        self.raw = np.asarray(roi_data.get("data"), dtype=np.float32)
        if self.raw.ndim != 2:
            raise ValueError(f"ROI data must be 2D, got ndim={self.raw.ndim}")
        if self.raw.size == 0:
            raise ValueError("ROI data is empty.")

        self.fs = float(roi_data.get("fs") or 1.0)
        self.dx = float(roi_data.get("dx") or 1.0)

        self._processors: list[ProcessorPlugin] = list(processors or default_processors())
        if not self._processors:
            raise ValueError("At least one processor is required.")
        self._processor = self._processors[0]

        self._denoised = self.raw.copy()
        self._param_dialogs: dict[str, ParamDialog] = {}

        self._show_db = True
        self._normalize_fk = True
        self._img_cmap = "seismic"
        self._fk_cmap = "cividis"
        self._center_img_zero = True

        self._img_auto_clim = True
        self._img_clim: tuple[float, float] | None = None
        self._fk_auto_clim = True
        self._fk_clim: tuple[float, float] | None = None

        self._fk_extent_full: tuple | None = None
        self._fk_limits: tuple | None = None

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._run_preview)

        self._build_center()
        self._run_preview()

    # =========================================================
    # ROIWindow hook — builds the right-dock widget
    # =========================================================

    def create_controls(self, parent=None) -> QWidget:
        root = QWidget(parent)
        root.setMinimumWidth(230)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner = QWidget()
        vbox = QVBoxLayout(inner)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(2)

        vbox.addWidget(self._build_algo_widget())
        vbox.addWidget(_hline())
        vbox.addWidget(self._build_display_section())
        vbox.addWidget(_hline())
        vbox.addWidget(self._build_metrics_section())
        vbox.addStretch(1)

        scroll.setWidget(inner)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(scroll, 1)
        outer.addWidget(_hline())
        outer.addWidget(self._build_apply_widget())

        self.lbl_status = QLabel("Ready")
        self.lbl_status.setWordWrap(True)
        small = self.lbl_status.font()
        small.setPointSizeF(small.pointSizeF() * 0.85)
        self.lbl_status.setFont(small)
        self.lbl_status.setContentsMargins(8, 2, 8, 4)
        outer.addWidget(self.lbl_status)

        self._sync_fk_limit_spins()
        return root

    # =========================================================
    # Center plot
    # =========================================================

    def _build_center(self):
        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)

        gs = self.figure.add_gridspec(2, 2)
        self.ax_raw = self.figure.add_subplot(gs[0, 0])
        self.ax_den = self.figure.add_subplot(gs[0, 1])
        self.ax_fk_raw = self.figure.add_subplot(gs[1, 0])
        self.ax_fk_den = self.figure.add_subplot(gs[1, 1])

        nt, nc = self.raw.shape
        ext = [0.0, float(nt), 0.0, float(nc)]

        self.im_raw = self.ax_raw.imshow(
            self.raw.T, origin="lower", aspect="auto", cmap=self._img_cmap, extent=ext
        )
        self.im_den = self.ax_den.imshow(
            self._denoised.T, origin="lower", aspect="auto", cmap=self._img_cmap, extent=ext
        )

        fk0, f_ax, k_ax = self._compute_fk_display(self.raw)
        ext_fk = [float(k_ax.min()), float(k_ax.max()), float(f_ax.min()), float(f_ax.max())]
        self._fk_extent_full = tuple(ext_fk)
        self._fk_limits = self._fk_extent_full

        self.im_fk_raw = self.ax_fk_raw.imshow(
            fk0, origin="lower", aspect="auto", cmap=self._fk_cmap, extent=ext_fk
        )
        self.im_fk_den = self.ax_fk_den.imshow(
            fk0, origin="lower", aspect="auto", cmap=self._fk_cmap, extent=ext_fk
        )
        self.cbar_fk_raw = self.figure.colorbar(self.im_fk_raw, ax=self.ax_fk_raw, shrink=0.82)
        self.cbar_fk_den = self.figure.colorbar(self.im_fk_den, ax=self.ax_fk_den, shrink=0.82)

        for ax, title in zip(
            [self.ax_raw, self.ax_den, self.ax_fk_raw, self.ax_fk_den],
            ["Raw ROI", "Denoised ROI", "Raw F-K", "Denoised F-K"],
        ):
            ax.set_title(title)

        self.ax_raw.set_xlabel("Time index")
        self.ax_den.set_xlabel("Time index")
        self.ax_fk_raw.set_xlabel("k")
        self.ax_fk_den.set_xlabel("k")
        self.ax_raw.set_ylabel("Channel")
        self.ax_den.set_ylabel("Channel")
        self.ax_fk_raw.set_ylabel("f (Hz)")
        self.ax_fk_den.set_ylabel("f (Hz)")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    # =========================================================
    # Control widgets
    # =========================================================

    def _build_algo_widget(self) -> QWidget:
        """Algorithm selector — single-column, flat."""
        w = QWidget()
        vbox = QVBoxLayout(w)
        vbox.setContentsMargins(0, 6, 0, 6)
        vbox.setSpacing(5)

        # Algorithm combo — full width
        self.cb_method = QComboBox()
        for proc in self._processors:
            self.cb_method.addItem(proc.label, proc.method_key)
        self.cb_method.currentIndexChanged.connect(self._on_method_changed)
        vbox.addWidget(self.cb_method)

        # Configure button — full width, shown always (disabled when no params)
        self.btn_configure = QPushButton("Configure Parameters…")
        self.btn_configure.clicked.connect(self._on_configure_clicked)
        self._update_configure_btn()
        vbox.addWidget(self.btn_configure)

        # Device selector (shared by both preview and Apply to Full Data)
        self.cb_device = QComboBox()
        devices = _available_devices()
        for label, key in devices:
            self.cb_device.addItem(label, key)
        gpu_idx = next((i for i, (_, k) in enumerate(devices) if k != "cpu"), -1)
        if gpu_idx >= 0:
            self.cb_device.setCurrentIndex(gpu_idx)
        self.cb_device.setToolTip(
            "CPU: NumPy/SciPy, always available.\n"
            "GPU: auto-selects CUDA (NVIDIA) or MPS (Apple Silicon).\n"
            "GPU is greyed out for algorithms without a GPU implementation.\n"
            "Applies to both preview and Apply to Full Data."
        )
        vbox.addWidget(self.cb_device)
        self._update_device_combo()

        # Auto-preview checkbox
        self.ck_auto = QCheckBox("Auto-preview on change")
        self.ck_auto.setChecked(True)
        vbox.addWidget(self.ck_auto)

        # Preview + Reset side-by-side (only 2 symmetric buttons — acceptable)
        row_btns = QWidget()
        hb = QHBoxLayout(row_btns)
        hb.setContentsMargins(0, 0, 0, 0)
        hb.setSpacing(6)
        btn_preview = QPushButton("▶  Preview")
        btn_reset = QPushButton("↺  Reset")
        btn_preview.clicked.connect(self._run_preview)
        btn_reset.clicked.connect(self._reset_defaults)
        hb.addWidget(btn_preview, 1)
        hb.addWidget(btn_reset, 1)
        vbox.addWidget(row_btns)

        return w

    def _build_display_section(self) -> CollapsibleSection:
        """Display options — collapsible, collapsed by default."""
        sec = CollapsibleSection(
            "Display",
            collapsed=True,
            collapsed_marker="▶",
            expanded_marker="▼",
            body_margins=(4, 2, 4, 6),
            body_spacing=4,
        )

        # Checkboxes
        self.ck_show_db = QCheckBox("F-K in dB")
        self.ck_show_db.setChecked(True)
        self.ck_show_db.toggled.connect(self._on_display_changed)

        self.ck_norm_fk = QCheckBox("Normalize FK channels")
        self.ck_norm_fk.setChecked(True)
        self.ck_norm_fk.toggled.connect(self._on_display_changed)

        self.ck_center_zero = QCheckBox("Zero-center image")
        self.ck_center_zero.setChecked(True)
        self.ck_center_zero.toggled.connect(self._on_display_changed)

        for ck in (self.ck_show_db, self.ck_norm_fk, self.ck_center_zero):
            sec.add(ck)

        # Colormaps — single column via QFormLayout
        cmap_form_w = QWidget()
        cmap_form = QFormLayout(cmap_form_w)
        cmap_form.setVerticalSpacing(4)
        cmap_form.setContentsMargins(0, 4, 0, 0)
        cmap_form.setLabelAlignment(Qt.AlignLeft)
        self.cb_img_cmap = QComboBox()
        self.cb_img_cmap.addItems(["seismic", "RdBu_r", "gray", "viridis", "magma"])
        self.cb_img_cmap.setCurrentText("seismic")
        self.cb_img_cmap.currentTextChanged.connect(self._on_display_changed)
        self.cb_fk_cmap = QComboBox()
        self.cb_fk_cmap.addItems(["cividis", "viridis", "magma", "plasma", "gray"])
        self.cb_fk_cmap.setCurrentText("cividis")
        self.cb_fk_cmap.currentTextChanged.connect(self._on_display_changed)
        cmap_form.addRow("Image:", self.cb_img_cmap)
        cmap_form.addRow("FK:", self.cb_fk_cmap)
        sec.add(cmap_form_w)

        sec.add(_hline())
        sec.add(self._small_label("Color limits"))
        sec.add(self._build_clim_widget())

        sec.add(_hline())
        sec.add(self._small_label("F-K view range"))
        sec.add(self._build_fkrange_widget())

        return sec

    def _build_clim_widget(self) -> QWidget:
        """Single-column color-limit controls (no percentile controls)."""
        w = QWidget()
        form = QFormLayout(w)
        form.setVerticalSpacing(4)
        form.setContentsMargins(0, 0, 0, 0)
        form.setLabelAlignment(Qt.AlignLeft)

        self.sp_img_vmin = self._spin_float(-1e12, 1e12, -1.0, step=0.1)
        self.sp_img_vmax = self._spin_float(-1e12, 1e12,  1.0, step=0.1)
        form.addRow("Img min:", self.sp_img_vmin)
        form.addRow("Img max:", self.sp_img_vmax)
        form.addRow(self._btn_pair("Set", "Auto",
                                   self._apply_img_clim_from_ui, self._set_img_auto_clim))

        form.addRow(_hline())

        self.sp_fk_clim_vmin = self._spin_float(-1e12, 1e12, -1.0, step=0.1)
        self.sp_fk_clim_vmax = self._spin_float(-1e12, 1e12,  1.0, step=0.1)
        form.addRow("FK min:", self.sp_fk_clim_vmin)
        form.addRow("FK max:", self.sp_fk_clim_vmax)
        form.addRow(self._btn_pair("Set", "Auto",
                                   self._apply_fk_clim_from_ui, self._set_fk_auto_clim))

        return w

    def _build_fkrange_widget(self) -> QWidget:
        """Single-column F-K view range controls."""
        w = QWidget()
        form = QFormLayout(w)
        form.setVerticalSpacing(4)
        form.setContentsMargins(0, 0, 0, 0)
        form.setLabelAlignment(Qt.AlignLeft)

        self.sp_kmin = self._spin_float(-1e12, 1e12, -1.0, step=0.1)
        self.sp_kmax = self._spin_float(-1e12, 1e12,  1.0, step=0.1)
        self.sp_fmin = self._spin_float(-1e12, 1e12,  0.0, step=0.1)
        self.sp_fmax = self._spin_float(-1e12, 1e12,  1.0, step=0.1)
        form.addRow("k min:", self.sp_kmin)
        form.addRow("k max:", self.sp_kmax)
        form.addRow("f min:", self.sp_fmin)
        form.addRow("f max:", self.sp_fmax)
        form.addRow(self._btn_pair("Apply", "Reset",
                                   self._apply_fk_limits_from_ui, self._reset_fk_limits))

        return w

    def _build_metrics_section(self) -> CollapsibleSection:
        """Metrics — collapsible, expanded by default."""
        sec = CollapsibleSection(
            "Metrics",
            collapsed=False,
            collapsed_marker="▶",
            expanded_marker="▼",
            body_margins=(4, 2, 4, 6),
            body_spacing=4,
        )

        metrics_w = QWidget()
        form = QFormLayout(metrics_w)
        form.setVerticalSpacing(2)
        form.setContentsMargins(0, 0, 0, 0)
        self.lbl_raw_rms   = QLabel("—")
        self.lbl_den_rms   = QLabel("—")
        self.lbl_gain      = QLabel("—")
        self.lbl_residual  = QLabel("—")
        form.addRow("Raw RMS:",    self.lbl_raw_rms)
        form.addRow("Denoised:",   self.lbl_den_rms)
        form.addRow("Attenuation:", self.lbl_gain)
        form.addRow("Residual:",   self.lbl_residual)
        sec.add(metrics_w)

        return sec

    def _build_apply_widget(self) -> QWidget:
        """Apply-to-full-data controls, pinned at bottom."""
        w = QWidget()
        vbox = QVBoxLayout(w)
        vbox.setContentsMargins(8, 6, 8, 6)
        vbox.setSpacing(5)

        self.cb_chunk_mode = QComboBox()
        self.cb_chunk_mode.addItem("No chunking",            "none")
        self.cb_chunk_mode.addItem("Channel chunks",         "channel")
        self.cb_chunk_mode.addItem("Block chunks (time-first)", "block")
        self.cb_chunk_mode.setCurrentIndex(1)   # default: channel
        self.cb_chunk_mode.setToolTip(
            "No chunking: load full data into memory at once.\n"
            "Channel chunks: full time axis per chunk — recommended for FK filter.\n"
            "Block chunks: 2-D blocks, time as outer loop —\n"
            "  best for time-local ops (bandpass, SVD); FK loses frequency resolution."
        )
        vbox.addWidget(self.cb_chunk_mode)

        self.btn_apply_full = QPushButton("Apply to Full Data…")
        self.btn_apply_full.setMinimumHeight(32)
        self.btn_apply_full.clicked.connect(self._on_apply_full_clicked)
        vbox.addWidget(self.btn_apply_full)

        return w

    # =========================================================
    # Param dialog management
    # =========================================================

    def _get_param_dialog(self) -> ParamDialog:
        key = self._processor.method_key
        if key not in self._param_dialogs:
            dlg = ParamDialog(
                self._processor.param_spec,
                title=f"{self._processor.label} — Parameters",
                parent=self,
            )
            dlg.params_changed.connect(self._on_params_changed)
            self._param_dialogs[key] = dlg
        return self._param_dialogs[key]

    def _update_configure_btn(self):
        if hasattr(self, "btn_configure"):
            has_params = bool(self._processor.param_spec)
            self.btn_configure.setEnabled(has_params)
            self.btn_configure.setToolTip(
                "Open the parameter dialog for the selected algorithm."
                if has_params
                else "This algorithm has no configurable parameters."
            )

    def _update_device_combo(self):
        """Enable/disable GPU options based on whether the current processor supports GPU."""
        if not hasattr(self, "cb_device"):
            return
        supports = bool(getattr(self._processor, "supports_gpu", False))
        model = self.cb_device.model()
        for i in range(self.cb_device.count()):
            key = self.cb_device.itemData(i)
            item = model.item(i)
            if key != "cpu":
                item.setEnabled(supports)
        # If the selected device is now disabled, fall back to CPU
        cur_key = self.cb_device.currentData()
        if cur_key != "cpu" and not supports:
            self.cb_device.setCurrentIndex(0)

    # =========================================================
    # Control events
    # =========================================================

    def _on_method_changed(self):
        key = str(self.cb_method.currentData() or "")
        for proc in self._processors:
            if proc.method_key == key:
                self._processor = proc
                break
        self._update_configure_btn()
        self._update_device_combo()
        self._schedule_preview()

    def _on_configure_clicked(self):
        self._get_param_dialog().show_and_raise()

    def _on_params_changed(self):
        self._schedule_preview()

    def _on_display_changed(self):
        self._show_db = bool(self.ck_show_db.isChecked())
        self._normalize_fk = bool(self.ck_norm_fk.isChecked())
        self._img_cmap = str(self.cb_img_cmap.currentText() or "seismic")
        self._fk_cmap = str(self.cb_fk_cmap.currentText() or "cividis")
        self._center_img_zero = bool(self.ck_center_zero.isChecked())
        self._refresh_visuals()

    def _reset_defaults(self):
        dlg = self._get_param_dialog()
        dlg.set_params(self._processor.default_params())
        self._img_auto_clim = True
        self._fk_auto_clim = True
        self._run_preview()

    def _on_apply_full_clicked(self):
        params = self._current_params()
        chunk_mode = str(self.cb_chunk_mode.currentData()) if hasattr(self, "cb_chunk_mode") else "channel"
        device = str(self.cb_device.currentData()) if hasattr(self, "cb_device") else "cpu"
        self.apply_requested.emit(self._processor.method_key, params, chunk_mode, device)

    def _schedule_preview(self):
        if not hasattr(self, "ck_auto") or not self.ck_auto.isChecked():
            return
        self._preview_timer.start(180)

    # =========================================================
    # F-K view range
    # =========================================================

    def _apply_fk_limits_from_ui(self):
        kmin = float(self.sp_kmin.value())
        kmax = float(self.sp_kmax.value())
        fmin = float(self.sp_fmin.value())
        fmax = float(self.sp_fmax.value())
        self._apply_fk_limits(kmin, kmax, fmin, fmax)

    def _apply_fk_limits(self, kmin, kmax, fmin, fmax):
        if kmax < kmin: kmin, kmax = kmax, kmin
        if fmax < fmin: fmin, fmax = fmax, fmin
        self._fk_limits = (kmin, kmax, fmin, fmax)
        self.ax_fk_raw.set_xlim(kmin, kmax)
        self.ax_fk_raw.set_ylim(fmin, fmax)
        self.ax_fk_den.set_xlim(kmin, kmax)
        self.ax_fk_den.set_ylim(fmin, fmax)
        self.canvas.draw_idle()

    def _reset_fk_limits(self):
        if self._fk_extent_full is None:
            return
        self._fk_limits = self._fk_extent_full
        kmin, kmax, fmin, fmax = self._fk_extent_full
        self._set_fk_limit_values(kmin, kmax, fmin, fmax)
        self._apply_fk_limits(kmin, kmax, fmin, fmax)

    def _set_fk_limit_values(self, kmin, kmax, fmin, fmax):
        for sp in (self.sp_kmin, self.sp_kmax, self.sp_fmin, self.sp_fmax):
            sp.blockSignals(True)
        self.sp_kmin.setValue(float(kmin))
        self.sp_kmax.setValue(float(kmax))
        self.sp_fmin.setValue(float(fmin))
        self.sp_fmax.setValue(float(fmax))
        for sp in (self.sp_kmin, self.sp_kmax, self.sp_fmin, self.sp_fmax):
            sp.blockSignals(False)

    def _sync_fk_limit_spins(self):
        if self._fk_limits is not None and all(
            hasattr(self, n) for n in ("sp_kmin", "sp_kmax", "sp_fmin", "sp_fmax")
        ):
            self._set_fk_limit_values(*self._fk_limits)

    # =========================================================
    # Color controls
    # =========================================================

    def _apply_img_clim_from_ui(self):
        vmin, vmax = float(self.sp_img_vmin.value()), float(self.sp_img_vmax.value())
        if vmax < vmin: vmin, vmax = vmax, vmin
        self._img_auto_clim = False
        self._img_clim = (vmin, vmax)
        self._refresh_visuals()

    def _apply_fk_clim_from_ui(self):
        vmin, vmax = float(self.sp_fk_clim_vmin.value()), float(self.sp_fk_clim_vmax.value())
        if vmax < vmin: vmin, vmax = vmax, vmin
        self._fk_auto_clim = False
        self._fk_clim = (vmin, vmax)
        self._refresh_visuals()

    def _set_img_auto_clim(self):
        self._img_auto_clim = True
        self._refresh_visuals()

    def _set_fk_auto_clim(self):
        self._fk_auto_clim = True
        self._refresh_visuals()

    # =========================================================
    # Preview pipeline
    # =========================================================

    def _safe_fs_dx(self) -> tuple[float, float]:
        fs = float(self.fs) if np.isfinite(self.fs) and self.fs > 0 else 1.0
        dx = float(self.dx) if np.isfinite(self.dx) and self.dx > 0 else 1.0
        return fs, dx

    def _current_params(self) -> dict:
        fs, dx = self._safe_fs_dx()
        key = self._processor.method_key
        raw = self._param_dialogs[key].get_params() if key in self._param_dialogs else self._processor.default_params()
        return self._processor.validate_params(raw, fs=fs, dx=dx)

    def _run_preview(self):
        try:
            fs, dx = self._safe_fs_dx()
            params = self._current_params()
            device = self.cb_device.currentData() if hasattr(self, "cb_device") else "cpu"
            den = self._processor.run(self.raw, fs=fs, dx=dx, params=params, device=device)
            den = np.asarray(den, dtype=np.float32)
            if den.shape != self.raw.shape:
                raise ValueError(f"Shape mismatch: {den.shape} != {self.raw.shape}")
            self._denoised = den
            self._refresh_visuals()
            if hasattr(self, "lbl_status"):
                self.lbl_status.setText("Preview updated.")
        except Exception as e:
            if hasattr(self, "lbl_status"):
                self.lbl_status.setText(f"Preview failed: {type(e).__name__}: {e}")

    # Maximum pixels sent to matplotlib for image panels (time × channel)
    _MAX_DISPLAY_T = 1024
    _MAX_DISPLAY_C = 512

    def _refresh_visuals(self):
        nt, nc = self.raw.shape
        ext_img = [0.0, float(nt), 0.0, float(nc)]

        # Downsample for display to keep matplotlib responsive with large ROIs
        raw_disp, _, _ = self._downsample_2d(self.raw, max_t=self._MAX_DISPLAY_T, max_c=self._MAX_DISPLAY_C)
        den_disp, _, _ = self._downsample_2d(self._denoised, max_t=self._MAX_DISPLAY_T, max_c=self._MAX_DISPLAY_C)

        for im, data in [(self.im_raw, raw_disp), (self.im_den, den_disp)]:
            im.set_data(data.T)
            im.set_extent(ext_img)
            im.set_cmap(self._img_cmap)

        img_lo, img_hi = self._compute_img_clim()
        self.im_raw.set_clim(img_lo, img_hi)
        self.im_den.set_clim(img_lo, img_hi)
        if hasattr(self, "sp_img_vmin"):
            self._set_spin_pair(self.sp_img_vmin, self.sp_img_vmax, img_lo, img_hi)

        fk_raw, f_ax, k_ax = self._compute_fk_display(self.raw)
        fk_den, _, _ = self._compute_fk_display(self._denoised)
        ext_fk = [float(k_ax.min()), float(k_ax.max()), float(f_ax.min()), float(f_ax.max())]
        self._fk_extent_full = tuple(ext_fk)
        if self._fk_limits is None:
            self._fk_limits = self._fk_extent_full

        for im, fk in [(self.im_fk_raw, fk_raw), (self.im_fk_den, fk_den)]:
            im.set_data(fk)
            im.set_extent(ext_fk)
            im.set_cmap(self._fk_cmap)

        fk_lo, fk_hi = self._compute_fk_clim(fk_raw, fk_den)
        self.im_fk_raw.set_clim(fk_lo, fk_hi)
        self.im_fk_den.set_clim(fk_lo, fk_hi)
        if hasattr(self, "sp_fk_clim_vmin"):
            self._set_spin_pair(self.sp_fk_clim_vmin, self.sp_fk_clim_vmax, fk_lo, fk_hi)

        if self._fk_limits is not None:
            self._apply_fk_limits(*self._fk_limits)

        fk_label = "Power (dB)" if self._show_db else "Power"
        self.cbar_fk_raw.set_label(fk_label)
        self.cbar_fk_den.set_label(fk_label)

        self._sync_fk_limit_spins()
        self._update_metrics()
        self.canvas.draw_idle()

    # =========================================================
    # Helpers
    # =========================================================

    def _compute_fk_display(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(data, dtype=np.float32)
        x_ds, st, sc = self._downsample_2d(x, max_t=2048, max_c=1024)
        if self._normalize_fk:
            x_ds = self._normalize_by_channel(x_ds)
        dt_eff = (1.0 / max(self.fs, 1e-12)) * st
        dx_eff = max(self.dx * sc, 1e-12)
        spec, f_axis, k_axis = fk_ops.fk_transform(
            data=x_ds, dt=dt_eff, dx=dx_eff,
            backend="scipy", return_power=True, shift=True,
        )
        spec, f_axis, k_axis = np.asarray(spec), np.asarray(f_axis), np.asarray(k_axis)
        pos = f_axis >= 0
        spec, f_axis = spec[pos, :], f_axis[pos]
        if self._show_db:
            eps = 1e-12 * float(np.max(spec)) + 1e-12
            spec = 10.0 * np.log10(spec + eps)
        return spec.astype(np.float32, copy=False), f_axis, k_axis

    def _compute_img_clim(self) -> tuple[float, float]:
        if not self._img_auto_clim and self._img_clim is not None:
            return self._img_clim
        lo_p, hi_p = self._get_percentile_bounds()
        vals = np.concatenate([self.raw.ravel(), self._denoised.ravel()])
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            self._img_clim = (-1.0, 1.0)
            return self._img_clim
        if self._center_img_zero:
            vmax = float(np.percentile(np.abs(vals), hi_p))
            vmax = vmax if vmax > 1e-12 else 1.0
            self._img_clim = (-vmax, vmax)
        else:
            vmin = float(np.percentile(vals, lo_p))
            vmax = float(np.percentile(vals, hi_p))
            if vmax <= vmin: vmax = vmin + 1e-6
            self._img_clim = (vmin, vmax)
        return self._img_clim

    def _compute_fk_clim(self, fk_raw, fk_den) -> tuple[float, float]:
        if not self._fk_auto_clim and self._fk_clim is not None:
            return self._fk_clim
        lo_p, hi_p = self._get_percentile_bounds()
        vals = np.concatenate([fk_raw.ravel(), fk_den.ravel()])
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            self._fk_clim = (0.0, 1.0)
            return self._fk_clim
        vmin = float(np.percentile(vals, lo_p))
        vmax = float(np.percentile(vals, hi_p))
        if vmax <= vmin: vmax = vmin + 1e-6
        self._fk_clim = (vmin, vmax)
        return self._fk_clim

    def _get_percentile_bounds(self) -> tuple[float, float]:
        return 1.0, 99.0

    def _update_metrics(self):
        raw_rms = float(np.sqrt(np.mean(self.raw ** 2)))
        den_rms = float(np.sqrt(np.mean(self._denoised ** 2)))
        res_rms = float(np.sqrt(np.mean((self.raw - self._denoised) ** 2)))
        att_db = 20.0 * np.log10(raw_rms / den_rms) if den_rms > 1e-12 and raw_rms > 1e-12 else 0.0
        self.lbl_raw_rms.setText(f"{raw_rms:.4g}")
        self.lbl_den_rms.setText(f"{den_rms:.4g}")
        self.lbl_gain.setText(f"{att_db:+.2f} dB")
        self.lbl_residual.setText(f"{res_rms:.4g}")

    @staticmethod
    def _normalize_by_channel(x: np.ndarray) -> np.ndarray:
        med = np.median(x, axis=0, keepdims=True)
        mad = 1.4826 * np.median(np.abs(x - med), axis=0, keepdims=True) + 1e-12
        return (x - med) / mad

    @staticmethod
    def _downsample_2d(x: np.ndarray, *, max_t: int, max_c: int) -> tuple[np.ndarray, int, int]:
        nt, nc = x.shape
        st = max(1, int(np.ceil(nt / max_t)))
        sc = max(1, int(np.ceil(nc / max_c)))
        return x[::st, ::sc], st, sc

    @staticmethod
    def _set_spin_pair(sp0: QDoubleSpinBox, sp1: QDoubleSpinBox, v0: float, v1: float):
        sp0.blockSignals(True); sp1.blockSignals(True)
        sp0.setValue(float(v0)); sp1.setValue(float(v1))
        sp0.blockSignals(False); sp1.blockSignals(False)

    @staticmethod
    def _spin_float(
        vmin: float, vmax: float, value: float, *, step: float = 0.1
    ) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setDecimals(4)
        s.setRange(float(vmin), float(vmax))
        s.setSingleStep(float(step))
        s.setValue(float(value))
        s.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return s

    @staticmethod
    def _small_label(text: str) -> QLabel:
        lbl = QLabel(text)
        f = lbl.font()
        f.setPointSizeF(f.pointSizeF() * 0.88)
        lbl.setFont(f)
        return lbl

    @staticmethod
    def _btn_pair(label_a: str, label_b: str, slot_a, slot_b) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        btn_a = QPushButton(label_a)
        btn_b = QPushButton(label_b)
        btn_a.setFixedWidth(54)
        btn_b.setFixedWidth(54)
        btn_a.clicked.connect(slot_a)
        btn_b.clicked.connect(slot_b)
        h.addWidget(btn_a)
        h.addWidget(btn_b)
        h.addStretch(1)
        return row
