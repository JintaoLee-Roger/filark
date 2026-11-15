# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""ROI window for local analysis, processing, and result export."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

from PySide6.QtCore import QObject, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QWidget,
)
from PySide6.QtCore import Qt

from filark.pipeline import run_processor_to_file
from filark.processing.processors import ProcessorPlugin, default_processors


# ---------------------------------------------------------------------------
# Background worker: process + save
# ---------------------------------------------------------------------------

class _ApplyWorker(QObject):
    """Runs full-data processing in a worker thread via the pipeline service."""

    finished = Signal(str)       # save path on success
    failed = Signal(str)         # error message on failure
    progress = Signal(int, int)  # done, total (channels or time samples)

    def __init__(
        self,
        source,
        processor: ProcessorPlugin,
        params: dict,
        save_path: str,
        chunk_mode: str = "channel",   # "none" | "channel" | "block"
        device: str = "cpu",
    ):
        super().__init__()
        self._source = source
        self._processor = processor
        self._params = params
        self._save_path = save_path
        self._chunk_mode = str(chunk_mode)
        self._device = str(device)

    def run(self) -> None:
        try:
            progress_cb = lambda done, total: self.progress.emit(int(done), int(total))
            cancel_cb = lambda: bool(QThread.currentThread().isInterruptionRequested())
            result = run_processor_to_file(
                self._source,
                self._processor,
                self._params,
                self._save_path,
                chunk_mode=self._chunk_mode,
                device=self._device,
                progress_cb=progress_cb,
                cancel_cb=cancel_cb,
            )
            self.finished.emit(result.save_path)

        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# ROI window
# ---------------------------------------------------------------------------

class ROIWindow(QMainWindow):
    """
    Floating window that holds a single ROI analysis view (central widget)
    and a dock containing the view's controls.

    Parameters
    ----------
    roi_data : dict
        Keys: ``data`` (2-D ndarray, nt×nc), ``fs``, ``dx``, ``roi``.
    view_type : str
        ``"fk"`` or ``"denoise"`` / ``"preprocess"``.
    source : Tape-like, optional
        Full data source.  When provided and ``view_type`` is ``"denoise"``,
        the "Apply to Full Data" button becomes functional.
    processors : list[ProcessorPlugin], optional
        Processor plugins for the denoise view.  Defaults to built-in list.
    """

    apply_finished = Signal(str)  # emitted with the save-file path on success

    def __init__(
        self,
        roi_data,
        view_type: str = "fk",
        source=None,
        processors: Sequence[ProcessorPlugin] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.roi_data = roi_data
        self.view_type = view_type
        self._source = source
        self._processors: list[ProcessorPlugin] = list(processors or default_processors())

        self._apply_thread: QThread | None = None
        self._apply_worker: _ApplyWorker | None = None
        self._current_chunk_mode: str = "channel"
        self._current_device: str = "cpu"
        self._t0: float = 0.0
        self._progress_text: str = ""

        # Elapsed-time ticker (fires every second while processing)
        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(1000)
        self._tick_timer.timeout.connect(self._tick)

        self._view = self._create_view(view_type)
        self.setCentralWidget(self._view)

        controls = None
        if hasattr(self._view, "create_controls"):
            controls = self._view.create_controls(parent=self)

        if controls is not None:
            dock = QDockWidget("Controls", self)
            dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
            dock.setFeatures(
                QDockWidget.DockWidgetMovable
                | QDockWidget.DockWidgetFloatable
                | QDockWidget.DockWidgetClosable
            )
            dock.setWidget(controls)
            self.addDockWidget(Qt.RightDockWidgetArea, dock)
            # Keep the controls panel compact; user can widen if needed
            self.resizeDocks([dock], [260], Qt.Horizontal)

        if hasattr(self._view, "apply_requested"):
            self._view.apply_requested.connect(self._handle_apply_requested)
        if self._source is None and hasattr(self._view, "btn_apply_full"):
            self._view.btn_apply_full.setEnabled(False)
            self._view.btn_apply_full.setToolTip("Apply to full data is only available for default time-channel sources.")

        self._build_status_bar()
        self.setWindowTitle(f"ROI View — {view_type}")

    def _build_status_bar(self) -> None:
        """Add a non-modal progress widget to the status bar."""
        self._sb_widget = QWidget()
        hb = QHBoxLayout(self._sb_widget)
        hb.setContentsMargins(4, 0, 4, 0)
        hb.setSpacing(8)
        self._lbl_sb = QLabel()
        self._btn_sb_cancel = QPushButton("Cancel")
        self._btn_sb_cancel.setFixedHeight(20)
        self._btn_sb_cancel.clicked.connect(self._cancel_apply)
        hb.addWidget(self._lbl_sb, 1)
        hb.addWidget(self._btn_sb_cancel)
        self.statusBar().addWidget(self._sb_widget, 1)
        self._sb_widget.hide()

    def _create_view(self, view_type: str):
        if view_type == "fk":
            from .views.fk_view import FKView
            return FKView(self.roi_data)
        if view_type in ("denoise", "preprocess"):
            from .views.denoise_view import DenoiseView
            return DenoiseView(self.roi_data, processors=self._processors)
        if view_type == "stft":
            from .views.stft_view import STFTView
            return STFTView(self.roi_data)
        if view_type == "psd":
            from .views.psd_view import PSDView
            return PSDView(self.roi_data)
        if view_type == "xcorr":
            from .views.xcorr_view import XcorrView
            return XcorrView(self.roi_data)
        if view_type == "tp":
            from .views.tp_view import TpView
            return TpView(self.roi_data)
        if view_type == "rms":
            from .views.rms_view import RMSView
            return RMSView(self.roi_data)
        if view_type == "svd":
            from .views.svd_view import SVDView
            return SVDView(self.roi_data)
        raise ValueError(view_type)

    # ------------------------------------------------------------------
    # Apply-to-full-data handler
    # ------------------------------------------------------------------

    def _handle_apply_requested(
        self,
        method_key: str,
        params: dict,
        chunk_mode: str = "channel",
        device: str = "cpu",
    ) -> None:
        if self._source is None:
            QMessageBox.warning(
                self,
                "Apply to Full Data",
                "No source data is available.\n"
                "This ROI window was opened without a reference to the full dataset.",
            )
            return

        if self._apply_thread is not None and self._apply_thread.isRunning():
            QMessageBox.information(
                self, "Apply to Full Data", "An apply operation is already in progress."
            )
            return

        processor = next(
            (p for p in self._processors if p.method_key == method_key), None
        )
        if processor is None:
            QMessageBox.critical(self, "Apply Failed", f"Unknown processor: {method_key!r}")
            return

        # --- Save dialog ---
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Processed Data", "", "NumPy Array (*.npy)",
        )
        if not save_path:
            return
        if not save_path.endswith(".npy"):
            save_path += ".npy"

        if hasattr(self._view, "btn_apply_full"):
            self._view.btn_apply_full.setEnabled(False)

        self._current_chunk_mode = str(chunk_mode)
        self._current_device = str(device)
        self._progress_text = ""
        self._t0 = time.monotonic()

        # Show non-modal status bar widget
        dev_label = f" [{device}]" if device != "cpu" else ""
        self._lbl_sb.setText(f"Applying {method_key}{dev_label}…  00:00")
        self._sb_widget.show()
        self._tick_timer.start()

        # --- Worker thread ---
        self._apply_worker = _ApplyWorker(
            self._source, processor, params, save_path,
            chunk_mode=str(chunk_mode), device=str(device),
        )
        self._apply_thread = QThread(self)
        self._apply_worker.moveToThread(self._apply_thread)

        self._apply_thread.started.connect(self._apply_worker.run)
        self._apply_worker.finished.connect(self._on_apply_finished)
        self._apply_worker.failed.connect(self._on_apply_failed)
        self._apply_worker.progress.connect(self._on_apply_progress)
        self._apply_worker.finished.connect(self._apply_thread.quit)
        self._apply_worker.failed.connect(self._apply_thread.quit)
        self._apply_thread.finished.connect(self._cleanup_apply_thread)

        self._apply_thread.start()

    def _cancel_apply(self) -> None:
        if self._apply_thread is not None:
            self._apply_thread.requestInterruption()

    def _tick(self) -> None:
        elapsed = int(time.monotonic() - self._t0)
        m, s = divmod(elapsed, 60)
        method = getattr(self._apply_worker, "_processor", None)
        name = method.method_key if method else "…"
        dev = self._current_device
        dev_label = f" [{dev}]" if dev != "cpu" else ""
        self._lbl_sb.setText(
            f"Applying {name}{dev_label}…  {m:02d}:{s:02d}  {self._progress_text}"
        )

    def _on_apply_finished(self, save_path: str) -> None:
        elapsed = time.monotonic() - self._t0
        self._tick_timer.stop()
        self._sb_widget.hide()

        if hasattr(self._view, "lbl_status"):
            m, s = divmod(int(elapsed), 60)
            self._view.lbl_status.setText(
                f"Done ({m:02d}:{s:02d}) — saved to: {save_path}"
            )

        meta_path = str(Path(save_path).with_suffix("")) + "_meta.json"
        m, s = divmod(int(elapsed), 60)
        msg = f"Processing complete  ({m:02d}:{s:02d}).\n\nSaved to:\n  {save_path}"
        if Path(meta_path).exists():
            msg += f"\n\nMetadata:\n  {meta_path}"
        QMessageBox.information(self, "Apply Complete", msg)

        self.apply_finished.emit(save_path)

    def _on_apply_failed(self, error_msg: str) -> None:
        self._tick_timer.stop()
        self._sb_widget.hide()
        QMessageBox.critical(self, "Apply Failed", f"Processing failed:\n\n{error_msg}")
        if hasattr(self._view, "lbl_status"):
            self._view.lbl_status.setText(f"Apply failed: {error_msg}")

    def _on_apply_progress(self, done: int, total: int) -> None:
        total = max(1, int(total))
        done = max(0, min(int(done), total))
        unit = "samples" if self._current_chunk_mode == "block" else "ch"
        self._progress_text = f"({done}/{total} {unit})"
        self._tick()   # refresh label immediately

    def _cleanup_apply_thread(self) -> None:
        if hasattr(self._view, "btn_apply_full"):
            self._view.btn_apply_full.setEnabled(True)
        self._apply_thread = None
        self._apply_worker = None
