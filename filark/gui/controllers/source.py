"""Source lifecycle controller for the main GUI window."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QDialog, QMessageBox

from filark._exceptions import print_exception
from filark.io.protocols import Tape
from filark.io.tapeio import NpyTape

from ..widgets.goto_dialog import GoToDialog


def _print_exception(context: str) -> None:
    print_exception(context, prefix="SourceController")


def estimate_initial_clim(source, *, cap=256, k=5.0):
    """Estimate a stable initial color range from a small source sample."""
    a, b = source.shape[:2]
    if a == 0 or b == 0:
        return 0.0, 1.0

    s = min(cap, a, b)
    vals = np.asarray(source[:s, :s]).ravel()

    if vals.size == 0:
        return 0.0, 1.0

    ma = float(vals.mean())
    st = float(vals.std()) or 1.0
    return ma - k * st, ma + k * st


class SourceControllerMixin:
    """Own source loading, clearing, display initialization, and go-to navigation."""

    def _init_source_state(self, initial_source) -> None:
        self._source = None
        self._pending_initial_source = initial_source
        self._initial_source_load_scheduled = False

    def _schedule_initial_source_load(self) -> None:
        if self._pending_initial_source is None or self._initial_source_load_scheduled:
            return
        if not self.isVisible():
            return
        self._initial_source_load_scheduled = True
        QTimer.singleShot(0, self._load_pending_initial_source)

    def _load_pending_initial_source(self) -> None:
        source = self._pending_initial_source
        self._pending_initial_source = None
        self._initial_source_load_scheduled = False
        if source is None:
            return
        if self.set_source(source):
            QTimer.singleShot(0, self.canvas.initialize_if_ready)
            QTimer.singleShot(0, self.canvas.native.setFocus)

    def _apply_stream_processing_for_source(self, source: Tape) -> None:
        fs = float(getattr(source, "fs", 1.0) or 1.0)
        dx = float(getattr(source, "dx", 1.0) or 1.0)
        preprocess = self.cfg.build_stream_preprocess(fs=fs, dx=dx)
        postprocess = self.cfg.build_stream_postprocess(fs=fs, dx=dx)
        self.disp_model.set_stream_processing(preprocess, postprocess)

    def _source_switch_allowed(self, operation: str = "switching the source") -> bool:
        if not self._monitor_is_running():
            return True
        QMessageBox.information(
            self,
            "Monitor Running",
            f"A monitor run is still in progress.\nCancel it or wait for it to finish before {operation}.",
        )
        return False

    def set_source(self, source) -> bool:
        if not self._source_switch_allowed("switching the source"):
            return False

        if isinstance(source, (str, os.PathLike)):
            path = Path(str(source))
            if self._source is not None and not self._on_clear():
                return False
            return self.load_source_path(path)

        old = getattr(self, "_source", None)
        if old is not None and old is not source:
            try:
                old.close()
            except Exception:
                _print_exception("Failed to close previous source during source switch.")

        if isinstance(source, (np.ndarray, np.memmap)):
            source = NpyTape(
                source,
                dims="nt_nc",
                fs=1,
                dx=1,
                dx_unit="m",
            )
        elif isinstance(source, Tape):
            pass
        else:
            raise ValueError(f"unknown source: {source}")

        self._on_loaded(source)
        return True

    def load_source_path(self, path: Path) -> bool:
        return bool(self.load_panel.load_path(path))

    def _on_loaded(self, source):
        if not self._source_switch_allowed("switching the source"):
            if source is not None and source is not self._source:
                try:
                    source.close()
                except Exception:
                    _print_exception("Failed to close rejected source while monitor is running.")
            self.load_panel.set_current_source(self._source)
            return False
        self._stop_realtime_display()
        vmin, vmax = estimate_initial_clim(source)
        self._source = source
        self._clear_monitor_overlays()
        self._monitor_run_contexts.clear()
        self.detect_window.reset_monitor_feedback()
        self.load_panel.set_current_source(source)
        self.ann_panel.set_source_context(source, axis_metadata=getattr(self, "axis_metadata", None))
        self._apply_stream_processing_for_source(source)
        self.disp_model.set_source(source)
        self.viz_panel.set_clim(vmin, vmax)
        self.viz_panel.set_source_meta(getattr(source, "start_time", None))
        self._clear_compare_sources()
        self._refresh_menu_state()
        return True

    def _on_clear(self) -> bool:
        if not self._source_switch_allowed("clearing the source"):
            return False
        old = self._source
        self._source = None
        self.disp_model.clear()
        if old is not None:
            try:
                old.close()
            except Exception:
                _print_exception("Failed to close source while clearing the main window.")
        self.load_panel.reset_panel()
        self.ann_panel.clear_silent()
        self.ann_panel.set_source_context(None, axis_metadata=getattr(self, "axis_metadata", None))
        self.roi_panel.reset()
        self.viz_panel.reset()
        self._clear_compare_sources()
        self._refresh_menu_state()
        return True

    def _on_goto(self) -> None:
        if self._source is None:
            return
        dims = getattr(self._source, "dims", "nt_nc")
        nt = self._source.shape[0] if dims == "nt_nc" else self._source.shape[1]
        fs = float(getattr(self._source, "fs", 1.0)) or 1.0
        t0 = getattr(self._source, "start_time", None)

        dlg = GoToDialog(nt=nt, fs=fs, start_time=t0, parent=self, tape=self._source)
        if dlg.exec() != QDialog.Accepted:
            return
        sample = dlg.result_sample()
        if sample is not None:
            self.disp_model.goto_sample(sample)

    def _on_overview(self) -> None:
        """View ▸ Navigator…: full-read image of the file under the current view."""
        if self._source is None:
            return
        meta = getattr(self, "axis_metadata", None)
        if not getattr(meta, "is_default_time_channel", True):
            QMessageBox.information(self, "Navigator", "The navigator is only available for time-channel sources.")
            return
        try:
            from filark.explore.overview import file_range_at
            from filark.gui.overview import OverviewWindow
        except ImportError as exc:
            QMessageBox.warning(self, "Navigator", f"The navigator needs pyqtgraph: {exc}")
            return
        bounds = self.disp_model.current_view_bounds() or {"t0": 0, "t1": 1}
        start, stop = file_range_at(self._source, (int(bounds["t0"]) + int(bounds["t1"])) // 2)
        win = OverviewWindow(
            self._source,
            self.disp_model,
            start=start,
            stop=stop,
            filter_params=self.disp_model.display_filter_params(),
        )
        win.resize(*tuple(int(v) for v in self.cfg.roi_window_size))
        win.show()
        self._overview_windows.append(win)
        win.destroyed.connect(
            lambda *_: self._overview_windows.remove(win) if win in self._overview_windows else None
        )

    def _show_transform_status(self, state: dict) -> None:
        label = str(state.get("label") or state.get("active") or "")
        if not label:
            return
        self.statusBar().showMessage(f"Tape transform: {label}", 2000)

    def _on_source_transform_changed(self, _state: dict) -> None:
        if self._source is not None:
            self.ann_panel.set_source_context(self._source, axis_metadata=getattr(self, "axis_metadata", None))

    def _cycle_tape_transform(self, step: int) -> None:
        if self._source is None:
            return
        state = self.disp_model.cycle_source_transform(step)
        if state.get("enabled"):
            self._show_transform_status(state)

    def _reset_tape_transform(self) -> None:
        if self._source is None:
            return
        state = self.disp_model.reset_source_transform()
        if state.get("enabled"):
            self._show_transform_status(state)
