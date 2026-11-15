"""Analysis plugin controller for the main GUI window."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox, QVBoxLayout, QWidget

from filark._exceptions import print_exception


def _print_exception(context: str) -> None:
    print_exception(context, prefix="AnalysisController")


class AnalysisControllerMixin:
    """Own analysis plugin execution, figure windows, and channel picking."""

    def _init_analysis_state(self) -> None:
        self._next_analysis_overlay_uid: int = -(2 * 10 ** 9)
        self._analysis_pick_owner: int | None = None
        self._analysis_pick_callback = None

    def _on_analysis_run(self, plugin, params: dict, roi):
        if self._source is None:
            QMessageBox.information(self, "Analysis", "No data source loaded.")
            return

        src = self._source
        axis = getattr(self, "axis_metadata", None)
        if (
            axis is not None
            and not axis.is_default_time_channel
            and not bool(getattr(plugin, "supports_axis_metadata", False))
        ):
            QMessageBox.information(
                self,
                "Analysis",
                f"{plugin.label} only supports the default time-channel view.",
            )
            return
        fs = float(getattr(src, "fs", 1.0))
        dx = float(getattr(src, "dx", 1.0))
        dims = getattr(src, "dims", "nt_nc")
        nt = src.shape[0] if dims == "nt_nc" else src.shape[1]
        nc = src.shape[1] if dims == "nt_nc" else src.shape[0]

        try:
            params = plugin.validate_params(params, fs=fs, dx=dx, nt=nt, nc=nc)
        except Exception as exc:
            _print_exception(f"Analysis plugin '{plugin.label}' rejected parameters.")
            QMessageBox.warning(self, "Analysis - Invalid Params", str(exc))
            return

        roi_data = None
        if roi is not None and (roi.t1 > roi.t0 or roi.c1 > roi.c0):
            try:
                roi_data = self._extract_roi_data(roi)
            except Exception:
                _print_exception("Failed to extract ROI data for analysis plugin.")

        try:
            widget = plugin.create_widget(src, parent=None)
        except Exception as exc:
            _print_exception(f"Analysis plugin '{plugin.label}' failed in create_widget().")
            QMessageBox.warning(self, "Analysis Failed", f"create_widget() raised:\n{exc}")
            return

        if widget is not None:
            overlay_uid = self._alloc_analysis_overlay_uid()
            if hasattr(widget, "set_filark_context"):
                try:
                    widget.set_filark_context(
                        {
                            "current_view_bounds_getter": self.disp_model.current_view_bounds,
                            "initial_view_bounds": self.disp_model.current_view_bounds(),
                            "roi_getter": self._analysis_current_roi_data,
                            "axis": axis.as_dict() if axis is not None else None,
                            "goto_sample": self.disp_model.goto_sample,
                            "set_channel_marker": (
                                lambda channel, _uid=overlay_uid: self._set_analysis_channel_marker(_uid, channel)
                            ),
                            "clear_channel_marker": (
                                lambda _uid=overlay_uid: self._clear_analysis_channel_marker(_uid)
                            ),
                            "begin_channel_pick": (
                                lambda callback, _uid=overlay_uid: self._begin_analysis_channel_pick(_uid, callback)
                            ),
                            "cancel_channel_pick": (
                                lambda _uid=overlay_uid: self._cancel_analysis_channel_pick(_uid)
                            ),
                            "raise_main_window": self.raise_,
                        }
                    )
                except Exception:
                    _print_exception("Failed to inject FiLark context into analysis widget.")
            widget.setWindowFlags(Qt.Window)
            widget.setWindowTitle(plugin.label)
            widget.resize(*tuple(int(v) for v in self.cfg.analysis_widget_size))
            widget.show()
            if not hasattr(self, "_analysis_windows"):
                self._analysis_windows: list = []
            self._analysis_windows.append(widget)
            widget.destroyed.connect(
                lambda *_: self._analysis_windows.remove(widget)
                if widget in self._analysis_windows else None
            )
            return

        try:
            result = plugin.run(src, params=params, roi=roi_data)
        except Exception as exc:
            _print_exception(f"Analysis plugin '{plugin.label}' failed in run().")
            QMessageBox.warning(self, "Analysis Failed", str(exc))
            return

        if result is not None:
            self._show_mpl_figures(result, title=plugin.label)

    def _analysis_current_roi_data(self) -> dict | None:
        """Return the current ROI as an analysis ``roi`` dict, or None if no ROI is set."""
        if self._source is None:
            return None
        roi = self.roi_panel.current_roi()
        if roi.is_empty():
            return None
        return self._extract_roi_data(roi)

    def _alloc_analysis_overlay_uid(self) -> int:
        self._next_analysis_overlay_uid -= 1
        return int(self._next_analysis_overlay_uid)

    def _analysis_channel_count(self) -> int:
        src = self._source
        if src is None:
            return 0
        dims = getattr(src, "dims", "nt_nc")
        return int(src.shape[1] if dims == "nt_nc" else src.shape[0])

    def _analysis_channel_line_points(self, channel: int) -> list[tuple[float, float]]:
        src = self._source
        if src is None:
            return []
        dims = getattr(src, "dims", "nt_nc")
        nt = int(src.shape[0] if dims == "nt_nc" else src.shape[1])
        if nt <= 0:
            return []
        y = float(channel)
        return [(0.0, y), (float(max(0, nt - 1)), y)]

    def _set_analysis_channel_marker(self, uid: int, channel: int) -> None:
        nc = self._analysis_channel_count()
        if nc <= 0:
            return
        channel = max(0, min(int(channel), nc - 1))
        points = self._analysis_channel_line_points(channel)
        if not points:
            return
        payload = {
            "points": points,
            "color": str(self.cfg.analysis_channel_marker_color),
            "visible": True,
            "source_kind": "analysis_channel_marker",
            "selectable": False,
            "editable": False,
        }
        if self.disp_model.has_overlay(int(uid)):
            self.disp_model.update_overlay(int(uid), payload)
        else:
            self.disp_model.add_polyline_overlay(
                int(uid),
                payload,
                source_kind="analysis_channel_marker",
                selectable=False,
                editable=False,
            )

    def _clear_analysis_channel_marker(self, uid: int) -> None:
        self.disp_model.remove_overlay(int(uid))

    def _begin_analysis_channel_pick(self, owner_uid: int, callback) -> bool:
        if not callable(callback):
            return False
        self._analysis_pick_owner = int(owner_uid)
        self._analysis_pick_callback = callback
        try:
            self.canvas.native.setFocus()
        except Exception:
            _print_exception("Failed to focus canvas for channel pick.")
        return True

    def _cancel_analysis_channel_pick(self, owner_uid: int | None = None) -> None:
        if owner_uid is not None and self._analysis_pick_owner != int(owner_uid):
            return
        self._analysis_pick_owner = None
        self._analysis_pick_callback = None

    def _on_analysis_pick_mouse_press(self, ev) -> bool:
        callback = self._analysis_pick_callback
        if callback is None:
            return False
        button = getattr(ev, "button", None)
        if button not in (1, None):
            if button in (2, 3):
                self._cancel_analysis_channel_pick()
                return True
            return False
        pos = getattr(ev, "pos", None)
        if pos is None:
            self._cancel_analysis_channel_pick()
            return True
        nc = self._analysis_channel_count()
        if nc <= 0:
            self._cancel_analysis_channel_pick()
            return True
        try:
            _, y = self.disp_model.canvas_pos_to_world(pos)
            channel = max(0, min(int(round(float(y))), nc - 1))
            self._analysis_pick_owner = None
            self._analysis_pick_callback = None
            callback(channel)
        except Exception:
            self._cancel_analysis_channel_pick()
            _print_exception("Failed to resolve channel pick from main canvas.")
        return True

    def _show_mpl_figures(self, fig_or_list, title: str = "Analysis"):
        try:
            from matplotlib.backends.backend_qtagg import (
                FigureCanvasQTAgg,
                NavigationToolbar2QT,
            )
        except ImportError:
            QMessageBox.warning(
                self,
                title,
                "matplotlib is required to display analysis figures.\n"
                "Install it with: pip install matplotlib",
            )
            return

        figs = fig_or_list if isinstance(fig_or_list, list) else [fig_or_list]

        win = QWidget(None)
        win.setWindowFlags(Qt.Window)
        win.setWindowTitle(title)
        vbox = QVBoxLayout(win)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(4)

        for fig in figs:
            canvas = FigureCanvasQTAgg(fig)
            toolbar = NavigationToolbar2QT(canvas, win)
            vbox.addWidget(toolbar)
            vbox.addWidget(canvas, 1)

        win.resize(*tuple(int(v) for v in self.cfg.analysis_figure_window_size))
        win.show()

        if not hasattr(self, "_analysis_windows"):
            self._analysis_windows: list = []
        self._analysis_windows.append(win)
        win.destroyed.connect(
            lambda *_: self._analysis_windows.remove(win)
            if win in self._analysis_windows else None
        )
