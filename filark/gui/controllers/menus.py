"""Menu bar of the main window.

Layout rule
-----------
* The **left drawer** holds things that act on the picture currently on the
  canvas (Visuals, Annotation, ROI).
* The **menu bar** holds functions: opening/closing data, navigation commands
  that previously only had keyboard shortcuts, and the analysis windows.

Window-level shortcuts are registered once, here, as ``QAction`` shortcuts.
Single keys that the streaming canvas handles itself (R, S, +, -) are only
shown as hints in the menu text, so they are never handled twice.
"""

from __future__ import annotations

from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QDialogButtonBox, QMenu, QMessageBox

from filark.gui.roi.views.param_dialog import ParamDialog

from .layout import TAB_ROI

# Inspection views opened on the current ROI (key used by ROIWindow).
ROI_INSPECT_VIEWS: list[tuple[str, str]] = [
    ("F-K Spectrum", "fk"),
    ("Time-Frequency (STFT)", "stft"),
    ("Power Spectrum (PSD)", "psd"),
    ("Cross-correlation Gather", "xcorr"),
    ("τ-p (Radon)", "tp"),
    ("RMS Energy Profile", "rms"),
    ("SVD Singular Values", "svd"),
]


class MenuControllerMixin:
    """Build the menu bar and route its actions to the existing controllers."""

    # ------------------------------------------------------------------ build
    def _build_menu_bar(self) -> None:
        mb = self.menuBar()
        mb.setNativeMenuBar(True)  # macOS: system menu bar, takes no window space
        self._source_actions: list[QAction] = []
        self._roi_param_dialogs: dict[str, ParamDialog] = {}

        self._build_file_menu(mb.addMenu("&File"))
        self._build_view_menu(mb.addMenu("&View"))
        self._build_analyze_menu(mb.addMenu("&Analyze"))
        self._build_help_menu(mb.addMenu("&Help"))
        self._refresh_menu_state()

    def _action(
        self,
        menu: QMenu,
        text: str,
        slot,
        shortcut=None,
        *,
        needs_source: bool = False,
        tip: str = "",
    ) -> QAction:
        act = QAction(text, self)
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
        if tip:
            act.setToolTip(tip)
            act.setStatusTip(tip)
        act.triggered.connect(lambda *_: slot())
        menu.addAction(act)
        # Also attach to the window so the shortcut works while menus are closed.
        self.addAction(act)
        if needs_source:
            self._source_actions.append(act)
        return act

    def _build_file_menu(self, m: QMenu) -> None:
        self._action(m, "Open File…", self.load_panel.pick_file, QKeySequence.Open)
        self._action(m, "Open Folder…", self.load_panel.pick_dir, "Ctrl+Alt+O")
        m.addSeparator()
        self._action(
            m, "Add to Compare…", self.load_panel.request_compare, needs_source=True,
            tip="Select a processed .npy file and add it to the comparison bar.",
        )
        self._action(m, "Close Source", self._on_clear, QKeySequence.Close, needs_source=True)
        m.addSeparator()
        self._action(m, "Source Properties…", self.source_props_dialog.show_and_raise, "Ctrl+I", needs_source=True)
        self._action(m, "Export Subset…", self.load_panel.export_subset, needs_source=True)
        m.addSeparator()
        quit_act = self._action(m, "Quit", self.close, QKeySequence.Quit)
        quit_act.setMenuRole(QAction.QuitRole)

    def _build_view_menu(self, m: QMenu) -> None:
        cfg = self.cfg
        self._action(m, "Go to…", self._on_goto, str(cfg.shortcut_goto), needs_source=True)
        self._action(
            m, "Navigator…", self._on_overview, str(cfg.shortcut_overview), needs_source=True,
            tip="Whole-file (or whole-recording) image of the data; click to jump there.",
        )
        m.addSeparator()

        tm = m.addMenu("Transform")
        self._action(tm, "Next Transform", lambda: self._cycle_tape_transform(+1),
                     str(cfg.shortcut_tape_transform_next), needs_source=True)
        self._action(tm, "Previous Transform", lambda: self._cycle_tape_transform(-1),
                     str(cfg.shortcut_tape_transform_prev), needs_source=True)
        self._action(tm, "Raw (No Transform)", self._reset_tape_transform,
                     str(cfg.shortcut_tape_transform_raw), needs_source=True)

        self._action(m, "Previous Source", lambda: self._cycle_compare_source(-1),
                     str(cfg.shortcut_compare_prev), needs_source=True)
        self._action(m, "Next Source", lambda: self._cycle_compare_source(+1),
                     str(cfg.shortcut_compare_next), needs_source=True)
        m.addSeparator()

        pm = m.addMenu("Playback")
        self._action(pm, "Play / Pause    (R)", self._menu_toggle_playback, needs_source=True)
        self._action(pm, "Stop    (S)", self._menu_stop_playback, needs_source=True)
        self._action(pm, "Faster    (+)", lambda: self._menu_scale_playback(1.25), needs_source=True)
        self._action(pm, "Slower    (−)", lambda: self._menu_scale_playback(1 / 1.25), needs_source=True)
        m.addSeparator()

        self._action(m, "Toggle Side Panel", self._on_shortcut_collapse_ui, str(cfg.shortcut_collapse_ui))

    def _build_analyze_menu(self, m: QMenu) -> None:
        im = m.addMenu("Inspect ROI")
        for label, key in ROI_INSPECT_VIEWS:
            self._action(im, f"{label}…", lambda k=key: self._menu_run_roi_view(k), needs_source=True)

        act = self._action(
            m, "Process ROI…", lambda: self._menu_run_roi_view("denoise"), needs_source=True,
            tip="Preview processors on the ROI and apply them to the full data.",
        )
        act.setProperty("_needs_processors", True)
        m.addSeparator()

        self._action(m, "Detect Events…", self.detect_window.show_and_raise,
                     tip="Run an event detector over the view, the full source, or the live display.")

        analyses = list(self.cfg.analysis_list())
        if analyses:
            m.addSeparator()
            for plugin in analyses:
                self._action(m, f"{plugin.label}…", lambda p=plugin: self._menu_run_analysis(p), needs_source=True)

    def _build_help_menu(self, m: QMenu) -> None:
        self._action(m, "Keyboard Shortcuts    (?)", self._toggle_shortcut_help)

    # ----------------------------------------------------------------- state
    def _refresh_menu_state(self) -> None:
        has_source = getattr(self, "_source", None) is not None
        has_processors = bool(self.cfg.processor_list())
        for act in getattr(self, "_source_actions", []):
            enabled = has_source
            if act.property("_needs_processors"):
                enabled = enabled and has_processors
            act.setEnabled(enabled)

    # --------------------------------------------------------------- handlers
    def _menu_toggle_playback(self) -> None:
        canvas = self.canvas
        if getattr(canvas, "_rt_running", False):
            canvas.stop_realtime_autoscroll(reason="menu")
        else:
            canvas.start_realtime_autoscroll(reason="menu")

    def _menu_stop_playback(self) -> None:
        self.canvas.stop_realtime_autoscroll(reason="menu")

    def _menu_scale_playback(self, factor: float) -> None:
        speed = float(getattr(self.canvas, "_rt_speed", 1.0) or 1.0)
        self.canvas.set_realtime_speed(speed * float(factor))

    def _menu_run_roi_view(self, key: str) -> None:
        if self._source is None:
            return
        roi = self.roi_panel.current_roi()
        if roi.is_empty():
            self.navbar.select_tab(TAB_ROI)
            self.roi_panel.set_roi_mode(True)
            QMessageBox.information(
                self,
                "Select a region",
                "Drag on the canvas to select a region (ROI), then choose the tool again.",
            )
            return
        self._on_roi_annalysis(key, roi)

    def _menu_run_analysis(self, plugin) -> None:
        if self._source is None:
            return
        if not plugin.param_spec:
            self._on_analysis_run(plugin, plugin.default_params(), self.roi_panel.current_roi())
            return
        key = str(getattr(plugin, "key", id(plugin)))
        dlg = self._roi_param_dialogs.get(key)
        if dlg is None:
            dlg = ParamDialog(param_spec=plugin.param_spec, title=plugin.label, parent=self)
            run_btn = dlg.button_box.addButton("Run", QDialogButtonBox.AcceptRole)
            run_btn.setDefault(True)
            run_btn.clicked.connect(
                lambda *_ , p=plugin, d=dlg: self._on_analysis_run(p, d.get_params(), self.roi_panel.current_roi())
            )
            self._roi_param_dialogs[key] = dlg
        dlg.show_and_raise()
