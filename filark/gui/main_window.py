"""Main Qt window that assembles the viewer, side panels, and ROI tools."""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtWidgets import QMainWindow, QWidget

from .config import GuiConfig
from .controllers.layout import TAB_ANNOTATION
from .mpl_cursor import disable_matplotlib_wait_cursor
from .native_titlebar import (
    configure_native_titlebar,
    perform_native_titlebar_double_click,
    uses_macos_custom_titlebar,
)
from .controllers import (
    AnalysisControllerMixin,
    AnnotationMixin,
    CompareControllerMixin,
    EventControllerMixin,
    LayoutControllerMixin,
    MenuControllerMixin,
    MonitorControllerMixin,
    ROIControllerMixin,
    SourceControllerMixin,
)


class MainWindow(
    QMainWindow,
    SourceControllerMixin,
    AnnotationMixin,
    MonitorControllerMixin,
    AnalysisControllerMixin,
    CompareControllerMixin,
    ROIControllerMixin,
    LayoutControllerMixin,
    MenuControllerMixin,
    EventControllerMixin,
):
    def __init__(self, cfg: GuiConfig):
        super().__init__()
        self.cfg = cfg
        disable_matplotlib_wait_cursor()
        self._native_titlebar_style = configure_native_titlebar(self, cfg.theme, expand_macos=True)
        self._native_titlebar_hit_area = None
        self._native_titlebar_button_area_width = 120
        self._native_titlebar_safe_area_connected = False

        self.setWindowTitle("FiLark Viz")
        self.resize(*tuple(int(v) for v in cfg.window_size))
        self.setAcceptDrops(True)

        self._init_source_state(cfg.source)
        self._init_event_state()
        self._init_compare_state()
        self._init_monitor_state()
        self._init_analysis_state()

        self._build_main_layout()
        self._init_shortcuts()

    def _annotation_gate_should_be_active(self) -> bool:
        if self.roi_panel.ck_roi_mode.isChecked():
            return False
        if self.is_drawer_open:
            return self._current_tab_idx == TAB_ANNOTATION
        return self._last_open_tab_idx == TAB_ANNOTATION

    def _sync_annotation_gate(self) -> None:
        self.disp_model.set_annotation_gate(self._annotation_gate_should_be_active())

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._refresh_native_titlebar()
        self._ensure_native_titlebar_safe_area_updates()
        if uses_macos_custom_titlebar(self._native_titlebar_style):
            self._ensure_native_titlebar_hit_area()
        self._sync_native_titlebar_hit_area()
        self._schedule_initial_source_load()
        QTimer.singleShot(50, self._refresh_native_titlebar)
        QTimer.singleShot(250, self._refresh_native_titlebar)

    def _refresh_native_titlebar(self) -> None:
        self._native_titlebar_style = configure_native_titlebar(
            self,
            self.cfg.theme,
            expand_macos=True,
            update_qt_flags=False,
        )
        if uses_macos_custom_titlebar(self._native_titlebar_style):
            self._ensure_native_titlebar_hit_area()
        self._sync_native_titlebar_hit_area()

    def _ensure_native_titlebar_safe_area_updates(self) -> None:
        if self._native_titlebar_safe_area_connected:
            return
        handle = self.windowHandle()
        if handle is None or not hasattr(handle, "safeAreaMarginsChanged"):
            return
        try:
            handle.safeAreaMarginsChanged.connect(self._sync_native_titlebar_hit_area)
        except Exception:
            return
        self._native_titlebar_safe_area_connected = True

    def _ensure_native_titlebar_hit_area(self) -> None:
        if self._native_titlebar_hit_area is not None:
            return
        self._native_titlebar_hit_area = QWidget(self)
        self._native_titlebar_hit_area.setObjectName("NativeTitleBarHitArea")
        self._native_titlebar_hit_area.setProperty("_filark_titlebar_double_click", True)
        self._native_titlebar_hit_area.setStyleSheet("background: transparent; border: 0;")
        self._native_titlebar_hit_area.installEventFilter(self)
        self._native_titlebar_hit_area.hide()

    def _sync_native_titlebar_hit_area(self) -> None:
        hit_area = getattr(self, "_native_titlebar_hit_area", None)
        if hit_area is None:
            return
        if not uses_macos_custom_titlebar(self._native_titlebar_style):
            hit_area.hide()
            return
        top = max(0, self._safe_area_margin("top"))
        left = self._native_titlebar_button_area_width + self._safe_area_margin("left")
        width = max(0, self.width() - left)
        hit_area.setGeometry(left, 0, width, top)
        hit_area.setVisible(top > 0 and width > 0)
        hit_area.raise_()

    def _safe_area_margin(self, side: str) -> int:
        handle = self.windowHandle()
        if handle is None or not hasattr(handle, "safeAreaMargins"):
            return 0
        try:
            margins = handle.safeAreaMargins()
            return int(getattr(margins, side)())
        except Exception:
            return 0

    def closeEvent(self, event) -> None:
        self._shutdown_background_workers()
        super().closeEvent(event)

    def eventFilter(self, watched, event):
        if (
            event.type() == QEvent.Type.MouseButtonDblClick
            and watched.property("_filark_titlebar_double_click")
        ):
            button = getattr(event, "button", lambda: None)()
            if button == Qt.MouseButton.LeftButton:
                event.accept()
                return perform_native_titlebar_double_click(self)
        if self._filter_key_event(event):
            return True
        return super().eventFilter(watched, event)

    # Source lifecycle and Go-To methods live in SourceControllerMixin.

    # ROI methods live in ROIControllerMixin.

    # Monitor/realtime controller methods live in MonitorControllerMixin.

    # Analysis and comparison methods live in controller mixins.

    # =========================================
    # Drag-and-drop
    # =========================================
    def dragEnterEvent(self, event):
        self._handle_drag_enter(event)

    def dropEvent(self, event):
        self._handle_drop(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_drawer()
        self._sync_native_titlebar_hit_area()
