"""Keyboard and drag/drop event helpers for the main GUI window."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QApplication, QAbstractSpinBox, QComboBox, QDialog, QLineEdit, QPlainTextEdit, QTextEdit

from filark.gui.shortcut_help import ShortcutHelpDialog


class EventControllerMixin:
    """Own shortcut wiring, global key filtering, and source drop handling."""

    def _init_event_state(self) -> None:
        self._last_open_tab_idx: int = 0
        self._shortcut_help_dialog: ShortcutHelpDialog | None = None

    def _init_shortcuts(self) -> None:
        # All window-level shortcuts are QAction shortcuts owned by the menu bar.
        self._build_menu_bar()

    def _filter_key_event(self, event) -> bool:
        if event.type() == QEvent.KeyPress and self._is_shortcut_help_event(event):
            if not self._focus_accepts_text():
                self._toggle_shortcut_help()
                event.accept()
                return True
        if event.type() in (QEvent.KeyPress, QEvent.ShortcutOverride):
            focus_widget = QApplication.focusWidget()
            focus_window = focus_widget.window() if focus_widget is not None else None
            if (
                isinstance(focus_window, QDialog)
                and focus_window is not self
                and event.key() == Qt.Key_Escape
            ):
                focus_window.close()
                event.accept()
                return True
        if event.type() in (QEvent.KeyPress, QEvent.ShortcutOverride, QEvent.KeyRelease):
            if self._canvas_focus_active() and self._is_delete_like_event(event):
                event.accept()
                return True
        return False

    def _is_shortcut_help_event(self, event) -> bool:
        text = event.text() or ""
        if text == "?":
            return True
        modifiers = event.modifiers()
        return bool(event.key() == Qt.Key_Slash and modifiers & Qt.ShiftModifier)

    def _focus_accepts_text(self) -> bool:
        focus_widget = QApplication.focusWidget()
        if focus_widget is None:
            return False
        if isinstance(focus_widget, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)):
            return True
        if isinstance(focus_widget, QComboBox) and focus_widget.isEditable():
            return True
        return False

    def _toggle_shortcut_help(self) -> None:
        dlg = self._shortcut_help_dialog
        if dlg is not None and dlg.isVisible():
            dlg.close()
            return
        if dlg is None:
            dlg = ShortcutHelpDialog(self)
            self._shortcut_help_dialog = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _canvas_focus_active(self) -> bool:
        app = QApplication.instance()
        focus_widget = app.focusWidget() if app is not None else None
        if focus_widget is None:
            return False
        return focus_widget is self.canvas.native or self._canvas_container.isAncestorOf(focus_widget)

    def _is_delete_like_event(self, event) -> bool:
        if event.key() in (Qt.Key_Backspace, Qt.Key_Delete):
            return True
        text = event.text() or ""
        if text in ("\b", "\x7f"):
            return True
        return False

    def _on_shortcut_collapse_ui(self) -> None:
        active_window = QApplication.activeWindow()
        if isinstance(active_window, QDialog) and active_window is not self:
            active_window.close()
            return
        focus_widget = QApplication.focusWidget()
        if focus_widget is not None:
            focus_window = focus_widget.window()
            if isinstance(focus_window, QDialog) and focus_window is not self:
                focus_window.close()
                return
        if self.is_drawer_open:
            self.navbar.clear_selection()
            self.handle_panel_toggle(-1)
        else:
            self.navbar.select_tab(self._last_open_tab_idx)

    def _handle_drag_enter(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def _handle_drop(self, event) -> None:
        urls = event.mimeData().urls()
        if not urls:
            return
        path = Path(urls[0].toLocalFile())
        if not path.exists():
            return
        if not self._on_clear():
            event.ignore()
            return
        if self.load_source_path(path):
            event.acceptProposedAction()
        else:
            event.ignore()
