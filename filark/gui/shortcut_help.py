"""Keyboard shortcut data and the in-app help dialog."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class ShortcutEntry:
    group: str
    shortcut: str
    action: str


SHORTCUTS: tuple[ShortcutEntry, ...] = (
    ShortcutEntry("Application", "Esc", "Toggle the side panel, or close the active dialog."),
    ShortcutEntry("Application", "Ctrl/Cmd+G", "Go to sample/time/UTC position."),
    ShortcutEntry("Application", "Ctrl/Cmd+O", "Open a file."),
    ShortcutEntry("Application", "Ctrl/Cmd+Alt/Option+O", "Open a folder."),
    ShortcutEntry("Application", "Ctrl/Cmd+W", "Close the current source."),
    ShortcutEntry("Application", "Ctrl/Cmd+I", "Source properties (metadata and overrides)."),
    ShortcutEntry("Application", "Ctrl/Cmd+Shift+O", "Navigator: whole-file image of the data; click to jump."),
    ShortcutEntry("Application", "Ctrl/Cmd+Left", "Previous comparison source."),
    ShortcutEntry("Application", "Ctrl/Cmd+Right", "Next comparison source."),
    ShortcutEntry("Application", "?", "Show or hide this shortcut panel."),
    ShortcutEntry("Navigation", "Left / Right", "Pan time."),
    ShortcutEntry("Navigation", "Up / Down", "Pan channels."),
    ShortcutEntry("Navigation", "Page Up / Page Down", "Page channels."),
    ShortcutEntry("Navigation", "Alt/Option+Up / Alt/Option+Down", "Alternative page channel controls."),
    ShortcutEntry("Display Scale", "X", "Increase time compression."),
    ShortcutEntry("Display Scale", "Shift+X", "Decrease time compression."),
    ShortcutEntry("Display Scale", "Y", "Increase channel compression."),
    ShortcutEntry("Display Scale", "Shift+Y", "Decrease channel compression."),
    ShortcutEntry("Display Scale", "F", "Fit channels into the Y buffer."),
    ShortcutEntry("Display Scale", "Shift+F", "Reset Y compression."),
    ShortcutEntry("Display Scale", "A", "Reduce display aspect."),
    ShortcutEntry("Display Scale", "Shift+A", "Increase display aspect."),
    ShortcutEntry("Display Scale", "C", "Toggle cursor coordinate display."),
    ShortcutEntry("Tape Transform", "Alt/Option+T", "Next source-provided Tape transform."),
    ShortcutEntry("Tape Transform", "Alt/Option+Shift+T", "Previous source-provided Tape transform."),
    ShortcutEntry("Tape Transform", "Alt/Option+0", "Reset to Raw."),
    ShortcutEntry("Realtime Playback", "R", "Toggle realtime auto-scroll."),
    ShortcutEntry("Realtime Playback", "S", "Stop realtime auto-scroll."),
    ShortcutEntry("Realtime Playback", "+ / =", "Speed up playback."),
    ShortcutEntry("Realtime Playback", "- / _", "Slow down playback."),
    ShortcutEntry("Realtime Playback", "1..8", "Select configured realtime sample-rate presets."),
    ShortcutEntry("Diagnostics", "P", "Print realtime debug stats."),
    ShortcutEntry("Diagnostics", "Shift+P", "Print showcase timing and memory stats."),
    ShortcutEntry("Diagnostics", "B", "Run a short throughput benchmark."),
    ShortcutEntry("Diagnostics", "M", "Print current memory snapshot when available."),
    ShortcutEntry("Diagnostics", "Shift+M", "Print memory snapshot history when available."),
    ShortcutEntry("Annotation", "Ctrl+drag", "Draw bbox or paint selected/new mask."),
    ShortcutEntry("Annotation", "Ctrl+click, then click", "Build polyline."),
    ShortcutEntry("Annotation", "Enter / Right-click", "Finish current polyline."),
    ShortcutEntry("Annotation", "Esc", "Cancel current draw/edit session."),
    ShortcutEntry("Annotation", "Shift+click polyline segment", "Insert vertex."),
    ShortcutEntry("Annotation", "Alt/Option+click polyline vertex", "Delete vertex."),
    ShortcutEntry("Annotation", "Shift while mask painting", "Erase."),
    ShortcutEntry("Annotation", "Ctrl+Z / Ctrl+Y", "Mask brush undo/redo."),
)


class ShortcutHelpDialog(QDialog):
    """Non-modal searchable shortcut reference."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.Tool)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setModal(False)
        self.resize(720, 620)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        title = QLabel("Keyboard Shortcuts")
        title.setObjectName("ShortcutHelpTitle")
        root.addWidget(title)

        hint = QLabel(
            "Single-key shortcuts are exact: R is not Shift+R, and modified keys only trigger when listed."
        )
        hint.setObjectName("ShortcutHelpHint")
        hint.setWordWrap(True)
        root.addWidget(hint)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search shortcuts")
        root.addWidget(self.search)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        root.addWidget(self.scroll, 1)

        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(10)
        self.scroll.setWidget(self.content)

        self._sc_close = QShortcut(QKeySequence("Esc"), self)
        self._sc_close.setContext(Qt.WindowShortcut)
        self._sc_close.activated.connect(self.close)

        self.search.textChanged.connect(self._render)
        self._render("")

    def _render(self, query: str = "") -> None:
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        query = (query or "").strip().lower()
        groups: dict[str, list[ShortcutEntry]] = {}
        for entry in SHORTCUTS:
            haystack = f"{entry.group} {entry.shortcut} {entry.action}".lower()
            if query and query not in haystack:
                continue
            groups.setdefault(entry.group, []).append(entry)

        for group, entries in groups.items():
            box = QGroupBox(group)
            grid = QGridLayout(box)
            grid.setContentsMargins(10, 8, 10, 10)
            grid.setHorizontalSpacing(18)
            grid.setVerticalSpacing(6)
            for row, entry in enumerate(entries):
                key = QLabel(entry.shortcut)
                key.setObjectName("ShortcutKey")
                key.setTextInteractionFlags(Qt.TextSelectableByMouse)
                action = QLabel(entry.action)
                action.setObjectName("ShortcutAction")
                action.setWordWrap(True)
                grid.addWidget(key, row, 0)
                grid.addWidget(action, row, 1)
            grid.setColumnStretch(1, 1)
            self.content_layout.addWidget(box)

        if not groups:
            empty = QLabel("No shortcuts matched.")
            empty.setObjectName("ShortcutHelpHint")
            self.content_layout.addWidget(empty)
        self.content_layout.addStretch(1)
