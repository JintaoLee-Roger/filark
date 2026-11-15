"""Top bar for switching between multiple loaded sources."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QLabel, QPushButton, QWidget


@dataclass
class _SourceEntry:
    name: str
    source: object
    button: QPushButton


class SourceBar(QWidget):
    """Thin source-switch bar shown when multiple datasets are loaded."""

    source_activated = Signal(object)  # the Tape / source object

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._entries: list[_SourceEntry] = []

        self.setVisible(False)
        self.setFixedHeight(34)
        self.setObjectName("SourceBar")
        self.setStyleSheet(
            "#SourceBar {"
            "  background: #2a2a2a;"
            "  border-bottom: 1px solid #444;"
            "}"
            "#SourceBar QPushButton {"
            "  color: #ccc;"
            "  background: transparent;"
            "  border: 1px solid transparent;"
            "  border-radius: 4px;"
            "  padding: 2px 10px;"
            "  font-size: 12px;"
            "}"
            "#SourceBar QPushButton:checked {"
            "  color: #fff;"
            "  background: #0078d7;"
            "  border-color: #0078d7;"
            "}"
            "#SourceBar QPushButton:hover:!checked {"
            "  border-color: #666;"
            "}"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(6)

        self._compare_label = QLabel("Compare:")
        self._compare_label.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(self._compare_label)

        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)

        layout.addStretch(1)

        self._layout = layout

    def add_source(self, name: str, source: object, *, active: bool = False) -> None:
        """Add a source entry. If active=True, it becomes the selected one."""
        button = QPushButton(name)
        button.setCheckable(True)
        button.setFixedHeight(26)

        insert_pos = self._layout.count() - 1
        self._layout.insertWidget(insert_pos, button)

        self._btn_group.addButton(button)
        entry = _SourceEntry(name=name, source=source, button=button)
        self._entries.append(entry)

        if active or len(self._entries) == 1:
            button.setChecked(True)

        button.clicked.connect(lambda _checked, selected_entry=entry: self._on_btn_clicked(selected_entry))

        if len(self._entries) >= 2:
            self.setVisible(True)

    def set_active_name(self, name: str) -> None:
        for entry in self._entries:
            entry.button.setChecked(entry.name == name)

    def clear(self) -> None:
        """Remove all entries and hide the bar."""
        for entry in self._entries:
            self._btn_group.removeButton(entry.button)
            self._layout.removeWidget(entry.button)
            entry.button.deleteLater()
        self._entries.clear()
        self.setVisible(False)

    def _on_btn_clicked(self, entry: _SourceEntry) -> None:
        for source_entry in self._entries:
            source_entry.button.setChecked(source_entry is entry)
        self.source_activated.emit(entry.source)
