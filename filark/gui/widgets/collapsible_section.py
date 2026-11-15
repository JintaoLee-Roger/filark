"""Reusable collapsible section for dense side-panel layouts."""

from __future__ import annotations

from PySide6.QtWidgets import QPushButton, QSizePolicy, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    """Toggle button plus a hideable body container."""

    def __init__(
        self,
        title: str,
        *,
        collapsed: bool = False,
        parent: QWidget | None = None,
        collapsed_marker: str = ">",
        expanded_marker: str = "v",
        body_margins: tuple[int, int, int, int] = (0, 0, 0, 2),
        body_spacing: int = 0,
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._collapsed_marker = collapsed_marker
        self._expanded_marker = expanded_marker

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(4)

        self._toggle_button = QPushButton()
        self._toggle_button.setCheckable(True)
        self._toggle_button.setChecked(not collapsed)
        self._toggle_button.setFlat(True)
        self._toggle_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._toggle_button.clicked.connect(self._on_toggle)
        root_layout.addWidget(self._toggle_button)

        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(*body_margins)
        self._body_layout.setSpacing(body_spacing)
        self._body.setVisible(not collapsed)
        root_layout.addWidget(self._body)

        self._refresh_label()

    def add(self, child: QWidget) -> "CollapsibleSection":
        self._body_layout.addWidget(child)
        return self

    def add_layout(self, layout) -> "CollapsibleSection":
        self._body_layout.addLayout(layout)
        return self

    def _on_toggle(self, checked: bool) -> None:
        self._body.setVisible(checked)
        self._refresh_label()

    def _refresh_label(self) -> None:
        marker = self._expanded_marker if self._toggle_button.isChecked() else self._collapsed_marker
        self._toggle_button.setText(f" {marker}  {self._title}")
