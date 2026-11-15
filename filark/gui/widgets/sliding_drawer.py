"""Animated side drawer anchored to the navbar edge."""

from PySide6.QtWidgets import QWidget, QStackedWidget, QVBoxLayout
from PySide6.QtCore import QPropertyAnimation, QEasingCurve, QPoint


class SlidingDrawer(QWidget):
    """Slide-in drawer that hosts stacked side-panel pages."""

    def __init__(self, parent=None, width=300, animation_ms: int = 250):
        super().__init__(parent)
        self.target_width = width

        self.setObjectName("SlidingDrawer")

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedWidget(self)
        self.main_layout.addWidget(self.stack)

        self.anim = QPropertyAnimation(self, b"pos")
        self.anim.setDuration(max(0, int(animation_ms)))
        self.anim.setEasingCurve(QEasingCurve.OutCubic)

        self._pending_hide = False
        self.anim.finished.connect(self._on_anim_finished)

        self.hide()

    def add_module(self, widget: QWidget):
        self.stack.addWidget(widget)

    def set_content_widget(self, index: int):
        self.stack.setCurrentIndex(index)

    def _on_anim_finished(self):
        if self._pending_hide:
            self._pending_hide = False
            self.hide()

    def toggle(self, show: bool, anchor_x: int, parent_height: int):
        """Show or hide the drawer relative to the navbar right edge."""
        self.anim.stop()
        self._pending_hide = False

        self.resize(self.target_width, parent_height)

        if show:
            was_hidden = self.isHidden()
            self.show()
            self.raise_()

            end_pos = QPoint(anchor_x, 0)

            if was_hidden:
                self.move(-self.target_width, 0)

            self.anim.setStartValue(self.pos())
            self.anim.setEndValue(end_pos)
            self.anim.start()

        else:
            if self.isHidden():
                self.move(-self.target_width, 0)
                return

            end_pos = QPoint(-self.target_width, 0)
            self.anim.setStartValue(self.pos())
            self.anim.setEndValue(end_pos)
            self._pending_hide = True
            self.anim.start()
