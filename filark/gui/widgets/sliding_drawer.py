from PySide6.QtWidgets import (QWidget, QStackedWidget, QVBoxLayout)
from PySide6.QtCore import QPropertyAnimation, QEasingCurve, QPoint




class SlidingDrawer(QWidget):
    def __init__(self, parent=None, width=300):
        super().__init__(parent)
        self.target_width = width

        self.setObjectName("SlidingDrawer")

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedWidget(self)
        self.main_layout.addWidget(self.stack)

        self.anim = QPropertyAnimation(self, b"pos")
        self.anim.setDuration(250)
        self.anim.setEasingCurve(QEasingCurve.OutCubic)

        self._pending_hide = False
        self.anim.finished.connect(self._on_anim_finished)

        self.hide()

    # ===== 你缺的两个接口 =====
    def add_module(self, widget: QWidget):
        self.stack.addWidget(widget)

    def set_content_widget(self, index: int):
        self.stack.setCurrentIndex(index)

    # ===== 动画结束后 hide（避免抽屉盖住 NavBar）=====
    def _on_anim_finished(self):
        if self._pending_hide:
            self._pending_hide = False
            self.hide()

    def toggle(self, show: bool, anchor_x: int, parent_height: int):
        """
        anchor_x: NavBar 右边缘 x（通常=navbar_width）
        """
        self.anim.stop()
        self._pending_hide = False

        self.resize(self.target_width, parent_height)

        if show:
            was_hidden = self.isHidden()
            self.show()
            self.raise_()

            end_pos = QPoint(anchor_x, 0)

            # 关键：如果之前隐藏，把抽屉放到完全左侧（不覆盖 NavBar）
            if was_hidden:
                self.move(-self.target_width, 0)

            self.anim.setStartValue(self.pos())
            self.anim.setEndValue(end_pos)
            self.anim.start()

        else:
            if self.isHidden():
                self.move(-self.target_width, 0)
                return

            # 关键：关闭时移动到 x=-width，确保不盖住 NavBar
            end_pos = QPoint(-self.target_width, 0)
            self.anim.setStartValue(self.pos())
            self.anim.setEndValue(end_pos)
            self._pending_hide = True
            self.anim.start()

