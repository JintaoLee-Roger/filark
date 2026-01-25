from PySide6.QtWidgets import QFrame, QVBoxLayout, QToolButton, QButtonGroup, QSizePolicy
from PySide6.QtCore import Signal, QSize, Qt

class NavBar(QFrame):
    """左侧窄条导航栏，包含图标按钮"""
    
    # 信号：发射被点击的 Tab 索引，如果再次点击同一个则发射 -1 (表示关闭)
    idx_changed = Signal(int)

    def __init__(self, items: list[tuple[str, str]], parent=None):
        super().__init__(parent)
        self.setFixedWidth(50)  # 窄宽度
        self.setObjectName("NavBar")

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(10)
        self.layout.addStretch() # 顶部占位，如果希望图标居中可以去掉，或者根据需求调整

        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(False) # 允许取消选中
        self.btn_group.idClicked.connect(self._on_btn_clicked)
        
        self._current_idx = -1

        # 从底部向顶部的布局习惯，或者是从上到下，这里使用从上到下
        # 清空 layout 重新添加
        self._items = items
        self.setup_ui()

    def setup_ui(self):
        # 清除旧的 stretch
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for idx, (name, icon_text) in enumerate(self._items):
            btn = QToolButton()
            btn.setText(icon_text) # 实际开发建议使用 setIcon(QIcon(...))
            btn.setToolTip(name)
            btn.setCheckable(True)
            btn.setFixedSize(40, 40)
            
            self.layout.addWidget(btn)
            self.btn_group.addButton(btn, idx)
        
        self.layout.addStretch() # 底部挤压，使图标靠上

    def _on_btn_clicked(self, idx):
        sender = self.btn_group.button(idx)
        
        if idx == self._current_idx:
            # 如果点击的是当前已经打开的，则关闭它（取消选中状态）
            self.btn_group.setExclusive(False)
            sender.setChecked(False)
            self._current_idx = -1
            self.idx_changed.emit(-1)
        else:
            # 切换到新的
            self._current_idx = idx
            # 确保其他的被取消选中 (手动模拟 Exclusive 行为以支持 Toggle)
            for btn in self.btn_group.buttons():
                if btn is not sender:
                    btn.setChecked(False)
            sender.setChecked(True)
            self.idx_changed.emit(idx)