from PySide6.QtWidgets import QMainWindow, QDockWidget
from PySide6.QtCore import Qt

class ROIWindow(QMainWindow):
    def __init__(self, roi_data, view_type="fk", parent=None):
        super().__init__(parent)
        self.roi_data = roi_data
        self.view_type = view_type

        self._view = self._create_view(view_type)
        self.setCentralWidget(self._view)  # 图像区域 = 可自由拉伸的中心

        # controls 由 view 提供（A方式）
        controls = None
        if hasattr(self._view, "create_controls"):
            controls = self._view.create_controls(parent=self)

        if controls is not None:
            dock = QDockWidget("Controls", self)
            dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
            dock.setFeatures(
                QDockWidget.DockWidgetMovable |
                QDockWidget.DockWidgetFloatable |
                QDockWidget.DockWidgetClosable
            )
            dock.setWidget(controls)
            self.addDockWidget(Qt.RightDockWidgetArea, dock)

        self.setWindowTitle(f"ROI View - {view_type}")

    def _create_view(self, view_type):
        if view_type == "fk":
            from .views.fk_view import FKView
            return FKView(self.roi_data)
        raise ValueError(view_type)
