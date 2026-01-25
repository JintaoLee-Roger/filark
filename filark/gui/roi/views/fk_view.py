from __future__ import annotations

import numpy as np

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QDoubleSpinBox, QPushButton,
    QCheckBox, QGroupBox, QScrollArea, QSizePolicy, QLineEdit
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

from filark.dsp import fk as fifk  # 你提供的接口


def normlized(x, tfirst=False):
    # x: [ny, nt]
    if tfirst:
        x = x.T
    m = np.median(x, axis=1, keepdims=True)
    s = 1.4826 * np.median(np.abs(x - m), axis=1, keepdims=True) + 1e-12  # MAD
    x = (x - m) / s
    if tfirst:
        x = x.T
    return x


class FKView(QWidget):
    """
    FKView = ROIWindow 的 central widget（图像区域）
    ROIWindow 的 dock 会调用 create_controls() 来拿到控制面板（不会挤压图像）

    Changes (2026-01):
      - Add "Show dB" toggle (dB vs raw power).
      - Remove "Apply selection -> limits" checkbox (always True).
      - Remove "Clear selection" button.
      - Selection is now: rectangle -> set limits immediately -> then auto-clear selection.
      - Colorbar label follows the display mode.
    """
    fk_selection_changed = Signal(tuple)  # (kmin, kmax, fmin, fmax) analysis selection

    def __init__(self, roi_data, parent=None):
        super().__init__(parent)
        self.roi_data = roi_data

        # ---- display mode ----
        # True: show dB, False: show raw power
        self._show_db = True

        # 1) 从 roi_data 取数据与采样
        data = np.asarray(roi_data.get("data"))
        nt, nc = data.shape
        fs = float(roi_data.get("fs") or 1.0)
        dt = 1 / fs
        dx = float(roi_data.get("dx") or 1.0)

        # 2) 计算 f-k（返回 power spectrum, f_axis, k_axis）
        spectrum, f_axis, k_axis = fifk.fk_transform(
            data=normlized(data, True),
            dt=dt,
            dx=dx,
            backend="scipy",
            return_power=True,
            shift=True,
        )

        # 转成 numpy（以防 backend 不是 numpy）
        spectrum = np.asarray(spectrum)
        f_axis = np.asarray(f_axis)
        k_axis = np.asarray(k_axis)

        # 3) 只保留正频（更常用、更“干净”）
        pos = f_axis >= 0
        self.f_axis = f_axis[pos]
        self.k_axis = k_axis

        # ---- store both raw & display ----
        self.fk_raw = spectrum[pos, :]          # raw power
        self.fk = self._to_display(self.fk_raw) # display (dB or raw)

        # 4) extent（用于 imshow 的坐标映射）
        self.k_extent = (float(self.k_axis.min()), float(self.k_axis.max()))
        self.f_extent = (float(self.f_axis.min()), float(self.f_axis.max()))

        # 速度线
        self._vel_lines = []
        self._vel_values = [500.0, 1500.0, 3000.0]
        self._vel_lines_visible = False

        # 选择器
        self._selector: RectangleSelector | None = None
        self._selection = None  # (kmin,kmax,fmin,fmax)

        # 5) Matplotlib 画布
        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self._im = None
        self._cbar = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self._plot_initial()

    # =========================================================
    # ROIWindow(A) 会调用：create_controls()
    # =========================================================
    def create_controls(self, parent=None) -> QWidget:
        root = QWidget(parent)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea(root)
        scroll.setWidgetResizable(True)
        root_layout.addWidget(scroll)

        panel = QWidget()
        scroll.setWidget(panel)

        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(10)

        # --- Group 0: Spectrum Display ---
        g0 = QGroupBox("Spectrum Display")
        f0 = QFormLayout(g0)

        self.db_cb = QCheckBox("Show dB (log scale)")
        self.db_cb.setChecked(True)
        self.db_cb.toggled.connect(self.set_show_db)
        f0.addRow(self.db_cb)

        vbox.addWidget(g0)

        # --- Group 1: Appearance (cmap/clim) ---
        g1 = QGroupBox("Appearance")
        f1 = QFormLayout(g1)

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["cividis", "viridis", "magma", "plasma", "gray"])
        self.cmap_combo.setCurrentText("cividis")
        self.cmap_combo.currentTextChanged.connect(self.set_cmap)
        f1.addRow("cmap", self.cmap_combo)

        self.vmin_spin = self._spin()
        self.vmax_spin = self._spin()
        vmin, vmax = self._robust_minmax(self.fk)
        self.vmin_spin.setValue(vmin)
        self.vmax_spin.setValue(vmax)
        f1.addRow("vmin", self.vmin_spin)
        f1.addRow("vmax", self.vmax_spin)

        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        btn_apply = QPushButton("Apply")
        btn_auto = QPushButton("Auto")
        btn_apply.clicked.connect(lambda: self.set_clim(self.vmin_spin.value(), self.vmax_spin.value()))
        btn_auto.clicked.connect(self.autoscale_clim)
        h.addWidget(btn_apply)
        h.addWidget(btn_auto)
        h.addStretch(1)
        f1.addRow("clim", row)

        vbox.addWidget(g1)

        # --- Group 2: View Range (axis limits) ---
        g2 = QGroupBox("View Range (axis)")
        f2 = QFormLayout(g2)

        self.kmin_spin = self._spin()
        self.kmax_spin = self._spin()
        self.fmin_spin = self._spin()
        self.fmax_spin = self._spin()

        k0, k1 = self.k_extent
        f0_, f1_ = self.f_extent
        self.kmin_spin.setValue(k0)
        self.kmax_spin.setValue(k1)
        self.fmin_spin.setValue(f0_)
        self.fmax_spin.setValue(f1_)

        f2.addRow("k min", self.kmin_spin)
        f2.addRow("k max", self.kmax_spin)
        f2.addRow("f min", self.fmin_spin)
        f2.addRow("f max", self.fmax_spin)

        row2 = QWidget()
        h2 = QHBoxLayout(row2)
        h2.setContentsMargins(0, 0, 0, 0)
        btn_apply_lim = QPushButton("Apply")
        btn_reset_lim = QPushButton("Reset")
        btn_apply_lim.clicked.connect(self._apply_limits_from_ui)
        btn_reset_lim.clicked.connect(self.reset_limits)
        h2.addWidget(btn_apply_lim)
        h2.addWidget(btn_reset_lim)
        h2.addStretch(1)
        f2.addRow("limits", row2)

        vbox.addWidget(g2)

        # --- Group 3: Selection (fk-domain) ---
        # Removed:
        #   - "Apply selection -> limits" (always True now)
        #   - "Clear selection" button (auto-clear after apply)
        g3 = QGroupBox("Selection (fk-domain)")
        f3 = QFormLayout(g3)

        self.sel_cb = QCheckBox("Enable rectangle selection")
        self.sel_cb.toggled.connect(self.enable_fk_selector)
        f3.addRow(self.sel_cb)

        vbox.addWidget(g3)

        # --- Group 4: Velocity Lines (FK-specific) ---
        g4 = QGroupBox("Velocity Lines")
        f4 = QFormLayout(g4)

        self.vel_cb = QCheckBox("Show velocity lines")
        self.vel_cb.toggled.connect(self.set_velocity_lines_visible)
        f4.addRow(self.vel_cb)

        # 用输入框：逗号分隔
        self.vel_edit = QLineEdit()
        self.vel_edit.setPlaceholderText("e.g. 500,1500,3000")
        self.vel_edit.setText(",".join(str(int(v)) if float(v).is_integer() else str(v) for v in self._vel_values))
        f4.addRow("velocities", self.vel_edit)

        btn_apply_vel = QPushButton("Apply velocities")
        btn_apply_vel.clicked.connect(self._apply_velocities_from_ui)
        f4.addRow(btn_apply_vel)

        vbox.addWidget(g4)

        vbox.addStretch(1)

        # 初始把 clim 应用到图上
        self.set_clim(self.vmin_spin.value(), self.vmax_spin.value())
        self._update_colorbar_label()
        return root

    # =========================================================
    # 绘图与更新
    # =========================================================
    def _plot_initial(self):
        self.ax.clear()
        self._im = self.ax.imshow(
            self.fk,
            origin="lower",
            aspect="auto",
            extent=[self.k_extent[0], self.k_extent[1], self.f_extent[0], self.f_extent[1]],
            cmap="cividis",
        )
        self.ax.set_title("f-k")
        self.ax.set_xlabel("k")
        self.ax.set_ylabel("f (Hz)")

        # 一次性 colorbar（避免重复叠加）
        self._cbar = self.figure.colorbar(self._im, ax=self.ax, shrink=0.85)
        self._update_colorbar_label()

        # 初始范围
        self.ax.set_xlim(self.k_extent)
        self.ax.set_ylim(self.f_extent)

        self.canvas.draw_idle()

    # =========================================================
    # Spectrum display mode (dB/raw)
    # =========================================================
    def set_show_db(self, enabled: bool):
        enabled = bool(enabled)
        if enabled == self._show_db:
            return
        self._show_db = enabled

        # refresh display array
        self.fk = self._to_display(self.fk_raw)

        # update image
        if self._im is not None:
            self._im.set_data(self.fk)

            # if user has not touched clim, auto-update to robust range
            vmin, vmax = self._robust_minmax(self.fk)
            self._im.set_clim(vmin, vmax)

            if hasattr(self, "vmin_spin"):
                self.vmin_spin.blockSignals(True)
                self.vmax_spin.blockSignals(True)
                self.vmin_spin.setValue(vmin)
                self.vmax_spin.setValue(vmax)
                self.vmin_spin.blockSignals(False)
                self.vmax_spin.blockSignals(False)

        self._update_colorbar_label()
        self.canvas.draw_idle()

    def _to_display(self, power_spectrum: np.ndarray) -> np.ndarray:
        """
        power_spectrum: raw power (>=0), usually abs(FFT)**2
        """
        p = np.asarray(power_spectrum)
        if self._show_db:
            # Power in dB: 10*log10(P)
            eps = 1e-12 * float(np.max(p)) + 1e-12
            return 10.0 * np.log10(p + eps)
        return p

    def _update_colorbar_label(self):
        if self._cbar is None:
            return
        self._cbar.set_label("Power (dB)" if self._show_db else "Power")

    # =========================================================
    # Appearance controls
    # =========================================================
    def set_cmap(self, cmap: str):
        if self._im is None:
            return
        self._im.set_cmap(cmap)
        self.canvas.draw_idle()

    def set_clim(self, vmin: float, vmax: float):
        if self._im is None:
            return
        vmin, vmax = float(vmin), float(vmax)
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        self._im.set_clim(vmin, vmax)
        self.canvas.draw_idle()

    def autoscale_clim(self):
        if self._im is None:
            return
        vmin, vmax = self._robust_minmax(self.fk)
        self._im.set_clim(vmin, vmax)
        if hasattr(self, "vmin_spin"):
            self.vmin_spin.setValue(vmin)
            self.vmax_spin.setValue(vmax)
        self.canvas.draw_idle()

    # =========================================================
    # View range controls (axis limits)
    # =========================================================
    def _apply_limits_from_ui(self):
        kmin = float(self.kmin_spin.value())
        kmax = float(self.kmax_spin.value())
        fmin = float(self.fmin_spin.value())
        fmax = float(self.fmax_spin.value())
        self.set_limits(kmin, kmax, fmin, fmax)

    def set_limits(self, kmin: float, kmax: float, fmin: float, fmax: float):
        if kmax < kmin:
            kmin, kmax = kmax, kmin
        if fmax < fmin:
            fmin, fmax = fmax, fmin
        self.ax.set_xlim(kmin, kmax)
        self.ax.set_ylim(fmin, fmax)
        if self._vel_lines_visible:
            self._draw_velocity_lines()
        self.canvas.draw_idle()

    def reset_limits(self):
        self.ax.set_xlim(self.k_extent[0], self.k_extent[1])
        self.ax.set_ylim(self.f_extent[0], self.f_extent[1])
        if hasattr(self, "kmin_spin"):
            self.kmin_spin.setValue(self.k_extent[0])
            self.kmax_spin.setValue(self.k_extent[1])
            self.fmin_spin.setValue(self.f_extent[0])
            self.fmax_spin.setValue(self.f_extent[1])
        if self._vel_lines_visible:
            self._draw_velocity_lines()
        self.canvas.draw_idle()

    # =========================================================
    # FK selection (RectangleSelector)
    # =========================================================
    def enable_fk_selector(self, enabled: bool):
        if enabled:
            if self._selector is None:
                self._selector = RectangleSelector(
                    self.ax,
                    onselect=self._on_rect_select,
                    useblit=True,
                    button=[1],
                    interactive=True,
                    drag_from_anywhere=True,
                )
            self._selector.set_active(True)
        else:
            if self._selector is not None:
                self._selector.set_active(False)

    def _on_rect_select(self, eclick, erelease):
        if (
            eclick.xdata is None or eclick.ydata is None or
            erelease.xdata is None or erelease.ydata is None
        ):
            return

        k0, f0 = float(eclick.xdata), float(eclick.ydata)
        k1, f1 = float(erelease.xdata), float(erelease.ydata)

        kmin, kmax = (k0, k1) if k0 <= k1 else (k1, k0)
        fmin, fmax = (f0, f1) if f0 <= f1 else (f1, f0)

        # 1) 存起来（仍然保留，便于后续做统计/滤波）
        self._selection = (kmin, kmax, fmin, fmax)

        # 2) 写回 UI（selection 设置 k/f 范围）
        if hasattr(self, "kmin_spin"):
            self.kmin_spin.setValue(kmin)
            self.kmax_spin.setValue(kmax)
            self.fmin_spin.setValue(fmin)
            self.fmax_spin.setValue(fmax)

        # 3) 默认行为：立刻 apply selection -> limits（总是 True）
        self.set_limits(kmin, kmax, fmin, fmax)

        # 4) 发信号（以后做 filter 或统计都用这个）
        self.fk_selection_changed.emit((kmin, kmax, fmin, fmax))

        # 5) 默认：apply 后自动 clear selection（不再需要 Clear 按钮）
        self.clear_selection()

    def clear_selection(self):
        self._selection = None
        if self._selector is not None:
            # 清空可视化框：设成退化框（Matplotlib selector 的简易清除方式）
            try:
                self._selector.extents = (0, 0, 0, 0)
            except Exception:
                pass
        self.canvas.draw_idle()

    # =========================================================
    # Velocity lines (FK-specific)
    # =========================================================
    def _apply_velocities_from_ui(self):
        text = self.vel_edit.text().strip() if hasattr(self, "vel_edit") else ""
        vels = self._parse_velocities(text)
        if not vels:
            # 空输入就不更新（或你也可以清空速度线）
            return
        self.set_velocity_lines(vels)

    def set_velocity_lines(self, velocities: list[float]):
        self._vel_values = [float(v) for v in velocities if float(v) > 0]
        if self._vel_lines_visible:
            self._draw_velocity_lines()
            self.canvas.draw_idle()

    def set_velocity_lines_visible(self, visible: bool):
        self._vel_lines_visible = bool(visible)
        if self._vel_lines_visible:
            self._draw_velocity_lines()
        else:
            self._remove_velocity_lines()
        self.canvas.draw_idle()

    def _remove_velocity_lines(self):
        for ln in self._vel_lines:
            try:
                ln.remove()
            except Exception:
                pass
        self._vel_lines = []

    def _draw_velocity_lines(self):
        self._remove_velocity_lines()

        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        ymin, ymax = min(y0, y1), max(y0, y1)

        # 注意：这里用 f = v * |k|
        # 如果你未来把 k 轴改成 rad/m，需要改成 f = v * |k| / (2π)
        k = np.linspace(x0, x1, 400)
        kk = np.abs(k)

        for v in self._vel_values:
            f = v * kk
            mask = (f >= ymin) & (f <= ymax)
            if not np.any(mask):
                continue
            ln, = self.ax.plot(k[mask], f[mask], linewidth=1.0, alpha=0.7)
            self._vel_lines.append(ln)

    # =========================================================
    # Helpers
    # =========================================================
    def _spin(self) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setDecimals(6)
        s.setRange(-1e12, 1e12)
        s.setSingleStep(0.1)
        s.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        s.setMinimumWidth(120)
        return s

    def _robust_minmax(self, arr: np.ndarray, p_low=1.0, p_high=99.0):
        a = np.asarray(arr)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return 0.0, 1.0
        lo = float(np.percentile(a, p_low))
        hi = float(np.percentile(a, p_high))
        if hi == lo:
            hi = lo + 1e-6
        return lo, hi

    def _parse_velocities(self, text: str) -> list[float]:
        # 支持：空格、逗号、分号
        if not text:
            return []
        parts = [p.strip() for p in text.replace(";", ",").split(",")]
        out = []
        for p in parts:
            if not p:
                continue
            try:
                v = float(p)
                if v > 0:
                    out.append(v)
            except ValueError:
                continue
        # 去重并保持顺序
        dedup = []
        seen = set()
        for v in out:
            if v not in seen:
                dedup.append(v)
                seen.add(v)
        return dedup
