# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""Navigator window (View ▸ Navigator…): pyqtgraph image linked to the streaming canvas.

The streaming canvas stays a constant-cost, sample-faithful viewer.  This
window shows a *bounded full-read* overview of one time range (default: the
file under the current view): every sample is filtered at the original rate
and aggregated per bin (signed mean = amplitude, or RMS / peak in dB), so the
image is not an aliased decimation.  The two views are linked:

* clicking the overview moves the streaming canvas to that time/channel;
* the streaming canvas' visible window is drawn as a rectangle here.

Layout: one toolbar row of summary buttons that open small popovers —
Range, Statistic, Resolution, Filter (need Compute) │ Color (live) — and
the image filling the rest.  The view can only zoom *into* the computed
image (left-drag = box zoom, wheel = zoom); double-click returns to it.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPoint, QRectF, QSignalBlocker, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QDoubleValidator, QIcon, QImage, QKeySequence, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGraphicsRectItem,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from filark._exceptions import print_exception
from filark.explore.overview import (
    OverviewResult,
    compute_overview,
    default_bins,
    estimate_overview_bytes,
    file_range_at,
)
from filark.utils.time import tape_sample_to_unix

__all__ = ["OverviewWindow"]

CONFIRM_SECONDS = 30 * 60  # ask before reading more than 30 min of data
GAP_BRUSH = QColor(128, 128, 140, 170)
VIEW_PEN_COLOR = QColor(255, 140, 0)
AUTO_PCT = (2.0, 99.7)  # dB images
AUTO_ABS_PCT = 99.0  # amplitude images: symmetric ±p99(|x|)
DEFAULT_CMAP = {"mean": "seismic", "rms": "viridis", "maxabs": "viridis"}
CMAPS = ["viridis", "magma", "inferno", "plasma", "turbo", "cividis", "gray", "seismic", "RdBu_r", "coolwarm"]
AUTO_RES = (1920, 1080)  # Auto bin size: at most this many bins (cols × rows)
CONFIRM_IMAGE_BYTES = 1 << 30  # ask before allocating more than 1 GiB of bins
STATS = [("Amplitude", "mean"), ("RMS", "rms"), ("Peak", "maxabs")]
HINT = "Drag to zoom · double-click to reset · click to move the streaming view there."


def _layout_nt_nc(tape) -> tuple[int, int]:
    a, b = int(tape.shape[0]), int(tape.shape[1])
    return (a, b) if getattr(tape, "dims", "nt_nc") == "nt_nc" else (b, a)


class _TimeAxis(pg.AxisItem):
    """Bottom axis in logical samples, labelled with (gap-aware) UTC or seconds."""

    def __init__(self, tape, fs: float, **kw):
        super().__init__("bottom", **kw)
        self._tape = tape
        self._fs = float(fs) or 1.0
        self._utc = getattr(tape, "start_time", None) is not None

    def tickStrings(self, values, scale, spacing):  # noqa: N802 (Qt naming)
        out = []
        step_s = float(spacing) / self._fs
        for v in values:
            if self._utc:
                t = tape_sample_to_unix(self._tape, float(v))
                dt = _dt.datetime.fromtimestamp(float(t), tz=_dt.timezone.utc)
                if step_s < 1.0:
                    out.append(dt.strftime("%H:%M:%S.") + f"{dt.microsecond // 1000:03d}")
                else:
                    out.append(dt.strftime("%H:%M:%S"))
            else:
                out.append(f"{float(v) / self._fs:.4g}")
        return out


class _OverviewWorker(QThread):
    progress = Signal(int, int)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, kwargs: dict, parent=None):
        super().__init__(parent)
        self._kwargs = kwargs

    def run(self) -> None:
        try:
            res = compute_overview(
                **self._kwargs,
                progress_cb=lambda d, t: self.progress.emit(int(d), int(t)),
                cancel_cb=lambda: self.isInterruptionRequested(),
            )
            self.done.emit(res)
        except Exception as exc:  # reported in the window
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class _Popover(QToolButton):
    """Toolbar button showing a one-line summary; clicking opens ``panel`` below it."""

    def __init__(self, panel: QWidget, tooltip: str = "", parent=None):
        super().__init__(parent)
        self.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.setToolTip(tooltip)
        self.setCursor(Qt.PointingHandCursor)
        self._popup = QFrame(self, Qt.Popup)
        self._popup.setFrameShape(QFrame.StyledPanel)
        lay = QVBoxLayout(self._popup)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.addWidget(panel)
        self.clicked.connect(self.open)

    def setSummary(self, text: str) -> None:  # noqa: N802 (Qt naming)
        self.setText(f"{text}  ▾")

    def open(self) -> None:
        self._popup.adjustSize()
        pos = self.mapToGlobal(QPoint(0, self.height() + 2))
        screen = self.screen().availableGeometry()
        pos.setX(min(pos.x(), screen.right() - self._popup.width()))
        self._popup.move(pos)
        self._popup.show()

    def close_popup(self) -> None:
        self._popup.hide()


class OverviewWindow(QMainWindow):
    """Bounded full-read overview of one time range, linked to ``display_model``."""

    def __init__(
        self,
        tape,
        display_model,
        *,
        start: int,
        stop: int,
        filter_params: Optional[dict] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setWindowTitle("Navigator")
        self._tape = tape
        self._dm = display_model
        self._fs = float(getattr(tape, "fs", None) or 1.0)
        self._nt, self._nc = _layout_nt_nc(tape)
        self._worker: Optional[_OverviewWorker] = None
        self._result: Optional[OverviewResult] = None
        self._gap_items: list = []
        self._shown: Optional[np.ndarray] = None  # image as displayed (amplitude or dB)
        self._shown_stat: Optional[str] = None
        self._syncing_levels = False

        self._build_ui(filter_params)
        self._set_range(int(start), int(stop))

        self._follow = QTimer(self)
        self._follow.setInterval(100)  # ≤10 Hz viewport-rectangle updates
        self._follow.timeout.connect(self._update_view_rect)
        self._follow.start()

        QTimer.singleShot(0, self.compute)

    # ------------------------------------------------------------------ UI
    def _build_ui(self, filter_params: Optional[dict]) -> None:
        # ---- Range popover
        self.sp_start = QDoubleSpinBox()
        self.sp_start.setDecimals(3)
        self.sp_start.setSuffix(" s")
        self.sp_start.setRange(0.0, max(0.0, (self._nt - 1) / self._fs))
        self.sp_start.setToolTip("Elapsed seconds on the logical timeline.")
        self.sp_dur = QDoubleSpinBox()
        self.sp_dur.setDecimals(3)
        self.sp_dur.setSuffix(" s")
        self.sp_dur.setRange(1.0 / self._fs, max(1.0 / self._fs, self._nt / self._fs))
        self.btn_file = QPushButton("Current file")
        self.btn_file.setToolTip("The file under the streaming canvas' current view.")
        self.btn_all = QPushButton("Whole recording")
        self.btn_all.setToolTip("The whole source, including all files and gaps.")
        p_range = QWidget()
        f = QFormLayout(p_range)
        f.setContentsMargins(0, 0, 0, 0)
        f.addRow(self._hrow(self.btn_file, self.btn_all))
        f.addRow("Start", self.sp_start)
        f.addRow("Duration", self.sp_dur)
        self.pop_range = _Popover(p_range, "Time range to read")

        # ---- Statistic (segmented, no popover)
        self.grp_stat = QButtonGroup(self)
        self.grp_stat.setExclusive(True)
        seg = QWidget()
        sh = QHBoxLayout(seg)
        sh.setContentsMargins(0, 0, 0, 0)
        sh.setSpacing(0)
        tips = {
            "mean": "Signed bin mean — the ordinary amplitude image",
            "rms": "RMS energy per bin, in dB",
            "maxabs": "Peak |x| per bin, in dB",
        }
        for k, (label, key) in enumerate(STATS):
            b = QToolButton()
            b.setText(label)
            b.setCheckable(True)
            b.setToolTip(tips[key])
            b.setProperty("stat", key)
            b.setObjectName("seg")
            b.setChecked(k == 0)
            self.grp_stat.addButton(b, k)
            sh.addWidget(b)

        # ---- Resolution popover
        self.chk_auto_bin = QCheckBox(f"Auto (≈ {AUTO_RES[0]} × {AUTO_RES[1]})")
        self.chk_auto_bin.setChecked(True)
        self.chk_auto_bin.setToolTip("Smallest bins that keep the image within this size; follows the range.")
        self.btn_full_res = QPushButton("Full (every sample)")
        self.sp_tbin = QSpinBox()
        self.sp_tbin.setRange(1, max(1, self._nt))
        self.sp_tbin.setSuffix(" samples")
        self.sp_cbin = QSpinBox()
        self.sp_cbin.setRange(1, max(1, self._nc))
        self.sp_cbin.setSuffix(" ch")
        self.lbl_bins = QLabel("")
        self.lbl_bins.setObjectName("hint")
        p_res = QWidget()
        f = QFormLayout(p_res)
        f.setContentsMargins(0, 0, 0, 0)
        f.addRow(self._hrow(self.chk_auto_bin, self.btn_full_res))
        f.addRow("Time bin", self.sp_tbin)
        f.addRow("Channel bin", self.sp_cbin)
        f.addRow(self.lbl_bins)
        self.pop_res = _Popover(p_res, "Bin size: how many samples / channels each image pixel aggregates")

        # ---- Filter popover
        self.chk_filter = QCheckBox("Enable")
        self.chk_filter.setToolTip("Causal Butterworth filter at the original sampling rate, before binning.")
        self.cb_kind = QComboBox()
        self.cb_kind.addItems(["bandpass", "lowpass", "highpass"])
        nyq = 0.5 * self._fs
        self.sp_fmin = QDoubleSpinBox()
        self.sp_fmax = QDoubleSpinBox()
        for sp in (self.sp_fmin, self.sp_fmax):
            sp.setDecimals(3)
            sp.setRange(0.0, nyq * 0.999)
            sp.setSuffix(" Hz")
        self.sp_fmin.setValue(min(5.0, nyq * 0.4))
        self.sp_fmax.setValue(min(50.0, nyq * 0.9))
        self.sp_order = QSpinBox()
        self.sp_order.setRange(1, 12)
        self.sp_order.setValue(4)
        if filter_params:
            self.chk_filter.setChecked(True)
            self.cb_kind.setCurrentText(str(filter_params.get("kind", "bandpass")))
            if filter_params.get("fmin") is not None:
                self.sp_fmin.setValue(float(filter_params["fmin"]))
            if filter_params.get("fmax") is not None:
                self.sp_fmax.setValue(float(filter_params["fmax"]))
            self.sp_order.setValue(int(filter_params.get("order", 4)))
        p_filter = QWidget()
        f = QFormLayout(p_filter)
        f.setContentsMargins(0, 0, 0, 0)
        f.addRow(self.chk_filter)
        f.addRow("Type", self.cb_kind)
        f.addRow("fmin", self.sp_fmin)
        f.addRow("fmax", self.sp_fmax)
        f.addRow("Order", self.sp_order)
        self.pop_filter = _Popover(p_filter, "Filter applied before binning")

        # ---- Color popover (live, no recompute)
        self.cb_cmap = QComboBox()
        self.cb_cmap.addItems(CMAPS)
        self.cb_cmap.setCurrentText(DEFAULT_CMAP[self._stat()])
        self.chk_auto = QCheckBox("Auto limits")
        self.chk_auto.setChecked(True)
        self.chk_auto.setToolTip(
            f"Amplitude: ±{AUTO_ABS_PCT:g}th percentile of |x|. dB: {AUTO_PCT[0]:g}–{AUTO_PCT[1]:g} % percentiles.\n"
            "Dragging the colour bar switches to manual."
        )
        self.sp_vmin = QLineEdit()
        self.sp_vmax = QLineEdit()
        for ed in (self.sp_vmin, self.sp_vmax):
            ed.setValidator(QDoubleValidator(ed))
        p_color = QWidget()
        f = QFormLayout(p_color)
        f.setContentsMargins(0, 0, 0, 0)
        f.addRow("Colormap", self.cb_cmap)
        f.addRow(self.chk_auto)
        f.addRow("vmin", self.sp_vmin)
        f.addRow("vmax", self.sp_vmax)
        self.pop_color = _Popover(p_color, "Colormap and colour limits (applied immediately)")
        self.pop_color.setIconSize(QSize(36, 12))

        # ---- Compute (doubles as Cancel + progress)
        self.btn_compute = QPushButton("Compute")
        self.btn_compute.setDefault(True)
        self.btn_compute.setMinimumWidth(130)
        self.btn_compute.setToolTip("Read the range and build the image (Ctrl/Cmd+Return)")

        tb = QToolBar("Navigator", self)
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setContextMenuPolicy(Qt.PreventContextMenu)
        tb.addWidget(self.pop_range)
        tb.addWidget(seg)
        tb.addWidget(self.pop_res)
        tb.addWidget(self.pop_filter)
        tb.addSeparator()
        tb.addWidget(self.pop_color)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)
        tb.addWidget(self.btn_compute)
        self.addToolBar(Qt.TopToolBarArea, tb)
        self.setUnifiedTitleAndToolBarOnMac(True)

        # ---- plot
        bg = self.palette().window().color()
        self.plot = pg.PlotWidget(background=bg, axisItems={"bottom": _TimeAxis(self._tape, self._fs)})
        self.plot.setLabel("left", "Channel")
        self.plot.setLabel("bottom", "UTC" if getattr(self._tape, "start_time", None) is not None else "Time (s)")
        self.plot.showGrid(x=False, y=False)
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        vb = self.plot.getPlotItem().getViewBox()
        vb.setMouseMode(pg.ViewBox.RectMode)  # left-drag = box zoom
        self.img = pg.ImageItem()
        self.plot.addItem(self.img)
        stat0 = self._stat()
        self.cbar = pg.ColorBarItem(
            values=(0, 1), colorMap=self._colormap(DEFAULT_CMAP[stat0]),
            label="Amplitude" if stat0 == "mean" else "dB", interactive=True,
        )
        self.cbar.setImageItem(self.img, insert_in=self.plot.getPlotItem())

        self.view_rect = QGraphicsRectItem()
        pen = QPen(VIEW_PEN_COLOR)
        pen.setWidthF(2.0)
        pen.setCosmetic(True)
        self.view_rect.setPen(pen)
        self.view_rect.setZValue(20)
        self.view_rect.setVisible(False)
        self.plot.addItem(self.view_rect)
        self.setCentralWidget(self.plot)

        # ---- status bar: left = hover readout / messages, right = read size
        self.lbl_status = QLabel(HINT)
        self.lbl_reads = QLabel("")
        self.lbl_reads.setObjectName("hint")
        sb = self.statusBar()
        sb.setSizeGripEnabled(False)
        sb.addWidget(self.lbl_status, 1)
        sb.addPermanentWidget(self.lbl_reads)

        self.setStyleSheet(
            "QToolBar { spacing: 6px; padding: 4px 6px; }"
            "QToolButton#seg { padding: 3px 10px; }"
            "QLabel#hint { color: palette(placeholder-text); }"
        )
        self.resize(1300, 780)

        self._click_timer = QTimer(self)
        self._click_timer.setSingleShot(True)
        self._click_timer.setInterval(QApplication.doubleClickInterval())
        self._click_timer.timeout.connect(self._navigate_pending)
        self._pending_click: Optional[tuple[float, float]] = None

        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self.compute)
        self.btn_compute.clicked.connect(self._on_compute_clicked)
        self.btn_file.clicked.connect(self._use_current_file)
        self.btn_all.clicked.connect(self._use_whole_recording)
        self.cb_kind.currentTextChanged.connect(self._sync_filter_widgets)
        self.chk_filter.toggled.connect(self._sync_filter_widgets)
        for w in (self.sp_fmin, self.sp_fmax, self.sp_order):
            w.valueChanged.connect(self._sync_filter_widgets)
        self.sp_start.valueChanged.connect(self._on_range_edited)
        self.sp_dur.valueChanged.connect(self._on_range_edited)
        self.sp_tbin.valueChanged.connect(self._update_bins_label)
        self.sp_cbin.valueChanged.connect(self._update_bins_label)
        self.chk_auto_bin.toggled.connect(self._on_auto_bin_toggled)
        self.btn_full_res.clicked.connect(self._use_full_resolution)
        self.cb_cmap.currentTextChanged.connect(self._on_cmap_changed)
        self.chk_auto.toggled.connect(self._on_auto_toggled)
        self.sp_vmin.editingFinished.connect(self._on_levels_edited)
        self.sp_vmax.editingFinished.connect(self._on_levels_edited)
        self.cbar.sigLevelsChangeFinished.connect(self._on_cbar_dragged)
        self.plot.scene().sigMouseClicked.connect(self._on_click)
        self.plot.scene().sigMouseMoved.connect(self._on_move)
        self._sync_filter_widgets()
        self._on_auto_toggled(True)
        self._update_color_summary()
        self.sp_tbin.setEnabled(False)  # auto bins; values follow the range (_update_range_label)
        self.sp_cbin.setEnabled(False)

    def _stat(self) -> str:
        return str(self.grp_stat.checkedButton().property("stat"))

    @staticmethod
    def _hrow(*widgets) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        for x in widgets:
            h.addWidget(x)
        return w

    @staticmethod
    def _colormap(name: str):
        try:
            return pg.colormap.get(name)
        except Exception:
            return pg.colormap.get(name, source="matplotlib")

    def _sync_filter_widgets(self, *_):
        on = self.chk_filter.isChecked()
        kind = self.cb_kind.currentText()
        self.cb_kind.setEnabled(on)
        self.sp_order.setEnabled(on)
        self.sp_fmin.setEnabled(on and kind != "lowpass")
        self.sp_fmax.setEnabled(on and kind != "highpass")
        if not on:
            text = "Filter: off"
        elif kind == "lowpass":
            text = f"LP {self.sp_fmax.value():g} Hz"
        elif kind == "highpass":
            text = f"HP {self.sp_fmin.value():g} Hz"
        else:
            text = f"BP {self.sp_fmin.value():g}–{self.sp_fmax.value():g} Hz"
        self.pop_filter.setSummary(text)

    # ------------------------------------------------------------- display
    def _on_cmap_changed(self, name: str) -> None:
        self.cbar.setColorMap(self._colormap(name))
        self._update_color_summary()

    def _update_color_summary(self) -> None:
        cmap = self._colormap(self.cb_cmap.currentText())
        lut = cmap.getLookupTable(nPts=64, alpha=False).astype(np.uint8)
        row = np.ascontiguousarray(np.repeat(lut[None, :, :], 12, axis=0))
        qimg = QImage(row.data, row.shape[1], row.shape[0], 3 * row.shape[1], QImage.Format_RGB888).copy()
        self.pop_color.setIcon(QIcon(QPixmap.fromImage(qimg).scaled(36, 12)))
        if self.chk_auto.isChecked() or self._result is None:
            text = "Auto"
        else:
            lo, hi = self.cbar.levels()
            text = f"±{hi:.3g}" if np.isclose(lo, -hi) else f"{lo:.3g} … {hi:.3g}"
        self.pop_color.setSummary(text)

    def _on_auto_toggled(self, on: bool) -> None:
        self.sp_vmin.setEnabled(not on)
        self.sp_vmax.setEnabled(not on)
        if on:
            self._apply_auto_levels()
        self._update_color_summary()

    def _apply_auto_levels(self) -> None:
        res = self._result
        if self._shown is None or res is None:
            return
        finite = self._shown[np.isfinite(self._shown)]
        if res.stat == "mean":
            a = float(np.percentile(np.abs(finite), AUTO_ABS_PCT)) if finite.size else 0.0
            if a <= 0:
                a = float(np.max(np.abs(finite))) if finite.size else 0.0
            a = a or 1.0
            lo, hi = -a, a
        else:
            lo, hi = (np.percentile(finite, AUTO_PCT) if finite.size else (0.0, 1.0))
            if hi <= lo:
                hi = lo + 1.0
        self._set_levels(float(lo), float(hi))

    def _on_levels_edited(self, *_):
        if self._syncing_levels or self.chk_auto.isChecked():
            return
        lim = self._typed_levels()
        if lim is not None:
            self._set_levels(*lim)

    def _typed_levels(self) -> Optional[tuple[float, float]]:
        try:
            lo, hi = float(self.sp_vmin.text()), float(self.sp_vmax.text())
        except ValueError:
            return None
        return (lo, hi) if np.isfinite(lo) and np.isfinite(hi) and hi > lo else None

    def _on_cbar_dragged(self, *_):
        if self._syncing_levels:
            return
        lo, hi = self.cbar.levels()
        with QSignalBlocker(self.chk_auto):
            self.chk_auto.setChecked(False)
        self.sp_vmin.setEnabled(True)
        self.sp_vmax.setEnabled(True)
        self._set_levels(float(lo), float(hi))

    def _set_levels(self, lo: float, hi: float) -> None:
        self._syncing_levels = True
        try:
            self.sp_vmin.setText(f"{lo:.6g}")
            self.sp_vmax.setText(f"{hi:.6g}")
            self.cbar.setLevels((lo, hi))
        finally:
            self._syncing_levels = False
        self._update_color_summary()

    # --------------------------------------------------------------- range
    def _set_range(self, start: int, stop: int) -> None:
        """Set the range in samples (exact); the spin boxes only display it."""
        start = int(np.clip(start, 0, max(0, self._nt - 1)))
        stop = int(np.clip(stop, start + 1, self._nt))
        self._range = (start, stop)
        self._syncing = True
        try:
            self.sp_start.setValue(start / self._fs)
            self.sp_dur.setValue((stop - start) / self._fs)
        finally:
            self._syncing = False
        self._update_range_label()

    def _on_range_edited(self, *_):
        if getattr(self, "_syncing", False):
            return
        start = int(round(self.sp_start.value() * self._fs))
        start = int(np.clip(start, 0, max(0, self._nt - 1)))
        stop = int(np.clip(start + int(round(self.sp_dur.value() * self._fs)), start + 1, self._nt))
        self._range = (start, stop)
        self._update_range_label()

    def _range_samples(self) -> tuple[int, int]:
        return self._range

    def _update_range_label(self, *_):
        start, stop = self._range_samples()
        mb = estimate_overview_bytes(self._tape, start, stop) / 1e6
        t = tape_sample_to_unix(self._tape, start)
        dur = self._fmt_dur((stop - start) / self._fs)
        if t is not None:
            dt = _dt.datetime.fromtimestamp(float(t), tz=_dt.timezone.utc)
            self.pop_range.setSummary(f"{dt:%H:%M:%S} · {dur}")
            self.pop_range.setToolTip(f"{dt:%Y-%m-%d %H:%M:%S} UTC, {dur}")
        else:
            self.pop_range.setSummary(f"{start / self._fs:g} s · {dur}")
        self.lbl_reads.setText(f"reads {mb / 1e3:,.2f} GB" if mb >= 1e3 else f"reads {mb:,.0f} MB")
        if self.chk_auto_bin.isChecked():
            self._apply_auto_bins()
        self._update_bins_label()

    # ---------------------------------------------------------------- bins
    def _on_auto_bin_toggled(self, on: bool) -> None:
        self.sp_tbin.setEnabled(not on)
        self.sp_cbin.setEnabled(not on)
        if on:
            self._apply_auto_bins()
        self._update_bins_label()

    def _apply_auto_bins(self) -> None:
        start, stop = self._range_samples()
        tb, cb = default_bins(stop - start, self._nc, *AUTO_RES)
        self.sp_tbin.setValue(tb)
        self.sp_cbin.setValue(cb)

    def _use_full_resolution(self) -> None:
        self.chk_auto_bin.setChecked(False)
        self.pop_res.close_popup()
        self.sp_tbin.setValue(1)
        self.sp_cbin.setValue(1)

    def _image_shape(self) -> tuple[int, int]:
        start, stop = self._range_samples()
        cols = -(-(stop - start) // self.sp_tbin.value())
        rows = -(-self._nc // self.sp_cbin.value())
        return rows, cols

    def _image_bytes(self) -> int:
        rows, cols = self._image_shape()
        return rows * cols * 16  # float64 accumulator + float32 image + float32 display copy

    def _update_bins_label(self, *_):
        rows, cols = self._image_shape()
        t_bin = self.sp_tbin.value() / self._fs
        txt = (
            f"{self._fmt_bin(t_bin)} × {self.sp_cbin.value()} ch per bin\n"
            f"image {cols:,} × {rows:,} ≈ {self._image_bytes() / 1e6:,.3g} MB"
        )
        if self.sp_tbin.value() > 1:
            # a boxcar of length T has its first null at 1/T: content above ~0.5/T averages out
            txt += f"\nAmplitude keeps < ~{0.5 / t_bin:.3g} Hz"
        self.lbl_bins.setText(txt)
        full = self.sp_tbin.value() == 1 and self.sp_cbin.value() == 1
        size = f"{self._fmt_bin(t_bin)} × {self.sp_cbin.value()} ch"
        if full:
            self.pop_res.setSummary("Full")
        elif self.chk_auto_bin.isChecked():
            self.pop_res.setSummary(f"Auto · {size}")
        else:
            self.pop_res.setSummary(size)

    @staticmethod
    def _fmt_bin(sec: float) -> str:
        if sec >= 60:
            return f"{sec / 60:.3g} min"
        if sec >= 1:
            return f"{sec:.3g} s"
        if sec >= 1e-3:
            return f"{sec * 1e3:.3g} ms"
        return f"{sec * 1e6:.3g} µs"

    def _use_current_file(self) -> None:
        b = self._view_bounds()
        center = (b["t0"] + b["t1"]) // 2 if b else 0
        self._set_range(*file_range_at(self._tape, center))
        self.pop_range.close_popup()
        self.compute()

    def _use_whole_recording(self) -> None:
        self._set_range(0, self._nt)
        self.pop_range.close_popup()
        self.compute()

    # ------------------------------------------------------------- compute
    def compute(self) -> None:
        if self._worker is not None:
            return
        start, stop = self._range_samples()
        warn = []
        if (stop - start) / self._fs > CONFIRM_SECONDS:
            mb = estimate_overview_bytes(self._tape, start, stop) / 1e6
            warn.append(f"The range is {(stop - start) / self._fs / 60:.1f} min and reads about {mb:,.0f} MB.")
        if self._image_bytes() > CONFIRM_IMAGE_BYTES:
            rows, cols = self._image_shape()
            warn.append(f"The image is {cols:,} × {rows:,} bins and needs about {self._image_bytes() / 1e9:.1f} GB.")
        if warn:
            ans = QMessageBox.question(self, "Large overview", "\n".join(warn) + "\nContinue?")
            if ans != QMessageBox.Yes:
                return
        band = None
        kind = None
        if self.chk_filter.isChecked():
            kind = self.cb_kind.currentText()
            fmin = self.sp_fmin.value() if kind != "lowpass" else None
            fmax = self.sp_fmax.value() if kind != "highpass" else None
            band = (fmin, fmax)
        kwargs: dict[str, Any] = dict(
            tape=self._tape, start=start, stop=stop,
            time_bin=int(self.sp_tbin.value()), channel_bin=int(self.sp_cbin.value()),
            stat=self._stat(), band=band, filter_kind=kind,
            order=int(self.sp_order.value()),
        )
        self._worker = _OverviewWorker(kwargs, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_worker_finished)
        self.btn_compute.setText("Cancel  0%")
        self.lbl_status.setText("Computing…")
        self._worker.start()

    def _on_compute_clicked(self) -> None:
        if self._worker is None:
            self.compute()
        else:
            self.cancel()

    def cancel(self) -> None:
        if self._worker is not None:
            self._worker.requestInterruption()
            self.btn_compute.setText("Cancelling…")

    def _on_progress(self, done: int, total: int) -> None:
        if self._worker is not None and not self._worker.isInterruptionRequested():
            self.btn_compute.setText(f"Cancel  {100 * done // max(1, total)}%")

    def _on_failed(self, msg: str) -> None:
        self.lbl_status.setText(msg if "cancelled" not in msg else "Cancelled.")

    def _on_worker_finished(self) -> None:
        self._worker = None
        self.btn_compute.setText("Compute")

    def _on_done(self, res: OverviewResult) -> None:
        self._result = res
        prev = self._shown_stat or STATS[0][1]
        self._shown = self._display_values(res.image, res.stat)
        is_db = res.stat != "mean"
        self.cbar.getAxis("left").setLabel("dB" if is_db else "Amplitude")
        self._shown_stat = res.stat
        if (prev == "mean") != (res.stat == "mean"):
            # amplitude <-> dB: old limits and colour map no longer make sense
            self.cb_cmap.setCurrentText(DEFAULT_CMAP[res.stat])
            self.chk_auto.setChecked(True)
        # ImageItem default axis order is (x, y) = (time, channel)
        self.img.setImage(self._shown.T, autoLevels=False)
        lim = None if self.chk_auto.isChecked() else self._typed_levels()
        if lim is None:
            self._apply_auto_levels()
        else:
            self._set_levels(*lim)
        self.img.setRect(QRectF(res.start, res.c0, res.stop - res.start, res.c1 - res.c0))
        self._draw_gaps(res)
        vb = self.plot.getPlotItem().getViewBox()
        # never zoom out past the computed image; zooming in is unrestricted
        vb.setLimits(
            xMin=res.start, xMax=res.stop, yMin=res.c0, yMax=res.c1,
            maxXRange=res.stop - res.start, maxYRange=res.c1 - res.c0,
        )
        self._reset_zoom()
        self.lbl_status.setText(HINT)

    def _reset_zoom(self) -> None:
        res = self._result
        if res is None:
            return
        self.plot.getPlotItem().getViewBox().setRange(
            xRange=(res.start, res.stop), yRange=(res.c0, res.c1), padding=0.0
        )

    @staticmethod
    def _display_values(image: np.ndarray, stat: str) -> np.ndarray:
        if stat == "mean":
            return image
        with np.errstate(divide="ignore", invalid="ignore"):
            db = 20.0 * np.log10(np.maximum(image, 1e-30))
        db[~np.isfinite(image)] = np.nan
        return db

    def _draw_gaps(self, res: OverviewResult) -> None:
        for it in self._gap_items:
            self.plot.removeItem(it)
        self._gap_items.clear()
        for g0, g1, seconds in res.gaps:
            reg = pg.LinearRegionItem(values=(g0, g1), movable=False, brush=GAP_BRUSH)
            reg.setZValue(10)
            txt = pg.TextItem(f"NO DATA ({self._fmt_dur(seconds)})", color=VIEW_PEN_COLOR, anchor=(0, 0))
            txt.setPos(g1, res.c1)
            txt.setZValue(11)
            for it in (reg, txt):
                self.plot.addItem(it)
                self._gap_items.append(it)

    @staticmethod
    def _fmt_dur(s: float) -> str:
        if s >= 3600:
            return f"{s / 3600:.2f} h"
        if s >= 60:
            return f"{s / 60:.1f} min"
        return f"{s:.2f} s"

    # ------------------------------------------------------------- linking
    def _source_is_linked(self) -> bool:
        try:
            return self._dm.current_source() is self._tape
        except Exception:
            return False

    def _view_bounds(self) -> Optional[dict]:
        if not self._source_is_linked():
            return None
        try:
            return self._dm.current_view_bounds()
        except Exception:
            print_exception("Overview: current_view_bounds failed", prefix="Overview")
            return None

    def _update_view_rect(self) -> None:
        b = self._view_bounds()
        if not b:
            self.view_rect.setVisible(False)
            return
        r = QRectF(float(b["t0"]), float(b["c0"]), float(b["t1"] - b["t0"]), float(b["c1"] - b["c0"]))
        if self.view_rect.rect() != r:
            self.view_rect.setRect(r)
        self.view_rect.setVisible(True)

    def _scene_to_data(self, scene_pos) -> Optional[tuple[float, float]]:
        vb = self.plot.getPlotItem().getViewBox()
        if not vb.sceneBoundingRect().contains(scene_pos):
            return None
        p = vb.mapSceneToView(scene_pos)
        return float(p.x()), float(p.y())

    def _on_click(self, ev) -> None:
        if ev.button() != Qt.LeftButton or self._result is None:
            return
        xy = self._scene_to_data(ev.scenePos())
        if xy is None:
            return
        if ev.double():
            # double-click: back to the full image (cancel the pending single click)
            self._click_timer.stop()
            self._pending_click = None
            self._reset_zoom()
            return
        # single click navigates, but only once it is clear it is not a double-click
        self._pending_click = xy
        self._click_timer.start()

    def _navigate_pending(self) -> None:
        xy, self._pending_click = self._pending_click, None
        res = self._result
        if xy is None or res is None or not self._source_is_linked():
            return
        sample = int(np.clip(xy[0], 0, self._nt - 1))
        ch = int(np.clip(xy[1], 0, self._nc - 1))
        if not (res.start <= sample < res.stop):
            return
        try:
            self._dm.goto_sample(sample, center=True)
            b = self._dm.current_view_bounds()
            if b and str(b.get("y_mode")) != "fit_all":
                h = max(1, int(b["c1"] - b["c0"]))
                self._dm.set_channel_start(max(0, min(self._nc - h, ch - h // 2)))
        except Exception:
            print_exception("Overview: failed to move the streaming view", prefix="Overview")

    def _on_move(self, scene_pos) -> None:
        res = self._result
        if res is None:
            return
        xy = self._scene_to_data(scene_pos)
        if xy is None:
            return
        x, y = xy
        if not (res.start <= x < res.stop and res.c0 <= y < res.c1):
            return
        j = int(np.clip(np.searchsorted(res.col_edges, x, side="right") - 1, 0, res.image.shape[1] - 1))
        r = int(np.clip(np.searchsorted(res.row_edges, y, side="right") - 1, 0, res.image.shape[0] - 1))
        v = float(res.image[r, j])
        t = tape_sample_to_unix(self._tape, x)
        when = f"{x / self._fs:.3f} s"
        if t is not None:
            dt = _dt.datetime.fromtimestamp(float(t), tz=_dt.timezone.utc)
            when = dt.strftime("%H:%M:%S.") + f"{dt.microsecond // 1000:03d} UTC"
        if not np.isfinite(v):
            val = "no data"
        elif res.stat == "mean":
            val = f"{v:.4g}"
        else:
            val = f"{20 * np.log10(max(v, 1e-30)):.1f} dB"
        self.lbl_status.setText(f"{when}  ch={int(y)}  {val}")

    # ---------------------------------------------------------------- close
    def closeEvent(self, event) -> None:  # noqa: N802
        self._follow.stop()
        if self._worker is not None:
            self._worker.requestInterruption()
            self._worker.wait(5000)
        super().closeEvent(event)
