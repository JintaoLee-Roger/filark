"""1-D signal analysis viewer.

SignalViewer  — independent waveform viewer (no source required at open time)
SpectrumWindow — PSD + Spectrogram floating window
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal as sp_signal

from filark.utils.time import tape_sample_to_unix, tape_unix_to_sample

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QCursor
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from filark.dsp.temporal import bandpass_time, highpass_time, lowpass_time
from filark.io.tapeio import H5Tape, NpyTape


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nt_nc(source) -> tuple[int, int]:
    dims = getattr(source, "dims", "nt_nc")
    if dims == "nt_nc":
        return int(source.shape[0]), int(source.shape[1])
    return int(source.shape[1]), int(source.shape[0])


def _read_channel(source, channel: int, t0: int = 0, t1: int = 0) -> np.ndarray:
    """Read a single channel, optionally limited to [t0, t1) samples."""
    nt = _nt_nc(source)[0]
    end = min(nt, t1) if t1 > 0 else nt
    start = max(0, min(t0, end - 1))
    if getattr(source, "dims", "nt_nc") == "nt_nc":
        arr = source[start:end, int(channel)]
    else:
        arr = source[int(channel), start:end]
    return np.asarray(arr, dtype=np.float32).ravel()


def _downsample(indices: np.ndarray, values: np.ndarray, max_pts: int):
    n = len(values)
    if n <= max_pts:
        return indices, values
    step = max(1, n // max_pts)
    idx = np.arange(0, n, step)
    return indices[idx], values[idx]


def _wrap(layout) -> QWidget:
    w = QWidget()
    w.setLayout(layout)
    return w


def _try_load_path(path: Path):
    suf = path.suffix.lower()
    if suf in (".h5", ".hdf5"):
        return H5Tape(str(path))
    if suf == ".npy":
        return NpyTape(str(path), dims="nt_nc")
    raise ValueError(f"Unsupported file type: {suf!r}. Supported: .h5, .hdf5, .npy")


# ---------------------------------------------------------------------------
# SpectrumWindow
# ---------------------------------------------------------------------------

class SpectrumWindow(QWidget):
    """Floating PSD + Spectrogram analysis window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Spectrum Analysis")
        self.setWindowFlags(Qt.Window)
        self.resize(720, 580)

        self._trace: np.ndarray | None = None
        self._fs: float = 1.0
        self._channel: int = 0
        self._psd_color: str = "#26C6DA"

        # ---- PSD tab ----
        self._fig_psd = Figure(constrained_layout=True)
        self._canvas_psd = FigureCanvas(self._fig_psd)
        self._ax_psd = self._fig_psd.add_subplot(111)

        psd_ctrl = QWidget()
        psd_form = QFormLayout(psd_ctrl)
        psd_form.setContentsMargins(6, 6, 6, 6)

        self._sp_psd_fmin = QDoubleSpinBox()
        self._sp_psd_fmin.setRange(0.0, 1e6)
        self._sp_psd_fmin.setDecimals(2)
        self._sp_psd_fmin.setValue(0.0)
        self._sp_psd_fmax = QDoubleSpinBox()
        self._sp_psd_fmax.setRange(0.0, 1e6)
        self._sp_psd_fmax.setDecimals(2)
        self._sp_psd_fmax.setValue(500.0)
        self._sp_psd_ymin = QDoubleSpinBox()
        self._sp_psd_ymin.setRange(-300.0, 300.0)
        self._sp_psd_ymin.setValue(-120.0)
        self._sp_psd_ymax = QDoubleSpinBox()
        self._sp_psd_ymax.setRange(-300.0, 300.0)
        self._sp_psd_ymax.setValue(0.0)
        self._sp_psd_nperseg = QSpinBox()
        self._sp_psd_nperseg.setRange(16, 65536)
        self._sp_psd_nperseg.setValue(1024)

        self._btn_psd_color = QPushButton("Pick Color")
        self._btn_psd_color.clicked.connect(self._pick_psd_color)
        self._btn_psd_apply = QPushButton("Apply")
        self._btn_psd_apply.clicked.connect(self._draw_psd)

        btn_row_psd = QHBoxLayout()
        btn_row_psd.addWidget(self._btn_psd_color)
        btn_row_psd.addWidget(self._btn_psd_apply)

        psd_form.addRow("fmin (Hz):", self._sp_psd_fmin)
        psd_form.addRow("fmax (Hz):", self._sp_psd_fmax)
        psd_form.addRow("ymin (dB):", self._sp_psd_ymin)
        psd_form.addRow("ymax (dB):", self._sp_psd_ymax)
        psd_form.addRow("Seg length:", self._sp_psd_nperseg)
        psd_form.addRow("", _wrap(btn_row_psd))

        psd_page = QWidget()
        psd_lay = QVBoxLayout(psd_page)
        psd_lay.setContentsMargins(0, 0, 0, 0)
        psd_lay.addWidget(psd_ctrl)
        psd_lay.addWidget(self._canvas_psd, 1)

        # ---- Spectrogram tab ----
        self._fig_sg = Figure(constrained_layout=True)
        self._canvas_sg = FigureCanvas(self._fig_sg)
        self._ax_sg = self._fig_sg.add_subplot(111)
        self._sg_cbar = None

        sg_ctrl = QWidget()
        sg_form = QFormLayout(sg_ctrl)
        sg_form.setContentsMargins(6, 6, 6, 6)

        self._sp_sg_nperseg = QSpinBox()
        self._sp_sg_nperseg.setRange(16, 65536)
        self._sp_sg_nperseg.setValue(256)
        self._sp_sg_overlap = QSpinBox()
        self._sp_sg_overlap.setRange(0, 99)
        self._sp_sg_overlap.setSuffix(" %")
        self._sp_sg_overlap.setValue(75)
        self._sp_sg_fmin = QDoubleSpinBox()
        self._sp_sg_fmin.setRange(0.0, 1e6)
        self._sp_sg_fmin.setDecimals(2)
        self._sp_sg_fmin.setValue(0.0)
        self._sp_sg_fmax = QDoubleSpinBox()
        self._sp_sg_fmax.setRange(0.0, 1e6)
        self._sp_sg_fmax.setDecimals(2)
        self._sp_sg_fmax.setValue(500.0)
        self._cb_sg_cmap = QComboBox()
        self._cb_sg_cmap.addItems(["magma", "viridis", "plasma", "inferno", "jet", "hot", "RdBu_r"])
        self._sp_sg_vmin = QDoubleSpinBox()
        self._sp_sg_vmin.setRange(-300.0, 300.0)
        self._sp_sg_vmin.setValue(-100.0)
        self._sp_sg_vmax = QDoubleSpinBox()
        self._sp_sg_vmax.setRange(-300.0, 300.0)
        self._sp_sg_vmax.setValue(0.0)
        self._btn_sg_apply = QPushButton("Apply")
        self._btn_sg_apply.clicked.connect(self._draw_spectrogram)

        sg_form.addRow("Window (samples):", self._sp_sg_nperseg)
        sg_form.addRow("Overlap:", self._sp_sg_overlap)
        sg_form.addRow("fmin (Hz):", self._sp_sg_fmin)
        sg_form.addRow("fmax (Hz):", self._sp_sg_fmax)
        sg_form.addRow("Colormap:", self._cb_sg_cmap)
        sg_form.addRow("vmin (dB):", self._sp_sg_vmin)
        sg_form.addRow("vmax (dB):", self._sp_sg_vmax)
        sg_form.addRow("", self._btn_sg_apply)

        sg_page = QWidget()
        sg_lay = QVBoxLayout(sg_page)
        sg_lay.setContentsMargins(0, 0, 0, 0)
        sg_lay.addWidget(sg_ctrl)
        sg_lay.addWidget(self._canvas_sg, 1)

        tabs = QTabWidget()
        tabs.addTab(psd_page, "PSD")
        tabs.addTab(sg_page, "Spectrogram")

        root = QVBoxLayout(self)
        root.addWidget(tabs)

        # Spectrogram zoom state
        self._sg_rect_sel: RectangleSelector | None = None
        self._canvas_sg.mpl_connect("button_press_event", self._on_sg_press)

    def set_trace(self, trace: np.ndarray, fs: float, channel: int) -> None:
        self._trace = np.asarray(trace, dtype=np.float64)
        self._fs = float(fs)
        self._channel = int(channel)
        nyq = self._fs * 0.5
        if self._sp_psd_fmax.value() > nyq:
            self._sp_psd_fmax.setValue(nyq)
        if self._sp_sg_fmax.value() > nyq:
            self._sp_sg_fmax.setValue(nyq)
        self.setWindowTitle(f"Spectrum Analysis — Channel {channel}")
        self._draw_psd()
        self._draw_spectrogram()

    def _pick_psd_color(self) -> None:
        color = QColorDialog.getColor(QColor(self._psd_color), self)
        if color.isValid():
            self._psd_color = color.name()

    def _draw_psd(self) -> None:
        if self._trace is None:
            return
        ax = self._ax_psd
        ax.clear()
        nperseg = min(int(self._sp_psd_nperseg.value()), len(self._trace))
        freqs, psd = sp_signal.welch(self._trace, fs=self._fs, nperseg=nperseg)
        psd_db = 10.0 * np.log10(psd + 1e-30)
        fmin = float(self._sp_psd_fmin.value())
        fmax = float(self._sp_psd_fmax.value())
        mask = (freqs >= fmin) & (freqs <= fmax)
        ax.plot(freqs[mask], psd_db[mask], lw=1.0, color=self._psd_color)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (dB/Hz)")
        ax.set_title(f"PSD — Channel {self._channel}")
        ax.set_ylim(float(self._sp_psd_ymin.value()), float(self._sp_psd_ymax.value()))
        ax.grid(True, alpha=0.2)
        self._canvas_psd.draw_idle()

    def _draw_spectrogram(self) -> None:
        if self._trace is None:
            return
        ax = self._ax_sg
        ax.clear()
        if self._sg_cbar is not None:
            try:
                self._sg_cbar.remove()
            except Exception:
                pass
            self._sg_cbar = None
        nperseg = min(int(self._sp_sg_nperseg.value()), len(self._trace))
        noverlap = int(nperseg * self._sp_sg_overlap.value() / 100.0)
        freqs, times, Sxx = sp_signal.spectrogram(
            self._trace, fs=self._fs, nperseg=nperseg, noverlap=noverlap
        )
        Sxx_db = 10.0 * np.log10(Sxx + 1e-30)
        fmin = float(self._sp_sg_fmin.value())
        fmax = float(self._sp_sg_fmax.value())
        mask = (freqs >= fmin) & (freqs <= fmax)
        im = ax.pcolormesh(
            times, freqs[mask], Sxx_db[mask],
            cmap=self._cb_sg_cmap.currentText(),
            vmin=float(self._sp_sg_vmin.value()),
            vmax=float(self._sp_sg_vmax.value()),
            shading="auto",
        )
        self._sg_cbar = self._fig_sg.colorbar(im, ax=ax, pad=0.02)
        self._sg_cbar.set_label("dB")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"Spectrogram — Channel {self._channel}  [drag to zoom, dbl-click to reset]",
                     fontsize=9)
        self._canvas_sg.draw_idle()
        self._setup_sg_rect_sel()

    def _setup_sg_rect_sel(self) -> None:
        """(Re-)create the RectangleSelector for spectrogram zoom after each redraw."""
        self._sg_rect_sel = RectangleSelector(
            self._ax_sg,
            self._on_sg_zoom_select,
            useblit=True,
            button=[1],
            interactive=False,
            props=dict(facecolor="white", edgecolor="white", alpha=0.25, fill=True),
        )

    def _on_sg_zoom_select(self, eclick, erelease) -> None:
        x1 = min(eclick.xdata, erelease.xdata)
        x2 = max(eclick.xdata, erelease.xdata)
        y1 = min(eclick.ydata, erelease.ydata)
        y2 = max(eclick.ydata, erelease.ydata)
        if x2 - x1 > 0 and y2 - y1 > 0:
            self._ax_sg.set_xlim(x1, x2)
            self._ax_sg.set_ylim(y1, y2)
            self._canvas_sg.draw_idle()

    def _on_sg_press(self, event) -> None:
        if event.inaxes == self._ax_sg and event.dblclick:
            # Reset zoom by redrawing
            self._draw_spectrogram()


# ---------------------------------------------------------------------------
# SignalViewer
# ---------------------------------------------------------------------------

class SignalViewer(QWidget):
    """Independent 1-D signal viewer."""

    _MAX_OV_PTS = 8_000
    _MAX_DET_PTS = 12_000
    _WARN_SAMPLES = 2 * 60 * 1500  # 180 000 — warn before loading more than this

    def __init__(self, source=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Signal Viewer")
        self.resize(1100, 650)

        self._canvas_source = source
        self._source = None
        self._source_nt: int = 0   # total samples in the source
        self._nt: int = 0          # samples actually loaded (may be a sub-range)
        self._nc: int = 0
        self._channel: int = 0
        self._load_offset: int = 0  # source sample index of _raw[0]
        self._raw: np.ndarray | None = None
        self._trace: np.ndarray | None = None
        self._filter_desc: str = "raw"

        self._detail_t0: int = 0
        self._detail_t1: int = 0

        # Overview drag state
        self._ov_drag_mode: str | None = None  # "left" | "right"
        self._ov_drag_anchor_px: float = 0.0   # pixel-x at press (for robust off-axes drag)
        # Overview span artists (set in _draw_overview, None before first draw)
        self._ov_span = None
        self._ov_edge_left = None
        self._ov_edge_right = None

        # cursor artists (created in _draw_detail, None between redraws)
        self._cursor_dot = None
        self._cursor_vline = None
        self._cursor_hline = None
        self._cursor_text = None

        self._context: dict[str, Any] = {}
        self._spectrum_win: SpectrumWindow | None = None

        self._build_figure()
        self._build_controls()
        self._assemble_layout()
        self._bind_events()
        self._draw_empty()

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    def _build_figure(self) -> None:
        self._fig = Figure(constrained_layout=True)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setMinimumHeight(360)
        gs = self._fig.add_gridspec(2, 1, height_ratios=[1, 2.5], hspace=0.35)
        self._ax_ov = self._fig.add_subplot(gs[0])
        self._ax_det = self._fig.add_subplot(gs[1])

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def _build_controls(self) -> None:
        # ---- Source ----
        g_src = QGroupBox("Source")
        src_lay = QVBoxLayout(g_src)
        src_lay.setContentsMargins(6, 6, 6, 6)
        src_lay.setSpacing(6)
        self._lbl_source = QLabel("No source loaded")
        self._lbl_source.setWordWrap(True)
        self._btn_load_file = QPushButton("Load File…")
        self._btn_load_file.clicked.connect(self._pick_file)
        src_form = QFormLayout()
        src_form.setContentsMargins(0, 0, 0, 0)
        self._sp_load_t0 = QSpinBox()
        self._sp_load_t0.setRange(0, 0)
        self._sp_load_t0.setValue(0)
        self._sp_load_t0.setSingleStep(10_000)
        self._sp_load_t0.setToolTip("First sample to load (0 = start of data).")
        self._sp_load_t1 = QSpinBox()
        self._sp_load_t1.setRange(0, 0)
        self._sp_load_t1.setValue(0)
        self._sp_load_t1.setSpecialValueText("end")
        self._sp_load_t1.setSingleStep(10_000)
        self._sp_load_t1.setToolTip(
            "Last sample to load (0 / 'end' = load to end of data).\n"
            f"A warning appears when the range exceeds {self._WARN_SAMPLES:,} samples.\n"
            "When using 'Pick from Canvas', this range is pre-filled from\n"
            "the current canvas view."
        )
        src_form.addRow("Load from:", self._sp_load_t0)
        src_form.addRow("Load to:", self._sp_load_t1)
        src_lay.addWidget(self._lbl_source)
        src_lay.addWidget(self._btn_load_file)
        src_lay.addLayout(src_form)

        # ---- Channel ----
        g_ch = QGroupBox("Channel")
        f_ch = QFormLayout(g_ch)
        f_ch.setContentsMargins(6, 6, 6, 6)
        ch_row = QHBoxLayout()
        ch_row.setSpacing(4)
        self._btn_prev = QPushButton("◀")
        self._btn_prev.setFixedWidth(30)
        self._btn_prev.clicked.connect(lambda: self._set_channel(self._channel - 1))
        self._sp_ch = QSpinBox()
        self._sp_ch.setRange(0, 0)
        self._sp_ch.setKeyboardTracking(False)
        self._sp_ch.valueChanged.connect(self._set_channel)
        self._btn_next = QPushButton("▶")
        self._btn_next.setFixedWidth(30)
        self._btn_next.clicked.connect(lambda: self._set_channel(self._channel + 1))
        ch_row.addWidget(self._btn_prev)
        ch_row.addWidget(self._sp_ch, 1)
        ch_row.addWidget(self._btn_next)
        self._btn_pick = QPushButton("Pick from Canvas")
        self._btn_pick.setCheckable(True)
        self._btn_pick.setEnabled(False)
        self._btn_pick.clicked.connect(self._toggle_pick)
        f_ch.addRow(_wrap(ch_row))
        f_ch.addRow(self._btn_pick)

        # ---- Filter ----
        g_filt = QGroupBox("Filter")
        f_filt = QFormLayout(g_filt)
        f_filt.setContentsMargins(6, 6, 6, 6)
        self._cb_filt_kind = QComboBox()
        self._cb_filt_kind.addItems(["bandpass", "lowpass", "highpass"])
        self._cb_filt_kind.currentTextChanged.connect(self._update_filter_ui)
        self._sp_fmin = QDoubleSpinBox()
        self._sp_fmin.setRange(0.0, 1e6)
        self._sp_fmin.setDecimals(2)
        self._sp_fmin.setValue(5.0)
        self._sp_fmax = QDoubleSpinBox()
        self._sp_fmax.setRange(0.0, 1e6)
        self._sp_fmax.setDecimals(2)
        self._sp_fmax.setValue(200.0)
        self._sp_order = QSpinBox()
        self._sp_order.setRange(1, 20)
        self._sp_order.setValue(4)
        filt_btns = QHBoxLayout()
        self._btn_apply_filt = QPushButton("Apply")
        self._btn_apply_filt.clicked.connect(self._apply_filter)
        self._btn_clear_filt = QPushButton("Clear")
        self._btn_clear_filt.clicked.connect(self._clear_filter)
        filt_btns.addWidget(self._btn_apply_filt)
        filt_btns.addWidget(self._btn_clear_filt)
        self._lbl_filt = QLabel("raw")
        self._lbl_filt.setWordWrap(True)
        f_filt.addRow("Type:", self._cb_filt_kind)
        f_filt.addRow("fmin (Hz):", self._sp_fmin)
        f_filt.addRow("fmax (Hz):", self._sp_fmax)
        f_filt.addRow("Order:", self._sp_order)
        f_filt.addRow(_wrap(filt_btns))
        f_filt.addRow("Status:", self._lbl_filt)

        # ---- X-axis ----
        g_xaxis = QGroupBox("X Axis")
        f_xaxis = QFormLayout(g_xaxis)
        f_xaxis.setContentsMargins(6, 6, 6, 6)
        self._cb_xmode = QComboBox()
        self._cb_xmode.addItems(["samples", "time (s)", "UTC"])
        self._cb_xmode.currentIndexChanged.connect(self._redraw)
        f_xaxis.addRow("Mode:", self._cb_xmode)

        # ---- View Sync ----
        g_sync = QGroupBox("View Sync")
        sync_lay = QVBoxLayout(g_sync)
        sync_lay.setContentsMargins(6, 6, 6, 6)
        sync_lay.setSpacing(6)
        self._btn_push = QPushButton("Detail → Canvas")
        self._btn_push.setEnabled(False)
        self._btn_push.setToolTip("Navigate main canvas to the time range shown in Detail")
        self._btn_push.clicked.connect(self._push_to_canvas)
        self._btn_pull = QPushButton("Canvas → Detail")
        self._btn_pull.setEnabled(False)
        self._btn_pull.setToolTip("Set Detail range to match the main canvas visible window")
        self._btn_pull.clicked.connect(self._pull_from_canvas)
        sync_lay.addWidget(self._btn_push)
        sync_lay.addWidget(self._btn_pull)

        # ---- Spectrum ----
        self._btn_spectrum = QPushButton("Open Spectrum Window")
        self._btn_spectrum.clicked.connect(self._open_spectrum)
        self._btn_spectrum.setEnabled(False)

        self._ctrl_widget = QWidget()
        ctrl_lay = QVBoxLayout(self._ctrl_widget)
        ctrl_lay.setContentsMargins(0, 0, 0, 0)
        ctrl_lay.setSpacing(8)
        ctrl_lay.addWidget(g_src)
        ctrl_lay.addWidget(g_ch)
        ctrl_lay.addWidget(g_filt)
        ctrl_lay.addWidget(g_xaxis)
        ctrl_lay.addWidget(g_sync)
        ctrl_lay.addWidget(self._btn_spectrum)
        ctrl_lay.addStretch(1)

        self._update_filter_ui()

    def _assemble_layout(self) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._ctrl_widget)
        scroll.setFixedWidth(240)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        root.addWidget(scroll)
        root.addWidget(self._canvas, 1)

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _bind_events(self) -> None:
        self._canvas.setFocusPolicy(Qt.StrongFocus)
        self._canvas.mpl_connect("scroll_event", self._on_scroll)
        self._canvas.mpl_connect("button_press_event", self._on_press)
        self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._canvas.mpl_connect("button_release_event", self._on_release)
        self._canvas.mpl_connect("key_press_event", self._on_key_press)

    def _on_scroll(self, event) -> None:
        if event.inaxes != self._ax_det or self._source is None:
            return
        factor = 0.7 if event.button == "up" else 1.4
        cx_sample = (
            self._x_to_sample(event.xdata)
            if event.xdata is not None
            else (self._detail_t0 + self._detail_t1) // 2
        )
        width = self._detail_t1 - self._detail_t0
        ratio = max(0.0, min(1.0, (cx_sample - self._detail_t0) / max(1, width)))
        new_width = max(10, min(self._nt, int(width * factor)))
        new_t0 = max(0, min(self._nt - new_width, int(cx_sample - ratio * new_width)))
        self._detail_t0 = new_t0
        self._detail_t1 = new_t0 + new_width
        self._redraw()

    def _ov_pixel_to_xdata(self, px: float) -> float:
        """Convert display pixel-x to overview data-x (works even outside the axes)."""
        try:
            return float(self._ax_ov.transData.inverted().transform((px, 0))[0])
        except Exception:
            return 0.0

    def _on_press(self, event) -> None:
        if self._source is None:
            return
        if event.inaxes == self._ax_ov and event.button == 1 and event.xdata is not None:
            x = event.xdata
            x_t0 = float(self._sample_to_x(np.array([self._detail_t0]))[0])
            x_t1 = float(self._sample_to_x(np.array([self._detail_t1]))[0])
            # Pixel-based tolerance: ~8 pixels in data units
            ax_w = max(1.0, self._ax_ov.get_window_extent().width)
            xlim = self._ax_ov.get_xlim()
            tol = 8.0 * (xlim[1] - xlim[0]) / ax_w
            if abs(x - x_t0) <= tol:
                self._ov_drag_mode = "left"
                self._ov_drag_anchor_px = event.x
            elif abs(x - x_t1) <= tol:
                self._ov_drag_mode = "right"
                self._ov_drag_anchor_px = event.x
            elif x < x_t0 - tol or x > x_t1 + tol:
                # Click outside span → jump: center the current window on the click
                sample = max(0, min(self._nt, self._x_to_sample(x)))
                width = self._detail_t1 - self._detail_t0
                new_t0 = max(0, min(self._nt - width, sample - width // 2))
                self._detail_t0 = new_t0
                self._detail_t1 = new_t0 + width
                self._redraw()
        elif event.inaxes == self._ax_det:
            self._canvas.setFocus()

    def _on_motion(self, event) -> None:
        # Overview drag — use raw pixel coords so drag works even outside the axes
        if self._ov_drag_mode is not None:
            x = self._ov_pixel_to_xdata(event.x)
            sample = max(0, min(self._nt, self._x_to_sample(x)))
            if self._ov_drag_mode == "left":
                self._detail_t0 = max(0, min(self._detail_t1 - 1, sample))
            elif self._ov_drag_mode == "right":
                self._detail_t1 = max(self._detail_t0 + 1, min(self._nt, sample))
            self._update_ov_span_artist()
            self._draw_detail()
            self._canvas.draw_idle()
            return

        # Hover over overview — update cursor shape
        if event.inaxes == self._ax_ov and event.xdata is not None and self._source is not None:
            x = event.xdata
            x_t0 = float(self._sample_to_x(np.array([self._detail_t0]))[0])
            x_t1 = float(self._sample_to_x(np.array([self._detail_t1]))[0])
            ax_w = max(1.0, self._ax_ov.get_window_extent().width)
            xlim = self._ax_ov.get_xlim()
            tol = 8.0 * (xlim[1] - xlim[0]) / ax_w
            if abs(x - x_t0) <= tol or abs(x - x_t1) <= tol:
                self._canvas.setCursor(QCursor(Qt.SizeHorCursor))
            else:
                self._canvas.setCursor(QCursor(Qt.ArrowCursor))
            return
        if event.inaxes != self._ax_ov:
            self._canvas.unsetCursor()

        # Cursor crosshair
        if self._cursor_dot is None or self._trace is None:
            return
        in_det = event.inaxes == self._ax_det and event.xdata is not None
        if in_det:
            sample = max(0, min(self._nt - 1, self._x_to_sample(event.xdata)))
            x_disp = float(self._sample_to_x(np.array([sample]))[0])
            y_val = float(self._trace[sample])
            xlim = self._ax_det.get_xlim()
            ylim = self._ax_det.get_ylim()
            self._cursor_dot.set_data([x_disp], [y_val])
            self._cursor_dot.set_visible(True)
            if self._cursor_vline is not None:
                self._cursor_vline.set_data([x_disp, x_disp], list(ylim))
                self._cursor_vline.set_visible(True)
            if self._cursor_hline is not None:
                self._cursor_hline.set_data(list(xlim), [y_val, y_val])
                self._cursor_hline.set_visible(True)
            if self._cursor_text is not None:
                self._cursor_text.set_text(f"x: {self._fmt_x(x_disp)}\ny: {y_val:.6g}")
                self._cursor_text.set_visible(True)
        else:
            self._cursor_dot.set_visible(False)
            if self._cursor_vline is not None:
                self._cursor_vline.set_visible(False)
            if self._cursor_hline is not None:
                self._cursor_hline.set_visible(False)
            if self._cursor_text is not None:
                self._cursor_text.set_visible(False)
        self._canvas.draw_idle()

    def _on_release(self, event) -> None:
        if self._ov_drag_mode is not None:
            self._ov_drag_mode = None
            self._redraw(overview=True)  # refresh title with updated range

    def _on_key_press(self, event) -> None:
        if self._source is None or self._trace is None:
            return
        width = self._detail_t1 - self._detail_t0
        step = max(1, width // 10)
        if event.key == "left":
            new_t0 = max(0, self._detail_t0 - step)
            self._detail_t0 = new_t0
            self._detail_t1 = new_t0 + width
            self._redraw()
        elif event.key == "right":
            new_t0 = min(self._nt - width, self._detail_t0 + step)
            self._detail_t0 = new_t0
            self._detail_t1 = new_t0 + width
            self._redraw()
        elif event.key in ("up", "down"):
            factor = 0.7 if event.key == "up" else 1.4
            center = (self._detail_t0 + self._detail_t1) // 2
            new_width = max(10, min(self._nt, int(width * factor)))
            new_t0 = max(0, min(self._nt - new_width, center - new_width // 2))
            self._detail_t0 = new_t0
            self._detail_t1 = new_t0 + new_width
            self._redraw()

    # ------------------------------------------------------------------
    # Source loading
    # ------------------------------------------------------------------

    def _pick_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open signal file", "",
            "Data Files (*.h5 *.hdf5 *.npy);;All Files (*)",
        )
        if not path:
            return
        try:
            src = _try_load_path(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            return
        self._set_source(src, label=Path(path).name)
        self._raw = None
        self._trace = None
        self._cursor_dot = None
        self._draw_empty()

    def _set_source(self, source, label: str = "") -> None:
        self._source = source
        self._source_nt, self._nc = _nt_nc(source)
        self._nt = self._source_nt
        self._channel = 0
        self._load_offset = 0
        self._sp_ch.blockSignals(True)
        self._sp_ch.setRange(0, max(0, self._nc - 1))
        self._sp_ch.setValue(0)
        self._sp_ch.blockSignals(False)
        # Update load-range spinbox bounds and reset to full range
        self._sp_load_t0.setRange(0, max(0, self._source_nt - 1))
        self._sp_load_t0.setValue(0)
        self._sp_load_t1.setRange(0, self._source_nt)
        self._sp_load_t1.setValue(0)  # 0 = "end"
        fs = getattr(source, "fs", None)
        fs_str = f"fs={float(fs):.6g} Hz" if fs not in (None, 0) else "fs=?"
        self._lbl_source.setText(f"{label}\nnt={self._source_nt}, nc={self._nc}, {fs_str}")
        if fs not in (None, 0):
            nyq = float(fs) * 0.5
            self._sp_fmax.setValue(min(self._sp_fmax.value(), nyq))
        # Detail range: use canvas view bounds if available
        bounds = self._context.get("initial_view_bounds")
        if bounds and isinstance(bounds, dict):
            t0 = max(0, int(bounds.get("t0", 0)))
            t1 = min(self._source_nt, int(bounds.get("t1", self._source_nt)))
            self._detail_t0 = t0
            self._detail_t1 = t1 if t1 > t0 else min(self._source_nt, t0 + 5000)
        else:
            self._detail_t0 = 0
            self._detail_t1 = min(self._source_nt, 5000)
        self._btn_spectrum.setEnabled(True)
        self._btn_push.setEnabled(callable(self._context.get("goto_sample")))
        self._btn_pull.setEnabled(callable(self._context.get("current_view_bounds_getter")))

    def _source_changed_from_canvas(self, source) -> None:
        self._canvas_source = source

    # ------------------------------------------------------------------
    # Channel
    # ------------------------------------------------------------------

    def _set_channel(self, ch: int) -> None:
        if self._source is None:
            if self._canvas_source is not None:
                self._set_source(self._canvas_source, label="canvas source")
            else:
                return
        ch = max(0, min(int(ch), self._nc - 1))
        if ch == self._channel and self._raw is not None:
            return
        self._channel = ch
        self._sp_ch.blockSignals(True)
        self._sp_ch.setValue(ch)
        self._sp_ch.blockSignals(False)
        self._load_channel(ch)

    def _load_channel(self, ch: int) -> None:
        t0 = int(self._sp_load_t0.value())
        t1_raw = int(self._sp_load_t1.value())
        t1 = t1_raw if t1_raw > 0 else self._source_nt
        # Clamp to valid range
        t0 = max(0, min(t0, self._source_nt - 1))
        t1 = max(t0 + 1, min(t1, self._source_nt))
        load_n = t1 - t0
        # Warn before loading a large range
        if load_n > self._WARN_SAMPLES:
            fs = getattr(self._source, "fs", None)
            dur_str = (
                f"{load_n / float(fs):.1f} s"
                if fs not in (None, 0)
                else f"{load_n:,} samples"
            )
            msg = QMessageBox(self)
            msg.setWindowTitle("Large Data Warning")
            msg.setIcon(QMessageBox.Warning)
            msg.setText(
                f"The selected range <b>[{t0:,} : {t1:,}]</b> contains "
                f"<b>{load_n:,} samples</b> ({dur_str}), "
                f"which exceeds the threshold of "
                f"{self._WARN_SAMPLES:,} samples (2 min × 1500 Hz).<br><br>"
                f"Loading may be slow or use significant memory.<br>"
                f"You can narrow the <i>Load from / Load to</i> range in the "
                f"Source panel to load only the portion you need.<br><br>"
                f"<b>Proceed with loading {load_n:,} samples?</b>"
            )
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            msg.setDefaultButton(QMessageBox.Cancel)
            if msg.exec() != QMessageBox.Ok:
                return
        self._raw = _read_channel(self._source, ch, t0, t1)
        self._load_offset = t0
        self._nt = len(self._raw)
        self._detail_t0 = 0
        self._detail_t1 = min(self._nt, 5000)
        self._trace = self._raw.copy()
        self._filter_desc = "raw"
        self._lbl_filt.setText("raw")
        self._cursor_dot = None
        self._redraw()
        self._update_spectrum_if_open()

    def _toggle_pick(self, checked: bool) -> None:
        cb = self._context.get("begin_channel_pick")
        cancel = self._context.get("cancel_channel_pick")
        raise_main = self._context.get("raise_main_window")
        if checked and callable(cb):
            cb(self._on_channel_picked)
            if callable(raise_main):
                raise_main()
        elif not checked and callable(cancel):
            cancel()

    def _on_channel_picked(self, channel: int) -> None:
        self._btn_pick.setChecked(False)
        if self._source is None and self._canvas_source is not None:
            self._set_source(self._canvas_source, label="canvas source")
        self._set_channel(channel)
        self.raise_()
        self.activateWindow()

    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------

    def _update_filter_ui(self) -> None:
        kind = self._cb_filt_kind.currentText()
        self._sp_fmin.setEnabled(kind != "lowpass")
        self._sp_fmax.setEnabled(kind != "highpass")

    def _apply_filter(self) -> None:
        if self._raw is None or self._source is None:
            return
        fs = getattr(self._source, "fs", None)
        if fs in (None, 0):
            QMessageBox.warning(self, "Filter", "Source has no valid fs.")
            return
        fs = float(fs)
        kind = self._cb_filt_kind.currentText()
        order = int(self._sp_order.value())
        data = self._raw.reshape(-1, 1).astype(np.float32)
        try:
            if kind == "bandpass":
                out = bandpass_time(data, fs=fs, fmin=float(self._sp_fmin.value()),
                                    fmax=float(self._sp_fmax.value()), order=order, zero_phase=True)
                self._filter_desc = f"BP {self._sp_fmin.value():.4g}–{self._sp_fmax.value():.4g} Hz, ord={order}"
            elif kind == "lowpass":
                out = lowpass_time(data, fs=fs, fmax=float(self._sp_fmax.value()),
                                   order=order, zero_phase=True)
                self._filter_desc = f"LP <{self._sp_fmax.value():.4g} Hz, ord={order}"
            else:
                out = highpass_time(data, fs=fs, fmin=float(self._sp_fmin.value()),
                                    order=order, zero_phase=True)
                self._filter_desc = f"HP >{self._sp_fmin.value():.4g} Hz, ord={order}"
        except Exception as exc:
            QMessageBox.warning(self, "Filter Failed", str(exc))
            return
        self._trace = np.asarray(out, dtype=np.float32).ravel()
        self._lbl_filt.setText(self._filter_desc)
        self._cursor_dot = None
        self._redraw()
        self._update_spectrum_if_open()

    def _clear_filter(self) -> None:
        if self._raw is None:
            return
        self._trace = self._raw.copy()
        self._filter_desc = "raw"
        self._lbl_filt.setText("raw")
        self._cursor_dot = None
        self._redraw()
        self._update_spectrum_if_open()

    # ------------------------------------------------------------------
    # X-axis helpers
    # ------------------------------------------------------------------

    def _sample_to_x(self, samples: np.ndarray) -> np.ndarray:
        """Convert 0-based indices into _raw to display coordinates."""
        mode = self._cb_xmode.currentText()
        # Shift by load offset so x-axis shows absolute position in the source
        actual = samples.astype(np.float64) + self._load_offset
        if mode == "samples":
            return actual
        fs = float(getattr(self._source, "fs", None) or 1.0) if self._source else 1.0
        t_rel = actual / fs
        if mode == "time (s)":
            return t_rel
        abs_t = tape_sample_to_unix(self._source, actual) if self._source else None
        if abs_t is not None:
            return np.asarray(abs_t, dtype=np.float64)  # gap-aware absolute time
        return t_rel

    def _x_to_sample(self, x: float) -> int:
        """Convert display coordinate to 0-based index into _raw."""
        mode = self._cb_xmode.currentText()
        if mode == "samples":
            return int(round(x)) - self._load_offset
        fs = float(getattr(self._source, "fs", None) or 1.0) if self._source else 1.0
        if mode == "time (s)":
            return int(round(x * fs)) - self._load_offset
        n = tape_unix_to_sample(self._source, x) if self._source else None
        if n is None:
            return int(round(x * fs)) - self._load_offset
        return int(n) - self._load_offset

    def _fmt_x(self, x_disp: float) -> str:
        """Format a display x value as a human-readable string."""
        mode = self._cb_xmode.currentText()
        if mode == "samples":
            return f"{int(round(x_disp)):,}"
        if mode == "time (s)":
            return f"{x_disp:.4f} s"
        try:
            dt = datetime.fromtimestamp(float(x_disp), tz=timezone.utc)
            return dt.strftime("%H:%M:%S.%f")[:-3] + " UTC"
        except Exception:
            return f"{x_disp:.4f} s"

    def _x_label(self) -> str:
        mode = self._cb_xmode.currentText()
        if mode == "samples":
            return "Sample index"
        if mode == "time (s)":
            return "Time (s)"
        t0 = getattr(self._source, "start_time", None) if self._source else None
        if t0 is not None:
            try:
                dt = datetime.fromtimestamp(float(t0), tz=timezone.utc)
                return f"Time (s)  [origin: {dt.strftime('%Y-%m-%d %H:%M:%S UTC')}]"
            except Exception:
                pass
        return "Time (s, since epoch)"

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_empty(self) -> None:
        for ax, title in [(self._ax_ov, "Overview"), (self._ax_det, "Detail")]:
            ax.clear()
            ax.text(0.5, 0.5, "No source loaded\nUse 'Load File…' or pick a channel",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="#888888")
            ax.set_title(title, fontsize=9)
        self._canvas.draw_idle()

    def _redraw(self, *, overview: bool = True) -> None:
        if self._trace is None or self._source is None:
            self._draw_empty()
            return
        if overview:
            self._draw_overview()
        else:
            self._update_ov_span_artist()
        self._draw_detail()
        self._canvas.draw_idle()

    def _update_ov_span_artist(self) -> None:
        """Fast path: update overview span polygon + edge lines without clearing axes."""
        if self._ov_span is None or self._ov_edge_left is None or self._ov_edge_right is None:
            return
        x_t0 = float(self._sample_to_x(np.array([self._detail_t0]))[0])
        x_t1 = float(self._sample_to_x(np.array([self._detail_t1]))[0])
        try:
            xy = self._ov_span.get_xy()
            xy[0, 0] = x_t0; xy[1, 0] = x_t0
            xy[2, 0] = x_t1; xy[3, 0] = x_t1
            if len(xy) > 4:
                xy[4, 0] = x_t0
            self._ov_span.set_xy(xy)
            self._ov_edge_left.set_xdata([x_t0, x_t0])
            self._ov_edge_right.set_xdata([x_t1, x_t1])
        except Exception:
            pass

    def _draw_overview(self) -> None:
        ax = self._ax_ov
        ax.clear()
        self._ov_span = None
        self._ov_edge_left = None
        self._ov_edge_right = None
        idx_all = np.arange(self._nt)
        idx_ds, y_ds = _downsample(idx_all, self._trace, self._MAX_OV_PTS)
        x_ds = self._sample_to_x(idx_ds)
        ax.plot(x_ds, y_ds, lw=0.7, color="#4FC3F7")
        x_t0 = float(self._sample_to_x(np.array([self._detail_t0]))[0])
        x_t1 = float(self._sample_to_x(np.array([self._detail_t1]))[0])
        self._ov_span = ax.axvspan(x_t0, x_t1, color="#FFD54F", alpha=0.28)
        self._ov_edge_left, = ax.plot([x_t0, x_t0], [0, 1],
                                      transform=ax.get_xaxis_transform(),
                                      color="#FFD54F", lw=1.5, ls="--", alpha=0.85)
        self._ov_edge_right, = ax.plot([x_t1, x_t1], [0, 1],
                                       transform=ax.get_xaxis_transform(),
                                       color="#FFD54F", lw=1.5, ls="--", alpha=0.85)
        n_det = self._detail_t1 - self._detail_t0
        fs = getattr(self._source, "fs", None)
        dur_str = f"{n_det / float(fs):.3f} s" if fs not in (None, 0) else f"{n_det} samp"
        abs_t0 = self._detail_t0 + self._load_offset
        abs_t1 = self._detail_t1 + self._load_offset
        ax.set_title(
            f"Overview — ch {self._channel}  |  {self._filter_desc}"
            f"  |  detail: [{abs_t0}:{abs_t1}] ({dur_str})",
            fontsize=9,
        )
        ax.set_ylabel("Amplitude", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.15)

    def _draw_detail(self) -> None:
        ax = self._ax_det
        ax.clear()
        t0, t1 = self._detail_t0, self._detail_t1
        idx_seg = np.arange(t0, t1)
        y_seg = self._trace[t0:t1]
        idx_ds, y_ds = _downsample(idx_seg, y_seg, self._MAX_DET_PTS)
        x_ds = self._sample_to_x(idx_ds)
        ax.plot(x_ds, y_ds, lw=0.9, color="#FF8A65")
        n_samples = t1 - t0
        fs = getattr(self._source, "fs", None)
        dur_str = f"{n_samples / float(fs):.3f} s" if fs not in (None, 0) else f"{n_samples} samples"
        abs_t0 = t0 + self._load_offset
        abs_t1 = t1 + self._load_offset
        ax.set_title(f"Detail  [{abs_t0} : {abs_t1}]  {dur_str}", fontsize=9)
        ax.set_xlabel(self._x_label(), fontsize=8)
        ax.set_ylabel("Amplitude", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.15)
        # Cursor dot — empty data so it never affects xlim/ylim
        self._cursor_dot, = ax.plot([], [], "o", color="#FFD54F", ms=7,
                                    zorder=5, visible=False)
        self._cursor_text = ax.text(
            0.99, 0.97, "", transform=ax.transAxes,
            ha="right", va="top", fontsize=8,
            color="#FFD54F", alpha=0.9,
            bbox=dict(boxstyle="round,pad=0.2", fc="#1a1a1a", ec="none", alpha=0.6),
            zorder=6, visible=False,
        )

    # ------------------------------------------------------------------
    # View Sync
    # ------------------------------------------------------------------

    def _push_to_canvas(self) -> None:
        goto = self._context.get("goto_sample")
        if callable(goto):
            goto((self._detail_t0 + self._detail_t1) // 2, center=True)

    def _pull_from_canvas(self) -> None:
        getter = self._context.get("current_view_bounds_getter")
        if not callable(getter):
            return
        bounds = getter()
        if not bounds:
            return
        self._detail_t0 = max(0, int(bounds.get("t0", 0)))
        self._detail_t1 = min(self._nt, int(bounds.get("t1", self._nt)))
        if self._detail_t1 <= self._detail_t0:
            self._detail_t1 = min(self._nt, self._detail_t0 + 1)
        self._redraw()

    # ------------------------------------------------------------------
    # Spectrum
    # ------------------------------------------------------------------

    def _open_spectrum(self) -> None:
        if self._spectrum_win is None:
            self._spectrum_win = SpectrumWindow()
        if self._trace is not None:
            fs = float(getattr(self._source, "fs", None) or 1.0)
            self._spectrum_win.set_trace(self._trace, fs, self._channel)
        self._spectrum_win.show()
        self._spectrum_win.raise_()

    def _update_spectrum_if_open(self) -> None:
        if (self._spectrum_win is not None
                and self._spectrum_win.isVisible()
                and self._trace is not None
                and self._source is not None):
            fs = float(getattr(self._source, "fs", None) or 1.0)
            self._spectrum_win.set_trace(self._trace, fs, self._channel)

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    def set_filark_context(self, context: dict | None) -> None:
        self._context = dict(context or {})
        self._btn_pick.setEnabled(callable(self._context.get("begin_channel_pick")))
        if self._source is not None:
            self._btn_push.setEnabled(callable(self._context.get("goto_sample")))
            self._btn_pull.setEnabled(callable(self._context.get("current_view_bounds_getter")))
