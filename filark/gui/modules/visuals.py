"""Side-panel controls for viewer rendering, scaling, and axis display."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import QSignalBlocker
from PySide6.QtGui import QStandardItem
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from filark.gui.display import DisplayModel
from filark.gui.widgets import CollapsibleSection
from filark.gui.widgets.typography import PanelTitle
from filark.io.streaming import StreamingSource
from filark.viz._aspect import DEFAULT_MAX_DISPLAY_ASPECT, DEFAULT_MIN_DISPLAY_ASPECT


@dataclass
class VisualState:
    vmin: float = 0.0
    vmax: float = 1.0
    cmap: str = "seismic"
    interpolation: str = "nearest"
    scale_x: float = 1.0
    scale_y: float = 1.0
    display_aspect: float = 1.0
    auto_display_aspect: bool = True
    display_aspect_min: float = DEFAULT_MIN_DISPLAY_ASPECT
    display_aspect_max: float = DEFAULT_MAX_DISPLAY_ASPECT
    downsample_mode: str = "drop"
    buffer_h: int = 0
    buffer_w: int = 0
    auto_buffer_resize: bool = True
    x_axis_mode: str = "sample"    # "sample" | "time"
    y_axis_mode: str = "channel"   # "channel" | "distance"


class VisualsPanel(QWidget):
    """
    Visual controls panel: color limits, rendering, scale and axis coordinates.
    """

    def __init__(self, display_model: DisplayModel, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.display_model = display_model
        self.state = VisualState()
        self._axis_metadata = None

        self._build_ui()
        self._wire()
        self._sync_scale_state(self.display_model.current_scale_state())
        self._sync_filter_limits_from_source()
        self._apply_all_to_canvas()

    def set_axis_metadata(self, axis_metadata) -> None:
        self._axis_metadata = axis_metadata
        x_role = str(getattr(getattr(axis_metadata, "x", None), "role", "time"))
        y_role = str(getattr(getattr(axis_metadata, "y", None), "role", "channel"))
        with QSignalBlocker(self.cb_x_axis), QSignalBlocker(self.cb_y_axis):
            if x_role != "time":
                self.cb_x_axis.setCurrentIndex(0)
                self.cb_x_axis.setEnabled(False)
                self.cb_x_axis.setToolTip("Coordinate conversion is disabled because the displayed x axis is not time.")
                self.state.x_axis_mode = "sample"
            else:
                self.cb_x_axis.setEnabled(True)
                self.cb_x_axis.setToolTip(
                    "Time (samples): raw sample index.\n"
                    "Time (s): elapsed seconds, using the source's sampling rate (fs).\n"
                    "UTC clock: absolute wall-clock time (HH:MM:SS). Requires start_time."
                )
            if y_role != "channel":
                self.cb_y_axis.setCurrentIndex(0)
                self.cb_y_axis.setEnabled(False)
                self.cb_y_axis.setToolTip("Distance conversion is disabled because the displayed y axis is not channel.")
                self.state.y_axis_mode = "channel"
            else:
                self.cb_y_axis.setEnabled(True)
                self.cb_y_axis.setToolTip(
                    "Channel index: shows raw channel number.\n"
                    "Distance: converts using the source's channel spacing (dx)."
                )
        self._apply_axis_mode_to_canvas()

    # ---------------- UI ----------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        self.setMinimumWidth(320)

        root.addWidget(PanelTitle("Visuals"))

        # --- Display (color limits + rendering + axis coordinates merged) ---
        g_disp = QGroupBox("")
        f_disp = QFormLayout(g_disp)
        self._configure_form_layout(f_disp)

        # color limits
        self.sp_vmin = QDoubleSpinBox()
        self.sp_vmin.setRange(-1e30, 1e30)
        self.sp_vmin.setDecimals(4)
        self.sp_vmin.setSingleStep(0.1)
        self.sp_vmin.setValue(-1)
        self.sp_vmin.setToolTip("Lower color limit (vmin).")

        self.sp_vmax = QDoubleSpinBox()
        self.sp_vmax.setRange(-1e30, 1e30)
        self.sp_vmax.setDecimals(4)
        self.sp_vmax.setSingleStep(0.1)
        self.sp_vmax.setValue(1)
        self.sp_vmax.setToolTip("Upper color limit (vmax).")

        self.chk_sym = QCheckBox("Symmetric")
        self.chk_sym.setToolTip("|vmin| = vmax — adjusting either mirrors the other")

        f_disp.addRow("clim (vmin-vmax):", self._make_hrow(self.sp_vmin, self.sp_vmax))
        f_disp.addRow(self.chk_sym)

        # rendering
        self.cb_cmap = QComboBox()
        self.cb_cmap.addItems([
            "seismic", "gray", "viridis", "RdBu_r", "RdBu", "magma", "inferno", "plasma",
            "turbo", "cividis", "coolwarm"
        ])

        self.cb_interp = QComboBox()
        self.cb_interp.addItems(["nearest", "linear"])

        self.cb_transform = QComboBox()
        self.cb_transform.setEnabled(False)
        self.cb_transform.addItem("Raw", "raw")

        f_disp.addRow("cmap:", self.cb_cmap)
        f_disp.addRow("interpolation:", self.cb_interp)
        f_disp.addRow("Tape transform:", self.cb_transform)

        # axis coordinates
        self.cb_x_axis = QComboBox()
        self.cb_x_axis.addItems(["Time (samples)", "Time (s)", "UTC clock"])
        self.cb_x_axis.setToolTip(
            "Time (samples): raw sample index.\n"
            "Time (s): elapsed seconds, using the source's sampling rate (fs).\n"
            "UTC clock: absolute wall-clock time (HH:MM:SS). Requires start_time."
        )

        self.cb_y_axis = QComboBox()
        self.cb_y_axis.addItems(["Channel index", "Distance"])
        self.cb_y_axis.setToolTip(
            "Channel index: shows raw channel number.\n"
            "Distance: converts using the source's channel spacing (dx)."
        )

        f_disp.addRow("X axis:", self.cb_x_axis)
        f_disp.addRow("Y axis:", self.cb_y_axis)

        root.addWidget(CollapsibleSection("Display", collapsed=False).add(g_disp))

        # --- View scale / overview ---
        g_scale = QGroupBox("")
        f_scale = QFormLayout(g_scale)
        self._configure_form_layout(f_scale)

        self.sp_scale_x = QDoubleSpinBox()
        self.sp_scale_x.setRange(1.0, 1e6)
        self.sp_scale_x.setDecimals(3)
        self.sp_scale_x.setSingleStep(1.0)
        self.sp_scale_x.setValue(1.0)
        self.sp_scale_x.setToolTip("Compression factor along time. Larger means wider overview.")

        self.sp_scale_y = QDoubleSpinBox()
        self.sp_scale_y.setRange(1.0, 1e6)
        self.sp_scale_y.setDecimals(3)
        self.sp_scale_y.setSingleStep(1.0)
        self.sp_scale_y.setValue(1.0)
        self.sp_scale_y.setToolTip("Compression factor along channels. Larger means more channels per view height.")

        # channel range
        self.sp_ch_start = QSpinBox()
        self.sp_ch_start.setRange(0, 999_999)
        self.sp_ch_start.setValue(0)
        self.sp_ch_start.setKeyboardTracking(False)
        self.sp_ch_start.setToolTip("First channel to display.")

        self.sp_ch_end = QSpinBox()
        self.sp_ch_end.setRange(1, 1_000_000)
        self.sp_ch_end.setValue(512)
        self.sp_ch_end.setKeyboardTracking(False)
        self.sp_ch_end.setToolTip("Last channel to display (exclusive).")

        self.btn_ch_go = QPushButton("Go")
        self.btn_ch_go.setToolTip(
            "Jump to the specified channel range.\n"
            "Adjusts scale_y so the range fits the buffer, then navigates to ch_start."
        )

        ch_range_row = QWidget()
        ch_range_layout = QHBoxLayout(ch_range_row)
        ch_range_layout.setContentsMargins(0, 0, 0, 0)
        ch_range_layout.setSpacing(6)
        ch_range_layout.addWidget(self.sp_ch_start, 1)
        ch_range_layout.addWidget(self.sp_ch_end, 1)
        ch_range_layout.addWidget(self.btn_ch_go)

        self.sp_display_aspect = QDoubleSpinBox()
        self.sp_display_aspect.setRange(self.state.display_aspect_min, self.state.display_aspect_max)
        self.sp_display_aspect.setDecimals(3)
        self.sp_display_aspect.setSingleStep(1.0)
        self.sp_display_aspect.setValue(1.0)
        self.sp_display_aspect.setToolTip(
            "X-axis compression relative to Y.\n"
            "2 means 1 channel height = 2 sample widths on screen (X compressed).\n"
            "1 = square pixels. Values < 1 stretch X."
        )

        self.chk_auto_aspect = QCheckBox("Auto aspect from canvas height")
        self.chk_auto_aspect.setChecked(True)
        self.chk_auto_aspect.setToolTip(
            "Automatically reduce display aspect when the channel count is much smaller than the canvas height."
        )

        self.btn_reset_aspect = QPushButton("1:1 aspect")
        self.btn_reset_aspect.setToolTip("Disable auto aspect and reset display aspect to 1.0.")

        self.cb_downsample = QComboBox()
        self.cb_downsample.addItem("Drop (fastest)", "drop")
        self.cb_downsample.addItem("Adaptive peak", "hybrid")
        self.cb_downsample.addItem("Exact peak", "peak")
        self.cb_downsample.addItem("Linear", "linear")
        self.cb_downsample.setToolTip(
            "Drop (fastest): direct sample dropping / nearest selection.\n"
            "Adaptive peak: preserves transients while capping IO.\n"
            "Exact peak: preserves transients without presampling.\n"
            "Linear: interpolate to the requested output size."
        )

        self.btn_fit_y = QPushButton("Fit Y")
        self.btn_fit_y.setToolTip("Compress Y so all channels fit into the current buffer height.")
        self.btn_reset_y = QPushButton("1:1 Y")
        self.btn_reset_y.setToolTip("Reset channel compression to local 1:1 view.")

        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(6)
        btn_layout.addWidget(self.btn_fit_y)
        btn_layout.addWidget(self.btn_reset_y)

        aspect_row = QWidget()
        aspect_layout = QHBoxLayout(aspect_row)
        aspect_layout.setContentsMargins(0, 0, 0, 0)
        aspect_layout.setSpacing(6)
        aspect_layout.addWidget(self.sp_display_aspect, 1)
        aspect_layout.addWidget(self.btn_reset_aspect)

        f_scale.addRow("scale (x-y):", self._make_hrow(self.sp_scale_x, self.sp_scale_y))
        f_scale.addRow("ch range:", ch_range_row)
        f_scale.addRow("display aspect:", aspect_row)
        f_scale.addRow(self.chk_auto_aspect)
        f_scale.addRow("overview:", self.cb_downsample)
        f_scale.addRow(btn_row)
        root.addWidget(CollapsibleSection("View scale", collapsed=True).add(g_scale))

        # --- Streaming buffer ---
        g_buf = QGroupBox("")
        f_buf = QFormLayout(g_buf)
        self._configure_form_layout(f_buf)

        self.sp_buffer_h = QSpinBox()
        self.sp_buffer_h.setRange(1, 1_000_000)
        self.sp_buffer_h.setSingleStep(64)
        self.sp_buffer_h.setValue(2048)
        self.sp_buffer_h.setKeyboardTracking(False)
        self.sp_buffer_h.setToolTip("Texture/buffer height used for streaming. Higher uses more memory but improves fidelity.")

        self.sp_buffer_w = QSpinBox()
        self.sp_buffer_w.setRange(1, 10_000_000)
        self.sp_buffer_w.setSingleStep(128)
        self.sp_buffer_w.setValue(2048)
        self.sp_buffer_w.setKeyboardTracking(False)
        self.sp_buffer_w.setToolTip(
            "Texture/buffer width used for streaming. Disable auto width first, then press Enter to apply."
        )

        self.chk_auto_buffer = QCheckBox("Auto fit width on resize")
        self.chk_auto_buffer.setChecked(True)
        self.chk_auto_buffer.setToolTip("Recompute buffer width from the current window size so fullscreen views stay filled.")

        f_buf.addRow("buffer (h-w):", self._make_hrow(self.sp_buffer_h, self.sp_buffer_w))
        f_buf.addRow(self.chk_auto_buffer)
        # --- Display filter ---
        g_filt = QGroupBox("")
        f_filt = QFormLayout(g_filt)
        self._configure_form_layout(f_filt)

        self.chk_filter_enable = QCheckBox("Enable")
        self.chk_filter_enable.setToolTip(
            "Apply a causal realtime IIR bandpass/lowpass/highpass filter to rendered data.\n"
            "State is preserved during forward X streaming and reset on seeks, jumps, or Y-window changes.\n"
            "You can edit parameters before enabling the filter."
        )

        self.cb_filter_kind = QComboBox()
        self.cb_filter_kind.addItems(["bandpass", "lowpass", "highpass"])

        self.sp_fmin = QDoubleSpinBox()
        self.sp_fmin.setRange(0.0, 1e6)
        self.sp_fmin.setDecimals(3)
        self.sp_fmin.setSingleStep(1.0)
        self.sp_fmin.setValue(5.0)
        self.sp_fmin.setToolTip("Lower cutoff frequency (Hz). Used by bandpass / highpass.")

        self.sp_fmax = QDoubleSpinBox()
        self.sp_fmax.setRange(0.0, 1e6)
        self.sp_fmax.setDecimals(3)
        self.sp_fmax.setSingleStep(1.0)
        self.sp_fmax.setValue(250.0)
        self.sp_fmax.setToolTip("Upper cutoff frequency (Hz). Used by bandpass / lowpass.")

        self.sp_filter_order = QSpinBox()
        self.sp_filter_order.setRange(1, 20)
        self.sp_filter_order.setValue(4)
        self.sp_filter_order.setToolTip("Butterworth filter order. Higher = steeper roll-off.")

        self.btn_filter_apply = QPushButton("Apply")
        self.btn_filter_apply.setToolTip("Apply / update the display filter with the current parameters when enabled.")
        self.btn_filter_apply.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        f_filt.addRow(self.chk_filter_enable)
        f_filt.addRow("Type:", self.cb_filter_kind)
        f_filt.addRow("fmin (Hz):", self.sp_fmin)
        f_filt.addRow("fmax (Hz):", self.sp_fmax)
        f_filt.addRow("Order:", self.sp_filter_order)
        f_filt.addRow(self.btn_filter_apply)
        self.lbl_filter_status = QLabel("")
        self.lbl_filter_status.setWordWrap(True)
        self.lbl_filter_status.setStyleSheet("color: #ff8c00;")
        self.lbl_filter_status.setVisible(False)
        f_filt.addRow(self.lbl_filter_status)
        root.addWidget(CollapsibleSection("Display filter", collapsed=True).add(g_filt))
        root.addWidget(CollapsibleSection("Streaming buffer", collapsed=True).add(g_buf))

        # --- Channel markers ---
        g_mark = QGroupBox("")
        f_mark = QFormLayout(g_mark)
        self._configure_form_layout(f_mark)

        self.le_marker_channels = QLineEdit()
        self.le_marker_channels.setPlaceholderText("e.g.  10, 50, 200")
        self.le_marker_channels.setToolTip(
            "Comma-separated source channel indices to mark with horizontal lines.\n"
            "Lines are drawn in camera space and work correctly with streaming sources."
        )

        self.cb_marker_color = QComboBox()
        self.cb_marker_color.addItems(["yellow", "cyan", "red", "lime", "white", "magenta"])
        self.cb_marker_color.setToolTip("Line color applied to all markers.")

        self.btn_marker_apply = QPushButton("Apply")
        self.btn_marker_apply.setToolTip("Draw horizontal lines at the specified channels.")
        self.btn_marker_clear = QPushButton("Clear")
        self.btn_marker_clear.setToolTip("Remove all channel marker lines.")

        mark_btn_row = QWidget()
        mark_btn_lay = QHBoxLayout(mark_btn_row)
        mark_btn_lay.setContentsMargins(0, 0, 0, 0)
        mark_btn_lay.setSpacing(6)
        mark_btn_lay.addWidget(self.btn_marker_apply)
        mark_btn_lay.addWidget(self.btn_marker_clear)

        f_mark.addRow("Channels:", self.le_marker_channels)
        f_mark.addRow("Color:", self.cb_marker_color)
        f_mark.addRow(mark_btn_row)
        root.addWidget(CollapsibleSection("Channel markers", collapsed=True).add(g_mark))

        root.addStretch(1)

    @staticmethod
    def _make_hrow(*widgets: QWidget) -> QWidget:
        """Place widgets side by side with equal stretch."""
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        for w in widgets:
            h.addWidget(w, 1)
        return row

    @staticmethod
    def _configure_form_layout(layout: QFormLayout) -> None:
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(4)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        layout.setRowWrapPolicy(QFormLayout.WrapAllRows)

    def _wire(self):
        # Most visual controls update immediately; buffer shape waits for editingFinished
        # so typing a large value does not recreate the texture on every digit.
        self.sp_vmin.valueChanged.connect(self._commit_clim)
        self.sp_vmax.valueChanged.connect(self._commit_clim)
        # editingFinished: apply symmetric mirror only when the user finishes typing
        self.sp_vmin.editingFinished.connect(self._on_vmin_editing_finished)
        self.sp_vmax.editingFinished.connect(self._on_vmax_editing_finished)
        self.cb_cmap.currentTextChanged.connect(self._on_cmap_changed)
        self.cb_interp.currentTextChanged.connect(self._on_interp_changed)
        self.cb_transform.currentIndexChanged.connect(self._on_transform_changed)
        self.sp_scale_x.valueChanged.connect(self._on_scale_x_changed)
        self.sp_scale_y.valueChanged.connect(self._on_scale_y_changed)
        self.sp_display_aspect.valueChanged.connect(self._on_display_aspect_changed)
        self.chk_auto_aspect.toggled.connect(self._on_auto_aspect_toggled)
        self.btn_reset_aspect.clicked.connect(self._on_reset_aspect_clicked)
        self.cb_downsample.currentIndexChanged.connect(self._on_downsample_mode_changed)
        self.btn_ch_go.clicked.connect(self._on_channel_range_go)
        self.btn_fit_y.clicked.connect(self._on_fit_y_clicked)
        self.btn_reset_y.clicked.connect(self._on_reset_y_clicked)
        self.sp_buffer_h.editingFinished.connect(self._on_buffer_h_editing_finished)
        self.sp_buffer_w.editingFinished.connect(self._on_buffer_w_editing_finished)
        self.chk_auto_buffer.toggled.connect(self._on_auto_buffer_toggled)
        self.cb_x_axis.currentIndexChanged.connect(self._on_axis_mode_changed)
        self.cb_y_axis.currentIndexChanged.connect(self._on_axis_mode_changed)
        self.display_model.scale_state_changed.connect(self._sync_scale_state)
        self.display_model.source_transform_changed.connect(self._sync_transform_state)

        # channel markers
        self.btn_marker_apply.clicked.connect(self._on_marker_apply)
        self.btn_marker_clear.clicked.connect(self._on_marker_clear)
        self.le_marker_channels.returnPressed.connect(self._on_marker_apply)

        # display filter
        self.chk_filter_enable.toggled.connect(self._on_filter_enable_toggled)
        self.cb_filter_kind.currentTextChanged.connect(self._on_filter_kind_changed)
        self.btn_filter_apply.clicked.connect(self._on_filter_apply)
        self.display_model.display_filter_status_changed.connect(self._on_display_filter_status)

    # ---------------- External API ----------------
    def set_clim(self, vmin: float, vmax: float):
        if vmax == vmin:
            vmax = vmin + 0.01

        self.state.vmin = float(vmin)
        self.state.vmax = float(vmax)

        with QSignalBlocker(self.sp_vmin), QSignalBlocker(self.sp_vmax):
            self.sp_vmin.setValue(self.state.vmin)
            self.sp_vmax.setValue(self.state.vmax)

        self._apply_clim_to_canvas()

    def reset(self, **_):
        """Reset visuals panel to default state."""
        default = VisualState()

        self.state.vmin = default.vmin
        self.state.vmax = default.vmax
        with QSignalBlocker(self.sp_vmin), QSignalBlocker(self.sp_vmax):
            self.sp_vmin.setValue(default.vmin)
            self.sp_vmax.setValue(default.vmax)

        self.state.cmap = default.cmap
        self.state.interpolation = default.interpolation
        with QSignalBlocker(self.cb_cmap), QSignalBlocker(self.cb_interp):
            self.cb_cmap.setCurrentText(default.cmap)
            self.cb_interp.setCurrentText(default.interpolation)

        self.state.scale_x = default.scale_x
        self.state.scale_y = default.scale_y
        self.state.display_aspect = default.display_aspect
        self.state.auto_display_aspect = default.auto_display_aspect
        aspect_min = self.state.display_aspect_min
        aspect_max = self.state.display_aspect_max
        self.state.display_aspect_min = aspect_min
        self.state.display_aspect_max = aspect_max
        self.state.downsample_mode = default.downsample_mode
        self.state.buffer_h = default.buffer_h
        self.state.buffer_w = default.buffer_w
        self.state.auto_buffer_resize = default.auto_buffer_resize
        with QSignalBlocker(self.sp_scale_x), QSignalBlocker(self.sp_scale_y), QSignalBlocker(self.sp_display_aspect), QSignalBlocker(self.chk_auto_aspect), QSignalBlocker(self.cb_downsample), QSignalBlocker(self.sp_buffer_h), QSignalBlocker(self.sp_buffer_w), QSignalBlocker(self.chk_auto_buffer):
            self.sp_scale_x.setValue(default.scale_x)
            self.sp_scale_y.setValue(default.scale_y)
            self.sp_display_aspect.setRange(aspect_min, aspect_max)
            self.sp_display_aspect.setValue(min(aspect_max, max(aspect_min, default.display_aspect)))
            self.chk_auto_aspect.setChecked(default.auto_display_aspect)
            self.cb_downsample.setCurrentIndex(0)
            self.sp_buffer_h.setValue(max(1, default.buffer_h or 1))
            self.sp_buffer_w.setValue(max(1, default.buffer_w or 1))
            self.chk_auto_buffer.setChecked(default.auto_buffer_resize)
        self.sp_display_aspect.setEnabled(not default.auto_display_aspect)
        self.btn_reset_aspect.setEnabled(not default.auto_display_aspect)
        self.sp_buffer_w.setEnabled(not default.auto_buffer_resize)

        self.state.x_axis_mode = default.x_axis_mode
        self.state.y_axis_mode = default.y_axis_mode
        with QSignalBlocker(self.cb_x_axis), QSignalBlocker(self.cb_y_axis):
            self.cb_x_axis.setCurrentIndex(0)
            self.cb_y_axis.setCurrentIndex(0)
        self.set_source_meta(None)
        self._sync_transform_state()

        # reset display filter
        with QSignalBlocker(self.chk_filter_enable):
            self.chk_filter_enable.setChecked(False)
        self.cb_filter_kind.setEnabled(True)
        self.sp_fmin.setEnabled(True)
        self.sp_fmax.setEnabled(True)
        self.sp_filter_order.setEnabled(True)
        self.btn_filter_apply.setEnabled(True)
        self.display_model.clear_display_filter()
        self._sync_filter_limits_from_source()

    def set_source_meta(self, start_time: float | None) -> None:
        """Enable or disable the 'UTC clock' X-axis option based on source availability."""
        model = self.cb_x_axis.model()
        utc_item: QStandardItem = model.item(2)
        has_utc = start_time is not None
        utc_item.setEnabled(has_utc)
        if not has_utc and self.cb_x_axis.currentIndex() == 2:
            self.cb_x_axis.setCurrentIndex(0)

        # Update channel range spinbox limits from the current source
        source = self.display_model.current_source()
        nc = 0
        if source is not None:
            shape = getattr(source, "shape", (0, 0))
            if isinstance(source, StreamingSource):
                nc = int(shape[0])  # logical (channel, time) shape
            else:
                # Raw Tape shape follows its storage layout; channels are axis 1 for nt_nc.
                nc = int(shape[1] if getattr(source, "dims", "nc_nt") == "nt_nc" else shape[0])
        if nc > 0:
            with QSignalBlocker(self.sp_ch_start), QSignalBlocker(self.sp_ch_end):
                self.sp_ch_start.setRange(0, nc - 1)
                self.sp_ch_end.setRange(1, nc)
                self.sp_ch_end.setValue(min(int(self.sp_ch_end.value()), nc))

        self._sync_filter_limits_from_source()
        self._sync_transform_state()

    def reapply_to_canvas(self):
        self._apply_all_to_canvas()

    # ---------------- Handlers ----------------
    def _on_vmin_editing_finished(self):
        if self.chk_sym.isChecked():
            amp = max(1e-6, abs(float(self.sp_vmin.value())))
            with QSignalBlocker(self.sp_vmin), QSignalBlocker(self.sp_vmax):
                self.sp_vmin.setValue(-amp)
                self.sp_vmax.setValue(amp)
            self._commit_clim()

    def _on_vmax_editing_finished(self):
        if self.chk_sym.isChecked():
            amp = max(1e-6, abs(float(self.sp_vmax.value())))
            with QSignalBlocker(self.sp_vmin), QSignalBlocker(self.sp_vmax):
                self.sp_vmin.setValue(-amp)
                self.sp_vmax.setValue(amp)
            self._commit_clim()

    def _commit_clim(self):
        vmin = float(self.sp_vmin.value())
        vmax = float(self.sp_vmax.value())

        if vmax == vmin:
            vmax = vmin + 1e-6
            with QSignalBlocker(self.sp_vmax):
                self.sp_vmax.setValue(vmax)

        if vmax < vmin:
            vmin, vmax = vmax, vmin
            with QSignalBlocker(self.sp_vmin), QSignalBlocker(self.sp_vmax):
                self.sp_vmin.setValue(vmin)
                self.sp_vmax.setValue(vmax)

        self.state.vmin, self.state.vmax = vmin, vmax
        self._apply_clim_to_canvas()

    def _on_cmap_changed(self, name: str):
        self.state.cmap = name
        self._apply_cmap_to_canvas()

    def _on_interp_changed(self, name: str):
        self.state.interpolation = name
        self._apply_interp_to_canvas()

    def _on_transform_changed(self):
        key = self.cb_transform.currentData()
        if key is None:
            return
        self.display_model.set_source_transform(str(key))

    def _on_scale_x_changed(self, value: float):
        self.state.scale_x = max(1.0, float(value))
        self.display_model.set_scale_x(self.state.scale_x)

    def _on_scale_y_changed(self, value: float):
        self.state.scale_y = max(1.0, float(value))
        self.display_model.set_scale_y(self.state.scale_y)

    def _on_display_aspect_changed(self, value: float):
        lo = float(self.state.display_aspect_min)
        hi = float(self.state.display_aspect_max)
        self.state.display_aspect = min(hi, max(lo, float(value)))
        if not self.chk_auto_aspect.isChecked():
            self.display_model.set_display_aspect(self.state.display_aspect)

    def _on_auto_aspect_toggled(self, checked: bool):
        self.state.auto_display_aspect = bool(checked)
        self.sp_display_aspect.setEnabled(not checked)
        self.btn_reset_aspect.setEnabled(not checked)
        self.display_model.set_auto_display_aspect(checked)

    def _on_reset_aspect_clicked(self):
        with QSignalBlocker(self.chk_auto_aspect):
            self.chk_auto_aspect.setChecked(False)
        self.sp_display_aspect.setEnabled(True)
        self.btn_reset_aspect.setEnabled(True)
        with QSignalBlocker(self.sp_display_aspect):
            self.sp_display_aspect.setValue(min(self.state.display_aspect_max, max(self.state.display_aspect_min, 1.0)))
        self.state.auto_display_aspect = False
        self.state.display_aspect = min(self.state.display_aspect_max, max(self.state.display_aspect_min, 1.0))
        self.display_model.set_display_aspect(self.state.display_aspect)

    def _on_downsample_mode_changed(self):
        mode = str(self.cb_downsample.currentData() or "drop")
        self.state.downsample_mode = mode
        self.display_model.set_downsample_mode(mode)

    def _on_channel_range_go(self):
        c0 = int(self.sp_ch_start.value())
        c1 = int(self.sp_ch_end.value())
        if c1 <= c0:
            c1 = c0 + 1
            with QSignalBlocker(self.sp_ch_end):
                self.sp_ch_end.setValue(c1)
        self.display_model.show_channel_range(c0, c1)

    def _on_fit_y_clicked(self):
        self.display_model.fit_y()

    def _on_reset_y_clicked(self):
        self.display_model.set_scale_y(1.0)

    def _on_buffer_h_editing_finished(self):
        value = int(self.sp_buffer_h.value())
        self.state.buffer_h = max(1, value)
        self.display_model.set_buffer_h(self.state.buffer_h)

    def _on_buffer_w_editing_finished(self):
        if self.chk_auto_buffer.isChecked():
            return
        value = int(self.sp_buffer_w.value())
        self.state.buffer_w = max(1, value)
        self.display_model.set_buffer_w(self.state.buffer_w)

    def _on_auto_buffer_toggled(self, checked: bool):
        self.state.auto_buffer_resize = bool(checked)
        self.sp_buffer_w.setEnabled(not checked)
        self.display_model.set_auto_buffer_resize(self.state.auto_buffer_resize)

    def _on_axis_mode_changed(self):
        x_idx = self.cb_x_axis.currentIndex()
        x_mode = ("time" if x_idx == 1 else "utc" if x_idx == 2 else "sample")
        y_mode = "distance" if self.cb_y_axis.currentIndex() == 1 else "channel"
        self.state.x_axis_mode = x_mode
        self.state.y_axis_mode = y_mode
        self._apply_axis_mode_to_canvas()

    def _sync_scale_state(self, state: dict):
        sx = max(1.0, float(state.get("scale_x", 1.0)))
        sy = max(1.0, float(state.get("scale_y", 1.0)))
        display_aspect_min = float(state.get("display_aspect_min", DEFAULT_MIN_DISPLAY_ASPECT) or DEFAULT_MIN_DISPLAY_ASPECT)
        display_aspect_max = float(state.get("display_aspect_max", DEFAULT_MAX_DISPLAY_ASPECT) or DEFAULT_MAX_DISPLAY_ASPECT)
        display_aspect = min(
            display_aspect_max,
            max(display_aspect_min, float(state.get("display_aspect", 1.0) or 1.0)),
        )
        auto_display_aspect = bool(state.get("auto_display_aspect", True))
        mode = str(state.get("downsample_mode", "drop"))
        buffer_h = max(1, int(state.get("buffer_h", 1) or 1))
        buffer_w = max(0, int(state.get("buffer_w", 0) or 0))
        auto_buffer_resize = bool(state.get("auto_buffer_resize", True))

        self.state.scale_x = sx
        self.state.scale_y = sy
        self.state.display_aspect = display_aspect
        self.state.auto_display_aspect = auto_display_aspect
        self.state.display_aspect_min = display_aspect_min
        self.state.display_aspect_max = display_aspect_max
        self.state.downsample_mode = mode
        self.state.buffer_h = buffer_h
        self.state.buffer_w = buffer_w
        self.state.auto_buffer_resize = auto_buffer_resize

        with QSignalBlocker(self.sp_scale_x), QSignalBlocker(self.sp_scale_y), QSignalBlocker(self.sp_display_aspect), QSignalBlocker(self.chk_auto_aspect), QSignalBlocker(self.cb_downsample), QSignalBlocker(self.sp_buffer_h), QSignalBlocker(self.sp_buffer_w), QSignalBlocker(self.chk_auto_buffer):
            self.sp_scale_x.setValue(sx)
            self.sp_scale_y.setValue(sy)
            self.sp_display_aspect.setRange(display_aspect_min, display_aspect_max)
            self.sp_display_aspect.setValue(display_aspect)
            self.chk_auto_aspect.setChecked(auto_display_aspect)
            idx = self.cb_downsample.findData(mode)
            self.cb_downsample.setCurrentIndex(idx if idx >= 0 else 0)
            self.sp_buffer_h.setValue(buffer_h)
            self.sp_buffer_w.setValue(max(1, buffer_w or 1))
            self.chk_auto_buffer.setChecked(auto_buffer_resize)
        self.sp_display_aspect.setEnabled(not auto_display_aspect)
        self.btn_reset_aspect.setEnabled(not auto_display_aspect)
        self.sp_buffer_w.setEnabled(not auto_buffer_resize)

    def _sync_transform_state(self, state: dict | None = None) -> None:
        if state is None:
            state = self.display_model.source_transform_state()
        options = list(state.get("options", []) or [])
        active = state.get("active")
        enabled = bool(state.get("enabled") and options)

        with QSignalBlocker(self.cb_transform):
            self.cb_transform.clear()
            if enabled:
                for option in options:
                    self.cb_transform.addItem(
                        str(option.get("label", option.get("key", ""))),
                        str(option.get("key", "")),
                    )
                idx = self.cb_transform.findData(str(active)) if active is not None else -1
                self.cb_transform.setCurrentIndex(idx if idx >= 0 else 0)
            else:
                self.cb_transform.addItem("Raw", "raw")
                self.cb_transform.setCurrentIndex(0)
        self.cb_transform.setEnabled(enabled and len(options) > 1)

    # ---------------- Channel markers ----------------
    def _on_marker_apply(self) -> None:
        text = self.le_marker_channels.text().strip()
        if not text:
            self.display_model.set_channel_markers([])
            return
        channels = []
        for part in text.split(","):
            part = part.strip()
            if part:
                try:
                    channels.append(int(part))
                except ValueError:
                    pass
        color = self.cb_marker_color.currentText()
        self.display_model.set_channel_markers(channels, [color] * len(channels))

    def _on_marker_clear(self) -> None:
        self.le_marker_channels.clear()
        self.display_model.set_channel_markers([])

    # ---------------- Display filter ----------------
    def _on_filter_enable_toggled(self, checked: bool):
        self._sync_filter_limits_from_source()
        if checked:
            self._on_filter_kind_changed(self.cb_filter_kind.currentText())
            self._on_filter_apply()
        else:
            self.display_model.clear_display_filter()

    def _on_display_filter_status(self, status: dict) -> None:
        suspended = str(status.get("state")) == "suspended"
        self.lbl_filter_status.setText(str(status.get("message", "")) if suspended else "")
        self.lbl_filter_status.setVisible(suspended)

    def _on_filter_kind_changed(self, kind: str):
        # grey-out irrelevant controls
        self.sp_fmin.setEnabled(kind != "lowpass")
        self.sp_fmax.setEnabled(kind != "highpass")

    def _on_filter_apply(self):
        if not self.chk_filter_enable.isChecked():
            return
        kind = self.cb_filter_kind.currentText()
        fmin = float(self.sp_fmin.value()) if kind != "lowpass" else None
        fmax = float(self.sp_fmax.value()) if kind != "highpass" else None
        order = int(self.sp_filter_order.value())
        self.display_model.set_display_filter(kind, fmin, fmax, order)

    def _sync_filter_limits_from_source(self) -> None:
        source = self.display_model.current_source()
        fs = float(getattr(source, "fs", 0.0) or 0.0) if source is not None else 0.0
        nyquist = 0.5 * fs if fs > 0.0 else 0.0
        max_cutoff = max(0.001, nyquist * 0.999) if nyquist > 0.0 else 1e6
        kind = self.cb_filter_kind.currentText() or "bandpass"

        fmin = float(self.sp_fmin.value())
        fmax = float(self.sp_fmax.value())
        default_fmax = min(250.0, max_cutoff)
        default_fmin = min(5.0, max(0.001, default_fmax * 0.5))

        fmax = max(0.001, min(fmax, max_cutoff))
        fmin = max(0.0, min(fmin, max_cutoff))

        if kind == "bandpass":
            if fmax <= 0.001:
                fmax = default_fmax
            if fmin <= 0.0:
                fmin = default_fmin
            if fmin >= fmax:
                fmin = max(0.001, min(default_fmin, fmax * 0.5))
            if fmin >= fmax:
                fmax = min(max_cutoff, max(fmin * 1.5, fmin + 0.001))
        elif kind == "lowpass":
            if fmax <= 0.001:
                fmax = default_fmax
        else:
            if fmin <= 0.0:
                fmin = default_fmin
            if fmin >= max_cutoff:
                fmin = max(0.001, min(default_fmin, max_cutoff * 0.5))

        with QSignalBlocker(self.sp_fmin), QSignalBlocker(self.sp_fmax):
            self.sp_fmin.setRange(0.0, max_cutoff)
            self.sp_fmax.setRange(0.001, max_cutoff)
            self.sp_fmin.setValue(fmin)
            self.sp_fmax.setValue(fmax)

    # ---------------- Canvas apply ----------------
    def _apply_all_to_canvas(self):
        self._apply_clim_to_canvas()
        self._apply_cmap_to_canvas()
        self._apply_interp_to_canvas()
        self._apply_display_aspect_to_canvas()
        self._apply_axis_mode_to_canvas()

    def _apply_clim_to_canvas(self):
        self.display_model.set_clim(self.state.vmin, self.state.vmax)

    def _apply_cmap_to_canvas(self):
        self.display_model.set_cmap(self.state.cmap)

    def _apply_interp_to_canvas(self):
        self.display_model.set_interpolation(self.state.interpolation)

    def _apply_display_aspect_to_canvas(self):
        self.display_model.set_auto_display_aspect(self.state.auto_display_aspect)
        if not self.state.auto_display_aspect:
            self.display_model.set_display_aspect(self.state.display_aspect)

    def _apply_axis_mode_to_canvas(self):
        self.display_model.set_axis_mode(self.state.x_axis_mode, self.state.y_axis_mode)
