#!/usr/bin/env python3
"""Source localization from first arrivals, as two FiLark analysis plugins.

The same algorithm is exposed in both forms an analysis plugin can take:

* ``SourceLocationFigure`` (mode 1): runs on the selected ROI and returns a
  matplotlib figure; FiLark builds the parameter dialog and the figure window.
* ``SourceLocationWidgetPlugin`` (mode 2): provides ``SourceLocationWidget``,
  its own Qt widget that reads the current ROI or the current view, re-runs on
  demand, and marks the located channel on the main canvas.

Algorithm (for a point source near a straight cable, e.g. a sparker shot):

1. band-pass filter the selected data;
2. pick the first arrival on each channel as the first sample, before the
   envelope peak, that exceeds ``onset_frac`` of that peak; channels whose
   envelope peak is below ``snr_min`` times its median are skipped;
3. fit the direct-wave hyperbola t(x) = t0 + sqrt(d^2 + (x - x0)^2) / v
   with a robust least-squares loss and a fixed velocity v.

x0 is the source position along the cable and d its distance from the cable.
With one cable, d does not tell on which side of the cable the source lies.

Usage::

    python scripts/source_location_demo.py -f /path/to/record.h5

Then select an ROI around the event and choose
Analyze > Source Location (figure) or Analyze > Source Location (widget).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import uniform_filter
from scipy.optimize import least_squares
from scipy.signal import hilbert

from filark.dsp.temporal import iir_bandpass_time
from filark.gui.plugins.analysis import AnalysisPlugin


# ----------------------------------------------------------------------
# Algorithm
# ----------------------------------------------------------------------

PARAM_SPEC = [
    {"key": "fmin", "label": "Low cut (Hz)", "type": "float", "default": 30.0,
     "min": 0.1, "max": 10000.0, "step": 1.0},
    {"key": "fmax", "label": "High cut (Hz)", "type": "float", "default": 200.0,
     "min": 0.1, "max": 10000.0, "step": 1.0},
    {"key": "velocity", "label": "Velocity (m/s)", "type": "float", "default": 1500.0,
     "min": 100.0, "max": 10000.0, "step": 10.0},
    {"key": "snr_min", "label": "Min. envelope SNR", "type": "float", "default": 4.0,
     "min": 1.0, "max": 100.0, "step": 0.5},
    {"key": "onset_frac", "label": "Onset level (x peak)", "type": "float", "default": 0.3,
     "min": 0.05, "max": 0.95, "step": 0.05},
]


@dataclass
class Location:
    filtered: np.ndarray          # [time, channel], each channel scaled to unit std
    t_start: int                  # absolute sample index of filtered[0]
    c_start: int                  # absolute channel index of filtered[:, 0]
    fs: float
    dx: float
    pick_channels: np.ndarray     # absolute channel indices
    pick_samples: np.ndarray      # absolute sample indices
    x0_channel: float
    x0_se_channel: float
    distance_m: float
    distance_se_m: float
    origin_s: float               # origin time, s from t_start
    velocity: float
    rms_ms: float

    def arrival_samples(self, channels: np.ndarray) -> np.ndarray:
        """Fitted arrival time (absolute sample index) at the given channels."""
        x = (np.asarray(channels, dtype=float) - self.x0_channel) * self.dx
        t = self.origin_s + np.sqrt(self.distance_m ** 2 + x ** 2) / self.velocity
        return self.t_start + t * self.fs

    def summary(self) -> str:
        return (
            f"source at channel {self.x0_channel:.1f} ± {self.x0_se_channel:.1f} "
            f"({self.x0_channel * self.dx:.0f} m), "
            f"{self.distance_m:.0f} ± {self.distance_se_m:.0f} m from the cable; "
            f"{self.pick_channels.size} picks, RMS {self.rms_ms:.1f} ms"
        )


def pick_first_arrivals(x: np.ndarray, *, snr_min: float, onset_frac: float):
    """Return (channel index, onset sample) of the channels with a clear arrival."""
    env = uniform_filter(np.abs(hilbert(x, axis=0)), size=(5, 3))
    peak = env.max(axis=0)
    snr = peak / (np.median(env, axis=0) + 1e-20)
    t_peak = env.argmax(axis=0)
    channels = np.flatnonzero(snr >= snr_min)
    onsets = np.empty(channels.size, dtype=int)
    for i, c in enumerate(channels):
        below = np.flatnonzero(env[: t_peak[c] + 1, c] < onset_frac * peak[c])
        onsets[i] = below[-1] + 1 if below.size else 0
    return channels, onsets


def locate_source(
    data,
    *,
    fs: float,
    dx: float,
    t_start: int = 0,
    c_start: int = 0,
    pad: tuple[int, int] = (0, 0),
    fmin: float = 30.0,
    fmax: float = 200.0,
    velocity: float = 1500.0,
    snr_min: float = 4.0,
    onset_frac: float = 0.3,
) -> Location:
    """Locate a point source from a [time, channel] window around one event.

    ``pad`` gives the number of extra samples before and after the window that
    ``data`` contains; they absorb the filter transients and are cut off before
    picking. ``t_start`` is the absolute sample index of the window itself.
    """
    x = iir_bandpass_time(np.asarray(data, dtype=np.float32), fs=fs, fmin=fmin, fmax=fmax)
    x = x[pad[0]: x.shape[0] - pad[1]]
    x = x / (x.std(axis=0, keepdims=True) + 1e-20)

    ch, onset = pick_first_arrivals(x, snr_min=snr_min, onset_frac=onset_frac)
    if ch.size < 5:
        raise ValueError(
            f"Only {ch.size} channels have a clear arrival; "
            "select a region around one event or lower the SNR threshold."
        )
    xs = ch * dx
    ts = onset / fs

    def residual(p):
        x0, d, t0 = p
        return t0 + np.sqrt(d * d + (xs - x0) ** 2) / velocity - ts

    i = int(np.argmin(ts))
    p0 = [xs[i], 10.0, ts[i] - 10.0 / velocity]
    res = least_squares(
        residual, p0, loss="soft_l1", f_scale=max(3e-3, 2.0 / fs),
        bounds=([xs.min() - 1e4, 0.0, -np.inf], [xs.max() + 1e4, np.inf, np.inf]),
    )
    r = residual(res.x)
    cov = np.linalg.pinv(res.jac.T @ res.jac) * np.mean(r ** 2)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    x0, d, t0 = res.x
    nc = x.shape[1]
    if not (0.0 <= x0 / dx <= nc - 1) or d > 2.0 * nc * dx:
        raise ValueError(
            "No consistent direct-wave hyperbola in the selected data; "
            "select a region that contains the apex of one event."
        )
    return Location(
        filtered=x, t_start=int(t_start), c_start=int(c_start), fs=float(fs), dx=float(dx),
        pick_channels=ch + int(c_start), pick_samples=onset + int(t_start),
        x0_channel=x0 / dx + c_start, x0_se_channel=se[0] / dx,
        distance_m=float(d), distance_se_m=float(se[1]),
        origin_s=float(t0), velocity=float(velocity),
        rms_ms=1e3 * float(np.sqrt(np.mean(r ** 2))),
    )


def plot_location(fig, loc: Location) -> None:
    """Draw the filtered data, the picks, and the fitted hyperbola."""
    fig.clear()
    ax = fig.add_subplot(111)
    nt, nc = loc.filtered.shape
    t0, c0 = loc.t_start, loc.c_start
    ax.imshow(
        loc.filtered.T, aspect="auto", origin="lower", cmap="seismic", vmin=-4, vmax=4,
        extent=[t0 - 0.5, t0 + nt - 0.5, c0 - 0.5, c0 + nc - 0.5], interpolation="nearest",
    )
    ax.plot(loc.pick_samples, loc.pick_channels, "k.", ms=4, label="first-arrival picks")
    cc = np.linspace(c0, c0 + nc - 1, 400)
    tt = loc.arrival_samples(cc)
    keep = tt <= t0 + nt
    ax.plot(tt[keep], cc[keep], "g-", lw=1.5, label="fitted direct wave")
    ax.axhline(loc.x0_channel, color="g", ls=":", lw=1)
    ax.set_xlim(t0 - 0.5, t0 + nt - 0.5)
    ax.set_ylim(c0 - 0.5, c0 + nc - 0.5)
    ax.set_xlabel("Time (sample)")
    ax.set_ylabel("Channel")
    ax.set_title(loc.summary(), fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()


def _source_fs_dx(source) -> tuple[float, float]:
    fs = float(getattr(source, "fs", None) or 1.0)
    dx = float(getattr(source, "dx", None) or 1.0)
    return fs, dx


def _read_block(source, t0: int, t1: int, c0: int, c1: int) -> np.ndarray:
    if getattr(source, "dims", "nt_nc") == "nt_nc":
        return np.asarray(source[t0:t1, c0:c1])
    return np.asarray(source[c0:c1, t0:t1]).T


def _nt(source) -> int:
    return int(source.shape[0] if getattr(source, "dims", "nt_nc") == "nt_nc" else source.shape[1])


def locate_in_source(source, t0: int, t1: int, c0: int, c1: int, *, fs: float, dx: float,
                     params: dict) -> Location:
    """Read the window with 0.2 s of padding on each side and locate the source."""
    p = int(round(0.2 * fs))
    a, b = max(0, t0 - p), min(_nt(source), t1 + p)
    data = _read_block(source, a, b, c0, c1)
    return locate_source(data, fs=fs, dx=dx, t_start=t0, c_start=c0,
                         pad=(t0 - a, b - t1), **params)


# ----------------------------------------------------------------------
# Mode 1: FiLark builds the parameter dialog and the figure window
# ----------------------------------------------------------------------

class SourceLocationFigure(AnalysisPlugin):
    label = "Source Location (figure)"
    key = "source_location_figure"
    param_spec = PARAM_SPEC

    def run(self, source, *, params, roi=None):
        if roi is None:
            raise ValueError("Select an ROI around the event first.")
        r = roi["roi"]
        loc = locate_in_source(source, r.t0, r.t1, r.c0, r.c1,
                               fs=float(roi["fs"]), dx=float(roi["dx"]), params=params)
        print(f"[source location] {loc.summary()}", flush=True)
        fig = Figure(figsize=(8, 6))
        plot_location(fig, loc)
        return fig


# ----------------------------------------------------------------------
# Mode 2: the plugin provides its own widget
# ----------------------------------------------------------------------

class SourceLocationWidgetPlugin(AnalysisPlugin):
    label = "Source Location (widget)"
    key = "source_location_widget"
    param_spec: list[dict] = []

    def create_widget(self, source, parent=None):
        return SourceLocationWidget(source, parent=parent)


class SourceLocationWidget(QWidget):
    """Parameter panel, locate button, result line, and an embedded figure."""

    MAX_SAMPLES = 20_000_000   # refuse to read more than this from the current view

    def __init__(self, source, parent=None):
        super().__init__(parent)
        self._source = source
        self._context: dict = {}

        self._spins: dict[str, QDoubleSpinBox] = {}
        form = QFormLayout()
        self._region = QComboBox()
        self._region.addItems(["Selected ROI", "Current view"])
        form.addRow("Data", self._region)
        for spec in PARAM_SPEC:
            sb = QDoubleSpinBox()
            sb.setRange(float(spec["min"]), float(spec["max"]))
            sb.setSingleStep(float(spec["step"]))
            sb.setDecimals(2)
            sb.setValue(float(spec["default"]))
            self._spins[spec["key"]] = sb
            form.addRow(spec["label"], sb)

        self._btn = QPushButton("Locate")
        self._btn.clicked.connect(self._locate)
        self._status = QLabel("Select an ROI around one event, then press Locate.")
        self._status.setWordWrap(True)

        left = QVBoxLayout()
        left.addLayout(form)
        left.addWidget(self._btn)
        left.addWidget(self._status)
        left.addStretch(1)
        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setFixedWidth(260)

        self._fig = Figure(figsize=(6, 4))
        self._canvas = FigureCanvasQTAgg(self._fig)
        right = QVBoxLayout()
        right.addWidget(NavigationToolbar2QT(self._canvas, self))
        right.addWidget(self._canvas, 1)

        lay = QHBoxLayout(self)
        lay.addWidget(left_w)
        lay.addLayout(right, 1)

    def set_filark_context(self, context: dict | None) -> None:
        self._context = dict(context or {})

    def _region_data(self):
        if self._region.currentIndex() == 0:
            getter = self._context.get("roi_getter")
            roi = getter() if callable(getter) else None
            if roi is None:
                raise ValueError("No ROI selected.")
            r = roi["roi"]
            return r.t0, r.t1, r.c0, r.c1, float(roi["fs"]), float(roi["dx"])
        getter = self._context.get("current_view_bounds_getter")
        b = getter() if callable(getter) else None
        if not b:
            raise ValueError("Current view is not available.")
        t0, t1, c0, c1 = (int(b[k]) for k in ("t0", "t1", "c0", "c1"))
        if (t1 - t0) * (c1 - c0) > self.MAX_SAMPLES:
            raise ValueError("The current view is too large; zoom in to the event.")
        fs, dx = _source_fs_dx(self._source)
        return t0, t1, c0, c1, fs, dx

    def _locate(self) -> None:
        try:
            t0, t1, c0, c1, fs, dx = self._region_data()
            params = {k: sb.value() for k, sb in self._spins.items()}
            loc = locate_in_source(self._source, t0, t1, c0, c1, fs=fs, dx=dx, params=params)
        except Exception as exc:
            self._status.setText(f"<span style='color:#c33'>{exc}</span>")
            return
        plot_location(self._fig, loc)
        self._canvas.draw_idle()
        self._status.setText(loc.summary())
        marker = self._context.get("set_channel_marker")
        if callable(marker):
            marker(int(round(loc.x0_channel)))

    def closeEvent(self, event) -> None:
        clear = self._context.get("clear_channel_marker")
        if callable(clear):
            clear()
        super().closeEvent(event)


# ----------------------------------------------------------------------
# Launcher
# ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    from filark.gui.app import run_app
    from filark.gui.config import GuiConfig

    cfg = GuiConfig(
        analyses=[SourceLocationFigure(), SourceLocationWidgetPlugin()],
        analysis_widget_size=(900, 560),
    )
    run_app(argv, cfg=cfg)


if __name__ == "__main__":
    main()
