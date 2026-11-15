"""Source-location example plugin: recover a known source from a synthetic shot."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "source_location_demo",
    Path(__file__).resolve().parents[1] / "scripts" / "source_location_demo.py",
)
demo = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = demo
_SPEC.loader.exec_module(demo)

FS, DX, V = 1550.0, 5.0, 1500.0


def synthetic_shot(x0_ch=60.3, d=40.0, t0=0.25, nt=1200, nc=120, seed=0):
    """Ricker wavelets on a direct-wave hyperbola plus Gaussian noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(nt) / FS
    ch = np.arange(nc)
    arrival = t0 + np.sqrt(d ** 2 + ((ch - x0_ch) * DX) ** 2) / V
    f0 = 80.0
    tau = t[:, None] - arrival[None, :] - 1.2 / f0      # onset ~1.2/f0 before the peak
    a = (np.pi * f0 * tau) ** 2
    data = (1 - 2 * a) * np.exp(-a)
    data += 0.05 * rng.standard_normal(data.shape)
    return data.astype(np.float32)


class _ArraySource:
    dims = "nt_nc"
    fs, dx = FS, DX

    def __init__(self, data):
        self._d = data
        self.shape = data.shape

    def __getitem__(self, key):
        return self._d[key]


def test_recovers_synthetic_source():
    src = _ArraySource(synthetic_shot())
    loc = demo.locate_in_source(
        src, 200, 1000, 0, 120, fs=FS, dx=DX,
        params=dict(fmin=20.0, fmax=200.0, velocity=V, snr_min=4.0, onset_frac=0.3),
    )
    assert abs(loc.x0_channel - 60.3) < 0.5
    assert abs(loc.distance_m - 40.0) < 8.0
    assert loc.pick_channels.size > 50


def test_rejects_window_without_event():
    src = _ArraySource(np.random.default_rng(1).standard_normal((1200, 120)).astype(np.float32))
    with pytest.raises(ValueError):
        demo.locate_in_source(
            src, 200, 1000, 0, 120, fs=FS, dx=DX,
            params=dict(fmin=20.0, fmax=200.0, velocity=V, snr_min=6.0, onset_frac=0.3),
        )


def test_figure_plugin_returns_figure():
    from filark.gui.modules.roi_panel import ROI

    src = _ArraySource(synthetic_shot())
    roi = {"data": None, "fs": FS, "dx": DX, "roi": ROI(200, 1000, 0, 120)}
    plugin = demo.SourceLocationFigure()
    fig = plugin.run(src, params=plugin.default_params(), roi=roi)
    assert fig.axes and "source at channel" in fig.axes[0].get_title()


def test_matplotlib_wait_cursor_disabled():
    from matplotlib.backend_bases import NavigationToolbar2

    from filark.gui.mpl_cursor import disable_matplotlib_wait_cursor

    disable_matplotlib_wait_cursor()

    class _Canvas:
        def set_cursor(self, cursor):
            raise AssertionError("wait cursor must not be set")

    tb = NavigationToolbar2.__new__(NavigationToolbar2)
    tb.canvas = _Canvas()
    with tb._wait_cursor_for_draw_cm():
        pass
