"""Built-in analysis plugins shipped with FiLark."""

from __future__ import annotations

from filark.gui.plugins.analysis import AnalysisPlugin
from filark.gui.widgets.signal_viewer import SignalViewer


class SignalViewerAnalysis(AnalysisPlugin):
    """1-D signal viewer: waveform, filter, PSD and spectrogram."""

    label = "Signal Viewer"
    key = "signal_viewer"
    param_spec: list[dict] = []

    def create_widget(self, source, parent=None):
        return SignalViewer(source, parent=parent)


def default_analyses() -> list:
    return [SignalViewerAnalysis()]
