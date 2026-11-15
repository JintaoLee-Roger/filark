"""GUI plugin protocols.

Only plugins whose lifecycle is fundamentally GUI-facing live here:

- ``MonitorPlugin`` scans a source and returns event-like overlays
- ``AnalysisPlugin`` creates figures/widgets for exploratory inspection

Processing plugins intentionally do *not* live here; they belong to
``filark.processing`` because they are shared between GUI and non-GUI flows.
"""

from .monitor import MonitorPlugin
from .analysis import AnalysisPlugin

__all__ = ["MonitorPlugin", "AnalysisPlugin"]
