"""Panel widgets and windows of the main desktop GUI.

Left drawer (things that act on the picture currently on the canvas):

- ``VisualsPanel`` controls rendering and view-scale parameters
- ``AnnotationPanel`` manages annotation intent and persistence
- ``ROIPanel`` holds the region the Analyze menu works on

Windows opened from the menu bar:

- ``LoadPanel`` / ``SourcePropertiesDialog`` load sources and show/override metadata
- ``DetectEventsWindow`` runs monitor (detector) plugins

They should stay thin: collect parameters, emit signals, and delegate real
work to ``DisplayModel``, ``processing``, or ``pipeline`` layers.
"""

from .load import LoadPanel, SourcePropertiesDialog
from .visuals import VisualsPanel
from .annotation import AnnotationPanel, LabelDialog
from .roi_panel import ROI, ROIPanel
from .detect_events import DetectEventsWindow

__all__ = [
    "LoadPanel",
    "SourcePropertiesDialog",
    "VisualsPanel",
    "AnnotationPanel",
    "LabelDialog",
    "ROI",
    "ROIPanel",
    "DetectEventsWindow",
]
