"""ROI-specific windows and workflows.

The ROI subsystem is the bridge between a selected region on the main canvas
and secondary windows that inspect, process, or export that region.
"""

from .roi_window import ROIWindow

__all__ = ["ROIWindow"]
