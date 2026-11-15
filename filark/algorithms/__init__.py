"""Algorithm-oriented workflows built on top of lower-level packages.

This package is reserved for higher-level analysis or detection workflows that
combine multiple DSP operators and domain heuristics.  In contrast:

- ``dsp`` contains reusable numerical operators
- ``processing`` contains user-facing processing APIs and processor schemas
- ``algorithms`` should contain end-to-end domain logic when such workflows
  mature beyond a single operator
"""

from .energy_detection import (
    BBoxDetectionEvent,
    BasePatchDetector,
    RelativeEnergyDetector,
    STALTADetector,
    TKEODetector,
    build_energy_patch_detector,
    merge_bbox_events,
    run_patch_detector_on_tape,
    EnergyDetectionMonitor,
)

__all__ = [
    "BBoxDetectionEvent",
    "BasePatchDetector",
    "RelativeEnergyDetector",
    "STALTADetector",
    "TKEODetector",
    "build_energy_patch_detector",
    "merge_bbox_events",
    "run_patch_detector_on_tape",
    "EnergyDetectionMonitor",
]
