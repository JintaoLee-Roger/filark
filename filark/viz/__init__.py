# Copyright (c) 2026 Jintao Li. 
# Zhejiang University (ZJU).
# 
# Licensed under the MIT License.

"""Vispy-based visualization stack.

The visualization layer owns the render-time model of the data:

- ``StreamingImage`` for GPU-backed texture/ring-buffer storage
- ``StreamingCamera`` for view constraints and navigation invariants
- ``BufferScheduler2D`` for deciding what data block to read next
- ``StreamingCanvas`` for wiring view, camera, scheduler, and overlays together

This package should stay ignorant of file-format details and GUI drawer state.
It consumes ``StreamingSource`` objects and focuses only on display behavior.
"""

from .visual import (
    StreamingImage,
    StreamingImageVisual,
    AxisWidget,
    AnnotationBBox,
    AnnotationBBoxVisual,
    AnnotationPolyline,
    AnnotationPolylineVisual,
    MaskOverlay,
    MaskOverlayVisual,
)
from .camera import StreamingCamera
from .scheduler import BufferScheduler2D
from .canvas import StreamingCanvas

__all__ = [
    "StreamingImage",
    "StreamingImageVisual",
    "StreamingCamera",
    "AxisWidget",
    "BufferScheduler2D",
    "StreamingCanvas",
    "AnnotationBBoxVisual",
    "AnnotationBBox",
    "AnnotationPolylineVisual",
    "AnnotationPolyline",
    "MaskOverlayVisual",
    "MaskOverlay",
]
