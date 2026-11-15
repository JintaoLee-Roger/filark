# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""Annotation-specific bbox visual for viewer overlays and editing."""

from __future__ import annotations

from typing import Optional

import numpy as np
from vispy import visuals
from vispy.scene.visuals import create_visual_node
from vispy.visuals import CompoundVisual

from filark._exceptions import print_exception
from ._annotation_visual_utils import bbox_handle_points, normalize_bbox

__all__ = ["AnnotationBBoxVisual", "AnnotationBBox"]


class AnnotationBBoxVisual(CompoundVisual):
    """Compound bbox visual with an outline rectangle and optional handles."""

    def __init__(
        self,
        *,
        color: str = "yellow",
        fill_color=(0, 0, 0, 0.001),
        border_width: float = 1.0,
    ) -> None:
        self._rect = visuals.RectangleVisual(
            center=(0.0, 0.0),
            width=1.0,
            height=1.0,
            color=fill_color,
            border_color=color,
            border_width=float(border_width),
        )
        self._handles = visuals.MarkersVisual()
        super().__init__([self._rect, self._handles])
        self.unfreeze()
        self.set_gl_state("translucent", depth_test=False, cull_face=False)

        self._bbox: Optional[dict[str, float]] = None
        self._visible = True
        self._handles_visible = False
        self._color = str(color)
        self._border_width = float(border_width)
        self._handle_face = "#FFFFFF"
        self._handle_edge = str(color)
        self._handle_size = 8.0
        self.clear()
        self.freeze()

    def close(self) -> None:
        try:
            self.parent = None
        except Exception:
            print_exception("Failed to detach AnnotationBBoxVisual during close().", prefix="AnnotationBBoxVisual")

    def clear(self) -> None:
        self._bbox = None
        self._rect.visible = False
        self._handles.set_data(pos=np.zeros((1, 2), dtype=np.float32), size=self._handle_size)
        self._handles.visible = False
        self.update()

    def set_visible(self, visible: bool) -> None:
        self._visible = bool(visible)
        self._sync()

    def set_color(self, color: str) -> None:
        self._color = str(color)
        self._sync()

    def set_border_width(self, width: float) -> None:
        self._border_width = max(1.0, float(width))
        self._sync()

    def set_handle_style(
        self,
        *,
        face_color: Optional[str] = None,
        edge_color: Optional[str] = None,
        size: Optional[float] = None,
    ) -> None:
        if face_color is not None:
            self._handle_face = str(face_color)
        if edge_color is not None:
            self._handle_edge = str(edge_color)
        if size is not None:
            self._handle_size = float(size)
        self._sync()

    def set_handles_visible(self, visible: bool) -> None:
        self._handles_visible = bool(visible)
        self._sync()

    def set_bbox(self, bbox: dict | None) -> None:
        self._bbox = normalize_bbox(bbox)
        self._sync()

    def show(
        self,
        bbox: dict | None,
        *,
        color: Optional[str] = None,
        border_width: Optional[float] = None,
        visible: bool = True,
        handles: bool = False,
        handle_face: Optional[str] = None,
        handle_edge: Optional[str] = None,
        handle_size: Optional[float] = None,
    ) -> None:
        if color is not None:
            self._color = str(color)
        if border_width is not None:
            self._border_width = max(1.0, float(border_width))
        if handle_face is not None or handle_edge is not None or handle_size is not None:
            self.set_handle_style(face_color=handle_face, edge_color=handle_edge, size=handle_size)
        self._visible = bool(visible)
        self._handles_visible = bool(handles)
        self._bbox = normalize_bbox(bbox)
        self._sync()

    def _sync(self) -> None:
        bbox = self._bbox
        if bbox is None or not self._visible:
            self._rect.visible = False
            self._handles.visible = False
            self.update()
            return

        x0, y0, x1, y1 = bbox["t0"], bbox["c0"], bbox["t1"], bbox["c1"]
        self._rect.center = (0.5 * (x0 + x1), 0.5 * (y0 + y1))
        self._rect.width = max(abs(x1 - x0), 1e-6)
        self._rect.height = max(abs(y1 - y0), 1e-6)
        self._rect.border_color = self._color
        self._rect._border_width = self._border_width
        self._rect._update()
        self._rect.visible = True

        if self._handles_visible:
            self._handles.set_data(
                pos=bbox_handle_points(bbox),
                face_color=self._handle_face,
                edge_color=self._handle_edge,
                edge_width=1.2,
                size=self._handle_size,
            )
            self._handles.visible = True
        else:
            self._handles.visible = False
        self.update()

AnnotationBBox = create_visual_node(AnnotationBBoxVisual)
AnnotationBBox.__name__ = "AnnotationBBox"
AnnotationBBox.__qualname__ = "AnnotationBBox"
