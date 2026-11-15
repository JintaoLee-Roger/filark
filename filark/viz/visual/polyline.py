# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""Annotation-specific polyline visual for viewer overlays and editing."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from vispy import visuals
from vispy.scene.visuals import create_visual_node
from vispy.visuals import CompoundVisual

from filark._exceptions import print_exception
from ._annotation_visual_utils import coerce_points

__all__ = ["AnnotationPolylineVisual", "AnnotationPolyline"]


class AnnotationPolylineVisual(CompoundVisual):
    """Compound polyline visual with line, markers, and cursor preview."""

    def __init__(
        self,
        *,
        color: str = "yellow",
        width: float = 2.0,
    ) -> None:
        self._line = visuals.LineVisual(
            pos=np.zeros((2, 2), dtype=np.float32),
            color=color,
            method="agg",
            antialias=True,
            connect="strip",
            width=width,
        )
        self._cursor = visuals.LineVisual(
            pos=np.zeros((2, 2), dtype=np.float32),
            color=color,
            method="agg",
            antialias=True,
            connect="strip",
            width=width,
        )
        self._markers = visuals.MarkersVisual()
        super().__init__([self._line, self._cursor, self._markers])
        self.unfreeze()
        self.set_gl_state("translucent", depth_test=False, cull_face=False)

        self._visible = True
        self._points = np.zeros((0, 2), dtype=np.float32)
        self._cursor_segment = np.zeros((0, 2), dtype=np.float32)
        self._line_color = str(color)
        self._line_width = float(width)
        self._markers_visible = False
        self._marker_face = "#FFFFFF"
        self._marker_edge = "#111111"
        self._marker_size = 7.0
        self._cursor_color = str(color)
        self._cursor_width = float(width)
        self.clear()
        self.freeze()

    def close(self) -> None:
        try:
            self.parent = None
        except Exception:
            print_exception(
                "Failed to detach AnnotationPolylineVisual during close().",
                prefix="AnnotationPolylineVisual",
            )

    def clear(self) -> None:
        self._points = np.zeros((0, 2), dtype=np.float32)
        self._cursor_segment = np.zeros((0, 2), dtype=np.float32)
        self._line.visible = False
        self._cursor.visible = False
        self._markers.set_data(pos=np.zeros((1, 2), dtype=np.float32), size=self._marker_size)
        self._markers.visible = False
        self.update()

    def set_visible(self, visible: bool) -> None:
        self._visible = bool(visible)
        self._sync()

    def set_points(self, points: Iterable[tuple[float, float]] | np.ndarray | None) -> None:
        self._points = coerce_points(points)
        self._sync()

    def set_cursor_segment(self, points: Iterable[tuple[float, float]] | np.ndarray | None) -> None:
        arr = coerce_points(points)
        self._cursor_segment = arr if len(arr) == 2 else np.zeros((0, 2), dtype=np.float32)
        self._sync()

    def set_line_style(self, *, color: Optional[str] = None, width: Optional[float] = None) -> None:
        if color is not None:
            self._line_color = str(color)
        if width is not None:
            self._line_width = float(width)
        self._sync()

    def set_marker_style(
        self,
        *,
        face_color: Optional[str] = None,
        edge_color: Optional[str] = None,
        size: Optional[float] = None,
    ) -> None:
        if face_color is not None:
            self._marker_face = str(face_color)
        if edge_color is not None:
            self._marker_edge = str(edge_color)
        if size is not None:
            self._marker_size = float(size)
        self._sync()

    def set_cursor_style(self, *, color: Optional[str] = None, width: Optional[float] = None) -> None:
        if color is not None:
            self._cursor_color = str(color)
        if width is not None:
            self._cursor_width = float(width)
        self._sync()

    def set_markers_visible(self, visible: bool) -> None:
        self._markers_visible = bool(visible)
        self._sync()

    def show(
        self,
        points: Iterable[tuple[float, float]] | np.ndarray | None,
        *,
        color: Optional[str] = None,
        width: Optional[float] = None,
        visible: bool = True,
        show_markers: bool = False,
        marker_face: Optional[str] = None,
        marker_edge: Optional[str] = None,
        marker_size: Optional[float] = None,
        cursor_segment: Iterable[tuple[float, float]] | np.ndarray | None = None,
        cursor_color: Optional[str] = None,
        cursor_width: Optional[float] = None,
    ) -> None:
        if color is not None or width is not None:
            self.set_line_style(color=color, width=width)
        if marker_face is not None or marker_edge is not None or marker_size is not None:
            self.set_marker_style(face_color=marker_face, edge_color=marker_edge, size=marker_size)
        if cursor_color is not None or cursor_width is not None:
            self.set_cursor_style(color=cursor_color, width=cursor_width)
        self._visible = bool(visible)
        self._markers_visible = bool(show_markers)
        self._points = coerce_points(points)
        cursor_arr = coerce_points(cursor_segment)
        self._cursor_segment = cursor_arr if len(cursor_arr) == 2 else np.zeros((0, 2), dtype=np.float32)
        self._sync()

    def _sync(self) -> None:
        if not self._visible or len(self._points) < 2:
            self._line.visible = False
        else:
            self._line.set_data(
                pos=self._points,
                color=self._line_color,
                connect="strip",
                width=self._line_width,
            )
            self._line.visible = True

        if self._visible and len(self._cursor_segment) == 2:
            self._cursor.set_data(
                pos=self._cursor_segment,
                color=self._cursor_color,
                connect="strip",
                width=self._cursor_width,
            )
            self._cursor.visible = True
        else:
            self._cursor.visible = False

        if self._visible and self._markers_visible and len(self._points) >= 1:
            self._markers.set_data(
                pos=self._points,
                face_color=self._marker_face,
                edge_color=self._marker_edge,
                edge_width=1.2,
                size=self._marker_size,
            )
            self._markers.visible = True
        else:
            self._markers.visible = False
        self.update()

AnnotationPolyline = create_visual_node(AnnotationPolylineVisual)
AnnotationPolyline.__name__ = "AnnotationPolyline"
AnnotationPolyline.__qualname__ = "AnnotationPolyline"
