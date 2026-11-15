# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""Fixed-capacity mask overlay visual for detection-style raster patches."""

from __future__ import annotations

from typing import Optional

import numpy as np
from vispy.color import Color
from vispy.gloo import Texture2D, VertexBuffer
from vispy.scene.visuals import create_visual_node
from vispy.visuals.visual import Visual

from filark._exceptions import print_exception

__all__ = ["MaskOverlayVisual", "MaskOverlay"]


_VERTEX_SHADER = r"""
attribute vec2 a_position;
attribute vec2 a_texcoord;
varying vec2 v_texcoord;

void main() {
    v_texcoord = a_texcoord;
    gl_Position = $transform(vec4(a_position, 0.0, 1.0));
}
"""


_FRAGMENT_SHADER = r"""
uniform sampler2D u_texture;
uniform vec4 u_color;
uniform float u_alpha;
uniform float u_threshold;
varying vec2 v_texcoord;

void main() {
    float value = texture2D(u_texture, v_texcoord).r;
    if (!(value > u_threshold)) {
        discard;
    }
    float alpha = clamp(value, 0.0, 1.0) * clamp(u_alpha, 0.0, 1.0) * u_color.a;
    if (alpha <= 0.0) {
        discard;
    }
    gl_FragColor = vec4(u_color.rgb, alpha);
}
"""


class MaskOverlayVisual(Visual):
    """Display one world-coordinate mask patch using a fixed-size texture.

    Input masks use semantic ``(time, channel)`` layout.  The texture stores
    them transposed as ``(channel, time)`` so the X axis remains time and Y axis
    remains channel in world coordinates.
    """

    def __init__(
        self,
        *,
        capacity_t: int = 512,
        capacity_c: int = 512,
        color: str = "#39FF14",
        alpha: float = 0.45,
        threshold: float = 0.0,
        interpolation: str = "nearest",
        **kwargs,
    ) -> None:
        self._capacity_t = int(max(1, capacity_t))
        self._capacity_c = int(max(1, capacity_c))
        self._visible = False
        self._color = str(color)
        self._alpha = float(alpha)
        self._threshold = float(threshold)
        self._bounds = (0.0, 0.0, 1.0, 1.0)
        self._valid_shape = (0, 0)

        Visual.__init__(self, vcode=_VERTEX_SHADER, fcode=_FRAGMENT_SHADER, **kwargs)
        self._draw_mode = "triangle_strip"

        verts = np.zeros((4, 2), dtype=np.float32)
        tex = np.zeros((4, 2), dtype=np.float32)
        self._v_position = VertexBuffer(verts)
        self._v_texcoord = VertexBuffer(tex)
        self._texture = Texture2D(
            np.zeros((self._capacity_c, self._capacity_t), dtype=np.float32),
            internalformat="r32f",
            format="red",
            interpolation=interpolation,
            wrapping=("clamp_to_edge", "clamp_to_edge"),
        )

        self.shared_program["a_position"] = self._v_position
        self.shared_program["a_texcoord"] = self._v_texcoord
        self.shared_program["u_texture"] = self._texture
        self._sync_uniforms()

        self.set_gl_state("translucent", depth_test=False, cull_face=False)
        self.clear()
        self.freeze()

    @property
    def capacity(self) -> tuple[int, int]:
        """Texture capacity as ``(time, channel)``."""
        return self._capacity_t, self._capacity_c

    def close(self) -> None:
        try:
            self.parent = None
        except Exception:
            print_exception("Failed to detach MaskOverlayVisual during close().", prefix="MaskOverlayVisual")

    def clear(self) -> None:
        self._visible = False
        self.visible = False
        self._valid_shape = (0, 0)
        self.update()

    def set_visible(self, visible: bool) -> None:
        self._visible = bool(visible)
        self.visible = bool(visible) and self._valid_shape[0] > 0 and self._valid_shape[1] > 0
        self.update()

    def show(
        self,
        mask,
        *,
        t0: float,
        c0: float,
        t1: Optional[float] = None,
        c1: Optional[float] = None,
        color: Optional[str] = None,
        alpha: Optional[float] = None,
        threshold: Optional[float] = None,
        visible: bool = True,
    ) -> None:
        arr = np.asarray(mask, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"mask must be 2D (time, channel), got shape={arr.shape}")
        dt, dc = int(arr.shape[0]), int(arr.shape[1])
        if dt <= 0 or dc <= 0:
            self.clear()
            return
        if dt > self._capacity_t or dc > self._capacity_c:
            raise ValueError(
                "mask patch exceeds visual capacity: "
                f"patch=({dt}, {dc}), capacity=({self._capacity_t}, {self._capacity_c})"
            )

        if color is not None:
            self._color = str(color)
        if alpha is not None:
            self._alpha = float(alpha)
        if threshold is not None:
            self._threshold = float(threshold)

        self._texture.set_data(np.ascontiguousarray(arr.T), offset=(0, 0))
        self._valid_shape = (dt, dc)
        self._bounds = (
            float(t0),
            float(c0),
            float(t1) if t1 is not None else float(t0) + float(dt),
            float(c1) if c1 is not None else float(c0) + float(dc),
        )
        self._sync_geometry()
        self._sync_uniforms()
        self._visible = bool(visible)
        self.visible = bool(visible)
        self.update()

    def _sync_geometry(self) -> None:
        t0, c0, t1, c1 = self._bounds
        if t1 < t0:
            t0, t1 = t1, t0
        if c1 < c0:
            c0, c1 = c1, c0
        verts = np.array(
            [[t0, c0], [t1, c0], [t0, c1], [t1, c1]],
            dtype=np.float32,
        )
        dt, dc = self._valid_shape
        tx = max(0.0, min(1.0, float(dt) / float(self._capacity_t)))
        ty = max(0.0, min(1.0, float(dc) / float(self._capacity_c)))
        tex = np.array(
            [[0.0, 0.0], [tx, 0.0], [0.0, ty], [tx, ty]],
            dtype=np.float32,
        )
        self._v_position.set_data(verts)
        self._v_texcoord.set_data(tex)

    def _sync_uniforms(self) -> None:
        rgba = Color(self._color).rgba
        self.shared_program["u_color"] = tuple(float(x) for x in rgba)
        self.shared_program["u_alpha"] = float(self._alpha)
        self.shared_program["u_threshold"] = float(self._threshold)

    def _compute_bounds(self, axis, view):
        t0, c0, t1, c1 = self._bounds
        if axis == 0:
            return (min(t0, t1), max(t0, t1))
        if axis == 1:
            return (min(c0, c1), max(c0, c1))
        return (0, 0)

    def _prepare_transforms(self, view):
        view.view_program.vert["transform"] = view.transforms.get_transform()

    def _prepare_draw(self, view):
        if not self._visible or self._valid_shape[0] <= 0 or self._valid_shape[1] <= 0:
            return False
        prg = view.view_program
        prg["a_position"] = self._v_position
        prg["a_texcoord"] = self._v_texcoord
        prg["u_texture"] = self._texture
        return True


MaskOverlay = create_visual_node(MaskOverlayVisual)
MaskOverlay.__name__ = "MaskOverlay"
MaskOverlay.__qualname__ = "MaskOverlay"
