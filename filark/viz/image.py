# Copyright (c) 2026 Jintao Li. 
# Zhejiang University (ZJU).
# 
# Licensed under the MIT License.


from typing import Optional, Tuple, Union
import numpy as np

from vispy.visuals.visual import Visual
from vispy.scene.visuals import create_visual_node
from vispy.gloo import Texture2D, VertexBuffer
from vispy.color import get_colormap
from vispy.visuals.shaders import Function, FunctionChain

__all__ = ["StreamingImage", "StreamingImageVisual"]

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
uniform float u_offset_x;      // normalized (wrap via fract)
uniform float u_offset_y;      // normalized (wrap via fract)
uniform float u_wrap_y;        // 0.0 or 1.0
varying vec2 v_texcoord;

void main() {
    vec2 texcoord = v_texcoord;

    // logical -> physical (ring)
    texcoord.x = fract(texcoord.x + u_offset_x);
    if (u_wrap_y > 0.5) {
        texcoord.y = fract(texcoord.y + u_offset_y);
    }

    gl_FragColor = $color_transform($get_data(texcoord));
}
"""

_TEXTURE_LOOKUP = r"""
vec4 texture_lookup(vec2 texcoord) {
    if (texcoord.x < 0.0 || texcoord.x > 1.0 ||
        texcoord.y < 0.0 || texcoord.y > 1.0) {
        discard;
    }
    return texture2D($texture, texcoord);
}
"""

_C2L_RED = r"float color_to_luminance(vec4 color) { return color.r; }"

_APPLY_CLIM_FLOAT = r"""
float apply_clim(float data) {
    // keep NaN as NaN
    if (!(data <= 0.0 || 0.0 <= data)) return data;

    data = clamp(data, min($clim.x, $clim.y), max($clim.x, $clim.y));
    data = (data - $clim.x) / ($clim.y - $clim.x);
    return data;
}
"""

_APPLY_GAMMA_FLOAT = r"""
float apply_gamma(float data) {
    if (!(data <= 0.0 || 0.0 <= data)) return data;
    return pow(data, $gamma);
}
"""


class StreamingImageVisual(Visual):
    """
    Streaming ring-buffer image visual.

    This visual displays a 2D image backed by a fixed-size GPU texture,
    and supports efficient streaming updates via push operations
    (left/right, optionally up/down) without reallocating the texture.

    Conceptually:
      - X axis is always a ring (time / samples).
      - Y axis can optionally be a ring (channels) if `wrap_y=True`.

    The visual itself does NOT decide *when* to stream.
    It only provides fast primitives; higher-level logic (scheduler/camera)
    controls when push/reset happens.

    Example
    -------
    >>> img = StreamingImageVisual(
    ...     shape=(1024, 4096),
    ...     cmap="seismic",
    ...     clim=(-5, 5),
    ...     wrap_y=False,
    ... )
    >>> img.set_data(initial_block)
    >>> img.push_right(new_block)   # append new samples on the right
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        cmap: Union[str, object] = "viridis",
        clim: Union[str, Tuple[float, float]] = "auto",
        gamma: float = 1.0,
        interpolation: str = "nearest",
        keep_cpu: bool = False,
        base: Optional[np.ndarray] = None,
        wrap_y: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        shape : (int, int)
            Texture shape as (height, width), i.e. (channels, samples).
            This is the fixed ring-buffer size on GPU.

        cmap : str or Colormap, optional
            Colormap used for rendering.

        clim : "auto" or (float, float), optional
            Color limits. Use "auto" for dynamic scaling.

        gamma : float, optional
            Gamma correction applied in the shader.

        interpolation : str, optional
            Texture interpolation method ("nearest", "linear", ...).

        keep_cpu : bool, optional
            If True, keep a CPU-side copy of the data buffer.
            Useful for debugging or read-back, but costs memory.

        base : ndarray or None, optional
            Initial image data with shape matching `shape`.
            If None, the texture is allocated but left uninitialized.

        wrap_y : bool, optional
            If True, enable ring behavior in Y direction
            (allows push_up / push_down).
            If False, Y is treated as a fixed axis.

        **kwargs
            Forwarded to `Visual`.
        """
        self._H, self._W = int(shape[0]), int(shape[1])
        self._channels = 1  # this implementation focuses on single-channel float
        self._wrap_y = bool(wrap_y)

        # ring bases (physical pixel indices)
        self._base_x_px = 0
        self._base_y_px = 0

        # state flags (ImageVisual-like)
        self._need_colortransform_update = True
        self._need_interpolation_update = False

        # display params
        self._cmap = get_colormap(cmap)
        self._gamma = float(gamma)
        if self._gamma <= 0:
            raise ValueError("gamma must be > 0")

        self._interpolation_names = ("nearest", "linear")
        if interpolation not in self._interpolation_names:
            raise ValueError(
                f"interpolation must be one of {self._interpolation_names}")
        self._interpolation = interpolation

        self._keep_cpu = bool(keep_cpu)
        self._cpu_buf = None

        # base data
        if base is None:
            base = np.zeros((self._H, self._W), dtype=np.float32)
        base = self._coerce_full(base)

        if self._keep_cpu:
            self._cpu_buf = base.copy()

        # clim (stored in python, applied in shader)
        self._clim = self._compute_clim(clim, base_for_auto=base)

        # Visual init
        Visual.__init__(self, vcode=_VERTEX_SHADER, fcode=_FRAGMENT_SHADER)
        self._draw_mode = "triangle_strip"

        # quad
        verts = np.array(
            [[0, 0], [self._W, 0], [0, self._H], [self._W, self._H]],
            dtype=np.float32,
        )
        tex = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
        self._v_position = VertexBuffer(verts)
        self._v_texcoord = VertexBuffer(tex)

        # texture
        self._texture = self._create_texture(base)

        # get_data
        self._data_lookup_fn = Function(_TEXTURE_LOOKUP)
        self.shared_program.frag["get_data"] = self._data_lookup_fn
        self._data_lookup_fn["texture"] = self._texture

        # bind
        self.shared_program["a_position"] = self._v_position
        self.shared_program["a_texcoord"] = self._v_texcoord
        self.shared_program["u_texture"] = self._texture

        # ring uniforms
        self.shared_program["u_wrap_y"] = 1.0 if self._wrap_y else 0.0
        self._apply_base()

        # build color transform chain
        self._color_chain = None
        self._f_clim = None
        self._f_gamma = None
        self._build_color_transform()
        self._need_colortransform_update = False

        self.set_gl_state("translucent", cull_face=False)
        self.freeze()

    # -------------------------
    # ImageVisual-like view init (optional, but you asked for this style)
    # -------------------------
    def view(self):
        """Get the VisualView for this visual (ImageVisual-like pattern)."""
        v = Visual.view(self)
        self._init_view(v)
        return v

    def _init_view(self, view):
        # per-view stash (keep minimal)
        view._need_method_update = True
        view._method_used = None

    # -------------------------
    # Properties (ImageVisual-like)
    # -------------------------
    @property
    def size(self):
        """Get size (width, height)."""
        return (self._W, self._H)

    @property
    def clim(self):
        """Get color limits used when rendering (cmin, cmax)."""
        return self._clim

    @clim.setter
    def clim(self, clim):
        self._clim = self._compute_clim(
            clim,
            base_for_auto=self._cpu_buf if self._cpu_buf is not None else None,
        )
        self._update_colortransform_clim()
        self.update()

    def _update_colortransform_clim(self):
        if self._need_colortransform_update:
            return
        # shortcut update (like ImageVisual)
        if self._color_chain is not None:
            self._color_chain[1]["clim"] = self._clim

    @property
    def cmap(self):
        """Get the colormap object applied to luminance data."""
        return self._cmap

    @cmap.setter
    def cmap(self, cmap):
        self._cmap = get_colormap(cmap)
        self._need_colortransform_update = True
        self.update()

    @property
    def gamma(self):
        """Get gamma used when rendering."""
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        if value <= 0:
            raise ValueError("gamma must be > 0")
        self._gamma = float(value)
        # shortcut update (like ImageVisual)
        if not self._need_colortransform_update and self._color_chain is not None:
            self._color_chain[2]["gamma"] = self._gamma
        self.update()

    @property
    def bad_color(self):
        """Color used to render NaN values."""
        return self._cmap.get_bad_color()

    @bad_color.setter
    def bad_color(self, color):
        self._cmap.set_bad_color(color)
        self._need_colortransform_update = True
        self.update()

    @property
    def interpolation(self):
        """Get interpolation algorithm name."""
        return self._interpolation

    @interpolation.setter
    def interpolation(self, i):
        if i not in self._interpolation_names:
            raise ValueError(
                f"interpolation must be one of {self._interpolation_names}")
        if self._interpolation != i:
            self._interpolation = i
            self._need_interpolation_update = True
            self.update()

    @property
    def interpolation_functions(self):
        """Get names of possible interpolation methods."""
        return self._interpolation_names

    @property
    def wrap_y(self):
        """Whether Y axis is ring-wrapped."""
        return self._wrap_y

    # -------------------------
    # Core push semantics (X ring)
    # -------------------------
    def push_left(self, chunk):
        """
        Left push: new enters LEFT, old shifts RIGHT.
        Example: [1 2 3 4 5] + 9 -> [9 1 2 3 4]
        chunk shape: (H, w)
        """
        chunk = self._coerce_x_chunk(chunk)
        w = int(chunk.shape[1])
        if w <= 0 or w > self._W:
            raise ValueError("Invalid chunk width")

        # logical shift right by w => base moves left by w
        self._base_x_px = (self._base_x_px - w) % self._W

        # write new chunk to new base_x
        self._write_block(chunk, x0=self._base_x_px, y0=self._base_y_px)

        self._apply_base()
        self.update()

    def push_right(self, chunk):
        """
        Right push: new enters RIGHT, old shifts LEFT.
        Example: [1 2 3 4 5] + 9 -> [2 3 4 5 9]
        chunk shape: (H, w)
        """
        chunk = self._coerce_x_chunk(chunk)
        w = int(chunk.shape[1])
        if w <= 0 or w > self._W:
            raise ValueError("Invalid chunk width")

        # logical shift left by w => base moves right by w
        self._base_x_px = (self._base_x_px + w) % self._W

        # freed region is old LEFT w columns => physical start = base_x - w
        x_write = (self._base_x_px - w) % self._W
        self._write_block(chunk, x0=x_write, y0=self._base_y_px)

        self._apply_base()
        self.update()

    # -------------------------
    # Optional Y ring push
    # -------------------------
    def push_up(self, chunk):
        """
        Up push: new enters TOP, old shifts DOWN.
        chunk shape: (h, W)
        """
        if not self._wrap_y:
            raise RuntimeError("wrap_y=False: Y ring is disabled")

        chunk = self._coerce_y_chunk(chunk)
        h = int(chunk.shape[0])
        if h <= 0 or h > self._H:
            raise ValueError("Invalid chunk height")

        self._base_y_px = (self._base_y_px - h) % self._H
        self._write_block(chunk, x0=self._base_x_px, y0=self._base_y_px)

        self._apply_base()
        self.update()

    def push_down(self, chunk):
        """
        Down push: new enters BOTTOM, old shifts UP.
        chunk shape: (h, W)
        """
        if not self._wrap_y:
            raise RuntimeError("wrap_y=False: Y ring is disabled")

        chunk = self._coerce_y_chunk(chunk)
        h = int(chunk.shape[0])
        if h <= 0 or h > self._H:
            raise ValueError("Invalid chunk height")

        self._base_y_px = (self._base_y_px + h) % self._H
        y_write = (self._base_y_px - h) % self._H
        self._write_block(chunk, x0=self._base_x_px, y0=y_write)

        self._apply_base()
        self.update()

    # -------------------------
    # Full reset / replace
    # -------------------------
    def set_data(self, data, *, reset_base=True, keep_view=True):
        """
        Replace the whole image data (H, W).

        reset_base=True:
            set base_x/base_y to 0,0 (most intuitive)
        keep_view=True:
            keep current logical view stable (data is assumed in logical coords),
            so we write data into physical texture at (base_x, base_y).
        """
        data = self._coerce_full(data)

        if reset_base:
            self._base_x_px = 0
            self._base_y_px = 0

        if keep_view:
            x0, y0 = self._base_x_px, self._base_y_px
        else:
            x0, y0 = 0, 0

        self._write_block(data, x0=x0, y0=y0)

        if self._keep_cpu:
            # keep a canonical logical copy (not physical)
            self._cpu_buf = data.copy()

        self._apply_base()
        self.update()

    def resize(self, shape, *, fill=0.0):
        newH, newW = map(int, shape)
        self._H, self._W = newH, newW
        base = np.full((newH, newW), float(fill), np.float32)

        if self._keep_cpu:
            self._cpu_buf = base.copy()
        else:
            self._cpu_buf = None

        self._base_x_px = 0
        self._base_y_px = 0

        self._replace_texture(self._create_texture(base))
        self._rebuild_quad()
        self._apply_base()
        self.update()


    # -------------------------
    # Internals
    # -------------------------
    def _apply_base(self):
        self.shared_program["u_offset_x"] = float(self._base_x_px) / float(self._W)
        self.shared_program["u_offset_y"] = float(self._base_y_px) / float(self._H)

    def _create_texture(self, base):
        wrap = ("repeat", "repeat") if self._wrap_y else ("repeat", "clamp_to_edge")
        tex = Texture2D(
            base.astype(np.float32, copy=False),
            internalformat="r32f",
            format="red",
            interpolation=self._interpolation,
            wrapping=wrap,
        )
        return tex
    
    def _rebuild_quad(self):
        """Rebuild quad geometry to match current (H,W)."""
        verts = np.array(
            [[0, 0], [self._W, 0], [0, self._H], [self._W, self._H]],
            dtype=np.float32,
        )
        # 这里不要新建 VertexBuffer 对象，直接 set_data，避免 program 里绑定失效
        self._v_position.set_data(verts)

        # texcoord 不依赖 H/W，保持不变即可；如果你未来想做非[0,1]映射，也可重设
        # self._v_texcoord.set_data(...)

    def _replace_texture(self, new_tex: Texture2D):
        """Swap texture object and rebind shader uniforms."""
        old = self._texture
        self._texture = new_tex

        # 绑定到 lookup function + program uniform
        self._data_lookup_fn["texture"] = self._texture
        self.shared_program["u_texture"] = self._texture

        # ring uniforms depends on H/W so base offsets need refresh
        self.shared_program["u_wrap_y"] = 1.0 if self._wrap_y else 0.0
        self._apply_base()

        # 让 old 释放引用（通常交给 GC；如果你有显式 delete 机制可放这里）
        _ = old


    def _ensure_interpolation_updated(self):
        if not self._need_interpolation_update:
            return
        # try in-place update first
        try:
            self._texture.interpolation = self._interpolation
        except Exception:
            # fallback: recreate texture and rebind
            old = self._texture
            # if we have cpu, use cpu; otherwise we cannot recover full data reliably
            if self._cpu_buf is None:
                raise RuntimeError(
                    "Recreating texture requires CPU copy, but keep_cpu=False. "
                    "Set keep_cpu=True or avoid interpolation changes that require recreation."
                )
            self._texture = self._create_texture(self._cpu_buf)
            self._data_lookup_fn["texture"] = self._texture
            self.shared_program["u_texture"] = self._texture
            # prevent old from being referenced
            _ = old
        self._need_interpolation_update = False

    def _build_color_transform(self):
        # single-channel: red -> luminance -> clim -> gamma -> cmap
        fclim = Function(_APPLY_CLIM_FLOAT)
        fgamma = Function(_APPLY_GAMMA_FLOAT)
        chain = FunctionChain(
            None,
            [Function(_C2L_RED), fclim, fgamma,
             Function(self._cmap.glsl_map)],
        )
        fclim["clim"] = self._clim
        fgamma["gamma"] = self._gamma

        self.shared_program.frag["color_transform"] = chain
        self.shared_program["texture2D_LUT"] = self._cmap.texture_lut()

        self._color_chain = chain
        self._f_clim = fclim
        self._f_gamma = fgamma

    def _compute_clim(self, clim, base_for_auto=None):
        if clim is None:
            clim = "auto"
        if isinstance(clim, str) and clim == "auto":
            if base_for_auto is None:
                # best-effort fallback
                return (0.0, 1.0)
            mn = np.nanmin(base_for_auto)
            mx = np.nanmax(base_for_auto)
            if not np.isfinite(mn) or not np.isfinite(mx) or mn == mx:
                return (0.0, 1.0)
            return (float(mn), float(mx))

        c0, c1 = clim
        c0, c1 = float(c0), float(c1)
        if c0 == c1:
            c1 = c0 + 1e-12
        return (c0, c1)

    def _coerce_full(self, data):
        data = np.asarray(data)
        if data.ndim != 2 or data.shape != (self._H, self._W):
            raise ValueError(f"data must be (H,W)==({self._H},{self._W}), got {data.shape}")
        return data.astype(np.float32, copy=False)

    def _coerce_x_chunk(self, chunk):
        chunk = np.asarray(chunk)
        if chunk.ndim != 2 or chunk.shape[0] != self._H:
            raise ValueError(f"x-chunk must be (H,w) with H={self._H}, got {chunk.shape}")
        return chunk.astype(np.float32, copy=False)

    def _coerce_y_chunk(self, chunk):
        chunk = np.asarray(chunk)
        if chunk.ndim != 2 or chunk.shape[1] != self._W:
            raise ValueError(f"y-chunk must be (h,W) with W={self._W}, got {chunk.shape}")
        return chunk.astype(np.float32, copy=False)

    def _write_block(self, block, *, x0, y0):
        """
        Write a block (h,w) into the *physical* texture at (x0,y0),
        splitting on ring boundaries. Offset order is (y, x) as you requested.

        If wrap_y=False, we still allow writing with y0 but we assume y0 will be 0
        for full-height x chunks; y wrapping isn't used in shader, and texture y wrapping is clamped.
        """
        block = np.asarray(block, dtype=np.float32, order="C")
        h, w = int(block.shape[0]), int(block.shape[1])
        if h <= 0 or w <= 0:
            return

        if self._keep_cpu and self._cpu_buf is not None:
            self._write_cpu_block(block, x0=x0, y0=y0)

        # split x
        x_parts = []
        if x0 + w <= self._W:
            x_parts.append((x0, 0, w))  # (x_write, x_src, w_len)
        else:
            w1 = self._W - x0
            x_parts.append((x0, 0, w1))
            x_parts.append((0, w1, w - w1))

        # split y (only needed if wrap_y and y0+h crosses H)
        y_parts = []
        if (not self._wrap_y) or (y0 + h <= self._H):
            y_parts.append((y0, 0, h))
        else:
            h1 = self._H - y0
            y_parts.append((y0, 0, h1))
            y_parts.append((0, h1, h - h1))

        # write up to 4 rectangles
        for (xw, xs, wl) in x_parts:
            for (yw, ys, hl) in y_parts:
                sub = block[ys:ys + hl, xs:xs + wl]
                self._texture.set_data(sub, offset=(yw, xw))  # (y,x)

    def _write_cpu_block(self, block, *, x0, y0):
        """Maintain a CPU copy in *physical layout* (same as texture) for debugging/export."""
        h, w = block.shape
        # split x
        if x0 + w <= self._W:
            x_slices = [(slice(x0, x0 + w), slice(0, w))]
        else:
            w1 = self._W - x0
            x_slices = [
                (slice(x0, self._W), slice(0, w1)),
                (slice(0, w - w1), slice(w1, w)),
            ]

        # split y
        if (not self._wrap_y) or (y0 + h <= self._H):
            y_slices = [(slice(y0, y0 + h), slice(0, h))]
        else:
            h1 = self._H - y0
            y_slices = [
                (slice(y0, self._H), slice(0, h1)),
                (slice(0, h - h1), slice(h1, h)),
            ]

        for (ydst, ysrc) in y_slices:
            for (xdst, xsrc) in x_slices:
                self._cpu_buf[ydst, xdst] = block[ysrc, xsrc]

    # -------------------------
    # Visual hooks
    # -------------------------
    def _compute_bounds(self, axis, view):
        if axis == 0:
            return (0, self._W)
        if axis == 1:
            return (0, self._H)
        return (0, 0)

    def _prepare_transforms(self, view):
        view.view_program.vert["transform"] = view.transforms.get_transform()

    def _prepare_draw(self, view):
        # late updates like ImageVisual
        if self._need_colortransform_update:
            self._build_color_transform()
            self._need_colortransform_update = False

        if self._need_interpolation_update:
            self._ensure_interpolation_updated()

        prg = view.view_program
        prg["a_position"] = self._v_position
        prg["a_texcoord"] = self._v_texcoord
        prg["u_texture"] = self._texture
        return True


StreamingImage = create_visual_node(StreamingImageVisual)
