# Copyright (c) 2026 Jintao Li. 
# Zhejiang University (ZJU).
# 
# Licensed under the MIT License.

from typing import Any, Callable, Optional, Sequence, Tuple

import math
import numpy as np
from vispy import scene
from vispy.visuals.transforms import STTransform
from vispy.geometry import Rect

from filark.io.streaming import StreamingSource
from .image import StreamingImage
from .camera import StreamingCamera
from .scheduler import BufferScheduler2D
from .axis import AxisWidget
from ._canvas_mixin import RealtimeAutoScrollMixin, KeyboardPanMixin


class StreamingCanvas(scene.SceneCanvas, RealtimeAutoScrollMixin, KeyboardPanMixin):
    """
    Top-level canvas orchestrator for DAS streaming visualization.

    StreamingCanvas is the ONLY owner that creates and wires together:
      - View
      - StreamingImageVisual
      - StreamingCamera
      - BufferScheduler2D

    Responsibilities:
      - Construct and connect camera ↔ scheduler
      - Expose deterministic public APIs for navigation and scaling
      - Own keyboard / realtime auto-scroll behaviors
      - Handle delayed initialization after first draw / resize

    Parameters (summary)
    --------------------
    source : array-like or StreamingSource
        Data source for streaming.
    buffer_h, buffer_w : int or None
        Streaming buffer size.
    wrap_y : bool
        Enable channel (Y) streaming.
    x_align, x_margin : int
        X streaming alignment and safety margin.
    """

    def __init__(
        self,
        source=None,
        size=(1200, 600),
        buffer_h=None,
        buffer_w=None,
        world_origin=(0.0, 0.0),
        world_scale=(1.0, 1.0),
        axis_units=("", ""),
        axis_labels=None,
        title="",
        wrap_y=False,
        cmap="seismic",
        clim=(-5.0, 5.0),
        interpolation="nearest",
        keep_cpu=True,
        x_align=256,
        x_margin=512,
        enable_coalesce=False,
        coalesce_hz=60,
        bgcolor="black",
        **kwargs,
    ):
        super().__init__(keys="interactive", size=size, bgcolor=bgcolor, title="FiLark Viewer", **kwargs)
        self.unfreeze()

        # ---- store config (no data) ----
        self._wrap_y = bool(wrap_y)
        self._world_origin = world_origin
        self._world_scale = world_scale
        self._axis_units = axis_units
        self._title = title
        self._cmap = cmap
        self._clim = clim
        self._interpolation = interpolation
        self._keep_cpu = bool(keep_cpu)

        self._x_align = int(x_align)
        self._x_margin = int(x_margin)
        self._enable_coalesce = bool(enable_coalesce)
        self._coalesce_hz = int(coalesce_hz)

        if axis_labels is None:
            axis_labels = ["Time", "Channels"]
            if axis_units[0]:
                axis_labels[0] = f"Time ({axis_units[0]})"
            if axis_units[1]:
                axis_labels[1] = f"Distance ({axis_units[1]})"
        self._axis_labels = axis_labels

        # ---- layout ----
        self.grid = self.central_widget.add_grid(margin=10)
        self.grid.spacing = 0
        border_color = "white" if bgcolor == "black" else "black"
        self.view = self.grid.add_view(row=1, col=1, border_color=border_color)

        # ---- image: create as "empty" placeholder ----
        # init with shape (1, 1)
        self.image = StreamingImage(
            shape=(1, 1),
            cmap=self._cmap,
            clim=self._clim,
            gamma=1.0,
            interpolation=self._interpolation,
            keep_cpu=self._keep_cpu,
            base=None,
            wrap_y=self._wrap_y,
            parent=self.view.scene,
        )
        self.image.visible = False
        self.image.transform = STTransform(translate=(0.0, 0.0))

        # ---- World layer: for physic coords ------
        self.world_layer = scene.Node(parent=self.view.scene)
        self.world_layer.transform = STTransform(scale=(1.0, 1.0, 1.0))

        # ---- Camera ----
        self.camera = StreamingCamera(
            ymin=0, ymax=1,
            xmin=0, xmax=1,
            y_home_height=1.0,
            wrap_y=self._wrap_y,
            aspect=1,
            interactive=True,
        )
        self.view.camera = self.camera

        # ---- scheduler initial ----
        self.scheduler = None
        self.source = None
        self.nc = self.nt = 0
        self.buffer_h = self.buffer_w = 0

        # ---- axis ----
        self._init_axis()

        if hasattr(self, "_init_keyboard_pan"):
            self._init_keyboard_pan()
        if hasattr(self, "_init_realtime_autoscroll"):
            self._init_realtime_autoscroll()

        # delayed init: waiting source
        self._did_init = False
        self.events.resize.connect(self._try_init_once)
        self.events.draw.connect(self._try_init_once)

        self.freeze()

        if source is not None:
            self.set_source(source, buffer_h=buffer_h, buffer_w=buffer_w)


    def set_source(
        self,
        source,
        *,
        buffer_h=None,
        buffer_w=None,
        reset_view=True,
    ):
        self.unfreeze()

        # 1) wrap source
        if not isinstance(source, StreamingSource):
            dims = "auto"
            if hasattr(source, "dims"):
                dims = source.dims
            source = StreamingSource(source, dims=dims)
        self.source = source
        self.nc, self.nt = self.source.shape

        # 2) decide buffer sizes
        if buffer_h is None:
            buffer_h = min(self.nc, 2048)
        self.buffer_h = int(buffer_h)

        if buffer_w is None:
            size = self.size
            view_w = int(size[0] / max(1, size[1]) * self.buffer_h + 2 * self._x_margin)
            align = int(self._x_align if self._x_align >= 8 else 64)
            buffer_w = int(math.ceil(view_w / align) * align)
        self.buffer_w = int(buffer_w)

        # 3) tell source buffer size
        self.source.set_buffer_size(self.buffer_h, self.buffer_w)

        # 4) resize image texture
        self.image.resize((self.buffer_h, self.buffer_w), fill=0.0)
        self.image.visible = True

        # 5) update camera bounds & home height
        self.camera._ymin = 0
        self.camera._ymax = self.nc
        self.camera._xmin = 0
        self.camera._xmax = self.nt
        self.camera._y_home_height = float(min(self.buffer_h, self.nc))

        # 6) rebuild scheduler
        if self.scheduler is not None:
            self.camera.remove_view_listener(self.scheduler.on_camera_changed)

        self.scheduler = BufferScheduler2D(
            source=self.source,
            image=self.image,
            camera=self.camera,
            buffer_h=self.buffer_h,
            buffer_w=self.buffer_w,
            x_align=self._x_align,
            x_margin=self._x_margin,
            y_align=1,
            wrap_y=self._wrap_y,
            enable_coalesce=self._enable_coalesce,
            coalesce_hz=self._coalesce_hz,
            request_draw=self.update,
        )
        self.camera.add_view_listener(self.scheduler.on_camera_changed)

        # 7) reset pan window / wrap bottom
        self.camera.set_y_pan_window(0, min(self.buffer_h, self.nc))
        if self._wrap_y:
            self.camera.set_y_bottom(0)

        # 8) axis scale
        self.xaxis.update_scale(float(self.source.scale_x))
        self.yaxis.update_scale(float(self.source.scale_y))

        # 9) allow delayed init to run again
        self._did_init = False

        # 10) optionally init immediately
        if reset_view:
            self._try_init_once()

        self.freeze()
        self.update()



    # -------------------------
    # Axis
    # -------------------------
    def _init_axis(self):
        self.title_label = scene.Label(self._title, color="white")
        self.title_label.height_max = 20
        self.grid.add_widget(self.title_label, row=0, col=0, col_span=2)

        r, g, b = self.bgcolor.rgba[:3]
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = "black" if luminance > 0.5 else "white"
        tick_color = (0.3, 0.3, 0.3) if luminance > 0.5 else (0.7, 0.7, 0.7)
        axis_color = (0, 0, 0) if luminance > 0.5 else (1, 1, 1)

        dz = self._world_scale[1]
        self.yaxis = AxisWidget(
            orientation="left",
            transform=lambda y: f"{float(y) * dz:g}",
            unit=self._axis_units[1],
            axis_label=self._axis_labels[1],
            axis_font_size=12,
            axis_label_margin=50,
            tick_label_margin=5,
            text_color=text_color,
            axis_color=axis_color,
            tick_color=tick_color,
        )
        self.yaxis.width_max = 80
        self.grid.add_widget(self.yaxis, row=1, col=0)

        fs = self._world_scale[0]
        self.xaxis = AxisWidget(
            orientation="bottom",
            transform=lambda t: f"{float(t) / fs:g}",
            unit=self._axis_units[0],
            axis_label=self._axis_labels[0],
            axis_font_size=12,
            axis_label_margin=50,
            tick_label_margin=20,
            text_color=text_color,
            axis_color=axis_color,
            tick_color=tick_color,
        )
        self.xaxis.height_max = 60
        self.grid.add_widget(self.xaxis, row=2, col=1)

        self.xaxis.link_view(self.view)
        self.yaxis.link_view(self.view)

    # -------------------------
    # delayed init
    # -------------------------
    def _try_init_once(self, event=None):
        if self._did_init:
            return
        if self.source is None or self.scheduler is None:
            return

        ok = self.camera.initialize_view(x0=0)
        if not ok:
            self.update()
            return

        self._did_init = True

        # trigger one scheduler sync and draw
        self.camera.view_changed()
        self.update()

    # -------------------------
    # deterministic Y APIs (GUI/keyboard)
    # -------------------------
    def set_channel_start(self, c0, *, hard_reset=False, sync_camera=True):
        """
        Move channel window deterministically.
        """
        self.scheduler.set_channel_start(c0, hard_reset=hard_reset)
        if sync_camera:
            # Update mouse-pan window to current buffer window
            self.camera.set_y_pan_window(self.scheduler.c0, self.scheduler.c0 + self.buffer_h)

            # keep camera bottom consistent with buffer origin (optional but recommended)
            if self._wrap_y:
                self.camera.set_y_bottom(self.scheduler.c0)

        self.update()

    def scroll_channels(self, delta, *, hard_reset=False, sync_camera=True):
        self.set_channel_start(
            self.scheduler.c0 + int(delta),
            hard_reset=hard_reset,
            sync_camera=sync_camera,
        )

    def page_channels(
        self,
        direction,
        overlap=0.2,
        *,
        hard_reset=False,
        sync_camera=True,
    ):
        step = int(round(self.buffer_h * (1.0 - float(overlap))))
        step = max(1, step)
        self.scroll_channels(
            step if direction > 0 else -step,
            hard_reset=hard_reset,
            sync_camera=sync_camera,
        )

    # -------------------------
    # scaling APIs (GUI/keyboard)
    # -------------------------
    def set_scale_x(self, scale_x: float):
        """
        Change x compression factor in source.
        We anchor at current VIEW LEFT in "scaled view coords".
        """
        if scale_x is None:
            return

        old_sx = float(self.source.scale_x)
        new_sx = float(scale_x)
        if new_sx < 1.0:
            new_sx = 1.0
        if new_sx == old_sx:
            return

        rect = self.camera.rect

        # anchor at view_left
        view_left_old = float(rect.left)
        view_left_new = view_left_old * old_sx / new_sx

        self.source.set_scale(scale_x=new_sx, scale_y=None)

        # update camera x bounds consistently (scale change changes world mapping)
        self.camera._xmin = self.camera._xmin * old_sx / new_sx
        if self.camera._xmax is not None:
            self.camera._xmax = self.camera._xmax * old_sx / new_sx

        # reset scheduler origin to anchored left
        self.scheduler.t0 = int(view_left_new)
        # full reset after scale
        self.scheduler._reset_full(self.scheduler.t0, self.scheduler.c0)

        # keep camera rect left consistent
        self.camera.rect = Rect(
            float(self.scheduler.t0),
            rect.bottom,
            rect.width,
            rect.height,
        )

        # axis
        self.xaxis.update_scale(new_sx)

        self.camera.view_changed()
        self._update_world_layer_transform()
        self.update()

    def set_scale_y(self, scale_y: float):
        """
        Change y compression factor in source.
        Supports fit_all switch when raw_h >= total channels.
        """
        if scale_y is None:
            return

        old_sy = float(self.source.scale_y)
        new_sy = float(scale_y)
        if new_sy < 1.0:
            new_sy = 1.0
        if new_sy == old_sy:
            return

        rect = self.camera.rect
        H = self.source.shape[0]
        bh = int(self.source.buffer_h or rect.height)

        raw_h_local = int(np.ceil(bh * new_sy))

        # fit_all mode
        if raw_h_local >= H:
            new_sy = H / max(1, bh)
            self.source.set_scale(
                scale_x=None,
                scale_y=new_sy,
                y_mode="fit_all",
            )
            self.scheduler.c0 = 0
            self.camera.set_y_pan_window(0, H)

            # update camera y bounds
            self.camera._ymin = self.camera._ymin * old_sy / new_sy
            self.camera._ymax = self.camera._ymax * old_sy / new_sy

            self.yaxis.update_scale(new_sy)

            self.scheduler._reset_full(self.scheduler.t0, self.scheduler.c0)

            # Update mouse-pan window to current buffer window
            self.camera.set_y_pan_window(self.scheduler.c0, self.scheduler.c0 + self.buffer_h)

            if self._wrap_y:
                self.camera.set_y_bottom(0)
            self.camera.view_changed()
            self.update()
            return

        # local mode
        self.source.set_scale(scale_x=None, scale_y=new_sy, y_mode="local")

        # bottom-anchored scaling
        view_bot_old = float(rect.bottom)
        view_bot_new = view_bot_old * (old_sy / new_sy)

        c_min = 0.0
        c_max = (H - raw_h_local) / new_sy
        c_new = float(np.clip(view_bot_new, c_min, c_max))

        self.scheduler.c0 = int(c_new)
        self.camera._ymin = self.camera._ymin * old_sy / new_sy
        self.camera._ymax = self.camera._ymax * old_sy / new_sy

        self.yaxis.update_scale(new_sy)

        self.scheduler._reset_full(self.scheduler.t0, self.scheduler.c0)
        if self._wrap_y:
            self.camera.set_y_bottom(self.scheduler.c0)
        self.camera.view_changed()
        self._update_world_layer_transform()
        self.update()


    def _update_world_layer_transform(self):
        sx = self.source.scale_x
        sy = self.source.scale_y
        
        sx = sx if sx > 1e-6 else 1.0
        sy = sy if sy > 1e-6 else 1.0

        # physic coords to pixel coords
        self.world_layer.transform.scale = (1.0/sx, 1.0/sy, 1.0)
        # self.world_layer.transform.translate = (tx, ty)


    def clear_source(self):
        self.unfreeze()

        if self.scheduler is not None:
            self.camera.remove_view_listener(self.scheduler.on_camera_changed)
            self.scheduler.close() if hasattr(self.scheduler, "close") else None
        self.scheduler = None
        self.source = None
        self.nc = self.nt = 0
        self.buffer_h = self.buffer_w = 0
        self._did_init = False

        self.image.resize((1, 1))
        self.image.visible = False

        if getattr(self, "world_layer", None) is not None:
            self._detach_all_children(self.world_layer)
            try:
                self.world_layer.transform = STTransform(scale=(1.0, 1.0, 1.0), translate=(0.0, 0.0, 0.0))
            except Exception:
                pass

        self.freeze()
        self.update()


    def _detach_all_children(self, node: scene.Node):
        """Detach all children nodes from a scene.Node safely."""
        # copy list to avoid modifying while iterating
        for ch in list(getattr(node, "children", [])):
            try:
                if hasattr(ch, "close") and callable(ch.close):
                    ch.close()
            except Exception:
                pass
            try:
                ch.parent = None
            except Exception:
                try:
                    ch._parent = None
                except Exception:
                    pass
