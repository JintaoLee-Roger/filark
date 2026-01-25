# Copyright (c) 2026 Jintao Li. 
# Zhejiang University (ZJU).
# 
# Licensed under the MIT License.


import numpy as np
from math import floor
from vispy import app
from vispy.visuals.transforms import STTransform

from filark.io.streaming import StreamingSource
from .image import StreamingImageVisual
from .camera import StreamingCamera


class BufferScheduler2D:
    """
    Streaming buffer scheduler for a 2D ring image.

    Maintains the world-space window origin (t0, c0) and keeps a
    StreamingImageVisual synchronized with camera X movement
    and explicit Y control.

    Design rules:
      - Camera drives X streaming only.
      - Y streaming is deterministic and API-driven (keyboard / GUI).
      - Mouse input never triggers Y push.
      - Image transform is always kept in sync.

    Parameters (summary)
    --------------------
    source : StreamingSource
        Data provider.
    image : StreamingImageVisual
        Ring-buffer image to update.
    camera : StreamingCamera
        Camera providing world X window.
    buffer_h, buffer_w : int
        Fixed buffer size.
    wrap_y : bool
        Enable Y ring behavior.
    """

    def __init__(
        self,
        *,
        source: StreamingSource,
        image: StreamingImageVisual,
        camera: StreamingCamera,
        buffer_h: int,
        buffer_w: int,
        x_align: int = 256,
        x_margin: int = 512,
        y_align: int = 1,
        wrap_y: bool = False,
        enable_coalesce: bool = False,
        coalesce_hz: int = 60,
        reset_ratio: float = 0.7,
        request_draw=None,
    ):
        """
        Parameters
        ----------
        source : StreamingSource
            Streaming data source.

        image : StreamingImageVisual
            Target ring-buffer image visual.

        camera : StreamingCamera
            Camera providing world X window.
            Only X movement is observed.

        buffer_h : int
            Buffer height in samples (Y dimension).

        buffer_w : int
            Buffer width in samples (X dimension).

        x_align : int, default 256
            Alignment unit for X buffer start.

        x_margin : int, default 512
            Safe margin before triggering X buffer shift.

        y_align : int, default 1
            Alignment unit for Y buffer start.
            Use 1 for per-row push.

        wrap_y : bool, default False
            Whether Y push_up / push_down is enabled.

        enable_coalesce : bool, default False
            If True, buffer updates are throttled via timer.

        coalesce_hz : int, default 60
            Flush rate when coalescing is enabled.

        reset_ratio : float, default 0.7
            Threshold ratio to force full buffer reset
            instead of incremental push.

        request_draw : callable or None
            Callback to request a redraw (e.g. canvas.update).
        """

        self.source = source
        self.image = image
        self.camera = camera

        self._wrap_y = bool(wrap_y)

        self.nc, self.nt = self.source.shape  # (channels, time)

        self.buffer_h = int(buffer_h)
        self.buffer_w = int(buffer_w)

        self.x_align = int(max(1, x_align))
        self.y_align = int(max(1, y_align))

        self.x_margin = int(max(0, x_margin))
        self.x_margin = min(self.x_margin, max(0, self.buffer_w // 2 - 1))

        self._reset_ratio = float(reset_ratio)

        self.request_draw = request_draw  # may be None

        # window origin (world coords)
        self.t0 = 0
        self.c0 = 0

        if not isinstance(getattr(self.image, "transform", None), STTransform):
            self.image.transform = STTransform(translate=(0, 0))

        # coalesce state for camera-driven X
        self._enable_coalesce = bool(enable_coalesce)
        self._dirty = False
        self._pending_t0 = None
        self._timer = None
        self._coalesce_hz = int(max(1, coalesce_hz))

        self._reset_full(self.t0, self.c0)

        if self._enable_coalesce:
            self.start_coalesce(self._coalesce_hz)

    # -----------------------
    # utilities
    # -----------------------
    def _draw(self):
        if callable(self.request_draw):
            self.request_draw()

    # -----------------------
    # coalesce control
    # -----------------------
    def start_coalesce(self, hz=60):
        hz = int(max(1, hz))
        self._coalesce_hz = hz
        self._enable_coalesce = True

        if self._timer is None:
            interval = 1.0 / float(hz)
            self._timer = app.Timer(
                interval=interval,
                connect=self._on_timer,
                start=True,
            )
        else:
            self._timer.interval = 1.0 / float(hz)
            self._timer.start()

    def stop_coalesce(self):
        self._enable_coalesce = False
        self._dirty = False
        self._pending_t0 = None
        if self._timer is not None:
            self._timer.stop()

    def _on_timer(self, event):
        self.flush_pending()

    def flush_pending(self):
        if (not self._enable_coalesce) or (not self._dirty):
            return
        self._dirty = False
        tgt_t0 = self.t0 if self._pending_t0 is None else int(self._pending_t0)
        self._pending_t0 = None
        self._seek_x(tgt_t0)

    # -----------------------
    # clamp + align
    # -----------------------
    def _clamp_t0(self, t0):
        if self.nt <= self.buffer_w:
            return 0
        return int(max(0, min(int(t0), self.nt - self.buffer_w)))

    def _clamp_c0(self, c0):
        if (not self._wrap_y) or (self.nc <= self.buffer_h):
            return 0
        return int(max(0, min(int(c0), self.nc - self.buffer_h)))

    @staticmethod
    def _align_start(raw, align, start_min, start_max):
        raw = float(raw)
        start_min = float(start_min)
        start_max = float(start_max)

        if start_max <= start_min:
            return int(start_min)

        raw = max(start_min, min(raw, start_max))
        a = int(max(1, align))

        if a == 1:
            return int(round(raw))

        # tail snap
        if raw >= start_max - 0.5 * a:
            return int(start_max)

        return int(floor(raw / a) * a)

    # -----------------------
    # read helpers
    # -----------------------
    def _read_window(self, c0, t0, h=None, w=None):
        h = self.buffer_h if h is None else int(h)
        w = self.buffer_w if w is None else int(w)
        return self.source.read(int(c0), int(t0), h, w)

    def _apply_translate(self):
        self.image.transform.translate = (float(self.t0), float(self.c0))

    # -----------------------
    # full reset
    # -----------------------
    def _reset_full(self, t0=None, c0=None):
        t0 = self.t0 if t0 is None else int(t0)
        c0 = self.c0 if c0 is None else int(c0)

        t0 = self._clamp_t0(t0)
        c0 = self._clamp_c0(c0)

        block = self._read_window(c0, t0)
        if block is None:
            return

        self.t0 = t0
        self.c0 = c0
        self.image.set_data(block, reset_base=True, keep_view=True)
        self._apply_translate()
        self._draw()

    # -----------------------
    # camera callback (X only)
    # -----------------------
    def on_camera_changed(self, cam: StreamingCamera):
        view_left = float(cam.rect.left)
        view_right = float(cam.rect.right)

        if self.nt <= self.buffer_w:
            return

        buf_left = self.t0
        buf_right = self.t0 + self.buffer_w
        safe_left = buf_left + self.x_margin
        safe_right = buf_right - self.x_margin

        if safe_left <= view_left and view_right <= safe_right:
            return

        view_center = 0.5 * (view_left + view_right)
        raw = view_center - self.buffer_w / 2.0
        tgt = self._align_start(raw, self.x_align, 0, self.nt - self.buffer_w)

        if self._enable_coalesce:
            self._pending_t0 = int(tgt)
            self._dirty = True
            return

        self._seek_x(int(tgt))

    # -----------------------
    # deterministic Y API (GUI/keyboard)
    # -----------------------
    def set_channel_start(self, c0, *, hard_reset=False):
        c0 = self._clamp_c0(int(c0))
        if hard_reset:
            self._reset_full(self.t0, c0)
        else:
            self._seek_y(c0)

    def step_channels(self, delta, *, hard_reset=False):
        self.set_channel_start(self.c0 + int(delta), hard_reset=hard_reset)

    def page_channels(self, direction, overlap=0.2, *, hard_reset=False):
        step = int(round(self.buffer_h * (1.0 - float(overlap))))
        step = max(1, step)
        self.step_channels(
            step if direction > 0 else -step,
            hard_reset=hard_reset,
        )

    # -----------------------
    # seek X (camera-driven)
    # -----------------------
    def _seek_x(self, tgt_t0):
        tgt_t0 = self._clamp_t0(tgt_t0)
        dx = int(tgt_t0 - self.t0)
        if dx == 0:
            return

        # big jump -> reset
        if (abs(dx) >= int(self._reset_ratio * self.buffer_w)) or (abs(dx) >= self.buffer_w):
            self._reset_full(tgt_t0, self.c0)
            return

        if dx > 0:
            self._push_right(dx)
        else:
            self._push_left(-dx)

        self._draw()

    # -----------------------
    # seek Y (GUI-driven)
    # -----------------------
    def _seek_y(self, tgt_c0):
        if not self._wrap_y:
            return

        tgt_c0 = self._clamp_c0(tgt_c0)

        # optional snap to align grid
        if self.y_align > 1:
            tgt_c0 = self._align_start(
                tgt_c0,
                self.y_align,
                0,
                self.nc - self.buffer_h,
            )

        dy = int(tgt_c0 - self.c0)
        if dy == 0:
            return

        # big jump -> reset
        if (abs(dy) >= int(self._reset_ratio * self.buffer_h)) or (abs(dy) >= self.buffer_h):
            self._reset_full(self.t0, tgt_c0)
            return

        if dy > 0:
            self._push_down(dy)
        else:
            self._push_up(-dy)

        self._draw()

    # -----------------------
    # push X
    # -----------------------
    def _push_right(self, dx):
        dx = int(dx)
        if dx <= 0:
            return

        max_dx = self.nt - (self.t0 + self.buffer_w)
        dx = min(dx, max_dx)
        if dx <= 0:
            return

        block = self.source.read(
            self.c0,
            self.t0 + self.buffer_w,
            self.buffer_h,
            dx,
        )
        if block is None:
            return

        self.image.push_right(block)
        self.t0 += dx
        self._apply_translate()

    def _push_left(self, dx):
        dx = int(dx)
        if dx <= 0:
            return

        dx = min(dx, self.t0)
        if dx <= 0:
            return

        new_t0 = self.t0 - dx
        block = self.source.read(self.c0, new_t0, self.buffer_h, dx)
        if block is None:
            return

        self.image.push_left(block)
        self.t0 = new_t0
        self._apply_translate()

    # -----------------------
    # push Y
    # -----------------------
    def _push_down(self, dy):
        dy = int(dy)
        if dy <= 0:
            return

        max_dy = self.nc - (self.c0 + self.buffer_h)
        dy = min(dy, max_dy)
        if dy <= 0:
            return

        block = self.source.read(
            self.c0 + self.buffer_h,
            self.t0,
            dy,
            self.buffer_w,
        )
        if block is None:
            return

        self.image.push_down(block)
        self.c0 += dy
        self._apply_translate()

    def _push_up(self, dy):
        dy = int(dy)
        if dy <= 0:
            return

        dy = min(dy, self.c0)
        if dy <= 0:
            return

        new_c0 = self.c0 - dy
        block = self.source.read(new_c0, self.t0, dy, self.buffer_w)
        if block is None:
            return

        self.image.push_up(block)
        self.c0 = new_c0
        self._apply_translate()
