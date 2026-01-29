# Copyright (c) 2026 Jintao Li. 
# Zhejiang University (ZJU).
# 
# Licensed under the MIT License.


import numpy as np
from vispy.scene.cameras import PanZoomCamera
from vispy.geometry import Rect


class StreamingCamera(PanZoomCamera):
    """
    Streaming-friendly PanZoom camera with strict invariants.

    Designed for large streaming images where the camera defines
    the authoritative world-visible window.

    Parameters (summary)
    --------------------
    ymin, ymax : float
        World Y bounds.
    xmin, xmax : float or None
        World X bounds (None = unbounded streaming).
    y_home_height : float or None
        Maximum allowed visible height in Y.
    wrap_y : bool
        Whether Y panning is allowed programmatically.
    """

    def __init__(
        self,
        ymin: float,
        ymax: float,
        xmin: float = 0.0,
        xmax: float | None = None,
        y_home_height: float | None = None,
        wheel_zoom_speed: float = 4.0,
        wrap_y: bool = False,
        min_height: float | None = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        ymin : float
            Minimum world Y coordinate.
        ymax : float
            Maximum world Y coordinate.

        xmin : float, default 0.0
            Minimum world X coordinate.
        xmax : float or None, default None
            Maximum world X coordinate.
            If None, X is treated as an unbounded streaming axis.

        y_home_height : float or None, default None
            Maximum allowed visible height in Y.
            If None, full [ymin, ymax] is allowed.

        wheel_zoom_speed : float, default 4.0
            Exponent factor controlling mouse wheel zoom sensitivity.

        wrap_y : bool, default False
            If True, allow Y panning via explicit API calls.
            Mouse input never changes Y regardless of this flag.

        min_height : float or None, default None
            Optional minimum visible height in Y (prevents over-zoom-in).

        **kwargs
            Forwarded to `PanZoomCamera`.
        """
        self._listeners = []

        super().__init__(**kwargs)

        self._ymin = float(ymin)
        self._ymax = float(ymax)
        self._xmin = float(xmin)
        self._xmax = None if xmax is None else float(xmax)
        self._y_pan_min = float(ymin)
        self._y_pan_max = float(ymax)

        self._wrap_y = bool(wrap_y)

        self._y_home_height = None if y_home_height is None else float(y_home_height)
        self._wheel_zoom_speed = float(wheel_zoom_speed)
        self._min_height = None if min_height is None else float(min_height)

        self._eps = 1e-6

    # ----------------------------
    # listener
    # ----------------------------
    def add_view_listener(self, fn):
        self._listeners.append(fn)

    def remove_view_listener(self, fn) -> bool:
        """Remove a previously added callback. Returns True if removed."""
        try:
            self._listeners.remove(fn)
            return True
        except ValueError:
            return False

    def clear_view_listeners(self):
        """Remove all listeners."""
        self._listeners.clear()

    def view_changed(self):
        for fn in self._listeners:
            fn(self)
        super().view_changed()

    # ----------------------------
    # helpers
    # ----------------------------
    def _H_full(self):
        return float(self._ymax - self._ymin)

    def _H_home(self):
        Hf = self._H_full()
        if self._y_home_height is None:
            return Hf
        return float(min(max(self._y_home_height, 1.0), Hf))

    def _view_aspect(self):
        vbr = getattr(self._viewbox, "rect", None)
        if vbr is None or vbr.width <= 0 or vbr.height <= 0:
            return None
        return float(vbr.width) / float(vbr.height)

    def _width_from_height(self, height):
        aspect = self._view_aspect()
        if aspect is not None:
            return float(height) * float(aspect)

        r = getattr(self, "rect", None)
        if r is not None and r.height > self._eps:
            return float(height) * float(r.width) / float(r.height)
        return float(height)
    
    def set_y_pan_window(self, y0: float, y1: float):
        """
        Set the allowed Y range for mouse panning (in world coords).
        Typically [scheduler.c0, scheduler.c0 + buffer_h] in local mode,
        or [0, nc] in fit_all mode.
        """
        y0 = float(y0)
        y1 = float(y1)
        if y1 < y0:
            y0, y1 = y1, y0

        # clamp to global bounds just in case
        y0 = max(self._ymin, min(y0, self._ymax))
        y1 = max(self._ymin, min(y1, self._ymax))
        if y1 <= y0 + self._eps:
            y1 = y0 + 1.0

        self._y_pan_min = y0
        self._y_pan_max = y1

        # re-apply rect to enforce new invariants
        r = self.rect
        self._apply_rect(left=r.left, bottom=r.bottom, height=r.height)


    def _apply_rect(self, left, bottom, height):
        """
        Single authority that enforces invariants.
        width always derived from height (aspect lock).
        """
        H_home = self._H_home()
        h = float(height)

        if self._min_height is not None:
            h = max(h, float(self._min_height))
        h = min(h, float(H_home))
        h = max(h, 1.0)

        w = max(self._width_from_height(h), 1.0)

        # Y rule (mouse-pan within pan window, regardless of wrap_y)
        y0 = float(self._y_pan_min)
        y1 = float(self._y_pan_max)
        span = float(y1 - y0)

        if h >= span - self._eps:
            b = y0
        else:
            b = float(bottom)
            b = max(y0, min(b, y1 - h))

        # X rule
        l = float(left)
        if self._xmax is None:
            l = max(self._xmin, l)
        else:
            total = float(self._xmax - self._xmin)
            if w > total + self._eps:
                l = self._xmin
            else:
                l = max(self._xmin, min(l, self._xmax - w))

        self.rect = Rect(l, b, w, h)

    # ----------------------------
    # init
    # ----------------------------
    def initialize_view(self, x0=0.0):
        vbr = getattr(self._viewbox, "rect", None)
        if vbr is None or vbr.width <= 10 or vbr.height <= 10:
            return False
        H0 = self._H_home()
        self._apply_rect(left=float(x0), bottom=self._ymin, height=H0)
        return True

    # ----------------------------
    # programmatic Y control (GUI/keyboard)
    # ----------------------------
    def set_y_bottom(self, bottom):
        """Programmatic Y move (allowed), mouse never calls this."""
        r = self.rect
        self._apply_rect(left=r.left, bottom=float(bottom), height=r.height)


    def _modifier_down(self, event, key: str) -> bool:
        mods = getattr(event, "modifiers", None)
        if not mods:
            return False
        # mods could be tuple/list/set of strings
        return key in mods

    def viewbox_mouse_event(self, event):
        if event.handled or not self.interactive:
            return

        if event.type == "gesture_zoom":
            event.handled = True
            return

        if event.type == "mouse_wheel":
            dx, dy = event.delta

            # horizontal wheel -> pan X
            if abs(dx) > abs(dy):
                scale = self.rect.width / 100.0
                self._pan_x(-float(dx) * scale)
                event.handled = True
                return
            
            # hold modifier -> wheel pans Y
            if self._modifier_down(event, "Shift"):
                # only meaningful when zoomed-in relative to current pan window
                r = self.rect
                span = float(self._y_pan_max - self._y_pan_min)
                allow_y = (r.height < span - self._eps)
                if allow_y:
                    # map wheel dy to world-units; tune with _y_wheel_pan_speed
                    # dy > 0 usually means scroll up; choose sign that feels natural
                    scale_y = (r.height / 100.0) * 6
                    self._pan_xy(0.0, float(dy) * scale_y)
                event.handled = True
                return

            # vertical wheel -> zoom, keep bottom fixed
            zoom_out = (dy < 0)
            factor = (1 + self.zoom_factor)**(-float(dy) * self._wheel_zoom_speed)
            self._zoom_about_center_xonly(float(factor), zoom_out=zoom_out)
            event.handled = True
            return

        # LMB drag -> pan X only
        if event.type in ("mouse_move", "mouse_press", "mouse_release"):
            if event.type == "mouse_move" and event.press_event is not None and 1 in event.buttons:
                p1 = np.array(event.last_event.pos, dtype=np.float32)[:2]
                p2 = np.array(event.pos, dtype=np.float32)[:2]
                p1s = self._transform.imap(p1)
                p2s = self._transform.imap(p2)
                d = (p1s - p2s)

                # Always allow X pan
                dx_world = float(d[0])

                # Allow Y pan only when zoomed-in relative to current pan window
                r = self.rect
                span = float(self._y_pan_max - self._y_pan_min)
                allow_y = (r.height < span - self._eps)
                dy_world = float(d[1]) if allow_y else 0.0

                self._pan_xy(dx_world, dy_world)
                event.handled = True
                return

            event.handled = True
            return

        super().viewbox_mouse_event(event)

    def _pan_x(self, dx):
        r = self.rect
        self._apply_rect(
            left=r.left + float(dx),
            bottom=r.bottom,
            height=r.height,
        )

    def _pan_xy(self, dx, dy):
        r = self.rect
        self._apply_rect(
            left=r.left + float(dx),
            bottom=r.bottom + float(dy),
            height=r.height,
        )

    def _zoom_about_center_xonly(self, factor, *, zoom_out):
        r = self.rect
        cx = float(r.center[0])
        new_h = float(r.height) * float(factor)

        H_home = self._H_home()
        if zoom_out and (new_h >= H_home - self._eps):
            new_h = H_home

        new_w = self._width_from_height(new_h)
        new_left = cx - float(new_w) / 2.0

        # keep bottom fixed (no Y drift)
        self._apply_rect(left=new_left, bottom=r.bottom, height=new_h)
