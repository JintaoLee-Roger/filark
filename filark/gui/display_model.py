# filark/gui/display_model.py
from __future__ import annotations

from typing import Optional, Dict, Any
import numpy as np

from PySide6.QtCore import QObject, Signal, Qt
from vispy import scene

from filark.viz.canvas import StreamingCanvas
from filark.io.protocols import Tape


# =========================
# Coord / Router / Visual
# =========================
class CoordMixin:
    """Coordinate utilities shared by all interaction modes."""

    def canvas_pos_to_image(self, pos) -> tuple[float, float]:
        tr = self._canvas.scene.node_transform(self._canvas.image)
        w = tr.map(np.array(pos, dtype=np.float32)[:2])
        return float(w[0]), float(w[1])

    def image_pixel_to_world(self, px: float, py: float) -> tuple[float, float]:
        sc = self._canvas.view.scene
        tr = self._canvas.image.node_transform(sc)  # image(local) -> scene(world)
        w = tr.map((px, py, 0))[:2]
        return float(w[0]), float(w[1])
    
    def canvas_pos_to_layer(self, pos) -> tuple[float, float]:
        # canvas(pixel) -> scene(world)
        tr = self._canvas.world_layer.node_transform(self._canvas.scene)
        # 注意：node_transform(A) 是 “layer_node -> A” 的变换
        # 我们要反过来：A -> layer_node，所以要 inverse
        p = tr.inverse.map((pos[0], pos[1], 0))
        return float(p[0]), float(p[1])

    def canvas_pos_to_world(self, pos) -> tuple[float, float]:
        # px, py = self.canvas_pos_to_image(pos)
        # return self.image_pixel_to_world(px, py)
        return self.canvas_pos_to_layer(pos)


class EventRouterMixin:
    """Route canvas events to active interaction handlers (modes/plugins)."""

    def _init_event_router(self):
        self._handlers = {}          # name -> (priority, handler)
        self._handler_order = []     # ordered names
        self._active_handler = None  # str|None

    def register_handler(self, name: str, handler, *, priority: int = 100):
        self._handlers[name] = (priority, handler)
        self._handler_order = [k for k, _ in sorted(self._handlers.items(), key=lambda kv: kv[1][0])]

    def set_active_handler(self, name: Optional[str]):
        self._active_handler = name

    def _dispatch(self, method_name: str, ev):
        if getattr(ev, "handled", False):
            return

        # 1) active first
        if self._active_handler is not None:
            _, h = self._handlers.get(self._active_handler, (None, None))
            if h is not None:
                fn = getattr(h, method_name, None)
                if fn is not None:
                    fn(ev)
                    if getattr(ev, "handled", False):
                        return

        # 2) then others by priority
        for name in self._handler_order:
            if name == self._active_handler:
                continue
            _, h = self._handlers[name]
            fn = getattr(h, method_name, None)
            if fn is None:
                continue
            fn(ev)
            if getattr(ev, "handled", False):
                return


class VisualMixin:
    def set_source(self, source):
        self._source = source
        self._canvas.set_source(source)

    def set_display_params(self, *, clim=None, cmap=None, interpolation=None):
        if clim is not None:
            self.set_clim(*clim)
        if cmap is not None:
            self.set_cmap(cmap)
        if interpolation is not None:
            self.set_interpolation(interpolation)

    def set_clim(self, vmin, vmax):
        self._clim = (vmin, vmax)
        self._canvas.image.clim = [vmin, vmax]

    def set_cmap(self, cmap):
        self._cmap = cmap
        self._canvas.image.cmap = cmap

    def set_interpolation(self, interp):
        self._interp = interp
        self._canvas.image.interpolation = interp


class RectGeomMixin(CoordMixin):
    def _update_rect(self, rect, p0, p1):
        x0, y0 = p0
        x1, y1 = p1
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        w = abs(x1 - x0)
        h = abs(y1 - y0)

        rect.center = (cx, cy)
        rect.width = max(w, 1e-6)
        rect.height = max(h, 1e-6)
        self._canvas.update()


# =========================
# DisplayModel
# =========================
class DisplayModel(QObject, VisualMixin, EventRouterMixin, RectGeomMixin):
    bbox_finished = Signal(dict)         # {"t0","c0","t1","c1","modifiers":[...]}
    polyline_finished = Signal(dict)     # {"points":[(t,c)...]}
    roi_committed = Signal(dict)         # {"t0","c0","t1","c1"}

    def __init__(self, canvas: StreamingCanvas):
        super().__init__()
        self._canvas = canvas
        self._source: Optional[Tape] = None

        # display params
        self._clim = None
        self._cmap = None
        self._interp = None

        # overlay nodes
        self._overlays: Dict[int, Any] = {}

        # -------------------------
        # Annotation state
        # -------------------------
        self.ann_enabled = True
        self.ann_mode = "bbox"          # "bbox" | "polyline" | "brush"
        self.ann_label = "event"
        self.ann_hold_ctrl = True

        # gate: only Annotation tab should set True
        self._ann_gate_active = False

        # ctrl fallback (ONLY for key shortcuts; never used for mouse gating)
        self._ctrl_down = False

        # -------------------------
        # ROI state
        # -------------------------
        self._roi_enable = False

        # -------------------------
        # Shared drag session for rectangles
        # -------------------------
        # None | "ann_bbox" | "roi"
        self._drag_kind: Optional[str] = None
        self._drag_rect = None
        self._drag_p0 = None
        self._drag_p1 = None

        # -------------------------
        # Polyline session
        # -------------------------
        self._poly_points = []
        self._poly_preview = None

        # init router + visuals
        self._init_event_router()
        self._init_bbox_preview()
        self._init_roi_preview()
        self._init_polyline_preview()

        # handlers
        self._bbox_handler = _SelfHandlerProxy(self, kind="bbox")
        self._poly_handler = _SelfHandlerProxy(self, kind="polyline")
        self._roi_handler = _SelfHandlerProxy(self, kind="roi")

        self.register_handler("bbox", self._bbox_handler, priority=10)
        self.register_handler("polyline", self._poly_handler, priority=10)
        self.register_handler("roi", self._roi_handler, priority=11)

        # bind events
        self._bind_canvas_events()

        # default camera interactive
        self._canvas.camera.interactive = True
        self._refresh_active_handler()

    # =========================================================
    # Previews
    # =========================================================
    def _init_bbox_preview(self):
        # Vispy backend may not render alpha=0 reliably; use tiny alpha to "look transparent".
        self._bbox_rect = scene.visuals.Rectangle(
            center=(0, 0),
            width=1,
            height=1,
            color=(0, 0, 0, 0.001),
            border_color="yellow",
            parent=self._canvas.world_layer,
        )
        self._bbox_rect.visible = False

    def _init_roi_preview(self):
        self._roi_rect = scene.visuals.Rectangle(
            center=(0, 0),
            width=1,
            height=1,
            color=(0, 0, 0, 0.001),
            border_color="red",
            parent=self._canvas.world_layer,
        )
        self._roi_rect.visible = False

    def _init_polyline_preview(self):
        self._poly_preview = scene.visuals.Line(
            pos=np.zeros((0, 2), dtype=np.float32),
            color="yellow",
            parent=self._canvas.world_layer,
            method="gl",
            width=2,
        )
        self._poly_preview.visible = False

    # =========================================================
    # Public API: gate / roi / mode
    # =========================================================
    def set_annotation_gate(self, active: bool):
        """Only Annotation panel/tab should set this True.
        When gate is off, annotation must not interfere with ROI/analysis.
        """
        self._ann_gate_active = bool(active)
        if not self._ann_gate_active:
            self._cancel_bbox_drag()
            self._cancel_polyline_session()
            self._annot_unlock_camera()
        self._refresh_active_handler()

    def set_analysis_roi_mode(self, enabled: bool):
        """ROI mode takes over mouse events; annotation must not run."""
        self._roi_enable = bool(enabled)
        if self._roi_enable:
            self._cancel_bbox_drag()
            self._cancel_polyline_session()
            self._roi_rect.visible = False
            self._canvas.camera.interactive = False
        else:
            self._drag_kind = None
            self._roi_rect.visible = False
            self._canvas.camera.interactive = True
            self._canvas.update()
        self._refresh_active_handler()

    def set_mode(self, mode: str):
        if mode not in ("bbox", "polyline", "brush"):
            mode = "bbox"
        self._cancel_bbox_drag()
        self._cancel_polyline_session()
        self.ann_mode = mode
        self._refresh_active_handler()

    def set_label(self, label: str):
        self.ann_label = (label or "event").strip() or "event"

    def set_annotation_enabled(self, enabled: bool):
        self.ann_enabled = bool(enabled)
        if not self.ann_enabled:
            self._cancel_bbox_drag()
            self._cancel_polyline_session()
            self._annot_unlock_camera()
        self._refresh_active_handler()

    def set_annotation_hold_ctrl(self, enabled: bool):
        self.ann_hold_ctrl = bool(enabled)

    def set_annotation_state(self, *, mode: str, label: str, enabled: bool = True, attrs_json: Optional[str] = None):
        self.set_annotation_enabled(enabled)
        self.set_mode(mode)
        self.set_label(label)

    def _set_roi_region(self, roi):
        """Called by analysis panel to show an existing ROI."""
        try:
            self._update_rect(self._roi_rect, (roi.t0, roi.c0), (roi.t1, roi.c1))
            self._roi_rect.visible = True
            self._canvas.update()
        except Exception:
            pass

    # =========================================================
    # Overlay API
    # =========================================================
    def add_bbox_overlay(self, uid: int, bbox: dict):
        rect = scene.visuals.Rectangle(
            center=(0, 0),
            width=1,
            height=1,
            color=(0, 0, 0, 0.001),
            border_color="yellow",
            parent=self._canvas.world_layer,
        )
        rect.visible = True
        self._overlays[int(uid)] = rect
        self._update_rect(rect, (bbox["t0"], bbox["c0"]), (bbox["t1"], bbox["c1"]))
        self._canvas.update()

    def add_polyline_overlay(self, uid: int, data: dict):
        pts = data.get("points", []) or []
        arr = np.array(pts, dtype=np.float32).reshape(-1, 2) if pts else np.zeros((0, 2), np.float32)
        line = scene.visuals.Line(pos=arr, color="yellow", parent=self._canvas.world_layer, method="gl", width=2)
        line.visible = True
        self._overlays[int(uid)] = line
        self._canvas.update()

    def update_overlay(self, uid: int, patch: dict):
        uid = int(uid)
        node = self._overlays.get(uid)
        if node is None:
            return

        if isinstance(node, scene.visuals.Rectangle):
            if all(k in patch for k in ("t0", "c0", "t1", "c1")):
                self._update_rect(node, (patch["t0"], patch["c0"]), (patch["t1"], patch["c1"]))
                self._canvas.update()
            return

        if isinstance(node, scene.visuals.Line):
            if "points" in patch:
                pts = patch.get("points", []) or []
                arr = np.array(pts, dtype=np.float32).reshape(-1, 2) if pts else np.zeros((0, 2), np.float32)
                node.set_data(pos=arr)
                self._canvas.update()
            return

    def remove_overlay(self, uid: int):
        uid = int(uid)
        node = self._overlays.pop(uid, None)
        if node is None:
            return
        node.parent = None
        node.visible = False
        self._canvas.update()

    def clear_overlays(self):
        for _, node in list(self._overlays.items()):
            node.parent = None
            node.visible = False
        self._overlays.clear()
        self._canvas.update()

    # =========================================================
    # Event binding
    # =========================================================
    def _bind_canvas_events(self):
        self._canvas.events.mouse_press.connect(self._on_mouse_press_global)
        self._canvas.events.mouse_move.connect(self._on_mouse_move_global)
        self._canvas.events.mouse_release.connect(self._on_mouse_release_global)
        self._canvas.events.key_press.connect(self._on_key_press_global)
        self._canvas.events.key_release.connect(self._on_key_release_global)

    def _on_mouse_press_global(self, ev):
        # If annotation gate is active, ensure focus so key events are reliable
        if self._ann_gate_active:
            try:
                self._canvas.native.setFocus()
            except Exception:
                pass
        self._dispatch("on_mouse_press", ev)

    def _on_mouse_move_global(self, ev):
        # If ctrl is required and currently not pressed, cancel bbox drag (but keep polyline session)
        if self.ann_hold_ctrl and (not self.is_ctrl_pressed(ev)):
            self._on_ctrl_released_during_mouse(ev)
        self._dispatch("on_mouse_move", ev)

    def _on_mouse_release_global(self, ev):
        self._dispatch("on_mouse_release", ev)
        if self.ann_hold_ctrl and (not self.is_ctrl_pressed(ev)):
            self._on_ctrl_released_during_mouse(ev)

    def _on_key_press_global(self, ev):
        if self._is_ctrl_key(ev):
            self._ctrl_down = True
        self._dispatch("on_key_press", ev)

    def _on_key_release_global(self, ev):
        if self._is_ctrl_key(ev):
            self._ctrl_down = False
            # releasing ctrl => cancel bbox drag, unlock camera if not in polyline session
            self._on_ctrl_released_during_key(ev)
        self._dispatch("on_key_release", ev)

    def _is_ctrl_key(self, ev) -> bool:
        k = getattr(getattr(ev, "key", None), "name", None) or str(getattr(ev, "key", ""))
        k = str(k).upper()
        return ("CTRL" in k) or ("CONTROL" in k)

    # =========================================================
    # Ctrl detection: two purposes with clear names
    # =========================================================
    def is_ctrl_pressed(self, ev) -> bool:
        """Return True iff Ctrl is pressed IN THIS EVENT.
        Used for mouse gating to prevent 'sticky ctrl' bugs.
        """
        # 1) vispy modifiers
        mods = getattr(ev, "modifiers", None)
        if mods is not None:
            mods = mods or []
            if ("Control" in mods) or ("CTRL" in mods):
                return True

        # 2) Qt native event modifiers
        native = getattr(ev, "native", None)
        if native is not None:
            try:
                return bool(native.modifiers() & Qt.ControlModifier)
            except Exception:
                pass

        return False

    def is_ctrl_shortcut(self, ev) -> bool:
        """Return True iff event corresponds to a Ctrl-based shortcut.
        Used for key shortcuts (e.g., Ctrl+Z). Allows fallback _ctrl_down.
        """
        mods = getattr(ev, "modifiers", None) or []
        if ("Control" in mods) or ("CTRL" in mods):
            return True
        return bool(self._ctrl_down)

    # =========================================================
    # Gate policy
    # =========================================================
    def _annotation_allowed_now(self) -> bool:
        if not self._ann_gate_active:
            return False
        if not self.ann_enabled:
            return False
        if self._roi_enable:
            return False
        if self.ann_mode not in ("bbox", "polyline"):
            return False
        return True

    def _refresh_active_handler(self):
        if self._roi_enable:
            self.set_active_handler("roi")
            return

        if not self._ann_gate_active or (not self.ann_enabled):
            self.set_active_handler(None)
            return

        if self.ann_mode in ("bbox", "polyline"):
            self.set_active_handler(self.ann_mode)
        else:
            self.set_active_handler(None)

    # =========================================================
    # Camera helpers
    # =========================================================
    def _annot_lock_camera(self):
        if self._roi_enable:
            return
        self._canvas.camera.interactive = False

    def _annot_unlock_camera(self):
        if self._roi_enable:
            return
        self._canvas.camera.interactive = True

    # =========================================================
    # Rectangle drag session (shared)
    # =========================================================
    def _begin_drag_rect(self, kind: str, rect_node, ev):
        self._drag_kind = kind
        self._drag_rect = rect_node
        x, y = self.canvas_pos_to_world(ev.pos)
        self._drag_p0 = (x, y)
        self._drag_p1 = (x, y)
        rect_node.visible = True
        self._update_rect(rect_node, self._drag_p0, self._drag_p1)
        self._canvas.update()

    def _move_drag_rect(self, ev) -> bool:
        if self._drag_kind is None:
            return False
        if hasattr(ev, "buttons") and (1 not in ev.buttons):
            return False
        x, y = self.canvas_pos_to_world(ev.pos)
        self._drag_p1 = (x, y)
        self._update_rect(self._drag_rect, self._drag_p0, self._drag_p1)
        return True

    def _end_drag_rect(self, ev):
        if self._drag_kind is None:
            return None

        x, y = self.canvas_pos_to_world(ev.pos)
        self._drag_p1 = (x, y)
        self._update_rect(self._drag_rect, self._drag_p0, self._drag_p1)

        t0, c0 = self._drag_p0
        t1, c1 = self._drag_p1

        kind = self._drag_kind
        rect = self._drag_rect

        self._drag_kind = None
        self._drag_rect = None
        self._drag_p0 = None
        self._drag_p1 = None

        rect.visible = False
        self._canvas.update()

        return kind, {"t0": t0, "c0": c0, "t1": t1, "c1": c1}

    # =========================================================
    # Polyline session helpers
    # =========================================================
    def _set_poly_preview(self, pts):
        arr = np.array(pts, dtype=np.float32).reshape(-1, 2) if pts else np.zeros((0, 2), np.float32)
        self._poly_preview.set_data(pos=arr)
        self._canvas.update()

    def _update_poly_preview(self, mouse_xy=None):
        pts = list(self._poly_points)
        if mouse_xy is not None and pts:
            pts = pts + [mouse_xy]
        self._poly_preview.visible = bool(pts)
        self._set_poly_preview(pts)

    def _cancel_polyline_session(self):
        self._poly_points = []
        self._poly_preview.visible = False
        self._set_poly_preview([])
        self._annot_unlock_camera()

    def _undo_polyline_point(self):
        if not self._poly_points:
            return
        self._poly_points.pop()
        if self._poly_points:
            self._poly_preview.visible = True
            self._update_poly_preview(mouse_xy=None)
        else:
            self._cancel_polyline_session()

    def _finish_polyline(self):
        if len(self._poly_points) < 2:
            self._cancel_polyline_session()
            return
        self.polyline_finished.emit({"points": list(self._poly_points)})
        self._cancel_polyline_session()

    # =========================================================
    # Ctrl release policy (important)
    # =========================================================
    def _on_ctrl_released_during_mouse(self, ev):
        """Mouse events sometimes don't carry modifiers; we only use is_ctrl_pressed(ev).
        When ctrl is not pressed:
          - cancel bbox drag immediately
          - do NOT cancel polyline session
          - unlock camera only if not in polyline session
        """
        if not self._ann_gate_active or self._roi_enable:
            return

        # cancel bbox dragging if active
        if self._drag_kind == "ann_bbox":
            self._cancel_bbox_drag()
            self._annot_unlock_camera()

        # unlock camera only if not in polyline session
        if not self._poly_points:
            self._annot_unlock_camera()

    def _on_ctrl_released_during_key(self, ev):
        """Key release is reliable for ctrl. Same policy as mouse cleanup."""
        if not self._ann_gate_active or self._roi_enable:
            return

        if self._drag_kind == "ann_bbox":
            self._cancel_bbox_drag()
            self._annot_unlock_camera()

        if not self._poly_points:
            self._annot_unlock_camera()

    # =========================================================
    # Cancel helpers
    # =========================================================
    def _cancel_bbox_drag(self):
        if self._drag_kind == "ann_bbox":
            if self._drag_rect is not None:
                self._drag_rect.visible = False
            self._drag_kind = None
            self._drag_rect = None
            self._drag_p0 = None
            self._drag_p1 = None
            self._canvas.update()

    # =========================================================
    # Handlers: bbox / polyline / roi
    # =========================================================
    # ---- bbox (annotation) ----
    def on_mouse_press_bbox(self, ev):
        if getattr(ev, "button", None) != 1:
            return
        if not self._annotation_allowed_now():
            return
        if self.ann_mode != "bbox":
            return

        # Mouse gating must use "event-real ctrl" only
        if self.ann_hold_ctrl and (not self.is_ctrl_pressed(ev)):
            return

        self._annot_lock_camera()
        self._begin_drag_rect("ann_bbox", self._bbox_rect, ev)
        ev.handled = True

    def on_mouse_move_bbox(self, ev):
        if self._drag_kind != "ann_bbox":
            return

        # if ctrl released => cancel immediately
        if self.ann_hold_ctrl and (not self.is_ctrl_pressed(ev)):
            self._cancel_bbox_drag()
            self._annot_unlock_camera()
            ev.handled = True
            return

        if self._move_drag_rect(ev):
            ev.handled = True

    def on_mouse_release_bbox(self, ev):
        if self._drag_kind != "ann_bbox":
            return
        if getattr(ev, "button", None) != 1:
            return

        # if ctrl released => treat as cancel
        if self.ann_hold_ctrl and (not self.is_ctrl_pressed(ev)):
            self._cancel_bbox_drag()
            self._annot_unlock_camera()
            ev.handled = True
            return

        ended = self._end_drag_rect(ev)
        if ended is None:
            return
        _, geom = ended

        self._annot_unlock_camera()

        mods = getattr(ev, "modifiers", None)
        geom["modifiers"] = list(mods) if mods else []
        self.bbox_finished.emit(geom)
        ev.handled = True

    # ---- polyline (annotation) ----
    def on_mouse_press_polyline(self, ev):
        if getattr(ev, "button", None) not in (1, 2):
            return
        if not self._annotation_allowed_now():
            return
        if self.ann_mode != "polyline":
            return

        b = getattr(ev, "button", None)

        if b == 1:
            # first point requires ctrl (if enabled)
            if not self._poly_points:
                if self.ann_hold_ctrl and (not self.is_ctrl_pressed(ev)):
                    return
                self._annot_lock_camera()
                try:
                    self._canvas.native.setFocus()
                except Exception:
                    pass

            x, y = self.canvas_pos_to_world(ev.pos)
            self._poly_points.append((x, y))
            self._poly_preview.visible = True
            self._update_poly_preview(mouse_xy=(x, y))
            ev.handled = True
            return

        if b == 2:
            self._finish_polyline()
            ev.handled = True
            return

    def on_mouse_move_polyline(self, ev):
        if not self._poly_points:
            return
        x, y = self.canvas_pos_to_world(ev.pos)
        self._update_poly_preview(mouse_xy=(x, y))
        ev.handled = True

    def on_mouse_release_polyline(self, ev):
        return

    def on_key_press_polyline(self, ev):
        key = ev.key.name if hasattr(ev.key, "name") else str(ev.key)
        k = str(key).upper()

        if not self._poly_points:
            return

        # Ctrl+Z: undo last point
        if k == "Z" and self.is_ctrl_shortcut(ev):
            self._undo_polyline_point()
            ev.handled = True
            return

        # Backspace/Delete: undo last point
        if k in ("BACKSPACE", "DELETE"):
            self._undo_polyline_point()
            ev.handled = True
            return

        if k in ("ENTER", "RETURN"):
            self._finish_polyline()
            ev.handled = True
            return

        if k in ("ESCAPE", "ESC"):
            self._cancel_polyline_session()
            ev.handled = True
            return

    # ---- roi (analysis) ----
    def on_mouse_press_roi(self, ev):
        if not self._roi_enable:
            return
        if getattr(ev, "button", None) != 1:
            return
        self._begin_drag_rect("roi", self._roi_rect, ev)
        ev.handled = True

    def on_mouse_move_roi(self, ev):
        if self._drag_kind != "roi":
            return
        if self._move_drag_rect(ev):
            ev.handled = True

    def on_mouse_release_roi(self, ev):
        if self._drag_kind != "roi":
            return
        if getattr(ev, "button", None) != 1:
            return
        ended = self._end_drag_rect(ev)
        if ended is None:
            return
        _, geom = ended
        self.roi_committed.emit(geom)
        ev.handled = True

    def clear(self):
        self.clear_overlays()
        # self._roi_rect.
        self._canvas.clear_source()


# =========================
# Router proxy
# =========================
class _SelfHandlerProxy:
    def __init__(self, owner: DisplayModel, kind: str):
        self._o = owner
        self._k = kind

    def on_mouse_press(self, ev):
        if self._k == "bbox":
            return self._o.on_mouse_press_bbox(ev)
        if self._k == "polyline":
            return self._o.on_mouse_press_polyline(ev)
        if self._k == "roi":
            return self._o.on_mouse_press_roi(ev)

    def on_mouse_move(self, ev):
        if self._k == "bbox":
            return self._o.on_mouse_move_bbox(ev)
        if self._k == "polyline":
            return self._o.on_mouse_move_polyline(ev)
        if self._k == "roi":
            return self._o.on_mouse_move_roi(ev)

    def on_mouse_release(self, ev):
        if self._k == "bbox":
            return self._o.on_mouse_release_bbox(ev)
        if self._k == "polyline":
            return self._o.on_mouse_release_polyline(ev)
        if self._k == "roi":
            return self._o.on_mouse_release_roi(ev)

    def on_key_press(self, ev):
        if self._k == "polyline":
            return self._o.on_key_press_polyline(ev)
