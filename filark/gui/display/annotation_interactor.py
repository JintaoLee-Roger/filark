"""Annotation and ROI interaction state machine for DisplayModel."""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
from vispy import scene

from filark._exceptions import print_exception
from filark.viz import AnnotationBBox, AnnotationPolyline, MaskOverlay

from .annotation_events import AnnotationEventMixin
from .overlays import AnnotationBBoxNode, AnnotationPolylineNode, MaskOverlayNode, OverlayRecord


def _print_exception(context: str) -> None:
    print_exception(context, prefix="DisplayModel")


class _AnnotationHandlerProxy:
    """Route generic canvas handler calls to one annotation interaction mode."""

    def __init__(self, owner, kind: str):
        self._owner = owner
        self._kind = str(kind)

    def on_mouse_press(self, ev):
        if self._kind == "edit":
            return self._owner.on_mouse_press_edit(ev)
        if self._kind == "bbox":
            return self._owner.on_mouse_press_bbox(ev)
        if self._kind == "polyline":
            return self._owner.on_mouse_press_polyline(ev)
        if self._kind == "mask":
            return self._owner.on_mouse_press_mask(ev)
        if self._kind == "roi":
            return self._owner.on_mouse_press_roi(ev)

    def on_mouse_move(self, ev):
        if self._kind == "edit":
            return self._owner.on_mouse_move_edit(ev)
        if self._kind == "bbox":
            return self._owner.on_mouse_move_bbox(ev)
        if self._kind == "polyline":
            return self._owner.on_mouse_move_polyline(ev)
        if self._kind == "mask":
            return self._owner.on_mouse_move_mask(ev)
        if self._kind == "roi":
            return self._owner.on_mouse_move_roi(ev)

    def on_mouse_release(self, ev):
        if self._kind == "edit":
            return self._owner.on_mouse_release_edit(ev)
        if self._kind == "bbox":
            return self._owner.on_mouse_release_bbox(ev)
        if self._kind == "polyline":
            return self._owner.on_mouse_release_polyline(ev)
        if self._kind == "mask":
            return self._owner.on_mouse_release_mask(ev)
        if self._kind == "roi":
            return self._owner.on_mouse_release_roi(ev)

    def on_key_press(self, ev):
        if self._kind == "edit":
            return self._owner.on_key_press_edit(ev)
        if self._kind == "polyline":
            return self._owner.on_key_press_polyline(ev)
        if self._kind == "mask":
            return self._owner.on_key_press_mask(ev)


class AnnotationEditSessionMixin:
    """Own bbox/polyline edit session lifecycle and previews."""

    def _begin_bbox_edit(self, ev, mode: str) -> None:
        record = self._selected_record()
        payload = self._selected_payload()
        if record is None or payload is None:
            return
        x, y = self.canvas_pos_to_world(ev.pos)
        self._bbox_edit_session = {
            "uid": record.uid,
            "mode": mode,
            "origin_mouse": (x, y),
            "origin_bbox": self._normalized_bbox(payload),
            "current_bbox": self._normalized_bbox(payload),
        }
        record.node.set_visible(False)
        self._annot_lock_camera()
        self._refresh_selection_visuals()

    def _bbox_from_edit_session(self, x: float, y: float) -> Optional[dict]:
        session = self._bbox_edit_session
        if session is None:
            return None
        origin = dict(session["origin_bbox"])
        dx = x - session["origin_mouse"][0]
        dy = y - session["origin_mouse"][1]
        mode = session["mode"]

        bbox = dict(origin)
        if mode == "move":
            bbox["t0"] = origin["t0"] + dx
            bbox["t1"] = origin["t1"] + dx
            bbox["c0"] = origin["c0"] + dy
            bbox["c1"] = origin["c1"] + dy
        else:
            if mode in ("left", "top_left", "bottom_left"):
                bbox["t0"] = origin["t0"] + dx
            if mode in ("right", "top_right", "bottom_right"):
                bbox["t1"] = origin["t1"] + dx
            if mode in ("top", "top_left", "top_right"):
                bbox["c0"] = origin["c0"] + dy
            if mode in ("bottom", "bottom_left", "bottom_right"):
                bbox["c1"] = origin["c1"] + dy
        return self._normalized_bbox(bbox)

    def _preview_bbox_edit(self, bbox: dict) -> None:
        session = self._bbox_edit_session
        if session is None:
            return
        session["current_bbox"] = dict(bbox)
        self._refresh_selection_visuals()

    def _finish_bbox_edit_session(self) -> Optional[tuple[int, dict]]:
        session = self._bbox_edit_session
        record = self._selected_record()
        self._bbox_edit_session = None
        self._annot_unlock_camera()
        if session is None or record is None:
            self._refresh_selection_visuals()
            return None
        final_bbox = self._rounded_bbox(session["current_bbox"])
        self._refresh_selection_visuals()
        return record.uid, final_bbox

    def _cancel_bbox_edit_session(self) -> None:
        session = self._bbox_edit_session
        self._bbox_edit_session = None
        if session is None:
            return
        record = self._selected_record()
        if record is not None:
            record.node.set_visible(self._annotation_visible(record.payload))
        self._annot_unlock_camera()
        self._refresh_selection_visuals()

    def _begin_poly_edit(self, ev, vertex_index: int) -> None:
        record = self._selected_record()
        payload = self._selected_payload()
        if record is None or payload is None:
            return
        points = [tuple(point) for point in (payload.get("points", []) or [])]
        if not points or not (0 <= vertex_index < len(points)):
            return
        self._poly_edit_session = {
            "uid": record.uid,
            "index": int(vertex_index),
            "current_points": list(points),
            "origin_points": list(points),
        }
        record.node.set_visible(False)
        self._annot_lock_camera()
        self._refresh_selection_visuals()

    def _delete_polyline_vertex(self, uid: int, points: list[tuple[float, float]], index: int) -> None:
        if len(points) <= 2 or not (0 <= index < len(points)):
            return
        new_points = list(points)
        new_points.pop(index)
        record = self._overlays.get(int(uid))
        if record is not None:
            record.payload["points"] = list(new_points)
        self.set_selected_annotation(uid, "polyline", {"points": new_points})
        self.annotation_geometry_edited.emit(uid, {"points": new_points})

    def _preview_poly_edit(self, points: list[tuple[float, float]]) -> None:
        session = self._poly_edit_session
        if session is None:
            return
        session["current_points"] = list(points)
        self._refresh_selection_visuals()

    def _finish_poly_edit_session(self) -> Optional[tuple[int, dict]]:
        session = self._poly_edit_session
        record = self._selected_record()
        self._poly_edit_session = None
        self._annot_unlock_camera()
        if session is None or record is None:
            self._refresh_selection_visuals()
            return None
        final_points = [tuple(map(float, point)) for point in session["current_points"]]
        self._refresh_selection_visuals()
        return record.uid, {"points": final_points}

    def _cancel_poly_edit_session(self) -> None:
        session = self._poly_edit_session
        self._poly_edit_session = None
        if session is None:
            return
        record = self._selected_record()
        if record is not None:
            record.node.set_visible(self._annotation_visible(record.payload))
        self._annot_unlock_camera()
        self._refresh_selection_visuals()


class AnnotationMaskSessionMixin:
    """Own mask brush editing, preview, and brush undo/redo."""

    def set_mask_brush(self, *, radius: int, mode: str = "paint") -> None:
        self._mask_brush_radius = int(max(1, radius))
        self._mask_brush_mode = "erase" if str(mode) == "erase" else "paint"

    def _cancel_mask_paint_session(self) -> None:
        session = self._mask_paint_session
        self._mask_paint_session = None
        if session is None:
            return
        if session.get("uid") is None and self._active_mask_node is not None:
            self._active_mask_node.clear()
        record = self._selected_record()
        if record is not None:
            record.node.set_visible(self._annotation_visible(record.payload))
        self._annot_unlock_camera()
        self._refresh_selection_visuals()

    def _begin_mask_paint_session(self, ev) -> bool:
        x, y = self.canvas_pos_to_world(ev.pos)
        radius_t, radius_c = self._mask_brush_data_radii(ev)
        record = self._selected_record()
        uid = None
        before = None
        if record is not None and record.anno_type == "mask" and record.editable:
            uid = int(record.uid)
            payload = self._copy_mask_payload(record.payload)
            before = self._copy_mask_payload(record.payload)
        else:
            payload = self._new_mask_payload(x, y, radius_t=radius_t, radius_c=radius_c)

        self._mask_paint_session = {"uid": uid, "before": before, "payload": payload, "changed": False}
        self._annot_lock_camera()
        return self._paint_mask_session_at(x, y, radius_t=radius_t, radius_c=radius_c, erase=self._mask_erase_for_event(ev))

    def _paint_mask_session_event(self, ev) -> bool:
        if self._mask_paint_session is None:
            return False
        x, y = self.canvas_pos_to_world(ev.pos)
        radius_t, radius_c = self._mask_brush_data_radii(ev)
        return self._paint_mask_session_at(x, y, radius_t=radius_t, radius_c=radius_c, erase=self._mask_erase_for_event(ev))

    def _finish_mask_paint_session(self):
        session = self._mask_paint_session
        self._mask_paint_session = None
        self._annot_unlock_camera()
        if session is None or not bool(session.get("changed", False)):
            if session is not None and session.get("uid") is None and self._active_mask_node is not None:
                self._active_mask_node.clear()
            self._refresh_selection_visuals()
            return None

        uid = session.get("uid")
        payload = self._trim_mask_payload(self._copy_mask_payload(session["payload"]))
        if uid is None:
            if self._active_mask_node is not None:
                self._active_mask_node.clear()
            self._refresh_selection_visuals()
            return None, payload

        before = session.get("before")
        if before is not None:
            self._push_mask_undo(int(uid), before=before, after=payload)
        self._refresh_selection_visuals()
        return int(uid), payload

    def _paint_mask_session_at(self, x: float, y: float, *, radius_t: float, radius_c: float, erase: bool) -> bool:
        session = self._mask_paint_session
        if session is None:
            return False
        payload = self._ensure_mask_payload_contains(
            self._copy_mask_payload(session["payload"]),
            x,
            y,
            radius_t=radius_t,
            radius_c=radius_c,
        )
        changed = self._paint_mask_payload(payload, x, y, radius_t=radius_t, radius_c=radius_c, erase=erase)
        session["payload"] = payload
        session["changed"] = bool(session.get("changed", False) or changed)
        self._apply_mask_paint_preview(payload, uid=session.get("uid"))
        return True

    def _mask_brush_data_radii(self, ev) -> tuple[float, float]:
        radius_px = float(max(1, self._mask_brush_radius))
        pos = getattr(ev, "pos", None)
        if pos is None:
            return radius_px, radius_px
        try:
            px, py = float(pos[0]), float(pos[1])
            cx, cy = self.canvas_pos_to_world((px, py))
            xp, _ = self.canvas_pos_to_world((px + radius_px, py))
            xm, _ = self.canvas_pos_to_world((px - radius_px, py))
            _, yp = self.canvas_pos_to_world((px, py + radius_px))
            _, ym = self.canvas_pos_to_world((px, py - radius_px))
            radius_t = max(abs(float(xp) - float(cx)), abs(float(xm) - float(cx)), 0.5)
            radius_c = max(abs(float(yp) - float(cy)), abs(float(ym) - float(cy)), 0.5)
            if not (np.isfinite(radius_t) and np.isfinite(radius_c)):
                return radius_px, radius_px
            return float(radius_t), float(radius_c)
        except Exception:
            _print_exception("failed to convert mask brush radius to data units")
            return radius_px, radius_px

    def _apply_mask_paint_preview(self, payload: dict, *, uid) -> None:
        if uid is None:
            self._ensure_active_mask_capacity(payload)
            self._active_mask_node.show(
                payload.get("mask"),
                t0=float(payload.get("t0", 0)),
                c0=float(payload.get("c0", 0)),
                t1=float(payload.get("t1", 0)),
                c1=float(payload.get("c1", 0)),
                color=str(payload.get("color", self._annotation_preview_color)),
                alpha=float(payload.get("alpha", 0.45)),
                threshold=float(payload.get("threshold", 0.0)),
                visible=True,
            )
            self._canvas.update()
            return

        record = self._overlays.get(int(uid))
        if record is None:
            return
        record.payload.update(payload)
        self._selected_annotation_payload = dict(record.payload)
        self._apply_overlay_record(record)

    def _ensure_active_mask_capacity(self, payload: dict) -> None:
        mask = np.asarray(payload.get("mask"), dtype=np.uint8)
        nt, nc = int(mask.shape[0]), int(mask.shape[1])
        node = self._active_mask_node
        if node is not None:
            cap_t, cap_c = node.capacity
            if cap_t >= nt and cap_c >= nc:
                return
            node.close()
        self._active_mask_node = MaskOverlay(
            parent=self._world_ui_parent,
            capacity_t=max(1, nt),
            capacity_c=max(1, nc),
            color=self._annotation_preview_color,
            alpha=0.45,
            threshold=0.0,
        )
        self._active_mask_node.order = 39

    def _mask_erase_for_event(self, ev) -> bool:
        return bool(self._mask_brush_mode == "erase" or self.is_shift_pressed(ev))

    def _copy_mask_payload(self, payload: dict) -> dict:
        out = dict(payload)
        mask = out.get("mask")
        out["mask"] = np.ascontiguousarray(np.asarray(mask, dtype=np.uint8)).copy() if mask is not None else np.zeros((1, 1), dtype=np.uint8)
        out["t0"] = int(round(float(out.get("t0", 0))))
        out["c0"] = int(round(float(out.get("c0", 0))))
        out["t1"] = int(round(float(out.get("t1", out["t0"] + out["mask"].shape[0]))))
        out["c1"] = int(round(float(out.get("c1", out["c0"] + out["mask"].shape[1]))))
        out.setdefault("alpha", 0.45)
        out.setdefault("threshold", 0.0)
        return out

    def _new_mask_payload(self, x: float, y: float, *, radius_t: float, radius_c: float) -> dict:
        margin_t = max(4, int(np.ceil(float(max(radius_t, 0.5)) * 0.25)))
        margin_c = max(4, int(np.ceil(float(max(radius_c, 0.5)) * 0.25)))
        t0, t1, c0, c1 = self._clamp_mask_bounds(
            int(np.floor(float(x) - float(radius_t))) - margin_t,
            int(np.ceil(float(x) + float(radius_t))) + margin_t + 1,
            int(np.floor(float(y) - float(radius_c))) - margin_c,
            int(np.ceil(float(y) + float(radius_c))) + margin_c + 1,
        )
        return {
            "t0": int(t0),
            "t1": int(t1),
            "c0": int(c0),
            "c1": int(c1),
            "mask": np.zeros((max(1, int(t1 - t0)), max(1, int(c1 - c0))), dtype=np.uint8),
            "alpha": 0.45,
            "threshold": 0.0,
        }

    def _ensure_mask_payload_contains(self, payload: dict, x: float, y: float, *, radius_t: float, radius_c: float) -> dict:
        payload = self._copy_mask_payload(payload)
        radius_t = float(max(0.5, radius_t))
        radius_c = float(max(0.5, radius_c))
        pad_t = max(2, int(np.ceil(radius_t * 0.25)))
        pad_c = max(2, int(np.ceil(radius_c * 0.25)))
        min_t = int(np.floor(float(x) - radius_t)) - pad_t
        max_t = int(np.ceil(float(x) + radius_t)) + pad_t + 1
        min_c = int(np.floor(float(y) - radius_c)) - pad_c
        max_c = int(np.ceil(float(y) + radius_c)) + pad_c + 1

        old_t0, old_t1 = int(payload["t0"]), int(payload["t1"])
        old_c0, old_c1 = int(payload["c0"]), int(payload["c1"])
        new_t0, new_t1, new_c0, new_c1 = self._clamp_mask_bounds(
            min(old_t0, min_t),
            max(old_t1, max_t),
            min(old_c0, min_c),
            max(old_c1, max_c),
        )
        if (new_t0, new_t1, new_c0, new_c1) == (old_t0, old_t1, old_c0, old_c1):
            return payload

        old = np.asarray(payload["mask"], dtype=np.uint8)
        expanded = np.zeros((max(1, new_t1 - new_t0), max(1, new_c1 - new_c0)), dtype=np.uint8)
        dst_t0 = max(0, old_t0 - new_t0)
        dst_c0 = max(0, old_c0 - new_c0)
        src_t0 = max(0, new_t0 - old_t0)
        src_c0 = max(0, new_c0 - old_c0)
        copy_t = min(old.shape[0] - src_t0, expanded.shape[0] - dst_t0)
        copy_c = min(old.shape[1] - src_c0, expanded.shape[1] - dst_c0)
        if copy_t > 0 and copy_c > 0:
            expanded[dst_t0:dst_t0 + copy_t, dst_c0:dst_c0 + copy_c] = old[
                src_t0:src_t0 + copy_t, src_c0:src_c0 + copy_c
            ]
        payload.update({"t0": int(new_t0), "t1": int(new_t1), "c0": int(new_c0), "c1": int(new_c1), "mask": expanded})
        return payload

    def _paint_mask_payload(self, payload: dict, x: float, y: float, *, radius_t: float, radius_c: float, erase: bool) -> bool:
        arr = np.asarray(payload["mask"], dtype=np.uint8)
        cx = float(x) - float(payload["t0"])
        cy = float(y) - float(payload["c0"])
        radius_t = float(max(0.5, radius_t))
        radius_c = float(max(0.5, radius_c))
        t0 = max(0, int(np.floor(cx - radius_t)))
        t1 = min(arr.shape[0], int(np.ceil(cx + radius_t)) + 1)
        c0 = max(0, int(np.floor(cy - radius_c)))
        c1 = min(arr.shape[1], int(np.ceil(cy + radius_c)) + 1)
        if t1 <= t0 or c1 <= c0:
            return False
        tt, cc = np.ogrid[t0:t1, c0:c1]
        disk = ((tt - cx) / radius_t) ** 2 + ((cc - cy) / radius_c) ** 2 <= 1.0
        view = arr[t0:t1, c0:c1]
        before = view.copy()
        view[disk] = 0 if erase else 255
        payload["mask"] = arr
        return not np.array_equal(before, view)

    def _trim_mask_payload(self, payload: dict) -> dict:
        payload = self._copy_mask_payload(payload)
        arr = np.asarray(payload["mask"], dtype=np.uint8)
        nonzero = np.argwhere(arr > 0)
        if nonzero.size == 0:
            payload["mask"] = np.zeros((1, 1), dtype=np.uint8)
            payload["t1"] = int(payload["t0"]) + 1
            payload["c1"] = int(payload["c0"]) + 1
            return payload
        lo_t, lo_c = nonzero.min(axis=0)
        hi_t, hi_c = nonzero.max(axis=0) + 1
        payload["mask"] = np.ascontiguousarray(arr[int(lo_t):int(hi_t), int(lo_c):int(hi_c)])
        payload["t0"] = int(payload["t0"]) + int(lo_t)
        payload["c0"] = int(payload["c0"]) + int(lo_c)
        payload["t1"] = int(payload["t0"]) + int(payload["mask"].shape[0])
        payload["c1"] = int(payload["c0"]) + int(payload["mask"].shape[1])
        return payload

    def _clamp_mask_bounds(self, t0: int, t1: int, c0: int, c1: int) -> tuple[int, int, int, int]:
        nt, nc = self._source_shape_for_annotation()
        if nt is not None:
            t0 = max(0, min(int(t0), int(nt) - 1))
            t1 = max(t0 + 1, min(int(t1), int(nt)))
        if nc is not None:
            c0 = max(0, min(int(c0), int(nc) - 1))
            c1 = max(c0 + 1, min(int(c1), int(nc)))
        return int(t0), int(t1), int(c0), int(c1)

    def _source_shape_for_annotation(self) -> tuple[int | None, int | None]:
        source = getattr(self, "_source", None)
        shape = getattr(source, "shape", None)
        dims = str(getattr(source, "dims", "nt_nc") or "nt_nc")
        if shape is not None and len(shape) == 2:
            if dims == "nt_nc":
                return int(shape[0]), int(shape[1])
            return int(shape[1]), int(shape[0])
        return None, None

    def _hit_mask_payload(self, payload: dict, x: float, y: float) -> bool:
        try:
            t0 = int(round(float(payload.get("t0", 0))))
            c0 = int(round(float(payload.get("c0", 0))))
            t1 = int(round(float(payload.get("t1", t0))))
            c1 = int(round(float(payload.get("c1", c0))))
            if not (t0 <= x < t1 and c0 <= y < c1):
                return False
            arr = np.asarray(payload.get("mask"), dtype=np.uint8)
            ti = int(np.floor(float(x))) - t0
            ci = int(np.floor(float(y))) - c0
            return bool(0 <= ti < arr.shape[0] and 0 <= ci < arr.shape[1] and arr[ti, ci] > 0)
        except Exception:
            _print_exception("failed to hit-test mask annotation")
            return False

    def _push_mask_undo(self, uid: int, *, before: dict, after: dict) -> None:
        if self._mask_payload_equal(before, after):
            return
        self._mask_undo_stack.append({"uid": int(uid), "before": before, "after": after})
        self._mask_redo_stack.clear()
        if len(self._mask_undo_stack) > 64:
            self._mask_undo_stack = self._mask_undo_stack[-64:]

    def _undo_mask_brush(self) -> bool:
        if not self._mask_undo_stack:
            return False
        entry = self._mask_undo_stack.pop()
        if not self._apply_mask_history_payload(int(entry["uid"]), entry["before"]):
            return False
        self._mask_redo_stack.append(entry)
        return True

    def _redo_mask_brush(self) -> bool:
        if not self._mask_redo_stack:
            return False
        entry = self._mask_redo_stack.pop()
        if not self._apply_mask_history_payload(int(entry["uid"]), entry["after"]):
            return False
        self._mask_undo_stack.append(entry)
        return True

    def _apply_mask_history_payload(self, uid: int, payload: dict) -> bool:
        record = self._overlays.get(int(uid))
        if record is None or record.anno_type != "mask":
            return False
        patch = self._copy_mask_payload(payload)
        record.payload.update(patch)
        self.set_selected_annotation(uid, "mask", dict(record.payload))
        self._apply_overlay_record(record)
        self.annotation_geometry_edited.emit(uid, patch)
        return True

    def _mask_payload_equal(self, a: dict, b: dict) -> bool:
        if [int(a.get(k, 0)) for k in ("t0", "t1", "c0", "c1")] != [int(b.get(k, 0)) for k in ("t0", "t1", "c0", "c1")]:
            return False
        return np.array_equal(np.asarray(a.get("mask")), np.asarray(b.get("mask")))


class AnnotationInteractionMixin(AnnotationEditSessionMixin, AnnotationMaskSessionMixin, AnnotationEventMixin):
    """Own annotation drawing, editing, ROI preview, and canvas event handling."""

    def _init_annotation_state(self) -> None:
        options = dict(getattr(self, "_annotation_options", {}) or {})
        self._selected_annotation_uid: Optional[int] = None
        self._selected_annotation_type: Optional[str] = None
        self._selected_annotation_payload: Optional[dict] = None
        self.ann_enabled = bool(options.get("enabled", True))
        mode = str(options.get("mode", "bbox") or "bbox")
        self.ann_mode = mode if mode in ("bbox", "polyline", "mask") else "bbox"
        self.ann_label = str(options.get("label", "event") or "event").strip() or "event"
        self.ann_hold_ctrl = bool(options.get("hold_ctrl", True))
        self._mask_brush_radius = int(max(1, options.get("mask_brush_radius", 16) or 16))
        self._mask_brush_mode = "erase" if str(options.get("mask_brush_mode", "paint")) == "erase" else "paint"
        self._annotation_preview_color = str(options.get("preview_color", "#FFD54F") or "#FFD54F")
        self._annotation_preview_width = float(options.get("preview_width", 3.0) or 3.0)
        self._annotation_preview_cursor_width = float(options.get("preview_cursor_width", 2.0) or 2.0)
        self._annotation_bbox_width = float(max(1.0, options.get("bbox_width", 2.0) or 2.0))
        self._annotation_polyline_width = float(max(1.0, options.get("polyline_width", 3.0) or 3.0))
        self._annotation_marker_size = int(options.get("marker_size", 9) or 9)
        self._annotation_handle_size = int(options.get("handle_size", 8) or 8)
        self._annotation_hit_slop_t_fraction = float(options.get("hit_slop_t_fraction", 0.015) or 0.015)
        self._annotation_hit_slop_t_min = float(options.get("hit_slop_t_min", 4.0) or 4.0)
        self._annotation_hit_slop_c_fraction = float(options.get("hit_slop_c_fraction", 0.03) or 0.03)
        self._annotation_hit_slop_c_min = float(options.get("hit_slop_c_min", 1.5) or 1.5)
        self._ann_gate_active = False
        self._ctrl_down = False
        self._roi_enable = False
        self._drag_kind: Optional[str] = None
        self._drag_rect = None
        self._drag_p0 = None
        self._drag_p1 = None
        self._poly_points: list[tuple[float, float]] = []
        self._poly_build_target: Optional[dict] = None
        self._active_bbox_node: Optional[AnnotationBBoxNode] = None
        self._active_polyline_node: Optional[AnnotationPolylineNode] = None
        self._active_mask_node: Optional[MaskOverlayNode] = None
        self._bbox_edit_session: Optional[dict] = None
        self._poly_edit_session: Optional[dict] = None
        self._mask_paint_session: Optional[dict] = None
        self._mask_undo_stack: list[dict] = []
        self._mask_redo_stack: list[dict] = []
        self._mouse_press_interceptors: list[Callable[[Any], bool]] = []

    def _init_annotation_interactor(self) -> None:
        self._init_annotation_state()
        self._init_annotation_nodes()
        self._init_roi_preview()
        self._bbox_handler = _AnnotationHandlerProxy(self, kind="bbox")
        self._poly_handler = _AnnotationHandlerProxy(self, kind="polyline")
        self._mask_handler = _AnnotationHandlerProxy(self, kind="mask")
        self._edit_handler = _AnnotationHandlerProxy(self, kind="edit")
        self._roi_handler = _AnnotationHandlerProxy(self, kind="roi")
        self.register_handler("edit", self._edit_handler, priority=5)
        self.register_handler("bbox", self._bbox_handler, priority=10)
        self.register_handler("polyline", self._poly_handler, priority=10)
        self.register_handler("mask", self._mask_handler, priority=10)
        self.register_handler("roi", self._roi_handler, priority=11)

    def _init_annotation_nodes(self) -> None:
        self._active_bbox_node = AnnotationBBox(
            parent=self._world_ui_parent,
            color="yellow",
            border_width=self._annotation_bbox_width,
        )
        self._active_bbox_node.order = 40
        self._active_polyline_node = AnnotationPolyline(
            parent=self._world_ui_parent,
            color="yellow",
            width=self._annotation_polyline_width,
        )
        self._active_polyline_node.order = 40
        self._active_mask_node = MaskOverlay(
            parent=self._world_ui_parent,
            capacity_t=1,
            capacity_c=1,
            color=self._annotation_preview_color,
            alpha=0.45,
            threshold=0.0,
        )
        self._active_mask_node.order = 39
        self._active_mask_node.clear()

    def _init_roi_preview(self) -> None:
        self._roi_rect = scene.visuals.Rectangle(
            center=(0, 0),
            width=1,
            height=1,
            color=(0, 0, 0, 0.001),
            border_color="red",
            parent=self._world_ui_parent,
        )
        self._roi_rect.visible = False

    def _ensure_preview_visuals(self) -> None:
        self._drag_kind = None
        self._drag_rect = None
        self._drag_p0 = None
        self._drag_p1 = None
        self._poly_points = []
        self._poly_build_target = None
        self._bbox_edit_session = None
        self._poly_edit_session = None
        self._mask_paint_session = None

        if self._active_bbox_node is not None:
            self._active_bbox_node.close()
        if self._active_polyline_node is not None:
            self._active_polyline_node.close()
        if self._active_mask_node is not None:
            self._active_mask_node.close()
        if self._roi_rect is not None:
            try:
                self._roi_rect.parent = None
            except Exception:
                _print_exception("failed to detach ROI preview node")

        self._init_annotation_nodes()
        self._init_roi_preview()
        self._refresh_selection_visuals()

    def set_annotation_gate(self, active: bool) -> None:
        self._ann_gate_active = bool(active)
        if not self._ann_gate_active:
            self._cancel_bbox_drag()
            self._cancel_bbox_edit_session()
            self._cancel_polyline_session()
            self._cancel_poly_edit_session()
            self._cancel_mask_paint_session()
            self._annot_unlock_camera()
        self._refresh_active_handler()

    def set_analysis_roi_mode(self, enabled: bool) -> None:
        self._roi_enable = bool(enabled)
        if self._roi_enable:
            self._cancel_bbox_drag()
            self._cancel_bbox_edit_session()
            self._cancel_polyline_session()
            self._cancel_poly_edit_session()
            self._cancel_mask_paint_session()
            self._roi_rect.visible = False
            self._canvas.camera.interactive = False
        else:
            self._drag_kind = None
            self._roi_rect.visible = False
            self._canvas.camera.interactive = True
            self._canvas.update()
        self._refresh_active_handler()

    def set_mode(self, mode: str) -> None:
        if mode not in ("bbox", "polyline", "mask"):
            mode = "bbox"
        self._cancel_bbox_drag()
        self._cancel_bbox_edit_session()
        self._cancel_polyline_session()
        self._cancel_poly_edit_session()
        self._cancel_mask_paint_session()
        self.ann_mode = mode
        self._refresh_active_handler()

    def set_label(self, label: str) -> None:
        self.ann_label = (label or "event").strip() or "event"

    def set_annotation_enabled(self, enabled: bool) -> None:
        self.ann_enabled = bool(enabled)
        if not self.ann_enabled:
            self._cancel_bbox_drag()
            self._cancel_bbox_edit_session()
            self._cancel_polyline_session()
            self._cancel_poly_edit_session()
            self._cancel_mask_paint_session()
            self._annot_unlock_camera()
        self._refresh_active_handler()

    def set_annotation_hold_ctrl(self, enabled: bool) -> None:
        self.ann_hold_ctrl = bool(enabled)

    def set_annotation_state(
        self,
        *,
        mode: str,
        label: str,
        enabled: bool = True,
        attrs_json: Optional[str] = None,
    ) -> None:
        self.set_annotation_enabled(enabled)
        self.set_mode(mode)
        self.set_label(label)

    def set_selected_annotation(self, uid: Optional[int], anno_type: Optional[str] = None, data: Optional[dict] = None) -> None:
        if (
            self._selected_annotation_uid != (None if uid is None else int(uid))
            or self._selected_annotation_type != (None if uid is None else anno_type)
        ):
            self._cancel_bbox_edit_session()
            self._cancel_poly_edit_session()
            self._cancel_mask_paint_session()
        self._selected_annotation_uid = None if uid is None else int(uid)
        self._selected_annotation_type = None if uid is None else anno_type
        self._selected_annotation_payload = None if uid is None else dict(data or {})
        refresh_virtual = getattr(self, "_refresh_annotation_virtual_overlays", None)
        if callable(refresh_virtual):
            refresh_virtual()
        self._refresh_selection_visuals()

    def set_roi_region(self, roi) -> None:
        try:
            self._update_rect(self._roi_rect, (roi.t0, roi.c0), (roi.t1, roi.c1))
            self._roi_rect.visible = True
            self._canvas.update()
        except Exception:
            _print_exception("failed to update ROI preview")

    def _hit_slop(self) -> tuple[float, float]:
        try:
            rect = self._canvas.camera.rect
            return (
                max(float(rect.width) * self._annotation_hit_slop_t_fraction, self._annotation_hit_slop_t_min),
                max(float(rect.height) * self._annotation_hit_slop_c_fraction, self._annotation_hit_slop_c_min),
            )
        except Exception:
            _print_exception("failed to compute hit slop")
            return 8.0, 3.0

    def _refresh_selection_visuals(self) -> None:
        if self._active_bbox_node is None or self._active_polyline_node is None:
            return

        bbox_payload = None
        poly_points = None

        if self._selected_annotation_uid is not None:
            if self._bbox_edit_session is not None and self._bbox_edit_session.get("uid") == self._selected_annotation_uid:
                bbox_payload = dict(self._bbox_edit_session["current_bbox"])
            elif self._poly_edit_session is not None and self._poly_edit_session.get("uid") == self._selected_annotation_uid:
                poly_points = list(self._poly_edit_session["current_points"])
            elif (
                self._poly_build_target is not None
                and self._poly_build_target.get("uid") == self._selected_annotation_uid
                and self._poly_points
            ):
                poly_points = list(self._poly_points)
            else:
                payload = self._selected_payload()
                if payload is not None and self._annotation_visible(payload):
                    if (self._selected_annotation_type or "") == "bbox":
                        bbox_payload = self._normalized_bbox(payload)
                    elif (self._selected_annotation_type or "") == "polyline":
                        poly_points = list(payload.get("points", []) or [])

        if bbox_payload is not None:
            payload = self._selected_payload()
            color = self._annotation_color(payload, self._annotation_preview_color)
            width = self._annotation_bbox_width
            if payload is not None:
                width = float(payload.get("width", payload.get("border_width", width)))
            self._active_bbox_node.show(
                bbox_payload,
                color=color,
                border_width=width,
                visible=True,
                handles=True,
                handle_face="#FFFFFF",
                handle_edge=color,
                handle_size=self._annotation_handle_size,
            )
        else:
            self._active_bbox_node.clear()

        if poly_points:
            payload = self._selected_payload()
            color = self._annotation_color(payload, self._annotation_preview_color)
            width = self._annotation_polyline_width
            if payload is not None:
                width = float(payload.get("width", width))
            self._active_polyline_node.show(
                poly_points,
                color=color,
                width=width,
                visible=True,
                show_markers=True,
                marker_face=color,
                marker_edge="#FFFFFF",
                marker_size=self._annotation_marker_size,
                cursor_segment=None,
                cursor_color=color,
                cursor_width=self._annotation_preview_cursor_width,
            )
        else:
            self._active_polyline_node.clear()

        self._canvas.update()

    def _hit_selected_bbox_mode(self, x: float, y: float) -> Optional[str]:
        if self._selected_annotation_type != "bbox":
            return None
        payload = self._selected_payload()
        if payload is None or not self._annotation_visible(payload):
            return None
        return self._hit_bbox_mode_for_payload(payload, x, y)

    def _hit_bbox_mode_for_payload(self, payload: dict, x: float, y: float) -> Optional[str]:
        bbox = self._normalized_bbox(payload)
        xtol, ytol = self._hit_slop()

        left = abs(x - bbox["t0"]) <= xtol
        right = abs(x - bbox["t1"]) <= xtol
        top = abs(y - bbox["c0"]) <= ytol
        bottom = abs(y - bbox["c1"]) <= ytol
        inside_x = bbox["t0"] <= x <= bbox["t1"]
        inside_y = bbox["c0"] <= y <= bbox["c1"]

        if left and top:
            return "top_left"
        if right and top:
            return "top_right"
        if left and bottom:
            return "bottom_left"
        if right and bottom:
            return "bottom_right"
        if left and (bbox["c0"] - ytol) <= y <= (bbox["c1"] + ytol):
            return "left"
        if right and (bbox["c0"] - ytol) <= y <= (bbox["c1"] + ytol):
            return "right"
        if top and (bbox["t0"] - xtol) <= x <= (bbox["t1"] + xtol):
            return "top"
        if bottom and (bbox["t0"] - xtol) <= x <= (bbox["t1"] + xtol):
            return "bottom"
        if inside_x and inside_y:
            return "move"
        return None

    def _hit_selected_poly_vertex(self, x: float, y: float) -> Optional[int]:
        if self._selected_annotation_type != "polyline":
            return None
        payload = self._selected_payload()
        if payload is None or not self._annotation_visible(payload):
            return None
        return self._hit_poly_vertex_in_points(list(payload.get("points", []) or []), x, y)

    def _hit_poly_vertex_in_points(self, points: list[tuple[float, float]], x: float, y: float) -> Optional[int]:
        if not points:
            return None
        xtol, ytol = self._hit_slop()
        best_index = None
        best_score = None
        for index, (px, py) in enumerate(points):
            if abs(x - px) > xtol or abs(y - py) > ytol:
                continue
            score = abs(x - px) + abs(y - py)
            if best_score is None or score < best_score:
                best_score = score
                best_index = index
        return best_index

    def _hit_poly_segment_in_points(self, points: list[tuple[float, float]], x: float, y: float) -> bool:
        return self._hit_poly_segment_index(points, x, y) is not None

    def _hit_poly_segment_index(self, points: list[tuple[float, float]], x: float, y: float) -> Optional[int]:
        if len(points) < 2:
            return None
        xtol, ytol = self._hit_slop()
        tol2 = float(max(xtol, ytol) ** 2)
        px = float(x)
        py = float(y)
        best_index = None
        best_distance = None
        for index, ((x0, y0), (x1, y1)) in enumerate(zip(points[:-1], points[1:])):
            vx = float(x1) - float(x0)
            vy = float(y1) - float(y0)
            seg2 = vx * vx + vy * vy
            if seg2 <= 1e-12:
                dx = px - float(x0)
                dy = py - float(y0)
                dist2 = dx * dx + dy * dy
                if dist2 <= tol2 and (best_distance is None or dist2 < best_distance):
                    best_index = index
                    best_distance = dist2
                continue
            t = ((px - float(x0)) * vx + (py - float(y0)) * vy) / seg2
            t = min(1.0, max(0.0, t))
            qx = float(x0) + t * vx
            qy = float(y0) + t * vy
            dx = px - qx
            dy = py - qy
            dist2 = dx * dx + dy * dy
            if dist2 <= tol2 and (best_distance is None or dist2 < best_distance):
                best_index = index
                best_distance = dist2
        return best_index

    def _overlay_hit_candidates(self) -> list[OverlayRecord]:
        records = list(self._overlays.values())
        if self._selected_annotation_uid is None:
            return list(reversed(records))
        selected = self._overlays.get(int(self._selected_annotation_uid))
        ordered = [record for record in reversed(records) if record.uid != self._selected_annotation_uid]
        if selected is not None:
            ordered.insert(0, selected)
        return ordered

    def _pick_overlay_hit(self, x: float, y: float) -> Optional[dict]:
        for record in self._overlay_hit_candidates():
            if not record.selectable:
                continue
            payload = dict(record.payload)
            if not self._annotation_visible(payload):
                continue
            if record.anno_type == "bbox":
                bbox_mode = self._hit_bbox_mode_for_payload(payload, x, y)
                if bbox_mode is None:
                    continue
                return {
                    "uid": record.uid,
                    "anno_type": "bbox",
                    "payload": payload,
                    "bbox_mode": bbox_mode,
                    "vertex_index": None,
                    "segment_hit": True,
                }
            if record.anno_type == "polyline":
                points = [tuple(point) for point in (payload.get("points", []) or [])]
                vertex_index = self._hit_poly_vertex_in_points(points, x, y)
                segment_index = self._hit_poly_segment_index(points, x, y)
                if vertex_index is None and segment_index is None:
                    continue
                return {
                    "uid": record.uid,
                    "anno_type": "polyline",
                    "payload": payload,
                    "bbox_mode": None,
                    "vertex_index": vertex_index,
                    "segment_index": segment_index,
                }
            if record.anno_type == "mask":
                if not self._hit_mask_payload(payload, x, y):
                    continue
                return {
                    "uid": record.uid,
                    "anno_type": "mask",
                    "payload": payload,
                    "bbox_mode": None,
                    "vertex_index": None,
                    "segment_index": None,
                }
        return None

    def _annotation_creation_in_progress(self) -> bool:
        return self._drag_kind == "ann_bbox" or bool(self._poly_points) or self._mask_paint_session is not None

    def _select_overlay_hit(self, hit: dict) -> None:
        uid = int(hit["uid"])
        record = self._overlays.get(uid)
        if record is None or not record.selectable:
            return
        anno_type = str(hit["anno_type"])
        payload = dict(hit["payload"])
        changed = (
            self._selected_annotation_uid != uid
            or self._selected_annotation_type != anno_type
        )
        self.set_selected_annotation(uid, anno_type, payload)
        if changed:
            self.annotation_selected.emit(uid)

    def _clear_annotation_selection(self) -> None:
        if self._selected_annotation_uid is None:
            return
        self.set_selected_annotation(None)
        self.annotation_selected.emit(None)

    def continue_selected_polyline(self, side: str) -> None:
        if side not in ("start", "end"):
            return
        record = self._selected_record()
        payload = self._selected_payload()
        if record is None or payload is None or record.anno_type != "polyline":
            return
        points = [tuple(map(float, point)) for point in (payload.get("points", []) or [])]
        if len(points) < 2:
            return
        self._cancel_polyline_session()
        self._poly_build_target = {"uid": record.uid, "side": side}
        self._poly_points = list(points)
        record.node.set_visible(False)
        self._annot_lock_camera()
        self._update_poly_preview(mouse_xy=None)
        try:
            self._canvas.native.setFocus()
        except Exception:
            _print_exception("failed to focus canvas while continuing selected polyline")
