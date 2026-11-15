"""Canvas event handlers for annotation interaction."""

from __future__ import annotations

from PySide6.QtCore import Qt

from filark._exceptions import print_exception


def _print_exception(context: str) -> None:
    print_exception(context, prefix="DisplayModel")

class AnnotationEventCoreMixin:
    """Own canvas event binding, modifier helpers, and shared camera/drag helpers."""

    def _bind_canvas_events(self) -> None:
        self._canvas.events.mouse_press.connect(self._on_mouse_press_global)
        self._canvas.events.mouse_move.connect(self._on_mouse_move_global)
        self._canvas.events.mouse_release.connect(self._on_mouse_release_global)
        self._canvas.events.key_press.connect(self._on_key_press_global)
        self._canvas.events.key_release.connect(self._on_key_release_global)

    def _on_mouse_press_global(self, ev) -> None:
        if self._ann_gate_active:
            try:
                self._canvas.native.setFocus()
            except Exception:
                _print_exception("failed to focus canvas on mouse press")
        if self._run_mouse_press_interceptors(ev):
            return
        self._dispatch("on_mouse_press", ev)

    def _on_mouse_move_global(self, ev) -> None:
        if self.ann_hold_ctrl and (not self.is_ctrl_pressed(ev)):
            self._on_ctrl_released_during_mouse(ev)
        self._dispatch("on_mouse_move", ev)

    def _on_mouse_release_global(self, ev) -> None:
        self._dispatch("on_mouse_release", ev)
        if self.ann_hold_ctrl and (not self.is_ctrl_pressed(ev)):
            self._on_ctrl_released_during_mouse(ev)

    def _on_key_press_global(self, ev) -> None:
        if self._is_ctrl_key(ev):
            self._ctrl_down = True
        self._dispatch("on_key_press", ev)
        key_name = getattr(getattr(ev, "key", None), "name", None)
        key_text = str(key_name if key_name is not None else getattr(ev, "key", "")).upper()
        if key_text in ("BACKSPACE", "DELETE") and not getattr(ev, "handled", False):
            ev.handled = True
            return
        if key_text in ("X", "Y", "F"):
            self._emit_scale_state_changed()

    def _on_key_release_global(self, ev) -> None:
        if self._is_ctrl_key(ev):
            self._ctrl_down = False
            self._on_ctrl_released_during_key(ev)
        self._dispatch("on_key_release", ev)

    def _is_ctrl_key(self, ev) -> bool:
        key = getattr(getattr(ev, "key", None), "name", None) or str(getattr(ev, "key", ""))
        key = str(key).upper()
        return ("CTRL" in key) or ("CONTROL" in key)

    def is_ctrl_pressed(self, ev) -> bool:
        modifiers = getattr(ev, "modifiers", None)
        if modifiers is not None:
            modifiers = modifiers or []
            if ("Control" in modifiers) or ("CTRL" in modifiers):
                return True
        native = getattr(ev, "native", None)
        if native is not None:
            try:
                return bool(native.modifiers() & Qt.ControlModifier)
            except Exception:
                _print_exception("failed to read native Ctrl modifier")
        return False

    def is_ctrl_shortcut(self, ev) -> bool:
        modifiers = getattr(ev, "modifiers", None) or []
        if ("Control" in modifiers) or ("CTRL" in modifiers):
            return True
        return bool(self._ctrl_down)

    def is_shift_pressed(self, ev) -> bool:
        modifiers = getattr(ev, "modifiers", None)
        if modifiers is not None:
            modifiers = modifiers or []
            if ("Shift" in modifiers) or ("SHIFT" in modifiers):
                return True
        native = getattr(ev, "native", None)
        if native is not None:
            try:
                return bool(native.modifiers() & Qt.ShiftModifier)
            except Exception:
                _print_exception("failed to read native Shift modifier")
        return False

    def is_alt_pressed(self, ev) -> bool:
        modifiers = getattr(ev, "modifiers", None)
        if modifiers is not None:
            modifiers = modifiers or []
            if ("Alt" in modifiers) or ("ALT" in modifiers) or ("Option" in modifiers) or ("OPTION" in modifiers):
                return True
        native = getattr(ev, "native", None)
        if native is not None:
            try:
                return bool(native.modifiers() & Qt.AltModifier)
            except Exception:
                _print_exception("failed to read native Alt/Option modifier")
        return False

    def _annotation_allowed_now(self) -> bool:
        if not self._ann_gate_active:
            return False
        if not self.ann_enabled:
            return False
        if self._roi_enable:
            return False
        return self.ann_mode in ("bbox", "polyline", "mask")

    def _annotation_edit_allowed_now(self) -> bool:
        if not self._ann_gate_active:
            return False
        if not self.ann_enabled:
            return False
        if self._roi_enable:
            return False
        if self._annotation_creation_in_progress():
            return False
        return True

    def _refresh_active_handler(self) -> None:
        if self._roi_enable:
            self.set_active_handler("roi")
            return
        if not self._ann_gate_active or not self.ann_enabled:
            self.set_active_handler(None)
            return
        self.set_active_handler(None)

    def _annot_lock_camera(self) -> None:
        if not self._roi_enable:
            self._canvas.camera.interactive = False

    def _annot_unlock_camera(self) -> None:
        if not self._roi_enable:
            self._canvas.camera.interactive = True

    def _begin_drag_rect(self, kind: str, rect_node, ev) -> None:
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

        kind = self._drag_kind
        geom = {"t0": self._drag_p0[0], "c0": self._drag_p0[1], "t1": self._drag_p1[0], "c1": self._drag_p1[1]}
        rect = self._drag_rect

        self._drag_kind = None
        self._drag_rect = None
        self._drag_p0 = None
        self._drag_p1 = None

        rect.visible = False
        self._canvas.update()
        return kind, geom

    def _on_ctrl_released_during_mouse(self, ev) -> None:
        if not self._ann_gate_active or self._roi_enable:
            return
        if self._drag_kind == "ann_bbox":
            self._cancel_bbox_drag()
            self._annot_unlock_camera()
        if self._mask_paint_session is not None:
            self._cancel_mask_paint_session()
        if not self._poly_points:
            self._annot_unlock_camera()

    def _on_ctrl_released_during_key(self, ev) -> None:
        if not self._ann_gate_active or self._roi_enable:
            return
        if self._drag_kind == "ann_bbox":
            self._cancel_bbox_drag()
            self._annot_unlock_camera()
        if self._mask_paint_session is not None:
            self._cancel_mask_paint_session()
        if not self._poly_points:
            self._annot_unlock_camera()

class AnnotationBBoxEventMixin:
    """Own bbox creation and bbox drag lifecycle handlers."""

    def _cancel_bbox_drag(self) -> None:
        if self._drag_kind == "ann_bbox":
            if self._active_bbox_node is not None:
                self._active_bbox_node.clear()
            self._drag_kind = None
            self._drag_rect = None
            self._drag_p0 = None
            self._drag_p1 = None
            if self._selected_annotation_uid is not None:
                self._refresh_selection_visuals()
            self._canvas.update()

    def on_mouse_press_bbox(self, ev) -> None:
        if getattr(ev, "button", None) != 1:
            return
        if not self._annotation_allowed_now() or self.ann_mode != "bbox":
            return

        if self.ann_hold_ctrl and (not self.is_ctrl_pressed(ev)):
            return

        self._annot_lock_camera()
        x, y = self.canvas_pos_to_world(ev.pos)
        self._drag_kind = "ann_bbox"
        self._drag_rect = None
        self._drag_p0 = (x, y)
        self._drag_p1 = (x, y)
        if self._active_bbox_node is not None:
            self._active_bbox_node.show(
                {"t0": x, "c0": y, "t1": x, "c1": y},
                color=self._annotation_preview_color,
                border_width=self._annotation_bbox_width,
                visible=True,
                handles=False,
            )
            self._canvas.update()
        ev.handled = True

    def on_mouse_move_bbox(self, ev) -> None:
        if self._bbox_edit_session is not None:
            x, y = self.canvas_pos_to_world(ev.pos)
            bbox = self._bbox_from_edit_session(x, y)
            if bbox is not None:
                self._preview_bbox_edit(bbox)
                ev.handled = True
            return

        if self._drag_kind != "ann_bbox":
            return

        if self.ann_hold_ctrl and (not self.is_ctrl_pressed(ev)):
            self._cancel_bbox_drag()
            self._annot_unlock_camera()
            ev.handled = True
            return

        x, y = self.canvas_pos_to_world(ev.pos)
        self._drag_p1 = (x, y)
        if self._active_bbox_node is not None:
            self._active_bbox_node.show(
                {"t0": self._drag_p0[0], "c0": self._drag_p0[1], "t1": x, "c1": y},
                color=self._annotation_preview_color,
                border_width=self._annotation_bbox_width,
                visible=True,
                handles=False,
            )
            self._canvas.update()
        ev.handled = True

    def on_mouse_release_bbox(self, ev) -> None:
        if self._bbox_edit_session is not None:
            if getattr(ev, "button", None) != 1:
                return
            finished = self._finish_bbox_edit_session()
            if finished is not None:
                uid, patch = finished
                self.annotation_geometry_edited.emit(uid, patch)
            ev.handled = True
            return

        if self._drag_kind != "ann_bbox" or getattr(ev, "button", None) != 1:
            return

        if self.ann_hold_ctrl and (not self.is_ctrl_pressed(ev)):
            self._cancel_bbox_drag()
            self._annot_unlock_camera()
            ev.handled = True
            return

        x, y = self.canvas_pos_to_world(ev.pos)
        geom = {"t0": self._drag_p0[0], "c0": self._drag_p0[1], "t1": x, "c1": y}
        self._drag_kind = None
        self._drag_rect = None
        self._drag_p0 = None
        self._drag_p1 = None
        if self._active_bbox_node is not None:
            self._active_bbox_node.clear()
        if self._selected_annotation_uid is not None:
            self._refresh_selection_visuals()
        self._annot_unlock_camera()
        modifiers = getattr(ev, "modifiers", None)
        geom["modifiers"] = list(modifiers) if modifiers else []
        self.bbox_finished.emit(geom)
        ev.handled = True

class AnnotationPolylineEventMixin:
    """Own polyline creation, continuation, preview, and key handlers."""

    def _update_poly_preview(self, mouse_xy=None) -> None:
        if self._active_polyline_node is None:
            return
        cursor_segment = None
        if mouse_xy is not None and self._poly_points:
            side = "end"
            if self._poly_build_target is not None:
                side = str(self._poly_build_target.get("side", "end"))
            anchor = self._poly_points[0] if side == "start" else self._poly_points[-1]
            cursor_segment = [mouse_xy, anchor] if side == "start" else [anchor, mouse_xy]
        self._active_polyline_node.show(
            self._poly_points,
            color=self._annotation_preview_color,
            width=self._annotation_polyline_width,
            visible=bool(self._poly_points),
            show_markers=bool(self._poly_points),
            marker_face=self._annotation_preview_color,
            marker_edge="#FFFFFF",
            marker_size=self._annotation_marker_size,
            cursor_segment=cursor_segment,
            cursor_color=self._annotation_preview_color,
            cursor_width=self._annotation_preview_cursor_width,
        )
        self._canvas.update()

    def _cancel_polyline_session(self) -> None:
        target = dict(self._poly_build_target or {})
        uid = target.get("uid")
        self._poly_points = []
        self._poly_build_target = None
        if self._active_polyline_node is not None:
            self._active_polyline_node.clear()
        if uid is not None:
            record = self._overlays.get(int(uid))
            if record is not None:
                record.node.set_visible(self._annotation_visible(record.payload))
        if self._selected_annotation_uid is not None:
            self._refresh_selection_visuals()
        self._annot_unlock_camera()

    def _undo_polyline_point(self) -> None:
        if not self._poly_points:
            return
        self._poly_points.pop()
        if self._poly_points:
            self._update_poly_preview(mouse_xy=None)
        else:
            self._cancel_polyline_session()

    def _finish_polyline(self) -> None:
        if len(self._poly_points) < 2:
            self._cancel_polyline_session()
            return
        points = list(self._poly_points)
        target = dict(self._poly_build_target or {})
        uid = target.get("uid")
        if uid is None:
            self.polyline_finished.emit({"points": points})
        else:
            self.annotation_geometry_edited.emit(int(uid), {"points": points})
        self._cancel_polyline_session()

    def on_mouse_press_polyline(self, ev) -> None:
        if getattr(ev, "button", None) not in (1, 2):
            return
        if not self._annotation_allowed_now() or self.ann_mode != "polyline":
            return

        button = getattr(ev, "button", None)
        if button == 1:
            if not self._poly_points:
                if self.ann_hold_ctrl and (not self.is_ctrl_pressed(ev)):
                    return
                self._poly_build_target = {"uid": None, "side": "end"}
                self._annot_lock_camera()
                try:
                    self._canvas.native.setFocus()
                except Exception:
                    _print_exception("failed to focus canvas while starting polyline")
            x, y = self.canvas_pos_to_world(ev.pos)
            side = "end"
            if self._poly_build_target is not None:
                side = str(self._poly_build_target.get("side", "end"))
            if side == "start":
                self._poly_points.insert(0, (x, y))
            else:
                self._poly_points.append((x, y))
            self._update_poly_preview(mouse_xy=None)
            ev.handled = True
            return

        if button == 2:
            self._finish_polyline()
            ev.handled = True

    def on_mouse_move_polyline(self, ev) -> None:
        if self._poly_edit_session is not None:
            x, y = self.canvas_pos_to_world(ev.pos)
            points = list(self._poly_edit_session["current_points"])
            points[self._poly_edit_session["index"]] = (x, y)
            self._preview_poly_edit(points)
            ev.handled = True
            return

        if not self._poly_points:
            return
        x, y = self.canvas_pos_to_world(ev.pos)
        self._update_poly_preview(mouse_xy=(x, y))
        ev.handled = True

    def on_mouse_release_polyline(self, ev) -> None:
        if self._poly_edit_session is None:
            return
        if getattr(ev, "button", None) != 1:
            return
        finished = self._finish_poly_edit_session()
        if finished is not None:
            uid, patch = finished
            self.annotation_geometry_edited.emit(uid, patch)
        ev.handled = True

    def on_key_press_polyline(self, ev) -> None:
        key = getattr(getattr(ev, "key", None), "name", None) or str(getattr(ev, "key", ""))
        key = str(key).upper()

        if key in ("BACKSPACE", "DELETE"):
            ev.handled = True
            return

        if self._poly_edit_session is not None and key in ("ESCAPE", "ESC"):
            self._cancel_poly_edit_session()
            ev.handled = True
            return

        if not self._poly_points:
            return

        if key == "Z" and self.is_ctrl_shortcut(ev):
            self._undo_polyline_point()
            ev.handled = True
            return

        if key in ("ENTER", "RETURN"):
            self._finish_polyline()
            ev.handled = True
            return

        if key in ("ESCAPE", "ESC"):
            self._cancel_polyline_session()
            ev.handled = True


class AnnotationMaskEventMixin:
    """Own mask brush paint, erase, and brush history handlers."""

    def on_mouse_press_mask(self, ev) -> None:
        if getattr(ev, "button", None) != 1:
            return
        if not self._annotation_allowed_now() or self.ann_mode != "mask":
            return
        if self.ann_hold_ctrl and (not self.is_ctrl_pressed(ev)):
            return
        if self._begin_mask_paint_session(ev):
            ev.handled = True

    def on_mouse_move_mask(self, ev) -> None:
        if self._mask_paint_session is None:
            return
        if hasattr(ev, "buttons") and (1 not in ev.buttons):
            return
        if self._paint_mask_session_event(ev):
            ev.handled = True

    def on_mouse_release_mask(self, ev) -> None:
        if self._mask_paint_session is None or getattr(ev, "button", None) != 1:
            return
        finished = self._finish_mask_paint_session()
        if finished is not None:
            uid, patch = finished
            if uid is None:
                self.mask_finished.emit(patch)
            else:
                self.annotation_geometry_edited.emit(uid, patch)
        ev.handled = True

    def on_key_press_mask(self, ev) -> None:
        key = getattr(getattr(ev, "key", None), "name", None) or str(getattr(ev, "key", ""))
        key = str(key).upper()

        if self._mask_paint_session is not None and key in ("ESCAPE", "ESC"):
            self._cancel_mask_paint_session()
            ev.handled = True
            return

        if key == "Z" and self.is_ctrl_shortcut(ev):
            ev.handled = self._undo_mask_brush()
            return

        if key == "Y" and self.is_ctrl_shortcut(ev):
            ev.handled = self._redo_mask_brush()


class AnnotationEditEventMixin:
    """Own committed bbox/polyline selection and edit mouse/key handlers."""

    def on_mouse_press_edit(self, ev) -> None:
        if getattr(ev, "button", None) != 1:
            return
        if not self._annotation_edit_allowed_now():
            return
        if self.ann_hold_ctrl and self.is_ctrl_pressed(ev):
            return

        x, y = self.canvas_pos_to_world(ev.pos)
        hit = self._pick_overlay_hit(x, y)
        if hit is None:
            self._clear_annotation_selection()
            ev.handled = True
            return

        self._select_overlay_hit(hit)
        if hit["anno_type"] == "bbox" and hit["bbox_mode"] is not None:
            record = self._overlays.get(int(hit["uid"]))
            if record is None or not record.editable:
                ev.handled = True
                return
            self._begin_bbox_edit(ev, str(hit["bbox_mode"]))
            ev.handled = True
            return
        if hit["anno_type"] == "polyline":
            record = self._overlays.get(int(hit["uid"]))
            if record is None or not record.editable:
                ev.handled = True
                return
            payload = dict(hit["payload"])
            points = [tuple(map(float, point)) for point in (payload.get("points", []) or [])]
            uid = int(hit["uid"])
            vertex_index = hit.get("vertex_index")
            segment_index = hit.get("segment_index")
            if vertex_index is not None and self.is_alt_pressed(ev):
                self._delete_polyline_vertex(uid, points, int(vertex_index))
                ev.handled = True
                return
            if segment_index is not None and self.is_shift_pressed(ev):
                insert_index = int(segment_index) + 1
                new_points = list(points)
                new_points.insert(insert_index, (x, y))
                record = self._overlays.get(uid)
                if record is not None:
                    record.payload["points"] = list(new_points)
                self.set_selected_annotation(uid, "polyline", {"points": new_points})
                self.annotation_geometry_edited.emit(uid, {"points": new_points})
                self._begin_poly_edit(ev, insert_index)
                ev.handled = True
                return
            if vertex_index is not None:
                self._begin_poly_edit(ev, int(vertex_index))
            ev.handled = True
            return

        if hit["anno_type"] == "mask":
            ev.handled = True

    def on_mouse_move_edit(self, ev) -> None:
        if self._bbox_edit_session is not None:
            x, y = self.canvas_pos_to_world(ev.pos)
            bbox = self._bbox_from_edit_session(x, y)
            if bbox is not None:
                self._preview_bbox_edit(bbox)
                ev.handled = True
            return

        if self._poly_edit_session is not None:
            x, y = self.canvas_pos_to_world(ev.pos)
            points = list(self._poly_edit_session["current_points"])
            points[self._poly_edit_session["index"]] = (x, y)
            self._preview_poly_edit(points)
            ev.handled = True

    def on_mouse_release_edit(self, ev) -> None:
        if getattr(ev, "button", None) != 1:
            return

        if self._bbox_edit_session is not None:
            finished = self._finish_bbox_edit_session()
            if finished is not None:
                uid, patch = finished
                self.annotation_geometry_edited.emit(uid, patch)
            ev.handled = True
            return

        if self._poly_edit_session is not None:
            finished = self._finish_poly_edit_session()
            if finished is not None:
                uid, patch = finished
                self.annotation_geometry_edited.emit(uid, patch)
            ev.handled = True

    def on_key_press_edit(self, ev) -> None:
        key = getattr(getattr(ev, "key", None), "name", None) or str(getattr(ev, "key", ""))
        key = str(key).upper()
        if key not in ("ESCAPE", "ESC"):
            return
        if self._bbox_edit_session is not None:
            self._cancel_bbox_edit_session()
            ev.handled = True
            return
        if self._poly_edit_session is not None:
            self._cancel_poly_edit_session()
            ev.handled = True

class AnnotationROIEventMixin:
    """Own ROI drag handlers."""

    def on_mouse_press_roi(self, ev) -> None:
        if not self._roi_enable or getattr(ev, "button", None) != 1:
            return
        self._begin_drag_rect("roi", self._roi_rect, ev)
        ev.handled = True

    def on_mouse_move_roi(self, ev) -> None:
        if self._drag_kind == "roi" and self._move_drag_rect(ev):
            ev.handled = True

    def on_mouse_release_roi(self, ev) -> None:
        if self._drag_kind != "roi" or getattr(ev, "button", None) != 1:
            return
        ended = self._end_drag_rect(ev)
        if ended is None:
            return
        _, geom = ended
        self.roi_committed.emit(geom)
        ev.handled = True

class AnnotationEventMixin(
    AnnotationEventCoreMixin,
    AnnotationBBoxEventMixin,
    AnnotationPolylineEventMixin,
    AnnotationMaskEventMixin,
    AnnotationEditEventMixin,
    AnnotationROIEventMixin,
):
    """Compose annotation canvas event handlers by interaction mode."""
