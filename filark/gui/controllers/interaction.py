"""Annotation and ROI interaction controllers for the main Qt window."""

from __future__ import annotations

from PySide6.QtWidgets import QMessageBox

from ..roi import ROIWindow


class AnnotationMixin:
    """Bridge annotation panel events to DisplayModel overlays."""

    def _on_annot_intent_changed(self, intent: dict):
        mode = intent.get("mode", "bbox")
        label = intent.get("preset_label", "event")
        self.disp_model.set_mode(mode)
        self.disp_model.set_label(label)
        self.disp_model.set_mask_brush(
            radius=int(intent.get("mask_brush_radius", 16) or 16),
            mode=str(intent.get("mask_brush_mode", "paint") or "paint"),
        )

    def _on_annot_selection_changed(self, uid, recenter: bool = True):
        if uid is None:
            self.disp_model.set_selected_annotation(None)
            return
        item = self.ann_panel.store.get(uid)
        if item is None:
            self.disp_model.set_selected_annotation(None)
            return
        self.disp_model.set_selected_annotation(uid, item.anno_type, self.ann_panel.overlay_payload_for_item(item))
        if not recenter:
            return
        if item.anno_type == "bbox":
            t0_ann = float(item.data.get("t0", 0))
            t1_ann = float(item.data.get("t1", 0))
            t_center = (t0_ann + t1_ann) / 2.0
        elif item.anno_type == "polyline":
            points = item.data.get("points", []) or []
            if not points:
                return
            ts = [point[0] for point in points]
            t0_ann, t1_ann = min(ts), max(ts)
            t_center = (t0_ann + t1_ann) / 2.0
        elif item.anno_type == "mask":
            t0_ann = float(item.data.get("t0", 0))
            t1_ann = float(item.data.get("t1", 0))
            t_center = (t0_ann + t1_ann) / 2.0
        else:
            return
        bounds = self.disp_model.current_view_bounds()
        if bounds is not None:
            view_t0, view_t1 = float(bounds["t0"]), float(bounds["t1"])
            if view_t0 <= t0_ann and t1_ann <= view_t1:
                return
        self.disp_model.goto_sample(int(round(t_center)), center=True)

    def _on_ann_overlay_add(self, uid: int, anno_type: str, data: dict):
        if anno_type == "bbox":
            self.disp_model.add_bbox_overlay(uid, data)
        elif anno_type == "polyline":
            self.disp_model.add_polyline_overlay(uid, data)
        elif anno_type == "mask":
            self.disp_model.add_mask_overlay(uid, data, source_kind="annotation", selectable=True, editable=True)

    def _on_ann_overlay_add_many(self, records: list):
        add_many = getattr(self.disp_model, "add_annotation_overlays", None)
        if callable(add_many):
            add_many(records)
            return
        for uid, anno_type, data in records:
            self._on_ann_overlay_add(uid, anno_type, data)

    def _on_ann_overlay_update(self, uid: int, patch: dict):
        self.disp_model.update_overlay(uid, patch)

    def _on_ann_overlay_delete(self, uid: int):
        self.disp_model.remove_overlay(uid)

    def _on_ann_overlay_clear(self):
        self.disp_model.clear_overlays_by_kind("annotation")


class ROIControllerMixin:
    """Own ROI mode, ROI data extraction, and ROI analysis windows."""

    def _on_roi_mode(self, enabled: bool):
        enabled = bool(enabled)

        self.ann_panel.suspend(enabled)
        self.disp_model.set_annotation_enabled(not enabled)
        self.disp_model.set_analysis_roi_mode(enabled)

        if enabled:
            self.disp_model.set_annotation_gate(False)
        else:
            self._sync_annotation_gate()

    def _on_roi_committed(self, roi_dict: dict):
        self.roi_panel.set_roi(**roi_dict, commit=True)

    def _on_roi_annalysis(self, mode, roi):
        roi_data = self._extract_roi_data(roi)
        if roi_data["data"].size == 0:
            return

        try:
            win = ROIWindow(
                roi_data=roi_data,
                view_type=mode,
                source=self._source
                if getattr(getattr(self, "axis_metadata", None), "is_default_time_channel", True)
                else None,
                processors=self.cfg.processor_list(),
            )
        except ValueError:
            QMessageBox.information(
                self,
                "Analysis",
                f"ROI view '{mode}' is not yet available.",
            )
            return

        win.resize(*tuple(int(v) for v in self.cfg.roi_window_size))
        win.show()

        win.apply_finished.connect(self._on_denoise_apply_finished)

        self._roi_windows.append(win)
        win.destroyed.connect(
            lambda *_: self._roi_windows.remove(win) if win in self._roi_windows else None
        )

    def _extract_roi_data(self, roi):
        t0, t1, c0, c1 = roi.t0, roi.t1, roi.c0, roi.c1
        if self._source.dims == "nt_nc":
            data = self._source[t0:t1, c0:c1]
        else:
            data = self._source[c0:c1, t0:t1].T
        axis = getattr(self, "axis_metadata", None)
        if axis is not None:
            data = axis.canonical_nt_nc(data)
            fs = axis.time_fs(self._source)
            dx = axis.space_dx(self._source)
            axis_info = axis.as_dict()
            axis_transposed = bool(axis.requires_roi_transpose)
        else:
            fs = self._source.fs
            dx = self._source.dx
            axis_info = None
            axis_transposed = False

        return {
            "data": data,
            "fs": fs,
            "dx": dx,
            "dx_unit": getattr(self._source, "dx_unit", "m"),
            "roi": roi,
            "axis": axis_info,
            "axis_transposed": axis_transposed,
            "time_label": getattr(axis, "time_label", "Time") if axis is not None else "Time",
            "space_label": getattr(axis, "space_label", "Channel") if axis is not None else "Channel",
        }
