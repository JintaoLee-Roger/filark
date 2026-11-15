"""Overlay registry and realtime detection pool for DisplayModel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, TypeAlias

import numpy as np

from filark._exceptions import print_exception
from filark.utils.perf import elapsed_ms, log_perf, perf_debug_enabled, perf_now
from filark.viz import (
    AnnotationBBox,
    AnnotationBBoxVisual,
    AnnotationPolyline,
    AnnotationPolylineVisual,
    MaskOverlay,
    MaskOverlayVisual,
)

AnnotationBBoxNode: TypeAlias = AnnotationBBoxVisual
AnnotationPolylineNode: TypeAlias = AnnotationPolylineVisual
MaskOverlayNode: TypeAlias = MaskOverlayVisual
AnnotationOverlayNode: TypeAlias = AnnotationBBoxVisual | AnnotationPolylineVisual | MaskOverlayVisual


def _print_exception(context: str) -> None:
    print_exception(context, prefix="DisplayModel")


@dataclass
class OverlayRecord:
    """Committed canvas overlay tracked by uid."""

    uid: int
    anno_type: str
    node: AnnotationOverlayNode
    payload: dict
    source_kind: str = "annotation"
    selectable: bool = True
    editable: bool = True


@dataclass
class VirtualOverlayRecord:
    """Annotation overlay stored in model space, rendered only when needed."""

    uid: int
    anno_type: str
    payload: dict
    source_kind: str = "annotation"
    selectable: bool = True
    editable: bool = True
    bounds: tuple[float, float, float, float] | None = None
    revision: int = 0


class OverlayRegistryMixin:
    """Own committed overlays and pooled realtime detection visuals."""

    def _init_overlay_state(self) -> None:
        options = dict(getattr(self, "_annotation_options", {}) or {})
        self._overlays: Dict[int, OverlayRecord] = {}
        self._realtime_detection_pools: dict[str, list[dict]] = {"bbox": [], "polyline": [], "mask": []}
        self._realtime_detection_next_slot: dict[str, int] = {"bbox": 0, "polyline": 0, "mask": 0}
        self._realtime_mask_capacity = (512, 512)
        self._annotation_virtual_overlays: Dict[int, VirtualOverlayRecord] = {}
        self._annotation_render_pools: dict[str, list[dict]] = {"bbox": [], "polyline": [], "mask": []}
        self._annotation_render_slot_by_uid: dict[int, dict] = {}
        self._annotation_render_limits = {
            "bbox": int(max(0, options.get("render_bbox_pool", 64) or 64)),
            "polyline": int(max(0, options.get("render_polyline_pool", 64) or 64)),
            "mask": int(max(0, options.get("render_mask_pool", 12) or 12)),
        }
        self._annotation_mask_capacity = (
            int(max(1, options.get("render_mask_capacity_t", 512) or 512)),
            int(max(1, options.get("render_mask_capacity_c", 512) or 512)),
        )
        self._annotation_render_margin_fraction = float(options.get("render_margin_fraction", 0.35) or 0.35)
        self._annotation_virtual_batch_depth = 0
        self._annotation_refresh_stats = {
            "calls": 0,
            "total_ms": 0.0,
            "max_ms": 0.0,
            "last_ms": 0.0,
            "applied": 0,
            "skipped": 0,
        }
        self._defer_overlay_canvas_update = False

    def has_overlay(self, uid: int) -> bool:
        uid = int(uid)
        return uid in self._overlays or uid in self._annotation_virtual_overlays

    def overlay_count(self) -> int:
        return len(self._overlays)

    def add_bbox_overlay(
        self,
        uid: int,
        bbox: dict,
        *,
        source_kind: str = "annotation",
        selectable: bool = True,
        editable: bool = True,
    ) -> None:
        payload = dict(bbox)
        payload.setdefault("source_kind", source_kind)
        payload.setdefault("selectable", bool(selectable))
        payload.setdefault("editable", bool(editable))
        if str(payload.get("source_kind", source_kind)) == "annotation":
            self._upsert_annotation_virtual_overlay(int(uid), "bbox", payload, selectable=selectable, editable=editable)
            return
        bbox_node = AnnotationBBox(
            parent=self._world_overlay_parent,
            color=str(payload.get("color", "yellow")),
            border_width=float(payload.get("width", payload.get("border_width", 1.0))),
        )
        bbox_node.order = 20
        record = OverlayRecord(
            uid=int(uid),
            anno_type="bbox",
            node=bbox_node,
            payload=payload,
            source_kind=str(payload.get("source_kind", source_kind)),
            selectable=bool(payload.get("selectable", selectable)),
            editable=bool(payload.get("editable", editable)),
        )
        self._overlays[record.uid] = record
        self._apply_overlay_record(record)

    def add_polyline_overlay(
        self,
        uid: int,
        data: dict,
        *,
        source_kind: str = "annotation",
        selectable: bool = True,
        editable: bool = True,
    ) -> None:
        payload = dict(data)
        payload.setdefault("source_kind", source_kind)
        payload.setdefault("selectable", bool(selectable))
        payload.setdefault("editable", bool(editable))
        if str(payload.get("source_kind", source_kind)) == "annotation":
            self._upsert_annotation_virtual_overlay(int(uid), "polyline", payload, selectable=selectable, editable=editable)
            return
        polyline_node = AnnotationPolyline(
            parent=self._world_overlay_parent,
            color=str(payload.get("color", "yellow")),
            width=2,
        )
        polyline_node.order = 20
        record = OverlayRecord(
            uid=int(uid),
            anno_type="polyline",
            node=polyline_node,
            payload=payload,
            source_kind=str(payload.get("source_kind", source_kind)),
            selectable=bool(payload.get("selectable", selectable)),
            editable=bool(payload.get("editable", editable)),
        )
        self._overlays[record.uid] = record
        self._apply_overlay_record(record)

    def add_mask_overlay(
        self,
        uid: int,
        data: dict,
        *,
        source_kind: str = "detection",
        selectable: bool = False,
        editable: bool = False,
    ) -> None:
        payload = dict(data)
        payload.setdefault("source_kind", source_kind)
        payload.setdefault("selectable", bool(selectable))
        payload.setdefault("editable", bool(editable))
        if str(payload.get("source_kind", source_kind)) == "annotation":
            self._upsert_annotation_virtual_overlay(int(uid), "mask", payload, selectable=selectable, editable=editable)
            return
        mask = payload.get("mask")
        if mask is None:
            raise ValueError("mask overlay payload requires a 'mask' array")
        arr = self._coerce_mask_array(mask)
        node = MaskOverlay(
            parent=self._world_overlay_parent,
            capacity_t=max(1, int(arr.shape[0])),
            capacity_c=max(1, int(arr.shape[1])),
            color=str(payload.get("color", "#39FF14")),
            alpha=float(payload.get("alpha", 0.45)),
            threshold=float(payload.get("threshold", 0.0)),
        )
        node.order = 18
        record = OverlayRecord(
            uid=int(uid),
            anno_type="mask",
            node=node,
            payload=payload,
            source_kind=str(payload.get("source_kind", source_kind)),
            selectable=bool(payload.get("selectable", selectable)),
            editable=bool(payload.get("editable", editable)),
        )
        self._overlays[record.uid] = record
        self._apply_overlay_record(record)

    def add_annotation_overlays(self, records: Iterable[tuple[int, str, dict] | dict]) -> None:
        """Upsert many annotation overlays without materializing all nodes."""
        self._annotation_virtual_batch_depth += 1
        try:
            for raw in records:
                if isinstance(raw, dict):
                    uid = int(raw.get("uid"))
                    anno_type = str(raw.get("anno_type", raw.get("type", "")))
                    payload = dict(raw.get("payload", raw.get("data", {})) or {})
                else:
                    uid, anno_type, payload = raw
                    uid = int(uid)
                    anno_type = str(anno_type)
                    payload = dict(payload)
                if anno_type not in ("bbox", "polyline", "mask"):
                    continue
                payload.setdefault("source_kind", "annotation")
                payload.setdefault("selectable", True)
                payload.setdefault("editable", True)
                self._upsert_annotation_virtual_overlay(uid, anno_type, payload, refresh=False)
        finally:
            self._annotation_virtual_batch_depth = max(0, self._annotation_virtual_batch_depth - 1)
        self._refresh_annotation_virtual_overlays()

    def _upsert_annotation_virtual_overlay(
        self,
        uid: int,
        anno_type: str,
        payload: dict,
        *,
        selectable: bool = True,
        editable: bool = True,
        refresh: bool = True,
    ) -> None:
        uid = int(uid)
        anno_type = str(anno_type)
        if anno_type not in ("bbox", "polyline", "mask"):
            return
        payload = dict(payload)
        payload.setdefault("source_kind", "annotation")
        payload.setdefault("selectable", bool(selectable))
        payload.setdefault("editable", bool(editable))
        old = self._annotation_virtual_overlays.get(uid)
        if old is not None and old.anno_type != anno_type:
            self._deactivate_annotation_virtual_uid(uid)
        record = VirtualOverlayRecord(
            uid=uid,
            anno_type=anno_type,
            payload=payload,
            source_kind="annotation",
            selectable=bool(payload.get("selectable", selectable)),
            editable=bool(payload.get("editable", editable)),
            bounds=self._annotation_payload_bounds(anno_type, payload),
            revision=(int(old.revision) + 1) if old is not None else 1,
        )
        self._annotation_virtual_overlays[uid] = record
        active = self._overlays.get(uid)
        if active is not None and active.source_kind == "annotation":
            active.anno_type = anno_type
            active.payload = dict(payload)
            active.selectable = record.selectable
            active.editable = record.editable
        if self._selected_annotation_uid == uid:
            self._selected_annotation_type = anno_type
            self._selected_annotation_payload = dict(payload)
        if refresh and self._annotation_virtual_batch_depth <= 0:
            self._refresh_annotation_virtual_overlays()

    def _ensure_annotation_render_pool_kind(self, kind: str) -> None:
        kind = str(kind)
        limit = int(self._annotation_render_limits.get(kind, 0))
        pool = self._annotation_render_pools.setdefault(kind, [])
        while len(pool) < limit:
            if kind == "bbox":
                node = AnnotationBBox(
                    parent=self._world_overlay_parent,
                    color="yellow",
                    border_width=1.0,
                )
            elif kind == "polyline":
                node = AnnotationPolyline(
                    parent=self._world_overlay_parent,
                    color="yellow",
                    width=2.0,
                )
            elif kind == "mask":
                cap_t, cap_c = self._annotation_mask_capacity
                node = MaskOverlay(
                    parent=self._world_overlay_parent,
                    capacity_t=cap_t,
                    capacity_c=cap_c,
                    color="#39FF14",
                    alpha=0.45,
                    threshold=0.0,
                )
            else:
                return
            node.order = 18 if kind == "mask" else 20
            if hasattr(node, "set_visible"):
                node.set_visible(False)
            pool.append({"node": node, "active": False, "uid": None, "revision": 0})

    def _claim_annotation_render_slot(self, kind: str) -> dict | None:
        self._ensure_annotation_render_pool_kind(kind)
        pool = self._annotation_render_pools.get(str(kind), [])
        for slot in pool:
            if not bool(slot.get("active", False)):
                return slot
        return None

    def _deactivate_annotation_virtual_uid(self, uid: int) -> None:
        uid = int(uid)
        record = self._overlays.pop(uid, None)
        slot = self._annotation_render_slot_by_uid.pop(uid, None)
        if slot is not None:
            slot["active"] = False
            slot["uid"] = None
            slot["revision"] = 0
            node = slot.get("node")
            if hasattr(node, "set_visible"):
                node.set_visible(False)
            return
        if record is not None:
            record.node.close()

    def _annotation_payload_bounds(self, anno_type: str, payload: dict) -> tuple[float, float, float, float] | None:
        if anno_type == "bbox":
            bbox = self._normalized_bbox(payload)
            return bbox["t0"], bbox["t1"], bbox["c0"], bbox["c1"]
        if anno_type == "polyline":
            points = [tuple(point) for point in (payload.get("points", []) or [])]
            if not points:
                return None
            ts = [float(point[0]) for point in points]
            cs = [float(point[1]) for point in points]
            return min(ts), max(ts), min(cs), max(cs)
        if anno_type == "mask":
            mask = payload.get("mask")
            shape_t = shape_c = 1
            if mask is not None:
                try:
                    shape_t = max(1, int(getattr(mask, "shape", [1, 1])[0]))
                    shape_c = max(1, int(getattr(mask, "shape", [1, 1])[1]))
                except Exception:
                    shape_t = shape_c = 1
            t0 = float(payload.get("t0", 0.0))
            c0 = float(payload.get("c0", 0.0))
            t1 = float(payload.get("t1", t0 + shape_t))
            c1 = float(payload.get("c1", c0 + shape_c))
            return min(t0, t1), max(t0, t1), min(c0, c1), max(c0, c1)
        return None

    def _expanded_annotation_view_bounds(self) -> tuple[float, float, float, float] | None:
        bounds = self.current_view_bounds()
        if bounds is None:
            return None
        t0 = float(bounds.get("t0", 0.0))
        t1 = float(bounds.get("t1", t0))
        c0 = float(bounds.get("c0", 0.0))
        c1 = float(bounds.get("c1", c0))
        margin = max(0.0, float(self._annotation_render_margin_fraction))
        dt = max(1.0, abs(t1 - t0))
        dc = max(1.0, abs(c1 - c0))
        return min(t0, t1) - dt * margin, max(t0, t1) + dt * margin, min(c0, c1) - dc * margin, max(c0, c1) + dc * margin

    @staticmethod
    def _bounds_intersect(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
        at0, at1, ac0, ac1 = a
        bt0, bt1, bc0, bc1 = b
        return at1 >= bt0 and bt1 >= at0 and ac1 >= bc0 and bc1 >= ac0

    def _annotation_render_candidates(self) -> dict[str, list[VirtualOverlayRecord]]:
        view = self._expanded_annotation_view_bounds()
        selected_uid = self._selected_annotation_uid
        center_t = center_c = 0.0
        if view is not None:
            center_t = (view[0] + view[1]) / 2.0
            center_c = (view[2] + view[3]) / 2.0

        by_kind: dict[str, list[tuple[float, VirtualOverlayRecord]]] = {"bbox": [], "polyline": [], "mask": []}
        for record in self._annotation_virtual_overlays.values():
            if not self._annotation_visible(record.payload):
                continue
            bounds = record.bounds
            is_selected = selected_uid is not None and int(record.uid) == int(selected_uid)
            if bounds is None:
                if not is_selected:
                    continue
                score = -1.0
            elif view is not None and (not is_selected) and (not self._bounds_intersect(bounds, view)):
                continue
            else:
                rt = (bounds[0] + bounds[1]) / 2.0
                rc = (bounds[2] + bounds[3]) / 2.0
                score = 0.0 if view is None else abs(rt - center_t) + abs(rc - center_c)
                if is_selected:
                    score = -1.0
            by_kind.setdefault(record.anno_type, []).append((score, record))

        out: dict[str, list[VirtualOverlayRecord]] = {"bbox": [], "polyline": [], "mask": []}
        for kind, pairs in by_kind.items():
            limit = int(self._annotation_render_limits.get(kind, 0))
            if limit <= 0:
                continue
            pairs.sort(key=lambda item: (item[0], item[1].uid))
            out[kind] = [record for _score, record in pairs[:limit]]
        return out

    def _activate_annotation_virtual_record(self, virtual: VirtualOverlayRecord) -> bool:
        active = self._overlays.get(int(virtual.uid))
        if active is None:
            slot = self._claim_annotation_render_slot(virtual.anno_type)
            if slot is None:
                return False
            slot["active"] = True
            slot["uid"] = int(virtual.uid)
            slot["revision"] = 0
            self._annotation_render_slot_by_uid[int(virtual.uid)] = slot
            active = OverlayRecord(
                uid=int(virtual.uid),
                anno_type=str(virtual.anno_type),
                node=slot["node"],
                payload=dict(virtual.payload),
                source_kind="annotation",
                selectable=bool(virtual.selectable),
                editable=bool(virtual.editable),
            )
            self._overlays[active.uid] = active
        else:
            slot = self._annotation_render_slot_by_uid.get(int(virtual.uid))
            if (
                slot is not None
                and int(slot.get("revision", 0) or 0) == int(virtual.revision)
                and active.anno_type == virtual.anno_type
            ):
                return False
            active.anno_type = str(virtual.anno_type)
            active.payload = dict(virtual.payload)
            active.selectable = bool(virtual.selectable)
            active.editable = bool(virtual.editable)
        self._apply_overlay_record(active)
        slot = self._annotation_render_slot_by_uid.get(int(virtual.uid))
        if slot is not None:
            slot["revision"] = int(virtual.revision)
        return True

    def _reset_annotation_refresh_stats(self) -> None:
        self._annotation_refresh_stats = {
            "calls": 0,
            "total_ms": 0.0,
            "max_ms": 0.0,
            "last_ms": 0.0,
            "applied": 0,
            "skipped": 0,
        }

    def _record_annotation_refresh_stats(self, ms: float, *, applied: int = 0, skipped: int = 0) -> None:
        stats = getattr(self, "_annotation_refresh_stats", None)
        if stats is None:
            self._reset_annotation_refresh_stats()
            stats = self._annotation_refresh_stats
        ms = max(0.0, float(ms))
        stats["calls"] = int(stats.get("calls", 0)) + 1
        stats["total_ms"] = float(stats.get("total_ms", 0.0)) + ms
        stats["max_ms"] = max(float(stats.get("max_ms", 0.0)), ms)
        stats["last_ms"] = ms
        stats["applied"] = int(stats.get("applied", 0) or 0) + int(max(0, applied))
        stats["skipped"] = int(stats.get("skipped", 0) or 0) + int(max(0, skipped))

    def realtime_annotation_stats(self, *, reset: bool = False) -> dict:
        if reset:
            self._reset_annotation_refresh_stats()
        stats = getattr(self, "_annotation_refresh_stats", None)
        if stats is None:
            self._reset_annotation_refresh_stats()
            stats = self._annotation_refresh_stats

        virtual_by_kind = {"bbox": 0, "polyline": 0, "mask": 0}
        for record in self._annotation_virtual_overlays.values():
            if record.anno_type in virtual_by_kind:
                virtual_by_kind[record.anno_type] += 1

        active_by_kind = {"bbox": 0, "polyline": 0, "mask": 0}
        for record in self._overlays.values():
            if record.source_kind == "annotation" and record.anno_type in active_by_kind:
                active_by_kind[record.anno_type] += 1

        pool_slots_by_kind = {}
        pool_active_by_kind = {}
        for kind, pool in self._annotation_render_pools.items():
            pool_slots_by_kind[kind] = len(pool)
            pool_active_by_kind[kind] = sum(1 for slot in pool if bool(slot.get("active", False)))

        calls = int(stats.get("calls", 0) or 0)
        total_ms = float(stats.get("total_ms", 0.0) or 0.0)
        return {
            "thread": "main",
            "calls": calls,
            "total_ms": total_ms,
            "avg_ms": total_ms / max(1, calls),
            "max_ms": float(stats.get("max_ms", 0.0) or 0.0),
            "last_ms": float(stats.get("last_ms", 0.0) or 0.0),
            "applied": int(stats.get("applied", 0) or 0),
            "skipped": int(stats.get("skipped", 0) or 0),
            "virtual_count": int(sum(virtual_by_kind.values())),
            "virtual_bbox": int(virtual_by_kind["bbox"]),
            "virtual_polyline": int(virtual_by_kind["polyline"]),
            "virtual_mask": int(virtual_by_kind["mask"]),
            "active_total": int(sum(active_by_kind.values())),
            "active_bbox": int(active_by_kind["bbox"]),
            "active_polyline": int(active_by_kind["polyline"]),
            "active_mask": int(active_by_kind["mask"]),
            "pool_slots_total": int(sum(pool_slots_by_kind.values())),
            "pool_slots_bbox": int(pool_slots_by_kind.get("bbox", 0)),
            "pool_slots_polyline": int(pool_slots_by_kind.get("polyline", 0)),
            "pool_slots_mask": int(pool_slots_by_kind.get("mask", 0)),
            "pool_active_total": int(sum(pool_active_by_kind.values())),
            "pool_active_bbox": int(pool_active_by_kind.get("bbox", 0)),
            "pool_active_polyline": int(pool_active_by_kind.get("polyline", 0)),
            "pool_active_mask": int(pool_active_by_kind.get("mask", 0)),
            "mask_capacity_t": int(self._annotation_mask_capacity[0]),
            "mask_capacity_c": int(self._annotation_mask_capacity[1]),
        }

    def _refresh_annotation_virtual_overlays(self) -> None:
        stats_start = perf_now()
        applied = 0
        skipped = 0
        try:
            if self._annotation_virtual_batch_depth > 0:
                return
            if not self._annotation_virtual_overlays:
                return
            wanted = self._annotation_render_candidates()
            wanted_uids = {int(record.uid) for records in wanted.values() for record in records}
            old_defer = bool(self._defer_overlay_canvas_update)
            self._defer_overlay_canvas_update = True
            try:
                for uid, record in list(self._overlays.items()):
                    if record.source_kind != "annotation" or uid in wanted_uids:
                        continue
                    self._deactivate_annotation_virtual_uid(uid)
                for records in wanted.values():
                    for record in records:
                        if self._activate_annotation_virtual_record(record):
                            applied += 1
                        else:
                            skipped += 1
            finally:
                self._defer_overlay_canvas_update = old_defer
            self._canvas.update()
        finally:
            self._record_annotation_refresh_stats(elapsed_ms(stats_start), applied=applied, skipped=skipped)

    def ensure_realtime_detection_pool(
        self,
        *,
        bbox_count: int = 32,
        polyline_count: int = 32,
        mask_count: int = 12,
        mask_capacity_t: int = 512,
        mask_capacity_c: int = 512,
    ) -> None:
        """Pre-create hidden detection-only visuals for realtime reuse."""
        self._realtime_mask_capacity = (int(max(1, mask_capacity_t)), int(max(1, mask_capacity_c)))
        self._ensure_realtime_detection_pool_kind("bbox", int(bbox_count))
        self._ensure_realtime_detection_pool_kind("polyline", int(polyline_count))
        self._ensure_realtime_detection_pool_kind("mask", int(mask_count))
        self._canvas.update()

    def _ensure_realtime_detection_pool_kind(self, kind: str, count: int) -> None:
        count = max(0, int(count))
        pool = self._realtime_detection_pools.setdefault(kind, [])
        while len(pool) < count:
            if kind == "bbox":
                node = AnnotationBBox(
                    parent=self._world_overlay_parent,
                    color="yellow",
                    border_width=1.0,
                )
            elif kind == "polyline":
                node = AnnotationPolyline(
                    parent=self._world_overlay_parent,
                    color="yellow",
                    width=2.0,
                )
            elif kind == "mask":
                cap_t, cap_c = self._realtime_mask_capacity
                node = MaskOverlay(
                    parent=self._world_overlay_parent,
                    capacity_t=cap_t,
                    capacity_c=cap_c,
                    color="#39FF14",
                    alpha=0.45,
                    threshold=0.0,
                )
            else:
                return
            node.order = 18 if kind == "mask" else 20
            if hasattr(node, "set_visible"):
                node.set_visible(False)
            pool.append({"node": node, "active": False, "seq": 0, "payload": None, "owner_uid": None})

    def show_realtime_detection(self, kind: str, payload: dict):
        """Show pooled realtime detection visuals and return claimed slots."""
        kind = str(kind)
        pool = self._realtime_detection_pools.get(kind)
        if not pool:
            self.ensure_realtime_detection_pool()
            pool = self._realtime_detection_pools.get(kind)
        if not pool:
            return None

        owner_uid = payload.get("_uid")
        if kind == "mask":
            return self._show_realtime_mask_detection(pool, payload, owner_uid=owner_uid)

        slot_indices, replaced_uids = self._claim_realtime_detection_slots(kind, pool, 1)
        if not slot_indices:
            return None
        for old_uid in replaced_uids:
            self._hide_realtime_detection_owner(kind, old_uid)
        slot_idx = slot_indices[0]
        slot = pool[slot_idx]
        self._activate_realtime_slot(slot, payload, owner_uid=owner_uid)
        if kind == "bbox":
            bbox = self._normalized_bbox(payload)
            self._apply_bbox_geometry(
                slot["node"],
                bbox,
                color=str(payload.get("color", "yellow")),
                width=float(payload.get("width", payload.get("border_width", 1.0))),
                visible=bool(payload.get("visible", True)),
            )
        elif kind == "polyline":
            self._apply_polyline_geometry(
                slot["node"],
                list(payload.get("points", []) or []),
                color=str(payload.get("color", "yellow")),
                width=float(payload.get("width", 2.0)),
                visible=bool(payload.get("visible", True)),
            )
        return {"slots": [slot_idx], "replaced_uids": sorted(replaced_uids)}

    def _claim_realtime_detection_slot(self, kind: str, pool: list[dict]) -> int:
        for idx, slot in enumerate(pool):
            if not bool(slot.get("active", False)):
                self._realtime_detection_next_slot[kind] = (idx + 1) % len(pool)
                return idx
        idx = int(self._realtime_detection_next_slot.get(kind, 0)) % len(pool)
        self._realtime_detection_next_slot[kind] = (idx + 1) % len(pool)
        return idx

    def _claim_realtime_detection_slots(self, kind: str, pool: list[dict], count: int) -> tuple[list[int], set[int]]:
        count = min(max(0, int(count)), len(pool))
        if count <= 0:
            return [], set()
        selected: list[int] = []
        replaced: set[int] = set()
        for idx, slot in enumerate(pool):
            if len(selected) >= count:
                break
            if not bool(slot.get("active", False)):
                selected.append(idx)
        cursor = int(self._realtime_detection_next_slot.get(kind, 0)) % len(pool)
        attempts = 0
        while len(selected) < count and attempts < len(pool) * 2:
            idx = cursor % len(pool)
            cursor += 1
            attempts += 1
            if idx in selected:
                continue
            old_uid = pool[idx].get("owner_uid")
            if old_uid is not None:
                try:
                    replaced.add(int(old_uid))
                except Exception:
                    pass
            selected.append(idx)
        self._realtime_detection_next_slot[kind] = cursor % len(pool)
        return selected, replaced

    @staticmethod
    def _activate_realtime_slot(slot: dict, payload: dict, *, owner_uid) -> None:
        slot["active"] = True
        slot["seq"] = int(slot.get("seq", 0)) + 1
        slot["payload"] = dict(payload)
        slot["owner_uid"] = None if owner_uid is None else int(owner_uid)

    def _hide_realtime_detection_owner(self, kind: str, owner_uid: int) -> list[int]:
        hidden: list[int] = []
        pool = self._realtime_detection_pools.get(str(kind), [])
        for idx, slot in enumerate(pool):
            if slot.get("owner_uid") != int(owner_uid):
                continue
            slot["active"] = False
            slot["payload"] = None
            slot["owner_uid"] = None
            node = slot.get("node")
            if hasattr(node, "set_visible"):
                node.set_visible(False)
            hidden.append(idx)
        return hidden

    def _show_realtime_mask_detection(self, pool: list[dict], payload: dict, *, owner_uid):
        mask = self._coerce_mask_array(payload.get("mask"))
        cap_t, cap_c = self._realtime_mask_capacity
        tiles = list(self._iter_mask_tiles(mask, cap_t=cap_t, cap_c=cap_c))
        if not tiles:
            return None
        slot_indices, replaced_uids = self._claim_realtime_detection_slots("mask", pool, len(tiles))
        if not slot_indices:
            return None
        for old_uid in replaced_uids:
            self._hide_realtime_detection_owner("mask", old_uid)

        t0 = float(payload.get("t0", 0.0))
        c0 = float(payload.get("c0", 0.0))
        t_step = float(payload.get("t_step", self._mask_step(payload, "t", mask.shape[0])))
        c_step = float(payload.get("c_step", self._mask_step(payload, "c", mask.shape[1])))
        shown_slots: list[int] = []
        for slot_idx, (tile_t0, tile_c0, tile) in zip(slot_indices, tiles):
            slot = pool[slot_idx]
            tile_payload = dict(payload)
            tile_payload["mask"] = tile
            tile_payload["t0"] = t0 + float(tile_t0) * t_step
            tile_payload["c0"] = c0 + float(tile_c0) * c_step
            tile_payload["t1"] = t0 + float(tile_t0 + tile.shape[0]) * t_step
            tile_payload["c1"] = c0 + float(tile_c0 + tile.shape[1]) * c_step
            self._activate_realtime_slot(slot, tile_payload, owner_uid=owner_uid)
            self._apply_mask_geometry(
                slot["node"],
                tile_payload,
                visible=bool(tile_payload.get("visible", True)),
            )
            shown_slots.append(int(slot_idx))
        return {"slots": shown_slots, "replaced_uids": sorted(replaced_uids)}

    @staticmethod
    def _coerce_mask_array(mask) -> np.ndarray:
        arr = np.asarray(mask, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"mask must be 2D (time, channel), got shape={arr.shape}")
        return np.ascontiguousarray(arr)

    @staticmethod
    def _mask_step(payload: dict, axis: str, n: int) -> float:
        if axis == "t":
            if "t1" in payload and n > 0:
                return max(1e-12, (float(payload["t1"]) - float(payload.get("t0", 0.0))) / float(n))
        if axis == "c":
            if "c1" in payload and n > 0:
                return max(1e-12, (float(payload["c1"]) - float(payload.get("c0", 0.0))) / float(n))
        return 1.0

    @staticmethod
    def _iter_mask_tiles(mask, *, cap_t: int, cap_c: int):
        arr = np.asarray(mask, dtype=np.float32)
        nt, nc = int(arr.shape[0]), int(arr.shape[1])
        cap_t = max(1, int(cap_t))
        cap_c = max(1, int(cap_c))
        for t0 in range(0, nt, cap_t):
            for c0 in range(0, nc, cap_c):
                tile = arr[t0:min(nt, t0 + cap_t), c0:min(nc, c0 + cap_c)]
                if tile.size == 0 or not np.any(tile > 0):
                    continue
                yield t0, c0, np.ascontiguousarray(tile)

    def hide_realtime_detection(self, kind: str, slot_idx: int) -> None:
        pool = self._realtime_detection_pools.get(str(kind), [])
        idx = int(slot_idx)
        if idx < 0 or idx >= len(pool):
            return
        slot = pool[idx]
        slot["active"] = False
        slot["payload"] = None
        slot["owner_uid"] = None
        node = slot.get("node")
        if hasattr(node, "set_visible"):
            node.set_visible(False)
        if not self._defer_overlay_canvas_update:
            self._canvas.update()

    def clear_realtime_detections(self) -> None:
        for kind, pool in self._realtime_detection_pools.items():
            self._realtime_detection_next_slot[kind] = 0
            for slot in pool:
                slot["active"] = False
                slot["payload"] = None
                slot["owner_uid"] = None
                node = slot.get("node")
                if hasattr(node, "set_visible"):
                    node.set_visible(False)
        self._canvas.update()

    def begin_overlay_batch_update(self) -> None:
        self._defer_overlay_canvas_update = True

    def end_overlay_batch_update(self) -> None:
        self._defer_overlay_canvas_update = False
        self._canvas.update()

    def update_overlay(self, uid: int, patch: dict) -> None:
        uid = int(uid)
        virtual = self._annotation_virtual_overlays.get(uid)
        if virtual is not None:
            payload = dict(virtual.payload)
            payload.update(dict(patch))
            selectable = bool(payload.get("selectable", virtual.selectable))
            editable = bool(payload.get("editable", virtual.editable))
            self._upsert_annotation_virtual_overlay(
                uid,
                virtual.anno_type,
                payload,
                selectable=selectable,
                editable=editable,
            )
            return

        record = self._overlays.get(uid)
        if record is None:
            return
        record.payload.update(dict(patch))
        if "source_kind" in patch:
            record.source_kind = str(patch["source_kind"])
        if "selectable" in patch:
            record.selectable = bool(patch["selectable"])
        if "editable" in patch:
            record.editable = bool(patch["editable"])
        self._apply_overlay_record(record)

    def remove_overlay(self, uid: int) -> None:
        uid = int(uid)
        virtual = self._annotation_virtual_overlays.pop(uid, None)
        if virtual is not None:
            self._deactivate_annotation_virtual_uid(uid)
            if self._selected_annotation_uid == uid:
                self._selected_annotation_uid = None
                self._selected_annotation_type = None
                self._selected_annotation_payload = None
                self._refresh_selection_visuals()
            self._canvas.update()
            return

        record = self._overlays.pop(uid, None)
        if record is None:
            return
        record.node.close()
        if self._selected_annotation_uid == uid:
            self._selected_annotation_uid = None
            self._selected_annotation_type = None
            self._selected_annotation_payload = None
            self._refresh_selection_visuals()
        self._canvas.update()

    def clear_overlays(self) -> None:
        for uid in list(self._annotation_virtual_overlays):
            self._deactivate_annotation_virtual_uid(uid)
        self._annotation_virtual_overlays.clear()
        for pool in self._annotation_render_pools.values():
            for slot in pool:
                node = slot.get("node")
                try:
                    node.close()
                except Exception:
                    _print_exception("failed to close annotation render pool node")
        self._annotation_render_pools = {"bbox": [], "polyline": [], "mask": []}
        self._annotation_render_slot_by_uid.clear()

        for record in list(self._overlays.values()):
            record.node.close()
        self._overlays.clear()
        self._selected_annotation_uid = None
        self._selected_annotation_type = None
        self._selected_annotation_payload = None
        self._refresh_selection_visuals()
        self._canvas.update()

    def clear_overlays_by_kind(self, source_kind: str) -> None:
        source_kind = str(source_kind)
        removed_any = False
        selected_removed = False
        if source_kind == "annotation":
            for uid in list(self._annotation_virtual_overlays):
                if self._selected_annotation_uid == uid:
                    selected_removed = True
                self._deactivate_annotation_virtual_uid(uid)
            removed_any = bool(self._annotation_virtual_overlays) or removed_any
            self._annotation_virtual_overlays.clear()
            for pool in self._annotation_render_pools.values():
                for slot in pool:
                    slot["active"] = False
                    slot["uid"] = None
                    slot["revision"] = 0
                    node = slot.get("node")
                    if hasattr(node, "set_visible"):
                        node.set_visible(False)
            self._annotation_render_slot_by_uid.clear()
        for uid, record in list(self._overlays.items()):
            if record.source_kind != source_kind:
                continue
            if source_kind == "annotation":
                self._deactivate_annotation_virtual_uid(uid)
            else:
                record.node.close()
                self._overlays.pop(uid, None)
            removed_any = True
            if self._selected_annotation_uid == uid:
                selected_removed = True
        if selected_removed:
            self._selected_annotation_uid = None
            self._selected_annotation_type = None
            self._selected_annotation_payload = None
            self._refresh_selection_visuals()
        if removed_any:
            self._canvas.update()

    def _apply_overlay_record(self, record: OverlayRecord) -> None:
        perf_on = perf_debug_enabled()
        total_start = perf_now() if perf_on else 0.0
        color = str(record.payload.get("color", "yellow"))
        visible = bool(record.payload.get("visible", True))

        if record.anno_type == "bbox":
            bbox = self._normalized_bbox(record.payload)
            fallback_width = getattr(self, "_annotation_bbox_width", 1.0) if record.source_kind == "annotation" else 1.0
            width = float(record.payload.get("width", record.payload.get("border_width", fallback_width)))
            self._apply_bbox_geometry(record.node, bbox, color=color, width=width, visible=visible)
        elif record.anno_type == "polyline":
            points = list(record.payload.get("points", []) or [])
            fallback_width = getattr(self, "_annotation_polyline_width", 2.0) if record.source_kind == "annotation" else 2.0
            width = float(record.payload.get("width", fallback_width))
            self._apply_polyline_geometry(record.node, points, color=color, width=width, visible=visible)
        elif record.anno_type == "mask":
            self._ensure_mask_record_capacity(record)
            self._apply_mask_geometry(record.node, record.payload, visible=visible)

        if self._selected_annotation_uid == record.uid:
            self._selected_annotation_payload = dict(record.payload)
            if self._selected_annotation_type is None:
                self._selected_annotation_type = record.anno_type
            self._refresh_selection_visuals()
        if perf_on:
            log_perf(
                "overlay.apply",
                uid=int(record.uid),
                anno_type=str(record.anno_type),
                visible=int(visible),
                overlay_count=int(len(self._overlays)),
                ms=elapsed_ms(total_start),
            )

    def _annotation_color(self, payload: dict | None, fallback: str = "yellow") -> str:
        if payload is None:
            return fallback
        return str(payload.get("color", fallback))

    def _annotation_visible(self, payload: dict | None) -> bool:
        return bool(True if payload is None else payload.get("visible", True))

    def _normalized_bbox(self, payload: dict) -> dict:
        t0 = float(payload.get("t0", 0))
        c0 = float(payload.get("c0", 0))
        t1 = float(payload.get("t1", 0))
        c1 = float(payload.get("c1", 0))
        if t1 < t0:
            t0, t1 = t1, t0
        if c1 < c0:
            c0, c1 = c1, c0
        return {"t0": t0, "c0": c0, "t1": t1, "c1": c1}

    def _rounded_bbox(self, bbox: dict) -> dict:
        normalized = self._normalized_bbox(bbox)
        return {key: int(round(value)) for key, value in normalized.items()}

    def _apply_bbox_geometry(
        self,
        bbox_node: AnnotationBBoxNode,
        bbox: dict,
        *,
        color: str,
        width: float,
        visible: bool,
    ) -> None:
        bbox_node.show(
            bbox,
            color=color,
            border_width=max(1.0, float(width)),
            visible=visible,
            handles=False,
        )
        if not self._defer_overlay_canvas_update:
            self._canvas.update()

    def _apply_polyline_geometry(
        self,
        polyline_node: AnnotationPolylineNode,
        points,
        *,
        color: str,
        width: float,
        visible: bool,
    ) -> None:
        polyline_node.show(
            points,
            color=color,
            width=max(1.0, float(width)),
            visible=visible,
            show_markers=False,
            cursor_segment=None,
        )
        if not self._defer_overlay_canvas_update:
            self._canvas.update()

    def _apply_mask_geometry(
        self,
        mask_node: MaskOverlayNode,
        payload: dict,
        *,
        visible: bool,
    ) -> None:
        mask_node.show(
            payload.get("mask"),
            t0=float(payload.get("t0", 0.0)),
            c0=float(payload.get("c0", 0.0)),
            t1=float(payload["t1"]) if "t1" in payload else None,
            c1=float(payload["c1"]) if "c1" in payload else None,
            color=str(payload.get("color", "#39FF14")),
            alpha=float(payload.get("alpha", 0.45)),
            threshold=float(payload.get("threshold", 0.0)),
            visible=visible,
        )
        if not self._defer_overlay_canvas_update:
            self._canvas.update()

    def _ensure_mask_record_capacity(self, record: OverlayRecord) -> None:
        mask = np.asarray(record.payload.get("mask"), dtype=np.float32)
        if mask.ndim != 2:
            return
        cap_t, cap_c = record.node.capacity
        if cap_t >= int(mask.shape[0]) and cap_c >= int(mask.shape[1]):
            return
        old = record.node
        order = getattr(old, "order", 18)
        try:
            old.close()
        except Exception:
            _print_exception("failed to replace undersized mask overlay")
        node = MaskOverlay(
            parent=self._world_overlay_parent,
            capacity_t=max(1, int(mask.shape[0])),
            capacity_c=max(1, int(mask.shape[1])),
            color=str(record.payload.get("color", "#39FF14")),
            alpha=float(record.payload.get("alpha", 0.45)),
            threshold=float(record.payload.get("threshold", 0.0)),
        )
        node.order = order
        record.node = node
        slot = self._annotation_render_slot_by_uid.get(int(record.uid))
        if slot is not None:
            slot["node"] = node

    def _selected_record(self) -> Optional[OverlayRecord]:
        if self._selected_annotation_uid is None:
            return None
        return self._overlays.get(int(self._selected_annotation_uid))

    def _selected_payload(self) -> Optional[dict]:
        record = self._selected_record()
        if record is not None:
            return dict(record.payload)
        if self._selected_annotation_payload is None:
            return None
        return dict(self._selected_annotation_payload)
