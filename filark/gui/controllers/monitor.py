"""Monitor and realtime detection controller for the main GUI window."""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QMessageBox

from filark._exceptions import print_exception
from filark.io.tapeio import NpyTape
from filark.utils.annotations import describe_source

from .monitor_workers import MonitorWorker, RealtimeDisplayWorker, scale_event_coords
from .monitor_writer import MonitorAnnotationWriter


def _print_exception(context: str) -> None:
    print_exception(context, prefix="MonitorController")

class MonitorCoreMixin:
    """Own monitor state, source prep, worker launch, cancel, and shutdown."""

    def _init_monitor_state(self) -> None:
        self._monitor_overlay_uids: set[int] = set()
        self._next_monitor_overlay_uid: int = -(10 ** 9)
        self._active_monitor_run_token: int = 0
        self._monitor_thread = None
        self._monitor_worker = None
        self._monitor_run_contexts: dict[int, dict] = {}
        self._realtime_covered: dict | None = None
        self._realtime_uid_ct: dict[int, float] = {}
        self._realtime_uid_slot: dict[int, set[tuple[str, int]]] = {}
        self._realtime_slot_uid: dict[tuple[str, int], int] = {}
        self._realtime_display_plugin = None
        self._realtime_display_params: dict | None = None
        self._realtime_display_thread = None
        self._realtime_display_worker = None
        self._realtime_display_stats = self._new_realtime_display_stats()
        self._monitor_result_writer: MonitorAnnotationWriter | None = None
        self._monitor_result_writer_last_status: dict | None = None

    def _new_realtime_display_stats(self) -> dict[str, float | int]:
        return {
            "submitted": 0,
            "completed": 0,
            "dropped": 0,
            "failed": 0,
            "events_added": 0,
            "queue": 0,
            "last_run_ms": 0.0,
            "last_queue_wait_ms": 0.0,
            "last_overlay_ms": 0.0,
            "last_overlay_count": 0,
            "overlay_ms_total": 0.0,
            "overlay_calls": 0,
            "overlay_ms_max": 0.0,
            "last_evict_ms": 0.0,
            "evict_ms_total": 0.0,
            "evict_calls": 0,
            "evict_ms_max": 0.0,
        }

    def _monitor_is_running(self) -> bool:
        return bool(
            (self._monitor_thread is not None and self._monitor_thread.isRunning())
            or self._realtime_display_plugin is not None
        )

    def _build_monitor_source(self, scope: str):
        if self._source is None:
            raise RuntimeError("No source loaded.")

        scope = str(scope or "current_view")
        if scope == "full_source":
            return self._source, {
                "scope": "full_source",
                "scope_label": "full source",
                "origin_t": 0,
                "origin_c": 0,
            }, False

        bounds = self.disp_model.current_view_bounds()
        if bounds is None:
            raise RuntimeError("Current view bounds are unavailable.")

        t0 = int(bounds["t0"])
        t1 = int(bounds["t1"])
        c0 = int(bounds["c0"])
        c1 = int(bounds["c1"])
        if t1 <= t0 or c1 <= c0:
            raise RuntimeError("Current view is empty.")

        if self._source.dims == "nt_nc":
            patch = self._source[t0:t1, c0:c1]
        else:
            patch = self._source[c0:c1, t0:t1].T
        data = np.asarray(patch)
        if data.ndim != 2 or data.size == 0:
            raise RuntimeError("Current view data is empty.")

        subsource = NpyTape.from_array(
            data,
            dims="nt_nc",
            fs=float(getattr(self._source, "fs", 1.0) or 1.0),
            dx=float(getattr(self._source, "dx", 1.0) or 1.0),
            dx_unit=str(getattr(self._source, "dx_unit", "m") or "m"),
        )
        subsource.monitor_scope = "current_view"
        subsource.monitor_event_kind = "current"
        subsource.monitor_origin_t = int(t0)
        subsource.monitor_origin_c = int(c0)
        subsource.monitor_trigger_t0 = int(t0)
        subsource.monitor_trigger_t1 = int(t1)
        subsource.monitor_trigger_c0 = int(c0)
        subsource.monitor_trigger_c1 = int(c1)
        subsource.monitor_scale_x = 1.0
        subsource.monitor_scale_y = 1.0
        return subsource, {
            "scope": "current_view",
            "scope_label": "current view",
            "origin_t": int(t0),
            "origin_c": int(c0),
            "bounds": {"t0": int(t0), "t1": int(t1), "c0": int(c0), "c1": int(c1)},
        }, True

    # -------------------------
    # Monitor plugin
    # -------------------------

    def _on_monitor_run(self, plugin, params: dict, options: dict):
        if self._source is None:
            QMessageBox.information(self, "Monitor", "No data source loaded.")
            return
        axis = getattr(self, "axis_metadata", None)
        if axis is not None and not axis.is_default_time_channel:
            QMessageBox.information(
                self,
                "Monitor",
                "Monitor plugins only support the default time-channel view.",
            )
            return
        if self._monitor_is_running():
            QMessageBox.information(
                self,
                "Monitor",
                "A monitor run is already in progress.\nCancel it or wait for it to finish.",
            )
            return

        scope = str((options or {}).get("scope", "current_view") or "current_view")
        if scope == "realtime_display":
            self._start_monitor_result_writer(plugin, params, options or {}, scope=scope)
            self._start_realtime_display(plugin, params)
            return

        try:
            src, context, owns_source = self._build_monitor_source(scope)
        except Exception as exc:
            _print_exception("Failed to prepare monitor source.")
            QMessageBox.warning(self, "Monitor", str(exc))
            return

        fs = float(getattr(src, "fs", 1.0))
        dx = float(getattr(src, "dx", 1.0))
        dims = getattr(src, "dims", "nt_nc")
        nt = src.shape[0] if dims == "nt_nc" else src.shape[1]
        nc = src.shape[1] if dims == "nt_nc" else src.shape[0]

        try:
            params = plugin.validate_params(params, fs=fs, dx=dx, nt=nt, nc=nc)
        except Exception as exc:
            _print_exception(f"Monitor plugin '{plugin.label}' rejected parameters.")
            QMessageBox.warning(self, "Monitor — Invalid Params", str(exc))
            return

        self._active_monitor_run_token += 1
        run_token = int(self._active_monitor_run_token)
        self._clear_monitor_overlays()
        self._monitor_run_contexts[run_token] = dict(context)
        self._start_monitor_result_writer(plugin, params, options or {}, scope=scope)
        self.detect_window.set_monitor_busy(True)
        self.detect_window.set_monitor_progress(None, None)
        self.detect_window.set_monitor_status(
            f"Running {plugin.label} on {context['scope_label']}…"
        )
        self._launch_monitor_thread(plugin, src, params, context, owns_source, run_token)

    def _launch_monitor_thread(self, plugin, src, params, context, owns_source, run_token: int):
        """Start a monitor worker in a background QThread."""
        thread = QThread(self)
        worker = MonitorWorker(plugin, src, params, run_token, owns_source)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.batch_ready.connect(
            lambda evts, streamed, token: self._on_monitor_batch(evts, plugin.label, token, streamed)
        )
        worker.progress.connect(self._on_monitor_progress)
        worker.finished.connect(
            lambda evts, streamed, cancelled, token: self._on_monitor_finished(evts, plugin.label, token, streamed, cancelled)
        )
        worker.failed.connect(self._on_monitor_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_monitor_thread_finished)
        self._monitor_thread = thread
        self._monitor_worker = worker
        thread.start()

    def _shutdown_background_workers(self) -> None:
        self._stop_realtime_display()
        self._finish_monitor_result_writer(cancelled=True, update_ui=False)
        if self._monitor_worker is not None:
            self._monitor_worker.cancel()
        monitor_thread = self._monitor_thread
        if monitor_thread is not None and monitor_thread.isRunning():
            monitor_thread.wait(1000)
        realtime_thread = self._realtime_display_thread
        realtime_worker = self._realtime_display_worker
        if realtime_worker is not None:
            realtime_worker.stop()
        if realtime_thread is not None and realtime_thread.isRunning():
            realtime_thread.quit()
            if not realtime_thread.wait(1000):
                realtime_thread.terminate()
                realtime_thread.wait(1000)

    def _on_monitor_cancel(self) -> None:
        if self._realtime_display_plugin is not None:
            self._stop_active_realtime_display_monitor(cancelled=True, clear_overlays=True)
            return
        if self._monitor_worker is None:
            return
        self._monitor_worker.cancel()
        self.detect_window.set_monitor_status("Cancelling monitor run…")
        self.detect_window.btn_monitor_cancel.setEnabled(False)

    def _on_canvas_realtime_autoscroll_state_changed(self, running: bool, reason: str) -> None:
        """Stop display-driven detection when the canvas realtime switch is turned off."""
        if bool(running) or self._realtime_display_plugin is None:
            return
        self._stop_active_realtime_display_monitor(cancelled=True, clear_overlays=False)
        self.detect_window.set_monitor_status(f"Realtime monitor stopped with autoscroll ({reason}).")

    def _stop_active_realtime_display_monitor(self, *, cancelled: bool, clear_overlays: bool) -> None:
        self._stop_realtime_display()
        if self._monitor_worker is not None:
            self._monitor_worker.cancel()
        if clear_overlays:
            self._realtime_evict_all()
        self.detect_window.reset_monitor_feedback()
        self._finish_monitor_result_writer(cancelled=cancelled)

    def _on_monitor_thread_finished(self) -> None:
        self._monitor_thread = None
        self._monitor_worker = None

    def _start_monitor_result_writer(self, plugin, params: dict, options: dict, *, scope: str) -> None:
        self._finish_monitor_result_writer(cancelled=True, update_ui=False)
        if not bool((options or {}).get("save_enabled", False)):
            return
        path = str((options or {}).get("save_path", "") or "").strip()
        if not path:
            return
        session = {
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "plugin_key": getattr(plugin, "key", None),
            "plugin_label": getattr(plugin, "label", "monitor"),
            "scope": str(scope),
            "params": dict(params or {}),
        }
        try:
            self._monitor_result_writer = MonitorAnnotationWriter(
                path,
                source=describe_source(self._source, axis_metadata=getattr(self, "axis_metadata", None)),
                session=session,
            )
            self._monitor_result_writer_last_status = None
            if hasattr(self.detect_window, "set_monitor_save_status"):
                self.detect_window.set_monitor_save_status(f"Appending detections to {path}")
        except Exception as exc:
            self._monitor_result_writer = None
            _print_exception("Failed to start monitor result writer.")
            QMessageBox.warning(self, "Monitor Save Failed", str(exc))

    def _finish_monitor_result_writer(self, *, cancelled: bool, update_ui: bool = True) -> dict | None:
        writer = self._monitor_result_writer
        if writer is None:
            return None
        self._monitor_result_writer = None
        status = writer.close(cancelled=cancelled)
        self._monitor_result_writer_last_status = dict(status)
        if update_ui and hasattr(self.detect_window, "set_monitor_save_status"):
            if status.get("error"):
                self.detect_window.set_monitor_save_status(f"Save failed: {status['error']}", error=True)
            elif status.get("alive"):
                self.detect_window.set_monitor_save_status(
                    f"Save still closing: {int(status.get('items', 0))} detection(s) written to {status.get('path', '')}",
                    error=True,
                )
            else:
                dropped = int(status.get("dropped", 0) or 0)
                suffix = f", {dropped} dropped" if dropped > 0 else ""
                self.detect_window.set_monitor_save_status(
                    f"Saved {int(status.get('items', 0))} detection(s){suffix} to {status.get('path', '')}"
                )
        return status

    def _save_monitor_result_event(self, anno_type: str, payload: dict) -> None:
        writer = self._monitor_result_writer
        if writer is None:
            return
        writer.enqueue(anno_type, payload)

class RealtimeMonitorOverlayMixin:
    """Own realtime detection add/evict bookkeeping."""

    def _realtime_evict(self, t0: int, t1: int, *, drop_future: bool = True):
        """Remove realtime events whose center_t is no longer useful for the view."""
        to_remove = [
            uid for uid, ct in self._realtime_uid_ct.items()
            if ct < t0 or (drop_future and ct >= t1)
        ]
        begin = getattr(self.disp_model, "begin_overlay_batch_update", None)
        end = getattr(self.disp_model, "end_overlay_batch_update", None)
        if to_remove and callable(begin):
            begin()
        try:
            for uid in to_remove:
                self._release_realtime_uid(uid)
        finally:
            if to_remove and callable(end):
                end()

    def _realtime_evict_all(self):
        """Remove all realtime events from canvas."""
        begin = getattr(self.disp_model, "begin_overlay_batch_update", None)
        end = getattr(self.disp_model, "end_overlay_batch_update", None)
        if callable(begin):
            begin()
        try:
            for uid in list(self._realtime_uid_ct):
                self._release_realtime_uid(uid)
        finally:
            if callable(end):
                end()
        if hasattr(self.disp_model, "clear_realtime_detections"):
            self.disp_model.clear_realtime_detections()
        self._realtime_slot_uid.clear()
        self._realtime_uid_slot.clear()
        self._realtime_uid_ct.clear()

    def _release_realtime_uid(self, uid: int) -> None:
        uid = int(uid)
        slots = self._realtime_uid_slot.pop(uid, None)
        if slots is not None:
            for slot in list(slots):
                kind, slot_idx = slot
                if self._realtime_slot_uid.get(slot) == uid:
                    self._realtime_slot_uid.pop(slot, None)
                if hasattr(self.disp_model, "hide_realtime_detection"):
                    self.disp_model.hide_realtime_detection(kind, slot_idx)
        else:
            self.disp_model.remove_overlay(uid)
        self._monitor_overlay_uids.discard(uid)
        self._realtime_uid_ct.pop(uid, None)

    def _forget_realtime_uid(self, uid: int) -> None:
        uid = int(uid)
        slots = self._realtime_uid_slot.pop(uid, set())
        for slot in list(slots):
            if self._realtime_slot_uid.get(slot) == uid:
                self._realtime_slot_uid.pop(slot, None)
        self._monitor_overlay_uids.discard(uid)
        self._realtime_uid_ct.pop(uid, None)

    def _realtime_add_events(
        self,
        events: list[dict],
        *,
        label: str,
        origin_t: int,
        origin_c: int,
        owned_t0: int | None,
        owned_t1: int | None,
        visible_t0: int | None = None,
        visible_t1: int | None = None,
    ) -> int:
        """Add realtime detection events, tracking their center_t for later eviction."""
        added = 0
        begin = getattr(self.disp_model, "begin_overlay_batch_update", None)
        end = getattr(self.disp_model, "end_overlay_batch_update", None)
        if callable(begin):
            begin()
        try:
            for event in events:
                normalized = self._normalize_monitor_event(
                    event, fallback_label=label, origin_t=origin_t, origin_c=origin_c
                )
                if normalized is None:
                    continue
                geometry_type, payload = normalized

                # Filter to only the owned region (exclude patch-context overlap).
                if geometry_type == "bbox":
                    center_t = (float(payload["t0"]) + float(payload["t1"])) / 2.0
                elif geometry_type == "polyline":
                    pts = payload.get("points", [])
                    center_t = sum(p[0] for p in pts) / len(pts) if pts else 0.0
                elif geometry_type == "mask":
                    center_t = (float(payload["t0"]) + float(payload["t1"])) / 2.0
                else:
                    continue

                meta = payload.get("meta", {}) if isinstance(payload.get("meta", {}), dict) else {}
                event_owned_t0 = meta.get("owned_t0", owned_t0)
                event_owned_t1 = meta.get("owned_t1", owned_t1)
                if event_owned_t0 is not None and center_t < float(event_owned_t0):
                    continue
                if event_owned_t1 is not None and center_t >= float(event_owned_t1):
                    continue

                self._save_monitor_result_event(geometry_type, payload)

                if visible_t0 is not None and center_t < visible_t0:
                    continue
                if visible_t1 is not None and center_t > visible_t1:
                    continue

                uid = self._allocate_monitor_uid()
                slot_idx = None
                show_pooled = getattr(self.disp_model, "show_realtime_detection", None)
                if callable(show_pooled):
                    pooled_payload = dict(payload)
                    pooled_payload["_uid"] = uid
                    slot_idx = show_pooled(geometry_type, pooled_payload)
                if slot_idx is None:
                    if geometry_type == "bbox":
                        self.disp_model.add_bbox_overlay(
                            uid, payload, source_kind="detection", selectable=False, editable=False
                        )
                    elif geometry_type == "polyline":
                        self.disp_model.add_polyline_overlay(
                            uid, payload, source_kind="detection", selectable=False, editable=False
                        )
                    elif geometry_type == "mask":
                        self.disp_model.add_mask_overlay(
                            uid, payload, source_kind="detection", selectable=False, editable=False
                        )
                else:
                    if isinstance(slot_idx, dict):
                        slot_indices = list(slot_idx.get("slots", []) or [])
                        replaced_uids = list(slot_idx.get("replaced_uids", []) or [])
                    elif isinstance(slot_idx, (list, tuple, set)):
                        slot_indices = list(slot_idx)
                        replaced_uids = []
                    else:
                        slot_indices = [int(slot_idx)]
                        replaced_uids = []
                    for old_uid in replaced_uids:
                        if int(old_uid) != uid:
                            self._forget_realtime_uid(int(old_uid))
                    slot_keys = {(geometry_type, int(idx)) for idx in slot_indices}
                    for slot_key in slot_keys:
                        old_uid = self._realtime_slot_uid.get(slot_key)
                        if old_uid is not None and old_uid != uid:
                            self._forget_realtime_uid(old_uid)
                        self._realtime_slot_uid[slot_key] = uid
                    self._realtime_uid_slot[uid] = slot_keys
                self._monitor_overlay_uids.add(uid)
                self._realtime_uid_ct[uid] = center_t
                added += 1
        finally:
            if callable(end):
                end()
        return added

class DisplayRealtimeMonitorMixin:
    """Own display chunk queueing and display-driven realtime callbacks."""

    def _start_realtime_display(self, plugin, params: dict):
        """Start display-driven realtime: canvas push callbacks trigger detection."""
        self._stop_realtime_display()
        self._realtime_display_plugin = plugin
        self._realtime_display_params = dict(params)
        self._reset_realtime_display_stats()
        self._realtime_covered = None
        ensure_pool = getattr(self.disp_model, "ensure_realtime_detection_pool", None)
        if callable(ensure_pool):
            ensure_pool(
                bbox_count=int(params.get("bbox_pool_size", self.cfg.realtime_detection_bbox_pool)),
                polyline_count=int(params.get("polyline_pool_size", self.cfg.realtime_detection_polyline_pool)),
                mask_count=int(params.get("mask_pool_size", self.cfg.realtime_detection_mask_pool)),
                mask_capacity_t=int(params.get("mask_capacity_t", self.cfg.realtime_detection_mask_capacity_t)),
                mask_capacity_c=int(params.get("mask_capacity_c", self.cfg.realtime_detection_mask_capacity_c)),
            )
        self._start_realtime_display_worker(plugin)
        self.disp_model.add_chunk_listener(self._on_chunk_ready)
        self.detect_window.set_monitor_busy(True)
        self.detect_window.set_monitor_status(
            f"Realtime {plugin.label} (from display) active — detection on rendered data."
        )
        if bool(params.get("process_initial_current", False)):
            current = self.disp_model.current_display_chunk_event()
            if current is not None:
                self._enqueue_realtime_display_event(current)

    def _stop_realtime_display(self):
        """Stop display-driven realtime and remove canvas callback."""
        if self._realtime_display_plugin is not None:
            try:
                self.disp_model.remove_chunk_listener(self._on_chunk_ready)
            except Exception:
                pass
            self._realtime_display_plugin = None
            self._realtime_display_params = None
            self._realtime_covered = None
            self._clear_realtime_display_contexts()
            self._realtime_display_stats["queue"] = 0
        worker = self._realtime_display_worker
        if worker is not None:
            worker.stop()
        thread = self._realtime_display_thread
        if thread is not None and thread.isRunning():
            thread.quit()

    def _start_realtime_display_worker(self, plugin) -> None:
        thread = QThread(self)
        worker = RealtimeDisplayWorker(plugin)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.batch_ready.connect(self._on_realtime_display_batch)
        worker.finished.connect(self._on_realtime_display_finished)
        worker.failed.connect(self._on_realtime_display_failed)
        worker.stopped.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda w=worker, t=thread: self._on_realtime_display_thread_finished(w, t))
        self._realtime_display_thread = thread
        self._realtime_display_worker = worker
        thread.start()

    def _on_realtime_display_thread_finished(self, worker, thread) -> None:
        if self._realtime_display_worker is worker:
            self._realtime_display_worker = None
        if self._realtime_display_thread is thread:
            self._realtime_display_thread = None

    def _clear_realtime_display_contexts(self) -> None:
        for token, context in list(self._monitor_run_contexts.items()):
            if context.get("display_realtime"):
                self._monitor_run_contexts.pop(token, None)

    def _reset_realtime_display_stats(self) -> None:
        self._realtime_display_stats.clear()
        self._realtime_display_stats.update(self._new_realtime_display_stats())

    def realtime_display_stats(self) -> dict:
        stats = dict(self._realtime_display_stats)
        worker = self._realtime_display_worker
        if worker is not None:
            stats["queue"] = worker.queue_size()
        stats["overlays"] = len(self._realtime_uid_ct)
        stats["active"] = bool(self._realtime_display_plugin is not None)
        return stats

    def _on_chunk_ready(self, event):
        """Callback from scheduler: new processed chunk available for detection."""
        self._enqueue_realtime_display_event(event)

    def _enqueue_realtime_display_event(self, event) -> None:
        plugin = self._realtime_display_plugin
        params = self._realtime_display_params
        worker = self._realtime_display_worker
        if plugin is None or params is None or worker is None:
            return

        kind = str(getattr(event, "kind", "chunk"))
        if kind == "reset" and not bool(params.get("process_reset_chunks", False)):
            return

        block = np.asarray(getattr(event, "block", None), dtype=np.float32)
        if block.ndim != 2 or block.size == 0:
            return

        bounds = self.disp_model.current_view_bounds()
        if bounds is not None:
            if int(getattr(event, "t1", 0)) <= int(bounds["t0"]):
                return
            evict_start = time.perf_counter()
            self._realtime_evict(int(bounds["t0"]), int(bounds["t1"]), drop_future=False)
            evict_ms = (time.perf_counter() - evict_start) * 1000.0
            self._realtime_display_stats["last_evict_ms"] = float(evict_ms)
            self._realtime_display_stats["evict_ms_total"] = (
                float(self._realtime_display_stats.get("evict_ms_total", 0.0) or 0.0) + float(evict_ms)
            )
            self._realtime_display_stats["evict_calls"] = int(
                self._realtime_display_stats.get("evict_calls", 0) or 0
            ) + 1
            self._realtime_display_stats["evict_ms_max"] = max(
                float(self._realtime_display_stats.get("evict_ms_max", 0.0) or 0.0),
                float(evict_ms),
            )

        self._active_monitor_run_token += 1
        run_token = int(self._active_monitor_run_token)
        sx = float(getattr(event, "scale_x", 1.0) or 1.0)
        sy = float(getattr(event, "scale_y", 1.0) or 1.0)
        context = {
            "realtime": True,
            "display_realtime": True,
            "scope_label": "display chunk",
            "label": plugin.label,
            "origin_t": int(getattr(event, "t0", 0)),
            "origin_c": int(getattr(event, "c0", 0)),
            "owned_t0": int(getattr(event, "t0", 0)),
            "owned_t1": int(getattr(event, "t1", 0)),
            "new_covered_t0": int(getattr(event, "t0", 0)),
            "new_covered_t1": int(getattr(event, "t1", 0)),
            "new_covered_c0": int(getattr(event, "c0", 0)),
            "new_covered_c1": int(getattr(event, "c1", 0)),
            "scale_x": sx,
            "scale_y": sy,
            "allow_future_events": True,
        }
        self._monitor_run_contexts[run_token] = context
        task = {
            "run_token": run_token,
            "kind": kind,
            "block": np.array(block, copy=True, order="C"),
            "t0": context["origin_t"],
            "t1": context["owned_t1"],
            "c0": context["origin_c"],
            "c1": context["new_covered_c1"],
            "scale_x": sx,
            "scale_y": sy,
            "params": dict(params),
            "fs": float(getattr(self._source, "fs", 1.0) or 1.0),
            "dx": float(getattr(self._source, "dx", 1.0) or 1.0),
            "dx_unit": str(getattr(self._source, "dx_unit", "m") or "m"),
            "enqueued_at": time.perf_counter(),
        }
        max_queue = max(1, int(params.get("display_queue_max", self.cfg.realtime_display_queue_max)))
        dropped, qsize = worker.enqueue(task, max_queue=max_queue)
        self._realtime_display_stats["submitted"] = int(self._realtime_display_stats.get("submitted", 0)) + 1
        self._realtime_display_stats["dropped"] = int(self._realtime_display_stats.get("dropped", 0)) + len(dropped)
        self._realtime_display_stats["queue"] = int(qsize)
        for token in dropped:
            self._monitor_run_contexts.pop(int(token), None)

    def _apply_realtime_display_events(self, events: list[dict], context: dict, label: str) -> int:
        if not events:
            self._realtime_display_stats["last_overlay_ms"] = 0.0
            self._realtime_display_stats["last_overlay_count"] = 0
            return 0
        overlay_start = time.perf_counter()
        scale_x = float(context.get("scale_x", 1.0)) or 1.0
        scale_y = float(context.get("scale_y", 1.0)) or 1.0
        if scale_x != 1.0 or scale_y != 1.0:
            events = scale_event_coords(list(events), scale_x=scale_x, scale_y=scale_y)
        bounds = self.disp_model.current_view_bounds()
        allow_future = bool(context.get("allow_future_events", False))
        count = self._realtime_add_events(
            list(events),
            label=label,
            origin_t=int(context.get("origin_t", 0)),
            origin_c=int(context.get("origin_c", 0)),
            owned_t0=context.get("owned_t0"),
            owned_t1=context.get("owned_t1"),
            visible_t0=int(bounds["t0"]) if bounds is not None else None,
            visible_t1=None if allow_future else (int(bounds["t1"]) if bounds is not None else None),
        )
        if count > 0:
            self._realtime_display_stats["events_added"] = int(self._realtime_display_stats.get("events_added", 0)) + count
        overlay_ms = (time.perf_counter() - overlay_start) * 1000.0
        self._realtime_display_stats["last_overlay_ms"] = float(overlay_ms)
        self._realtime_display_stats["last_overlay_count"] = int(count)
        self._realtime_display_stats["overlay_ms_total"] = (
            float(self._realtime_display_stats.get("overlay_ms_total", 0.0) or 0.0) + float(overlay_ms)
        )
        self._realtime_display_stats["overlay_calls"] = int(
            self._realtime_display_stats.get("overlay_calls", 0) or 0
        ) + 1
        self._realtime_display_stats["overlay_ms_max"] = max(
            float(self._realtime_display_stats.get("overlay_ms_max", 0.0) or 0.0),
            float(overlay_ms),
        )
        return count

    def _on_realtime_display_batch(self, events: list[dict], streamed: bool, run_token: int) -> None:
        if self._realtime_display_plugin is None:
            return
        context = self._monitor_run_contexts.get(int(run_token), {})
        if not context.get("display_realtime"):
            return
        count = self._apply_realtime_display_events(list(events), context, str(context.get("label", "monitor")))
        if count > 0:
            self.detect_window.set_monitor_status(
                f"Realtime {context.get('label', 'monitor')}: {len(self._realtime_uid_ct)} detection(s) on screen."
            )
        _ = streamed

    def _on_realtime_display_finished(
        self,
        events: list[dict],
        streamed: bool,
        cancelled: bool,
        run_token: int,
        metrics: dict,
    ) -> None:
        context = self._monitor_run_contexts.pop(int(run_token), {})
        if self._realtime_display_plugin is None or not context.get("display_realtime"):
            return
        if streamed:
            events = []
        count = 0 if cancelled else self._apply_realtime_display_events(
            list(events), context, str(context.get("label", "monitor"))
        )
        nct0 = context.get("new_covered_t0")
        nct1 = context.get("new_covered_t1")
        ncc0 = context.get("new_covered_c0")
        ncc1 = context.get("new_covered_c1")
        if nct0 is not None and nct1 is not None:
            self._realtime_covered = {
                "t0": int(nct0),
                "t1": int(nct1),
                "c0": int(ncc0 or 0),
                "c1": int(ncc1 or 0),
            }
        worker = self._realtime_display_worker
        self._realtime_display_stats["completed"] = int(self._realtime_display_stats.get("completed", 0)) + 1
        self._realtime_display_stats["last_run_ms"] = float(metrics.get("run_ms", 0.0) or 0.0)
        self._realtime_display_stats["last_queue_wait_ms"] = float(metrics.get("queue_wait_ms", 0.0) or 0.0)
        self._realtime_display_stats["queue"] = worker.queue_size() if worker is not None else 0
        total_rt = len(self._realtime_uid_ct)
        self.detect_window.set_monitor_status(
            f"Realtime {context.get('label', 'monitor')}: {total_rt} detection(s) on screen."
            if total_rt else f"Realtime {context.get('label', 'monitor')} active — no detections in view."
        )
        _ = count

    def _on_realtime_display_failed(self, message: str, run_token: int, metrics: dict) -> None:
        context = self._monitor_run_contexts.pop(int(run_token), {})
        if self._realtime_display_plugin is None or not context.get("display_realtime"):
            return
        self._realtime_display_stats["failed"] = int(self._realtime_display_stats.get("failed", 0)) + 1
        self._realtime_display_stats["last_run_ms"] = float(metrics.get("run_ms", 0.0) or 0.0)
        worker = self._realtime_display_worker
        self._realtime_display_stats["queue"] = worker.queue_size() if worker is not None else 0
        self.detect_window.set_monitor_status(f"Realtime monitor failed: {message}", error=True)

class MonitorOverlayMixin:
    """Own detection event normalization and non-realtime overlay insertion."""

    def _allocate_monitor_uid(self) -> int:
        self._next_monitor_overlay_uid -= 1
        return int(self._next_monitor_overlay_uid)

    def add_monitor_overlay(self, geometry_type: str, payload: dict) -> int:
        uid = self._allocate_monitor_uid()
        kind = str(geometry_type).strip().lower()
        if kind == "bbox":
            self.disp_model.add_bbox_overlay(
                uid,
                payload,
                source_kind="detection",
                selectable=False,
                editable=False,
            )
        elif kind == "polyline":
            self.disp_model.add_polyline_overlay(
                uid,
                payload,
                source_kind="detection",
                selectable=False,
                editable=False,
            )
        elif kind == "mask":
            self.disp_model.add_mask_overlay(
                uid,
                payload,
                source_kind="detection",
                selectable=False,
                editable=False,
            )
        else:
            raise ValueError(f"Unsupported monitor overlay geometry type: {geometry_type!r}")
        self._monitor_overlay_uids.add(uid)
        return uid

    def _clear_monitor_overlays(self) -> None:
        for uid in list(self._monitor_overlay_uids):
            if uid in self._realtime_uid_slot:
                self._release_realtime_uid(uid)
            else:
                self.disp_model.remove_overlay(uid)
        self._monitor_overlay_uids.clear()

    def _normalize_monitor_event(
        self,
        event: dict,
        *,
        fallback_label: str,
        origin_t: int = 0,
        origin_c: int = 0,
    ) -> tuple[str, dict] | None:
        if not isinstance(event, dict):
            return None
        geometry_type = str(event.get("geometry_type", "") or "").strip().lower()
        if not geometry_type:
            geometry_type = "mask" if event.get("mask") is not None else "polyline" if event.get("points") else "bbox"
        default_color = "#39FF14" if geometry_type == "mask" else "#FF8A00"

        payload = {
            "label": str(event.get("label", fallback_label)),
            "color": str(event.get("color", default_color)),
            "visible": bool(event.get("visible", True)),
            "score": float(event.get("score", 0.0)),
            "scale": float(event.get("scale", 1.0)),
            "width": float(event.get("width", event.get("line_width", 2.0))),
            "alpha": float(event.get("alpha", 0.45)),
            "threshold": float(event.get("threshold", 0.0)),
            "algorithm": event.get("algorithm", fallback_label),
            "meta": dict(event.get("meta", {}) or {}),
            "source_kind": "detection",
            "selectable": False,
            "editable": False,
        }
        if geometry_type == "bbox":
            try:
                payload.update(
                    {
                        "t0": int(event.get("t0", 0)) + int(origin_t),
                        "t1": int(event.get("t1", 0)) + int(origin_t),
                        "c0": int(event.get("c0", 0)) + int(origin_c),
                        "c1": int(event.get("c1", 0)) + int(origin_c),
                    }
                )
            except Exception:
                return None
            return "bbox", payload

        if geometry_type == "polyline":
            points = [
                (float(point[0]) + float(origin_t), float(point[1]) + float(origin_c))
                for point in (event.get("points", []) or [])
            ]
            if len(points) < 2:
                return None
            payload["points"] = points
            return "polyline", payload

        if geometry_type == "mask":
            mask = event.get("mask")
            if mask is None:
                return None
            arr = np.asarray(mask, dtype=np.float32)
            if arr.ndim != 2 or arr.size == 0:
                return None
            try:
                t0 = float(event.get("t0", 0)) + float(origin_t)
                c0 = float(event.get("c0", 0)) + float(origin_c)
                t1 = float(event.get("t1", float(event.get("t0", 0)) + arr.shape[0])) + float(origin_t)
                c1 = float(event.get("c1", float(event.get("c0", 0)) + arr.shape[1])) + float(origin_c)
            except Exception:
                return None
            payload.update({"mask": np.ascontiguousarray(arr), "t0": t0, "c0": c0, "t1": t1, "c1": c1})
            return "mask", payload

        return None

    def _overlay_monitor_events(
        self,
        events: list[dict],
        *,
        fallback_label: str,
        origin_t: int = 0,
        origin_c: int = 0,
        save: bool = True,
    ) -> int:
        added = 0
        for event in events:
            normalized = self._normalize_monitor_event(
                event,
                fallback_label=fallback_label,
                origin_t=origin_t,
                origin_c=origin_c,
            )
            if normalized is None:
                continue
            geometry_type, payload = normalized
            if save:
                self._save_monitor_result_event(geometry_type, payload)
            self.add_monitor_overlay(geometry_type, payload)
            added += 1
        return added

class MonitorCallbackMixin:
    """Own monitor batch/progress/finished/failed callbacks."""

    def _on_monitor_batch(self, events: list[dict], label: str, run_token: int, streamed: bool = True):
        if int(run_token) != int(self._active_monitor_run_token):
            return
        if not events:
            return
        context = self._monitor_run_contexts.get(int(run_token), {})
        if context.get("realtime"):
            scale_x = float(context.get("scale_x", 1.0)) or 1.0
            scale_y = float(context.get("scale_y", 1.0)) or 1.0
            if scale_x != 1.0 or scale_y != 1.0:
                events = scale_event_coords(list(events), scale_x=scale_x, scale_y=scale_y)
            bounds = self.disp_model.current_view_bounds()
            allow_future = bool(context.get("allow_future_events", False))
            count = self._realtime_add_events(
                list(events),
                label=label,
                origin_t=int(context.get("origin_t", 0)),
                origin_c=int(context.get("origin_c", 0)),
                owned_t0=context.get("owned_t0"),
                owned_t1=context.get("owned_t1"),
                visible_t0=int(bounds["t0"]) if bounds is not None else None,
                visible_t1=None if allow_future else (int(bounds["t1"]) if bounds is not None else None),
            )
            total_rt = len(self._realtime_uid_ct)
            if count > 0:
                self.detect_window.set_monitor_status(
                    f"Realtime {label}: {total_rt} detection(s) on screen."
                )
            return
        count = self._overlay_monitor_events(
            list(events),
            fallback_label=label,
            origin_t=int(context.get("origin_t", 0)),
            origin_c=int(context.get("origin_c", 0)),
        )
        if count > 0:
            self.detect_window.set_monitor_status(
                f"Running {label} on {context.get('scope_label', 'source')}… showing {len(self._monitor_overlay_uids)} detection(s)."
            )
        _ = streamed

    def _on_monitor_progress(self, done: int, total: int, run_token: int) -> None:
        if int(run_token) != int(self._active_monitor_run_token):
            return
        context = self._monitor_run_contexts.setdefault(int(run_token), {})
        context["done"] = int(done)
        context["total"] = int(total)
        self.detect_window.set_monitor_progress(done, total)
        self.detect_window.set_monitor_status(
            f"Running {context.get('scope_label', 'source')} monitor… {int(done)}/{int(total)} patch(es) processed."
        )

    def _on_monitor_finished(self, events: list[dict], label: str, run_token: int, streamed: bool, cancelled: bool):
        """Overlay monitor results on the main canvas."""
        context = self._monitor_run_contexts.pop(int(run_token), {})
        if int(run_token) != int(self._active_monitor_run_token):
            return

        # ── Realtime incremental path ──────────────────────────────────
        if context.get("realtime"):
            scale_x = float(context.get("scale_x", 1.0)) or 1.0
            scale_y = float(context.get("scale_y", 1.0)) or 1.0
            if streamed:
                events = []
            # For display-driven mode (scale_x > 1), detected positions are in
            # view-column units; scale them back to source-sample space.
            if scale_x != 1.0 or scale_y != 1.0:
                events = scale_event_coords(list(events), scale_x=scale_x, scale_y=scale_y)
            bounds = self.disp_model.current_view_bounds()
            allow_future = bool(context.get("allow_future_events", False))
            count = self._realtime_add_events(
                list(events),
                label=label,
                origin_t=int(context.get("origin_t", 0)),
                origin_c=int(context.get("origin_c", 0)),
                owned_t0=context.get("owned_t0"),
                owned_t1=context.get("owned_t1"),
                visible_t0=int(bounds["t0"]) if bounds is not None else None,
                visible_t1=None if allow_future else (int(bounds["t1"]) if bounds is not None else None),
            )
            # Update covered range now that this strip is done.
            nct0 = context.get("new_covered_t0")
            nct1 = context.get("new_covered_t1")
            ncc0 = context.get("new_covered_c0")
            ncc1 = context.get("new_covered_c1")
            if nct0 is not None and nct1 is not None:
                self._realtime_covered = {
                    "t0": int(nct0), "t1": int(nct1),
                    "c0": int(ncc0 or 0), "c1": int(ncc1 or 0),
                }
            total_rt = len(self._realtime_uid_ct)
            self.detect_window.set_monitor_status(
                f"Realtime {label}: {total_rt} detection(s) on screen."
                if total_rt else f"Realtime {label} active — no detections in view."
            )
            return

        # ── Normal (batch) path ────────────────────────────────────────
        if streamed:
            self._clear_monitor_overlays()
        count = self._overlay_monitor_events(
            list(events),
            fallback_label=label,
            origin_t=int(context.get("origin_t", 0)),
            origin_c=int(context.get("origin_c", 0)),
            save=not bool(streamed),
        )
        self.detect_window.set_monitor_busy(False)
        self._finish_monitor_result_writer(cancelled=bool(cancelled))
        total = int(context.get("total", 0) or 0)
        done = int(context.get("done", total) or total)
        if cancelled:
            self.detect_window.set_monitor_progress(done if total > 0 else None, total if total > 0 else None)
            self.detect_window.set_monitor_status(
                f"{label} cancelled after {done}/{total or '?'} patch(es). {count} detection(s) kept on canvas."
            )
        elif count == 0:
            self.detect_window.set_monitor_progress(done if total > 0 else 0, total if total > 0 else 1)
            self.detect_window.set_monitor_status(
                f"{label} finished on {context.get('scope_label', 'source')}: no events detected."
            )
        else:
            self.detect_window.set_monitor_progress(done if total > 0 else 1, total if total > 0 else 1)
            self.detect_window.set_monitor_status(
                f"{label} finished on {context.get('scope_label', 'source')}: {count} detection(s) shown on canvas."
            )

    def _on_monitor_failed(self, message: str, run_token: int) -> None:
        self._monitor_run_contexts.pop(int(run_token), None)
        if int(run_token) != int(self._active_monitor_run_token):
            return
        self._clear_monitor_overlays()
        self.detect_window.set_monitor_busy(False)
        self._finish_monitor_result_writer(cancelled=True)
        self.detect_window.set_monitor_progress(0, 1)
        self.detect_window.set_monitor_status(f"Monitor failed: {message}", error=True)
        QMessageBox.warning(self, "Monitor Failed", message)

class MonitorControllerMixin(
    MonitorCoreMixin,
    RealtimeMonitorOverlayMixin,
    DisplayRealtimeMonitorMixin,
    MonitorOverlayMixin,
    MonitorCallbackMixin,
):
    """Compose monitor execution, realtime modes, and detection overlays."""
