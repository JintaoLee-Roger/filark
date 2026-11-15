"""Qt workers used by GUI monitor execution paths."""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np
from PySide6.QtCore import QObject, Signal

from filark._exceptions import print_exception
from filark.io.tapeio import NpyTape


class MonitorWorker(QObject):
    """Background worker that runs a monitor plugin on a data source."""

    batch_ready = Signal(list, bool, int)
    progress = Signal(int, int, int)
    finished = Signal(list, bool, bool, int)
    failed = Signal(str, int)

    def __init__(self, plugin, source, params, run_token, owns_source):
        super().__init__()
        self._plugin = plugin
        self._source = source
        self._params = params
        self._run_token = int(run_token)
        self._owns_source = bool(owns_source)
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self):
        try:
            streamed = False

            def _emit_batch(batch_events):
                nonlocal streamed
                batch = list(batch_events or [])
                if not batch:
                    return
                streamed = True
                self.batch_ready.emit(batch, True, self._run_token)

            def _emit_progress(done, total):
                self.progress.emit(int(done), int(total), self._run_token)

            events = self._plugin.run(
                self._source,
                params=self._params,
                progress_cb=_emit_progress,
                cancel_cb=lambda: self._cancelled,
                result_cb=_emit_batch,
            )
            self.finished.emit(list(events or []), streamed, bool(self._cancelled), self._run_token)
        except Exception as exc:
            print_exception(
                f"Monitor plugin '{self._plugin.label}' failed in worker run().",
                prefix="MonitorWorker",
            )
            self.failed.emit(str(exc), self._run_token)
        finally:
            if self._owns_source and hasattr(self._source, "close"):
                try:
                    self._source.close()
                except Exception:
                    print_exception("Failed to close temporary monitor source.", prefix="MonitorWorker")


class RealtimeDisplayWorker(QObject):
    """Persistent worker for display-driven realtime monitor chunks."""

    batch_ready = Signal(list, bool, int)
    finished = Signal(list, bool, bool, int, dict)
    failed = Signal(str, int, dict)
    stopped = Signal()

    def __init__(self, plugin):
        super().__init__()
        self._plugin = plugin
        self._cond = threading.Condition()
        self._queue = deque()
        self._stopping = False

    def enqueue(self, task: dict, *, max_queue: int) -> tuple[list[int], int]:
        dropped: list[int] = []
        with self._cond:
            if self._stopping:
                return [int(task.get("run_token", -1))], len(self._queue)
            self._queue.append(dict(task))
            limit = max(1, int(max_queue))
            while len(self._queue) > limit:
                old = self._queue.popleft()
                dropped.append(int(old.get("run_token", -1)))
            self._cond.notify()
            return dropped, len(self._queue)

    def queue_size(self) -> int:
        with self._cond:
            return len(self._queue)

    def stop(self) -> None:
        with self._cond:
            self._stopping = True
            self._queue.clear()
            self._cond.notify_all()

    def run(self) -> None:
        try:
            while True:
                with self._cond:
                    while not self._queue and not self._stopping:
                        self._cond.wait()
                    if self._stopping and not self._queue:
                        break
                    task = self._queue.popleft()
                self._run_task(task)
        finally:
            self.stopped.emit()

    def _cancelled(self) -> bool:
        with self._cond:
            return bool(self._stopping)

    def _run_task(self, task: dict) -> None:
        run_token = int(task.get("run_token", -1))
        now = time.perf_counter()
        metrics = {
            "queue_wait_ms": max(0.0, (now - float(task.get("enqueued_at", now))) * 1000.0),
            "run_ms": 0.0,
            "events": 0,
        }
        started = time.perf_counter()
        src = None
        streamed = False
        try:
            block = np.asarray(task.get("block"), dtype=np.float32)
            data = block.T
            if data.ndim != 2 or data.size == 0:
                self.finished.emit([], False, bool(self._cancelled()), run_token, metrics)
                return

            src = NpyTape.from_array(
                data,
                dims="nt_nc",
                fs=float(task.get("fs", 1.0) or 1.0) / max(1.0, float(task.get("scale_x", 1.0) or 1.0)),
                dx=float(task.get("dx", 1.0) or 1.0),
                dx_unit=str(task.get("dx_unit", "m") or "m"),
            )
            src.monitor_scope = "realtime_display"
            src.monitor_event_kind = str(task.get("kind", "chunk") or "chunk")
            src.monitor_origin_t = int(task.get("t0", 0) or 0)
            src.monitor_origin_c = int(task.get("c0", 0) or 0)
            src.monitor_trigger_t0 = int(task.get("t0", 0) or 0)
            src.monitor_trigger_t1 = int(task.get("t1", 0) or 0)
            src.monitor_trigger_c0 = int(task.get("c0", 0) or 0)
            src.monitor_trigger_c1 = int(task.get("c1", 0) or 0)
            src.monitor_scale_x = float(task.get("scale_x", 1.0) or 1.0)
            src.monitor_scale_y = float(task.get("scale_y", 1.0) or 1.0)
            nt, nc = data.shape
            params = self._plugin.validate_params(
                dict(task.get("params", {}) or {}),
                fs=src.fs,
                dx=src.dx,
                nt=nt,
                nc=nc,
            )

            def _emit_batch(batch_events):
                nonlocal streamed
                batch = list(batch_events or [])
                if not batch:
                    return
                streamed = True
                self.batch_ready.emit(batch, True, run_token)

            events = self._plugin.run(
                src,
                params=params,
                progress_cb=None,
                cancel_cb=self._cancelled,
                result_cb=_emit_batch,
            )
            event_list = list(events or [])
            metrics["run_ms"] = (time.perf_counter() - started) * 1000.0
            metrics["events"] = len(event_list)
            self.finished.emit(event_list, streamed, bool(self._cancelled()), run_token, metrics)
        except Exception as exc:
            metrics["run_ms"] = (time.perf_counter() - started) * 1000.0
            print_exception(
                f"Realtime monitor plugin '{self._plugin.label}' failed in worker run().",
                prefix="RealtimeDisplayWorker",
            )
            self.failed.emit(str(exc), run_token, metrics)
        finally:
            if src is not None and hasattr(src, "close"):
                try:
                    src.close()
                except Exception:
                    print_exception("Failed to close realtime display monitor source.", prefix="RealtimeDisplayWorker")


def scale_event_coords(events: list[dict], *, scale_x: float, scale_y: float = 1.0) -> list[dict]:
    """Scale detected display coords back to source sample/channel space."""
    out = []
    for ev in events:
        ev = dict(ev)
        geometry_type = ev.get("geometry_type", "bbox")
        if geometry_type == "bbox":
            ev["t0"] = int(round(float(ev.get("t0", 0)) * scale_x))
            ev["t1"] = int(round(float(ev.get("t1", 0)) * scale_x))
            ev["c0"] = int(round(float(ev.get("c0", 0)) * scale_y))
            ev["c1"] = int(round(float(ev.get("c1", 0)) * scale_y))
        elif geometry_type == "polyline":
            ev["points"] = [
                (float(p[0]) * scale_x, float(p[1]) * scale_y)
                for p in (ev.get("points") or [])
            ]
        elif geometry_type == "mask":
            t0 = float(ev.get("t0", 0))
            c0 = float(ev.get("c0", 0))
            mask = np.asarray(ev.get("mask"), dtype=np.float32)
            ev["t0"] = int(round(t0 * scale_x))
            ev["c0"] = int(round(c0 * scale_y))
            ev["t1"] = int(round(float(ev.get("t1", t0 + mask.shape[0])) * scale_x))
            ev["c1"] = int(round(float(ev.get("c1", c0 + mask.shape[1])) * scale_y))
        out.append(ev)
    return out
