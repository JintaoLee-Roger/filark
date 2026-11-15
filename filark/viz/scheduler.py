# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""Scheduler that keeps the GPU ring buffer aligned with the current viewport.

Two execution modes, selected at construction time
--------------------------------------------------
``async_io=False`` (default)
    All operations run synchronously on the main thread.  The original
    blocking behaviour — no blank areas, predictable timing.

``async_io=True``
    A dedicated ``filark-io`` daemon thread owns all ``source.read()`` calls.
    The main (VisPy) thread owns all GPU operations.  A 120 Hz flush timer
    drains the result queue and applies GPU updates.

    Blank-area prevention
    ~~~~~~~~~~~~~~~~~~~~~
    Only one horizontal push request is in-flight at a time
    (tracked by ``_PendingPush``).  The margin check uses ``self.t0``
    (last *applied* origin), not an optimistic pending state.  When the
    camera approaches within ``_blank_guard`` samples of the actual buffer
    edge and a push is still in-flight, the flush timer blocks briefly on
    the result queue (≤ ``_BLANK_WAIT_S`` seconds) rather than letting the
    camera expose an unloaded region.

    Architecture note
    ~~~~~~~~~~~~~~~~~
    The threading infrastructure (worker, request/result queues) properly
    belongs to the IO layer.  It lives here as a pragmatic convenience;
    future work should move it to ``io/async_source.py`` and expose a
    non-blocking ``read_async(callback)`` interface.
"""

import queue
import threading
from dataclasses import dataclass
from math import floor

import numpy as np
from vispy import app
from vispy.visuals.transforms import STTransform

from filark.io.streaming import StreamingSource
from filark.utils.perf import elapsed_ms, log_perf, next_perf_id, perf_debug_enabled, perf_now
from ._replay_cache import StatefulReplayCache2D
from .visual import StreamingImageVisual
from .camera import StreamingCamera


# ---------------------------------------------------------------------------
# Async IO — request / result dataclasses
# ---------------------------------------------------------------------------

_PRIO_RESET    = 0
_PRIO_PUSH     = 1
_PRIO_PREFETCH = 2

# Max time to block main thread waiting for an in-flight push result
# before the camera would expose a blank area.
_BLANK_WAIT_S = 0.15


@dataclass
class _IOReq:
    """One IO operation submitted to the worker thread."""
    kind: str        # 'reset' | 'push_right' | 'prefetch_right' | 'push_left' | 'push_down' | 'push_up'
    gen: int
    read_c0: int
    read_t0: int
    read_h: int
    read_w: int
    result_t0: int   # self.t0 after this op is applied
    result_c0: int   # self.c0 after this op is applied
    expected_t0: int # pre-condition on self.t0 (-1 = skip)
    expected_c0: int # pre-condition on self.c0 (-1 = skip)
    stateful: bool = False
    reason: str = ""


@dataclass
class _IOResult:
    """Completed IO result, waiting to be applied on the main thread."""
    kind: str
    gen: int
    block: object       # np.ndarray | None
    result_t0: int
    result_c0: int
    expected_t0: int
    expected_c0: int
    dx: int = 0
    fetch_ms: float = 0.0
    filter_ms: float = 0.0


@dataclass(frozen=True)
class ChunkEvent2D:
    """A processed display chunk that has just become available to the viewer."""

    kind: str
    block: np.ndarray
    t0: int
    t1: int
    c0: int
    c1: int
    scale_x: float
    scale_y: float
    view_t0: int
    view_t1: int
    view_c0: int
    view_c1: int


@dataclass
class _SchedulerTelemetry:
    """Last applied IO/upload timing sample consumed by realtime stats."""

    fetch_ms: float = 0.0
    filter_ms: float = 0.0
    upload_ms: float = 0.0
    io_ms: float = 0.0
    dx: int = 0

    def record(
        self,
        *,
        fetch_ms: float = 0.0,
        filter_ms: float = 0.0,
        upload_ms: float = 0.0,
        dx: int = 0,
    ) -> None:
        self.fetch_ms = float(fetch_ms)
        self.filter_ms = float(filter_ms)
        self.upload_ms = float(upload_ms)
        self.io_ms = float(fetch_ms) + float(filter_ms)
        self.dx = int(dx)

    def clear(self) -> None:
        self.record()

    def snapshot(self, *, reset: bool = False) -> dict[str, float | int]:
        payload = {
            "fetch_ms": float(self.fetch_ms),
            "filter_ms": float(self.filter_ms),
            "upload_ms": float(self.upload_ms),
            "io_ms": float(self.io_ms),
            "dx": int(self.dx),
        }
        if reset:
            self.clear()
        return payload


@dataclass
class _PendingPush:
    """Applied-buffer push currently waiting for async IO."""

    kind: str | None = None

    @property
    def in_flight(self) -> bool:
        return self.kind is not None

    def start(self, kind: str) -> None:
        self.kind = str(kind)

    def clear(self, kind: str | None = None) -> None:
        if kind is not None and self.kind != kind:
            return
        self.kind = None


@dataclass
class _PendingReset:
    """Full reset target currently waiting for async IO."""

    t0: int | None = None
    c0: int | None = None

    @property
    def in_flight(self) -> bool:
        return self.t0 is not None and self.c0 is not None

    def start(self, t0: int, c0: int) -> None:
        self.t0 = int(t0)
        self.c0 = int(c0)

    def clear(self) -> None:
        self.t0 = None
        self.c0 = None


@dataclass(frozen=True)
class _PendingCameraState:
    """Camera state that should be committed only after a reset block uploads."""

    camera_rect: tuple[float, float, float] | None = None
    y_pan_window: tuple[float, float] | None = None

    @classmethod
    def from_values(cls, *, camera_rect=None, y_pan_window=None):
        rect = None
        window = None
        if camera_rect is not None:
            left, bottom, height = camera_rect
            rect = (float(left), float(bottom), float(height))
        if y_pan_window is not None:
            y0, y1 = y_pan_window
            window = (float(y0), float(y1))
        if rect is None and window is None:
            return None
        return cls(camera_rect=rect, y_pan_window=window)


@dataclass
class _PrefetchBlock:
    """Reusable async right-prefetch result."""

    expected_t0: int
    expected_c0: int
    read_t0: int
    read_c0: int
    block: np.ndarray
    fetch_ms: float = 0.0
    filter_ms: float = 0.0


class _AsyncReadWorker:
    """Small request/result pipe for source reads owned by BufferScheduler2D."""

    def __init__(self, *, execute, is_current) -> None:
        self._execute = execute
        self._is_current = is_current
        self._req_q: queue.PriorityQueue = queue.PriorityQueue()
        self._res_q: queue.SimpleQueue = queue.SimpleQueue()
        self._seq = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="filark-io",
        )
        self._thread.start()

    def submit(self, priority: int, req: _IOReq) -> None:
        self._seq += 1
        self._req_q.put_nowait((int(priority), self._seq, req))

    def get_nowait(self) -> _IOResult:
        return self._res_q.get_nowait()

    def get(self, *, timeout: float) -> _IOResult:
        return self._res_q.get(timeout=float(timeout))

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                _prio, _seq, req = self._req_q.get(timeout=0.05)
            except queue.Empty:
                continue
            if not self._is_current(req):
                continue
            self._res_q.put(self._execute(req))


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class BufferScheduler2D:
    """
    Streaming buffer scheduler for a 2D ring image.

    Maintains the world-space window origin (t0, c0) and keeps a
    StreamingImageVisual synchronised with camera X movement and explicit
    Y control.

    Design rules
    ------------
    - Camera drives X streaming only.
    - Y streaming is deterministic and API-driven (keyboard / GUI).
    - Mouse input never triggers Y push.
    - Image transform is always kept in sync.
    - When ``async_io=False`` (default): all IO on main thread, no blanks.
    - When ``async_io=True``: IO on worker thread, one push in-flight at a
      time, blank guard blocks briefly if camera nears unloaded edge.
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
        window_cache_blocks: int = 4,
        checkpoint_stride: int = 256,
        checkpoint_limit: int = 32,
        request_draw=None,
        async_io: bool = False,
        blank_guard: int = 128,
    ):
        self.source = source
        self.image = image
        self.camera = camera

        self._wrap_y = bool(wrap_y)
        self._async_io = bool(async_io)
        source_lock = getattr(self.source, "_filark_async_lock", None)
        if source_lock is None:
            source_lock = threading.RLock()
            try:
                setattr(self.source, "_filark_async_lock", source_lock)
            except Exception:
                pass
        self._source_lock = source_lock

        self.nc, self.nt = self.source.shape

        self._telemetry = _SchedulerTelemetry()

        self._chunk_listeners: list = []

        self.buffer_h = int(buffer_h)
        self.buffer_w = int(buffer_w)
        self.x_align  = int(max(1, x_align))
        self.y_align  = int(max(1, y_align))
        self.x_margin = int(max(0, x_margin))
        self.x_margin = min(self.x_margin, max(0, self.buffer_w // 2 - 1))
        self._reset_ratio = float(reset_ratio)
        self._blank_guard = int(max(0, blank_guard))

        self.request_draw = request_draw

        self._replay_cache = StatefulReplayCache2D(
            source=self.source,
            image=self.image,
            buffer_h=self.buffer_h,
            buffer_w=self.buffer_w,
            window_cache_blocks=window_cache_blocks,
            checkpoint_stride=checkpoint_stride,
            checkpoint_limit=checkpoint_limit,
        )

        self.t0 = 0
        self.c0 = 0

        if not isinstance(getattr(self.image, "transform", None), STTransform):
            self.image.transform = STTransform(translate=(0, 0))

        # ---- coalesce (rate-limit camera X events) ----
        self._enable_coalesce = bool(enable_coalesce)
        self._dirty = False
        self._pending_cam_t0 = None
        self._coalesce_timer = None
        self._coalesce_hz = int(max(1, coalesce_hz))

        # ---- async IO state (used only when async_io=True) ----
        self._gen: int = 0
        self._push = _PendingPush()
        self._reset = _PendingReset()
        self._prefetch_in_flight: bool = False
        self._prefetch_right: _PrefetchBlock | None = None
        self._clamping_camera: bool = False
        self._camera_event_suppression_depth: int = 0
        self._pending_reset_camera_state: _PendingCameraState | None = None
        self._pending_cam_c0: int | None = None
        self._worker: _AsyncReadWorker | None = None
        self._flush_timer: app.Timer | None = None

        # ---- synchronous init (before worker starts) ----
        with self._source_lock:
            self.source.reset_processing_state()
        self._reset_full_sync(self.t0, self.c0, reason="scheduler_init")

        # ---- start async machinery if requested ----
        if self._async_io:
            self._worker = _AsyncReadWorker(
                execute=self._worker_execute,
                is_current=lambda req: int(req.gen) == int(self._gen),
            )
            self._worker.start()
            self._flush_timer = app.Timer(
                interval=1.0 / 120.0,
                connect=self._on_flush_timer,
                start=True,
            )

        if self._enable_coalesce:
            self.start_coalesce(self._coalesce_hz)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _draw(self):
        if callable(self.request_draw):
            self.request_draw()

    def telemetry_snapshot(self, *, reset: bool = False) -> dict[str, float | int]:
        """Return the latest applied IO/upload timing sample."""
        return self._telemetry.snapshot(reset=reset)

    @property
    def coalesce_hz(self) -> int:
        return int(self._coalesce_hz)

    def _has_stateful_processing(self) -> bool:
        with self._source_lock:
            return self._replay_cache.has_stateful_processing()

    def _can_snapshot_processing(self) -> bool:
        with self._source_lock:
            return self._replay_cache.can_snapshot_processing()

    def _maybe_store_state_anchor(self, *, block=None):
        with self._source_lock:
            self._replay_cache.maybe_store_state_anchor(t0=self.t0, c0=self.c0, block=block)

    def _read_window_from_checkpoint(self, c0, t0, h=None, w=None, debug_context=None):
        with self._source_lock:
            return self._replay_cache.read_window(c0=c0, t0=t0, h=h, w=w,
                                                  debug_context=debug_context)

    def mutate_source(self, fn):
        """Run a source mutation while async reads are quiescent."""
        with self._source_lock:
            return fn(self.source)

    # ------------------------------------------------------------------
    # Coalesce control (camera X rate-limiting)
    # ------------------------------------------------------------------

    def start_coalesce(self, hz=60):
        hz = int(max(1, hz))
        self._coalesce_hz = hz
        self._enable_coalesce = True
        if self._coalesce_timer is None:
            self._coalesce_timer = app.Timer(
                interval=1.0 / float(hz),
                connect=self._on_coalesce_timer,
                start=True,
            )
        else:
            self._coalesce_timer.interval = 1.0 / float(hz)
            self._coalesce_timer.start()

    def stop_coalesce(self):
        self._enable_coalesce = False
        self._dirty = False
        self._pending_cam_t0 = None
        if self._coalesce_timer is not None:
            self._coalesce_timer.stop()

    def _on_coalesce_timer(self, event):
        self.flush_pending()

    def flush_pending(self):
        if (not self._enable_coalesce) or (not self._dirty):
            return
        self._dirty = False
        tgt = self.t0 if self._pending_cam_t0 is None else int(self._pending_cam_t0)
        self._pending_cam_t0 = None
        self._seek_x(tgt)

    # ------------------------------------------------------------------
    # Async IO — worker thread
    # ------------------------------------------------------------------

    def _submit_req(self, priority: int, req: _IOReq) -> None:
        if self._worker is None:
            return
        self._worker.submit(priority, req)

    def _worker_execute(self, req: _IOReq) -> _IOResult:
        with self._source_lock:
            t_io = perf_now()
            if req.kind in ("push_left", "push_up", "push_down"):
                self.source.reset_processing_state()
            if req.kind == "reset" and req.stateful:
                block = self.source.read_with_history(req.read_c0, req.read_t0,
                                                       req.read_h, req.read_w)
            else:
                block = self.source.read(req.read_c0, req.read_t0,
                                         req.read_h, req.read_w)
            total_ms = elapsed_ms(t_io)
            filter_ms = float(getattr(self.source, "_bench_last_filter_ms", 0.0))
        fetch_ms   = max(0.0, total_ms - filter_ms)
        return _IOResult(
            kind=req.kind,
            gen=req.gen,
            block=block,
            result_t0=req.result_t0,
            result_c0=req.result_c0,
            expected_t0=req.expected_t0,
            expected_c0=req.expected_c0,
            dx=req.read_w if req.kind in ("push_right", "prefetch_right") else 0,
            fetch_ms=fetch_ms,
            filter_ms=filter_ms,
        )

    # ------------------------------------------------------------------
    # Async IO — flush timer (main thread, 120 Hz)
    # ------------------------------------------------------------------

    def _on_flush_timer(self, event) -> None:
        changed = self._drain_result_queue()

        # Blank-area guard: if camera is near the actual (applied) buffer
        # edge and a push is still in-flight, block briefly.
        if self._push.in_flight and self._push.kind in ("push_left", "push_right"):
            kind = self._push.kind
            try:
                view_left = float(self.camera.rect.left)
                view_right = float(self.camera.rect.right)
            except Exception:
                view_left = 0.0
                view_right = 0.0
            buf_left = self.t0
            buf_right = self.t0 + self.buffer_w
            should_wait = (
                (kind == "push_left" and view_left < buf_left + self._blank_guard)
                or (kind == "push_right" and view_right > buf_right - self._blank_guard)
            )
            if should_wait:
                try:
                    if self._worker is None:
                        return
                    result = self._worker.get(timeout=_BLANK_WAIT_S)
                    if self._apply_result(result):
                        changed = True
                except queue.Empty:
                    pass

        if changed:
            self._draw()

    def _drain_result_queue(self) -> bool:
        changed = False
        if self._worker is None:
            return changed
        while True:
            try:
                result = self._worker.get_nowait()
            except queue.Empty:
                break
            if self._apply_result(result):
                changed = True
        return changed

    def _apply_result(self, result: _IOResult) -> bool:
        if result.gen != self._gen:
            return False
        if result.block is None:
            if result.kind == "prefetch_right":
                self._clear_prefetch()
            elif result.kind == "reset":
                self._pending_reset_camera_state = None
                self._clear_pending_reset()
            elif result.kind.startswith("push_"):
                self._clear_pending_push(result.kind)
            return False

        if result.kind != "reset":
            if result.expected_t0 >= 0 and result.expected_t0 != self.t0:
                if result.kind == "prefetch_right":
                    self._clear_prefetch()
                elif result.kind.startswith("push_"):
                    self._clear_pending_push(result.kind)
                return False
            if result.expected_c0 >= 0 and result.expected_c0 != self.c0:
                if result.kind == "prefetch_right":
                    self._clear_prefetch()
                elif result.kind.startswith("push_"):
                    self._clear_pending_push(result.kind)
                return False

        t_upload = perf_now()

        if result.kind == "reset":
            self._apply_block(result.block, result.result_t0, result.result_c0)
            self._apply_pending_reset_camera_state()
            self._maybe_store_state_anchor(block=result.block)
            self._fire_chunk_ready("reset", result.block, result.result_t0, result.result_c0)
            self._telemetry.record(
                fetch_ms=result.fetch_ms,
                filter_ms=result.filter_ms,
                upload_ms=elapsed_ms(t_upload),
            )
            self._clear_pending_reset()
            self._resume_pending_seeks()

        elif result.kind == "push_right":
            dx = result.dx
            self._fire_chunk_ready("push_right", result.block, self.t0 + self.buffer_w, self.c0)
            self.image.push_right(result.block)
            self.t0 = result.result_t0
            self._apply_translate()
            self._telemetry.record(
                fetch_ms=result.fetch_ms,
                filter_ms=result.filter_ms,
                upload_ms=elapsed_ms(t_upload),
                dx=dx,
            )
            self._maybe_store_state_anchor()
            self._clear_pending_push("push_right")
            # Proactive prefetch (low priority, best-effort)
            self._submit_async_prefetch_right()
            self._resume_pending_seeks()

        elif result.kind == "prefetch_right":
            self._prefetch_in_flight = False
            self._prefetch_right = _PrefetchBlock(
                expected_t0=int(result.expected_t0),
                expected_c0=int(result.expected_c0),
                read_t0=int(result.expected_t0 + self.buffer_w),
                read_c0=int(result.expected_c0),
                block=np.array(result.block, copy=True, order="C"),
                fetch_ms=float(result.fetch_ms),
                filter_ms=float(result.filter_ms),
            )

        elif result.kind == "push_left":
            dx = self.t0 - result.result_t0
            self._fire_chunk_ready("push_left", result.block, result.result_t0, self.c0)
            self.image.push_left(result.block)
            self.t0 = result.result_t0
            self._apply_translate()
            self._telemetry.record(
                fetch_ms=result.fetch_ms,
                filter_ms=result.filter_ms,
                upload_ms=elapsed_ms(t_upload),
                dx=dx,
            )
            self._maybe_store_state_anchor()
            self._clear_pending_push("push_left")
            self._resume_pending_seeks()

        elif result.kind == "push_down":
            self._fire_chunk_ready("push_down", result.block, self.t0, self.c0 + self.buffer_h)
            self.image.push_down(result.block)
            self.c0 = result.result_c0
            self._apply_translate()
            self._sync_camera_y_to_buffer()
            self._telemetry.record(
                fetch_ms=result.fetch_ms,
                filter_ms=result.filter_ms,
                upload_ms=elapsed_ms(t_upload),
            )
            self._clear_pending_push("push_down")
            self._resume_pending_seeks()

        elif result.kind == "push_up":
            self._fire_chunk_ready("push_up", result.block, self.t0, result.result_c0)
            self.image.push_up(result.block)
            self.c0 = result.result_c0
            self._apply_translate()
            self._sync_camera_y_to_buffer()
            self._telemetry.record(
                fetch_ms=result.fetch_ms,
                filter_ms=result.filter_ms,
                upload_ms=elapsed_ms(t_upload),
            )
            self._clear_pending_push("push_up")
            self._resume_pending_seeks()

        return True

    def _clear_pending_push(self, kind: str | None = None) -> None:
        self._push.clear(kind)

    def _start_pending_reset(self, t0: int, c0: int) -> None:
        self._reset.start(t0, c0)

    def _clear_pending_reset(self) -> None:
        self._reset.clear()

    def _clear_prefetch(self) -> None:
        self._prefetch_in_flight = False
        self._prefetch_right = None

    def _invalidate_async_reads(
        self,
        *,
        clear_pending_x: bool = True,
        clear_pending_y: bool = True,
    ) -> None:
        self._gen += 1
        self._clear_pending_push()
        self._clear_pending_reset()
        self._clear_prefetch()
        if clear_pending_x:
            self._pending_cam_t0 = None
        if clear_pending_y:
            self._pending_cam_c0 = None

    def begin_camera_event_suppression(self) -> None:
        self._camera_event_suppression_depth += 1

    def end_camera_event_suppression(self) -> None:
        self._camera_event_suppression_depth = max(0, self._camera_event_suppression_depth - 1)

    def _set_pending_reset_camera_state(self, *, camera_rect=None, y_pan_window=None) -> None:
        self._pending_reset_camera_state = _PendingCameraState.from_values(
            camera_rect=camera_rect,
            y_pan_window=y_pan_window,
        )

    def _apply_pending_reset_camera_state(self) -> None:
        state = self._pending_reset_camera_state
        self._pending_reset_camera_state = None
        if not state:
            return
        self.begin_camera_event_suppression()
        try:
            y_pan_window = state.y_pan_window
            if y_pan_window is not None:
                self.camera.set_y_pan_window(float(y_pan_window[0]), float(y_pan_window[1]))
            camera_rect = state.camera_rect
            if camera_rect is not None:
                left, bottom, height = camera_rect
                self.camera.apply_rect(left=float(left), bottom=float(bottom), height=float(height))
            self.camera.view_changed()
        finally:
            self.end_camera_event_suppression()

    def _sync_camera_y_to_buffer(self) -> None:
        self.begin_camera_event_suppression()
        try:
            self.camera.set_y_pan_window(self.c0, self.c0 + self.buffer_h)
            self.camera.set_y_bottom(self.c0)
            self.camera.view_changed()
        finally:
            self.end_camera_event_suppression()

    def _camera_rect_tuple(self):
        try:
            rect = self.camera.rect
            return (float(rect.left), float(rect.bottom), float(rect.height))
        except Exception:
            return None

    def _camera_y_state_for_c0(self, c0: int):
        rect = self._camera_rect_tuple()
        if rect is None:
            return None, (int(c0), int(c0) + self.buffer_h)
        left, _bottom, height = rect
        return (float(left), float(c0), float(height)), (int(c0), int(c0) + self.buffer_h)

    def _resume_pending_seeks(self) -> None:
        if self._push.in_flight or self._reset.in_flight:
            return

        pending_c0 = self._pending_cam_c0
        self._pending_cam_c0 = None
        if pending_c0 is not None and int(pending_c0) != int(self.c0):
            self._seek_y(int(pending_c0))
            if self._push.in_flight or self._reset.in_flight:
                return

        pending_t0 = self._pending_cam_t0
        self._pending_cam_t0 = None
        if pending_t0 is not None and int(pending_t0) != int(self.t0):
            self._seek_x(int(pending_t0))

    def _submit_async_prefetch_right(self) -> None:
        """Submit a low-priority read for the next chunk (best-effort).

        Uses x_margin as the prefetch size: it matches the typical per-push
        chunk size triggered when the camera approaches the buffer edge.
        """
        if self._has_stateful_processing():
            return
        if self._prefetch_in_flight:
            return
        if self._prefetch_right is not None:
            if (
                int(self._prefetch_right.expected_t0) == int(self.t0)
                and int(self._prefetch_right.expected_c0) == int(self.c0)
            ):
                return
            self._prefetch_right = None
        max_dx = self._view_nt() - (self.t0 + self.buffer_w)
        if max_dx <= 0:
            return
        dx = min(max(self.x_align, self.x_margin), max_dx)
        req = _IOReq(
            kind="prefetch_right",
            gen=self._gen,
            read_c0=self.c0,
            read_t0=self.t0 + self.buffer_w,
            read_h=self.buffer_h,
            read_w=dx,
            result_t0=self.t0 + dx,
            result_c0=self.c0,
            expected_t0=self.t0,
            expected_c0=self.c0,
        )
        self._prefetch_in_flight = True
        self._submit_req(_PRIO_PREFETCH, req)

    def add_chunk_listener(self, fn) -> None:
        if fn not in self._chunk_listeners:
            self._chunk_listeners.append(fn)

    def remove_chunk_listener(self, fn) -> bool:
        try:
            self._chunk_listeners.remove(fn)
            return True
        except ValueError:
            return False

    def _fire_chunk_ready(self, kind: str, block, view_t0: int, view_c0: int):
        if not self._chunk_listeners:
            return
        try:
            sx = float(getattr(self.source, "scale_x", 1.0)) or 1.0
            sy = float(getattr(self.source, "scale_y", 1.0)) or 1.0
            h = int(block.shape[0])
            w = int(block.shape[1])
            event = ChunkEvent2D(
                kind=str(kind),
                block=np.asarray(block),
                t0=int(view_t0 * sx),
                t1=int((view_t0 + w) * sx),
                c0=int(view_c0 * sy),
                c1=int((view_c0 + h) * sy),
                scale_x=float(sx),
                scale_y=float(sy),
                view_t0=int(view_t0),
                view_t1=int(view_t0 + w),
                view_c0=int(view_c0),
                view_c1=int(view_c0 + h),
            )
            for fn in tuple(self._chunk_listeners):
                try:
                    fn(event)
                except Exception:
                    pass
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Clamp + align
    # ------------------------------------------------------------------

    def _view_nt(self) -> int:
        sx = max(1e-6, float(getattr(self.source, "scale_x", 1.0)))
        return max(1, int(np.ceil(float(self.nt) / sx)))

    def _view_nc(self) -> int:
        if str(getattr(self.source, "y_mode", "local")) == "fit_all":
            return int(self.buffer_h)
        sy = max(1e-6, float(getattr(self.source, "scale_y", 1.0)))
        return max(1, int(np.ceil(float(self.nc) / sy)))

    def _clamp_t0(self, t0):
        view_nt = self._view_nt()
        if view_nt <= self.buffer_w:
            return 0
        return int(max(0, min(int(t0), view_nt - self.buffer_w)))

    def _clamp_c0(self, c0):
        view_nc = self._view_nc()
        if (not self._wrap_y) or (view_nc <= self.buffer_h):
            return 0
        return int(max(0, min(int(c0), view_nc - self.buffer_h)))

    @staticmethod
    def align_start(raw, align, start_min, start_max):
        raw = float(raw)
        start_min = float(start_min)
        start_max = float(start_max)
        if start_max <= start_min:
            return int(start_min)
        raw = max(start_min, min(raw, start_max))
        a = int(max(1, align))
        if a == 1:
            return int(round(raw))
        if raw >= start_max - 0.5 * a:
            return int(start_max)
        return int(floor(raw / a) * a)

    # ------------------------------------------------------------------
    # Read helpers (sync, main thread only)
    # ------------------------------------------------------------------

    def _read_window(self, c0, t0, h=None, w=None):
        h = self.buffer_h if h is None else int(h)
        w = self.buffer_w if w is None else int(w)
        with self._source_lock:
            return self.source.read(int(c0), int(t0), h, w)

    def _read_window_with_history(self, c0, t0, h=None, w=None):
        h = self.buffer_h if h is None else int(h)
        w = self.buffer_w if w is None else int(w)
        with self._source_lock:
            if self._has_stateful_processing():
                return self.source.read_with_history(int(c0), int(t0), h, w)
            return self.source.read(int(c0), int(t0), h, w)

    def _debug_context(self, op: str, **fields) -> dict:
        ctx = {
            "trace": next_perf_id("sched"),
            "op": str(op),
            "buffer_h": int(self.buffer_h),
            "buffer_w": int(self.buffer_w),
            "stateful": int(self._has_stateful_processing()),
            "wrap_y": int(self._wrap_y),
        }
        ctx.update(fields)
        return ctx

    def _apply_translate(self):
        self.image.transform.translate = (float(self.t0), float(self.c0))

    def _apply_block(self, block, t0: int, c0: int) -> None:
        self.t0 = int(t0)
        self.c0 = int(c0)
        self.image.set_data(block, reset_base=True, keep_view=True)
        self._apply_translate()

    # ------------------------------------------------------------------
    # Synchronous full reset (init, checkpoint hit, or sync mode reset)
    # ------------------------------------------------------------------

    def _reset_full_sync(
        self,
        t0=None,
        c0=None,
        *,
        reason: str = "reset_full_sync",
        camera_rect=None,
        y_pan_window=None,
    ):
        perf_on = perf_debug_enabled()
        total_start = perf_now() if perf_on else 0.0
        t0 = self.t0 if t0 is None else int(t0)
        c0 = self.c0 if c0 is None else int(c0)
        self._set_pending_reset_camera_state(camera_rect=camera_rect, y_pan_window=y_pan_window)
        t0 = self._clamp_t0(t0)
        c0 = self._clamp_c0(c0)

        ctx = self._debug_context("reset_full_sync", reason=reason, t0=t0, c0=c0)

        t_io = perf_now()
        block = self._read_window_from_checkpoint(c0, t0, debug_context=ctx)
        read_mode = "checkpoint"
        if block is None:
            read_mode = "source"
            with self._source_lock:
                if self._has_stateful_processing():
                    block = self.source.read_with_history(c0, t0, self.buffer_h,
                                                           self.buffer_w, debug_context=ctx)
                else:
                    block = self.source.read(c0, t0, self.buffer_h,
                                              self.buffer_w, debug_context=ctx)
                filter_ms = float(getattr(self.source, "_bench_last_filter_ms", 0.0))
        else:
            filter_ms = 0.0
        total_io_ms = elapsed_ms(t_io)
        fetch_ms = max(0.0, total_io_ms - filter_ms)
        if block is None:
            self._pending_reset_camera_state = None
            self._telemetry.clear()
            if perf_on:
                log_perf("scheduler.reset_full_sync", **ctx, read_mode=read_mode,
                         result="none", ms=elapsed_ms(total_start))
            return

        t_upload = perf_now()
        self._apply_block(block, t0, c0)
        self._apply_pending_reset_camera_state()
        self._maybe_store_state_anchor(block=block)
        self._fire_chunk_ready("reset", block, t0, c0)
        self._telemetry.record(
            fetch_ms=fetch_ms,
            filter_ms=filter_ms,
            upload_ms=elapsed_ms(t_upload),
        )
        self._draw()
        if perf_on:
            log_perf("scheduler.reset_full_sync", **ctx, read_mode=read_mode,
                     result="ok", ms=elapsed_ms(total_start))

    # ------------------------------------------------------------------
    # Full reset (sync or async depending on mode)
    # ------------------------------------------------------------------

    def reset_full(
        self,
        t0=None,
        c0=None,
        *,
        reason: str = "reset_full",
        camera_rect=None,
        y_pan_window=None,
    ):
        t0 = self._clamp_t0(self.t0 if t0 is None else int(t0))
        c0 = self._clamp_c0(self.c0 if c0 is None else int(c0))

        if not self._async_io:
            # Sync mode: read on main thread directly.
            self._reset_full_sync(
                t0,
                c0,
                reason=reason,
                camera_rect=camera_rect,
                y_pan_window=y_pan_window,
            )
            return

        # Async mode: invalidate all pending ops, fast-path checkpoint.
        self._invalidate_async_reads()
        self._set_pending_reset_camera_state(camera_rect=camera_rect, y_pan_window=y_pan_window)
        new_gen = self._gen

        ctx = self._debug_context("reset_full", reason=reason, t0=t0, c0=c0)
        block = self._read_window_from_checkpoint(c0, t0, debug_context=ctx)
        if block is not None:
            self._apply_block(block, t0, c0)
            self._apply_pending_reset_camera_state()
            self._maybe_store_state_anchor(block=block)
            self._fire_chunk_ready("reset", block, t0, c0)
            self._draw()
            return

        # Submit async reset (highest priority).
        self._start_pending_reset(t0, c0)
        req = _IOReq(
            kind="reset",
            gen=new_gen,
            read_c0=c0, read_t0=t0,
            read_h=self.buffer_h, read_w=self.buffer_w,
            result_t0=t0, result_c0=c0,
            expected_t0=-1, expected_c0=-1,
            stateful=self._has_stateful_processing(),
            reason=reason,
        )
        self._submit_req(_PRIO_RESET, req)

    # ------------------------------------------------------------------
    # Camera callback (X only)
    # ------------------------------------------------------------------

    def on_camera_changed(self, cam: StreamingCamera):
        if self._clamping_camera or self._camera_event_suppression_depth > 0:
            return
        view_left  = float(cam.rect.left)
        view_right = float(cam.rect.right)

        view_nt = self._view_nt()
        if view_nt <= self.buffer_w:
            return

        # Always use the *applied* (not optimistic) state for margin check.
        # This prevents the camera from outrunning the loaded buffer.
        buf_left   = self.t0
        buf_right  = self.t0 + self.buffer_w
        safe_left  = buf_left  + self.x_margin
        safe_right = buf_right - self.x_margin

        if safe_left <= view_left and view_right <= safe_right:
            return

        view_center = 0.5 * (view_left + view_right)
        raw = view_center - self.buffer_w / 2.0
        tgt = self.align_start(raw, self.x_align, 0, view_nt - self.buffer_w)

        if self._async_io and self._reset.in_flight:
            self._pending_cam_t0 = int(tgt)
            return

        if self._enable_coalesce:
            self._pending_cam_t0 = int(tgt)
            self._dirty = True
            return

        if self._async_io and self._push.in_flight:
            self._clamp_camera_to_loaded_x(view_left=view_left, view_right=view_right)
            self._pending_cam_t0 = int(tgt)
            return

        self._seek_x(int(tgt))

    def _clamp_camera_to_loaded_x(self, *, view_left: float, view_right: float) -> None:
        width = max(1.0, float(view_right - view_left))
        buf_left = float(self.t0)
        buf_right = float(self.t0 + self.buffer_w)
        if width >= float(self.buffer_w):
            new_left = buf_left
        else:
            new_left = min(max(float(view_left), buf_left), buf_right - width)
        if abs(new_left - float(view_left)) <= 1e-6:
            return
        try:
            rect = self.camera.rect
            self._clamping_camera = True
            self.camera.apply_rect(left=new_left, bottom=float(rect.bottom), height=float(rect.height))
        finally:
            self._clamping_camera = False

    # ------------------------------------------------------------------
    # Y API (GUI / keyboard)
    # ------------------------------------------------------------------

    def set_channel_start(self, c0, *, hard_reset=False):
        c0 = self._clamp_c0(int(c0))
        if hard_reset:
            self.reset_full(self.t0, c0)
        else:
            self._seek_y(c0)

    def step_channels(self, delta, *, hard_reset=False):
        self.set_channel_start(self.c0 + int(delta), hard_reset=hard_reset)

    def page_channels(self, direction, overlap=0.2, *, hard_reset=False):
        step = max(1, int(round(self.buffer_h * (1.0 - float(overlap)))))
        self.step_channels(step if direction > 0 else -step, hard_reset=hard_reset)

    # ------------------------------------------------------------------
    # Seek X
    # ------------------------------------------------------------------

    def _seek_x(self, tgt_t0: int) -> None:
        tgt_t0 = self._clamp_t0(tgt_t0)
        dx = int(tgt_t0 - self.t0)
        if dx == 0:
            return

        if abs(dx) >= int(self._reset_ratio * self.buffer_w) or abs(dx) >= self.buffer_w:
            camera_rect = self._camera_rect_tuple()
            if self._async_io and camera_rect is not None:
                try:
                    rect = self.camera.rect
                    self._clamp_camera_to_loaded_x(view_left=float(rect.left), view_right=float(rect.right))
                except Exception:
                    pass
            self.reset_full(tgt_t0, self.c0, reason="seek_x_big_jump", camera_rect=camera_rect)
            return

        if dx < 0 and self._has_stateful_processing():
            camera_rect = self._camera_rect_tuple()
            if self._async_io and camera_rect is not None:
                try:
                    rect = self.camera.rect
                    self._clamp_camera_to_loaded_x(view_left=float(rect.left), view_right=float(rect.right))
                except Exception:
                    pass
            self.reset_full(tgt_t0, self.c0, reason="seek_x_backward_stateful", camera_rect=camera_rect)
            return

        if self._async_io:
            if self._reset.in_flight:
                self._pending_cam_t0 = int(tgt_t0)
                return
            if dx > 0:
                if self._push.in_flight:
                    if self._push.kind == "push_right":
                        self._pending_cam_t0 = int(tgt_t0)
                        return
                    if self._push.kind not in ("push_left", "push_right"):
                        self._pending_cam_t0 = int(tgt_t0)
                        return
                    self._invalidate_async_reads(clear_pending_y=False)
                    dx = int(tgt_t0 - self.t0)
                    if dx <= 0:
                        return
                if self._try_apply_prefetch_right(dx):
                    self._draw()
                else:
                    self._submit_async_push_right(dx)
            else:
                self._clear_prefetch()
                if self._push.in_flight:
                    if self._push.kind == "push_left":
                        self._pending_cam_t0 = int(tgt_t0)
                        return
                    if self._push.kind not in ("push_left", "push_right"):
                        self._pending_cam_t0 = int(tgt_t0)
                        return
                    self._invalidate_async_reads(clear_pending_y=False)
                dx_actual = int(tgt_t0 - self.t0)
                if dx_actual < 0 and not self._push.in_flight:
                    self._submit_async_push_left(-dx_actual)
        else:
            if dx > 0:
                self._push_right(dx)
            else:
                self._push_left(-dx)
            self._draw()

    # ------------------------------------------------------------------
    # Seek Y
    # ------------------------------------------------------------------

    def _seek_y(self, tgt_c0: int) -> None:
        if not self._wrap_y:
            return
        tgt_c0 = self._clamp_c0(tgt_c0)
        if self.y_align > 1:
            tgt_c0 = self.align_start(tgt_c0, self.y_align, 0, self._view_nc() - self.buffer_h)
        dy = int(tgt_c0 - self.c0)
        if dy == 0:
            return
        if self._has_stateful_processing():
            camera_rect, y_pan_window = self._camera_y_state_for_c0(tgt_c0)
            self.reset_full(
                self.t0,
                tgt_c0,
                reason="seek_y_stateful",
                camera_rect=camera_rect,
                y_pan_window=y_pan_window,
            )
            return
        if abs(dy) >= int(self._reset_ratio * self.buffer_h) or abs(dy) >= self.buffer_h:
            camera_rect, y_pan_window = self._camera_y_state_for_c0(tgt_c0)
            self.reset_full(
                self.t0,
                tgt_c0,
                reason="seek_y_big_jump",
                camera_rect=camera_rect,
                y_pan_window=y_pan_window,
            )
            return

        if self._async_io:
            if self._reset.in_flight or self._push.in_flight:
                self._pending_cam_c0 = int(tgt_c0)
                return
            self._invalidate_async_reads(clear_pending_x=False)
            if dy > 0:
                self._submit_async_push_down(dy)
            else:
                self._submit_async_push_up(-dy)
        else:
            if dy > 0:
                self._push_down(dy)
            else:
                self._push_up(-dy)
            self._draw()

    # ------------------------------------------------------------------
    # Async push request builders
    # ------------------------------------------------------------------

    def _submit_async_push_right(self, dx: int) -> None:
        dx = min(int(dx), self._view_nt() - (self.t0 + self.buffer_w))
        if dx <= 0:
            return
        req = _IOReq(
            kind="push_right",
            gen=self._gen,
            read_c0=self.c0,
            read_t0=self.t0 + self.buffer_w,
            read_h=self.buffer_h,
            read_w=dx,
            result_t0=self.t0 + dx,
            result_c0=self.c0,
            expected_t0=self.t0,
            expected_c0=self.c0,
        )
        self._push.start("push_right")
        self._submit_req(_PRIO_PUSH, req)

    def _try_apply_prefetch_right(self, dx: int) -> bool:
        cache = self._prefetch_right
        if cache is None:
            return False
        if int(cache.expected_t0) != int(self.t0):
            self._prefetch_right = None
            return False
        if int(cache.expected_c0) != int(self.c0):
            self._prefetch_right = None
            return False
        block = np.asarray(cache.block)
        dx = int(dx)
        if dx <= 0 or block.ndim != 2 or block.shape[1] < dx:
            return False

        chunk = np.array(block[:, :dx], copy=True, order="C")
        self._fire_chunk_ready("push_right", chunk, self.t0 + self.buffer_w, self.c0)
        t_upload = perf_now()
        self.image.push_right(chunk)
        self.t0 += dx
        self._apply_translate()
        self._telemetry.record(
            fetch_ms=float(cache.fetch_ms),
            filter_ms=float(cache.filter_ms),
            upload_ms=elapsed_ms(t_upload),
            dx=dx,
        )
        self._maybe_store_state_anchor()

        if block.shape[1] > dx:
            self._prefetch_right = _PrefetchBlock(
                expected_t0=int(self.t0),
                expected_c0=int(self.c0),
                read_t0=int(self.t0 + self.buffer_w),
                read_c0=int(self.c0),
                block=np.array(block[:, dx:], copy=True, order="C"),
            )
        else:
            self._prefetch_right = None
            self._submit_async_prefetch_right()
        return True

    def _submit_async_push_left(self, dx: int) -> None:
        dx = min(int(dx), self.t0)
        if dx <= 0:
            return
        new_t0 = self.t0 - dx
        req = _IOReq(
            kind="push_left",
            gen=self._gen,
            read_c0=self.c0,
            read_t0=new_t0,
            read_h=self.buffer_h,
            read_w=dx,
            result_t0=new_t0,
            result_c0=self.c0,
            expected_t0=self.t0,
            expected_c0=self.c0,
        )
        self._push.start("push_left")
        self._submit_req(_PRIO_PUSH, req)

    def _submit_async_push_down(self, dy: int) -> None:
        dy = min(int(dy), self._view_nc() - (self.c0 + self.buffer_h))
        if dy <= 0:
            return
        req = _IOReq(
            kind="push_down",
            gen=self._gen,
            read_c0=self.c0 + self.buffer_h,
            read_t0=self.t0,
            read_h=dy,
            read_w=self.buffer_w,
            result_t0=self.t0,
            result_c0=self.c0 + dy,
            expected_t0=self.t0,
            expected_c0=self.c0,
        )
        self._push.start("push_down")
        self._submit_req(_PRIO_PUSH, req)

    def _submit_async_push_up(self, dy: int) -> None:
        dy = min(int(dy), self.c0)
        if dy <= 0:
            return
        new_c0 = self.c0 - dy
        req = _IOReq(
            kind="push_up",
            gen=self._gen,
            read_c0=new_c0,
            read_t0=self.t0,
            read_h=dy,
            read_w=self.buffer_w,
            result_t0=self.t0,
            result_c0=new_c0,
            expected_t0=self.t0,
            expected_c0=self.c0,
        )
        self._push.start("push_up")
        self._submit_req(_PRIO_PUSH, req)

    # ------------------------------------------------------------------
    # Sync push_right / push_left / push_down / push_up (sync mode)
    # ------------------------------------------------------------------

    def _push_right(self, dx):
        perf_on = perf_debug_enabled()
        total_start = perf_now() if perf_on else 0.0
        dx = int(dx)
        if dx <= 0:
            return
        max_dx = self._view_nt() - (self.t0 + self.buffer_w)
        dx = min(dx, max_dx)
        if dx <= 0:
            return

        with self._source_lock:
            t_io = perf_now()
            block = self.source.read(
                self.c0, self.t0 + self.buffer_w, self.buffer_h, dx,
                debug_context=self._debug_context("push_right", dx=dx, t0=self.t0, c0=self.c0),
            )
            total_io_ms = elapsed_ms(t_io)
            filter_ms = float(getattr(self.source, "_bench_last_filter_ms", 0.0))
        fetch_ms = max(0.0, total_io_ms - filter_ms)
        if block is None:
            self._telemetry.clear()
            return

        t_upload = perf_now()
        self._fire_chunk_ready("push_right", block, self.t0 + self.buffer_w, self.c0)
        self.image.push_right(block)
        self.t0 += dx
        self._apply_translate()
        self._telemetry.record(
            fetch_ms=fetch_ms,
            filter_ms=filter_ms,
            upload_ms=elapsed_ms(t_upload),
            dx=dx,
        )
        self._maybe_store_state_anchor()
        if perf_on:
            log_perf("scheduler.push_right", dx=dx, t0=self.t0, c0=self.c0,
                     ms=elapsed_ms(total_start))

    def _push_left(self, dx):
        perf_on = perf_debug_enabled()
        total_start = perf_now() if perf_on else 0.0
        dx = int(dx)
        if dx <= 0:
            return
        dx = min(dx, self.t0)
        if dx <= 0:
            return
        new_t0 = self.t0 - dx
        with self._source_lock:
            t_io = perf_now()
            self.source.reset_processing_state()
            block = self.source.read(
                self.c0, new_t0, self.buffer_h, dx,
                debug_context=self._debug_context("push_left", dx=dx, t0=new_t0, c0=self.c0),
            )
            total_io_ms = elapsed_ms(t_io)
            filter_ms = float(getattr(self.source, "_bench_last_filter_ms", 0.0))
        fetch_ms = max(0.0, total_io_ms - filter_ms)
        if block is None:
            self._telemetry.clear()
            return
        t_upload = perf_now()
        self._fire_chunk_ready("push_left", block, new_t0, self.c0)
        self.image.push_left(block)
        self.t0 = new_t0
        self._apply_translate()
        self._telemetry.record(
            fetch_ms=fetch_ms,
            filter_ms=filter_ms,
            upload_ms=elapsed_ms(t_upload),
            dx=dx,
        )
        if perf_on:
            log_perf("scheduler.push_left", dx=dx, t0=self.t0, c0=self.c0,
                     ms=elapsed_ms(total_start))

    def _push_down(self, dy):
        perf_on = perf_debug_enabled()
        total_start = perf_now() if perf_on else 0.0
        dy = int(dy)
        if dy <= 0:
            return
        dy = min(dy, self._view_nc() - (self.c0 + self.buffer_h))
        if dy <= 0:
            return
        with self._source_lock:
            t_io = perf_now()
            self.source.reset_processing_state()
            block = self.source.read(
                self.c0 + self.buffer_h, self.t0, dy, self.buffer_w,
                debug_context=self._debug_context("push_down", dy=dy, t0=self.t0,
                                                   c0=self.c0 + self.buffer_h),
            )
            total_io_ms = elapsed_ms(t_io)
            filter_ms = float(getattr(self.source, "_bench_last_filter_ms", 0.0))
        fetch_ms = max(0.0, total_io_ms - filter_ms)
        if block is None:
            self._telemetry.clear()
            return
        t_upload = perf_now()
        self._fire_chunk_ready("push_down", block, self.t0, self.c0 + self.buffer_h)
        self.image.push_down(block)
        self.c0 += dy
        self._apply_translate()
        self._telemetry.record(
            fetch_ms=fetch_ms,
            filter_ms=filter_ms,
            upload_ms=elapsed_ms(t_upload),
        )
        if perf_on:
            log_perf("scheduler.push_down", dy=dy, t0=self.t0, c0=self.c0,
                     ms=elapsed_ms(total_start))

    def _push_up(self, dy):
        perf_on = perf_debug_enabled()
        total_start = perf_now() if perf_on else 0.0
        dy = int(dy)
        if dy <= 0:
            return
        dy = min(dy, self.c0)
        if dy <= 0:
            return
        new_c0 = self.c0 - dy
        with self._source_lock:
            t_io = perf_now()
            self.source.reset_processing_state()
            block = self.source.read(
                new_c0, self.t0, dy, self.buffer_w,
                debug_context=self._debug_context("push_up", dy=dy, t0=self.t0, c0=new_c0),
            )
            total_io_ms = elapsed_ms(t_io)
            filter_ms = float(getattr(self.source, "_bench_last_filter_ms", 0.0))
        fetch_ms = max(0.0, total_io_ms - filter_ms)
        if block is None:
            self._telemetry.clear()
            return
        t_upload = perf_now()
        self._fire_chunk_ready("push_up", block, self.t0, new_c0)
        self.image.push_up(block)
        self.c0 = new_c0
        self._apply_translate()
        self._telemetry.record(
            fetch_ms=fetch_ms,
            filter_ms=filter_ms,
            upload_ms=elapsed_ms(t_upload),
        )
        if perf_on:
            log_perf("scheduler.push_up", dy=dy, t0=self.t0, c0=self.c0,
                     ms=elapsed_ms(total_start))

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def close(self):
        if self._coalesce_timer is not None:
            self._coalesce_timer.stop()
        if self._async_io:
            if self._flush_timer is not None:
                self._flush_timer.stop()
            if self._worker is not None:
                self._worker.stop()
