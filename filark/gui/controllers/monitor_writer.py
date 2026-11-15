"""Background writer for monitor results saved as annotation JSON."""

from __future__ import annotations

import copy
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from filark._exceptions import print_exception
from filark.utils.annotations import encode_annotation_data


_DEFAULT_LABEL_COLORS = (
    "#ffd54f",
    "#4fc3f7",
    "#ff8a65",
    "#81c784",
    "#ba68c8",
    "#f06292",
    "#aed581",
    "#7986cb",
)


class MonitorAnnotationWriter:
    """Append monitor detections to an annotation-compatible JSON file."""

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        source: dict | None,
        session: dict,
        flush_every: int = 16,
        flush_interval_s: float = 2.0,
        max_queue: int = 4096,
    ) -> None:
        self.path = Path(path).expanduser()
        self._source = _json_safe(source or {})
        self._session = _json_safe(session)
        self._flush_every = max(1, int(flush_every))
        self._flush_interval_s = max(0.1, float(flush_interval_s))
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=max(1, int(max_queue)))
        self._thread = threading.Thread(target=self._run, name="filark-monitor-writer", daemon=True)
        self._labels: set[str] = set()
        self._label_styles: dict[str, dict] = {}
        self._next_uid = 1
        self._items_end_pos = 0
        self._has_items = False
        self._dropped = 0
        self._drop_lock = threading.Lock()
        self._error: str | None = None
        self._closed = False
        self._thread.start()

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def item_count(self) -> int:
        return int(self._next_uid - 1)

    def enqueue(self, anno_type: str, payload: dict) -> None:
        if self._closed:
            return
        try:
            self._queue.put_nowait(("event", (str(anno_type), dict(payload))))
        except queue.Full:
            with self._drop_lock:
                self._dropped += 1

    def close(self, *, cancelled: bool = False, timeout_s: float = 3.0) -> dict:
        if not self._closed:
            self._closed = True
            try:
                self._queue.put(("close", {"cancelled": bool(cancelled)}), timeout=max(0.1, float(timeout_s)))
            except queue.Full:
                with self._drop_lock:
                    self._dropped += 1
        self._thread.join(max(0.1, float(timeout_s)))
        with self._drop_lock:
            dropped = int(self._dropped)
        return {
            "path": str(self.path),
            "items": int(self.item_count),
            "dropped": dropped,
            "error": self._error,
            "alive": bool(self._thread.is_alive()),
        }

    def _run(self) -> None:
        pending_items: list[dict] = []
        last_flush = time.perf_counter()
        closing = False
        cancelled = False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._write_initial_file(cancelled=False)
            while not closing:
                try:
                    kind, payload = self._queue.get(timeout=self._flush_interval_s)
                except queue.Empty:
                    if pending_items:
                        self._flush_items(pending_items, cancelled=cancelled)
                        pending_items = []
                        last_flush = time.perf_counter()
                    continue

                if kind == "event":
                    anno_type, event_payload = payload
                    item = self._build_item(anno_type, event_payload)
                    if item is not None:
                        pending_items.append(item)
                elif kind == "close":
                    cancelled = bool(payload.get("cancelled", False))
                    closing = True

                now = time.perf_counter()
                if len(pending_items) >= self._flush_every or (pending_items and now - last_flush >= self._flush_interval_s):
                    self._flush_items(pending_items, cancelled=cancelled)
                    pending_items = []
                    last_flush = now
            self._flush_items(pending_items, cancelled=cancelled)
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            print_exception("Monitor annotation writer failed.", prefix="MonitorWriter")

    def _build_item(self, anno_type: str, payload: dict) -> dict | None:
        anno_type = str(anno_type)
        if anno_type not in ("bbox", "polyline", "mask"):
            return None
        label = str(payload.get("label") or payload.get("algorithm") or "detection").strip() or "detection"
        data = self._annotation_data(anno_type, payload)
        if data is None:
            return None
        style = self._label_styles.get(label)
        if style is None:
            color = str(payload.get("color") or _DEFAULT_LABEL_COLORS[len(self._label_styles) % len(_DEFAULT_LABEL_COLORS)])
            style = {"color": color, "visible": True}
            self._label_styles[label] = style
        self._labels.add(label)
        uid = int(self._next_uid)
        self._next_uid += 1
        return {
            "uid": uid,
            "label": label,
            "anno_type": anno_type,
            "data": encode_annotation_data(data),
        }

    def _annotation_data(self, anno_type: str, payload: dict) -> dict | None:
        base = {
            "visible": bool(payload.get("visible", True)),
            "reviewed": False,
            "confidence": _maybe_float(payload.get("score")),
            "score": _maybe_float(payload.get("score")),
            "algorithm": payload.get("algorithm"),
            "meta": _json_safe(payload.get("meta", {}) or {}),
        }
        base = {k: v for k, v in base.items() if v is not None}

        if anno_type == "bbox":
            try:
                base.update(
                    {
                        "t0": int(round(float(payload.get("t0", 0)))),
                        "t1": int(round(float(payload.get("t1", 0)))),
                        "c0": int(round(float(payload.get("c0", 0)))),
                        "c1": int(round(float(payload.get("c1", 0)))),
                    }
                )
            except Exception:
                return None
            return base

        if anno_type == "polyline":
            points = payload.get("points", []) or []
            if len(points) < 2:
                return None
            base["points"] = [(float(p[0]), float(p[1])) for p in points]
            return base

        mask = payload.get("mask")
        if mask is None:
            return None
        arr = np.asarray(mask)
        if arr.ndim != 2 or arr.size == 0:
            return None
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0)
            arr = np.asarray(np.rint(arr * 255.0), dtype=np.uint8)
        else:
            arr = np.asarray(arr, dtype=np.uint8)
        try:
            base.update(
                {
                    "t0": int(round(float(payload.get("t0", 0)))),
                    "t1": int(round(float(payload.get("t1", float(payload.get("t0", 0)) + arr.shape[0])))),
                    "c0": int(round(float(payload.get("c0", 0)))),
                    "c1": int(round(float(payload.get("c1", float(payload.get("c0", 0)) + arr.shape[1])))),
                    "mask": np.ascontiguousarray(arr),
                    "alpha": float(payload.get("alpha", 0.45)),
                    "threshold": float(payload.get("threshold", 0.0)),
                }
            )
        except Exception:
            return None
        return base

    def _base_payload(self, *, cancelled: bool) -> dict:
        session = dict(self._session)
        session["cancelled"] = bool(cancelled)
        session["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        return {
            "version": 1,
            "intent": {
                "mode": "bbox",
                "preset_label": "detection",
                "confirm_each": False,
            },
            "source": dict(self._source),
            "monitor_session": session,
        }

    def _footer(self, *, cancelled: bool) -> bytes:
        base = self._base_payload(cancelled=cancelled)
        tail = {
            "next_uid": int(self._next_uid),
            "labels": sorted(self._labels),
            "label_styles": {label: dict(style) for label, style in sorted(self._label_styles.items())},
        }
        session = base["monitor_session"]
        text = (
            "\n  ],\n"
            f"  \"next_uid\": {_json_compact(tail['next_uid'])},\n"
            f"  \"labels\": {_json_compact(tail['labels'])},\n"
            f"  \"label_styles\": {_json_compact(tail['label_styles'])},\n"
            f"  \"monitor_session\": {_json_compact(session)}\n"
            "}\n"
        )
        return text.encode("utf-8")

    def _write_initial_file(self, *, cancelled: bool) -> None:
        base = self._base_payload(cancelled=cancelled)
        text = (
            "{\n"
            "  \"version\": 1,\n"
            f"  \"intent\": {_json_compact(base['intent'])},\n"
            f"  \"source\": {_json_compact(base['source'])},\n"
            "  \"items\": ["
        )
        prefix = text.encode("utf-8")
        with self.path.open("wb") as file:
            file.write(prefix)
            self._items_end_pos = int(file.tell())
            file.write(self._footer(cancelled=cancelled))
            file.flush()

    def _flush_items(self, items: list[dict], *, cancelled: bool) -> None:
        with self.path.open("r+b") as file:
            file.seek(self._items_end_pos)
            chunk = b""
            if items:
                parts: list[str] = []
                if self._has_items:
                    parts.append(",\n")
                else:
                    parts.append("\n")
                for index, item in enumerate(items):
                    if index > 0:
                        parts.append(",\n")
                    parts.append("    ")
                    parts.append(_json_compact(item))
                chunk = "".join(parts).encode("utf-8")
                file.write(chunk)
                self._items_end_pos += len(chunk)
                self._has_items = True
            file.write(self._footer(cancelled=cancelled))
            file.truncate()
            file.flush()


def _maybe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return copy.deepcopy(str(value))


def _json_compact(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
