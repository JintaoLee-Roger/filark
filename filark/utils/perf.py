"""Lightweight runtime-switchable performance logging for FiLark."""

from __future__ import annotations

import os
import sys
import threading
import time
from itertools import count
from typing import Any


def _parse_bool_env(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


_ENABLED = _parse_bool_env(os.environ.get("FILARK_DEBUG_PERF"))
_LOCK = threading.Lock()
_SEQ = count(1)
_SESSION_START = time.perf_counter()
_FIELD_PRIORITY = {
    "total_ms": 0,
    "op_ms": 1,
    "result": 2,
    "reason": 3,
    "read_mode": 4,
    "cache_mode": 5,
    "apply_ms": 6,
    "cache_ms": 7,
    "slice_ms": 8,
    "preprocess_ms": 9,
    "downsample_ms": 10,
    "postprocess_ms": 11,
    "restore_ms": 12,
    "trace": 13,
    "op": 14,
}


def perf_debug_enabled() -> bool:
    return bool(_ENABLED)


def set_perf_debug(enabled: bool) -> None:
    global _ENABLED
    global _SESSION_START
    _ENABLED = bool(enabled)
    if _ENABLED:
        _SESSION_START = time.perf_counter()
    log_perf("perf.toggle", enabled=int(_ENABLED))


def perf_now() -> float:
    return time.perf_counter()


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - float(start)) * 1000.0


def next_perf_id(prefix: str = "perf") -> str:
    return f"{prefix}-{next(_SEQ)}"


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, bool):
        return "1" if value else "0"
    if value is None:
        return "none"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_format_value(v) for v in value) + "]"
    text = str(value)
    return text.replace("\n", "\\n").replace("\r", "\\r")


def log_perf(event: str, /, **fields: Any) -> None:
    if not _ENABLED:
        return
    now = time.perf_counter()
    payload_fields = dict(fields)
    if "ms" in payload_fields and "op_ms" not in payload_fields:
        payload_fields["op_ms"] = payload_fields.pop("ms")
    payload_fields.setdefault("total_ms", (now - _SESSION_START) * 1000.0)
    ordered_items = sorted(
        payload_fields.items(),
        key=lambda item: (_FIELD_PRIORITY.get(item[0], 1000), item[0]),
    )
    payload = " ".join(f"{key}={_format_value(value)}" for key, value in ordered_items)
    line = f"[FiLarkPerf] event={event}"
    if payload:
        line = f"{line} {payload}"
    with _LOCK:
        print(line, file=sys.stderr, flush=True)
