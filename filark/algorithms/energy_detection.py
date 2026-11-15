# Copyright (c) 2026 Kai Tong & Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

r"""Patch-based DAS energy detectors and monitor-friendly source scanning.

This module collects three deterministic patch detectors that were previously
implemented as research prototypes:

- relative energy
- STA / LTA
- TKEO

Each detector consumes a local patch :math:`P \in \mathbb{R}^{T \times C}` and
returns a score map, a binary mask, and local bbox events extracted from
connected components in the mask.

For GUI integration, ``EnergyDetectionMonitor`` wraps these patch detectors in a
monitor-compatible class that scans a full source in overlapping patches and
returns canvas-ready detection events.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Final, Iterable, Literal, Sequence

import numpy as np
from scipy.ndimage import find_objects, label, uniform_filter1d

DetectorMethod = Literal["energy", "stalta", "tkeo"]

__all__ = [
    "BBoxDetectionEvent",
    "BasePatchDetector",
    "RelativeEnergyDetector",
    "STALTADetector",
    "TKEODetector",
    "build_energy_patch_detector",
    "merge_bbox_events",
    "run_patch_detector_on_tape",
    "EnergyDetectionMonitor",
]


_DETECTOR_COLORS: Final[dict[str, str]] = {
    "energy": "#FF8A00",
    "stalta": "#00B8D9",
    "tkeo": "#FF4FB3",
}


@dataclass(frozen=True)
class BBoxDetectionEvent:
    """Detection bbox in global or patch-local sample/channel coordinates."""

    t0: int
    t1: int
    c0: int
    c1: int
    score: float
    scale: float = 1.0
    label: str = "event"
    color: str = "#FF8A00"
    geometry_type: str = "bbox"
    meta: dict = field(default_factory=dict)

    @property
    def center_t(self) -> float:
        return 0.5 * (float(self.t0) + float(self.t1))

    @property
    def center_c(self) -> float:
        return 0.5 * (float(self.c0) + float(self.c1))

    def shifted(self, *, dt: int = 0, dc: int = 0) -> "BBoxDetectionEvent":
        return replace(
            self,
            t0=int(self.t0 + dt),
            t1=int(self.t1 + dt),
            c0=int(self.c0 + dc),
            c1=int(self.c1 + dc),
        )

    def merged_with(self, other: "BBoxDetectionEvent") -> "BBoxDetectionEvent":
        meta = dict(self.meta)
        if other.meta:
            meta.update(other.meta)
        return BBoxDetectionEvent(
            t0=min(int(self.t0), int(other.t0)),
            t1=max(int(self.t1), int(other.t1)),
            c0=min(int(self.c0), int(other.c0)),
            c1=max(int(self.c1), int(other.c1)),
            score=max(float(self.score), float(other.score)),
            scale=float(self.scale if self.score >= other.score else other.scale),
            label=str(self.label or other.label),
            color=str(self.color or other.color),
            geometry_type="bbox",
            meta=meta,
        )

    def to_dict(self) -> dict:
        payload = {
            "geometry_type": "bbox",
            "t0": int(self.t0),
            "t1": int(self.t1),
            "c0": int(self.c0),
            "c1": int(self.c1),
            "score": float(self.score),
            "scale": float(self.scale),
            "label": str(self.label),
            "color": str(self.color),
        }
        if self.meta:
            payload["meta"] = dict(self.meta)
        return payload


def _as_patch(patch) -> np.ndarray:
    arr = np.asarray(patch, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"patch must be 2-D (T, C), got shape={arr.shape}")
    return arr


def _require_positive(name: str, value: int | float) -> float:
    value = float(value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return value


def _normalize_scales(values: int | Sequence[int], *, fallback: int) -> tuple[int, ...]:
    if isinstance(values, (int, np.integer)):
        out = [int(values)]
    else:
        out = [int(v) for v in values]
    out = sorted({max(1, int(v)) for v in out})
    if not out:
        out = [max(1, int(fallback))]
    return tuple(out)


def _smooth_multiscale(data: np.ndarray, scales: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return max-across-scales score and the winning scale at each pixel."""
    if not scales:
        raise ValueError("scales cannot be empty")
    scales = tuple(int(max(1, s)) for s in scales)
    if len(scales) == 1:
        size = scales[0]
        score = (
            uniform_filter1d(data, size=size, axis=0, mode="nearest")
            if size > 1 else np.array(data, copy=True, dtype=np.float32)
        )
        scale_map = np.full(score.shape, size, dtype=np.int32)
        return score.astype(np.float32, copy=False), scale_map

    best = np.full(data.shape, -np.inf, dtype=np.float32)
    scale_map = np.full(data.shape, scales[0], dtype=np.int32)
    for size in scales:
        current = (
            uniform_filter1d(data, size=size, axis=0, mode="nearest")
            if size > 1 else np.array(data, copy=False, dtype=np.float32)
        )
        current = np.asarray(current, dtype=np.float32)
        better = current > best
        best[better] = current[better]
        scale_map[better] = int(size)
    return best, scale_map


def _bbox_close(a: BBoxDetectionEvent, b: BBoxDetectionEvent, *, gap_t: int, gap_c: int) -> bool:
    dist_t = max(0, max(a.t0, b.t0) - min(a.t1, b.t1))
    dist_c = max(0, max(a.c0, b.c0) - min(a.c1, b.c1))
    return dist_t <= int(gap_t) and dist_c <= int(gap_c)


def merge_bbox_events(
    events: Iterable[BBoxDetectionEvent],
    *,
    gap_t: int = 0,
    gap_c: int = 0,
) -> list[BBoxDetectionEvent]:
    """Merge bbox events that overlap or are separated by small gaps."""
    pending = sorted(
        list(events),
        key=lambda ev: (int(ev.t0), int(ev.c0), int(ev.t1), int(ev.c1)),
    )
    if not pending:
        return []

    merged: list[BBoxDetectionEvent] = []
    while pending:
        base = pending.pop(0)
        changed = True
        while changed:
            changed = False
            keep: list[BBoxDetectionEvent] = []
            for other in pending:
                if _bbox_close(base, other, gap_t=gap_t, gap_c=gap_c):
                    base = base.merged_with(other)
                    changed = True
                else:
                    keep.append(other)
            pending = keep
        merged.append(base)
    return sorted(merged, key=lambda ev: (int(ev.t0), int(ev.c0), int(ev.t1), int(ev.c1)))


class BasePatchDetector:
    """Base class for patch-local DAS event detectors."""

    name: str = "base"
    default_label: str = "event"
    default_color: str = "#FFD54F"

    def _extract_events(
        self,
        mask: np.ndarray,
        score: np.ndarray,
        *,
        scale_map: np.ndarray | None = None,
        min_duration: int = 2,
        min_channels: int = 2,
        merge_gap_t: int = 0,
        merge_gap_c: int = 0,
        label_name: str | None = None,
        color: str | None = None,
    ) -> list[BBoxDetectionEvent]:
        mask = np.asarray(mask, dtype=bool)
        score = np.asarray(score, dtype=np.float32)
        if mask.shape != score.shape:
            raise ValueError("mask and score must have the same shape")

        labeled, _ = label(mask)
        slices = find_objects(labeled)
        events: list[BBoxDetectionEvent] = []
        label_name = str(label_name or self.default_label)
        color = str(color or self.default_color)

        for component_id, slc in enumerate(slices, start=1):
            if slc is None:
                continue
            t_slice, c_slice = slc
            t0, t1 = int(t_slice.start), int(t_slice.stop)
            c0, c1 = int(c_slice.start), int(c_slice.stop)
            if (t1 - t0) < int(min_duration) or (c1 - c0) < int(min_channels):
                continue

            local_mask = labeled[slc] == component_id
            if not np.any(local_mask):
                continue

            local_score = np.where(local_mask, score[slc], -np.inf)
            peak_index = int(np.argmax(local_score))
            peak_tc = np.unravel_index(peak_index, local_score.shape)
            peak_score = float(local_score[peak_tc])
            scale = 1.0
            if scale_map is not None:
                scale = float(scale_map[slc][peak_tc])

            events.append(
                BBoxDetectionEvent(
                    t0=t0,
                    t1=t1,
                    c0=c0,
                    c1=c1,
                    score=peak_score,
                    scale=scale,
                    label=label_name,
                    color=color,
                )
            )

        if merge_gap_t > 0 or merge_gap_c > 0:
            events = merge_bbox_events(events, gap_t=merge_gap_t, gap_c=merge_gap_c)
        return events


class RelativeEnergyDetector(BasePatchDetector):
    """Patch detector based on energy relative to patch-local background."""

    name = "energy"
    default_label = "energy"
    default_color = _DETECTOR_COLORS["energy"]

    def __init__(
        self,
        *,
        threshold: float = 8.0,
        smooth_sizes: int | Sequence[int] = (15, 31),
        background_percentile: float = 50.0,
        min_duration: int = 2,
        min_channels: int = 2,
        merge_gap_t: int = 8,
        merge_gap_c: int = 1,
        label_name: str | None = None,
        color: str | None = None,
    ) -> None:
        self.threshold = float(threshold)
        self.smooth_sizes = _normalize_scales(smooth_sizes, fallback=15)
        self.background_percentile = float(background_percentile)
        self.min_duration = int(min_duration)
        self.min_channels = int(min_channels)
        self.merge_gap_t = int(merge_gap_t)
        self.merge_gap_c = int(merge_gap_c)
        self.label_name = str(label_name or self.default_label)
        self.color = str(color or self.default_color)

    def __call__(self, patch: np.ndarray) -> dict:
        x = _as_patch(patch)
        energy = np.square(x, out=np.empty_like(x, dtype=np.float32))
        background = np.percentile(energy, self.background_percentile, axis=0).astype(np.float32, copy=False)
        floor = max(1e-10, float(np.percentile(energy, 5.0)))
        background = np.maximum(background, floor)
        relative = energy / background[None, :]
        score, scale_map = _smooth_multiscale(relative, self.smooth_sizes)
        mask = score >= self.threshold
        events = self._extract_events(
            mask,
            score,
            scale_map=scale_map,
            min_duration=self.min_duration,
            min_channels=self.min_channels,
            merge_gap_t=self.merge_gap_t,
            merge_gap_c=self.merge_gap_c,
            label_name=self.label_name,
            color=self.color,
        )
        return {"score": score, "mask": mask, "events": [event.to_dict() for event in events]}


class STALTADetector(BasePatchDetector):
    """Patch detector based on STA/LTA ratio on patch-local energy."""

    name = "stalta"
    default_label = "stalta"
    default_color = _DETECTOR_COLORS["stalta"]

    def __init__(
        self,
        *,
        sta_windows: int | Sequence[int] = (20, 40),
        lta_window: int = 200,
        threshold: float = 4.0,
        min_duration: int = 2,
        min_channels: int = 2,
        merge_gap_t: int = 8,
        merge_gap_c: int = 1,
        label_name: str | None = None,
        color: str | None = None,
    ) -> None:
        self.sta_windows = _normalize_scales(sta_windows, fallback=20)
        self.lta_window = int(max(lta_window, max(self.sta_windows) + 1))
        self.threshold = float(threshold)
        self.min_duration = int(min_duration)
        self.min_channels = int(min_channels)
        self.merge_gap_t = int(merge_gap_t)
        self.merge_gap_c = int(merge_gap_c)
        self.label_name = str(label_name or self.default_label)
        self.color = str(color or self.default_color)

    def __call__(self, patch: np.ndarray) -> dict:
        x = _as_patch(patch)
        cf = np.square(x, out=np.empty_like(x, dtype=np.float32))
        lta = uniform_filter1d(cf, size=self.lta_window, axis=0, mode="nearest").astype(np.float32, copy=False)
        lta = np.maximum(lta, 1e-10)
        score, scale_map = _smooth_multiscale(cf, self.sta_windows)
        score = score / lta
        mask = score >= self.threshold
        events = self._extract_events(
            mask,
            score,
            scale_map=scale_map,
            min_duration=self.min_duration,
            min_channels=self.min_channels,
            merge_gap_t=self.merge_gap_t,
            merge_gap_c=self.merge_gap_c,
            label_name=self.label_name,
            color=self.color,
        )
        return {"score": score, "mask": mask, "events": [event.to_dict() for event in events]}


class TKEODetector(BasePatchDetector):
    """Patch detector based on the Teager–Kaiser energy operator."""

    name = "tkeo"
    default_label = "tkeo"
    default_color = _DETECTOR_COLORS["tkeo"]

    def __init__(
        self,
        *,
        threshold: float = 12.0,
        smooth_sizes: int | Sequence[int] = (9, 21),
        background_percentile: float = 50.0,
        min_duration: int = 2,
        min_channels: int = 2,
        merge_gap_t: int = 8,
        merge_gap_c: int = 1,
        label_name: str | None = None,
        color: str | None = None,
    ) -> None:
        self.threshold = float(threshold)
        self.smooth_sizes = _normalize_scales(smooth_sizes, fallback=9)
        self.background_percentile = float(background_percentile)
        self.min_duration = int(min_duration)
        self.min_channels = int(min_channels)
        self.merge_gap_t = int(merge_gap_t)
        self.merge_gap_c = int(merge_gap_c)
        self.label_name = str(label_name or self.default_label)
        self.color = str(color or self.default_color)

    def __call__(self, patch: np.ndarray) -> dict:
        x = _as_patch(patch)
        tkeo = np.zeros_like(x, dtype=np.float32)
        if x.shape[0] >= 3:
            center = np.square(x[1:-1], out=np.empty_like(x[1:-1], dtype=np.float32))
            neighbor = x[:-2] * x[2:]
            tkeo[1:-1] = np.abs(center - neighbor, dtype=np.float32)
        score_raw, scale_map = _smooth_multiscale(tkeo, self.smooth_sizes)
        background = np.percentile(score_raw, self.background_percentile, axis=0).astype(np.float32, copy=False)
        floor = max(1e-10, float(np.percentile(score_raw, 5.0)))
        background = np.maximum(background, floor)
        score = score_raw / background[None, :]
        mask = score >= self.threshold
        events = self._extract_events(
            mask,
            score,
            scale_map=scale_map,
            min_duration=self.min_duration,
            min_channels=self.min_channels,
            merge_gap_t=self.merge_gap_t,
            merge_gap_c=self.merge_gap_c,
            label_name=self.label_name,
            color=self.color,
        )
        return {"score": score, "mask": mask, "events": [event.to_dict() for event in events]}


def build_energy_patch_detector(
    method: DetectorMethod,
    *,
    threshold: float,
    smooth_size: int,
    sta_window: int,
    lta_window: int,
    background_percentile: float,
    min_duration: int,
    min_channels: int,
    merge_gap_t: int,
    merge_gap_c: int,
    label_name: str,
    color: str | None = None,
) -> BasePatchDetector:
    """Build one of the supported energy-style patch detectors."""
    method = str(method).strip().lower()
    if method == "energy":
        return RelativeEnergyDetector(
            threshold=threshold,
            smooth_sizes=smooth_size,
            background_percentile=background_percentile,
            min_duration=min_duration,
            min_channels=min_channels,
            merge_gap_t=merge_gap_t,
            merge_gap_c=merge_gap_c,
            label_name=label_name,
            color=color,
        )
    if method == "stalta":
        return STALTADetector(
            sta_windows=sta_window,
            lta_window=lta_window,
            threshold=threshold,
            min_duration=min_duration,
            min_channels=min_channels,
            merge_gap_t=merge_gap_t,
            merge_gap_c=merge_gap_c,
            label_name=label_name,
            color=color,
        )
    if method == "tkeo":
        return TKEODetector(
            threshold=threshold,
            smooth_sizes=smooth_size,
            background_percentile=background_percentile,
            min_duration=min_duration,
            min_channels=min_channels,
            merge_gap_t=merge_gap_t,
            merge_gap_c=merge_gap_c,
            label_name=label_name,
            color=color,
        )
    raise ValueError(f"Unknown energy detection method: {method!r}")


def _semantic_shape(source) -> tuple[int, int]:
    dims = getattr(source, "dims", "nt_nc")
    raw_shape = tuple(int(v) for v in getattr(source, "shape"))
    if dims == "nt_nc":
        return raw_shape[0], raw_shape[1]
    return raw_shape[1], raw_shape[0]


def _read_patch_nt_nc(source, *, t0: int, t1: int, c0: int, c1: int) -> np.ndarray:
    dims = getattr(source, "dims", "nt_nc")
    if dims == "nt_nc":
        patch = source[t0:t1, c0:c1]
    else:
        patch = source[c0:c1, t0:t1].T
    return np.asarray(patch, dtype=np.float32)


def _sliding_starts(length: int, patch: int, step: int) -> list[int]:
    patch = int(max(1, patch))
    step = int(max(1, step))
    if patch >= length:
        return [0]
    starts = list(range(0, max(1, length - patch + 1), step))
    last = int(length - patch)
    if starts[-1] != last:
        starts.append(last)
    return starts


def _event_owned_by_patch(
    event: BBoxDetectionEvent,
    *,
    t0: int,
    c0: int,
    step_t: int,
    step_c: int,
    nt_total: int,
    nc_total: int,
    is_last_t: bool,
    is_last_c: bool,
) -> bool:
    own_t1 = nt_total if is_last_t else min(nt_total, t0 + step_t)
    own_c1 = nc_total if is_last_c else min(nc_total, c0 + step_c)
    return (t0 <= event.center_t < own_t1) and (c0 <= event.center_c < own_c1)


def run_patch_detector_on_tape(
    source,
    detector: BasePatchDetector,
    *,
    patch_nt: int,
    patch_nc: int,
    step_t: int,
    step_c: int,
    global_merge_gap_t: int = 0,
    global_merge_gap_c: int = 0,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
    result_cb: Callable[[list[dict]], None] | None = None,
) -> list[dict]:
    """Run a patch detector over a full source and return global bbox events."""
    nt_total, nc_total = _semantic_shape(source)
    patch_nt = int(min(max(1, patch_nt), nt_total))
    patch_nc = int(min(max(1, patch_nc), nc_total))
    step_t = int(min(max(1, step_t), patch_nt))
    step_c = int(min(max(1, step_c), patch_nc))

    t_starts = _sliding_starts(nt_total, patch_nt, step_t)
    c_starts = _sliding_starts(nc_total, patch_nc, step_c)
    total = len(t_starts) * len(c_starts)
    done = 0
    events: list[BBoxDetectionEvent] = []

    for ci, c0 in enumerate(c_starts):
        c1 = min(nc_total, c0 + patch_nc)
        is_last_c = ci == (len(c_starts) - 1)
        for ti, t0 in enumerate(t_starts):
            if cancel_cb is not None and cancel_cb():
                merged = merge_bbox_events(events, gap_t=global_merge_gap_t, gap_c=global_merge_gap_c)
                return [event.to_dict() for event in merged]

            t1 = min(nt_total, t0 + patch_nt)
            is_last_t = ti == (len(t_starts) - 1)
            patch = _read_patch_nt_nc(source, t0=t0, t1=t1, c0=c0, c1=c1)
            result = detector(patch)
            patch_events: list[BBoxDetectionEvent] = []
            for raw in result.get("events", []) or []:
                local_event = BBoxDetectionEvent(
                    t0=int(raw["t0"]),
                    t1=int(raw["t1"]),
                    c0=int(raw["c0"]),
                    c1=int(raw["c1"]),
                    score=float(raw.get("score", 0.0)),
                    scale=float(raw.get("scale", 1.0)),
                    label=str(raw.get("label", detector.default_label)),
                    color=str(raw.get("color", detector.default_color)),
                    meta=dict(raw.get("meta", {}) or {}),
                ).shifted(dt=t0, dc=c0)
                if not _event_owned_by_patch(
                    local_event,
                    t0=t0,
                    c0=c0,
                    step_t=step_t,
                    step_c=step_c,
                    nt_total=nt_total,
                    nc_total=nc_total,
                    is_last_t=is_last_t,
                    is_last_c=is_last_c,
                ):
                    continue
                patch_events.append(local_event)

            if patch_events:
                events.extend(patch_events)
                if result_cb is not None:
                    result_cb([event.to_dict() for event in patch_events])

            done += 1
            if progress_cb is not None:
                progress_cb(done, total)

    merged = merge_bbox_events(events, gap_t=global_merge_gap_t, gap_c=global_merge_gap_c)
    return [event.to_dict() for event in merged]


class EnergyDetectionMonitor:
    """Monitor-compatible wrapper around the patch-based energy detectors."""

    label = "Energy Detection"
    key = "energy_detection"
    supports_gpu = False
    param_spec = [
        {
            "key": "method",
            "label": "Method",
            "type": "choice",
            "choices": ["energy", "stalta", "tkeo"],
            "default": "energy",
            "tooltip": "Patch detector used to score local DAS events.",
        },
        {
            "key": "threshold",
            "label": "Threshold",
            "type": "float",
            "min": 0.1,
            "max": 1e6,
            "default": 15.0,
            "step": 0.5,
            "tooltip": "Detection threshold applied to the chosen score map.",
        },
        {
            "key": "smooth_size",
            "label": "Smooth window",
            "type": "int",
            "min": 1,
            "max": 8192,
            "default": 15,
            "tooltip": "Temporal smoothing window for relative-energy and TKEO modes.",
        },
        {
            "key": "sta_window",
            "label": "STA window",
            "type": "int",
            "min": 2,
            "max": 8192,
            "default": 20,
            "tooltip": "STA window length for STA/LTA mode.",
        },
        {
            "key": "lta_window",
            "label": "LTA window",
            "type": "int",
            "min": 4,
            "max": 65536,
            "default": 200,
            "tooltip": "LTA window length for STA/LTA mode.",
        },
        {
            "key": "background_percentile",
            "label": "Background percentile",
            "type": "float",
            "min": 1.0,
            "max": 90.0,
            "default": 70.0,
            "step": 1.0,
            "tooltip": "Patch-local background percentile for relative normalization.",
        },
        {
            "key": "patch_nt",
            "label": "Patch time",
            "type": "int",
            "min": 32,
            "max": 1_000_000,
            "default": 2048,
            "tooltip": "Patch length along time (samples).",
        },
        {
            "key": "patch_nc",
            "label": "Patch channels",
            "type": "int",
            "min": 8,
            "max": 1_000_000,
            "default": 512,
            "tooltip": "Patch height along channels.",
        },
        {
            "key": "step_t",
            "label": "Time step",
            "type": "int",
            "min": 1,
            "max": 1_000_000,
            "default": 1024,
            "tooltip": "Patch stride along time. Smaller values increase overlap.",
        },
        {
            "key": "step_c",
            "label": "Channel step",
            "type": "int",
            "min": 1,
            "max": 1_000_000,
            "default": 256,
            "tooltip": "Patch stride along channels. Smaller values increase overlap.",
        },
        {
            "key": "merge_gap_t",
            "label": "Merge gap t",
            "type": "int",
            "min": 0,
            "max": 1_000_000,
            "default": 32,
            "tooltip": "Merge nearby bbox detections separated by <= this many samples.",
        },
        {
            "key": "merge_gap_c",
            "label": "Merge gap c",
            "type": "int",
            "min": 0,
            "max": 1_000_000,
            "default": 4,
            "tooltip": "Merge nearby bbox detections separated by <= this many channels.",
        },
        {
            "key": "min_duration",
            "label": "Min duration",
            "type": "int",
            "min": 1,
            "max": 1_000_000,
            "default": 10,
            "tooltip": "Reject detections thinner than this many samples.",
        },
        {
            "key": "min_channels",
            "label": "Min channels",
            "type": "int",
            "min": 1,
            "max": 1_000_000,
            "default": 10,
            "tooltip": "Reject detections thinner than this many channels.",
        },
    ]

    def default_params(self) -> dict:
        return {entry["key"]: entry["default"] for entry in self.param_spec}

    def validate_params(self, params: dict, *, fs: float, dx: float, nt: int, nc: int) -> dict:  # noqa: ARG002
        method = str(params.get("method", "energy")).strip().lower()
        if method not in {"energy", "stalta", "tkeo"}:
            method = "energy"
        patch_nt = max(32, min(int(params.get("patch_nt", 2048)), int(nt)))
        patch_nc = max(8, min(int(params.get("patch_nc", 512)), int(nc)))
        step_t = max(1, min(int(params.get("step_t", max(1, patch_nt // 2))), patch_nt))
        step_c = max(1, min(int(params.get("step_c", max(1, patch_nc // 2))), patch_nc))
        sta_window = max(2, int(params.get("sta_window", 20)))
        lta_window = max(sta_window + 1, int(params.get("lta_window", 200)))
        return {
            "method": method,
            "threshold": max(0.1, float(params.get("threshold", 15.0))),
            "smooth_size": max(1, int(params.get("smooth_size", 15))),
            "sta_window": sta_window,
            "lta_window": lta_window,
            "background_percentile": min(90.0, max(1.0, float(params.get("background_percentile", 70.0)))),
            "patch_nt": patch_nt,
            "patch_nc": patch_nc,
            "step_t": step_t,
            "step_c": step_c,
            "merge_gap_t": max(0, int(params.get("merge_gap_t", 32))),
            "merge_gap_c": max(0, int(params.get("merge_gap_c", 4))),
            "min_duration": max(1, int(params.get("min_duration", 10))),
            "min_channels": max(1, int(params.get("min_channels", 10))),
        }

    def run(
        self,
        source,
        *,
        params: dict,
        progress_cb: Callable[[int, int], None] | None = None,
        cancel_cb: Callable[[], bool] | None = None,
        result_cb: Callable[[list[dict]], None] | None = None,
        device: str = "cpu",  # noqa: ARG002
    ) -> list[dict]:
        method = str(params["method"])
        detector = build_energy_patch_detector(
            method,
            threshold=float(params["threshold"]),
            smooth_size=int(params["smooth_size"]),
            sta_window=int(params["sta_window"]),
            lta_window=int(params["lta_window"]),
            background_percentile=float(params["background_percentile"]),
            min_duration=int(params["min_duration"]),
            min_channels=int(params["min_channels"]),
            merge_gap_t=int(params["merge_gap_t"]),
            merge_gap_c=int(params["merge_gap_c"]),
            label_name=method,
            color=_DETECTOR_COLORS.get(method, "#FF8A00"),
        )

        events = run_patch_detector_on_tape(
            source,
            detector,
            patch_nt=int(params["patch_nt"]),
            patch_nc=int(params["patch_nc"]),
            step_t=int(params["step_t"]),
            step_c=int(params["step_c"]),
            global_merge_gap_t=int(params["merge_gap_t"]),
            global_merge_gap_c=int(params["merge_gap_c"]),
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
            result_cb=result_cb,
        )
        for event in events:
            event.setdefault("algorithm", method)
        return events
