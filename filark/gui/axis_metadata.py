"""Lightweight display-axis metadata used at GUI boundaries."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


_ROLE_LABELS = {
    "time": "Time",
    "channel": "Channel",
    "trace": "Trace",
    "distance": "Distance",
    "depth": "Depth",
    "offset": "Offset",
    "receiver": "Receiver",
    "sample": "Sample",
    "index": "Index",
}

_TIME_ROLES = {"time"}
_DOWNWARD_Y_ROLES = {"time", "depth"}
_SPACE_ROLES = {"channel", "trace", "distance", "offset", "receiver", "depth", "index"}


def _normalize_role(value: str | None, fallback: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "", text)
    return text or fallback


def _infer_role(label: str | None, fallback: str) -> str:
    text = str(label or "").strip().lower()
    if not text:
        return fallback
    for role in ("trace", "channel", "offset", "receiver", "depth", "distance", "sample", "time"):
        if role in text:
            return role
    return fallback


def _infer_unit(label: str | None, explicit: str | None) -> str:
    if explicit is not None:
        return str(explicit)
    match = re.search(r"\(([^()]*)\)", str(label or ""))
    return match.group(1).strip() if match else ""


def _label_for(role: str, label: str | None, unit: str) -> str:
    if label is not None:
        return str(label)
    base = _ROLE_LABELS.get(role, role.replace("_", " ").title())
    return f"{base} ({unit})" if unit else base


def _resolve_invert_y(value: bool | str, y_role: str) -> bool:
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "auto":
            return y_role in _DOWNWARD_Y_ROLES
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        raise ValueError("invert_y must be bool or 'auto'")
    return bool(value)


def _float_attr(obj: Any, names: tuple[str, ...], fallback: float) -> float:
    for name in names:
        value = getattr(obj, name, None)
        if value is None:
            continue
        try:
            value = float(value)
        except Exception:
            continue
        if value > 0:
            return value
    return float(fallback)


@dataclass(frozen=True)
class AxisSpec:
    """One displayed axis."""

    role: str
    label: str
    unit: str = ""

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "label": self.label, "unit": self.unit}


@dataclass(frozen=True)
class AxisMetadata:
    """Display-axis roles plus resolved visual direction."""

    x: AxisSpec
    y: AxisSpec
    invert_y: bool = False

    @property
    def time_axis(self) -> str | None:
        if self.x.role in _TIME_ROLES:
            return "x"
        if self.y.role in _TIME_ROLES:
            return "y"
        return None

    @property
    def space_axis(self) -> str | None:
        if self.time_axis == "x":
            return "y"
        if self.time_axis == "y":
            return "x"
        if self.x.role in _SPACE_ROLES:
            return "x"
        if self.y.role in _SPACE_ROLES:
            return "y"
        return None

    @property
    def is_default_time_channel(self) -> bool:
        return self.x.role == "time" and self.y.role == "channel"

    @property
    def requires_roi_transpose(self) -> bool:
        return self.time_axis == "y"

    @property
    def time_label(self) -> str:
        axis = self.time_axis
        return self.y.label if axis == "y" else self.x.label

    @property
    def space_label(self) -> str:
        axis = self.space_axis
        if axis == "x":
            return self.x.label
        if axis == "y":
            return self.y.label
        return "Space"

    def canonical_nt_nc(self, display_xy_data):
        """Return data with time on axis 0 when the displayed y axis is time."""
        if self.requires_roi_transpose:
            return display_xy_data.T
        return display_xy_data

    def time_fs(self, source: Any) -> float:
        if self.time_axis == "y":
            return _float_attr(source, ("y_fs", "time_fs", "fs"), 1.0)
        return _float_attr(source, ("x_fs", "time_fs", "fs"), 1.0)

    def space_dx(self, source: Any) -> float:
        if self.space_axis == "x":
            return _float_attr(source, ("x_dx", "space_dx", "dx"), 1.0)
        return _float_attr(source, ("y_dx", "space_dx", "dx"), 1.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "x": self.x.as_dict(),
            "y": self.y.as_dict(),
            "invert_y": bool(self.invert_y),
            "time_axis": self.time_axis,
            "space_axis": self.space_axis,
            "is_default_time_channel": bool(self.is_default_time_channel),
        }


def build_axis_metadata(
    *,
    x_role: str | None = None,
    y_role: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    x_unit: str | None = None,
    y_unit: str | None = None,
    invert_y: bool | str = "auto",
) -> AxisMetadata:
    """Build GUI axis metadata from config fields."""
    xr = _normalize_role(x_role, _infer_role(x_label, "time"))
    yr = _normalize_role(y_role, _infer_role(y_label, "channel"))
    xu = _infer_unit(x_label, x_unit)
    yu = _infer_unit(y_label, y_unit)
    return AxisMetadata(
        x=AxisSpec(role=xr, label=_label_for(xr, x_label, xu), unit=xu),
        y=AxisSpec(role=yr, label=_label_for(yr, y_label, yu), unit=yu),
        invert_y=_resolve_invert_y(invert_y, yr),
    )
