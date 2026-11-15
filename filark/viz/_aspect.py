"""Helpers for aspect-aware display sizing in the streaming viewer."""

from __future__ import annotations

import math

DEFAULT_MIN_DISPLAY_ASPECT = 0.05
DEFAULT_MAX_DISPLAY_ASPECT = 20.0
DEFAULT_DISPLAY_ASPECT_LIMITS = (DEFAULT_MIN_DISPLAY_ASPECT, DEFAULT_MAX_DISPLAY_ASPECT)


def normalize_display_aspect_limits(
    limits: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Validate and normalize display-aspect clamp limits."""
    if limits is None:
        limits = DEFAULT_DISPLAY_ASPECT_LIMITS
    if len(limits) != 2:
        raise ValueError("display_aspect_limits must contain exactly two values")
    lo = float(limits[0])
    hi = float(limits[1])
    if (not math.isfinite(lo)) or (not math.isfinite(hi)) or lo <= 0.0 or hi <= 0.0:
        raise ValueError("display_aspect_limits must be finite positive numbers")
    if hi < lo:
        raise ValueError("display_aspect_limits max must be >= min")
    return lo, hi


def clamp_display_aspect(
    value: float,
    *,
    default: float = 1.0,
    limits: tuple[float, float] | None = None,
) -> float:
    """Return a finite positive display aspect clamped to a safe range."""
    lo, hi = normalize_display_aspect_limits(limits)
    if value is None:
        value = default
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("display_aspect must be a finite positive number")
    return float(min(hi, max(lo, value)))


def auto_display_aspect_for_buffer(
    buffer_h: int,
    viewport_h_px: float,
    *,
    limits: tuple[float, float] | None = None,
) -> float:
    """Choose an aspect that keeps low-channel views from collapsing width.

    display_aspect > 1 compresses X (more samples shown per Y unit).
    For low buffer_h we auto-increase aspect so X fills the viewport.
    """
    bh = max(1, int(buffer_h))
    vh = max(1.0, float(viewport_h_px))
    return clamp_display_aspect(max(1.0, vh / float(bh)), limits=limits)


def recommended_buffer_width(
    *,
    buffer_h: int,
    viewport_size: tuple[float, float],
    x_margin: int,
    x_align: int,
    display_aspect: float,
    display_aspect_limits: tuple[float, float] | None = None,
) -> int:
    """Recommend a buffer width from viewport geometry and display aspect."""
    bh = max(1, int(buffer_h))
    vw = max(1.0, float(viewport_size[0]))
    vh = max(1.0, float(viewport_size[1]))
    aspect = clamp_display_aspect(display_aspect, limits=display_aspect_limits)
    align = int(x_align if x_align >= 8 else 64)
    visible_w = (vw / vh) * float(bh) * aspect
    total_w = max(1.0, visible_w + 2.0 * float(max(0, x_margin)))
    return int(math.ceil(total_w / align) * align)
