# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""
MonitorPlugin — base class for batch / streaming detection plugins.

A monitor plugin analyses a full Tape and returns a structured event list.
The framework renders each event as a canvas overlay and optionally sends
events to the AnnotationStore.

Interface
---------
Subclass ``MonitorPlugin`` and implement ``run()``.

``param_spec`` follows the same format as ``ProcessorPlugin``::

    {
        "key": str,
        "label": str,
        "type": "int" | "float" | "bool" | "choice" | "str" | "file"
                | "int_range" | "float_range",
        "default": ...,
        # type-specific extras (min/max/step/choices/...)
    }

``run()`` returns a list of overlay-like event dicts.  The minimal bbox form is::

    {
        "geometry_type": "bbox",
        "t0": int, "t1": int,
        "c0": int, "c1": int,
        "label": str,
    }

Plugins may also return polyline geometry such as::

    {
        "geometry_type": "polyline",
        "points": [(t, c), ...],
        "label": str,
    }

Optional fields such as ``score``, ``color``, ``algorithm``, and ``meta`` are
passed through to the canvas overlay layer.
"""

from __future__ import annotations

from typing import Callable


class MonitorPlugin:
    """Base class for DAS monitoring / detection plugins."""

    label: str = "Unknown Monitor"
    key: str = "unknown_monitor"
    param_spec: list[dict] = []
    supports_gpu: bool = False

    def default_params(self) -> dict:
        return {e["key"]: e["default"] for e in self.param_spec}

    def validate_params(
        self,
        params: dict,
        *,
        fs: float,
        dx: float,
        nt: int,
        nc: int,
    ) -> dict:
        """Clamp / normalise params before use. Override in subclasses."""
        return dict(params)

    def run(
        self,
        source,
        *,
        params: dict,
        progress_cb: Callable[[int, int], None] | None = None,
        cancel_cb: Callable[[], bool] | None = None,
        device: str = "cpu",
        result_cb: Callable[[list[dict]], None] | None = None,
    ) -> list[dict]:
        """
        Run detection over the full source Tape.

        Parameters
        ----------
        source : Tape
            Full data tape.
        params : dict
            Validated parameter dict (from ``validate_params``).
        progress_cb : callable(done, total) | None
            Progress callback for UI progress bar.
        cancel_cb : callable() -> bool | None
            Returns True when the user requests cancellation.
        result_cb : callable(batch_events) | None
            Optional callback for progressive result streaming.  Plugins may
            call this with newly detected overlay events while the full run is
            still in progress.
        device : str
            "cpu" or a torch device string.

        Returns
        -------
        list[dict]
            Each dict should contain a geometry type and enough coordinates to
            render it on the main canvas, for example::

                {"geometry_type": "bbox", "t0": int, "t1": int, "c0": int, "c1": int, "label": str}
        """
        raise NotImplementedError
