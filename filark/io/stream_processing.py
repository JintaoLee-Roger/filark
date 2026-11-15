# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""Adapters and helpers for binding DSP transforms to streaming sources."""

from __future__ import annotations

from inspect import Parameter, signature
from typing import Any

import numpy as np

from filark.dsp.base import from_nt_nc, to_nt_nc
from filark.dsp.temporal import iir_filter_time_rt
from filark.dsp.transforms import Compose
from filark._exceptions import print_exception
from filark.io.streaming import StreamingSource
from filark.utils.perf import elapsed_ms, log_perf, perf_debug_enabled, perf_now


class StreamLayoutAdapter:
    """Adapt nt_nc-style DSP transforms to streaming nc_nt blocks."""

    def __init__(self, inner: Any):
        self.inner = inner

    def __call__(self, data):
        x = to_nt_nc(data, layout="nc_nt")
        y = self.inner(x)
        return from_nt_nc(y, layout="nc_nt")

    def is_stateful(self) -> bool:
        fn = getattr(self.inner, "is_stateful", None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                print_exception(f"StreamLayoutAdapter inner transform {self.inner!r} failed in is_stateful().", prefix="Stream")
        return callable(getattr(self.inner, "reset", None))

    def reset(self) -> None:
        reset = getattr(self.inner, "reset", None)
        if callable(reset):
            reset()

    def get_state(self):
        get_state = getattr(self.inner, "get_state", None)
        if callable(get_state):
            return get_state()
        return None

    def set_state(self, state) -> None:
        set_state = getattr(self.inner, "set_state", None)
        if callable(set_state):
            set_state(state)
        elif state is not None:
            raise ValueError(f"{self.inner!r} does not support set_state()")

    def warmup_samples(self, visible_w: int | None = None) -> int:
        fn = getattr(self.inner, "warmup_samples", None)
        if callable(fn):
            try:
                return int(max(0, fn(visible_w)))
            except TypeError:
                return int(max(0, fn()))
        return int(max(0, visible_w or 0))

    def __repr__(self) -> str:
        return f"StreamLayoutAdapter({self.inner!r})"


class RealtimeDisplaySOSFilter:
    """Stateful causal SOS filter for streaming display blocks [channels, time]."""

    def __init__(self, sos: np.ndarray):
        self.sos = np.asarray(sos, dtype=np.float64)
        self._zi: np.ndarray | None = None

    def reset(self) -> None:
        self._zi = None

    def is_stateful(self) -> bool:
        return True

    def get_state(self) -> np.ndarray | None:
        if self._zi is None:
            return None
        return np.array(self._zi, copy=True)

    def set_state(self, state: np.ndarray | None) -> None:
        if state is None:
            self._zi = None
            return
        self._zi = np.asarray(state, dtype=np.float64).copy()

    def warmup_samples(self, visible_w: int | None = None) -> int:
        pole_radius = 0.0
        for sec in self.sos:
            a0 = float(sec[3])
            if abs(a0) < 1e-12:
                continue
            roots = np.roots([1.0, float(sec[4]) / a0, float(sec[5]) / a0])
            if roots.size:
                pole_radius = max(pole_radius, float(np.max(np.abs(roots))))

        if pole_radius >= 0.999999:
            settle = 32768
        elif pole_radius > 0.0:
            settle = int(np.ceil(np.log(1e-4) / np.log(pole_radius)))
        else:
            settle = 0

        hint = int(max(0, visible_w or 0))
        return int(min(32768, max(256, hint, settle)))

    def __call__(self, block: np.ndarray) -> np.ndarray:
        start = perf_now() if perf_debug_enabled() else 0.0
        # iir_filter_time_rt owns precision handling; avoid an extra float64
        # copy of every float32 display block before SciPy's working copy.
        x = np.asarray(block)
        x_nt_nc = x.T
        try:
            y, self._zi = iir_filter_time_rt(x_nt_nc, self.sos, zi=self._zi)
        except ValueError:
            self._zi = None
            y, self._zi = iir_filter_time_rt(x_nt_nc, self.sos, zi=self._zi)
        if perf_debug_enabled():
            log_perf(
                "display_filter.call",
                ms=elapsed_ms(start),
                channels=int(block.shape[0]) if block.ndim >= 1 else 0,
                samples=int(block.shape[1]) if block.ndim >= 2 else 0,
                sections=int(self.sos.shape[0]) if self.sos.ndim >= 1 else 0,
            )
        return y.T.astype(np.float32, copy=False)


def compose_processing_chain(parts: list[Any]):
    parts = [part for part in parts if part is not None]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return Compose(parts)


def build_stream_callable(spec: Any, *, fs: float, dx: float):
    if spec is None:
        return None

    if isinstance(spec, (list, tuple)):
        parts = [
            build_stream_callable(part, fs=fs, dx=dx)
            for part in spec
        ]
        return compose_processing_chain(parts)

    if callable(spec):
        try:
            sig = signature(spec)
        except (TypeError, ValueError):
            sig = None

        if sig is not None:
            params = [
                p for p in sig.parameters.values()
                if p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
            ]
            required = [p for p in params if p.default is Parameter.empty]
            names = [p.name for p in required]

            if not required:
                return build_stream_callable(spec(), fs=fs, dx=dx)
            if len(required) == 1 and names[0] == "fs":
                return build_stream_callable(spec(fs), fs=fs, dx=dx)
            if len(required) == 1 and names[0] == "dx":
                return build_stream_callable(spec(dx), fs=fs, dx=dx)
            if len(required) == 2 and set(names) == {"fs", "dx"}:
                built = spec(fs, dx) if names == ["fs", "dx"] else spec(dx, fs)
                return build_stream_callable(built, fs=fs, dx=dx)

    mod = str(getattr(spec, "__module__", ""))
    if mod.startswith("filark.dsp.transforms"):
        return StreamLayoutAdapter(spec)
    return spec


def build_stream_preprocess(spec: Any, *, fs: float, dx: float):
    return build_stream_callable(spec, fs=float(fs), dx=float(dx))


def build_stream_postprocess(spec: Any, *, fs: float, dx: float):
    return build_stream_callable(spec, fs=float(fs), dx=float(dx))


def ensure_streaming_source(source):
    if isinstance(source, StreamingSource):
        return source
    dims = "auto"
    if hasattr(source, "dims"):
        dims = source.dims
    return StreamingSource(source, dims=dims)


class SourceProcessingController:
    """Bind GUI/runtime processing layers onto a StreamingSource."""

    def __init__(self) -> None:
        self.source = None
        self._base_preprocess = None
        self._base_postprocess = None
        self._stream_preprocess = None
        self._stream_postprocess = None
        self._display_postprocess = None

    def bind_source(self, source) -> None:
        self.source = source
        self._base_preprocess = getattr(source, "preprocess", None)
        self._base_postprocess = getattr(source, "postprocess", None)
        self.apply()

    def set_stream_processing(self, preprocess=None, postprocess=None) -> None:
        self._stream_preprocess = preprocess
        self._stream_postprocess = postprocess
        self.apply()

    def set_display_postprocess(self, postprocess=None) -> None:
        self._display_postprocess = postprocess
        self.apply()

    def apply(self) -> None:
        if self.source is None:
            return
        self.source.preprocess = compose_processing_chain([
            self._base_preprocess,
            self._stream_preprocess,
        ])
        self.source.postprocess = compose_processing_chain([
            self._base_postprocess,
            self._stream_postprocess,
            self._display_postprocess,
        ])
        reset = getattr(self.source, "reset_processing_state", None)
        if callable(reset):
            reset()
