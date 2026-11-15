# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""Execution helpers that run processors over full datasets and save outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .base import BasePipeline

__all__ = [
    "ProcessorRunResult",
    "run_processor_to_file",
    "run_nt_nc_transform_to_file",
]


def _norm_layout(source: Any) -> str:
    dims = str(getattr(source, "dims", "nt_nc") or "nt_nc").lower()
    return dims if dims in ("nt_nc", "nc_nt") else "nt_nc"


def _norm_fs_dx(source: Any) -> tuple[float, float]:
    fs = float(getattr(source, "fs", 1.0) or 1.0)
    dx = float(getattr(source, "dx", 1.0) or 1.0)
    if not np.isfinite(fs) or fs <= 0:
        fs = 1.0
    if not np.isfinite(dx) or dx <= 0:
        dx = 1.0
    return fs, dx


def _jsonable_params(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, (int, float, np.floating)):
            out[key] = float(value)
        elif isinstance(value, (list, tuple)):
            out[key] = [
                float(v) if isinstance(v, (int, float, np.floating)) else v
                for v in value
            ]
        else:
            out[key] = value
    return out


def _write_metadata(
    *,
    source: Any,
    save_path: str | Path,
    fs: float,
    dx: float,
    dims: str,
    method_key: str,
    chunk_mode: str,
    device: str,
    params: dict[str, Any],
) -> str:
    meta_path = str(Path(save_path).with_suffix("")) + "_meta.json"
    meta = {
        "fs": float(fs),
        "dx": float(dx),
        "dx_unit": getattr(source, "dx_unit", None),
        "gauge_length": getattr(source, "gauge_length", None),
        "dims": dims,
        "method": method_key,
        "chunk_mode": chunk_mode,
        "device": device,
        "params": _jsonable_params(params),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta_path


class _CallablePipeline(BasePipeline):
    def __init__(self, func: Callable[[np.ndarray], np.ndarray]):
        super().__init__()
        self._func = func

    def _run(self, data: np.ndarray) -> np.ndarray:
        out = self._func(data)
        return np.asarray(out, dtype=np.float32)


@dataclass(frozen=True)
class ProcessorRunResult:
    save_path: str
    meta_path: str
    params: dict[str, Any]
    fs: float
    dx: float
    dims: str
    chunk_mode: str
    device: str
    method_key: str


def run_nt_nc_transform_to_file(
    source: Any,
    save_path: str | Path,
    *,
    transform: Callable[[np.ndarray], np.ndarray],
    method_key: str = "transform",
    device: str = "cpu",
    chunk_channels: int | None = None,
    target_chunk_mb: float = 128.0,
    overlap_channels: int = 0,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> ProcessorRunResult:
    dims = _norm_layout(source)
    fs, dx = _norm_fs_dx(source)
    save_path = str(Path(save_path).expanduser().resolve())

    pipe = _CallablePipeline(transform)
    pipe(
        source,
        save_path,
        chunked=True,
        chunk_channels=chunk_channels,
        target_chunk_mb=target_chunk_mb,
        overlap_channels=overlap_channels,
        progress_cb=progress_cb,
        cancel_cb=cancel_cb,
    )
    meta_path = _write_metadata(
        source=source,
        save_path=save_path,
        fs=fs,
        dx=dx,
        dims=dims,
        method_key=method_key,
        chunk_mode="channel",
        device=device,
        params={},
    )
    return ProcessorRunResult(
        save_path=save_path,
        meta_path=meta_path,
        params={},
        fs=fs,
        dx=dx,
        dims=dims,
        chunk_mode="channel",
        device=device,
        method_key=method_key,
    )


def run_processor_to_file(
    source: Any,
    processor,
    params: dict[str, Any],
    save_path: str | Path,
    *,
    chunk_mode: str = "channel",
    device: str = "cpu",
    target_chunk_mb: float = 128.0,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> ProcessorRunResult:
    dims = _norm_layout(source)
    fs, dx = _norm_fs_dx(source)
    save_path = str(Path(save_path).expanduser().resolve())
    chunk_mode = str(chunk_mode)
    device = str(device)
    params_norm = processor.validate_params(dict(params), fs=fs, dx=dx)

    def _apply(block: np.ndarray) -> np.ndarray:
        out = processor.run(block, fs=fs, dx=dx, params=params_norm, device=device)
        return np.asarray(out, dtype=np.float32)

    pipe = _CallablePipeline(_apply)

    if chunk_mode == "none":
        if callable(cancel_cb) and cancel_cb():
            raise RuntimeError("Processing cancelled.")
        if callable(progress_cb):
            progress_cb(0, 1)
        data_native = np.asarray(source[:, :], dtype=np.float32)
        data_nt_nc = data_native if dims == "nt_nc" else data_native.T
        out = _apply(data_nt_nc)
        if out.shape != data_nt_nc.shape:
            raise ValueError(f"Output shape mismatch: {out.shape} != {data_nt_nc.shape}")
        out_native = out if dims == "nt_nc" else out.T
        np.save(save_path, out_native, allow_pickle=False)
        if callable(progress_cb):
            progress_cb(1, 1)
    elif chunk_mode == "channel":
        pipe(
            source,
            save_path,
            chunked=True,
            overlap_channels=int(processor.overlap_channels(params_norm)),
            target_chunk_mb=target_chunk_mb,
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
        )
    elif chunk_mode == "block":
        overlap_channels = int(processor.overlap_channels(params_norm))
        overlap_t = int(processor.overlap_time_samples(params_norm, fs))
        pipe(
            source,
            save_path,
            chunked=True,
            block_t=0,
            overlap_channels=overlap_channels,
            overlap_c=overlap_channels,
            overlap_t=overlap_t,
            taper_t=overlap_t > 0,
            target_chunk_mb=target_chunk_mb,
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
        )
    else:
        raise ValueError(f"Unknown chunk_mode: {chunk_mode!r}")

    meta_path = _write_metadata(
        source=source,
        save_path=save_path,
        fs=fs,
        dx=dx,
        dims=dims,
        method_key=str(getattr(processor, "method_key", "processor")),
        chunk_mode=chunk_mode,
        device=device,
        params=params_norm,
    )
    return ProcessorRunResult(
        save_path=save_path,
        meta_path=meta_path,
        params=params_norm,
        fs=fs,
        dx=dx,
        dims=dims,
        chunk_mode=chunk_mode,
        device=device,
        method_key=str(getattr(processor, "method_key", "processor")),
    )
