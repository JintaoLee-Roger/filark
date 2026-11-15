"""Utilities for exported annotation files and annotation-driven data slices."""

from __future__ import annotations

import base64
import json
import math
import os
from pathlib import Path
from typing import Any, Optional
import zlib

import numpy as np

from filark.io.fileset import BinFolderOpts, TapeFileSet
from filark.io.protocols import Tape
from filark.io.tapeio import BinTape, H5Tape, NpyTape


def describe_source(source: Any, *, axis_metadata: Any | None = None) -> dict[str, Any]:
    """Build a JSON-friendly source summary for annotation exports."""
    tape = _unwrap_tape(source)
    shape = getattr(tape, "shape", None)
    path = _source_path(tape)
    files = _source_files(tape)
    dtype = getattr(tape, "dtype", None)
    info = {
        "kind": type(tape).__name__,
        "name": _source_name(tape),
        "path": str(path) if path is not None else None,
        "files": files or None,
        "dims": getattr(tape, "dims", None),
        "shape": [int(x) for x in shape] if shape is not None else None,
        "dtype": str(dtype) if dtype is not None else None,
        "fs": _maybe_float(getattr(tape, "fs", None)),
        "dx": _maybe_float(getattr(tape, "dx", None)),
        "dx_unit": _maybe_str(getattr(tape, "dx_unit", None)),
        "gauge_length": _maybe_float(getattr(tape, "gauge_length", None)),
        "start_time": _maybe_float(getattr(tape, "start_time", None)),
    }
    transform = _source_transform_info(tape)
    if transform is not None:
        info["transform"] = transform
    axis = _axis_metadata_info(axis_metadata if axis_metadata is not None else getattr(tape, "axis_metadata", None))
    if axis is not None:
        info["axis"] = axis
    return info


def _axis_metadata_info(axis_metadata: Any | None) -> dict[str, Any] | None:
    if axis_metadata is None:
        return None
    if callable(axis_metadata):
        axis_metadata = axis_metadata()
    if hasattr(axis_metadata, "as_dict"):
        axis_metadata = axis_metadata.as_dict()
    if not isinstance(axis_metadata, dict):
        return None
    return json.loads(json.dumps(axis_metadata, default=str))


def _source_transform_info(tape: Any) -> dict[str, Any] | None:
    active = getattr(tape, "transform", None)
    if callable(active):
        active = active()

    raw_options = getattr(tape, "transform_options", None)
    if callable(raw_options):
        raw_options = raw_options()

    options: list[dict[str, str]] = []
    for item in raw_options or ():
        if isinstance(item, dict):
            key = item.get("key", item.get("value"))
            if key is None:
                continue
            label = item.get("label", key)
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            key, label = item[0], item[1]
        else:
            key = label = item
        options.append({"key": str(key), "label": str(label)})

    revision = getattr(tape, "transform_revision", None)
    if callable(revision):
        revision = revision()

    if active is None and not options and revision is None:
        return None

    label = ""
    if active is not None:
        for option in options:
            if option["key"] == str(active):
                label = option["label"]
                break
    return {
        "key": str(active) if active is not None else None,
        "label": label or (str(active) if active is not None else None),
        "revision": int(revision or 0),
        "options": options or None,
    }


def load_annotations(path_or_payload: str | os.PathLike | dict[str, Any]) -> dict[str, Any]:
    """Load an exported annotation payload from disk or return a copied payload."""
    if isinstance(path_or_payload, dict):
        return dict(path_or_payload)
    path = Path(path_or_payload)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Annotation file must contain a JSON object.")
    return payload


def annotation_source_info(path_or_payload: str | os.PathLike | dict[str, Any]) -> dict[str, Any] | None:
    """Return the exported source metadata block, if present."""
    payload = load_annotations(path_or_payload)
    source = payload.get("source")
    if not isinstance(source, dict):
        return None
    return dict(source)


def annotation_items(path_or_payload: str | os.PathLike | dict[str, Any]) -> list[dict[str, Any]]:
    """Return annotation items from an export payload."""
    payload = load_annotations(path_or_payload)
    items = payload.get("items", [])
    if not isinstance(items, list):
        raise ValueError("Annotation payload 'items' must be a list.")
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        copied = dict(item)
        data = copied.get("data", {})
        if isinstance(data, dict):
            copied["data"] = decode_annotation_data(data)
        out.append(copied)
    return out


def encode_annotation_data(data: dict[str, Any]) -> dict[str, Any]:
    """Encode annotation data into a JSON-friendly payload."""
    out = dict(data)
    mask = out.get("mask")
    if mask is None or _is_encoded_mask(mask):
        return out
    arr = np.ascontiguousarray(np.asarray(mask))
    if arr.ndim != 2:
        raise ValueError(f"Mask annotation must be 2D, got shape={arr.shape}.")
    raw = arr.tobytes(order="C")
    out["mask"] = {
        "encoding": "zlib_base64",
        "dtype": str(arr.dtype),
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "data": base64.b64encode(zlib.compress(raw, level=6)).decode("ascii"),
    }
    return out


def decode_annotation_data(data: dict[str, Any]) -> dict[str, Any]:
    """Decode JSON-friendly annotation data into runtime values."""
    out = dict(data)
    mask = out.get("mask")
    if not _is_encoded_mask(mask):
        return out
    dtype = np.dtype(str(mask.get("dtype", "uint8")))
    shape = mask.get("shape", [])
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError("Encoded mask annotation requires shape=[nt, nc].")
    raw = zlib.decompress(base64.b64decode(str(mask.get("data", "")).encode("ascii")))
    arr = np.frombuffer(raw, dtype=dtype).reshape((int(shape[0]), int(shape[1]))).copy()
    out["mask"] = arr
    return out


def _is_encoded_mask(mask: Any) -> bool:
    return isinstance(mask, dict) and str(mask.get("encoding", "")) == "zlib_base64"


def get_annotation_item(
    path_or_payload: str | os.PathLike | dict[str, Any],
    *,
    uid: int | None = None,
    label: str | None = None,
    index: int = 0,
) -> dict[str, Any]:
    """Select one annotation item by uid or label."""
    items = annotation_items(path_or_payload)
    if uid is not None:
        for item in items:
            if int(item.get("uid", -1)) == int(uid):
                return item
        raise KeyError(f"Annotation uid={uid} not found.")

    if label is not None:
        matches = [item for item in items if str(item.get("label", "")) == str(label)]
        if not matches:
            raise KeyError(f"Annotation label={label!r} not found.")
        if index < 0 or index >= len(matches):
            raise IndexError(index)
        return matches[index]

    if index < 0 or index >= len(items):
        raise IndexError(index)
    return items[index]


def annotation_bounds(
    item: dict[str, Any],
    *,
    pad_t: int = 0,
    pad_c: int = 0,
) -> tuple[int, int, int, int]:
    """Convert a bbox/polyline annotation into half-open slice bounds."""
    anno_type = str(item.get("anno_type", "bbox"))
    data = item.get("data", {})
    if not isinstance(data, dict):
        raise ValueError("Annotation item data must be a dict.")

    if anno_type == "bbox":
        t0 = float(data.get("t0", 0.0))
        t1 = float(data.get("t1", 0.0))
        c0 = float(data.get("c0", 0.0))
        c1 = float(data.get("c1", 0.0))
    elif anno_type == "polyline":
        pts = data.get("points", []) or []
        if not pts:
            raise ValueError("Polyline annotation has no points.")
        xs = [float(pt[0]) for pt in pts]
        ys = [float(pt[1]) for pt in pts]
        t0, t1 = min(xs), max(xs)
        c0, c1 = min(ys), max(ys)
    elif anno_type == "mask":
        t0 = float(data.get("t0", 0.0))
        t1 = float(data.get("t1", 0.0))
        c0 = float(data.get("c0", 0.0))
        c1 = float(data.get("c1", 0.0))
    else:
        raise NotImplementedError(f"Bounds extraction is not implemented for anno_type={anno_type!r}.")

    t0_i = math.floor(min(t0, t1)) - int(pad_t)
    t1_i = math.ceil(max(t0, t1)) + int(pad_t)
    c0_i = math.floor(min(c0, c1)) - int(pad_c)
    c1_i = math.ceil(max(c0, c1)) + int(pad_c)
    if t1_i <= t0_i:
        t1_i = t0_i + 1
    if c1_i <= c0_i:
        c1_i = c0_i + 1
    return t0_i, t1_i, c0_i, c1_i


def open_annotation_source(
    path_or_payload: str | os.PathLike | dict[str, Any],
    *,
    override: str | os.PathLike | Tape | None = None,
) -> Tape:
    """Open the source referenced by an exported annotation file."""
    if isinstance(override, Tape):
        return override

    payload = load_annotations(path_or_payload)
    info = annotation_source_info(payload) or {}
    if override is not None:
        path = Path(override)
    else:
        raw_path = info.get("path")
        if raw_path:
            path = Path(raw_path)
        else:
            files = info.get("files") or []
            if len(files) == 1:
                path = Path(files[0])
            else:
                raise ValueError("Annotation export does not contain a single recoverable source path.")

    suffix = path.suffix.lower()
    dims = str(info.get("dims") or "nt_nc")
    fs = float(info.get("fs") or 1.0)
    dx = float(info.get("dx") or 1.0)
    dx_unit = str(info.get("dx_unit") or "m")
    raw_gauge = info.get("gauge_length")
    gauge_length = None if raw_gauge is None else float(raw_gauge)
    start_time = info.get("start_time")
    start_time = float(start_time) if start_time is not None else None

    if suffix in (".h5", ".hdf5"):
        return H5Tape(str(path), start_time=start_time)
    if suffix == ".npy":
        return NpyTape(
            str(path),
            dims=dims,
            fs=fs,
            dx=dx,
            dx_unit=dx_unit,
            gauge_length=gauge_length,
            start_time=start_time,
        )
    if suffix in (".dat", ".bin"):
        shape = info.get("shape")
        dtype = info.get("dtype")
        if not shape or len(shape) != 2 or not dtype:
            raise ValueError("BIN annotation exports require shape and dtype metadata to reopen the source.")
        raw0, raw1 = int(shape[0]), int(shape[1])
        nt = raw0 if dims == "nt_nc" else raw1
        nc = raw1 if dims == "nt_nc" else raw0
        return BinTape(
            path=str(path),
            nt=nt,
            nc=nc,
            dtype=np.dtype(dtype),
            dims=dims,
            fs=fs,
            dx=dx,
            dx_unit=dx_unit,
            gauge_length=gauge_length,
            start_time=start_time,
        )

    files = info.get("files") or []
    if files:
        return _open_fileset(files, info)
    raise ValueError(f"Unsupported annotation source suffix: {suffix or '<none>'}")


def slice_annotation_data(
    source_or_payload: Tape | str | os.PathLike | dict[str, Any],
    *,
    annotation: dict[str, Any] | None = None,
    uid: int | None = None,
    label: str | None = None,
    index: int = 0,
    pad_t: int = 0,
    pad_c: int = 0,
) -> dict[str, Any]:
    """Extract a semantic nt_nc data block for one annotation."""
    owns_source = False
    payload: dict[str, Any] | None = None

    if isinstance(source_or_payload, Tape):
        tape = source_or_payload
        if annotation is None:
            raise ValueError("When passing a Tape directly, provide the annotation item explicitly.")
    else:
        payload = load_annotations(source_or_payload)
        tape = open_annotation_source(payload)
        owns_source = True

    try:
        item = dict(annotation) if annotation is not None else get_annotation_item(
            payload if payload is not None else {"items": []},
            uid=uid,
            label=label,
            index=index,
        )
        t0, t1, c0, c1 = annotation_bounds(item, pad_t=pad_t, pad_c=pad_c)
        nt, nc = _semantic_shape(tape)
        t0 = max(0, min(t0, nt))
        t1 = max(t0 + 1, min(t1, nt))
        c0 = max(0, min(c0, nc))
        c1 = max(c0 + 1, min(c1, nc))

        if tape.dims == "nt_nc":
            data = np.asarray(tape[t0:t1, c0:c1])
        else:
            data = np.asarray(tape[c0:c1, t0:t1]).T

        return {
            "data": data,
            "bounds": {"t0": t0, "t1": t1, "c0": c0, "c1": c1},
            "annotation": item,
            "source": describe_source(tape),
        }
    finally:
        if owns_source:
            tape.close()


def _unwrap_tape(source: Any) -> Any:
    data = getattr(source, "data", None)
    return data if data is not None else source


def _source_path(source: Any) -> Path | None:
    for attr in ("p", "path", "source"):
        value = getattr(source, attr, None)
        if isinstance(value, (str, os.PathLike)):
            return Path(value)
    return None


def _source_files(source: Any) -> list[str]:
    files = []
    for tape in getattr(source, "_files", []):
        path = _source_path(tape)
        if path is not None:
            files.append(str(path))
    return files


def _source_name(source: Any) -> str:
    path = _source_path(source)
    if path is not None:
        return path.name
    files = _source_files(source)
    if files:
        return Path(files[0]).name
    return type(source).__name__


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _maybe_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _semantic_shape(tape: Tape) -> tuple[int, int]:
    raw0, raw1 = tape.shape
    if tape.dims == "nt_nc":
        return int(raw0), int(raw1)
    return int(raw1), int(raw0)


def _open_fileset(files: list[str], info: dict[str, Any]) -> Tape:
    dims = str(info.get("dims") or "nt_nc")
    fs = float(info.get("fs") or 1.0)
    dx = float(info.get("dx") or 1.0)
    dx_unit = str(info.get("dx_unit") or "m")
    raw_gauge = info.get("gauge_length")
    gauge_length = None if raw_gauge is None else float(raw_gauge)
    shape = info.get("shape")
    dtype = info.get("dtype")
    suffix = Path(files[0]).suffix.lower() if files else ""

    if suffix in (".h5", ".hdf5"):
        return TapeFileSet(files)
    if suffix == ".npy":
        return TapeFileSet(
            files,
            npy_dims=dims,
            npy_fs=fs,
            npy_dx=dx,
            npy_dx_unit=dx_unit,
            npy_gauge_length=gauge_length,
        )
    if suffix in (".dat", ".bin"):
        if not shape or len(shape) != 2 or not dtype:
            raise ValueError("BIN filesets require shape and dtype metadata.")
        raw0, raw1 = int(shape[0]), int(shape[1])
        nt = raw0 if dims == "nt_nc" else raw1
        nc = raw1 if dims == "nt_nc" else raw0
        return TapeFileSet(
            files,
            bin_opts=BinFolderOpts(
                nt=nt,
                nc=nc,
                dtype=np.dtype(dtype),
                dims=dims,
                fs=fs,
                dx=dx,
                dx_unit=dx_unit,
                gauge_length=gauge_length,
            ),
        )
    raise ValueError("Unsupported fileset annotation source.")
