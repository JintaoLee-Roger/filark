"""Standalone user extension for annotating the Caltech event Zarr dataset."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from filark.gui.app import run_app
from filark.gui.config import GuiConfig


class CaltechZarrTape:
    """Expose data=(event, channel, time) as one continuous nt_nc tape."""

    dims = "nt_nc"
    dx_unit = "m"
    gauge_length: Optional[float] = None
    start_time: Optional[float] = None

    def __init__(
        self,
        path: str | Path,
        *,
        gap_nt: int = 200,
        transform: str = "raw",
        fk_vmin: float = 800.0,
        fk_vmax: float = 5000.0,
        fk_taper_width: float = 200.0,
        fk_cache_size: int = 4,
    ) -> None:
        try:
            import zarr
        except ModuleNotFoundError as exc:
            raise ImportError("This script needs zarr. Install it with: pip install zarr") from exc

        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        self.group = zarr.open_group(str(self.path), mode="r")
        self.data = self.group["data"]
        self.valid_nt = np.asarray(self.group["valid_nt"][:], dtype=np.int64)
        self.event_id = np.asarray(self.group["event_id"][:]) if "event_id" in self.group else None

        if self.data.ndim != 3:
            raise ValueError(f"Expected data=(event, channel, time), got shape={self.data.shape}")
        if self.valid_nt.shape[0] != self.data.shape[0]:
            raise ValueError("valid_nt length does not match number of events")

        self.n_events = int(self.data.shape[0])
        self.nc = int(self.data.shape[1])
        self.event_capacity_nt = int(self.data.shape[2])
        self.gap_nt = int(max(0, gap_nt))
        self.fs = float(self.group.attrs.get("fs", 1.0))
        self.dx = float(self.group.attrs.get("dx", 1.0))
        self.valid_nt = np.clip(self.valid_nt, 0, self.event_capacity_nt)
        self.fk_vmin = float(fk_vmin)
        self.fk_vmax = float(fk_vmax)
        self.fk_taper_width = float(fk_taper_width)
        self.fk_cache_size = int(max(0, fk_cache_size))
        self._fk_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._transform = "raw"
        self._transform_revision = 0
        self.set_transform(transform)

        self.event_starts = np.empty(self.n_events, dtype=np.int64)
        self.event_stops = np.empty(self.n_events, dtype=np.int64)
        cursor = 0
        for i, nt in enumerate(self.valid_nt):
            self.event_starts[i] = cursor
            cursor += int(nt)
            self.event_stops[i] = cursor
            if i < self.n_events - 1:
                cursor += self.gap_nt
        self.nt = int(cursor)

    @property
    def shape(self) -> tuple[int, int]:
        return self.nt, self.nc

    @property
    def dtype(self) -> np.dtype:
        if self.transform == "fk":
            return np.dtype(np.float32)
        return np.dtype(self.data.dtype)

    @property
    def ndim(self) -> int:
        return 2

    @property
    def transform_options(self) -> list[dict[str, str]]:
        return [
            {"key": "raw", "label": "Raw"},
            {"key": "fk", "label": f"FK {self.fk_vmin:g}-{self.fk_vmax:g} m/s"},
        ]

    @property
    def transform(self) -> str:
        return self._transform

    @property
    def transform_revision(self) -> int:
        return self._transform_revision

    def set_transform(self, key: str) -> str:
        key = str(key)
        valid = {option["key"] for option in self.transform_options}
        if key not in valid:
            raise ValueError(f"Unknown transform {key!r}; valid options: {sorted(valid)}")
        if key != self._transform:
            self._transform = key
            self._transform_revision += 1
        return self._transform

    def cycle_transform(self, step: int = 1) -> str:
        keys = [option["key"] for option in self.transform_options]
        idx = keys.index(self._transform) if self._transform in keys else 0
        return self.set_transform(keys[(idx + int(step)) % len(keys)])

    def __getitem__(self, key: Any) -> np.ndarray:
        if isinstance(key, tuple):
            tkey = key[0]
            ckey = key[1] if len(key) > 1 else slice(None)
        else:
            tkey = key
            ckey = slice(None)

        channel_key, channel_count, channel_scalar = self._normalize_channel_key(ckey)

        if isinstance(tkey, (int, np.integer)):
            t = self._normalize_index(int(tkey), self.nt)
            out = self._read_slice(t, t + 1, 1, channel_key, channel_count, channel_scalar)
            return out[0, 0] if channel_scalar else out[0]

        if isinstance(tkey, slice):
            start, stop, step = tkey.indices(self.nt)
            if step <= 0:
                samples = np.arange(start, stop, step, dtype=np.int64)
                out = self._read_samples(samples, channel_key, channel_count, channel_scalar)
            else:
                out = self._read_slice(start, stop, step, channel_key, channel_count, channel_scalar)
            return out[:, 0] if channel_scalar else out

        samples = np.asarray(tkey, dtype=np.int64)
        samples = np.where(samples < 0, samples + self.nt, samples)
        if np.any((samples < 0) | (samples >= self.nt)):
            raise IndexError("time index out of range")
        out = self._read_samples(samples.reshape(-1), channel_key, channel_count, channel_scalar)
        if channel_scalar:
            out = out[:, 0]
        return out.reshape(samples.shape + (() if channel_scalar else (channel_count,)))

    def _read_slice(
        self,
        start: int,
        stop: int,
        step: int,
        channel_key: Any,
        channel_count: int,
        channel_scalar: bool,
    ) -> np.ndarray:
        out_nt = len(range(start, stop, step))
        out = np.zeros((out_nt, channel_count), dtype=self.dtype)
        if out_nt == 0 or channel_count == 0:
            return out

        event_idx = int(np.searchsorted(self.event_stops, start, side="right"))
        while event_idx < self.n_events:
            event_start = int(self.event_starts[event_idx])
            event_stop = int(self.event_stops[event_idx])
            if event_start >= stop:
                break
            if event_stop <= start:
                event_idx += 1
                continue

            seg_start = max(start, event_start)
            seg_stop = min(stop, event_stop)
            remainder = (seg_start - start) % step
            first = seg_start if remainder == 0 else seg_start + (step - remainder)
            if first < seg_stop:
                rel0 = first - event_start
                rel1 = seg_stop - event_start
                out0 = (first - start) // step
                block = self._read_event_block_nt_nc(
                    event_idx,
                    slice(rel0, rel1, step),
                    channel_key,
                    channel_scalar=channel_scalar,
                )
                out[out0:out0 + block.shape[0], :] = block

            event_idx += 1

        return out

    def _read_samples(
        self,
        samples: np.ndarray,
        channel_key: Any,
        channel_count: int,
        channel_scalar: bool,
    ) -> np.ndarray:
        out = np.zeros((samples.size, channel_count), dtype=self.dtype)
        for out_idx, sample in enumerate(samples):
            hit = self.event_for_sample(int(sample))
            if hit is None:
                continue
            event_idx, rel_t = hit
            if self.transform == "fk":
                raw = np.asarray(self._filtered_event_nt_nc(event_idx)[int(rel_t), channel_key], dtype=self.dtype)
            else:
                raw = np.asarray(self.data[event_idx, channel_key, int(rel_t)], dtype=self.dtype)
            if channel_scalar:
                out[out_idx, 0] = raw
            else:
                out[out_idx, :] = raw.reshape(-1)
        return out

    def _read_raw_event_nt_nc(self, event_idx: int) -> np.ndarray:
        nt = int(self.valid_nt[event_idx])
        if nt <= 0:
            return np.empty((0, self.nc), dtype=np.float32)
        return np.asarray(self.data[int(event_idx), :, :nt], dtype=np.float32).T

    def _filtered_event_nt_nc(self, event_idx: int) -> np.ndarray:
        event_idx = int(event_idx)
        cached = self._fk_cache.get(event_idx)
        if cached is not None:
            self._fk_cache.move_to_end(event_idx)
            return cached

        raw = self._read_raw_event_nt_nc(event_idx)
        if raw.shape[0] < 2 or raw.shape[1] < 2:
            filtered = raw.astype(np.float32, copy=True)
        else:
            from filark.dsp.fk import fk_fan_filter

            filtered = fk_fan_filter(
                raw,
                dt=1.0 / float(self.fs),
                dx=float(self.dx),
                vmin=self.fk_vmin,
                vmax=self.fk_vmax,
                taper_width=self.fk_taper_width,
            )
        filtered = np.ascontiguousarray(filtered, dtype=np.float32)

        if self.fk_cache_size > 0:
            self._fk_cache[event_idx] = filtered
            self._fk_cache.move_to_end(event_idx)
            while len(self._fk_cache) > self.fk_cache_size:
                self._fk_cache.popitem(last=False)
        return filtered

    def _read_event_block_nt_nc(
        self,
        event_idx: int,
        time_key: slice,
        channel_key: Any,
        *,
        channel_scalar: bool,
    ) -> np.ndarray:
        if self.transform == "fk":
            arr = np.asarray(self._filtered_event_nt_nc(event_idx)[time_key, channel_key], dtype=self.dtype)
            if channel_scalar:
                return arr.reshape(-1, 1)
            if arr.ndim == 1:
                return arr.reshape(-1, 1)
            return arr

        if isinstance(channel_key, np.ndarray):
            raw = self.data[event_idx, :, time_key][channel_key, :]
        else:
            raw = self.data[event_idx, channel_key, time_key]
        return self._event_block_to_nt_nc(raw, channel_scalar)

    def _event_block_to_nt_nc(self, raw: Any, channel_scalar: bool) -> np.ndarray:
        arr = np.asarray(raw, dtype=self.dtype)
        if channel_scalar:
            return arr.reshape(-1, 1)
        if arr.ndim == 1:
            return arr.reshape(1, -1).T
        return arr.T

    def event_for_sample(self, sample: int) -> tuple[int, int] | None:
        sample = self._normalize_index(sample, self.nt)
        event_idx = int(np.searchsorted(self.event_stops, sample, side="right"))
        if event_idx >= self.n_events:
            return None
        start = int(self.event_starts[event_idx])
        stop = int(self.event_stops[event_idx])
        if start <= sample < stop:
            return event_idx, sample - start
        return None

    def _normalize_channel_key(self, ckey: Any) -> tuple[Any, int, bool]:
        if isinstance(ckey, (int, np.integer)):
            return self._normalize_index(int(ckey), self.nc), 1, True
        if isinstance(ckey, slice):
            start, stop, step = ckey.indices(self.nc)
            return slice(start, stop, step), len(range(start, stop, step)), False
        channels = np.asarray(ckey, dtype=np.int64)
        channels = np.where(channels < 0, channels + self.nc, channels)
        if np.any((channels < 0) | (channels >= self.nc)):
            raise IndexError("channel index out of range")
        return channels, int(channels.size), False

    @staticmethod
    def _normalize_index(index: int, size: int) -> int:
        if index < 0:
            index += size
        if index < 0 or index >= size:
            raise IndexError(index)
        return int(index)

    def close(self) -> None:
        self._fk_cache.clear()
        self.group = None
        self.data = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open a Caltech event Zarr dataset for annotation.")
    parser.add_argument("path", help="Path to the .zarr directory.")
    parser.add_argument("--gap-nt", type=int, default=200, help="Zero-filled samples between adjacent events.")
    parser.add_argument("--transform", choices=("raw", "fk"), default="raw", help="Initial Tape transform.")
    parser.add_argument("--fk-vmin", type=float, default=800.0, help="FK min apparent velocity to keep (m/s).")
    parser.add_argument("--fk-vmax", type=float, default=5000.0, help="FK max apparent velocity to keep (m/s).")
    parser.add_argument("--fk-taper-width", type=float, default=200.0, help="FK velocity taper width.")
    parser.add_argument("--fk-cache-size", type=int, default=4, help="Number of filtered events to keep in memory.")
    parser.add_argument("--theme", choices=("dark", "light"), default="dark")
    parser.add_argument("--annotation-mode", choices=("bbox", "polyline", "mask"), default="polyline")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tape = CaltechZarrTape(
        args.path,
        gap_nt=args.gap_nt,
        transform=args.transform,
        fk_vmin=args.fk_vmin,
        fk_vmax=args.fk_vmax,
        fk_taper_width=args.fk_taper_width,
        fk_cache_size=args.fk_cache_size,
    )
    cfg = GuiConfig(
        source=tape,
        theme=args.theme,
        annotation_default_mode=args.annotation_mode,
    )
    run_app(argv=[], cfg=cfg)


if __name__ == "__main__":
    main()
