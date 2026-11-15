# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""Lightweight memory monitor for FiLark.

Tracks RSS (physical RAM) and GPU VRAM at key lifecycle events and on demand.

RSS  — Resident Set Size: physical memory pages actually mapped into the process.
       Includes Python interpreter overhead, all numpy arrays, HDF5 page cache,
       VisPy buffers, etc. Measured via psutil.

VRAM — GPU memory allocated by PyTorch (``torch.cuda.memory_allocated``).
       Reflects ring-buffer textures loaded via OpenGL *are not counted here*
       (OpenGL VRAM is managed by the driver, not torch).  If you need total
       GPU VRAM including OpenGL, use a system tool (e.g. ``nvidia-smi``).

Usage
-----
    from filark.utils.memory import MemoryMonitor
    mon = MemoryMonitor()
    mon.snapshot("startup")
    # ... load data ...
    mon.snapshot("after load")
    mon.print_report()
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from typing import Optional


# ── optional dependencies ────────────────────────────────────────────────────

def _rss_mb() -> float:
    """Current process RSS in MB, or nan if psutil is unavailable."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 ** 2)
    except Exception:
        return float("nan")


def _vram_mb() -> float:
    """PyTorch CUDA allocated memory in MB, or nan if unavailable.

    Note: OpenGL VRAM (ring-buffer textures) is *not* included here.
    Use ``nvidia-smi`` or a system-level tool for total GPU memory.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)
    except Exception:
        pass
    return float("nan")


def _psutil_available() -> bool:
    try:
        import psutil  # noqa: F401
        return True
    except ImportError:
        return False


def _torch_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ── snapshot dataclass ───────────────────────────────────────────────────────

@dataclass
class MemSnapshot:
    label: str
    wall_time: float          # time.perf_counter() at capture
    rss_mb: float             # nan if psutil unavailable
    vram_mb: float            # nan if torch/CUDA unavailable

    @property
    def wall_elapsed(self) -> float:
        return time.perf_counter() - self.wall_time

    def fmt_rss(self) -> str:
        return f"{self.rss_mb:.1f} MB" if not math.isnan(self.rss_mb) else "n/a"

    def fmt_vram(self) -> str:
        return f"{self.vram_mb:.1f} MB" if not math.isnan(self.vram_mb) else "n/a"


# ── monitor ──────────────────────────────────────────────────────────────────

class MemoryMonitor:
    """Collects named memory snapshots and reports deltas.

    Parameters
    ----------
    print_on_snapshot : bool
        If True, print each snapshot to stderr immediately when taken.
    """

    def __init__(self, *, print_on_snapshot: bool = False):
        self._snapshots: list[MemSnapshot] = []
        self._print_on_snapshot = bool(print_on_snapshot)
        self._t0 = time.perf_counter()

    # ── public API ───────────────────────────────────────────────────────────

    def snapshot(self, label: str) -> MemSnapshot:
        """Take and store a memory snapshot with a descriptive label."""
        s = MemSnapshot(
            label=label,
            wall_time=time.perf_counter(),
            rss_mb=_rss_mb(),
            vram_mb=_vram_mb(),
        )
        self._snapshots.append(s)
        if self._print_on_snapshot:
            self._print_one(s, len(self._snapshots) - 1)
        return s

    def current(self) -> MemSnapshot:
        """Take a snapshot without storing it."""
        return MemSnapshot(
            label="<current>",
            wall_time=time.perf_counter(),
            rss_mb=_rss_mb(),
            vram_mb=_vram_mb(),
        )

    def print_current(self, label: str = "now") -> None:
        """Print a one-line memory summary to stderr (no storage)."""
        s = self.current()
        rss_str  = f"RSS={s.fmt_rss()}"
        vram_str = f"VRAM={s.fmt_vram()}"
        # delta vs first snapshot
        delta_str = ""
        if self._snapshots:
            first = self._snapshots[0]
            if not math.isnan(s.rss_mb) and not math.isnan(first.rss_mb):
                delta_str = f"  ΔRSS={s.rss_mb - first.rss_mb:+.1f} MB vs start"
        elapsed = time.perf_counter() - self._t0
        print(f"[MEM] {label:30s}  {rss_str:>14}  {vram_str:>14}  "
              f"t={elapsed:.1f}s{delta_str}",
              file=sys.stderr, flush=True)

    def print_report(self) -> None:
        """Print the full snapshot history with deltas to stderr."""
        if not self._snapshots:
            print("[MEM] No snapshots recorded.", file=sys.stderr)
            return

        has_psutil = _psutil_available()
        has_cuda   = _torch_cuda_available()

        print("[MEM] ── Memory Snapshot Report ──────────────────────────────", file=sys.stderr)
        if not has_psutil:
            print("[MEM]  RSS: n/a  (install psutil: pip install psutil)", file=sys.stderr)
        if not has_cuda:
            print("[MEM]  VRAM(torch): n/a  (no CUDA or PyTorch not installed; column omitted on macOS)", file=sys.stderr)
        print(f"[MEM]  {'#':>3}  {'Label':<30}  {'t(s)':>7}  {'RSS':>10}  "
              f"{'ΔRSS':>9}  {'VRAM':>10}  {'ΔVRAM':>9}", file=sys.stderr)
        print("[MEM]  " + "-" * 85, file=sys.stderr)

        first = self._snapshots[0]
        for i, s in enumerate(self._snapshots):
            t_str = f"{s.wall_time - self._t0:.1f}"
            rss_str  = f"{s.rss_mb:.1f}" if not math.isnan(s.rss_mb) else "n/a"
            vram_str = f"{s.vram_mb:.1f}" if not math.isnan(s.vram_mb) else "n/a"

            d_rss  = s.rss_mb  - first.rss_mb
            d_vram = s.vram_mb - first.vram_mb
            d_rss_str  = f"{d_rss:+.1f}"  if not math.isnan(d_rss)  else "n/a"
            d_vram_str = f"{d_vram:+.1f}" if not math.isnan(d_vram) else "n/a"

            print(f"[MEM]  {i:>3}  {s.label:<30}  {t_str:>7}  "
                  f"{rss_str:>9}M  {d_rss_str:>8}M  "
                  f"{vram_str:>9}M  {d_vram_str:>8}M",
                  file=sys.stderr)

        # summary: net delta from first to last
        last = self._snapshots[-1]
        net_rss  = last.rss_mb  - first.rss_mb
        net_vram = last.vram_mb - first.vram_mb
        print("[MEM]  " + "-" * 85, file=sys.stderr)
        net_rss_str  = f"{net_rss:+.1f} MB"  if not math.isnan(net_rss)  else "n/a"
        net_vram_str = f"{net_vram:+.1f} MB" if not math.isnan(net_vram) else "n/a"
        print(f"[MEM]  Net change (first→last):  RSS {net_rss_str}   VRAM {net_vram_str}",
              file=sys.stderr)
        print("[MEM] ────────────────────────────────────────────────────────", file=sys.stderr)

    def clear(self) -> None:
        """Remove all stored snapshots."""
        self._snapshots.clear()

    def __len__(self) -> int:
        return len(self._snapshots)
