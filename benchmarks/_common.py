"""Shared helpers for the FiLark benchmark scripts.

Every script writes into ``<out>/<name>_<timestamp>/``:

* ``system.json``  — hardware / software description of the run
* ``*.csv``        — raw per-trial or per-sample measurements
* ``summary.csv``  — aggregated statistics (mean, std, min, max, n)
"""

from __future__ import annotations

import csv
import ctypes
import ctypes.util
import datetime as _dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def make_run_dir(out_root: str | Path, name: str) -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = Path(out_root).expanduser() / f"{name}_{stamp}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json(path: Path, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def summarize(rows: Sequence[dict], group_keys: Sequence[str], value_keys: Sequence[str]) -> list[dict]:
    """Group ``rows`` by ``group_keys`` and report mean/std/min/max/n of ``value_keys``."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r.get(k) for k in group_keys), []).append(r)
    out = []
    for gk, rs in groups.items():
        row = dict(zip(group_keys, gk))
        for vk in value_keys:
            vals = np.array([float(r[vk]) for r in rs if r.get(vk) is not None], dtype=float)
            vals = vals[np.isfinite(vals)]
            row[f"{vk}_n"] = int(vals.size)
            if vals.size:
                row[f"{vk}_mean"] = float(vals.mean())
                row[f"{vk}_std"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
                row[f"{vk}_min"] = float(vals.min())
                row[f"{vk}_max"] = float(vals.max())
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# System description
# ---------------------------------------------------------------------------

def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def system_info(extra_paths: Iterable[str | Path] = ()) -> dict:
    info: dict = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_count_logical": os.cpu_count(),
    }
    if sys.platform == "darwin":
        info["cpu"] = _sh(["sysctl", "-n", "machdep.cpu.brand_string"])
        info["model"] = _sh(["sysctl", "-n", "hw.model"])
        try:
            info["ram_gb"] = int(_sh(["sysctl", "-n", "hw.memsize"])) / 1024**3
        except ValueError:
            pass
    else:
        for line in _sh(["lscpu"]).splitlines():
            if line.startswith("Model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
        try:
            with open("/proc/meminfo") as f:
                info["ram_gb"] = int(f.readline().split()[1]) / 1024**2
        except Exception:
            pass
    for mod in ("numpy", "scipy", "h5py", "torch", "vispy", "PySide6", "dascore", "daspy", "filark"):
        try:
            m = __import__(mod)
            info[f"ver_{mod}"] = getattr(m, "__version__", "?")
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
            info["gpu_mem_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            info["nvidia_driver"] = _sh(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            info["gpu"] = "Apple MPS (integrated, unified memory)"
    except Exception:
        pass
    disks = {}
    for p in extra_paths:
        p = Path(p)
        disks[str(p)] = _sh(["df", "-h", str(p)]).splitlines()[-1:] or ""
    if disks:
        info["storage"] = disks
    if sys.platform == "darwin":
        # Display count, resolution, scaling and refresh rate change render/upload cost.
        disp = _sh(["system_profiler", "SPDisplaysDataType"])
        info["displays"] = [ln.strip() for ln in disp.splitlines()
                            if ln.strip() and not ln.strip().startswith(("Graphics/Displays", "Chipset", "Type", "Bus", "Vendor", "Metal"))]
        info["power"] = _sh(["pmset", "-g", "batt"]).splitlines()[:1]
    repo = Path(__file__).resolve().parents[1]
    info["git_head"] = _sh(["git", "-C", str(repo), "rev-parse", "HEAD"])
    info["git_dirty"] = _sh(["git", "-C", str(repo), "status", "--short", "--untracked-files=no"]).splitlines()
    return info


# ---------------------------------------------------------------------------
# Memory probes
# ---------------------------------------------------------------------------

def rss_mb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 1024**2


class _RUsageInfoV2(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
    ]


_libproc = None


def phys_footprint_mb() -> float:
    """macOS ``phys_footprint`` (what Activity Monitor shows as "Memory").

    On Apple silicon this includes GPU/IOAccelerator allocations owned by the
    process (OpenGL/Metal textures live in unified memory), so it is the
    closest single number to "CPU + GPU memory" on this platform.
    Returns nan on other platforms.
    """
    global _libproc
    if sys.platform != "darwin":
        return float("nan")
    try:
        if _libproc is None:
            _libproc = ctypes.CDLL(ctypes.util.find_library("proc"))
        ri = _RUsageInfoV2()
        rc = _libproc.proc_pid_rusage(os.getpid(), 2, ctypes.byref(ri))  # RUSAGE_INFO_V2
        if rc != 0:
            return float("nan")
        return ri.ri_phys_footprint / 1024**2
    except Exception:
        return float("nan")


def disk_read_bytes() -> float:
    """Bytes read from storage by this process so far (nan if unavailable)."""
    try:
        import psutil
        return float(psutil.Process().io_counters().read_bytes)
    except Exception:
        pass
    if sys.platform == "darwin":
        try:
            if _libproc is None:
                phys_footprint_mb()
            ri = _RUsageInfoV2()
            if _libproc.proc_pid_rusage(os.getpid(), 2, ctypes.byref(ri)) == 0:
                return float(ri.ri_diskio_bytesread)
        except Exception:
            pass
    return float("nan")


def nvidia_process_mem_mb() -> float:
    """Total GPU memory used by this process according to nvidia-smi (incl. OpenGL)."""
    out = _sh(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
    pid = str(os.getpid())
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == pid:
            try:
                return float(parts[1])
            except ValueError:
                pass
    return float("nan")


# ---------------------------------------------------------------------------
# OS page cache
# ---------------------------------------------------------------------------

def drop_os_cache(verbose: bool = True) -> bool:
    """Flush the OS file cache so the next read comes from storage.

    macOS: ``sudo -n purge``; Linux: ``sync; echo 3 > /proc/sys/vm/drop_caches``.
    Needs password-less sudo for that command (``sudo -v`` beforehand is enough
    for a short session). Returns True on success.
    """
    if sys.platform == "darwin":
        cmd = ["sudo", "-n", "purge"]
    else:
        cmd = ["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0
    if verbose:
        msg = f"{time.perf_counter() - t0:.1f}s" if ok else f"FAILED: {r.stderr.strip()}"
        print(f"[cache] drop OS cache: {msg}", flush=True)
    return ok


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def list_files(path: str | Path, pattern: str = "*.h5", limit: int | None = None) -> list[Path]:
    p = Path(path).expanduser()
    files = [p] if p.is_file() else sorted(f for f in p.glob(pattern) if not f.name.startswith("._"))
    if limit:
        files = files[: int(limit)]
    if not files:
        raise FileNotFoundError(f"no files matching {pattern} under {p}")
    return files


def open_fileset(files: Sequence[Path], **kwargs):
    from filark.io.fileset import TapeFileSet
    return TapeFileSet([str(f) for f in files], **kwargs)


def describe_source(tape, files: Sequence[Path] | None = None) -> dict:
    """Essential acquisition parameters for the data-provenance table."""
    shape = tuple(int(v) for v in tape.shape)
    dims = str(getattr(tape, "dims", "nt_nc"))
    nt, nc = (shape[0], shape[1]) if dims == "nt_nc" else (shape[1], shape[0])
    fs = float(getattr(tape, "fs", float("nan")))
    d = {
        "dims": dims,
        "n_channels": nc,
        "n_samples": nt,
        "fs_hz": fs,
        "dx_m": float(getattr(tape, "dx", float("nan"))),
        "duration_s": nt / fs if fs > 0 else float("nan"),
        "dtype": str(getattr(tape, "dtype", "")),
        "n_gaps": len(getattr(tape, "gaps", []) or []),
    }
    if files:
        d["n_files"] = len(files)
        d["bytes_on_disk"] = int(sum(Path(f).stat().st_size for f in files))
        d["gb_on_disk"] = d["bytes_on_disk"] / 1024**3
        d["first_file"] = str(files[0])
    return d


def timer() -> float:
    return time.perf_counter()
