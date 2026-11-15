# Copyright (c) 2026 Jintao Li. 
# Zhejiang University (ZJU).
# 
# Licensed under the MIT License.

"""Concrete tape implementations for NumPy, HDF5, SEG-Y, and related inputs."""

from __future__ import annotations
from dataclasses import dataclass, field
import datetime
import os
from pathlib import Path
from typing import Literal, Optional, Tuple, Any, Union
import h5py
import numpy as np

Dims = Literal["nt_nc", "nc_nt"]


class H5Tape:

    def __init__(
        self,
        path: str,
        datakey: str = 'Acquisition/Raw[0]/RawData',
        fs_key: Tuple = None,
        dx_key: Tuple = None,
        dims_key: Tuple = None,
        dxunit_key: Tuple = None,
        gauge_length_key: Tuple = None,
        start_time: Optional[float] = None,
    ):
        self.p = Path(path)
        if not self.p.exists():
            raise FileNotFoundError(self.p)

        self._datakey = datakey
        self.file = h5py.File(path, 'r')
        self.dataset = self.file[datakey]
        # Cached so metadata stays available after ``release()``.
        self._shape = tuple(int(v) for v in self.dataset.shape)
        self._dtype = self.dataset.dtype
        self._ndim = int(self.dataset.ndim)

        if fs_key is None:
            fs_key = ('Acquisition/Raw[0]', 'OutputDataRate')
        try:
            self._fs = float(self.file[fs_key[0]].attrs.get(fs_key[1], 1.0))
        except Exception:
            self._fs = 1.0

        if dx_key is None:
            dx_key = ('Acquisition', 'SpatialSamplingInterval')
        try:
            self._dx = float(self.file[dx_key[0]].attrs.get(dx_key[1], 1.0))
        except Exception:
            self._dx = 1.0

        if dxunit_key is None:
            dxunit_key = ('Acquisition', 'SpatialSamplingIntervalUnit')
        try:
            raw_unit = self.file[dxunit_key[0]].attrs.get(dxunit_key[1], 'm')
            self._dx_unit = str(raw_unit, 'utf-8') if isinstance(raw_unit, np.bytes_) else str(raw_unit)
        except Exception:
            self._dx_unit = 'm'

        self._gauge_length: Optional[float] = None
        if gauge_length_key is None:
            gauge_length_candidates = [
                (datakey, "gauge_length"),
                (datakey, "GaugeLength"),
                ("Acquisition", "GaugeLength"),
                ("Acquisition/Raw[0]", "GaugeLength"),
            ]
        else:
            gauge_length_candidates = [gauge_length_key]
        for group_key, attr_key in gauge_length_candidates:
            try:
                raw_gauge = self.file[group_key].attrs.get(attr_key, None)
                if raw_gauge is None:
                    continue
                if isinstance(raw_gauge, np.ndarray):
                    if raw_gauge.size == 0:
                        continue
                    raw_gauge = raw_gauge.reshape(-1)[0]
                self._gauge_length = float(raw_gauge)
                break
            except Exception:
                continue

        if dims_key is None:
            dims_key = ('Acquisition/Raw[0]/RawData', 'Dimensions')
        try:
            raw_dims = self.file[dims_key[0]].attrs.get(dims_key[1], None)
        except Exception:
            raw_dims = None
        if raw_dims is not None:
            if isinstance(raw_dims, np.bytes_):
                raw_dims = str(raw_dims, 'utf-8')
            if isinstance(raw_dims, str):
                raw_dims = str(raw_dims.split(',')[0])
            elif isinstance(raw_dims, (list, np.ndarray)):
                raw_dims = str(raw_dims[0])
                if raw_dims.startswith("b'time'"):
                    raw_dims = 'time'
            else:
                raw_dims = None
        if raw_dims is not None and raw_dims.lower() != 'time':
            self._dims = 'nc_nt'
        else:
            self._dims = 'nt_nc'

        # Start time: accept explicit value or try common H5 attribute paths
        if start_time is not None:
            self._start_time: Optional[float] = float(start_time)
        else:
            self._start_time = self._try_read_start_time()

    def _try_read_start_time(self) -> Optional[float]:
        """Try to read acquisition start time from common H5 attribute locations."""
        # ISO 8601 string candidates (e.g. '1970-01-21T06:38:09.360281+00:00')
        iso_candidates = [
            ('Acquisition/Raw[0]/RawData', 'PartStartTime'),
            ('Acquisition/Raw[0]', 'PartStartTime'),
        ]
        for group_key, attr_key in iso_candidates:
            try:
                val = self.file[group_key].attrs.get(attr_key, None)
                if val is not None:
                    s = val.decode() if isinstance(val, (bytes, bytearray)) else str(val)
                    dt = datetime.datetime.fromisoformat(s)
                    return dt.timestamp()
            except Exception:
                continue

        # Numeric Unix timestamp candidates
        numeric_candidates = [
            ('Acquisition/Raw[0]', 'GPSTimeStamp'),
            ('Acquisition/Raw[0]', 'LocalTimestamp'),
            ('Acquisition', 'AcquisitionStart'),
            ('Acquisition', 'MinimumLEPTimestamp'),
        ]
        for group_key, attr_key in numeric_candidates:
            try:
                val = self.file[group_key].attrs.get(attr_key, None)
                if val is not None:
                    return float(val)
            except Exception:
                continue
        return None

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def fs(self) -> float:
        return self._fs

    @fs.setter
    def fs(self, value: float):
        self._fs = float(value)

    @property
    def dx(self) -> float:
        return self._dx

    @dx.setter
    def dx(self, value: float):
        self._dx = float(value)

    @property
    def dims(self):
        return self._dims

    @dims.setter
    def dims(self, value: str):
        self._dims = str(value)

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def dx_unit(self) -> str:
        return self._dx_unit

    @dx_unit.setter
    def dx_unit(self, value: str):
        self._dx_unit = str(value)

    @property
    def gauge_length(self) -> Optional[float]:
        return self._gauge_length

    @gauge_length.setter
    def gauge_length(self, value: Optional[float]):
        self._gauge_length = None if value is None else float(value)

    @property
    def start_time(self) -> Optional[float]:
        return self._start_time

    @property
    def is_open(self) -> bool:
        return self.file is not None

    def _ensure_open(self) -> None:
        if self.file is None:
            self.file = h5py.File(self.p, 'r')
            self.dataset = self.file[self._datakey]

    def release(self) -> None:
        """Close the HDF5 handle but keep the metadata; the next read reopens it.

        Each open HDF5 dataset keeps its own chunk cache, so a file set over
        many files bounds memory by releasing handles it is not using.
        """
        if self.file is not None:
            self.file.close()
        self.file = None
        self.dataset = None

    def __getitem__(self, key: Any) -> np.ndarray:
        self._ensure_open()
        return self.dataset[key]

    def close(self) -> None:
        self.release()




ArrayLike2D = Union[np.ndarray, np.memmap]

@dataclass
class NpyTape:
    source: Union[str, os.PathLike, ArrayLike2D]
    dims: Dims
    fs: float = 1.0
    dx: float = 1.0
    dx_unit: str = "m"
    gauge_length: Optional[float] = None
    start_time: Optional[float] = None

    mmap_mode: Optional[str] = "c"
    _arr: Optional[ArrayLike2D] = field(init=False, default=None, repr=False)
    _owns_ref: bool = field(init=False, default=False, repr=False)

    def __post_init__(self):
        if self.dims not in ("nt_nc", "nc_nt"):
            raise ValueError(f"Invalid dims: {self.dims}")

        src = self.source

        if isinstance(src, (str, os.PathLike)):
            self._arr = np.load(str(src), mmap_mode=self.mmap_mode)
            self._owns_ref = True
        elif isinstance(src, (np.ndarray, np.memmap)):
            self._arr = src
            self._owns_ref = False
        else:
            raise TypeError(
                "source must be a file path or a numpy ndarray/memmap, "
                f"got {type(src)!r}"
            )

        if self._arr.ndim != 2:
            raise ValueError(f"Only 2D supported, got ndim={self._arr.ndim}")

    @classmethod
    def from_path(
        cls,
        path: Union[str, os.PathLike],
        *,
        dims: Dims,
        fs: float = 1.0,
        dx: float = 1.0,
        dx_unit: str = "m",
        gauge_length: Optional[float] = None,
        mmap_mode: Optional[str] = "c",
    ) -> "NpyTape":
        return cls(path, dims=dims, fs=fs, dx=dx, dx_unit=dx_unit, gauge_length=gauge_length, mmap_mode=mmap_mode)

    @classmethod
    def from_array(
        cls,
        arr: ArrayLike2D,
        *,
        dims: Dims,
        fs: float = 1.0,
        dx: float = 1.0,
        dx_unit: str = "m",
        gauge_length: Optional[float] = None,
    ) -> "NpyTape":
        return cls(arr, dims=dims, fs=fs, dx=dx, dx_unit=dx_unit, gauge_length=gauge_length, mmap_mode=None)

    @property
    def arr(self) -> ArrayLike2D:
        if self._arr is None:
            raise RuntimeError("NpyTape is closed.")
        return self._arr

    @property
    def shape(self) -> Tuple[int, int]:
        return self.arr.shape

    @property
    def dtype(self) -> np.dtype:
        return self.arr.dtype

    @property
    def ndim(self) -> int:
        return self.arr.ndim

    def __getitem__(self, key: Any) -> np.ndarray:
        return self.arr[key]

    def close(self) -> None:
        self._arr = None

    def __enter__(self) -> "NpyTape":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()



@dataclass
class BinTape:
    path: str
    nt: int
    nc: int
    dtype: np.dtype
    dims: Dims
    fs: float = 1.0
    dx: float = 1.0
    dx_unit: str = "m"
    gauge_length: Optional[float] = None
    start_time: Optional[float] = None
    order: str = "C"          # memmap reshape order
    strict_size: bool = True  # True: file size must match exactly; False: allow extra bytes

    _arr: Optional[np.ndarray] = field(init=False, default=None, repr=False)

    def __post_init__(self):
        p = Path(self.path)
        if not p.exists():
            raise FileNotFoundError(self.path)
        if not p.is_file():
            raise ValueError(f"Not a file: {self.path}")

        # normalize dtype
        self.dtype = np.dtype(self.dtype)

        if self.dims not in ("nt_nc", "nc_nt"):
            raise ValueError(f"Invalid dims: {self.dims}")

        nt = int(self.nt)
        nc = int(self.nc)
        if nt <= 0 or nc <= 0:
            raise ValueError(f"nt/nc must be positive, got nt={nt}, nc={nc}")

        expected_count = nt * nc
        itemsize = int(self.dtype.itemsize)
        expected_bytes = expected_count * itemsize

        actual_bytes = p.stat().st_size
        if actual_bytes < expected_bytes:
            raise ValueError(
                f"File too small for declared shape.\n"
                f"path={self.path}\n"
                f"expected at least {expected_bytes} bytes (= nt*nc*dtype.itemsize = {nt}*{nc}*{itemsize}),\n"
                f"but file has {actual_bytes} bytes."
            )

        if self.strict_size and actual_bytes != expected_bytes:
            raise ValueError(
                f"File size mismatch (strict_size=True).\n"
                f"path={self.path}\n"
                f"expected exactly {expected_bytes} bytes (= {expected_count} elements of {self.dtype}),\n"
                f"but file has {actual_bytes} bytes.\n"
                f"If the file contains a header or trailing bytes, set strict_size=False."
            )

        # interpret file as flat memmap then reshape
        mm = np.memmap(self.path, mode="c", dtype=self.dtype, offset=0, shape=(expected_count,))
        if self.dims == "nt_nc":
            self._arr = mm.reshape((nt, nc), order=self.order)
        else:
            self._arr = mm.reshape((nc, nt), order=self.order)

    @property
    def shape(self) -> Tuple[int, int]:
        assert self._arr is not None
        return self._arr.shape

    @property
    def ndim(self) -> int:
        assert self._arr is not None
        return self._arr.ndim

    def __getitem__(self, key: Any) -> np.ndarray:
        assert self._arr is not None
        return self._arr[key]

    def close(self) -> None:
        # release references so OS can close mapping when GC happens
        self._arr = None


class ArrayTape:
    """Wrap any 2-D array-like object as a Tape.

    ``data`` must support:

    - ``data.shape`` → ``(int, int)``
    - ``data.dtype`` → ``np.dtype``
    - ``data[key]`` → ``np.ndarray``   (numpy-compatible slice notation)

    This works with ``h5py.Dataset``, ``zarr`` arrays, ``numpy`` arrays /
    memmaps, ``segyio`` trace wrappers, or any custom ``__getitem__``
    adapter.  Pass ``close_fn`` for lifecycle cleanup (e.g.
    ``close_fn=lambda: f.close()`` when ``data`` is an open file's dataset).

    All metadata (``dims``, ``fs``, ``dx``, etc.) default to safe values and
    can be changed at any time via the setters or the load panel's
    "Apply Overrides" button.

    Examples
    --------
    >>> arr = np.load("data.npy")
    >>> tape = ArrayTape(arr, dims="nt_nc", fs=500.0, dx=1.0)

    >>> import h5py
    >>> f = h5py.File("data.h5", "r")
    >>> tape = ArrayTape(f["data"], fs=1000.0, close_fn=f.close)

    >>> import segyio, numpy as np
    >>> with segyio.open("data.sgy") as f:
    ...     arr = np.stack([f.trace[i] for i in range(f.tracecount)], axis=1)
    >>> tape = ArrayTape(arr, dims="nt_nc", fs=500.0)
    """

    def __init__(
        self,
        data,
        *,
        dims: str = "nt_nc",
        fs: float = 1.0,
        dx: float = 1.0,
        dx_unit: str = "m",
        gauge_length: Optional[float] = None,
        start_time: Optional[float] = None,
        close_fn=None,
    ):
        if not hasattr(data, "shape") or len(data.shape) < 2:
            raise ValueError("ArrayTape requires a 2D array-like with a .shape attribute.")
        if data.shape[0] == 0 or data.shape[1] == 0:
            raise ValueError(f"ArrayTape data must be non-empty, got shape {data.shape}.")
        if dims not in ("nt_nc", "nc_nt"):
            raise ValueError(f"dims must be 'nt_nc' or 'nc_nt', got {dims!r}.")

        self._data = data
        self._dims = str(dims)
        self._fs = float(fs)
        self._dx = float(dx)
        self._dx_unit = str(dx_unit)
        self._gauge_length: Optional[float] = None if gauge_length is None else float(gauge_length)
        self._start_time: Optional[float] = float(start_time) if start_time is not None else None
        self._close_fn = close_fn

    # ---------- Tape protocol ----------

    @property
    def shape(self) -> Tuple[int, int]:
        return self._data.shape[0], self._data.shape[1]

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._data.dtype)

    @property
    def ndim(self) -> int:
        return 2

    @property
    def dims(self) -> str:
        return self._dims

    @dims.setter
    def dims(self, value: str) -> None:
        self._dims = str(value)

    @property
    def fs(self) -> float:
        return self._fs

    @fs.setter
    def fs(self, value: float) -> None:
        self._fs = float(value)

    @property
    def dx(self) -> float:
        return self._dx

    @dx.setter
    def dx(self, value: float) -> None:
        self._dx = float(value)

    @property
    def dx_unit(self) -> str:
        return self._dx_unit

    @dx_unit.setter
    def dx_unit(self, value: str) -> None:
        self._dx_unit = str(value)

    @property
    def gauge_length(self) -> Optional[float]:
        return self._gauge_length

    @gauge_length.setter
    def gauge_length(self, value: Optional[float]) -> None:
        self._gauge_length = None if value is None else float(value)

    @property
    def start_time(self) -> Optional[float]:
        return self._start_time

    def __getitem__(self, key: Any) -> np.ndarray:
        return np.asarray(self._data[key])

    def close(self) -> None:
        if self._close_fn is not None:
            try:
                self._close_fn()
            except Exception:
                pass
        self._data = None  # type: ignore[assignment]
        self._close_fn = None
