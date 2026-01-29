from __future__ import annotations
from dataclasses import dataclass, field
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
    ):
        self.p = Path(path)
        if not self.p.exists():
            raise FileNotFoundError(self.p)

        self.file = h5py.File(path, 'r')
        self.dataset = self.file[datakey]
        if fs_key is None:
            fs_key = ('Acquisition/Raw[0]', 'OutputDataRate')

        if dx_key is None:
            dx_key = ('Acquisition', 'SpatialSamplingInterval')

        self._fs = self.file[fs_key[0]].attrs.get(fs_key[1], 1.0)
        self._dx = self.file[dx_key[0]].attrs.get(dx_key[1], 1.0)

        if dxunit_key is None:
            dxunit_key = ('Acquisition', 'SpatialSamplingIntervalUnit')
        self._dx_unit = self.file[dxunit_key[0]].attrs.get(dxunit_key[1], 'm')
        self._dx_unit = str(self._dx_unit, 'utf-8') if isinstance(self._dx_unit, np.bytes_) else str(self._dx_unit)
        if dims_key is None:
            dims_key = ('Acquisition/Raw[0]/RawData', 'Dimensions')
        self._dims = self.file[dims_key[0]].attrs.get(dims_key[1], None)
        if self._dims is not None:
            if isinstance(self._dims, np.bytes_):
                self._dims = str(self._dims, 'utf-8')
            if isinstance(self._dims, str):
                self._dims = str(self._dims.split(',')[0])
            elif isinstance(self._dims, (list, np.ndarray)):
                self._dims = str(self._dims[0])
                if self._dims.startswith("b'time'"):
                    self._dims = 'time'
            else:
                self._dims is None 
        if self._dims is not None and self._dims.lower() != 'time':
            self._dims = 'nc_nt'
        else:
            self._dims = 'nt_nc'

    @property
    def shape(self) -> Tuple[int, int]:
        return self.dataset.shape

    @property
    def dtype(self) -> np.dtype:
        return self.dataset.dtype

    @property
    def fs(self) -> float:
        return self._fs

    @property
    def dx(self) -> float:
        return self._dx

    @property
    def dims(self):
        return self._dims

    @property
    def ndim(self) -> int:
        return self.dataset.ndim

    @property
    def dx_unit(self) -> str:
        return self._dx_unit

    def __getitem__(self, key: Any) -> np.ndarray:
        return self.dataset[key]

    def close(self) -> None:
        self.file.close()




ArrayLike2D = Union[np.ndarray, np.memmap]

@dataclass
class NpyTape:
    source: Union[str, os.PathLike, ArrayLike2D]
    dims: Dims
    fs: float = 1.0
    dx: float = 1.0
    dx_unit: str = "m"

    mmap_mode: Optional[str] = "r" 
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
        mmap_mode: Optional[str] = "r",
    ) -> "NpyTape":
        return cls(path, dims=dims, fs=fs, dx=dx, dx_unit=dx_unit, mmap_mode=mmap_mode)

    @classmethod
    def from_array(
        cls,
        arr: ArrayLike2D,
        *,
        dims: Dims,
        fs: float = 1.0,
        dx: float = 1.0,
        dx_unit: str = "m",
    ) -> "NpyTape":
        return cls(arr, dims=dims, fs=fs, dx=dx, dx_unit=dx_unit, mmap_mode=None)

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
        mm = np.memmap(self.path, mode="r", dtype=self.dtype, offset=0, shape=(expected_count,))
        if self.dims == "nt_nc":
            self._arr = mm.reshape((nt, nc), order=self.order)
        else:
            self._arr = mm.reshape((nc, nt), order=self.order)

    @property
    def shape(self) -> Tuple[int, int]:
        assert self._arr is not None
        return self._arr.shape

    @property
    def dtype_(self) -> np.dtype:
        return np.dtype(self.dtype)

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
