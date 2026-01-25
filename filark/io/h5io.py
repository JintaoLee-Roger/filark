from __future__ import annotations
from typing import Tuple, Any
import h5py
import numpy as np


class H5Source:

    def __init__(
        self,
        filepath: str,
        datakey: str = 'Acquisition/Raw[0]/RawData',
        fs_key: str = None,
        dx_key: str = None,
        dims_key: str = None,
        dxunit_key: str = None,
    ):
        self.file = h5py.File(filepath, 'r')
        self.dataset = self.file[datakey]  # 假设数据存储在根目录下的 'data' 数据集
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
