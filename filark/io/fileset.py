import h5py 
import numpy as np
from pathlib import Path
from .h5io import H5Source

class H5DASFileSet:
    def __init__(self, file_paths, **kwargs):
        if not isinstance(file_paths, list):
            file_paths = list(sorted(Path(file_paths).glob('*.h5')))
            file_paths = [f for f in file_paths if not f.name.startswith('.')]

        # print(file_paths)
        self.files = [H5Source(str(p), **kwargs) for p in file_paths]

        
        # 预计算每个文件的 shape 和起始位置
        self.shapes = [f.shape for f in self.files]
        self.nts = [s[0] for s in self.shapes]
        self.cum_nts = np.cumsum([0] + self.nts)
        
        # 基础属性
        self.nt = self.cum_nts[-1]
        self.nc = self.shapes[0][1] if self.files else 0
        self._dtype = self.files[0].dtype
        self._ndim = 2

    @property
    def shape(self):
        return (self.nt, self.nc)

    @property
    def ndim(self):
        return self._ndim

    @property
    def dtype(self):
        return self._dtype

    @property
    def fs(self):
        return self.files[0].fs if self.files else None

    @property
    def dx(self):
        return self.files[0].dx if self.files else None

    @property
    def dx_unit(self):
        return self.files[0].dx_unit if self.files else None

    @property
    def dims(self):
        return self.files[0].dims if self.files else None

    def __len__(self):
        return self.nt

    def __getitem__(self, key):
        # 1. 格式化 key (支持 int, slice, tuple)
        if isinstance(key, tuple):
            row_key = key[0]
            col_key = key[1] if len(key) > 1 else slice(None)
        else:
            row_key = key
            col_key = slice(None)

        # 2. 处理单行索引用例 (int)
        if isinstance(row_key, (int, np.integer)):
            if row_key < 0: row_key += self.nt
            idx = np.searchsorted(self.cum_nts, row_key, side='right') - 1
            rel_idx = row_key - self.cum_nts[idx]
            return self.files[idx][rel_idx, col_key]

        # 3. 处理切片索引用例 (slice)
        elif isinstance(row_key, slice):
            start, stop, step = row_key.indices(self.nt)
            if step != 1:
                # 步长不为1时，退化为简单提取（h5py 对非连续提取支持较慢）
                return np.array([self.__getitem__((i, col_key)) for i in range(start, stop, step)])

            # 跨文件切片核心逻辑
            p_start = np.searchsorted(self.cum_nts, start, side='right') - 1
            p_stop = np.searchsorted(self.cum_nts, stop - 1, side='right') - 1

            chunks = []
            for i in range(p_start, p_stop + 1):
                f_start = max(start, self.cum_nts[i])
                f_stop = min(stop, self.cum_nts[i+1])
                
                rel_start = f_start - self.cum_nts[i]
                rel_stop = f_stop - self.cum_nts[i]
                
                # 直接从 H5 dataset 切片，这是 IO 发生的地方
                data = self.files[i][rel_start:rel_stop, col_key]
                chunks.append(data)
            
            return np.concatenate(chunks, axis=0) if chunks else np.array([], dtype=self.dtype)

    def close(self):
        """记得关闭所有文件句柄"""
        for f in self.files:
            f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

