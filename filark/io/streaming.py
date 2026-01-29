from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Callable, Literal, Union
import numpy as np
from .protocols import Tape

YMode = Literal["local", "fit_all"]
Layout = Literal["auto", "nc_nt", "nt_nc"]  # (nc, nt) or (nt, nc)

# ---- resize helpers: keep yours as-is ----
def _resize_1d_linear(arr: np.ndarray, out_len: int, axis: int) -> np.ndarray:
    if out_len <= 0:
        raise ValueError("out_len must be > 0")
    in_len = arr.shape[axis]
    if in_len == out_len:
        return arr

    x = np.moveaxis(arr, axis, -1)
    in_len = x.shape[-1]
    if in_len <= 1:
        y = np.repeat(x, out_len, axis=-1)
        return np.moveaxis(y, -1, axis)

    src_pos = np.linspace(0.0, in_len - 1, out_len, dtype=np.float32)
    left = np.floor(src_pos).astype(np.int64)
    right = np.clip(left + 1, 0, in_len - 1)
    t = (src_pos - left).astype(np.float32)

    xl = np.take(x, left, axis=-1)
    xr = np.take(x, right, axis=-1)
    y = xl * (1.0 - t) + xr * t
    return np.moveaxis(y, -1, axis)

def _resize_2d_separable_linear(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    tmp = _resize_1d_linear(img, out_w, axis=1)
    out = _resize_1d_linear(tmp, out_h, axis=0)
    return out

@dataclass
class StreamingSource:
    """
    Streaming data source with explicit view↔raw coordinate mapping.

    Coordinate system:
      - X: time / sample (width)
      - Y: channel (height)

    Scaling model (scale >= 1.0):
      view_x = raw_x / scale_x
      view_y = raw_y / scale_y

    The source is responsible for:
      - Mapping view-space requests to raw data slices
      - Applying optional preprocessing / postprocessing
      - Handling out-of-bounds behavior
    """

    data: Union[np.ndarray, np.memmap, Tape]
    """Underlying 2D data array (raw coordinates)."""

    buffer_w: Optional[int] = None
    """Streaming buffer width in view coordinates."""

    buffer_h: Optional[int] = None
    """Streaming buffer height in view coordinates."""

    scale_x: float = 1.0
    """X-axis compression factor (>= 1.0)."""

    scale_y: float = 1.0
    """Y-axis compression factor (>= 1.0)."""

    y_mode: YMode = "local"
    """Y scaling mode: 'local' or 'fit_all'."""

    layout: Layout = "auto"
    """Raw data layout: 'auto', 'nc_nt', or 'nt_nc'."""

    dims: Layout = "auto"
    """Raw data layout: 'auto', 'nc_nt', or 'nt_nc'."""

    preprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None
    """Optional preprocessing applied on raw slices."""

    postprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None
    """Optional postprocessing applied after resizing."""

    return_none_on_oob: bool = True
    """If True, return None when request is fully out-of-bounds."""


    def __post_init__(self):
        if self.data.ndim != 2:
            raise ValueError("StreamingSource expects 2D data")
        if self.scale_x < 1.0 or self.scale_y < 1.0:
            raise ValueError("scale_x/scale_y are compression factors and must be >= 1.0")

        if self.dims != 'auto':
            self.layout = self.dims
        self._layout = self._infer_layout(self.data.shape, self.layout)
        self._base_h, self._base_w = self._logical_shape_from_data_shape(self.data.shape, self._layout)
        self.layout = self._layout 
        self.dims = self._layout 


    # ---- layout helpers ----
    @staticmethod
    def _infer_layout(shape: Tuple[int, int], layout: Layout) -> Layout:
        if layout != "auto":
            return layout
        a, b = int(shape[0]), int(shape[1])
        # heuristic: larger dim tends to be time(nt)
        return "nc_nt" if b >= a else "nt_nc"

    @staticmethod
    def _logical_shape_from_data_shape(shape: Tuple[int, int], layout: Layout) -> Tuple[int, int]:
        if layout == "nc_nt":
            nc, nt = int(shape[0]), int(shape[1])
            return nc, nt
        if layout == "nt_nc":
            nt, nc = int(shape[0]), int(shape[1])
            return nc, nt
        raise ValueError(f"unknown layout: {layout}")

    def _slice_raw(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """raw 坐标切片，但返回始终是 (y, x) 排列"""
        if self._layout == "nc_nt":
            return self.data[y0:y1, x0:x1]
        else:
            return self.data[x0:x1, y0:y1].T  # (nt,nc)->(nc,nt)

    # ---- metadata ----
    @property
    def shape(self) -> Tuple[int, int]:
        return (self._base_h, self._base_w)

    @property
    def dtype(self):
        return self.data.dtype

    def set_buffer_size(self, buffer_h: int, buffer_w: int):
        if buffer_w <= 0 or buffer_h <= 0:
            raise ValueError("buffer_w/buffer_h must be positive")
        self.buffer_w = int(buffer_w)
        self.buffer_h = int(buffer_h)

    def set_scale(self, scale_x: Optional[float] = None, scale_y: Optional[float] = None, y_mode: Optional[YMode] = None):
        if scale_x is not None:
            if scale_x < 1.0:
                raise ValueError("scale_x is compression factor, must be >= 1.0")
            self.scale_x = float(scale_x)
        if scale_y is not None:
            if scale_y < 1.0:
                raise ValueError("scale_y is compression factor, must be >= 1.0")
            self.scale_y = float(scale_y)
        if y_mode is not None:
            self.y_mode = y_mode

    # ---- helpers ----
    @staticmethod
    def _ceil_int(x: float) -> int:
        return int(np.ceil(x))

    def _clip0(self, v0: int, src_len: int, base_len: int) -> int:
        if src_len >= base_len:
            return 0
        return max(0, min(int(v0), base_len - src_len))

    @staticmethod
    def _as_int_scale(s: float, tol: float = 1e-6) -> Optional[int]:
        """If s is (almost) an integer >= 1, return that integer; else None."""
        r = int(round(s))
        if r >= 1 and abs(s - r) <= tol:
            return r
        return None

    def read(
        self,
        y0: int,
        x0: int,
        buffer_h: Optional[int] = None,
        buffer_w: Optional[int] = None,
        *,
        scale_y: Optional[float] = None,
        scale_x: Optional[float] = None,
        y_mode: Optional[YMode] = None,
        pad_value: float = 0.0,
        out_dtype: Optional[np.dtype] = None,
        return_none_on_oob: Optional[bool] = None,
    ) -> Optional[np.ndarray]:
        if (buffer_h is None and self.buffer_h is None) or (buffer_w is None and self.buffer_w is None):
            raise ValueError("buffer size not init")

        bw = int(buffer_w if buffer_w is not None else self.buffer_w)
        bh = int(buffer_h if buffer_h is not None else self.buffer_h)
        sx = float(scale_x if scale_x is not None else self.scale_x)
        sy = float(scale_y if scale_y is not None else self.scale_y)
        ym = (y_mode if y_mode is not None else self.y_mode)
        none_on_oob = self.return_none_on_oob if return_none_on_oob is None else bool(return_none_on_oob)

        if bw <= 0 or bh <= 0:
            raise ValueError("buffer size must be positive")
        if sx < 1.0 or sy < 1.0:
            raise ValueError("scale_x/scale_y are compression factors and must be >= 1.0")

        sx_i = self._as_int_scale(sx)
        sy_i = self._as_int_scale(sy)

        # view -> raw（你的定义：raw = view * scale）
        raw_x0 = int(np.floor(x0 * sx))
        raw_y0 = int(np.floor(y0 * sy))

        # 决定 src 尺寸：若是整数 scale，则用精确乘法（保证 decimate 后刚好是 bw/bh）
        if sx_i is not None:
            src_w = max(1, bw * sx_i)
        else:
            src_w = max(1, self._ceil_int(bw * sx))

        if ym == "fit_all":
            src_h = self._base_h
            raw_y0_eff = 0
        else:
            if sy_i is not None:
                src_h = max(1, bh * sy_i)
            else:
                src_h = max(1, self._ceil_int(bh * sy))
            raw_y0_eff = self._clip0(raw_y0, src_h, self._base_h)

        raw_x0_eff = self._clip0(raw_x0, src_w, self._base_w)

        if none_on_oob:
            if raw_x0 >= self._base_w or raw_y0 >= self._base_h:
                return None

        y1 = min(self._base_h, raw_y0_eff + src_h)
        x1 = min(self._base_w, raw_x0_eff + src_w)

        raw = self._slice_raw(raw_y0_eff, y1, raw_x0_eff, x1)

        if raw.shape[0] != src_h or raw.shape[1] != src_w:
            if none_on_oob and (raw.shape[0] == 0 or raw.shape[1] == 0):
                return None
            pad_h = src_h - raw.shape[0]
            pad_w = src_w - raw.shape[1]
            raw = np.pad(
                raw,
                pad_width=((0, max(0, pad_h)), (0, max(0, pad_w))),
                mode="constant",
                constant_values=pad_value,
            )

        if self.preprocess is not None:
            raw = self.preprocess(raw)

        # -------------------------
        # 整数 scale => drop/decimate（local 模式）
        # fit_all 模式始终走 resize（你要求 full channel 才出现 resize）
        # -------------------------
        if ym != "fit_all" and (sx_i is not None or sy_i is not None):
            out = raw

            # X 方向：整数 scale 直接抽取，否则插值到 bw
            if sx_i is not None:
                out = out[:, ::sx_i]
                # 理论上应当正好是 bw；但为了极端情况下安全：
                if out.shape[1] != bw:
                    out = out[:, :bw] if out.shape[1] > bw else np.pad(
                        out, ((0, 0), (0, bw - out.shape[1])), mode="constant", constant_values=pad_value
                    )
            else:
                out = _resize_1d_linear(out.astype(np.float32, copy=False), out_len=bw, axis=1)

            # Y 方向：整数 scale 直接抽取，否则插值到 bh
            if sy_i is not None:
                out = out[::sy_i, :]
                if out.shape[0] != bh:
                    out = out[:bh, :] if out.shape[0] > bh else np.pad(
                        out, ((0, bh - out.shape[0]), (0, 0)), mode="constant", constant_values=pad_value
                    )
            else:
                out = _resize_1d_linear(out.astype(np.float32, copy=False), out_len=bh, axis=0)

        else:
            # 非整数 或 fit_all：保持原逻辑（线性 resize）
            out = _resize_2d_separable_linear(raw.astype(np.float32, copy=False), out_h=bh, out_w=bw)

        if self.postprocess is not None:
            out = self.postprocess(out)

        if out_dtype is not None:
            out = out.astype(out_dtype, copy=False)

        return out

    def index_map(
        self,
        y0: int,
        x0: int,
        *,
        buffer_w: Optional[int] = None,
        buffer_h: Optional[int] = None,
        scale_x: Optional[float] = None,
        scale_y: Optional[float] = None,
        y_mode: Optional[YMode] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        bw = int(buffer_w if buffer_w is not None else self.buffer_w)
        bh = int(buffer_h if buffer_h is not None else self.buffer_h)
        sx = float(scale_x if scale_x is not None else self.scale_x)
        sy = float(scale_y if scale_y is not None else self.scale_y)
        ym = (y_mode if y_mode is not None else self.y_mode)

        sx_i = self._as_int_scale(sx)
        sy_i = self._as_int_scale(sy)

        if sx_i is not None:
            src_w = max(1, bw * sx_i)
        else:
            src_w = max(1, self._ceil_int(bw * sx))

        raw_x0 = float(np.floor(x0 * sx))
        raw_x0_eff = float(self._clip0(int(raw_x0), src_w, self._base_w))

        if ym == "fit_all":
            src_h = self._base_h
            raw_y0_eff = 0.0
        else:
            if sy_i is not None:
                src_h = max(1, bh * sy_i)
            else:
                src_h = max(1, self._ceil_int(bh * sy))
            raw_y0 = float(np.floor(y0 * sy))
            raw_y0_eff = float(self._clip0(int(raw_y0), src_h, self._base_h))

        # 整数 scale + local：index_map 要对应 “drop” 的采样位置
        if ym != "fit_all" and sx_i is not None:
            xs = raw_x0_eff + (np.arange(bw, dtype=np.float32) * sx_i)
        else:
            xs = raw_x0_eff + np.linspace(0.0, src_w - 1, bw, dtype=np.float32)

        if ym != "fit_all" and sy_i is not None:
            ys = raw_y0_eff + (np.arange(bh, dtype=np.float32) * sy_i)
        else:
            ys = raw_y0_eff + np.linspace(0.0, src_h - 1, bh, dtype=np.float32)

        return ys, xs
