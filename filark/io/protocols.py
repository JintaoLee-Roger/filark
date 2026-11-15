# Copyright (c) 2026 Jintao Li. 
# Zhejiang University (ZJU).
# 
# Licensed under the MIT License.

"""Protocol types that define the minimal tape/source interfaces."""

from __future__ import annotations
from typing import Protocol, runtime_checkable, Tuple, Any, Optional, Mapping, Sequence
import numpy as np


@runtime_checkable
class Tape(Protocol):
    """Minimal protocol for a 2-D DAS data source.

    Attributes
    ----------
    shape : (int, int)
        Raw shape of the underlying array.
    dtype : np.dtype
    fs : float
        Sampling rate in Hz.
    dx : float
        Channel spacing in ``dx_unit``.
    dx_unit : str
        Physical unit of ``dx`` (e.g. ``"m"``).
    gauge_length : float or None
        Gauge length in ``dx_unit``. ``None`` if unknown.
    dims : str
        Layout of the raw array: ``"nt_nc"`` (time × channel) or
        ``"nc_nt"`` (channel × time).
    ndim : int
        Always 2.
    start_time : float or None
        UTC start time as a Unix timestamp (seconds since 1970-01-01).
        ``None`` if unknown.
    """

    @property
    def shape(self) -> Tuple[int, int]: ...
    @property
    def dtype(self) -> np.dtype: ...
    @property
    def fs(self) -> float: ...
    @property
    def dx(self) -> float: ...
    @property
    def dx_unit(self) -> str: ...
    @property
    def gauge_length(self) -> Optional[float]: ...
    @property
    def dims(self) -> str: ...
    @property
    def ndim(self) -> int: ...
    @property
    def start_time(self) -> Optional[float]: ...
    def __getitem__(self, key: Any) -> np.ndarray: ...
    def close(self) -> None: ...


@runtime_checkable
class TransformableTape(Tape, Protocol):
    """Optional Tape capability for switching source-level transforms."""

    @property
    def transform_options(self) -> Sequence[Mapping[str, Any]]: ...
    @property
    def transform(self) -> str: ...
    @property
    def transform_revision(self) -> int: ...
    def set_transform(self, key: str) -> str: ...
    def cycle_transform(self, step: int = 1) -> str: ...
