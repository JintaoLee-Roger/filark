# Copyright (c) 2026 Jintao Li. 
# Zhejiang University (ZJU).
# 
# Licensed under the MIT License.

"""I/O and streaming-source infrastructure.

Public API::

    from filark.io import ArrayTape, H5Tape, NpyTape, BinTape

``ArrayTape`` is the most flexible entry point — wrap any 2D array-like
(h5py Dataset, zarr, segyio, numpy, …) as a Tape with a single call.


This package is responsible for turning files, folders, and arrays into a
uniform Tape/StreamingSource model.  It owns:

- file-backed source adapters such as H5/NumPy/BIN tapes
- multi-file stitching and folder loaders
- streaming read/downsample logic used by the viewer
- source-side processing adapters that are still tied to streaming semantics

It should not own rendering policy or Qt widget behavior.
"""

from .tapeio import ArrayTape, H5Tape, NpyTape, BinTape
from .protocols import Tape

__all__ = ["ArrayTape", "H5Tape", "NpyTape", "BinTape", "Tape"]
