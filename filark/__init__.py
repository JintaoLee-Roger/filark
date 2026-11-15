# Copyright (c) 2026 Jintao Li. 
# Zhejiang University (ZJU).
# 
# Licensed under the MIT License.

from __future__ import annotations

"""FiLark package root.

The project is intentionally split into a few stable layers:

- ``io``: file/tape abstractions and streaming reads
- ``dsp``: pure array-oriented signal processing kernels
- ``pipeline``: chunked execution and file-writing orchestration
- ``processing``: user-facing processing APIs and processor schemas
- ``viz``: vispy rendering, camera, scheduler, and GPU buffer management
- ``gui``: Qt orchestration and plugin-driven desktop UI

Keeping these boundaries explicit makes it easier to evolve the viewer and the
research code independently.
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("filark")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

__all__ = ["__version__"]
