# Copyright (c) 2026 Jintao Li. 
# Zhejiang University (ZJU).
# 
# Licensed under the MIT License.

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("filark")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

__all__ = ["__version__"]
