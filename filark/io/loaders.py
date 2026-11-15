# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""Loader registry and default loader definitions for GUI/source discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from .fileset import TapeFileSet
from .protocols import Tape
from .tapeio import H5Tape


class Loader(Protocol):
    suffixes: Sequence[str]

    def load(self, path: Path) -> Tape:
        ...


@dataclass(frozen=True)
class H5Loader:
    suffixes: tuple[str, ...] = (".h5", ".hdf5")

    def load(self, path: Path) -> Tape:
        if path.is_dir():
            return TapeFileSet(path, suffixes=self.suffixes)
        return H5Tape(str(path))


def default_loaders() -> tuple[Loader, ...]:
    return (H5Loader(),)
