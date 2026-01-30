from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

import numpy as np

from .tapeio import H5Tape, NpyTape, BinTape
from .protocols import Tape


@dataclass
class BinFolderOpts:
    nt: int
    nc: int
    dtype: np.dtype
    dims: str              # "nt_nc" / "nc_nt"
    fs: float = 1.0
    dx: float = 1.0
    dx_unit: str = "m"
    order: str = "C"


class TapeFileSet(Tape):
    def __init__(
        self,
        file_paths: str | Sequence[str | Path],
        *,
        # npy NEED dims
        npy_dims: Optional[str] = None,
        npy_fs: float = 1.0,
        npy_dx: float = 1.0,
        npy_dx_unit: str = "m",
        # bin NEED completed params
        bin_opts: Optional[BinFolderOpts] = None,
        h5_kwargs: Optional[dict] = None,
        suffixes: Sequence[str] = (".h5", ".hdf5", ".npy", ".dat", ".bin"),
    ):
        self._h5_kwargs = h5_kwargs or {}
        self._files: List[Tape] = []

        paths = self._normalize_paths(file_paths, suffixes=suffixes)
        if not paths:
            raise FileNotFoundError(f"No files found for {file_paths}")

        for p in paths:
            suf = p.suffix.lower()
            if suf in (".h5", ".hdf5"):
                self._files.append(H5Tape(str(p), **self._h5_kwargs))
            elif suf == ".npy":
                if npy_dims not in ("nt_nc", "nc_nt"):
                    raise ValueError("npy_dims must be provided for NPY folders (nt_nc / nc_nt).")
                self._files.append(NpyTape(str(p), dims=npy_dims, fs=npy_fs, dx=npy_dx, dx_unit=npy_dx_unit))
            elif suf in (".dat", ".bin"):
                if bin_opts is None:
                    raise ValueError("bin_opts must be provided for BIN/DAT folders.")
                self._files.append(BinTape(
                    path=str(p),
                    nt=bin_opts.nt,
                    nc=bin_opts.nc,
                    dtype=bin_opts.dtype,
                    dims=bin_opts.dims,
                    fs=bin_opts.fs,
                    dx=bin_opts.dx,
                    dx_unit=bin_opts.dx_unit,
                    order=bin_opts.order,
                ))
            else:
                raise ValueError(f"Unsupported suffix: {p}")

        self._validate_and_init()

    # --------------------
    # path helpers
    # --------------------
    def _normalize_paths(self, file_paths: str | Sequence[str | Path], *, suffixes: Sequence[str]) -> List[Path]:
        if isinstance(file_paths, (str, Path)):
            root = Path(file_paths)
            if root.is_dir():
                out: List[Path] = []
                for suf in suffixes:
                    out.extend(sorted(root.glob(f"*{suf}")))
                out = [p for p in out if not p.name.startswith(".")]
                return out
            else:
                return [root]
        else:
            return [Path(p) for p in file_paths]

    # --------------------
    # init + validation
    # --------------------
    def _validate_and_init(self):
        if not self._files:
            raise ValueError("Empty fileset.")

        dims0 = getattr(self._files[0], "dims", None)
        if dims0 not in ("nt_nc", "nc_nt"):
            raise ValueError(f"Invalid dims for first tape: {dims0}")

        def semantic_nc(t: Tape) -> int:
            return t.shape[1] if t.dims == "nt_nc" else t.shape[0]

        def semantic_nt(t: Tape) -> int:
            return t.shape[0] if t.dims == "nt_nc" else t.shape[1]

        nc0 = semantic_nc(self._files[0])
        dt0 = self._files[0].dtype
        self._dims = dims0
        self._dtype = dt0
        self._ndim = 2

        nts = []
        for i, t in enumerate(self._files):
            if t.ndim != 2:
                raise ValueError(f"Only 2D tapes are supported, file index {i} has ndim={t.ndim}")
            if t.dims != self._dims:
                raise ValueError(f"dims mismatch at index {i}: {t.dims} vs {self._dims}")
            if semantic_nc(t) != nc0:
                raise ValueError(f"nc mismatch at index {i}: {semantic_nc(t)} vs {nc0}")
            if t.dtype != dt0:
                raise ValueError(f"dtype mismatch at index {i}: {t.dtype} vs {dt0}")
            nts.append(semantic_nt(t))

        self._nts = nts
        self._cum_nts = np.cumsum([0] + self._nts)  # length = nfiles+1
        self._nt_total = int(self._cum_nts[-1])
        self._nc = int(nc0)

    # --------------------
    # Tape interface
    # --------------------
    @property
    def shape(self):
        if self._dims == "nt_nc":
            return (self._nt_total, self._nc)
        else:
            return (self._nc, self._nt_total)

    @property
    def ndim(self):
        return self._ndim

    @property
    def dtype(self):
        return self._dtype

    @property
    def fs(self):
        return self._files[0].fs if self._files else None

    @property
    def dx(self):
        return self._files[0].dx if self._files else None

    @property
    def dx_unit(self):
        return self._files[0].dx_unit if self._files else None

    @property
    def dims(self):
        return self._dims

    def __len__(self):
        return self._nt_total

    # --------------------
    # indexing
    # --------------------
    def __getitem__(self, key: Any) -> np.ndarray:
        if isinstance(key, tuple):
            k0 = key[0]
            k1 = key[1] if len(key) > 1 else slice(None)
        else:
            k0 = key
            k1 = slice(None)

        # time axis depends on dims
        time_axis = 0 if self._dims == "nt_nc" else 1

        # normalize selectors for (axis0, axis1)
        if time_axis == 0:
            ax0_key, ax1_key = k0, k1
        else:
            ax0_key, ax1_key = k1, k0

        # helper: take from a single tape with (axis0, axis1) keys
        def take_one(t: Tape, a0, a1):
            return t[a0, a1]

        # helper: concat along time axis (in tape's native axis order)
        def cat(chunks):
            return np.concatenate(chunks, axis=time_axis)

        # --------
        # Case 1: time index is int
        # --------
        tkey = ax1_key if time_axis == 0 else ax0_key
        okey = ax0_key if time_axis == 0 else ax1_key

        if isinstance(tkey, (int, np.integer)):
            i = int(tkey)
            if i < 0:
                i += self._nt_total
            if i < 0 or i >= self._nt_total:
                raise IndexError(i)

            fidx = int(np.searchsorted(self._cum_nts, i, side="right") - 1)
            rel = i - int(self._cum_nts[fidx])
            t = self._files[fidx]

            if time_axis == 0:
                # t[time, other]
                return take_one(t, rel, okey)
            else:
                # t[other, time]
                return take_one(t, okey, rel)

        # --------
        # Case 2: time index is slice
        # --------
        if isinstance(tkey, slice):
            start, stop, step = tkey.indices(self._nt_total)

            if step != 1:
                # correct but slower: gather then stack/concat
                # keep numpy semantics: result axis position follows original indexing
                if time_axis == 0:
                    return np.array([self[(i, k1)] if isinstance(key, tuple) else self[i] for i in range(start, stop, step)],
                                    dtype=self.dtype)
                else:
                    # time axis is 1: build by concatenating along axis=1
                    parts = []
                    for i in range(start, stop, step):
                        parts.append(self[(k0, i)] if isinstance(key, tuple) else self[:, i])
                    return np.stack(parts, axis=time_axis)  # shape handling best-effort

            if start >= stop:
                # empty slice: construct empty with correct rank
                if time_axis == 0:
                    # (0, nc') where nc' depends on okey
                    # simplest: take a small sample to infer shape if possible
                    return np.empty((0, 0), dtype=self.dtype) if okey == slice(None) else np.empty((0,), dtype=self.dtype)
                else:
                    return np.empty((0, 0), dtype=self.dtype) if okey == slice(None) else np.empty((0,), dtype=self.dtype)

            p_start = int(np.searchsorted(self._cum_nts, start, side="right") - 1)
            p_stop = int(np.searchsorted(self._cum_nts, stop - 1, side="right") - 1)

            chunks = []
            for fidx in range(p_start, p_stop + 1):
                f0 = int(self._cum_nts[fidx])
                f1 = int(self._cum_nts[fidx + 1])

                s0 = max(start, f0)
                s1 = min(stop, f1)

                rel0 = s0 - f0
                rel1 = s1 - f0

                t = self._files[fidx]
                if time_axis == 0:
                    chunks.append(t[rel0:rel1, okey])   # ✅ no transpose
                else:
                    chunks.append(t[okey, rel0:rel1])   # ✅ no transpose

            if not chunks:
                # return truly empty, but keep dtype
                return np.empty((0,), dtype=self.dtype)

            return cat(chunks)

        raise TypeError(f"Unsupported time-axis index type: {type(tkey)}")


    # --------------------
    # lifecycle
    # --------------------
    @property
    def files(self) -> List[Tape]:
        return self._files

    def close(self):
        for t in self._files:
            try:
                t.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

