# Data Loading

FiLark separates data access from the GUI. The viewer expects a Tape-like source;
loaders and dialogs are only ways to create one.

## Built-In Sources

| Source | Class | Notes |
|---|---|---|
| HDF5 | `H5Tape` | Reads HDF5 datasets and metadata when keys are known. |
| NPY | `NpyTape` | File-backed or ndarray-backed data; requires `dims`. |
| Raw binary | `BinTape` | Requires `nt`, `nc`, `dtype`, `dims`, and metadata. |
| In-memory array | `ArrayTape` | Useful for scripts and generated data. |
| Folder | `TapeFileSet` | Stitches compatible files into one logical time stream. |

All of these live under `filark.io`.

## Layout

FiLark supports two storage layouts:

- `nt_nc`: array shape is `(time, channel)`
- `nc_nt`: array shape is `(channel, time)`

Viewer coordinates, annotations, and monitor outputs are always semantic
`(t, c)` coordinates even if the file is stored as `nc_nt`.

For non-default display conventions, declare roles in `GuiConfig` instead of
changing the viewer internals:

```python
cfg = GuiConfig(
    source=my_gather_tape,
    x_role="trace",
    y_role="time",
    x_label="Trace",
    y_label="Time (ms)",
    invert_y="auto",
)
```

ROI views receive time-first blocks after this adapter step. Monitor plugins
still require the default time-channel view.

## Launch With A Tape

```python
from filark.gui.app import run_app
from filark.io.tapeio import H5Tape

source = H5Tape(
    "data/example.h5",
    datakey="Acquisition/Raw[0]/RawData",
)

run_app(source=source)
```

## Load A Folder

```python
from filark.gui.app import run_app
from filark.io.fileset import TapeFileSet

source = TapeFileSet("data/DS", suffixes=(".h5", ".hdf5"))
run_app(source=source)
```

For NPY folders and binary folders, pass the metadata FiLark cannot infer from
the file itself:

```python
import numpy as np
from filark.io.fileset import BinFolderOpts, TapeFileSet

source = TapeFileSet(
    "data/bin_folder",
    suffixes=(".dat", ".bin"),
    bin_opts=BinFolderOpts(
        nt=93000,
        nc=6455,
        dtype=np.float32,
        dims="nt_nc",
        fs=1550,
        dx=5,
        dx_unit="m",
    ),
)
```

### Timestamp gaps between files

When every file provides a `start_time`, `TapeFileSet` checks that each file
starts where the previous one ended (`start_time + nt / fs`). Deviations within
`gap_tolerance` (default 10 ms, at least half a sample) are timestamp jitter and
are ignored. Larger deviations follow `gap_policy`:

| `gap_policy` | Gap (missing data) | Overlap (duplicated data) |
|---|---|---|
| `"fill"` (default) | replaced by a short separator of `gap_samples` fill samples (default 100, or the true gap length if shorter) | duplicated leading samples of the later file are dropped |
| `"raise"` | `ValueError` | `ValueError` |
| `"ignore"` | files are concatenated as-is (legacy behaviour) | concatenated as-is |

Files are ordered by `start_time` for `"fill"` and `"raise"`. Separators
allocate no memory; reads inside them return `fill_value`. Because a long
outage occupies only `gap_samples` samples, the sample index is no longer
`start_time + index / fs` across gaps: use `source.sample_to_time(index)` and
`source.time_to_sample(t)` (the GUI axis, cursor read-out and Go-to dialog do).

```python
source = TapeFileSet("data/campaign", suffixes=(".h5",))
for g in source.gaps:
    print(g.kind, g.start, g.stop, f"missing {g.seconds:.3f} s")
print(source.segments)   # contiguous [start, stop) sample ranges
```

### Open file handles

Opening a folder reads each file's metadata once and then closes it. During
reads `TapeFileSet` keeps at most `max_open_files` HDF5 files open (default 6,
least recently used are closed and reopened on demand). Every open HDF5
dataset holds its own chunk cache, so this keeps memory bounded however many
files a session browses through, and keeps the number of file descriptors
small for folders with thousands of files.

```python
source = TapeFileSet("data/campaign", max_open_files=6)
```

Every gap or overlap is a segment boundary. Stateful stream processing (for
example the causal IIR display filter) is reset at the start of each segment,
separator columns are rendered blank, and the viewer marks each separator
with a "NO DATA (duration)" label showing the true missing duration. Chunked pipelines process each segment separately, so no
chunk, halo or filter state spans a discontinuity.

## Custom Loaders

Use custom loaders when your filenames or proprietary metadata encode enough
information to build a Tape without asking the user.

```python
from pathlib import Path
import numpy as np

from filark.gui.app import run_app
from filark.gui.config import GuiConfig
from filark.io.tapeio import BinTape

class MyBinLoader:
    suffixes = (".dat", ".bin")

    def load(self, path):
        path = Path(path)
        if path.is_dir():
            raise IsADirectoryError
        nt, nc = 93000, 6455
        return BinTape(path, nt, nc, np.float32, "nt_nc", fs=1550, dx=5)

cfg = GuiConfig(loaders=[MyBinLoader()])
run_app(cfg=cfg)
```

Loader order is:

1. `GuiConfig.loaders`
2. `GuiConfig.builtins`
3. GUI fallback dialogs

If a loader fails on a folder, FiLark can apply it to individual files and then
stitch compatible children with `TapeFileSet`.

## Metadata Checklist

For robust browsing, annotation export, and ROI extraction, provide:

- layout: `dims`
- sample rate: `fs`
- channel spacing: `dx`
- spacing unit: `dx_unit`
- optional gauge length: `gauge_length`
- optional absolute start time: `start_time`

These fields are also written into annotation exports when available.
