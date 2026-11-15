# Getting Started

This page takes you from installation to a first viewer session.

## Installation Profiles

Most users should install the desktop viewer profile:

```bash
pip install "filark[gui]"
```

Install the GUI viewer plus PyTorch-backed operators:

```bash
pip install "filark[all]"
```

Install only the core package for server-side I/O, DSP, pipelines, and headless
scripting:

```bash
pip install filark
```

Development checkout:

```bash
git clone https://github.com/JintaoLee-Roger/filark.git
cd filark
pip install -e ".[all]"
```

The GUI requires `PySide6`. GPU operators use PyTorch when available. `JAX` and
`CuPy` are left as manual installs because their wheels depend on platform and
accelerator setup.

```{note}
FiLark keeps GUI dependencies optional at the package level, but the documentation
recommends `filark[gui]` because the desktop viewer is the main workflow for most
users. Python packaging supports positive extras such as `filark[gui]`, but it
does not provide a clean cross-tooling way to install a package with default GUI
dependencies and then subtract them with something like `filark[no-gui]`.
```

## Launch The GUI

Command line:

```bash
filark
filark --theme light
filark --f path/to/data.h5
```

Python:

```python
from filark.gui.app import run_app

run_app()
run_app(theme="light")
run_app(source="path/to/data.h5")
```

Configuration object:

```python
from filark.gui.app import run_app
from filark.gui.config import GuiConfig

cfg = GuiConfig(
    theme="dark",
    async_io=True,
    default_buffer_h=2048,
)

run_app(cfg=cfg, source="path/to/data.h5")
```

## Open A Source

FiLark can start from paths, arrays, or Tape objects.

```python
from filark.gui.app import run_app
from filark.io.tapeio import H5Tape, NpyTape, BinTape

h5 = H5Tape("data/example.h5")
run_app(source=h5)

npy = NpyTape("data/example.npy", dims="nt_nc", fs=1550, dx=5, dx_unit="m")
run_app(source=npy)

bin_tape = BinTape(
    "data/example.dat",
    nt=93000,
    nc=6455,
    dtype="float32",
    dims="nt_nc",
    fs=1550,
    dx=5,
    dx_unit="m",
)
run_app(source=bin_tape)
```

For a folder of compatible files:

```python
from filark.io.fileset import TapeFileSet

source = TapeFileSet("data/DS", suffixes=(".h5", ".hdf5"))
run_app(source=source)
```

**File ▸ Open File…** and **Open Folder…** open sources interactively. For raw
formats whose metadata cannot be read from the file, a dialog asks for layout,
dimensions, sample rate, spacing, dtype, and optional gauge length. **File ▸
Source Properties…** shows and overrides the metadata of the loaded source.

## First GUI Session

1. Open a source with **File ▸ Open File…** / **Open Folder…** (or drop it on the window), or pass it to `run_app`.
2. Use the arrow keys to pan time and channels; scroll or pinch to zoom the time
   axis, and hold `Shift` while scrolling to pan channels.
3. Press `F` to fit the visible channel range.
4. Press `X` or `Y` to adjust display downsampling.
5. Open the Annotation panel, choose a mode, and draw with `Ctrl` held.
6. Open the ROI panel to select a region, then use the **Analyze** menu to inspect
   it, process it, or run event detectors (**Analyze ▸ Detect Events…**).
7. Use **View ▸ Navigator…** for a whole-file image of the data. The toolbar holds **Range**, **Amplitude / RMS / Peak**, **Resolution** (bin size; Auto ≈ 1920 × 1080, or Full = every sample) and **Filter**, which apply on **Compute** (Ctrl/Cmd+Return), and **Color** (colormap, limits), which applies immediately. Click the image to jump, drag a box to zoom in, double-click to return to the full image. A bin-averaged amplitude only keeps content below roughly 0.5 / bin length.

The left side panel holds what acts on the current picture (Visuals,
Annotation, ROI); the menu bar holds functions (File, View, Analyze, Help).

For keyboard details, see {doc}`keyboard_shortcuts`.

## Example Demos

The repository includes scripts that launch configured GUI sessions:

```bash
python scripts/realtime_monitor_demo.py --synthetic
python scripts/source_location_demo.py -f path/to/record.h5
```

The realtime demo registers a fake monitor plugin that emits bbox, polyline, and
mask events. The source-location demo registers two analysis plugins that locate
a point source from first arrivals (see {doc}`gui_plugins`).
