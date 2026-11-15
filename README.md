# FiLark

FiLark is a streaming-first Python framework for Distributed Acoustic Sensing
(DAS) data access, visualization, processing, annotation, and external analysis
integration.

This repository contains the complete FiLark `0.0.5` release.

The project is built around one practical constraint: DAS recordings are often
too large to treat as ordinary in-memory arrays. FiLark therefore keeps data
access, visualization, annotation, processing, and algorithm integration aligned
around streaming windows instead of one-off extracted snippets.

## Demo Videos

Click a thumbnail to watch the demo on YouTube.

<table>
  <tr>
    <td width="50%">
      <a href="https://www.youtube.com/watch?v=27FLH0Z71fk&autoplay=1">
        <img src="https://img.youtube.com/vi/27FLH0Z71fk/hqdefault.jpg" alt="Interactive exploration, ROI analysis, and full-data processing" width="100%">
      </a>
      <br>
      <strong>Exploration and Processing</strong>
      <br>
      <a href="https://www.bilibili.com/video/BV1HiLF6kEcg/?share_source=copy_web&amp;vd_source=cc7827632680dba5ffd3be7e2003fa47">Bilibili backup</a>
    </td>
    <td width="50%">
      <a href="https://www.youtube.com/watch?v=hUlHTDGpIC8&autoplay=1">
        <img src="https://img.youtube.com/vi/hUlHTDGpIC8/hqdefault.jpg" alt="Large-directory browsing, real-time display filtering, auto-scroll, and annotation" width="100%">
      </a>
      <br>
      <strong>Large Data and Annotation</strong>
      <br>
      <a href="https://www.bilibili.com/video/BV1pvLF6VELW/?share_source=copy_web&amp;vd_source=cc7827632680dba5ffd3be7e2003fa47">Bilibili backup</a>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <a href="https://www.youtube.com/watch?v=HptglyXhii0&autoplay=1">
        <img src="https://img.youtube.com/vi/HptglyXhii0/hqdefault.jpg" alt="Simulated real-time monitoring, overlay results, and manual recheck" width="100%">
      </a>
      <br>
      <strong>Monitoring and Recheck</strong>
      <br>
      <a href="https://www.bilibili.com/video/BV1rvLF6VEmJ/?share_source=copy_web&amp;vd_source=cc7827632680dba5ffd3be7e2003fa47">Bilibili backup</a>
    </td>
    <td width="50%">
      <a href="https://www.youtube.com/watch?v=9oRkvGtCsSA&autoplay=1">
        <img src="https://img.youtube.com/vi/9oRkvGtCsSA/hqdefault.jpg" alt="Source localization with figure and widget analysis plugins" width="100%">
      </a>
      <br>
      <strong>Source Localization Plugins</strong>
      <br>
      <a href="https://www.bilibili.com/video/BV1Desp6yE49/">Bilibili backup</a>
    </td>
  </tr>
</table>

## What FiLark Provides

- A Qt + VisPy desktop viewer for large DAS arrays.
- A fixed-size OpenGL ring-buffer renderer for deterministic navigation through
  long recordings.
- Tape abstractions for H5, NPY, binary files, arrays, and stitched folders.
- Annotation tools for bbox, polyline, and mask labels in time-channel
  coordinates.
- ROI analysis windows for PSD, STFT, F-K, tau-p, RMS, SVD, cross-correlation,
  and denoising previews.
- Monitor plugins that connect external algorithms to the viewer and render
  bbox, polyline, or mask results on the main canvas.
- DSP operators and chunked pipelines that can be used without the GUI.

The documentation describes the complete `0.0.5` release, including the desktop
viewer, annotation workflow, monitor integration, and scripting interfaces.

## Installation

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

`JAX` and `CuPy` are not installed automatically because their wheels are
platform-specific.

## Launch The Viewer

From the command line:

```bash
filark
filark --theme light
filark --f path/to/data.h5
```

From Python:

```python
from filark.gui.app import run_app

run_app()
run_app(theme="light")
run_app(source="path/to/data.h5")
```

With a configuration object:

```python
from filark.gui.app import run_app
from filark.gui.config import GuiConfig

cfg = GuiConfig(
    theme="dark",
    default_buffer_h=2048,
    async_io=True,
)

run_app(cfg=cfg, source="path/to/data.h5")
```

If `PySide6` is missing, the GUI entry points raise a clear install message with
the exact `python -m pip install "filark[gui]"` command.

FiLark intentionally keeps desktop GUI dependencies in an optional extra even
though the GUI is the main user-facing workflow. Standard Python packaging
handles `filark[gui]` cleanly, but it does not provide a reliable inverse extra
such as `filark[no-gui]` when GUI dependencies are part of the default install.

## Open Common DAS Sources

```python
from filark.gui.app import run_app
from filark.io.fileset import TapeFileSet
from filark.io.tapeio import BinTape, H5Tape, NpyTape

# HDF5 source.
tape = H5Tape("data/example.h5")
run_app(source=tape)

# NPY file or ndarray.
tape = NpyTape("data/example.npy", dims="nt_nc", fs=1550, dx=5, dx_unit="m")
run_app(source=tape)

# Raw binary source.
tape = BinTape(
    "data/example.dat",
    nt=93000,
    nc=6455,
    dtype="float32",
    dims="nt_nc",
    fs=1550,
    dx=5,
    dx_unit="m",
)
run_app(source=tape)

# Folder stitched as one logical stream.
tape = TapeFileSet("data/DS", suffixes=(".h5", ".hdf5"))
run_app(source=tape)
```

The GUI (**File ▸ Open File… / Open Folder…**) can also open `.h5`, `.npy`, `.dat`, `.bin`, and folders.
When metadata is not recoverable from the file, the GUI asks for layout,
sample rate, channel spacing, dtype, dimensions, and optional gauge length.

## Extend The GUI

Most project-specific behavior should be registered through `GuiConfig` rather
than hard-coded into the application.

```python
from filark.gui.app import run_app
from filark.gui.config import GuiConfig

cfg = GuiConfig(
    loaders=[...],    # custom source loaders
    processors=[...], # ROI processing plugins
    monitors=[...],   # detection / monitoring plugins
    analyses=[...],   # custom analysis windows or matplotlib figures
)

run_app(cfg=cfg)
```

`scripts/source_location_demo.py` is a complete example: it registers the same
first-arrival localization algorithm once as a figure plugin and once as a
widget plugin.

```bash
python scripts/source_location_demo.py -f path/to/record.h5
```

For display-side preprocessing, pass transform objects or factories:

```python
import filark.dsp.transforms as T
from filark.gui.app import run_app
from filark.gui.config import GuiConfig

cfg = GuiConfig(
    stream_preprocess=[
        T.DeMeanTime(),
        T.AGCTime(half_window=50),
    ],
)

run_app(cfg=cfg, source="data/example.h5")
```

## Repository Map

- `filark/io`: Tape interfaces, file readers, folder stitching, streaming
  source adapters.
- `filark/viz`: VisPy canvas, ring-buffer image visual, scheduler, camera,
  annotation overlay visuals.
- `filark/gui`: Qt application, panels, controllers, annotation UI, monitor
  workers, ROI windows, plugin surfaces.
- `filark/dsp`: Array-first DSP functions, chunk helpers, object-style
  transforms, and PyTorch-backed GPU variants.
- `filark/pipeline`: Chunked processing execution for large sources.
- `filark/processing`: GUI-facing processor plugins and processing helpers.
- `filark/algorithms`: Built-in or example detection algorithms.
- `filark/utils`: Annotation export/import utilities, coordinates, memory, and
  timing helpers.
- `scripts`: runnable examples, including a realtime monitor demo and a
  source-location example implemented as analysis plugins.
- `benchmarks`: the scripts behind the performance measurements in the paper.
- `docs`: the Sphinx documentation site.

## Documentation

Build the docs locally:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

Main pages:

- `docs/getting_started.md`
- `docs/concepts.md`
- `docs/io_loader_guide.md`
- `docs/visualization.md`
- `docs/annotations.md`
- `docs/monitoring.md`
- `docs/dsp_pipeline.md`
- `docs/gui_plugins.md`

## License

FiLark is released under the MIT License.
