# GUI Plugin Guide

FiLark extension points are registered through `GuiConfig`. This keeps
project-specific logic outside the main window while still making it available
from the GUI.

## Extension Surfaces

| Surface | Config field | Base class / protocol | Purpose |
|---|---|---|---|
| Loader | `loaders` | object with `suffixes` and `load(path)` | Create Tape sources from files/folders. |
| Processor | `processors` | `filark.processing.processors.ProcessorPlugin` | ROI processing and full-data apply from ROI tools. |
| Monitor | `monitors` | `filark.gui.plugins.monitor.MonitorPlugin` | Detection/monitoring overlays. |
| Analysis | `analyses` | `filark.gui.plugins.analysis.AnalysisPlugin` | Custom figures or Qt widgets. |

Register all of them the same way:

```python
from filark.gui.app import run_app
from filark.gui.config import GuiConfig

cfg = GuiConfig(
    loaders=[...],
    processors=[...],
    monitors=[...],
    analyses=[...],
)

run_app(cfg=cfg)
```

## Parameter Schema

Processor, monitor, and analysis plugins use a `param_spec` list. The GUI builds
dialogs from it.

```python
param_spec = [
    {
        "key": "fmin",
        "label": "Low cut (Hz)",
        "type": "float",
        "default": 5.0,
        "min": 0.0,
        "max": 1000.0,
        "step": 1.0,
    },
    {
        "key": "use_gpu",
        "label": "Use GPU",
        "type": "bool",
        "default": False,
    },
]
```

Common types include `int`, `float`, `bool`, `choice`, `str`, `file`,
`int_range`, and `float_range`.

## Processor Plugins

Processor plugins run on ROI-sized arrays and return arrays with matching
semantics.

```python
import numpy as np
from filark.processing.processors import ProcessorPlugin

class ScaleProcessor(ProcessorPlugin):
    label = "Scale"
    method_key = "scale"
    param_spec = [
        {"key": "factor", "label": "Factor", "type": "float", "default": 1.0},
    ]

    def run(self, data, *, fs, dx, params, device="cpu"):
        return np.asarray(data) * float(params["factor"])
```

Optional methods:

- `validate_params(params, fs, dx)`
- `overlap_channels(params)`
- `overlap_time_samples(params, fs)`

## Monitor Plugins

Monitor plugins consume a Tape-like source and emit event dictionaries. They may
stream partial result batches through `result_cb`.

```python
from filark.gui.plugins.monitor import MonitorPlugin

class DemoMonitor(MonitorPlugin):
    label = "Demo Monitor"
    key = "demo_monitor"
    supports_gpu = False

    def run(self, source, *, params, progress_cb=None, cancel_cb=None, device="cpu", result_cb=None):
        events = [{"geometry_type": "bbox", "t0": 0, "t1": 100, "c0": 0, "c1": 20}]
        if result_cb is not None:
            result_cb(events)
        return events
```

See {doc}`monitoring` for event formats and realtime behavior.

## Analysis Plugins

Analysis plugins either return matplotlib figures or provide their own Qt widget.

```python
from filark.gui.plugins.analysis import AnalysisPlugin

class MyAnalysis(AnalysisPlugin):
    label = "My Analysis"
    key = "my_analysis"

    def run(self, source, *, params, roi=None):
        # Return a matplotlib Figure, list of Figures, or None.
        return None
```

FiLark builds the parameter dialog from `param_spec` and shows the returned
figures in a window with the matplotlib toolbar. `roi` is `None` when no ROI is
selected; otherwise it is a dict with the ROI block (`data`, time by channel),
`fs`, `dx`, and the ROI bounds (`roi.t0`, `roi.t1`, `roi.c0`, `roi.c1`).

For a custom widget, override `create_widget(source, parent=None)` and return a
`QWidget`. If the widget defines `set_filark_context(context)`, FiLark passes a
dict of callables, including `roi_getter()` (the current ROI as the same dict
that `run()` receives, or `None`), `current_view_bounds_getter()`,
`goto_sample(t)`, `set_channel_marker(channel)` and `clear_channel_marker()`.

`scripts/source_location_demo.py` implements the same algorithm in both forms:
it filters the selected data, picks first arrivals, and fits a direct-wave
hyperbola to locate a source along the cable.

## Loader Plugins

Loader plugins are intentionally lightweight. See {doc}`io_loader_guide` for a
full example.

```python
class Loader:
    suffixes = (".dat",)

    def load(self, path):
        return tape
```

## What Should Stay Out Of Core

Keep model-specific checkpoints, project-specific dashboards, and experiment
scripts as plugins or scripts. Add behavior to FiLark core only when it improves
the generic browsing, annotation, monitoring, or processing workflow.
