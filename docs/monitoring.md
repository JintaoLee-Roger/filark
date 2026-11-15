# Monitoring

Monitor plugins connect algorithms to the viewer. A monitor receives a Tape-like
source and returns event dictionaries that the GUI renders as overlays.

## Monitor Modes

The **Analyze ▸ Detect Events…** window can run monitors in different scopes:

- current view: run on the visible window;
- full source: run on the loaded source;
- realtime display: consume chunks as the display scrolls.

Realtime display monitoring is intended for algorithm-in-the-loop demos and live
inspection. It uses a persistent worker thread and bounded queues so slow
algorithms do not block the OpenGL thread.

## Event Geometry

Monitors may emit:

```python
{"geometry_type": "bbox", "t0": 10, "t1": 40, "c0": 4, "c1": 12, "label": "event"}
```

```python
{"geometry_type": "polyline", "points": [(10, 4), (20, 8), (40, 12)], "label": "moveout"}
```

```python
{
    "geometry_type": "mask",
    "t0": 10,
    "t1": 80,
    "c0": 20,
    "c1": 90,
    "mask": mask_array,
    "label": "segmentation",
}
```

Optional fields such as `score`, `color`, `alpha`, `threshold`, `algorithm`, and
`meta` are passed through to overlay and export paths when supported.

## Minimal Plugin

```python
from filark.gui.plugins.monitor import MonitorPlugin

class ThresholdMonitor(MonitorPlugin):
    label = "Threshold"
    key = "threshold"
    param_spec = [
        {"key": "threshold", "label": "Threshold", "type": "float", "default": 5.0},
    ]

    def run(self, source, *, params, progress_cb=None, cancel_cb=None, device="cpu", result_cb=None):
        # Return event dictionaries in source coordinates.
        return [
            {
                "geometry_type": "bbox",
                "t0": 1000,
                "t1": 1200,
                "c0": 40,
                "c1": 80,
                "label": "candidate",
            }
        ]
```

Register it:

```python
from filark.gui.app import run_app
from filark.gui.config import GuiConfig

cfg = GuiConfig(monitors=[ThresholdMonitor()])
run_app(cfg=cfg, source="data/example.h5")
```

## Progressive Results

For long scans, call `result_cb(batch_events)` whenever a batch is ready. The
GUI can render partial results before the full run finishes.

Always check `cancel_cb()` inside long loops:

```python
if cancel_cb is not None and cancel_cb():
    return events
```

## Saving Results

The Detect Events window can save monitor output as annotation JSON. The writer
appends event batches instead of rewriting the whole file on every update. Saved
results can be imported through the Annotation panel for manual review and
correction.

## Realtime Performance

Realtime monitoring has three separate costs:

- IO/display fetch;
- algorithm worker time;
- overlay update/render time.

If an algorithm is slower than realtime, the worker should skip stale display
chunks or process the latest relevant chunk rather than forcing the GUI to wait.
The viewer still owns rendering and user interaction on the main thread.

Use the demo script to inspect behavior:

```bash
python scripts/realtime_monitor_demo.py --synthetic --quiet-monitor
```

It registers a fake monitor whose run time scales with the amount of data, so
that the effect of a slow algorithm on the display can be observed.
