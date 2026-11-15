# Streaming Processing In The Viewer

Use viewer-side stream processing when the displayed data should be transformed
as it is browsed. This is different from ROI processing and different from a
full-source pipeline run.

## Configuration Fields

`GuiConfig` exposes:

- `stream_preprocess`
  Runs before the display block is formed.
- `stream_postprocess`
  Runs after the display block is formed and before render-time display filters.
- `on_source_loaded`
  Optional hook to inspect or replace a Tape immediately after loading.

Use these hooks for lightweight display-oriented transforms such as de-mean,
normalization, AGC, or band-limited browsing aids.

## Static Transform Chain

```python
import filark.dsp.transforms as T
from filark.gui.app import run_app
from filark.gui.config import GuiConfig

cfg = GuiConfig(
    stream_preprocess=[
        T.DeMeanTime(),
        T.ZScoreTime(robust=True),
        T.AGCTime(half_window=50),
    ],
)

run_app(cfg=cfg, source="data/example.h5")
```

## Metadata-Aware Factory

Pass a callable when the transform parameters depend on source metadata.

```python
import filark.dsp.transforms as T
from filark.gui.app import run_app
from filark.gui.config import GuiConfig

def build_chain(*, fs, dx):
    return [
        T.DeMeanTime(),
        T.AGCTime(half_window=max(8, int(fs * 0.1))),
    ]

cfg = GuiConfig(stream_preprocess=build_chain)
run_app(cfg=cfg, source="data/example.h5")
```

## Stateful Transforms

Some transforms preserve state across streaming chunks. This is useful for
causal filters and running statistics. FiLark resets or snapshots transform
state when the viewer needs a full reset, seeks backward, or rebuilds the
buffer.

Use the realtime transform classes from `filark.dsp.transforms` when causal
state matters:

```python
import filark.dsp.transforms as T

chain = [
    T.RealtimeIIRBandpassFilter(fs=1550, fmin=5, fmax=120),
    T.RealtimeAGC(half_window=80),
]
```

## When Not To Use This

Do not use viewer stream hooks for:

- expensive algorithms that should run in a monitor worker;
- ROI-only experiments;
- full-source export processing;
- project-specific plots or dashboards.

Use {doc}`monitoring`, {doc}`dsp_pipeline`, or {doc}`gui_plugins` for those
workflows.
