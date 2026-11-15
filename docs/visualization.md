# Visualization

The main viewer is built for long DAS recordings where loading the whole array
is impractical. It keeps a fixed display buffer and updates only the region that
navigation exposes.

## Main Components

- `filark.viz.StreamingCanvas`: the VisPy canvas used by the GUI.
- `filark.viz.scheduler`: decides when to push incremental chunks or reset the
  visible buffer.
- `filark.viz.visual.image.StreamingImage`: GPU texture/ring-buffer visual.
- `filark.viz.camera`: deterministic time/channel navigation.
- `filark.gui.display.DisplayModel`: Qt-facing bridge for overlays,
  annotation interaction, and canvas state.

The `viz` package does not depend on GUI panels. It is the reusable
visualization layer.

## Ring Buffer Rendering

The image texture has a fixed capacity. During time-axis navigation, newly
visible data are fetched and uploaded into the vacated texture region while the
shader maps logical data coordinates to the physical ring-buffer location.

This keeps memory tied to display resolution, not recording duration.

## Async IO

`GuiConfig(async_io=True)` is the default. In async mode:

- data fetch and display downsampling run away from the Qt/OpenGL thread;
- texture upload and draw calls remain on the GUI thread;
- big jumps, scale changes, and channel changes use full reset paths;
- stale async results are ignored when the view has already moved elsewhere.

If data cannot be fetched before the viewport reaches the next region, FiLark
waits for the prepared block instead of clearing unrelated old data.

## Scale Controls

The viewer separates visual scale from source coordinates:

- `X` / `Shift+X`: compress or expand time-axis display scale.
- `Y` / `Shift+Y`: compress or expand channel-axis display scale.
- `F`: fit the channel axis into the current Y buffer.
- `Shift+F`: return to local 1:1 channel scale.
- `A` / `Shift+A`: adjust display aspect.

These controls change how much data is represented by the display buffer, not
the source metadata.

## Axis Labels and Direction

Custom tapes can reuse the viewer for gather-like data where the displayed
axes are not DAS time/channel semantics. Configure the GUI labels and y-axis
screen direction with `GuiConfig`:

```python
from filark.gui import GuiConfig, run_app

cfg = GuiConfig(
    source=my_tape,
    x_role="trace",
    y_role="time",
    x_label="Trace",
    y_label="Time (ms)",
    invert_y="auto",
)
run_app(cfg=cfg)
```

`invert_y="auto"` keeps the underlying data coordinates unchanged, but draws
smaller y values at the top when the displayed y axis is time or depth. This is
useful for seismic displays where time or depth increases downward.

When `x_role="trace"` and `y_role="time"`, ROI views receive a canonical
time-by-trace array. Monitor plugins remain restricted to the default
`x_role="time", y_role="channel"` DAS view.

## Realtime Auto-Scroll

Press `R` to toggle realtime auto-scroll and `S` to stop. Speed and assumed
sample rate can be controlled from shortcuts. Realtime stats can be printed with
`P` or `Shift+P`.

Realtime auto-scroll is useful for:

- checking whether a source can be browsed at acquisition speed;
- measuring IO/render bottlenecks;
- pairing monitor plugins with a moving display window.

For monitor behavior during realtime auto-scroll, see {doc}`monitoring`.

## Overlay Types

FiLark renders three annotation/detection geometries:

- bbox: axis-aligned time-channel rectangle;
- polyline: ordered time-channel points;
- mask: small raster patch projected into a time-channel bounding box.

Mask texture resolution and projected data bounds are intentionally separate.
This lets a compact mask represent a larger region without allocating one mask
pixel for every data sample.
