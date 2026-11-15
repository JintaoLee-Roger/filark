# Extensions

FiLark is designed to be extended without turning the core GUI into a collection
of experiment-specific tools.

## Good Extension Candidates

- custom storage loaders;
- project-specific monitor plugins;
- model wrappers for segmentation or event detection;
- ROI analysis widgets;
- display preprocessing chains;
- processor plugins for repeatable denoising or filtering workflows.

## Where To Put New Work

| Need | Best place |
|---|---|
| Open a proprietary file layout | custom loader |
| Change every block shown in the viewer | `stream_preprocess` / `stream_postprocess` |
| Run an algorithm and draw results | `MonitorPlugin` |
| Process an ROI and preview output | `ProcessorPlugin` |
| Open a custom plot/dashboard | `AnalysisPlugin` |
| Run offline over a full source | `filark.pipeline` |

## Adjacent Scripts

The `scripts` directory is a good home for runnable examples that combine
FiLark with project assets:

- `realtime_monitor_demo.py`: fake realtime monitor with bbox, polyline, and
  mask outputs;
- `source_location_demo.py`: first-arrival source localization as a figure
  plugin and as a widget plugin;
- `caltech_zarr_annotation.py`: custom loader and annotation setup for a Zarr
  event dataset;
- `segy_script.py`: custom loader for SEG-Y files (requires `cigsegy`).

Scripts are allowed to be model-specific. Core GUI modules should not be.

## Documentation Rule

When adding an extension, document:

1. how to register it through `GuiConfig`;
2. which coordinate system it emits or consumes;
3. whether it runs on the GUI thread, worker thread, or external process;
4. whether results are temporary overlays or persistent annotations;
5. how outputs can be saved and reloaded.
