# Concepts

This page explains the vocabulary used throughout FiLark. It is intentionally
short and architectural; implementation details live in the task guides.

## Streaming-First

FiLark treats long DAS recordings as streams of time-channel windows. A user may
be browsing an archived HDF5 folder, a memory-mapped NPY file, or a simulated
live source, but the rest of the system sees the same basic pattern:

1. read a bounded window from a source,
2. optionally process that window,
3. render or analyze the result,
4. keep coordinates tied to the original source.

This is different from a batch-first workflow where users first extract a short
segment, load it as an array, and then run every tool on that temporary block.

## Package Boundaries

FiLark is split so that the GUI does not own storage or numerical algorithms.

| Package | Role |
|---|---|
| `filark.io` | Tape protocol, H5/NPY/binary readers, folder stitching, stream adapters. |
| `filark.viz` | VisPy canvas, camera, scheduler, GPU image visual, annotation overlay visuals. |
| `filark.gui` | Qt panels, controllers, ROI windows, plugin UI, monitor workers. |
| `filark.dsp` | Array-first DSP functions, chunking utilities, transform objects, GPU variants. |
| `filark.pipeline` | Chunked processing execution over large Tape sources. |
| `filark.processing` | GUI-facing processor plugin wrappers. |
| `filark.algorithms` | Detection algorithms and examples. |
| `filark.utils` | Annotation JSON, coordinate, memory, performance, and time helpers. |

The most important dependency direction is:

```text
GUI -> viz/io/dsp/processing/plugins
viz -> no GUI dependency
io/dsp/pipeline -> no GUI dependency
```

## Tape Sources

A Tape is a 2-D DAS source with metadata. FiLark expects:

- `shape`
- `dtype`
- `dims`: `"nt_nc"` or `"nc_nt"`
- `fs`: sample rate in Hz
- `dx`: channel spacing
- `dx_unit`
- optional `gauge_length`
- array-style slicing or read helpers

The GUI can work with built-in Tape classes such as `H5Tape`, `NpyTape`,
`BinTape`, `ArrayTape`, and `TapeFileSet`, or with custom sources that follow
the protocol.

## Time-Channel Coordinates

The canonical semantic layout is time by channel:

```text
t: sample index along time
c: channel index along fiber
```

Even when a file is stored as `nc_nt`, annotations and monitor outputs are
expressed in the same semantic coordinate system. This keeps annotation JSON,
monitor events, ROI extraction, and display overlays consistent across file
layouts.

For gather-like sources, the GUI can declare display-axis roles such as
`x_role="trace"` and `y_role="time"`. The viewer still renders 2-D index
coordinates, while GUI adapters use the roles for labels, annotation metadata,
and ROI handoff to time-first algorithms. Realtime monitor plugins are kept on
the default time-channel coordinate system.

## Viewer And Scheduler

The main canvas keeps a fixed display buffer instead of loading the full data
array. When the viewport moves, the scheduler decides whether the move can be
served by a small incremental push or whether a full reset is needed.

In async mode, data fetch runs away from the GUI thread. The UI thread owns the
OpenGL context and only uploads/render data when a prepared block is ready.

## Annotation And Detection Are Related But Not Identical

Both annotation and monitor results can be rendered as bbox, polyline, or mask
overlays. The difference is ownership:

- annotation items live in the annotation store and are editable;
- monitor results are detection overlays unless they are saved/imported as
  annotations.

This separation lets detector output be reviewed manually without forcing every
temporary overlay to become a persistent label.
