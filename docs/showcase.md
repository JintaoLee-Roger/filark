# Showcase Workflows

This page describes the workflows FiLark is meant to support. It is not a
benchmark report and does not duplicate the paper.

## Video Overview

- [Video S1](https://www.youtube.com/watch?v=27FLH0Z71fk) ([Bilibili backup](https://www.bilibili.com/video/BV1HiLF6kEcg/?share_source=copy_web&vd_source=cc7827632680dba5ffd3be7e2003fa47)): interactive exploration, ROI analysis, and full-data processing.
- [Video S2](https://www.youtube.com/watch?v=hUlHTDGpIC8) ([Bilibili backup](https://www.bilibili.com/video/BV1pvLF6VELW/?share_source=copy_web&vd_source=cc7827632680dba5ffd3be7e2003fa47)): large-directory browsing, display filtering, auto-scroll, and annotation.
- [Video S3](https://www.youtube.com/watch?v=HptglyXhii0) ([Bilibili backup](https://www.bilibili.com/video/BV1rvLF6VEmJ/?share_source=copy_web&vd_source=cc7827632680dba5ffd3be7e2003fa47)): simulated real-time monitoring, overlay results, and manual recheck.
- [Video S4](https://www.youtube.com/watch?v=9oRkvGtCsSA) ([Bilibili backup](https://www.bilibili.com/video/BV1Desp6yE49/)): source localization with figure and widget analysis plugins.

## Browse A Large Recording

1. Open a file or folder-backed source.
2. Use `F` and `Y` to fit many channels into view.
3. Use `X` to compress time when scanning long intervals.
4. Pan or run realtime auto-scroll.
5. Return to local scale when a candidate event appears.

The viewer keeps memory tied to the display buffer, so the same workflow applies
to short files and long stitched recordings.

## Inspect A Region Of Interest

1. Open the ROI panel in the left side bar.
2. Enable "Drag to select" and draw a region.
3. Open PSD, STFT, F-K, tau-p, RMS, SVD, or xcorr views from **Analyze ▸ Inspect
   ROI**, or the denoising view from **Analyze ▸ Process ROI…**.
4. Tune parameters locally before deciding whether to export or run a full
   source workflow.

This keeps the main canvas focused on context while heavier analysis lives in a
separate window.

## Create Labels While Browsing

1. Open the Annotation panel.
2. Choose bbox, polyline, or mask brush.
3. Draw labels in source coordinates.
4. Use label styles and visibility to manage dense sessions.
5. Export annotation JSON.

The exported file contains both labels and source metadata, so later scripts can
recover the original data region.

## Review Detector Output

1. Register a `MonitorPlugin` through `GuiConfig`.
2. Run it on the current view, full source, or realtime display.
3. Save monitor output as annotation JSON.
4. Import the JSON in the Annotation panel.
5. Correct bbox, polyline, or mask labels manually.

This lets algorithm results become reviewed labels without mixing temporary
detection overlays with persistent annotation state.

## Locate A Source From First Arrivals

1. Launch `scripts/source_location_demo.py` with a record that contains a shot.
2. Select an ROI around the event in the ROI panel.
3. Run **Analyze ▸ Source Location (figure)**: FiLark builds the parameter
   dialog and shows the returned figure with the picks and the fitted
   direct-wave hyperbola.
4. Or run **Analyze ▸ Source Location (widget)**, which keeps its own
   parameter panel, re-runs on demand, and marks the located channel on the
   main canvas.

The same algorithm is wrapped in both forms, which shows the two ways an
analysis plugin can integrate with the viewer.

## Use FiLark Without The GUI

1. Open a Tape from `filark.io`.
2. Build an array transform chain from `filark.dsp.transforms`.
3. Execute in memory for small arrays or use `filark.pipeline` for large
   sources.
4. Reuse annotation JSON with `filark.utils.annotations` to extract labeled
   regions.

The same source metadata and coordinate conventions are used by the GUI and
headless workflows.
