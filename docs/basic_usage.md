# Basic Usage

This page is a task-oriented walkthrough for a typical FiLark session. The demo
videos provide longer examples, while the sections below keep individual
workflows easy to find.

## Demo Videos

| Video | Workflow | Links |
| --- | --- | --- |
| S1 | Interactive exploration, ROI analysis, and full-data processing | [YouTube](https://www.youtube.com/watch?v=27FLH0Z71fk) / [Bilibili backup](https://www.bilibili.com/video/BV1HiLF6kEcg/?share_source=copy_web&vd_source=cc7827632680dba5ffd3be7e2003fa47) |
| S2 | Large-directory browsing, real-time display filtering, auto-scroll, and annotation | [YouTube](https://www.youtube.com/watch?v=hUlHTDGpIC8) / [Bilibili backup](https://www.bilibili.com/video/BV1pvLF6VELW/?share_source=copy_web&vd_source=cc7827632680dba5ffd3be7e2003fa47) |
| S3 | Simulated real-time monitoring, overlay results, and manual recheck | [YouTube](https://www.youtube.com/watch?v=HptglyXhii0) / [Bilibili backup](https://www.bilibili.com/video/BV1rvLF6VEmJ/?share_source=copy_web&vd_source=cc7827632680dba5ffd3be7e2003fa47) |
| S4 | Source localization with figure and widget analysis plugins | [YouTube](https://www.youtube.com/watch?v=9oRkvGtCsSA) / [Bilibili backup](https://www.bilibili.com/video/BV1Desp6yE49/) |

## 1. Load A Source

Open a single H5/NPY/binary source or a folder-backed stream with **File ▸ Open
File…** or **Open Folder…**. **File ▸ Source Properties…** shows the metadata
that matters for DAS interpretation: layout, sample rate, channel spacing, dtype,
dimensions, start time, and optional gauge length.

The key point for the viewer is that the source is treated as a continuous Tape,
not as a pre-extracted snippet.

## 2. Navigate The Stream

Use mouse and keyboard navigation to move through time and channels:

- pan with the arrow keys, by dragging (time), or with `Shift`+scroll (channels);
- zoom the time axis by scrolling or pinching;
- fit the visible channel range with `F`;
- adjust time/channel scale with `X`, `Y`, `Shift+X`, and `Shift+Y`;
- jump directly to a time or sample index with **View ▸ Go to…**.

Videos S1 and S2 show the fixed-size GPU buffer as an interaction model:
the viewport changes, but the viewer does not need to upload the entire dataset.

## 3. Adjust The Display

The display can change without modifying the source data. The Visuals panel
holds the colormap, color limits, display scale, and the real-time display
filter.

```{note}
Display filters are viewer-side tools. They are useful for exploration, but they
do not rewrite the underlying Tape source.
```

## 4. Inspect An ROI

Select an ROI in the ROI panel and open diagnostic views such as PSD, STFT,
F-K, tau-p, RMS, SVD, or cross-correlation from **Analyze ▸ Inspect ROI**, or a
denoising preview from **Analyze ▸ Process ROI…**.

The ROI step is the bridge between interactive visual browsing and processing:
it lets a user test a transform on a small region before committing to the full
source.

## 5. Process And Compare

After an ROI operation looks reasonable, submit the same operation to the full
dataset. FiLark can open the original and processed results in linked tabs so
that view positions remain synchronized during comparison.

Video S1 shows an ROI-tested operation applied to the full source and then
compared in linked tabs.

## 6. Annotate Events

Use the Annotation panel to draw:

- bbox labels for compact events;
- polyline labels for curving or elongated patterns;
- mask labels for pixel-level regions.

Mask brush width is screen-pixel based while saved masks are stored in data
coordinates. Video S2 shows bbox and polyline annotation, export, reload, and
review.

## 7. Run A Monitor

Open **Analyze ▸ Detect Events…**, choose a monitor, its scope (current view,
full source, or realtime display) and an optional output file, and run it while
the viewer remains interactive. Results appear as bbox, polyline, and mask
overlays.

## 8. Recheck Saved Results

Saved monitor output should be reloadable through the annotation module. This is
important because detector output is rarely final: a user should be able to
review, adjust, delete, or add labels after inference.

Video S3 shows monitor output saved, reopened, and reviewed through the
annotation workflow.

## 9. Run An Analysis Plugin

Analysis plugins registered through `GuiConfig(analyses=[...])` appear at the
bottom of the **Analyze** menu and work on the selected ROI.
`scripts/source_location_demo.py` adds two of them that locate a point source,
such as a sparker shot, from its first arrivals: select an ROI around the event
and choose **Source Location (figure)** or **Source Location (widget)**. The
widget version also marks the located channel on the main canvas. Video S4
shows this workflow.
