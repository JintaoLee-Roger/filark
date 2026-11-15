# Guides

These pages are organized by task rather than by package name.

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Data Loading
Open H5, NPY, binary, array, and folder-backed sources.
+++
{doc}`io_loader_guide`
:::

:::{grid-item-card} Basic Usage
A task-oriented walkthrough from loading through monitoring recheck.
+++
{doc}`basic_usage`
:::

:::{grid-item-card} Visualization
Understand the streaming canvas, async IO, scaling, and realtime auto-scroll.
+++
{doc}`visualization`
:::

:::{grid-item-card} Streaming Processing
Attach lightweight preprocessing to the main viewer.
+++
{doc}`gui_stream_processing`
:::

:::{grid-item-card} Annotation
Draw bbox, polyline, and mask labels; export and import JSON.
+++
{doc}`annotations`
:::

:::{grid-item-card} Monitoring
Run detector plugins and save results as annotation JSON.
+++
{doc}`monitoring`
:::

:::{grid-item-card} DSP And Pipelines
Use array-first DSP, transform objects, GPU variants, and chunked execution.
+++
{doc}`dsp_pipeline`
:::

:::{grid-item-card} GUI Plugins
Register custom loaders, processors, monitors, and analyses.
+++
{doc}`gui_plugins`
:::

:::{grid-item-card} Keyboard Shortcuts
Navigation, scale control, realtime auto-scroll, annotation, and diagnostics.
+++
{doc}`keyboard_shortcuts`
:::
::::

```{toctree}
:maxdepth: 1
:hidden:

io_loader_guide
visualization
gui_stream_processing
annotations
monitoring
dsp_pipeline
gui_plugins
keyboard_shortcuts
```
