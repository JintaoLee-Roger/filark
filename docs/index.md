# FiLark Documentation

<div class="landing-hero">
  <div class="landing-hero-main">
    <div class="hero-kicker">Distributed acoustic sensing</div>
    <div class="hero-title">
      Streaming-first DAS visualization, annotation, and analysis integration.
    </div>
    <div class="hero-body">
      FiLark 0.0.5 keeps large DAS recordings navigable without forcing every
      workflow through short pre-extracted snippets. This complete release
      includes the desktop viewer, annotation workflow, monitor integration,
      and scriptable I/O, DSP, pipeline, and visualization layers.
    </div>
    <div class="hero-actions">
      <a class="hero-link" href="getting_started.html">Start with setup</a>
      <a class="hero-link" href="concepts.html">Read architecture</a>
    </div>
  </div>
</div>

## Demo Videos

The following videos show FiLark workflows from interactive browsing to
annotation, monitoring, review, and plugin-based analysis:

<div class="video-grid">
  <div class="video-card">
    <iframe src="https://www.youtube.com/embed/27FLH0Z71fk" title="Exploration and processing demo" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
    <strong>Exploration and Processing</strong>
    <a class="video-backup" href="https://www.bilibili.com/video/BV1HiLF6kEcg/?share_source=copy_web&amp;vd_source=cc7827632680dba5ffd3be7e2003fa47">Bilibili backup</a>
  </div>
  <div class="video-card">
    <iframe src="https://www.youtube.com/embed/hUlHTDGpIC8" title="Large data and annotation demo" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
    <strong>Large Data and Annotation</strong>
    <a class="video-backup" href="https://www.bilibili.com/video/BV1pvLF6VELW/?share_source=copy_web&amp;vd_source=cc7827632680dba5ffd3be7e2003fa47">Bilibili backup</a>
  </div>
  <div class="video-card">
    <iframe src="https://www.youtube.com/embed/HptglyXhii0" title="Monitoring and recheck demo" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
    <strong>Monitoring and Recheck</strong>
    <a class="video-backup" href="https://www.bilibili.com/video/BV1rvLF6VEmJ/?share_source=copy_web&amp;vd_source=cc7827632680dba5ffd3be7e2003fa47">Bilibili backup</a>
  </div>
  <div class="video-card">
    <iframe src="https://www.youtube.com/embed/9oRkvGtCsSA" title="Source localization plugins demo" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
    <strong>Source Localization Plugins</strong>
    <a class="video-backup" href="https://www.bilibili.com/video/BV1Desp6yE49/">Bilibili backup</a>
  </div>
</div>

## Choose A Path

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Getting Started
Install FiLark, launch the GUI, and open a first data source.
+++
{doc}`getting_started`
:::

:::{grid-item-card} Basic Usage
Walk through loading, navigation, ROI analysis, annotation, monitoring, and recheck.
+++
{doc}`basic_usage`
:::

:::{grid-item-card} Concepts
Understand the streaming-first design, package boundaries, and coordinate model.
+++
{doc}`concepts`
:::

:::{grid-item-card} Guides
Task-oriented pages for loading, visualization, annotation, monitoring, DSP, and plugins.
+++
{doc}`guides`
:::

:::{grid-item-card} Showcase Workflows
Read how the pieces fit together in common DAS analysis sessions.
+++
{doc}`showcase`
:::
::::

## Core Capabilities

::::{grid} 1 2 2 3
:gutter: 2

:::{grid-item-card} Streaming I/O
H5, NPY, binary, array, and folder-backed sources are exposed through one
metadata-aware Tape interface.
:::

:::{grid-item-card} Bounded-Memory Viewer
The main canvas uses a fixed-size GPU texture and updates only newly visible
regions while navigating.
:::

:::{grid-item-card} Display Processing
Attach lightweight stream preprocessing and display filters for exploratory
browsing.
:::

:::{grid-item-card} Coordinate-Stable Labels
Annotations and monitor events are stored in time-channel coordinates, not
display pixels.
:::

:::{grid-item-card} Algorithm-In-The-Loop
Monitor plugins connect external algorithms to the viewer while the interface
remains responsive.
:::

:::{grid-item-card} ROI Analysis
Open PSD, STFT, F-K, tau-p, RMS, SVD, xcorr, and denoising views from selected
regions.
:::

:::{grid-item-card} Headless Processing
Use `filark.io`, `filark.dsp`, and `filark.pipeline` without starting Qt.
:::
::::

## Project Notes

This site documents the complete FiLark 0.0.5 release. It borrows the project
motivation from the FiLark paper, but it is written for people who need to run,
configure, extend, or maintain FiLark.

```{toctree}
:maxdepth: 2
:hidden:

getting_started
basic_usage
concepts
guides
showcase
extensions
```
