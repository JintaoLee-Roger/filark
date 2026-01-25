# FiLark

FiLark (**Fi**ber **Lark**) — a lightweight, high-performance framework for **streaming-first** Big Data visualization and analysis in Distributed Fiber Optic Sensing (DAS).

In fiber optic sensing, the sheer data volume quickly overwhelms traditional workflows. FiLark is built to feel “swift and light”—enabling you to **navigate, inspect, and annotate** massive DAS arrays with deterministic control and low-latency rendering.

FiLark bridges raw storage and visual insight, focusing on:

- **High-Efficiency Visualization**: VisPy (OpenGL) rendering for fluid zoom/pan over huge arrays.
- **Streaming as a First-Class Citizen**: designed around incremental data feeds, scheduling, and real-time playback.

---

## ⚠️ Project Status (Early Version)

FiLark is in an **early-stage / experimental** phase. APIs and GUI behaviors may change, and some features are still being stabilized.  
If you encounter issues, please open an issue with logs + a minimal reproduction.

That said, **streaming is the core design point**: most GUI decisions (controls, scheduler behavior, auto-scroll) are implemented to stay deterministic under continuous incoming data.

---

## 🚀 Getting Started

### Installation

Install via PyPI:

```bash
pip install filark
```

Install in editable mode for development:

```bash
git clone https://github.com/JintaoLee-Roger/filark.git
cd filark
pip install -e . 
```

### Fast Usage

Run: 

```bash
filark
```


## 🖥️ GUI & Visualization Controls

FiLark’s GUI emphasizes **deterministic keyboard control**.
The goal is to keep navigation predictable even when data is streaming continuously.

### Deterministic Camera / Scheduler Navigation

* **Left / Right**
  Pan **X** via camera → propagates to scheduler X streaming
  *(camera → scheduler → streaming)*

* **Up / Down**
  Pan **Y** via scheduler (deterministic)

  * optional camera Y sync (if enabled)

* **Y / Shift + Y**
  `scale_y` (zoom in/out Y)

* **X / Shift + X**
  `scale_x` (zoom in/out X)

---

## ⏱️ Realtime Auto-Scroll (StreamingCanvas)

A simplified realtime auto-scroll mixin is included for `StreamingCanvas`, featuring **two-level statistics** and a small demo benchmark mode.

### Keys

* **R** : toggle realtime auto-scroll
* **S** : stop
* **1..8** : set `fs` (samples/sec)
* **+ / =** : speed up
* **- / _** : speed down
* **P** : print **DEBUG** stats
* **Shift + P** : print **SHOWCASE** stats *(for demos / papers)*
* **B** : run a short benchmark
  *(push speed up to a strong cap, report max sustained throughput)*

This mode is designed to validate that the rendering + scheduling loop can sustain **high-throughput streaming** while keeping interactions responsive.

---

## ✍️ Annotation

FiLark provides lightweight annotation tools intended for rapid inspection and labeling during exploration.

### Add / Edit Shapes

* Hold **Ctrl** to add:

  * **BBox** (bounding boxes)
  * **Polyline** (multi-point line)

### Undo

* **Ctrl + Z**
  Undo the **last polyline point** (step-wise rollback while drawing)

> Annotation UX is still evolving in this early version. Expect changes as the labeling workflow becomes more feature-complete.

---
