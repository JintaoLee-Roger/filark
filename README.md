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

Run in Teminal: 

```bash
filark

# select theme ('light' or 'dark', default is 'dark')
filark --theme light

# open a file 
filark --f /Usx/xxx/xxx/xx58.h5
```

Run in python:

```python
from filark.gui.app import run_app

run_app()

# light theme
run_app(theme='light')
```


From source:

```python
from filark.gui.app import run_app

# 1. from str (h5 file)
run_app(source='data/xxxxx.h5')

# 2. from str (npy file)
run_app(source='data/xxxxx.npy')

# 3. from str (binary file)
run_app(source='data/xxxxx.dat')

# 4. from np array
import numpy as np
data = np.load('data/xxxxx.npy')
run_app(source=data)

# 5. from Tape
# 5.1 h5tape
from filark.io.tapeio import H5Tape, NpyTape, BinTape
tape = H5Tape('data/xxx.h5')

# 5.2 h5tape with keys
tape = H5Tape('data/xxx.h5', 
              datakey="Acquisition/Raw[0]/RawData", 
              fs_key=('Acquisition/Raw[0]', 'OutputDataRate'),
              dx_key=('Acquisition', 'SpatialSamplingInterval'),
              dims_key=('Acquisition/Raw[0]/RawData', 'Dimensions'),
              dxunit_key=('Acquisition', 'SpatialSamplingIntervalUnit'))

# 5.3 NpyTape
tape = NpyTape('data/xxx.npy', 
               dims="nt_nc", # "nt_nc" or "nc_nt"
               fs=1550,
               dx=5,
               dx_unit="m")

# 5.4 BinTape
tape = BinTape('data/xxx.dat', 
               nt=93000,
               nc=6455,
               dtype='float32', # np.dtype(dtype)
               dims="nt_nc", # "nt_nc" or "nc_nt"
               fs=1550,
               dx=5,
               dx_unit="m")

# 5.5 Array
d = np.load('data/xxx.npy')
tape = NpyTape(d, 
               dims="nt_nc", # "nt_nc" or "nc_nt"
               fs=1550,
               dx=5,
               dx_unit="m")


run_app(source=tape)
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
