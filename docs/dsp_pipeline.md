# DSP And Pipelines

FiLark's DSP layer is usable without Qt. The GUI imports it, but the functions
and pipelines are ordinary Python tools for DAS arrays.

## Array Layout

Algorithm code uses semantic `nt_nc` layout: time by channel. Helpers in
`filark.dsp.base` convert between file layout and this canonical form.

## Function API

The modules under `filark.dsp` expose array-first functions:

- `temporal`: IIR/FIR/FFT filters, AGC, running mean, z-score, whitening,
  envelope, clipping;
- `spatial`: channel smoothing, gradients, Laplacian, resampling, gain
  equalization;
- `fk`: f-k transforms, fan filters, directional wedge denoising;
- `decomposition`: truncated SVD and local SVD denoising;
- `tp`: linear/parabolic Radon transforms and sparse variants;
- `xcorr`: correlation utilities;
- `spectral`: PSD, RMS, spectral centroid, STFT-related helpers;
- `gpu`: PyTorch-backed variants for selected heavy operators.

Example:

```python
import numpy as np
from filark.dsp.temporal import demean_time, iir_bandpass_time

data = np.asarray(data, dtype=np.float32)  # shape: nt x nc
data = demean_time(data)
data = iir_bandpass_time(data, fs=1550, fmin=5, fmax=120)
```

## Transform Objects

`filark.dsp.transforms` wraps common functions as composable objects:

```python
import filark.dsp.transforms as T

pipe = T.Compose([
    T.DeMeanTime(),
    T.IIRBandpassTime(fs=1550, fmin=5, fmax=120),
    T.AGCTime(half_window=80),
])

out = pipe(data)
```

Realtime transform classes preserve state across chunks:

```python
pipe = T.Compose([
    T.RealtimeIIRBandpassFilter(fs=1550, fmin=5, fmax=120),
    T.RealtimeAGC(half_window=80),
])
```

## Chunk Helpers

`filark.dsp.chunk` provides iterators and wrappers for channel chunks and 2-D
tiles. Use these when an operation needs overlap or halo regions.

Important helpers include:

- `iter_channel_slices`
- `iter_tiles_2d`
- `estimate_chunk_channels`
- `estimate_block_size`
- `apply_channel_chunks`
- `apply_block_chunks`
- `apply_tiled_overlap`
- `make_chunked`

## Pipeline Layer

`filark.pipeline.BasePipeline` provides one entry point for in-memory and
large-source execution. It can run a callable over full arrays or over chunked
Tape sources with overlap.

Typical use:

```python
from filark.pipeline.base import BasePipeline

pipeline = BasePipeline(algorithm=lambda x: pipe(x))
out = pipeline(source, chunked=True, block_t=4096, block_c=512)
```

Use the pipeline layer when the output should be saved or when a full source is
too large for memory.

## GUI Processor Plugins

ROI processing in the GUI uses `ProcessorPlugin` objects from
`filark.processing.processors`. They describe UI parameters and process one
ROI-sized block. Chunking and export are handled by the ROI window and pipeline
layer.

For registration details, see {doc}`gui_plugins`.
