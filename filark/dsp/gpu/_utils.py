# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""Internal helpers shared across gpu sub-modules."""

from __future__ import annotations

import numpy as np


def require_torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise ImportError(
            "GPU operators require PyTorch.\n"
            "Install it with one of:\n"
            "  pip install torch\n"
            "  pip install 'filark[gpu]'\n"
            "  pip install 'filark[all]'"
        ) from exc


def get_device(device: str = "auto"):
    """Return the best available :class:`torch.device`.

    Priority: CUDA > MPS (Apple Silicon) > CPU.
    """
    torch = require_torch()
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def to_tensor(data: np.ndarray, dev, dtype=None):
    torch = require_torch()
    if dtype is None:
        dtype = torch.float32
    return torch.tensor(np.asarray(data, dtype=np.float32), device=dev, dtype=dtype)


def to_numpy(t) -> np.ndarray:
    t = t.detach()
    if t.device.type != "cpu":
        t = t.cpu()
    return t.numpy()


def build_fft_mask_1d(
    freqs_np: np.ndarray,
    *,
    fs: float,
    kind: str,
    fmin: float | None,
    fmax: float | None,
    taper_hz: float,
) -> np.ndarray:
    """Build a 1-D frequency mask (numpy) for rfft output."""
    nyq = 0.5 * fs
    mask = np.zeros_like(freqs_np, dtype=np.float32)

    if kind == "bandpass":
        lo, hi = float(fmin), float(fmax)       # type: ignore[arg-type]
    elif kind == "lowpass":
        lo, hi = 0.0, float(fmax)               # type: ignore[arg-type]
    else:  # highpass
        lo, hi = float(fmin), nyq               # type: ignore[arg-type]

    mask[(freqs_np >= lo) & (freqs_np <= hi)] = 1.0

    if taper_hz > 0:
        if kind != "highpass":
            hi1 = min(nyq, hi + taper_hz)
            right = (freqs_np > hi) & (freqs_np <= hi1)
            if right.any():
                a = (freqs_np[right] - hi) / max(hi1 - hi, 1e-12)
                mask[right] = (0.5 * (1.0 + np.cos(np.pi * a))).astype(np.float32)
        if kind != "lowpass":
            lo1 = max(0.0, lo - taper_hz)
            left = (freqs_np >= lo1) & (freqs_np < lo)
            if left.any():
                a = (freqs_np[left] - lo1) / max(lo - lo1, 1e-12)
                mask[left] = (0.5 * (1.0 - np.cos(np.pi * a))).astype(np.float32)
    return mask


# Peak device bytes per (time sample x channel) of one rfft-mask-irfft pass:
# float32 input + complex64 spectrum + masked spectrum + float32 output + cuFFT
# workspace. Measured ~5.0x the float32 input on CUDA (47500 ch x 36000 samples
# peaked at 32.6 GB); 6 leaves a margin.
_FFT_BYTES_PER_SAMPLE = 6 * 4
_FFT_FREE_FRACTION = 0.8


def fft_channel_batch_size(nt: int, nc: int, dev) -> int:
    """Channels per batch so one FFT pass fits in currently free device memory.

    Each channel is filtered over its full duration, so batching over channels
    is exact. Only CUDA reports free memory; other devices use one batch.
    """
    torch = require_torch()
    if dev.type != "cuda" or nt <= 0 or nc <= 0:
        return max(nc, 1)
    free, _total = torch.cuda.mem_get_info(dev)
    # Blocks cached by PyTorch's allocator count as used for the driver but are reusable.
    free += torch.cuda.memory_reserved(dev) - torch.cuda.memory_allocated(dev)
    per_channel = nt * _FFT_BYTES_PER_SAMPLE
    return int(min(nc, max(1, (free * _FFT_FREE_FRACTION) // per_channel)))


def fft_filter_gpu(
    x_np: np.ndarray,
    *,
    mask_np: np.ndarray,
    dev,
    max_channels_per_batch: int | None = None,
) -> np.ndarray:
    """Apply a pre-built 1-D rfft mask on GPU. x_np is [nt, nc].

    Large inputs are processed in channel batches sized to free device memory
    (see :func:`fft_channel_batch_size`) instead of relying on the driver to
    spill into host memory; results are identical to a single pass.
    """
    torch = require_torch()
    nt, nc = x_np.shape[0], (x_np.shape[1] if x_np.ndim > 1 else 1)
    mask = torch.tensor(mask_np, device=dev, dtype=torch.float32).unsqueeze(1)

    def one(block):
        x = to_tensor(block, dev)
        X = torch.fft.rfft(x, dim=0)
        X *= mask
        return torch.fft.irfft(X, n=nt, dim=0)

    batch = max_channels_per_batch or fft_channel_batch_size(nt, nc, dev)
    if x_np.ndim < 2 or batch >= nc:
        return to_numpy(one(x_np)).astype(np.float32, copy=False)

    out = np.empty((nt, nc), dtype=np.float32)
    out_t = torch.from_numpy(out)
    for a in range(0, nc, batch):
        b = min(nc, a + batch)
        y = one(x_np[:, a:b])
        out_t[:, a:b].copy_(y)
        del y
    return out
