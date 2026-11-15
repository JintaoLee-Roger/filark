"""Channel-batched GPU FFT filtering must equal the single-pass result exactly."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from filark.dsp.gpu._utils import build_fft_mask_1d, fft_channel_batch_size, fft_filter_gpu  # noqa: E402
from filark.dsp.gpu.temporal import fft_bandpass_time_gpu  # noqa: E402
from filark.dsp.temporal import fft_bandpass_time  # noqa: E402


def _case(nt=1000, nc=37, fs=200.0):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((nt, nc)).astype(np.float32)
    freqs = np.fft.rfftfreq(nt, d=1.0 / fs).astype(np.float32)
    mask = build_fft_mask_1d(freqs, fs=fs, kind="bandpass", fmin=15, fmax=32, taper_hz=0.0)
    return x, mask


@pytest.mark.parametrize("batch", [1, 5, 36])
def test_batched_equals_single_pass(batch):
    x, mask = _case()
    dev = torch.device("cpu")
    full = fft_filter_gpu(x, mask_np=mask, dev=dev)
    part = fft_filter_gpu(x, mask_np=mask, dev=dev, max_channels_per_batch=batch)
    assert part.dtype == np.float32 and part.shape == x.shape
    np.testing.assert_array_equal(part, full)


def test_batched_matches_cpu_reference_and_keeps_input():
    x, mask = _case()
    x0 = x.copy()
    y = fft_filter_gpu(x, mask_np=mask, dev=torch.device("cpu"), max_channels_per_batch=4)
    ref = fft_bandpass_time(x, fs=200.0, fmin=15, fmax=32)
    assert np.linalg.norm(y - ref) / np.linalg.norm(ref) < 1e-5
    np.testing.assert_array_equal(x, x0)


def test_noncontiguous_input_layout():
    x, _ = _case()
    xt = np.ascontiguousarray(x.T)          # stored (nc, nt), like OOI files
    a = fft_bandpass_time_gpu(xt, fs=200.0, fmin=15, fmax=32, device="cpu", layout="nc_nt")
    b = fft_bandpass_time_gpu(x, fs=200.0, fmin=15, fmax=32, device="cpu")
    np.testing.assert_array_equal(a, b.T)


def test_batch_size_only_limits_cuda():
    assert fft_channel_batch_size(36000, 47500, torch.device("cpu")) == 47500
