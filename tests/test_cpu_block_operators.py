"""Block-wise CPU operators must match the whole-array SciPy results."""
import numpy as np
import pytest
from scipy import signal

import filark.dsp.temporal as temporal
from filark.dsp.temporal import analytic_envelope, build_iir_sos, detrend_time, iir_filter_time


def _data(nt, nc, dtype, seed=31):
    rng = np.random.default_rng(seed)
    t = np.arange(nt)[:, None]
    x = 200.0 * rng.normal(size=(nt, nc)) + 0.3 * t * rng.normal(size=(1, nc)) + rng.normal(size=(1, nc)) * 500
    return x.astype(dtype)


@pytest.mark.parametrize('zero_phase', [False, True])
@pytest.mark.parametrize('block_channels', [1, 5, 1000])
def test_iir_kernel_fallback_bit_identical(monkeypatch, zero_phase, block_channels):
    data = _data(3000, 37, np.float32)
    sos = build_iir_sos(fs=1550, kind='bandpass', fmin=1, fmax=50, order=4)
    fn = signal.sosfiltfilt if zero_phase else signal.sosfilt
    expected = fn(sos, data.astype(np.float64), axis=0).astype(np.float32)
    monkeypatch.setattr(temporal, '_BLOCK_BYTES', 3100 * 8 * block_channels)
    kw = dict(fs=1550, kind='bandpass', fmin=1, fmax=50, order=4, zero_phase=zero_phase)
    np.testing.assert_array_equal(iir_filter_time(data, **kw), expected)
    monkeypatch.setattr(temporal, '_sosfilt_kernel', None)
    np.testing.assert_array_equal(iir_filter_time(data, **kw), expected)


@pytest.mark.parametrize('kind,kw', [('lowpass', dict(fmax=50)), ('highpass', dict(fmin=5))])
@pytest.mark.parametrize('order', [1, 3, 4])
def test_iir_zero_phase_other_designs(kind, kw, order):
    # Odd orders leave zero coefficients, which shorten sosfiltfilt's padding.
    data = _data(2000, 9, np.float32)
    sos = signal.butter(order, list(kw.values())[0], fs=1550, btype=kind, output='sos')
    expected = signal.sosfiltfilt(sos, data.astype(np.float64), axis=0).astype(np.float32)
    actual = iir_filter_time(data, fs=1550, kind=kind, order=order, zero_phase=True, **kw)
    np.testing.assert_array_equal(actual, expected)


def test_iir_short_record_falls_back_to_causal():
    data = _data(20, 3, np.float32)
    sos = build_iir_sos(fs=1550, kind='bandpass', fmin=1, fmax=50, order=4)
    with pytest.warns(RuntimeWarning, match='short sequence'):
        actual = iir_filter_time(data, fs=1550, kind='bandpass', fmin=1, fmax=50, order=4, zero_phase=True)
    np.testing.assert_array_equal(actual, signal.sosfilt(sos, data.astype(np.float64), axis=0).astype(np.float32))


@pytest.mark.parametrize('kind', ['linear', 'constant'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int32])
@pytest.mark.parametrize('block_rows', [1, 7, 100000])
def test_detrend_matches_scipy(monkeypatch, kind, dtype, block_rows):
    data = _data(3001, 13, dtype)
    original = data.copy()
    monkeypatch.setattr(temporal, '_BLOCK_BYTES', 13 * 8 * block_rows)
    actual = detrend_time(data, kind=kind)
    expected = signal.detrend(data.astype(np.float64), axis=0, type=kind)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5 * np.abs(expected).max())
    np.testing.assert_array_equal(data, original)


def test_detrend_single_sample_and_bad_kind():
    assert np.allclose(detrend_time(np.ones((1, 4), np.float32)), 0.0)
    with pytest.raises(ValueError):
        detrend_time(np.ones((5, 4), np.float32), kind='quadratic')


@pytest.mark.parametrize('nt', [1, 2, 3000, 3001])
@pytest.mark.parametrize('dtype,tol', [(np.float32, 2e-6), (np.float64, 1e-7), (np.int32, 1e-7)])
@pytest.mark.parametrize('nc', [1, 7, 200])
def test_envelope_matches_scipy(nt, dtype, tol, nc):
    data = _data(nt, nc, dtype)
    original = data.copy()
    actual = analytic_envelope(data)
    expected = np.abs(signal.hilbert(data.astype(np.float64), axis=0))
    assert actual.dtype == np.float32 and actual.shape == data.shape
    assert np.linalg.norm(actual - expected) <= tol * np.linalg.norm(expected)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize('nt,nx', [(512, 64), (513, 65), (512, 65), (513, 64)])
@pytest.mark.parametrize('kw', [
    dict(vmin=1000., vmax=8000.),
    dict(vmin=1000., vmax=8000., taper_width=200.),
    dict(fmin=5., fmax=200., kmin_abs=1e-3, kmax_abs=0.05),
])
def test_fk_gpu_half_spectrum_matches_full(nt, nx, kw):
    torch = pytest.importorskip('torch')
    from filark.dsp.gpu.fk import fk_fan_filter_gpu
    dt, dx = 1 / 1550., 5.112
    x = _data(nt, nx, np.float32)
    actual = fk_fan_filter_gpu(x, dt=dt, dx=dx, device='cpu', **kw)

    f = np.abs(np.fft.fftfreq(nt, dt)).astype(np.float32)[:, None]
    k = np.abs(np.fft.fftfreq(nx, dx)).astype(np.float32)[None, :]
    v = f / (k + 1e-12)
    m = np.ones((nt, nx), np.float32); tw = kw.get('taper_width', 0.)
    if 'vmin' in kw:
        lo, hi = kw['vmin'], kw['vmax']
        z = (v >= lo) & (v < lo + tw)
        m[v < lo] = 0; m[z] = (0.5 * (1 - np.cos(np.pi * (v[z] - lo) / max(tw, 1e-12))))
        z = (v > hi - tw) & (v <= hi)
        m[v > hi] = 0; m[z] *= 0.5 * (1 + np.cos(np.pi * (v[z] - (hi - tw)) / max(tw, 1e-12)))
    if 'fmin' in kw:
        m[(f < kw['fmin'])[:, 0], :] = 0; m[(f > kw['fmax'])[:, 0], :] = 0
        m[:, (k < kw['kmin_abs'])[0]] = 0; m[:, (k > kw['kmax_abs'])[0]] = 0
    expected = np.fft.ifft2(np.fft.fft2(x.astype(np.float64)) * m).real
    assert np.linalg.norm(actual - expected) <= 1e-4 * np.linalg.norm(expected)
