"""Removing redundant input copies must preserve double-precision IIR results."""
import numpy as np
import pytest
from scipy import signal

from filark.dsp.temporal import build_iir_sos, iir_filter_time, iir_filter_time_rt
from filark.io.stream_processing import RealtimeDisplaySOSFilter


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int16])
@pytest.mark.parametrize('zero_phase', [False, True])
def test_offline_matches_explicit_double_reference(dtype, zero_phase):
    # Strided input exercises SciPy's layout conversion as well as precision.
    data=(np.random.default_rng(25).normal(size=(12400,8))*100).astype(dtype)[::2,::2]
    original=data.copy()
    sos=build_iir_sos(fs=1550,kind='bandpass',fmin=1,fmax=50,order=4)
    fn=signal.sosfiltfilt if zero_phase else signal.sosfilt
    expected=fn(sos,data.astype(np.float64),axis=0).astype(np.float32)
    actual=iir_filter_time(data,fs=1550,kind='bandpass',fmin=1,fmax=50,order=4,zero_phase=zero_phase)
    assert actual.dtype==np.float32
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(data,original)


@pytest.mark.parametrize('sos_dtype',[np.float32,np.float64])
def test_streaming_double_state_and_display_match_reference(sos_dtype):
    data=np.random.default_rng(26).normal(size=(6200,8)).astype(np.float32)
    original=data.copy()
    sos=build_iir_sos(fs=1550,kind='bandpass',fmin=1,fmax=50,order=4).astype(sos_dtype)
    reference,state=signal.sosfilt(sos.astype(np.float64),data.astype(np.float64),axis=0,
                                  zi=np.zeros((len(sos),2,data.shape[1]),dtype=np.float64))
    zi=None; chunks=[]; display=RealtimeDisplaySOSFilter(sos); images=[]
    for block in np.array_split(data,7):
        y,zi=iir_filter_time_rt(block,sos,zi=zi)
        assert y.dtype==np.float32 and zi.dtype==np.float64
        chunks.append(y)
        images.append(display(block.T).T)
    np.testing.assert_array_equal(np.concatenate(chunks),reference.astype(np.float32))
    np.testing.assert_array_equal(zi,state)
    np.testing.assert_array_equal(np.concatenate(images),reference.astype(np.float32))
    np.testing.assert_array_equal(data,original)


@pytest.mark.parametrize('dtype', [np.float32, np.int16])
@pytest.mark.parametrize('zero_phase', [False, True])
def test_offline_channel_blocks_bit_identical(monkeypatch, dtype, zero_phase):
    # Force many channel blocks (including a ragged last block) and compare
    # against SciPy on the whole array in double precision.
    import filark.dsp.temporal as temporal
    data=(np.random.default_rng(27).normal(size=(3000,37))*100).astype(dtype)
    sos=build_iir_sos(fs=1550,kind='bandpass',fmin=1,fmax=50,order=4)
    fn=signal.sosfiltfilt if zero_phase else signal.sosfilt
    expected=fn(sos,data.astype(np.float64),axis=0).astype(np.float32)
    monkeypatch.setattr(temporal,'_BLOCK_BYTES',(3000+54)*8*5 if zero_phase else 3000*8*5)  # 5 channels per block
    actual=iir_filter_time(data,fs=1550,kind='bandpass',fmin=1,fmax=50,order=4,zero_phase=zero_phase)
    assert actual.dtype==np.float32 and actual.flags.f_contiguous
    np.testing.assert_array_equal(actual,expected)
