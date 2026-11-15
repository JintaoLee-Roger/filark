# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""
GPU-accelerated DSP operators using PyTorch.

Install with ``pip install torch``, ``pip install 'filark[gpu]'``, or
``pip install 'filark[all]'``.

All public functions accept NumPy arrays and return NumPy arrays;
device transfers are handled internally.

Sub-modules mirror the CPU dsp layout:

    dsp.gpu.temporal      FFT/FIR temporal filters
    dsp.gpu.fk            f-k fan filter
    dsp.gpu.xcorr         cross-correlation
    dsp.gpu.decomposition local SVD denoising
    dsp.gpu.nonlinear     anisotropic diffusion
    dsp.gpu.spatial       spatial-domain channel operators
    dsp.gpu.radon         Radon / tau-p transforms
    dsp.gpu.spectral      spectral analysis (STFT, PSD, RMS, centroid)

Usage
-----
>>> import filark.dsp.gpu as gdsp
>>> out = gdsp.temporal.fft_bandpass_time_gpu(data, fs=500, fmin=5, fmax=200)
>>> out = gdsp.fk.fk_fan_filter_gpu(data, dt=0.002, dx=5, vmin=400)
>>> out = gdsp.spatial.demean_space_gpu(data)
>>> R = gdsp.radon.linear_radon_forward_gpu(data, p_vals, dt=0.002, dx=5.0)
>>> f, psd = gdsp.spectral.psd_estimate_gpu(data, fs=500)
"""

from . import decomposition, fk, nonlinear, radon, spatial, spectral, temporal, xcorr
from ._utils import get_device
from .decomposition import local_svd_denoise_gpu
from .fk import fk_fan_filter_gpu
from .nonlinear import anisotropic_diffusion_2d_gpu
from .radon import (
    linear_radon_forward_gpu,
    linear_radon_inverse_gpu,
    linear_radon_sparse_gpu,
    parabolic_radon_forward_gpu,
    parabolic_radon_inverse_gpu,
    parabolic_radon_sparse_gpu,
)
from .spatial import (
    demean_space_gpu,
    detrend_space_gpu,
    equalize_channel_gain_gpu,
    gaussian_smooth_space_gpu,
    median_smooth_space_gpu,
    resample_channels_gpu,
    running_mean_space_gpu,
    spatial_gradient_gpu,
    spatial_laplacian_gpu,
    zscore_space_gpu,
)
from .spectral import (
    psd_estimate_gpu,
    rms_profile_gpu,
    spectral_centroid_gpu,
    stft_gather_gpu,
)
from .temporal import (
    fft_bandpass_time_gpu,
    fft_highpass_time_gpu,
    fft_lowpass_time_gpu,
    fir_filter_time_gpu,
    iir_filter_time_rt_gpu,
    iir_bandpass_time_rt_gpu,
    iir_lowpass_time_rt_gpu,
    iir_highpass_time_rt_gpu,
)
from .xcorr import xcorr_all_pairs_gpu, xcorr_gather_gpu

__all__ = [
    # sub-modules
    "temporal",
    "fk",
    "xcorr",
    "decomposition",
    "nonlinear",
    "spatial",
    "radon",
    "spectral",
    # device utility
    "get_device",
    # temporal – FFT / FIR
    "fft_bandpass_time_gpu",
    "fft_lowpass_time_gpu",
    "fft_highpass_time_gpu",
    "fir_filter_time_gpu",
    # temporal – realtime IIR
    "iir_filter_time_rt_gpu",
    "iir_bandpass_time_rt_gpu",
    "iir_lowpass_time_rt_gpu",
    "iir_highpass_time_rt_gpu",
    # fk
    "fk_fan_filter_gpu",
    # xcorr
    "xcorr_gather_gpu",
    "xcorr_all_pairs_gpu",
    # decomposition
    "local_svd_denoise_gpu",
    # nonlinear
    "anisotropic_diffusion_2d_gpu",
    # spatial
    "demean_space_gpu",
    "detrend_space_gpu",
    "zscore_space_gpu",
    "running_mean_space_gpu",
    "gaussian_smooth_space_gpu",
    "median_smooth_space_gpu",
    "equalize_channel_gain_gpu",
    "spatial_gradient_gpu",
    "spatial_laplacian_gpu",
    "resample_channels_gpu",
    # radon
    "linear_radon_forward_gpu",
    "linear_radon_inverse_gpu",
    "linear_radon_sparse_gpu",
    "parabolic_radon_forward_gpu",
    "parabolic_radon_inverse_gpu",
    "parabolic_radon_sparse_gpu",
    # spectral
    "stft_gather_gpu",
    "psd_estimate_gpu",
    "rms_profile_gpu",
    "spectral_centroid_gpu",
]
