# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""Research-oriented DSP operators for DAS arrays.

Design notes
------------
- The functions in this package are intentionally array-first and mostly
  framework-agnostic.  They should not know about Qt, vispy, file dialogs, or
  HDF5 layout quirks.
- The canonical layout for algorithm code is ``nt_nc`` (time x channels).
  Helper utilities in ``dsp.base`` convert between ``nt_nc`` and ``nc_nt`` so
  higher layers do not need to duplicate layout logic.
- ``transforms`` provides object-style wrappers suitable for composition and
  stateful streaming filters.
- ``gpu`` mirrors a subset of the CPU API for operators that benefit from
  PyTorch acceleration.

This module re-exports the most commonly used operators so scripts and
notebooks can work from a single namespace during exploratory research.
"""

from . import gpu, transforms
from .base import Layout, as_float2d, from_nt_nc, require_odd, require_positive, to_nt_nc
from .chunk import (
    apply_block_chunks,
    apply_channel_chunks,
    apply_tiled_overlap,
    estimate_block_size,
    estimate_chunk_channels,
    iter_channel_slices,
    iter_tiles_2d,
    make_chunked,
)
from .decomposition import local_svd_denoise, local_svd_denoise_chunked, truncated_svd_reconstruct
from .fk import (
    apply_velocity_filter,
    directional_wedge_denoise,
    directional_wedge_denoise_chunked,
    fk_fan_filter,
    fk_fan_filter_chunked,
    fk_transform,
    get_fk_axes,
)
from .nonlinear import anisotropic_diffusion_2d, structure_tensor_filter, structure_tensor_filter_chunked
from .spatial import (
    decimate_channels,
    demean_space,
    detrend_space,
    equalize_channel_gain,
    gaussian_smooth_space,
    median_smooth_space,
    resample_channels,
    running_mean_space,
    spatial_gradient,
    spatial_laplacian,
    zscore_space,
)
from .temporal import (
    agc_time,
    agc_time_chunked,
    analytic_envelope,
    bandpass_time,
    bandstop_time,
    bandstop_time_chunked,
    butterworth_time_filter,
    clip_amplitude,
    demean_time,
    detrend_time,
    fft_bandpass_time,
    fft_bandpass_time_chunked,
    fft_highpass_time,
    fft_highpass_time_chunked,
    fft_lowpass_time,
    fft_lowpass_time_chunked,
    fir_bandpass_time,
    fir_bandpass_time_chunked,
    fir_filter_time,
    fir_filter_time_chunked,
    fir_highpass_time,
    fir_highpass_time_chunked,
    fir_lowpass_time,
    fir_lowpass_time_chunked,
    gaussian_smooth_time,
    highpass_time,
    iir_bandpass_time,
    iir_bandpass_time_chunked,
    iir_filter_time,
    iir_filter_time_chunked,
    iir_highpass_time,
    iir_highpass_time_chunked,
    iir_lowpass_time,
    iir_lowpass_time_chunked,
    lowpass_time,
    median_smooth_time,
    median_smooth_time_chunked,
    notch_time,
    notch_time_chunked,
    one_bit_normalize,
    running_mean_time,
    temporal_whiten,
    zscore_time,
)
from .tp import (
    linear_radon_forward,
    linear_radon_inverse,
    linear_radon_sparse,
    linear_radon_sparse_chunked,
    parabolic_radon_forward,
    parabolic_radon_inverse,
    parabolic_radon_sparse,
    parabolic_radon_sparse_chunked,
    tp_transform,
)
from .spectral import (
    psd_estimate,
    rms_profile,
    spectral_centroid,
    stft_gather,
)
from .xcorr import (
    ambient_noise_xcorr,
    phase_xcorr,
    xcorr_all_pairs,
    xcorr_gather,
    xcorr_traces,
)

__all__ = [
    # submodules
    "transforms",
    "gpu",
    # base utilities
    "Layout",
    "as_float2d",
    "require_positive",
    "require_odd",
    "to_nt_nc",
    "from_nt_nc",
    # chunk utilities
    "estimate_chunk_channels",
    "estimate_block_size",
    "iter_channel_slices",
    "iter_tiles_2d",
    "apply_channel_chunks",
    "apply_tiled_overlap",
    "apply_block_chunks",
    "make_chunked",
    # f-k
    "get_fk_axes",
    "fk_transform",
    "apply_velocity_filter",
    "fk_fan_filter",
    "fk_fan_filter_chunked",
    "directional_wedge_denoise",
    "directional_wedge_denoise_chunked",
    # temporal – basic
    "demean_time",
    "detrend_time",
    "zscore_time",
    "running_mean_time",
    "gaussian_smooth_time",
    "median_smooth_time",
    "median_smooth_time_chunked",
    "agc_time",
    "agc_time_chunked",
    # temporal – IIR
    "iir_filter_time",
    "iir_bandpass_time",
    "iir_lowpass_time",
    "iir_highpass_time",
    "iir_filter_time_chunked",
    "iir_bandpass_time_chunked",
    "iir_lowpass_time_chunked",
    "iir_highpass_time_chunked",
    # temporal – bandstop / notch
    "bandstop_time",
    "bandstop_time_chunked",
    "notch_time",
    "notch_time_chunked",
    # temporal – FIR
    "fir_filter_time",
    "fir_bandpass_time",
    "fir_lowpass_time",
    "fir_highpass_time",
    "fir_filter_time_chunked",
    "fir_bandpass_time_chunked",
    "fir_lowpass_time_chunked",
    "fir_highpass_time_chunked",
    # temporal – FFT domain
    "fft_bandpass_time",
    "fft_bandpass_time_chunked",
    "fft_lowpass_time",
    "fft_lowpass_time_chunked",
    "fft_highpass_time",
    "fft_highpass_time_chunked",
    # temporal – aliases
    "butterworth_time_filter",
    "bandpass_time",
    "lowpass_time",
    "highpass_time",
    # temporal – amplitude / spectral
    "temporal_whiten",
    "one_bit_normalize",
    "clip_amplitude",
    "analytic_envelope",
    # spatial
    "demean_space",
    "detrend_space",
    "zscore_space",
    "running_mean_space",
    "gaussian_smooth_space",
    "median_smooth_space",
    "equalize_channel_gain",
    "spatial_gradient",
    "spatial_laplacian",
    "decimate_channels",
    "resample_channels",
    # tau-p / Radon
    "linear_radon_forward",
    "linear_radon_inverse",
    "linear_radon_sparse",
    "linear_radon_sparse_chunked",
    "tp_transform",
    "parabolic_radon_forward",
    "parabolic_radon_inverse",
    "parabolic_radon_sparse",
    "parabolic_radon_sparse_chunked",
    # decomposition
    "truncated_svd_reconstruct",
    "local_svd_denoise",
    "local_svd_denoise_chunked",
    # nonlinear
    "structure_tensor_filter",
    "structure_tensor_filter_chunked",
    "anisotropic_diffusion_2d",
    # spectral analysis
    "stft_gather",
    "psd_estimate",
    "rms_profile",
    "spectral_centroid",
    # cross-correlation
    "xcorr_traces",
    "xcorr_gather",
    "xcorr_all_pairs",
    "phase_xcorr",
    "ambient_noise_xcorr",
    # GPU sub-package (access via dsp.gpu.*)
    "gpu",
]
