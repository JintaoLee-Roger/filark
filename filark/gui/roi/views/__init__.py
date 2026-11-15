"""ROI analysis/process views.

Each module in this package presents one concrete ROI-centric view such as PSD,
STFT, FK, SVD, denoising, or xcorr.  The modules are Qt widgets, but the heavy
numerical work should remain in ``dsp`` or ``processing``.
"""
