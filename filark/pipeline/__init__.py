# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""Chunked execution and export orchestration.

The pipeline layer sits between pure processing logic and user interfaces.  It
knows how to iterate large sources in chunks, apply overlap/taper policy, and
write outputs to disk with metadata.  It should not define GUI schemas or DSP
operators itself.
"""

from .base import BasePipeline
from .denoise import DenoisePipeline
from .execution import ProcessorRunResult, run_nt_nc_transform_to_file, run_processor_to_file

__all__ = [
    "BasePipeline",
    "DenoisePipeline",
    "ProcessorRunResult",
    "run_processor_to_file",
    "run_nt_nc_transform_to_file",
]
