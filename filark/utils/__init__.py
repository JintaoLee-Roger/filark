# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""filark.utils — high-level, framework-agnostic DAS utility functions.

Covers time/space axis bookkeeping, deterministic measurement conversions,
annotation helpers, and other small utilities that do not belong cleanly to
``dsp`` (signal operators) or ``io`` (file access).
"""

from .time import (
    sample_to_time,
    time_to_sample,
    tape_time_axis,
    format_timestamp,
    format_duration,
    parse_timestamp,
)
from .coords import (
    tape_channel_axis,
    sample_to_coord,
    coord_to_sample,
)
from .measurements import (
    canonical_measurement_name,
    time_derivative,
    time_integral,
    strain_to_strain_rate,
    strain_rate_to_strain,
    strain_to_microstrain,
    microstrain_to_strain,
    strain_rate_to_microstrain_rate,
    microstrain_rate_to_strain_rate,
    extension_to_strain,
    strain_to_extension,
    extension_rate_to_strain_rate,
    strain_rate_to_extension_rate,
    phase_to_extension,
    extension_to_phase,
    phase_to_strain,
    strain_to_phase,
)
from .memory import MemoryMonitor
from .annotations import (
    describe_source,
    load_annotations,
    annotation_source_info,
    annotation_items,
    get_annotation_item,
    annotation_bounds,
    open_annotation_source,
    slice_annotation_data,
)

__all__ = [
    # memory
    "MemoryMonitor",
    # time
    "sample_to_time",
    "time_to_sample",
    "tape_time_axis",
    "format_timestamp",
    "format_duration",
    "parse_timestamp",
    # coords
    "tape_channel_axis",
    "sample_to_coord",
    "coord_to_sample",
    # measurements
    "canonical_measurement_name",
    "time_derivative",
    "time_integral",
    "strain_to_strain_rate",
    "strain_rate_to_strain",
    "strain_to_microstrain",
    "microstrain_to_strain",
    "strain_rate_to_microstrain_rate",
    "microstrain_rate_to_strain_rate",
    "extension_to_strain",
    "strain_to_extension",
    "extension_rate_to_strain_rate",
    "strain_rate_to_extension_rate",
    "phase_to_extension",
    "extension_to_phase",
    "phase_to_strain",
    "strain_to_phase",
    # annotations
    "describe_source",
    "load_annotations",
    "annotation_source_info",
    "annotation_items",
    "get_annotation_item",
    "annotation_bounds",
    "open_annotation_source",
    "slice_annotation_data",
]
