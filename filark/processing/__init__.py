"""High-level processing APIs built on top of ``dsp`` and ``pipeline``.

``filark.processing`` is the domain-facing layer for "take DAS data and run a
named processing workflow on it".  It deliberately sits above the pure array
operators in ``filark.dsp`` and below the GUI.  In practice this package
contains two different concepts:

- declarative processor plugins used by ROI/full-data processing UIs
- ``DASProcessor``, a metadata-aware bridge that adapts raw arrays/Tapes to a
  transform chain

Execution policy for large datasets lives in ``filark.pipeline`` rather than
here, which keeps the processing API usable from scripts, notebooks, and GUI
code alike.
"""

from .denoise import (
    DENOISE_METHOD_LABELS,
    DEFAULT_DENOISE_PARAMS,
    apply_denoise,
    clone_default_params,
)
from .das_processor import DASProcessor
from .processors import (
    BandpassProcessor,
    FKFanProcessor,
    GaussianProcessor,
    MovingAverageProcessor,
    ProcessorPlugin,
    RealtimeIIRProcessor,
    default_processors,
)

__all__ = [
    # denoise registry
    "DENOISE_METHOD_LABELS",
    "DEFAULT_DENOISE_PARAMS",
    "apply_denoise",
    "clone_default_params",
    # processor
    "DASProcessor",
    "ProcessorPlugin",
    "MovingAverageProcessor",
    "GaussianProcessor",
    "BandpassProcessor",
    "FKFanProcessor",
    "RealtimeIIRProcessor",
    "default_processors",
]
