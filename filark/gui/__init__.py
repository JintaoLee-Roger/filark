# Copyright (c) 2026 Jintao Li. 
# Zhejiang University (ZJU).
# 
# Licensed under the MIT License.

"""Qt GUI package for the desktop application.

``filark.gui`` is intentionally an orchestration layer.  It owns widgets,
panel composition, plugin wiring, and user interaction state, but it should
avoid implementing file-format logic or numerical DSP kernels directly.

The main sub-areas are:

- ``modules``: drawer panels such as Load, Visuals, Annotation, Algorithms
- ``plugins``: GUI-facing plugin protocols for monitor/analysis extensions
- ``roi``: ROI windows and analysis/process views derived from selections
- ``widgets``: reusable Qt building blocks shared by panels
"""

from .app import run_app
from .config import GuiConfig

__all__ = ["run_app", "GuiConfig"]
