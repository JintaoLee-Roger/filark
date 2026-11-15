# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""Keep matplotlib toolbars from switching to the wait cursor.

``NavigationToolbar2`` sets ``Cursors.WAIT`` around a draw. On macOS, Qt builds
that cursor from an image (``QImage::toCGImage``), which can abort the process
with SIGTRAP inside CoreGraphics. FiLark never needs the wait cursor in its
figure windows, so the context manager is replaced by a no-op once at startup.
"""

from __future__ import annotations

import contextlib


def disable_matplotlib_wait_cursor() -> None:
    try:
        from matplotlib.backend_bases import NavigationToolbar2
    except ImportError:
        return
    NavigationToolbar2._wait_cursor_for_draw_cm = lambda self: contextlib.nullcontext()
