"""Shared helpers for surfacing exceptions during interactive use."""

from __future__ import annotations

import sys
import traceback


def print_exception(context: str, *, prefix: str = "FiLark") -> None:
    """Print a short context line plus the active traceback to stderr."""
    print(f"[{prefix}] {context}", file=sys.stderr)
    traceback.print_exc()
