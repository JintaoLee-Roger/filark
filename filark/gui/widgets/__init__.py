"""Reusable Qt widgets shared by the desktop GUI.

These are presentation-level building blocks such as the navigation bar,
drawer, typography helpers, and source switcher.  They should stay generic and
avoid owning application-specific business logic.
"""

from .collapsible_section import CollapsibleSection

__all__ = ["CollapsibleSection"]
