# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""
AnalysisPlugin — base class for user-defined analysis plugins.

Two operation modes:

Mode 1 — matplotlib Figure
    Implement ``run()`` to return a ``matplotlib.figure.Figure``
    (or list of figures, or None).  The framework displays the figure(s)
    in a window equipped with ``NavigationToolbar2QT`` (save / zoom / pan
    come for free from matplotlib's own toolbar).

Mode 2 — custom Qt widget
    Implement ``create_widget()`` to return a ``QWidget``.
    The framework shows it as an independent floating window.
    The plugin is responsible for all UI logic and data access inside the widget.

If ``create_widget()`` returns ``None`` (the default), mode 1 is used.

``param_spec`` follows the same format as ``ProcessorPlugin``:
identical type tokens are supported, including the extended types
``str``, ``file``, ``int_range``, ``float_range``.

``param_spec`` is used only in mode 1 (the framework builds a param dialog
automatically).  Mode 2 plugins may leave it empty and manage their own UI.
"""

from __future__ import annotations


class AnalysisPlugin:
    """Base class for user-defined DAS analysis plugins."""

    label: str = "Unknown Analysis"
    key: str = "unknown_analysis"
    param_spec: list[dict] = []

    def default_params(self) -> dict:
        return {e["key"]: e["default"] for e in self.param_spec}

    def validate_params(
        self,
        params: dict,
        *,
        fs: float,
        dx: float,
        nt: int,
        nc: int,
    ) -> dict:
        """Clamp / normalise params before use. Override in subclasses."""
        return dict(params)

    # ------------------------------------------------------------------
    # Mode 1 — return matplotlib Figure(s)
    # ------------------------------------------------------------------

    def run(
        self,
        source,
        *,
        params: dict,
        roi=None,
    ):
        """
        Run analysis and return matplotlib Figure(s).

        Parameters
        ----------
        source : Tape
            Full data tape.
        params : dict
            Validated parameter dict.
        roi : dict | None
            ROI info dict with keys ``data``, ``fs``, ``dx``, ``roi``
            (same as what ``MainWindow._extract_roi_data`` produces).
            ``None`` means "use full source".

        Returns
        -------
        matplotlib.figure.Figure | list[Figure] | None
            Return a Figure or list of Figures to display in the framework
            window.  Return ``None`` if the plugin handles display itself
            (e.g. calls ``plt.show()`` internally) or delegates to mode 2.
        """
        return None

    # ------------------------------------------------------------------
    # Mode 2 — return a custom QWidget
    # ------------------------------------------------------------------

    def create_widget(self, source, parent=None):
        """
        Return a custom QWidget for mode 2, or None to use mode 1.

        The framework calls this first.  If a widget is returned it is shown
        as an independent floating window; ``run()`` is not called.
        """
        return None
