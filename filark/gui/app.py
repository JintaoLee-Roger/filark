# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""Qt application entry points for launching the FiLark GUI."""

from __future__ import annotations

import argparse
from importlib import resources
import sys
from typing import TYPE_CHECKING

from .config import GuiConfig

if TYPE_CHECKING:
    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow


_GUI_INSTALL_HINT = (
    "FiLark GUI dependencies are missing.\n"
    'Run this command, then try again:\n\n'
    '  pip install "filark[gui]"'
)


def _require_gui_runtime() -> tuple[type["QApplication"], type["MainWindow"]]:
    """Import Qt runtime objects only when the GUI is actually launched."""
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError as exc:
        raise ImportError(_GUI_INSTALL_HINT) from exc

    try:
        from .main_window import MainWindow
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("PySide6"):
            raise ImportError(_GUI_INSTALL_HINT) from exc
        raise

    return QApplication, MainWindow


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="filark", description="Filark GUI")
    p.add_argument("--f", "-f", default=None, help="Input file or stream spec")
    p.add_argument(
        "--backend",
        choices=["pyqt5", "pyqt6", "pyside6"],
        default=None,
    )
    p.add_argument(
        "--theme",
        type=str,
        choices=["dark", "light"],
        default="dark",
    )
    return p


def apply_theme(app: "QApplication", theme_name: str = "dark") -> None:
    if not theme_name.endswith(".qss"):
        theme_name += ".qss"
    qss = resources.files("filark.gui.styles").joinpath(theme_name).read_text(encoding="utf-8")
    app.setStyleSheet(qss)


def run_app(
    argv: list[str] | None = None,
    *,
    source=None,
    theme: str | None = None,
    display_aspect: float | None = None,
    display_aspect_limits: tuple[float, float] | None = None,
    auto_display_aspect: bool | None = None,
    x_role: str | None = None,
    y_role: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    x_unit: str | None = None,
    y_unit: str | None = None,
    invert_y: bool | str | None = None,
    async_io: bool | None = None,
    cfg: GuiConfig | None = None,
) -> None:
    """
    Launch Filark GUI.

    Parameters
    ----------
    argv:
        Command-line arguments (default: sys.argv[1:])
    source:
        Initial data source (Tape-like object), expert mode shortcut
    theme:
        GUI theme shortcut ("dark" or "light")
    display_aspect:
        Optional initial display aspect for low-channel or stretched views
    display_aspect_limits:
        Optional clamp range for display aspect, e.g. ``(0.001, 20.0)``
    auto_display_aspect:
        Whether to auto-adapt display aspect from the current canvas height
    x_role:
        Optional semantic role for the displayed x axis, e.g. ``"time"`` or ``"trace"``
    y_role:
        Optional semantic role for the displayed y axis, e.g. ``"channel"`` or ``"time"``
    x_label:
        Optional custom x-axis label
    y_label:
        Optional custom y-axis label
    x_unit:
        Optional x-axis unit label
    y_unit:
        Optional y-axis unit label
    invert_y:
        Whether to show smaller y coordinates at the top of the canvas.
        ``"auto"`` inverts y for time/depth y axes.
    async_io:
        Whether GUI streaming reads run on the background IO thread
    cfg:
        Full GUI configuration (advanced usage)
    """
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(argv)

    cfg = cfg or GuiConfig()

    # merge CLI arguments
    if args.f is not None:
        cfg.source = args.f
    if args.theme is not None:
        cfg.theme = args.theme

    # merge shortcut keyword arguments
    cfg.merge(
        source=source,
        theme=theme,
        display_aspect=display_aspect,
        display_aspect_limits=display_aspect_limits,
        auto_display_aspect=auto_display_aspect,
        x_role=x_role,
        y_role=y_role,
        x_label=x_label,
        y_label=y_label,
        x_unit=x_unit,
        y_unit=y_unit,
        invert_y=invert_y,
        async_io=async_io,
    )

    QApplication, MainWindow = _require_gui_runtime()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    apply_theme(app, cfg.theme)

    window = MainWindow(cfg=cfg)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()
