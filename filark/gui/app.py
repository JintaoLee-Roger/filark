# Copyright (c) 2026 Jintao Li. 
# Zhejiang University (ZJU).
# 
# Licensed under the MIT License.

import sys
from PySide6.QtWidgets import QApplication
from .main_window import MainWindow
from importlib import resources

import argparse
import sys

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="filark", description="Filark GUI")
    p.add_argument("--f", "-f", default=None, help="Input file or stream spec")
    p.add_argument("--backend", choices=["pyqt5", "pyqt6", "pyside6"], default=None)
    p.add_argument("--theme", type=str, choices=["dark", "light"], default="dark")
    return p



def apply_theme(app: QApplication, theme_name: str = "dark"):
    if not theme_name.endswith(".qss"):
        theme_name += ".qss"
    qss = resources.files("filark.gui.styles").joinpath(theme_name).read_text(encoding="utf-8")
    app.setStyleSheet(qss)


def run_app(argv: list[str] | None = None, source = None):
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(argv)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    apply_theme(app, args.theme)
    
    window = MainWindow(theme=args.theme, source=source)
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    run_app()