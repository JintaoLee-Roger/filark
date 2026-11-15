# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""
ParamDialog — floating, non-modal dialog built from a ``param_spec`` list.

The dialog is created lazily (once per processor) and hidden rather than
destroyed on close, so parameter values survive method switches.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

_SP_MAX_W = 120


# ---------------------------------------------------------------------------
# Composite widget wrappers for new param types
# ---------------------------------------------------------------------------

class _FileWidget(QWidget):
    """QLineEdit + Browse button for 'file' param type."""

    textChanged = Signal(str)

    def __init__(self, default: str = "", filter: str = "", parent=None):
        super().__init__(parent)
        self._filter = filter
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        self._le = QLineEdit(str(default))
        self._le.setPlaceholderText("(path)")
        self._btn = QPushButton("…")
        self._btn.setFixedWidth(28)
        h.addWidget(self._le, 1)
        h.addWidget(self._btn)
        self._le.textChanged.connect(self.textChanged)
        self._btn.clicked.connect(self._browse)

    def text(self) -> str:
        return self._le.text()

    def setText(self, v: str):
        self._le.setText(str(v))

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select file", self._le.text(), self._filter)
        if path:
            self._le.setText(path)


class _RangeWidget(QWidget):
    """Two spinboxes (lo, hi) for 'int_range' / 'float_range' param types."""

    valueChanged = Signal()
    editingFinished = Signal()

    def __init__(self, spec: dict, parent=None):
        super().__init__(parent)
        t = spec.get("type", "int_range")
        lo_def, hi_def = spec.get("default", [0, 100])
        lo_min = spec.get("min", -2_000_000_000)
        hi_max = spec.get("max", 2_000_000_000)
        step = spec.get("step", 1)

        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)

        if "float" in t:
            dec = int(spec.get("decimals", 4))
            self._lo = QDoubleSpinBox()
            self._lo.setDecimals(dec)
            self._lo.setRange(float(lo_min), float(hi_max))
            self._lo.setSingleStep(float(step))
            self._lo.setValue(float(lo_def))
            self._hi = QDoubleSpinBox()
            self._hi.setDecimals(dec)
            self._hi.setRange(float(lo_min), float(hi_max))
            self._hi.setSingleStep(float(step))
            self._hi.setValue(float(hi_def))
        else:
            self._lo = QSpinBox()
            self._lo.setRange(int(lo_min), int(hi_max))
            self._lo.setSingleStep(int(step))
            self._lo.setValue(int(lo_def))
            self._hi = QSpinBox()
            self._hi.setRange(int(lo_min), int(hi_max))
            self._hi.setSingleStep(int(step))
            self._hi.setValue(int(hi_def))

        self._lo.setMaximumWidth(_SP_MAX_W)
        self._hi.setMaximumWidth(_SP_MAX_W)
        lbl = QLabel("–")
        lbl.setFixedWidth(10)
        h.addWidget(self._lo)
        h.addWidget(lbl)
        h.addWidget(self._hi)

        self._lo.valueChanged.connect(self.valueChanged)
        self._hi.valueChanged.connect(self.valueChanged)
        self._lo.editingFinished.connect(self.editingFinished)
        self._hi.editingFinished.connect(self.editingFinished)

    def value(self):
        return [self._lo.value(), self._hi.value()]

    def setValue(self, v):
        self._lo.blockSignals(True)
        self._hi.blockSignals(True)
        self._lo.setValue(v[0])
        self._hi.setValue(v[1])
        self._lo.blockSignals(False)
        self._hi.blockSignals(False)


# ---------------------------------------------------------------------------
# Widget factory + helpers
# ---------------------------------------------------------------------------

def _build_widget(spec: dict):
    t = spec.get("type", "float")
    if t == "int":
        w = QSpinBox()
        w.setRange(int(spec.get("min", -2_000_000_000)), int(spec.get("max", 2_000_000_000)))
        w.setValue(int(spec.get("default", 0)))
        if "step" in spec:
            w.setSingleStep(int(spec["step"]))
        w.setMaximumWidth(_SP_MAX_W)
        w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    elif t == "float":
        w = QDoubleSpinBox()
        w.setDecimals(int(spec.get("decimals", 4)))
        w.setRange(float(spec.get("min", -1e12)), float(spec.get("max", 1e12)))
        w.setSingleStep(float(spec.get("step", 0.1)))
        w.setValue(float(spec.get("default", 0.0)))
        w.setMaximumWidth(_SP_MAX_W)
        w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    elif t == "bool":
        w = QCheckBox()
        w.setChecked(bool(spec.get("default", False)))
    elif t == "choice":
        w = QComboBox()
        for ch in spec.get("choices", []):
            w.addItem(str(ch), ch)
        idx = w.findData(spec.get("default"))
        if idx >= 0:
            w.setCurrentIndex(idx)
    elif t == "str":
        w = QLineEdit(str(spec.get("default", "")))
        if "placeholder" in spec:
            w.setPlaceholderText(str(spec["placeholder"]))
    elif t == "file":
        w = _FileWidget(
            default=str(spec.get("default", "")),
            filter=str(spec.get("filter", "")),
        )
    elif t in ("int_range", "float_range"):
        w = _RangeWidget(spec)
    else:
        raise ValueError(f"Unknown param type: {t!r}")

    if "tooltip" in spec:
        w.setToolTip(str(spec["tooltip"]))
    return w


def _connect_change(w, spec: dict, slot):
    t = spec.get("type", "float")
    if t in ("int", "float"):
        # Fire only on Enter or focus-out, not on every keystroke
        w.editingFinished.connect(slot)
    elif t == "bool":
        w.toggled.connect(slot)
    elif t == "choice":
        w.currentIndexChanged.connect(slot)
    elif t == "str":
        w.editingFinished.connect(slot)
    elif t == "file":
        w.textChanged.connect(slot)
    elif t in ("int_range", "float_range"):
        w.editingFinished.connect(slot)


def _read_widget(w, spec: dict):
    t = spec.get("type", "float")
    if t == "int":
        return int(w.value())
    if t == "float":
        return float(w.value())
    if t == "bool":
        return bool(w.isChecked())
    if t == "choice":
        return w.currentData()
    if t in ("str", "file"):
        return w.text()
    if t in ("int_range", "float_range"):
        return w.value()
    return None


class ParamDialog(QDialog):
    """
    Floating non-modal dialog built dynamically from a ``param_spec`` list.

    Parameters
    ----------
    param_spec : list[dict]
        Declarative parameter schema from a ``ProcessorPlugin``.
    title : str
        Dialog window title.
    parent : QWidget, optional
    """

    params_changed = Signal()

    def __init__(self, param_spec: list[dict], title: str = "Parameters", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(
            Qt.Dialog
            | Qt.WindowStaysOnTopHint
            | Qt.WindowCloseButtonHint
        )
        self.setModal(False)
        self.param_spec = param_spec

        layout = QVBoxLayout(self)
        self._widgets: dict[str, object] = {}

        if not param_spec:
            layout.addWidget(QLabel("This method has no configurable parameters."))
        else:
            form = QFormLayout()
            form.setVerticalSpacing(8)
            layout.addLayout(form)

            for spec in param_spec:
                key = spec["key"]
                label = spec.get("label", key)
                w = _build_widget(spec)
                _connect_change(w, spec, self.params_changed)
                self._widgets[key] = w
                form.addRow(f"{label}:", w)

        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.hide)
        layout.addWidget(btns)
        self.button_box = btns

        self.adjustSize()

    def get_params(self) -> dict:
        """Read current widget values into a dict."""
        spec_by_key = {s["key"]: s for s in self.param_spec}
        return {key: _read_widget(w, spec_by_key[key]) for key, w in self._widgets.items()}

    def set_params(self, params: dict) -> None:
        """Programmatically update widget values without emitting params_changed."""
        spec_by_key = {s["key"]: s for s in self.param_spec}
        for key, val in params.items():
            if key not in self._widgets:
                continue
            w = self._widgets[key]
            w.blockSignals(True)
            t = spec_by_key[key].get("type", "float")
            if t == "int":
                w.setValue(int(val))
            elif t == "float":
                w.setValue(float(val))
            elif t == "bool":
                w.setChecked(bool(val))
            elif t == "choice":
                idx = w.findData(val)
                if idx >= 0:
                    w.setCurrentIndex(idx)
            elif t in ("str", "file"):
                w.setText(str(val))
            elif t in ("int_range", "float_range"):
                w.setValue(list(val))
            w.blockSignals(False)

    def show_and_raise(self):
        """Show (or bring to front if already visible)."""
        if not self.isVisible():
            self.show()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, event):
        """Hide instead of destroying so params are preserved."""
        event.ignore()
        self.hide()
