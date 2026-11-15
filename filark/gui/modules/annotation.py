# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""Annotation side panel and in-memory store for viewer overlays."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QDoubleSpinBox,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from filark.gui.widgets.typography import PanelTitle
from filark.gui.axis_metadata import build_axis_metadata
from filark.utils.annotations import decode_annotation_data, describe_source, encode_annotation_data
from filark._exceptions import print_exception


_DEFAULT_LABEL_COLORS = (
    "#ffd54f",
    "#4fc3f7",
    "#ff8a65",
    "#81c784",
    "#ba68c8",
    "#f06292",
    "#aed581",
    "#7986cb",
)


@dataclass
class AnnotationItem:
    """Single viewer annotation and its per-item attributes."""

    uid: int
    label: str
    anno_type: str  # "bbox" | "polyline" | "mask"
    data: dict


@dataclass
class LabelStyle:
    """Per-label display style shared by all items of the same class."""

    color: str
    visible: bool = True

    def to_dict(self) -> dict:
        return {"color": self.color, "visible": bool(self.visible)}


def _coerce_color(value: str | None, fallback: str) -> str:
    color = QColor(str(value or fallback))
    return color.name().upper() if color.isValid() else fallback.upper()


class AnnotationStore:
    """Small in-memory annotation store with per-label style metadata."""

    def __init__(self) -> None:
        self._next_uid = 1
        self._items: Dict[int, AnnotationItem] = {}
        self._order: List[int] = []
        self._labels: set[str] = set()
        self._label_styles: Dict[str, LabelStyle] = {}

    def labels(self) -> List[str]:
        return sorted(self._labels)

    def items_in_order(self) -> List[AnnotationItem]:
        return [self._items[uid] for uid in self._order if uid in self._items]

    def _default_color_for_label(self, label: str) -> str:
        label = (label or "").strip() or "event"
        if label in self._label_styles:
            return self._label_styles[label].color
        index = len(self._label_styles) % len(_DEFAULT_LABEL_COLORS)
        return _DEFAULT_LABEL_COLORS[index]

    def ensure_label_style(self, label: str) -> LabelStyle:
        label = (label or "").strip() or "event"
        style = self._label_styles.get(label)
        if style is None:
            style = LabelStyle(color=self._default_color_for_label(label), visible=True)
            self._label_styles[label] = style
        return style

    def label_style(self, label: str) -> dict:
        return self.ensure_label_style(label).to_dict()

    def label_styles(self) -> dict[str, dict]:
        return {label: style.to_dict() for label, style in sorted(self._label_styles.items())}

    def set_label_style(self, label: str, *, color: str | None = None, visible: bool | None = None) -> dict:
        style = self.ensure_label_style(label)
        if color is not None:
            style.color = _coerce_color(color, style.color)
        if visible is not None:
            style.visible = bool(visible)
        return style.to_dict()

    def _rebuild_labels(self) -> None:
        self._labels = {
            str(item.label).strip()
            for item in self._items.values()
            if str(item.label).strip()
        }

    def add(self, label: str, anno_type: str, data: dict) -> AnnotationItem:
        label = (label or "event").strip() or "event"
        uid = self._next_uid
        self._next_uid += 1
        payload = dict(data)
        payload.setdefault("visible", True)
        item = AnnotationItem(uid=uid, label=label, anno_type=anno_type, data=payload)
        self._items[uid] = item
        self._order.append(uid)
        self._labels.add(label)
        self.ensure_label_style(label)
        return item

    def get(self, uid: int) -> Optional[AnnotationItem]:
        return self._items.get(int(uid))

    def update(self, uid: int, patch: dict) -> Optional[AnnotationItem]:
        item = self._items.get(int(uid))
        if item is None:
            return None
        item.data.update(dict(patch))
        item.data.setdefault("visible", True)
        return item

    def set_label(self, uid: int, label: str) -> Optional[AnnotationItem]:
        item = self._items.get(int(uid))
        if item is None:
            return None
        item.label = (label or "event").strip() or "event"
        self._rebuild_labels()
        self.ensure_label_style(item.label)
        return item

    def rename_label(self, old_label: str, new_label: str) -> list[AnnotationItem]:
        old_label = (old_label or "event").strip() or "event"
        new_label = (new_label or "event").strip() or "event"
        if old_label == new_label:
            return []

        renamed: list[AnnotationItem] = []
        for uid in self._order:
            item = self._items.get(uid)
            if item is None or item.label != old_label:
                continue
            item.label = new_label
            renamed.append(item)

        old_style = self._label_styles.pop(old_label, None)
        if new_label not in self._label_styles:
            if old_style is not None:
                self._label_styles[new_label] = old_style
            elif renamed:
                self.ensure_label_style(new_label)

        self._rebuild_labels()
        return renamed

    def set_item_visible(self, uid: int, visible: bool) -> Optional[AnnotationItem]:
        item = self._items.get(int(uid))
        if item is None:
            return None
        item.data["visible"] = bool(visible)
        return item

    def delete(self, uid: int) -> Optional[AnnotationItem]:
        uid = int(uid)
        item = self._items.pop(uid, None)
        if item is None:
            return None
        try:
            self._order.remove(uid)
        except ValueError:
            print_exception(f"AnnotationStore order list is missing uid={uid} during delete().", prefix="Annotation")
        self._rebuild_labels()
        return item

    def clear(self) -> None:
        self._items.clear()
        self._order.clear()
        self._labels.clear()
        self._label_styles.clear()

    def dump(self) -> dict:
        """Serialize annotations to a JSON-friendly dictionary."""
        return {
            "version": 1,
            "next_uid": int(self._next_uid),
            "labels": sorted(self._labels),
            "label_styles": self.label_styles(),
            "items": [
                {
                    "uid": int(item.uid),
                    "label": str(item.label),
                    "anno_type": str(item.anno_type),
                    "data": encode_annotation_data(dict(item.data)),
                }
                for item in self.items_in_order()
            ],
        }


class BBoxEditWidget(QWidget):
    """Numeric editor for the currently selected bounding box."""

    changed = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QFormLayout(self)
        self.lb_x0 = QLabel("t0:")
        self.lb_y0 = QLabel("c0:")
        self.lb_x1 = QLabel("t1:")
        self.lb_y1 = QLabel("c1:")
        self.sp_t0 = QSpinBox()
        self.sp_c0 = QSpinBox()
        self.sp_t1 = QSpinBox()
        self.sp_c1 = QSpinBox()

        for spin in (self.sp_t0, self.sp_c0, self.sp_t1, self.sp_c1):
            spin.setRange(-2_000_000_000, 2_000_000_000)
            spin.valueChanged.connect(self._on_change)

        layout.addRow(self.lb_x0, self.sp_t0)
        layout.addRow(self.lb_y0, self.sp_c0)
        layout.addRow(self.lb_x1, self.sp_t1)
        layout.addRow(self.lb_y1, self.sp_c1)

    def set_axis_metadata(self, axis_metadata) -> None:
        x = getattr(axis_metadata, "x", None)
        y = getattr(axis_metadata, "y", None)
        x_label = getattr(x, "label", "t")
        y_label = getattr(y, "label", "c")
        self.lb_x0.setText(f"{x_label} 0:")
        self.lb_x1.setText(f"{x_label} 1:")
        self.lb_y0.setText(f"{y_label} 0:")
        self.lb_y1.setText(f"{y_label} 1:")

    @staticmethod
    def _normalize_bbox(t0: int, c0: int, t1: int, c1: int) -> tuple[int, int, int, int]:
        if t1 < t0:
            t0, t1 = t1, t0
        if c1 < c0:
            c0, c1 = c1, c0
        return t0, c0, t1, c1

    def _on_change(self) -> None:
        t0, c0, t1, c1 = self._normalize_bbox(
            int(self.sp_t0.value()),
            int(self.sp_c0.value()),
            int(self.sp_t1.value()),
            int(self.sp_c1.value()),
        )
        self.changed.emit({"t0": t0, "c0": c0, "t1": t1, "c1": c1})

    def set_data(self, data: dict) -> None:
        t0, c0, t1, c1 = self._normalize_bbox(
            int(round(float(data.get("t0", 0)))),
            int(round(float(data.get("c0", 0)))),
            int(round(float(data.get("t1", 0)))),
            int(round(float(data.get("c1", 0)))),
        )
        for spin in (self.sp_t0, self.sp_c0, self.sp_t1, self.sp_c1):
            spin.blockSignals(True)
        self.sp_t0.setValue(t0)
        self.sp_c0.setValue(c0)
        self.sp_t1.setValue(t1)
        self.sp_c1.setValue(c1)
        for spin in (self.sp_t0, self.sp_c0, self.sp_t1, self.sp_c1):
            spin.blockSignals(False)


class PolylineInfoWidget(QWidget):
    """Lightweight editor for selected polyline metadata."""

    continue_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.lb_info = QLabel("Points: 0")
        self.lb_hint = QLabel(
            "Drag a vertex to move it. Shift-click a segment to insert a point. "
            "Alt/Option-click a vertex to delete it."
        )
        self.lb_hint.setWordWrap(True)
        btn_row = QHBoxLayout()
        self.btn_continue_start = QPushButton("Continue from start")
        self.btn_continue_end = QPushButton("Continue from end")
        btn_row.addWidget(self.btn_continue_start)
        btn_row.addWidget(self.btn_continue_end)
        layout.addWidget(self.lb_info)
        layout.addWidget(self.lb_hint)
        layout.addLayout(btn_row)
        layout.addStretch(1)

        self.btn_continue_start.clicked.connect(lambda: self.continue_requested.emit("start"))
        self.btn_continue_end.clicked.connect(lambda: self.continue_requested.emit("end"))

    def set_data(self, data: dict) -> None:
        points = data.get("points", []) or []
        self.lb_info.setText(f"Points: {len(points)}")


class MaskInfoWidget(QWidget):
    """Lightweight editor for selected mask metadata."""

    clear_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.lb_info = QLabel("Mask: 0 x 0")
        self.lb_bounds = QLabel("Bounds: -")
        self.lb_hint = QLabel("Ctrl-drag paints. Shift while painting erases. Ctrl+Z / Ctrl+Y undo and redo brush strokes.")
        self.lb_hint.setWordWrap(True)
        self.btn_clear = QPushButton("Clear Mask")
        layout.addWidget(self.lb_info)
        layout.addWidget(self.lb_bounds)
        layout.addWidget(self.lb_hint)
        layout.addWidget(self.btn_clear)
        layout.addStretch(1)
        self.btn_clear.clicked.connect(self.clear_requested.emit)
        self._x_label = "t"
        self._y_label = "c"

    def set_axis_metadata(self, axis_metadata) -> None:
        x = getattr(axis_metadata, "x", None)
        y = getattr(axis_metadata, "y", None)
        self._x_label = str(getattr(x, "label", "t"))
        self._y_label = str(getattr(y, "label", "c"))

    def set_data(self, data: dict) -> None:
        mask = data.get("mask")
        arr = np.asarray(mask) if mask is not None else np.zeros((0, 0), dtype=np.uint8)
        nt, nc = arr.shape if arr.ndim == 2 else (0, 0)
        nnz = int(np.count_nonzero(arr)) if arr.ndim == 2 else 0
        self.lb_info.setText(f"Mask: {int(nt)} x {int(nc)}, pixels={nnz}")
        self.lb_bounds.setText(
            "Bounds: "
            f"{self._x_label}=[{int(round(float(data.get('t0', 0))))}, {int(round(float(data.get('t1', 0))))}), "
            f"{self._y_label}=[{int(round(float(data.get('c0', 0))))}, {int(round(float(data.get('c1', 0))))})"
        )


class AnnotationAttributesWidget(QWidget):
    """Editor for per-item attributes that do not affect geometry."""

    changed = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._syncing = False

        layout = QFormLayout(self)
        self.sp_confidence = QDoubleSpinBox()
        self.sp_confidence.setRange(-1.0, 1.0)
        self.sp_confidence.setSingleStep(0.05)
        self.sp_confidence.setDecimals(3)
        self.sp_confidence.setSpecialValueText("Not set")

        self.chk_reviewed = QCheckBox("Reviewed")
        self.edit_note = QPlainTextEdit()
        self.edit_note.setMaximumHeight(100)

        layout.addRow("Confidence:", self.sp_confidence)
        layout.addRow("", self.chk_reviewed)
        layout.addRow("Note:", self.edit_note)

        self.sp_confidence.valueChanged.connect(self._emit_changed)
        self.chk_reviewed.toggled.connect(self._emit_changed)
        self.edit_note.textChanged.connect(self._emit_changed)

    def setEnabled(self, enabled: bool) -> None:  # type: ignore[override]
        super().setEnabled(enabled)
        self.sp_confidence.setEnabled(enabled)
        self.chk_reviewed.setEnabled(enabled)
        self.edit_note.setEnabled(enabled)

    def set_data(self, data: dict | None) -> None:
        self._syncing = True
        try:
            if data is None:
                self.sp_confidence.setValue(-1.0)
                self.chk_reviewed.setChecked(False)
                self.edit_note.clear()
                return
            confidence = data.get("confidence", None)
            self.sp_confidence.setValue(-1.0 if confidence is None else float(confidence))
            self.chk_reviewed.setChecked(bool(data.get("reviewed", False)))
            self.edit_note.setPlainText(str(data.get("note", "") or ""))
        finally:
            self._syncing = False

    def _emit_changed(self) -> None:
        if self._syncing:
            return
        confidence = float(self.sp_confidence.value())
        patch = {
            "confidence": None if confidence < 0 else confidence,
            "reviewed": bool(self.chk_reviewed.isChecked()),
            "note": self.edit_note.toPlainText().strip(),
        }
        self.changed.emit(patch)


class LabelStyleWidget(QWidget):
    """Per-label color and visibility controls."""

    style_changed = Signal(str, dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._syncing = False
        self._styles: dict[str, dict] = {}

        layout = QFormLayout(self)
        self.cb_label = QComboBox()
        self.btn_color = QPushButton("Color")
        self.btn_color.setMinimumWidth(120)
        self.chk_visible = QCheckBox("Visible")

        layout.addRow("Label:", self.cb_label)
        layout.addRow("Color:", self.btn_color)
        layout.addRow("", self.chk_visible)

        self.cb_label.currentTextChanged.connect(self._refresh_fields)
        self.btn_color.clicked.connect(self._pick_color)
        self.chk_visible.toggled.connect(self._emit_style_change)
        self._refresh_fields()

    def set_state(self, styles: dict[str, dict], *, selected_label: str | None = None) -> None:
        current = selected_label or self.cb_label.currentText().strip()
        self._styles = {str(label): dict(style) for label, style in styles.items()}
        labels = sorted(self._styles)

        self._syncing = True
        try:
            self.cb_label.blockSignals(True)
            self.cb_label.clear()
            self.cb_label.addItems(labels)
            self.cb_label.blockSignals(False)
            if current and current in labels:
                self.cb_label.setCurrentText(current)
            elif labels:
                self.cb_label.setCurrentIndex(0)
        finally:
            self._syncing = False
        self._refresh_fields()

    def current_label(self) -> str:
        return (self.cb_label.currentText() or "").strip()

    def _refresh_fields(self) -> None:
        label = self.current_label()
        style = self._styles.get(label)
        enabled = style is not None and bool(label)
        self._syncing = True
        try:
            self.btn_color.setEnabled(enabled)
            self.chk_visible.setEnabled(enabled)
            if not enabled:
                self.btn_color.setText("Color")
                self.btn_color.setStyleSheet("")
                self.chk_visible.setChecked(False)
                return
            color = _coerce_color(style.get("color"), "#FFD54F")
            self.btn_color.setText(color)
            self.btn_color.setStyleSheet(
                f"QPushButton {{ background-color: {color}; color: #111111; font-weight: 600; }}"
            )
            self.chk_visible.setChecked(bool(style.get("visible", True)))
        finally:
            self._syncing = False

    def _pick_color(self) -> None:
        label = self.current_label()
        if not label:
            return
        current = QColor(self._styles.get(label, {}).get("color", "#FFD54F"))
        color = QColorDialog.getColor(current, self, f"Select color for {label}")
        if not color.isValid():
            return
        hex_color = color.name().upper()
        self._styles.setdefault(label, {})["color"] = hex_color
        self._refresh_fields()
        self.style_changed.emit(label, {"color": hex_color})

    def _emit_style_change(self) -> None:
        if self._syncing:
            return
        label = self.current_label()
        if not label:
            return
        visible = bool(self.chk_visible.isChecked())
        self._styles.setdefault(label, {})["visible"] = visible
        self.style_changed.emit(label, {"visible": visible})


class LabelDialog(QDialog):
    """Simple confirm dialog for label selection."""

    def __init__(self, labels: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Label")
        self.setMinimumWidth(300)
        self.setModal(True)

        self.labels = labels

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        layout.addWidget(QLabel("Label (Type to filter):"))
        self.edit_label = QLineEdit()
        self.edit_label.setPlaceholderText("Enter or select a label...")
        layout.addWidget(self.edit_label)

        self.list_widget = QListWidget()
        self.list_widget.addItems(self.labels)
        self.list_widget.setAlternatingRowColors(True)
        layout.addWidget(self.list_widget)

        layout.addWidget(QLabel("Notes (Optional):"))
        self.edit_note = QPlainTextEdit()
        self.edit_note.setMaximumHeight(80)
        layout.addWidget(self.edit_note)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        buttons = QHBoxLayout()
        self.btn_ok = QPushButton("OK (Enter)")
        self.btn_ok.setDefault(True)
        self.btn_ok.setAutoDefault(True)
        self.btn_cancel = QPushButton("Cancel (Esc)")
        buttons.addStretch()
        buttons.addWidget(self.btn_cancel)
        buttons.addWidget(self.btn_ok)
        layout.addLayout(buttons)

        self.edit_label.setFocus()

        self.edit_label.textChanged.connect(self._filter_list)
        self.list_widget.itemClicked.connect(lambda item: self.edit_label.setText(item.text()))
        self.list_widget.itemDoubleClicked.connect(self.accept)
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

    def _filter_list(self, text: str) -> None:
        needle = (text or "").lower()
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            item.setHidden(needle not in item.text().lower())

    def get_data(self) -> Tuple[str, str]:
        return self.edit_label.text().strip(), self.edit_note.toPlainText().strip()

    def accept(self) -> None:  # type: ignore[override]
        label, note = self.get_data()
        if not label:
            self.edit_label.setPlaceholderText("LABEL CANNOT BE EMPTY!")
            return
        self.result_label = label
        self.result_note = note
        super().accept()


class _ToolDialog(QDialog):
    """Small non-modal tool window used for annotation editing popups."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.Tool)
        self.setWindowTitle(title)
        self.setModal(False)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(12, 12, 12, 12)
        self._layout.setSpacing(10)
        self._sc_close = QShortcut(QKeySequence("Esc"), self)
        self._sc_close.setContext(Qt.WindowShortcut)
        self._sc_close.activated.connect(self.close)

    def add_widget(self, widget: QWidget) -> None:
        self._layout.addWidget(widget)

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() == Qt.Key_Escape:
            event.accept()
            self.close()
            return
        super().keyPressEvent(event)


class AnnotationPanel(QWidget):
    """Qt side panel that owns annotations, styles, and export/import actions."""

    intent_changed = Signal(dict)
    selection_changed = Signal(object, bool)
    overlay_add = Signal(int, str, dict)
    overlay_add_many = Signal(list)
    overlay_update = Signal(int, dict)
    overlay_delete = Signal(int)
    overlay_clear = Signal()
    polyline_continue_requested = Signal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        default_mode: str = "bbox",
        default_label: str = "event",
        confirm_each: bool = False,
        mask_brush_radius: int = 16,
        mask_brush_mode: str = "paint",
        bbox_width: float = 2.0,
        polyline_width: float = 3.0,
    ) -> None:
        super().__init__(parent)

        self.store = AnnotationStore()
        self._uid_to_list_item: Dict[int, QListWidgetItem] = {}
        self._suspended = False
        self._programmatic_select = False
        self._syncing_list_items = False
        self._syncing_label_editor = False
        self._source_context: dict | None = None
        self._axis_metadata = build_axis_metadata()
        self._default_mode = default_mode if default_mode in ("bbox", "polyline", "mask") else "bbox"
        self._default_label = (default_label or "event").strip() or "event"
        self._default_confirm_each = bool(confirm_each)
        self._default_mask_brush_radius = int(max(1, mask_brush_radius))
        self._default_mask_brush_mode = str(mask_brush_mode or "paint")
        self._default_bbox_width = float(max(1.0, bbox_width))
        self._default_polyline_width = float(max(1.0, polyline_width))

        self._settings = QSettings()
        self._settings_key_export_dir = "annotation/export_dir"

        self._build_ui()
        self.set_axis_metadata(self._axis_metadata)
        self._wire_events()
        self._sync_label_controls()
        self._emit_intent()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        root.addWidget(PanelTitle("Annotation"))

        self.actions_group = QGroupBox("Actions")
        actions_layout = QHBoxLayout(self.actions_group)
        self.btn_export = QPushButton("Export")
        self.btn_import = QPushButton("Import")
        self.btn_clear = QPushButton("Clear")
        self.btn_delete = QPushButton("Delete")
        actions_layout.addWidget(self.btn_export)
        actions_layout.addWidget(self.btn_import)
        actions_layout.addWidget(self.btn_clear)
        actions_layout.addWidget(self.btn_delete)
        actions_layout.addStretch(1)
        root.addWidget(self.actions_group)

        self.preset_group = QGroupBox("Drawing Presets")
        preset_layout = QFormLayout(self.preset_group)
        self.cb_mode = QComboBox()
        self.cb_mode.addItem("BBox", "bbox")
        self.cb_mode.addItem("Polyline", "polyline")
        self.cb_mode.addItem("Mask Brush", "mask")
        mode_index = max(0, self.cb_mode.findData(self._default_mode))
        self.cb_mode.setCurrentIndex(mode_index)

        self.cb_label = QComboBox()
        self.cb_label.setEditable(True)
        self.cb_label.setInsertPolicy(QComboBox.NoInsert)
        self.cb_label.setEditText(self._default_label)
        self.ck_confirm_each = QCheckBox("Confirm each")
        self.ck_confirm_each.setChecked(self._default_confirm_each)
        self.sp_mask_radius = QSpinBox()
        self.sp_mask_radius.setRange(1, 2048)
        self.sp_mask_radius.setValue(self._default_mask_brush_radius)
        self.sp_mask_radius.setSingleStep(2)
        self.cb_mask_action = QComboBox()
        self.cb_mask_action.addItem("Paint", "paint")
        self.cb_mask_action.addItem("Erase", "erase")
        action_index = self.cb_mask_action.findData(self._default_mask_brush_mode)
        self.cb_mask_action.setCurrentIndex(max(0, action_index))

        preset_layout.addRow("Mode:", self.cb_mode)
        preset_layout.addRow("Label:", self.cb_label)
        preset_layout.addRow("Brush radius (px):", self.sp_mask_radius)
        preset_layout.addRow("Brush action:", self.cb_mask_action)
        preset_layout.addRow("", self.ck_confirm_each)
        root.addWidget(self.preset_group)

        tools_row = QHBoxLayout()
        self.btn_edit_selected = QPushButton("Edit Selected…")
        self.btn_label_styles = QPushButton("Label Styles…")
        self.btn_edit_selected.setToolTip(
            "Open the selected annotation editor.\n"
            "Use the popup for label, note, confidence, reviewed, and numeric geometry.\n"
            "Use the canvas itself for bbox drag/resize and polyline vertex dragging."
        )
        self.btn_label_styles.setToolTip(
            "Open per-label style controls.\n"
            "This changes overlay color and visibility for the whole label."
        )
        tools_row.addWidget(self.btn_edit_selected)
        tools_row.addWidget(self.btn_label_styles)
        tools_row.addStretch(1)
        root.addLayout(tools_row)

        self.style_group = QGroupBox("Label Styles")
        style_layout = QVBoxLayout(self.style_group)
        self.style_edit = LabelStyleWidget()
        style_layout.addWidget(self.style_edit)

        root.addWidget(QLabel("Annotation List:"))
        self.list_widget = QListWidget()
        self.list_widget.setObjectName("annList")
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        root.addWidget(self.list_widget, stretch=1)

        self._sc_delete_selected = QShortcut(QKeySequence("Delete"), self.list_widget)
        self._sc_delete_selected.setContext(Qt.WidgetShortcut)
        self._sc_backspace_delete = QShortcut(QKeySequence("Backspace"), self.list_widget)
        self._sc_backspace_delete.setContext(Qt.WidgetShortcut)
        self._sc_delete_selected_panel = QShortcut(QKeySequence("Delete"), self)
        self._sc_delete_selected_panel.setContext(Qt.WidgetWithChildrenShortcut)
        self._sc_backspace_delete_panel = QShortcut(QKeySequence("Backspace"), self)
        self._sc_backspace_delete_panel.setContext(Qt.WidgetWithChildrenShortcut)

        self.selected_group = QGroupBox("Selected Annotation")
        selected_layout = QFormLayout(self.selected_group)
        self.lb_selected_type = QLabel("No selection")
        self.edit_selected_label = QLineEdit()
        self.edit_selected_label.setPlaceholderText("Select an annotation to edit its label")
        self.edit_selected_label.setEnabled(False)
        self.btn_rename_selected_label = QPushButton("Rename This Label Everywhere…")
        self.btn_rename_selected_label.setEnabled(False)
        self.btn_rename_selected_label.setToolTip(
            "Rename the selected label for all annotations that currently use it."
        )
        selected_layout.addRow("Type:", self.lb_selected_type)
        selected_layout.addRow("Label:", self.edit_selected_label)
        selected_layout.addRow("", self.btn_rename_selected_label)

        self.attr_group = QGroupBox("Attributes")
        attr_layout = QVBoxLayout(self.attr_group)
        self.attr_edit = AnnotationAttributesWidget()
        self.attr_edit.setEnabled(False)
        attr_layout.addWidget(self.attr_edit)

        self.editor_stack = QStackedWidget()
        self.editor_stack.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.empty_edit = QLabel("Select an item to edit")
        self.empty_edit.setAlignment(Qt.AlignCenter)
        self.bbox_edit = BBoxEditWidget()
        self.poly_info = PolylineInfoWidget()
        self.mask_info = MaskInfoWidget()
        self.editor_stack.addWidget(self.empty_edit)
        self.editor_stack.addWidget(self.bbox_edit)
        self.editor_stack.addWidget(self.poly_info)
        self.editor_stack.addWidget(self.mask_info)

        self.geometry_group = QGroupBox("Geometry")
        geometry_layout = QVBoxLayout(self.geometry_group)
        geometry_layout.addWidget(self.editor_stack)

        self._style_dialog = _ToolDialog("Annotation Label Styles", parent=self)
        self._style_dialog.add_widget(self.style_group)
        self._style_dialog.resize(320, 180)

        self._inspector_dialog = _ToolDialog("Edit Annotation", parent=self)
        self._inspector_dialog.add_widget(self.selected_group)
        self._inspector_dialog.add_widget(self.attr_group)
        self._inspector_dialog.add_widget(self.geometry_group)
        self._inspector_dialog.resize(380, 460)

    def _wire_events(self) -> None:
        self.btn_export.clicked.connect(self._on_export_clicked)
        self.btn_import.clicked.connect(self._on_import_clicked)
        self.btn_clear.clicked.connect(self._on_clear_clicked)
        self.btn_delete.clicked.connect(self._on_delete_clicked)
        self.btn_edit_selected.clicked.connect(self._open_selected_editor)
        self.btn_label_styles.clicked.connect(self._open_label_styles)

        self.cb_mode.currentIndexChanged.connect(self._emit_intent)
        self.cb_label.currentTextChanged.connect(self._emit_intent)
        self.ck_confirm_each.toggled.connect(self._emit_intent)
        self.sp_mask_radius.valueChanged.connect(self._emit_intent)
        self.cb_mask_action.currentIndexChanged.connect(self._emit_intent)

        self.list_widget.itemSelectionChanged.connect(self._on_selection_changed)
        self.list_widget.itemDoubleClicked.connect(lambda _item: self._open_selected_editor())
        self.list_widget.itemChanged.connect(self._on_list_item_changed)
        self.list_widget.customContextMenuRequested.connect(self._on_list_context_menu)
        self._sc_delete_selected.activated.connect(self._on_delete_clicked)
        self._sc_backspace_delete.activated.connect(self._on_delete_clicked)
        self._sc_delete_selected_panel.activated.connect(self._on_delete_clicked)
        self._sc_backspace_delete_panel.activated.connect(self._on_delete_clicked)

        self.bbox_edit.changed.connect(self._on_bbox_patch)
        self.mask_info.clear_requested.connect(self._on_mask_clear_clicked)
        self.edit_selected_label.editingFinished.connect(self._on_selected_label_edited)
        self.btn_rename_selected_label.clicked.connect(self._on_rename_selected_label_clicked)
        self.attr_edit.changed.connect(self._on_selected_attr_patch)
        self.style_edit.style_changed.connect(self._on_label_style_changed)
        self.poly_info.continue_requested.connect(self.polyline_continue_requested.emit)

    def commit_geometry(self, anno_type: str, data: dict) -> None:
        """Commit a new annotation coming from the canvas interaction layer."""
        if self._suspended:
            return

        anno_type = str(anno_type)
        if anno_type not in ("bbox", "polyline", "mask"):
            return

        intent = self.get_intent()
        preset_label = intent["preset_label"]
        confirm_each = intent["confirm_each"]

        modifiers = data.get("modifiers", []) or []
        force_dialog = ("Shift" in modifiers) or ("SHIFT" in modifiers)
        need_dialog = bool(confirm_each) or bool(force_dialog)

        label = preset_label
        note = ""
        if need_dialog:
            dialog = LabelDialog(self.store.labels(), parent=self)
            dialog.edit_label.setText(preset_label)

            if dialog.exec() != QDialog.DialogCode.Accepted:
                return

            label = (getattr(dialog, "result_label", "") or "").strip()
            note = (getattr(dialog, "result_note", "") or "").strip()
            if not label:
                label, note = dialog.get_data()
                label = (label or preset_label).strip() or "event"

        geom = dict(data)
        geom.pop("modifiers", None)
        geom.setdefault("reviewed", False)
        if anno_type == "bbox":
            geom.setdefault("width", float(intent.get("bbox_width", self._default_bbox_width)))
        elif anno_type == "polyline":
            geom.setdefault("width", float(intent.get("polyline_width", self._default_polyline_width)))
        if note:
            geom["note"] = note

        item = self.store.add(label=label, anno_type=anno_type, data=geom)
        self._sync_label_controls(selected_label=item.label)
        self._ui_add_or_update_item(item, select=True)
        self.overlay_add.emit(item.uid, item.anno_type, self._overlay_payload(item))
        self.selection_changed.emit(item.uid, False)

    def clear_silent(self) -> None:
        self._clear_panel_state()
        self.overlay_clear.emit()

    def suspend(self, flag: bool) -> None:
        self._suspended = bool(flag)

    def set_axis_metadata(self, axis_metadata) -> None:
        self._axis_metadata = axis_metadata if axis_metadata is not None else build_axis_metadata()
        self.bbox_edit.set_axis_metadata(self._axis_metadata)
        self.mask_info.set_axis_metadata(self._axis_metadata)
        for item in self.store.items_in_order():
            self._ui_add_or_update_item(item, select=False)
        current = self._current_selected_uid()
        if current is not None:
            self._on_selection_changed()

    def set_source_context(self, source, *, axis_metadata=None) -> None:
        axis = axis_metadata if axis_metadata is not None else self._axis_metadata
        self._source_context = describe_source(source, axis_metadata=axis) if source is not None else None

    def get_intent(self) -> dict:
        mode = self.cb_mode.currentData() or "bbox"
        preset_label = (self.cb_label.currentText() or "").strip() or "event"
        confirm_each = bool(self.ck_confirm_each.isChecked())
        return {
            "mode": mode,
            "preset_label": preset_label,
            "confirm_each": confirm_each,
            "mask_brush_radius": int(self.sp_mask_radius.value()),
            "mask_brush_mode": str(self.cb_mask_action.currentData() or "paint"),
            "bbox_width": float(self._default_bbox_width),
            "polyline_width": float(self._default_polyline_width),
        }

    def set_labels(self, labels: List[str], *, keep_current: bool = True) -> None:
        current = (self.cb_label.currentText() or "").strip() or self._default_label
        self.cb_label.blockSignals(True)
        self.cb_label.clear()
        self.cb_label.addItems(sorted({str(label).strip() for label in labels if str(label).strip()}))
        if not self.cb_label.count():
            self.cb_label.setEditText(current)
        self.cb_label.blockSignals(False)
        if keep_current and current:
            self.cb_label.setCurrentText(current)

    def apply_geometry_patch(self, uid: int, patch: dict) -> None:
        item = self.store.update(uid, patch)
        if item is None:
            return
        self._ui_add_or_update_item(item, select=False)
        if uid == self._current_selected_uid():
            self._sync_editor_for_item(item)
        self.overlay_update.emit(uid, self._overlay_payload(item))

    def overlay_payload_for_item(self, item: AnnotationItem) -> dict:
        return self._overlay_payload(item)

    def _overlay_payload(self, item: AnnotationItem) -> dict:
        payload = dict(item.data)
        style = self.store.label_style(item.label)
        payload["label"] = item.label
        payload["color"] = style["color"]
        payload["visible"] = bool(style["visible"]) and bool(item.data.get("visible", True))
        return payload

    @staticmethod
    def _item_visible(item: AnnotationItem) -> bool:
        return bool(item.data.get("visible", True))

    def _effective_item_visible(self, item: AnnotationItem) -> bool:
        return bool(self.store.label_style(item.label).get("visible", True)) and self._item_visible(item)

    def _sync_label_controls(self, *, selected_label: str | None = None) -> None:
        self.set_labels(self.store.labels(), keep_current=True)
        self.style_edit.set_state(self.store.label_styles(), selected_label=selected_label)
        self._sync_section_visibility()

    def _sync_section_visibility(self, *, selected_item: AnnotationItem | None = None) -> None:
        if selected_item is None:
            uid = self._current_selected_uid()
            selected_item = None if uid is None else self.store.get(uid)

        has_items = bool(self.store.items_in_order())
        has_labels = bool(self.store.labels())
        has_selection = selected_item is not None

        self.btn_export.setEnabled(has_items)
        self.btn_clear.setEnabled(has_items)
        self.btn_delete.setEnabled(has_selection)
        self.btn_edit_selected.setEnabled(has_selection)
        self.btn_label_styles.setEnabled(has_labels)
        self.btn_rename_selected_label.setEnabled(has_selection)

        if not has_labels:
            self._style_dialog.hide()
        if not has_selection:
            self._inspector_dialog.hide()

    def _open_selected_editor(self) -> None:
        if self._current_selected_uid() is None:
            return
        self._inspector_dialog.adjustSize()
        self._inspector_dialog.show()
        self._inspector_dialog.raise_()
        self._inspector_dialog.activateWindow()

    def _open_label_styles(self) -> None:
        if not self.store.labels():
            return
        self._style_dialog.adjustSize()
        self._style_dialog.show()
        self._style_dialog.raise_()
        self._style_dialog.activateWindow()

    def _suggest_export_path(self) -> str:
        base_dir = str(self._settings.value(self._settings_key_export_dir, "") or "")
        if base_dir and os.path.isdir(base_dir):
            return os.path.join(base_dir, "annotations.json")
        return os.path.join(os.path.expanduser("~"), "annotations.json")

    def _on_export_clicked(self) -> None:
        if self.list_widget.count() <= 0:
            QMessageBox.information(self, "Export", "No annotations to export.")
            return

        default_path = self._suggest_export_path()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Annotations",
            default_path,
            "JSON (*.json);;All Files (*)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path = path + ".json"

        payload = self.store.dump()
        payload["intent"] = self.get_intent()
        if self._source_context is not None:
            payload["source"] = dict(self._source_context)

        try:
            with open(path, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", f"Failed to write file:\n{path}\n\n{exc}")
            return

        self._settings.setValue(self._settings_key_export_dir, os.path.dirname(path))
        QMessageBox.information(self, "Export", f"Exported to:\n{path}")

    def _on_import_clicked(self) -> None:
        base_dir = str(self._settings.value(self._settings_key_export_dir, "") or "")
        start_dir = base_dir if os.path.isdir(base_dir) else os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Annotations",
            start_dir,
            "JSON (*.json);;All Files (*)",
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except Exception as exc:
            QMessageBox.critical(self, "Import Failed", f"Failed to read file:\n{path}\n\n{exc}")
            return

        if not isinstance(payload, dict) or payload.get("version") != 1:
            QMessageBox.critical(self, "Import Failed", "Invalid annotation file (expected version=1).")
            return

        raw_items = payload.get("items", [])
        if not isinstance(raw_items, list):
            QMessageBox.critical(self, "Import Failed", "Invalid file: 'items' is not a list.")
            return

        label_styles = payload.get("label_styles", {})
        if not isinstance(label_styles, dict):
            label_styles = {}

        valid_items: list[tuple[str, str, dict]] = []
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            label = str(raw.get("label", "") or "event")
            anno_type = str(raw.get("anno_type", "bbox"))
            data = raw.get("data", {})
            if not isinstance(data, dict):
                data = {}
            if anno_type not in ("bbox", "polyline", "mask"):
                continue
            try:
                data = decode_annotation_data(data)
            except Exception:
                print_exception("Failed to decode an annotation item during import.", prefix="Annotation")
                continue
            valid_items.append((label, anno_type, data))

        if not valid_items:
            QMessageBox.information(self, "Import", "No valid annotations found in file.")
            return

        do_replace = True
        if self.list_widget.count() > 0:
            msg = QMessageBox(self)
            msg.setWindowTitle("Import Annotations")
            msg.setText(
                f"Found {len(valid_items)} annotation(s) in file.\n"
                f"You have {self.list_widget.count()} existing annotation(s)."
            )
            msg.setInformativeText("Replace all existing annotations, or merge (add) them?")
            btn_replace = msg.addButton("Replace", QMessageBox.DestructiveRole)
            btn_merge = msg.addButton("Merge", QMessageBox.AcceptRole)
            btn_cancel = msg.addButton("Cancel", QMessageBox.RejectRole)
            msg.setDefaultButton(btn_cancel)
            msg.exec()
            clicked = msg.clickedButton()
            if clicked is btn_cancel or clicked is None:
                return
            do_replace = clicked is btn_replace

        if do_replace:
            self._clear_panel_state()
            self.overlay_clear.emit()

        for label, style in label_styles.items():
            if not isinstance(style, dict):
                continue
            self.store.set_label_style(
                label,
                color=style.get("color"),
                visible=style.get("visible"),
            )

        overlay_records: list[tuple[int, str, dict]] = []
        self.list_widget.setUpdatesEnabled(False)
        try:
            for label, anno_type, data in valid_items:
                item = self.store.add(label=label, anno_type=anno_type, data=data)
                self._ui_add_or_update_item(item, select=False)
                overlay_records.append((item.uid, item.anno_type, self._overlay_payload(item)))
        finally:
            self.list_widget.setUpdatesEnabled(True)
        self.overlay_add_many.emit(overlay_records)

        self._sync_label_controls()
        self._settings.setValue(self._settings_key_export_dir, os.path.dirname(path))
        QMessageBox.information(self, "Import", f"Imported {len(valid_items)} annotation(s) from:\n{path}")

    def _emit_intent(self, *args) -> None:
        self.intent_changed.emit(self.get_intent())

    def _current_selected_uid(self) -> Optional[int]:
        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            return None
        return int(selected_items[0].data(Qt.UserRole))

    def select_uid(self, uid: Optional[int]) -> None:
        self._programmatic_select = True
        try:
            if uid is None:
                self.list_widget.clearSelection()
                return
            list_item = self._uid_to_list_item.get(int(uid))
            if list_item is None:
                return
            self.list_widget.setCurrentItem(list_item)
        finally:
            self._programmatic_select = False

    def _ui_add_or_update_item(self, item: AnnotationItem, *, select: bool) -> None:
        text = self._format_item_text(item)
        list_item = self._uid_to_list_item.get(item.uid)
        self._syncing_list_items = True
        try:
            if list_item is None:
                list_item = QListWidgetItem(text)
                list_item.setData(Qt.UserRole, item.uid)
                list_item.setFlags(list_item.flags() | Qt.ItemIsUserCheckable)
                self._uid_to_list_item[item.uid] = list_item
                self.list_widget.addItem(list_item)
            else:
                list_item.setText(text)
            list_item.setCheckState(Qt.Checked if self._item_visible(item) else Qt.Unchecked)
            self._apply_list_item_style(item, list_item)
        finally:
            self._syncing_list_items = False
        if select:
            self._programmatic_select = True
            self.list_widget.setCurrentItem(list_item)
            self._programmatic_select = False

    def _apply_list_item_style(self, item: AnnotationItem, list_item: QListWidgetItem) -> None:
        style = self.store.label_style(item.label)
        color = QColor(style["color"])
        font = list_item.font()
        effective_visible = self._effective_item_visible(item)
        item_visible = self._item_visible(item)
        if effective_visible:
            list_item.setForeground(color)
            font.setItalic(False)
        else:
            list_item.setForeground(color.darker(220))
            font.setItalic(True)
        list_item.setFont(font)
        if not style["visible"]:
            list_item.setToolTip("Hidden by label style.")
        elif not item_visible:
            list_item.setToolTip("Hidden for this annotation.")
        else:
            list_item.setToolTip("")

    def _ui_remove_item(self, uid: int) -> None:
        list_item = self._uid_to_list_item.pop(uid, None)
        if list_item is None:
            return
        row = self.list_widget.row(list_item)
        self.list_widget.takeItem(row)

    def _set_item_visible(self, uid: int, visible: bool) -> None:
        item = self.store.set_item_visible(uid, visible)
        if item is None:
            return
        self._ui_add_or_update_item(item, select=False)
        if uid == self._current_selected_uid():
            self._sync_editor_for_item(item)
        self.overlay_update.emit(uid, self._overlay_payload(item))

    def _set_label_visibility(self, label: str, visible: bool) -> None:
        if not label:
            return
        self._on_label_style_changed(label, {"visible": visible})

    def _on_list_item_changed(self, list_item: QListWidgetItem) -> None:
        if self._syncing_list_items:
            return
        uid_data = list_item.data(Qt.UserRole)
        if uid_data is None:
            return
        uid = int(uid_data)
        item = self.store.get(uid)
        if item is None:
            return
        visible = list_item.checkState() == Qt.Checked
        if visible == self._item_visible(item):
            return
        self._set_item_visible(uid, visible)

    def _on_list_context_menu(self, pos) -> None:
        list_item = self.list_widget.itemAt(pos)
        if list_item is None:
            return
        uid = int(list_item.data(Qt.UserRole))
        item = self.store.get(uid)
        if item is None:
            return

        self._programmatic_select = True
        self.list_widget.setCurrentItem(list_item)
        self._programmatic_select = False
        self._on_selection_changed()

        item_visible = self._item_visible(item)
        label_visible = bool(self.store.label_style(item.label).get("visible", True))

        menu = QMenu(self)
        act_item_visible = menu.addAction("Hide item" if item_visible else "Show item")
        act_label_visible = menu.addAction("Hide label" if label_visible else "Show label")
        act_rename_label = menu.addAction("Rename label…")
        menu.addSeparator()
        act_edit = menu.addAction("Edit…")
        act_delete = menu.addAction("Delete")

        action = menu.exec(self.list_widget.mapToGlobal(pos))
        if action is None:
            return
        if action is act_item_visible:
            self._set_item_visible(uid, not item_visible)
            return
        if action is act_label_visible:
            self._set_label_visibility(item.label, not label_visible)
            return
        if action is act_rename_label:
            self._rename_label_interactive(item.label)
            return
        if action is act_edit:
            self._open_selected_editor()
            return
        if action is act_delete:
            self._on_delete_clicked()

    def _clear_panel_state(self) -> None:
        self.store.clear()
        self._uid_to_list_item.clear()
        self.list_widget.clear()
        self._style_dialog.hide()
        self._inspector_dialog.hide()
        self._set_selected_label_editor(None)
        self.attr_edit.set_data(None)
        self.attr_edit.setEnabled(False)
        self.lb_selected_type.setText("No selection")
        self.editor_stack.setCurrentIndex(0)
        self._sync_label_controls()
        self._sync_section_visibility(selected_item=None)
        self.selection_changed.emit(None, False)

    def _on_selection_changed(self) -> None:
        uid = self._current_selected_uid()
        if uid is None:
            self._set_selected_label_editor(None)
            self.attr_edit.set_data(None)
            self.attr_edit.setEnabled(False)
            self.lb_selected_type.setText("No selection")
            self.editor_stack.setCurrentIndex(0)
            self._sync_section_visibility(selected_item=None)
            if not self._programmatic_select:
                self.selection_changed.emit(None, False)
            return

        item = self.store.get(uid)
        if item is None:
            self._set_selected_label_editor(None)
            self.attr_edit.set_data(None)
            self.attr_edit.setEnabled(False)
            self.lb_selected_type.setText("No selection")
            self.editor_stack.setCurrentIndex(0)
            self._sync_section_visibility(selected_item=None)
            if not self._programmatic_select:
                self.selection_changed.emit(None, False)
            return

        self._set_selected_label_editor(item)
        self.attr_edit.setEnabled(True)
        self.attr_edit.set_data(item.data)
        self.lb_selected_type.setText(item.anno_type)
        self._sync_editor_for_item(item)
        self.style_edit.set_state(self.store.label_styles(), selected_label=item.label)
        self._sync_section_visibility(selected_item=item)
        if not self._programmatic_select:
            self.selection_changed.emit(uid, True)

    def _sync_editor_for_item(self, item: AnnotationItem) -> None:
        if item.anno_type == "bbox":
            self.editor_stack.setCurrentIndex(1)
            self.bbox_edit.set_data(item.data)
        elif item.anno_type == "polyline":
            self.editor_stack.setCurrentIndex(2)
            self.poly_info.set_data(item.data)
        elif item.anno_type == "mask":
            self.editor_stack.setCurrentIndex(3)
            self.mask_info.set_data(item.data)
        else:
            self.editor_stack.setCurrentIndex(0)

    def _on_bbox_patch(self, patch: dict) -> None:
        uid = self._current_selected_uid()
        if uid is None:
            return
        item = self.store.update(uid, patch)
        if item is None:
            return
        self._ui_add_or_update_item(item, select=False)
        self.overlay_update.emit(uid, self._overlay_payload(item))

    def _on_mask_clear_clicked(self) -> None:
        uid = self._current_selected_uid()
        if uid is None:
            return
        item = self.store.get(uid)
        if item is None or item.anno_type != "mask":
            return
        mask = np.asarray(item.data.get("mask"), dtype=np.uint8)
        if mask.ndim != 2:
            return
        patch = {"mask": np.zeros_like(mask)}
        item = self.store.update(uid, patch)
        if item is None:
            return
        self._ui_add_or_update_item(item, select=False)
        self._sync_editor_for_item(item)
        self.overlay_update.emit(uid, self._overlay_payload(item))

    def _on_selected_attr_patch(self, patch: dict) -> None:
        uid = self._current_selected_uid()
        if uid is None:
            return
        item = self.store.update(uid, patch)
        if item is None:
            return
        self._ui_add_or_update_item(item, select=False)

    def _set_selected_label_editor(self, item: Optional[AnnotationItem]) -> None:
        self._syncing_label_editor = True
        try:
            if item is None:
                self.edit_selected_label.clear()
                self.edit_selected_label.setEnabled(False)
            else:
                self.edit_selected_label.setEnabled(True)
                self.edit_selected_label.setText(str(item.label))
        finally:
            self._syncing_label_editor = False

    def _on_selected_label_edited(self) -> None:
        if self._syncing_label_editor:
            return
        uid = self._current_selected_uid()
        if uid is None:
            self._set_selected_label_editor(None)
            return

        new_label = (self.edit_selected_label.text() or "").strip()
        item = self.store.get(uid)
        if item is None:
            self._set_selected_label_editor(None)
            return
        if not new_label:
            self._set_selected_label_editor(item)
            return
        if new_label == item.label:
            return

        item = self.store.set_label(uid, new_label)
        if item is None:
            return

        self._sync_label_controls(selected_label=item.label)
        self._ui_add_or_update_item(item, select=False)
        self.overlay_update.emit(uid, self._overlay_payload(item))

    def _on_rename_selected_label_clicked(self) -> None:
        uid = self._current_selected_uid()
        if uid is None:
            return
        item = self.store.get(uid)
        if item is None:
            return
        self._rename_label_interactive(item.label)

    def _rename_label_interactive(self, old_label: str) -> None:
        old_label = (old_label or "").strip()
        if not old_label:
            return

        new_label, accepted = QInputDialog.getText(
            self,
            "Rename Label",
            "New label name:",
            QLineEdit.Normal,
            old_label,
        )
        if not accepted:
            return

        new_label = (new_label or "").strip()
        if not new_label:
            QMessageBox.information(self, "Rename Label", "Label name cannot be empty.")
            return

        self._rename_label_everywhere(old_label, new_label)

    def _rename_label_everywhere(self, old_label: str, new_label: str) -> None:
        old_label = (old_label or "event").strip() or "event"
        new_label = (new_label or "event").strip() or "event"
        if old_label == new_label:
            return

        preset_label = (self.cb_label.currentText() or "").strip() or "event"
        renamed_items = self.store.rename_label(old_label, new_label)
        if not renamed_items:
            return

        self._sync_label_controls(selected_label=new_label)
        if preset_label == old_label:
            self.cb_label.setCurrentText(new_label)

        overlay_records: list[tuple[int, str, dict]] = []
        for item in renamed_items:
            self._ui_add_or_update_item(item, select=False)
            overlay_records.append((item.uid, item.anno_type, self._overlay_payload(item)))
        self.overlay_add_many.emit(overlay_records)

        selected_uid = self._current_selected_uid()
        if selected_uid is None:
            self._sync_section_visibility(selected_item=None)
            return

        selected_item = self.store.get(selected_uid)
        if selected_item is None:
            self._sync_section_visibility(selected_item=None)
            return

        self._set_selected_label_editor(selected_item)
        self.style_edit.set_state(self.store.label_styles(), selected_label=selected_item.label)
        self._sync_section_visibility(selected_item=selected_item)

    def _on_label_style_changed(self, label: str, patch: dict) -> None:
        style = self.store.set_label_style(label, color=patch.get("color"), visible=patch.get("visible"))
        self.style_edit.set_state(self.store.label_styles(), selected_label=label)
        overlay_records: list[tuple[int, str, dict]] = []
        for item in self.store.items_in_order():
            if item.label != label:
                continue
            list_item = self._uid_to_list_item.get(item.uid)
            if list_item is not None:
                self._apply_list_item_style(item, list_item)
            overlay_records.append((item.uid, item.anno_type, self._overlay_payload(item)))
        self.overlay_add_many.emit(overlay_records)
        _ = style

    def _on_delete_clicked(self) -> None:
        uid = self._current_selected_uid()
        if uid is None:
            return
        item = self.store.delete(uid)
        if item is None:
            return
        self._ui_remove_item(uid)
        self._sync_label_controls()
        self._set_selected_label_editor(None)
        self.attr_edit.set_data(None)
        self.attr_edit.setEnabled(False)
        self.lb_selected_type.setText("No selection")
        self.editor_stack.setCurrentIndex(0)
        self._sync_section_visibility(selected_item=None)
        self.selection_changed.emit(None, False)
        self.overlay_delete.emit(uid)

    def _on_clear_clicked(self) -> None:
        if self.list_widget.count() <= 0:
            return
        code = QMessageBox.question(
            self,
            "Clear All?",
            "This will remove all annotations in the panel and overlay.\nContinue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if code != QMessageBox.Yes:
            return
        self._clear_panel_state()
        self.overlay_clear.emit()

    def _format_item_text(self, item: AnnotationItem) -> str:
        note = item.data.get("note", "") or ""
        note_suffix = f'  "{note[:20]}{"…" if len(note) > 20 else ""}"' if note else ""
        confidence = item.data.get("confidence", None)
        confidence_suffix = f"  p={float(confidence):.2f}" if confidence is not None else ""
        reviewed_suffix = "  [reviewed]" if item.data.get("reviewed", False) else ""
        x_label = getattr(getattr(self._axis_metadata, "x", None), "label", "x")
        y_label = getattr(getattr(self._axis_metadata, "y", None), "label", "y")

        if item.anno_type == "bbox":
            t0 = int(round(float(item.data.get("t0", 0))))
            c0 = int(round(float(item.data.get("c0", 0))))
            t1 = int(round(float(item.data.get("t1", 0))))
            c1 = int(round(float(item.data.get("c1", 0))))
            return (
                f"{item.label} [bbox] {x_label}=[{t0},{t1}) {y_label}=[{c0},{c1})"
                f"{confidence_suffix}{reviewed_suffix}  #{item.uid}{note_suffix}"
            )
        if item.anno_type == "polyline":
            points = item.data.get("points", []) or []
            return (
                f"{item.label} [poly] n={len(points)}"
                f"{confidence_suffix}{reviewed_suffix}  #{item.uid}{note_suffix}"
            )
        if item.anno_type == "mask":
            mask = np.asarray(item.data.get("mask"))
            if mask.ndim == 2:
                nt, nc = int(mask.shape[0]), int(mask.shape[1])
                nnz = int(np.count_nonzero(mask))
            else:
                nt, nc, nnz = 0, 0, 0
            return (
                f"{item.label} [mask] {nt}x{nc} pixels={nnz}"
                f"{confidence_suffix}{reviewed_suffix}  #{item.uid}{note_suffix}"
            )
        return f"{item.label} [{item.anno_type}]  #{item.uid}{note_suffix}"
