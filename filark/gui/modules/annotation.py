# Copyright (c) 2026 Jintao Li. 
# Zhejiang University (ZJU).
# 
# Licensed under the MIT License.


from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

from PySide6.QtCore import Signal, Qt, QSettings
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout, QLabel,
    QPushButton, QComboBox, QSpinBox, QCheckBox, QListWidget,
    QListWidgetItem, QStackedWidget, QFrame, QPlainTextEdit, QDialog, QLineEdit,
    QMessageBox, QFileDialog,
)

from filark.gui.widgets.typography import PanelTitle


# -------------------------
# Data structures
# -------------------------
@dataclass
class AnnotationItem:
    uid: int
    label: str
    anno_type: str  # "bbox" | "polyline" | "brush"
    data: dict      # bbox: {t0,c0,t1,c1}  polyline: {points: [...]} ...


class AnnotationStore:
    """Very small in-memory store. Panel owns it."""
    def __init__(self):
        self._next_uid = 1
        self._items: Dict[int, AnnotationItem] = {}
        self._order: List[int] = []
        self._labels: set[str] = set()

    def labels(self) -> List[str]:
        return sorted(self._labels)

    def items_in_order(self) -> List[AnnotationItem]:
        return [self._items[uid] for uid in self._order if uid in self._items]

    def add(self, label: str, anno_type: str, data: dict) -> AnnotationItem:
        uid = self._next_uid
        self._next_uid += 1
        it = AnnotationItem(uid=uid, label=label, anno_type=anno_type, data=dict(data))
        self._items[uid] = it
        self._order.append(uid)
        if label:
            self._labels.add(label)
        return it

    def get(self, uid: int) -> Optional[AnnotationItem]:
        return self._items.get(int(uid))

    def update(self, uid: int, patch: dict) -> Optional[AnnotationItem]:
        it = self._items.get(int(uid))
        if it is None:
            return None
        it.data.update(dict(patch))
        return it

    def delete(self, uid: int) -> Optional[AnnotationItem]:
        uid = int(uid)
        it = self._items.pop(uid, None)
        if it is None:
            return None
        try:
            self._order.remove(uid)
        except ValueError:
            pass
        return it

    def clear(self):
        self._items.clear()
        self._order.clear()

    # -------- Export helpers --------
    def dump(self) -> dict:
        """Serialize store to a JSON-friendly dict (no numpy types)."""
        return {
            "version": 1,
            "next_uid": int(self._next_uid),
            "labels": sorted(self._labels),
            "items": [
                {
                    "uid": int(it.uid),
                    "label": str(it.label),
                    "anno_type": str(it.anno_type),
                    "data": dict(it.data),
                }
                for it in self.items_in_order()
            ],
        }


# -------------------------
# Sub-widgets
# -------------------------
class BBoxEditWidget(QWidget):
    changed = Signal(dict)  # patch: {"t0":..,"c0":..,"t1":..,"c1":..}

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QFormLayout(self)

        self.sp_t0 = QSpinBox(); self.sp_c0 = QSpinBox()
        self.sp_t1 = QSpinBox(); self.sp_c1 = QSpinBox()

        for sp in (self.sp_t0, self.sp_c0, self.sp_t1, self.sp_c1):
            sp.setRange(-2_000_000_000, 2_000_000_000)
            sp.valueChanged.connect(self._on_change)

        layout.addRow("t0:", self.sp_t0); layout.addRow("c0:", self.sp_c0)
        layout.addRow("t1:", self.sp_t1); layout.addRow("c1:", self.sp_c1)

    @staticmethod
    def _norm(t0: int, c0: int, t1: int, c1: int):
        if t1 < t0: t0, t1 = t1, t0
        if c1 < c0: c0, c1 = c1, c0
        return t0, c0, t1, c1

    def _on_change(self):
        t0, c0, t1, c1 = self._norm(
            int(self.sp_t0.value()), int(self.sp_c0.value()),
            int(self.sp_t1.value()), int(self.sp_c1.value())
        )
        self.changed.emit({"t0": t0, "c0": c0, "t1": t1, "c1": c1})

    def set_data(self, data: dict):
        t0 = int(data.get("t0", 0)); c0 = int(data.get("c0", 0))
        t1 = int(data.get("t1", 0)); c1 = int(data.get("c1", 0))
        t0, c0, t1, c1 = self._norm(t0, c0, t1, c1)

        for sp in (self.sp_t0, self.sp_c0, self.sp_t1, self.sp_c1):
            sp.blockSignals(True)
        self.sp_t0.setValue(t0); self.sp_c0.setValue(c0)
        self.sp_t1.setValue(t1); self.sp_c1.setValue(c1)
        for sp in (self.sp_t0, self.sp_c0, self.sp_t1, self.sp_t1):
            pass  # (留一个占位避免 lint 报 “redefined” 的误判)
        for sp in (self.sp_t0, self.sp_c0, self.sp_t1, self.sp_c1):
            sp.blockSignals(False)


class PolylineInfoWidget(QWidget):
    """Simplified: only show point count (no closed loop, no undo)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.lb_info = QLabel("Points: 0")
        layout.addWidget(self.lb_info)
        layout.addStretch(1)

    def set_data(self, data: dict):
        pts = data.get("points", []) or []
        self.lb_info.setText(f"Points: {len(pts)}")


class LabelDialog(QDialog):
    """Simple confirm dialog for label selection."""
    def __init__(self, labels: list[str], parent=None):
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

        h_btns = QHBoxLayout()
        self.btn_ok = QPushButton("OK (Enter)")
        self.btn_ok.setDefault(True)
        self.btn_ok.setAutoDefault(True)
        self.btn_cancel = QPushButton("Cancel (Esc)")
        h_btns.addStretch()
        h_btns.addWidget(self.btn_cancel)
        h_btns.addWidget(self.btn_ok)
        layout.addLayout(h_btns)

        self.edit_label.setFocus()

        # wire
        self.edit_label.textChanged.connect(self._filter_list)
        self.list_widget.itemClicked.connect(lambda item: self.edit_label.setText(item.text()))
        self.list_widget.itemDoubleClicked.connect(self.accept)
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

    def _filter_list(self, text: str):
        t = (text or "").lower()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setHidden(t not in item.text().lower())

    def get_data(self) -> Tuple[str, str]:
        return self.edit_label.text().strip(), self.edit_note.toPlainText().strip()

    def accept(self):
        label, note = self.get_data()
        if not label:
            self.edit_label.setPlaceholderText("LABEL CANNOT BE EMPTY!")
            return
        self.result_label = label
        self.result_note = note
        super().accept()


# -------------------------
# Main panel
# -------------------------
class AnnotationPanel(QWidget):
    # intent
    intent_changed = Signal(dict)  # {"mode":..., "preset_label":..., "confirm_each": bool}

    # selection (optional)
    selection_changed = Signal(object)  # uid or None

    # overlay commands -> DisplayModel
    overlay_add = Signal(int, str, dict)       # uid, anno_type, data
    overlay_update = Signal(int, dict)         # uid, patch
    overlay_delete = Signal(int)               # uid
    overlay_clear = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.store = AnnotationStore()
        self._uid_to_lwi: Dict[int, QListWidgetItem] = {}
        self._suspended = False  # ROI mode may suspend accepting new annotations

        # Remember export dir
        # 你可以在 app 启动时统一设置 organization/application name，
        # 这里也可独立使用（不影响功能）
        self._settings = QSettings()
        self._settings_key_export_dir = "annotation/export_dir"

        self._build_ui()
        self._wire_events()
        self._emit_intent()

    # -------- UI --------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = PanelTitle("Annotation")
        root.addWidget(title)

        # actions
        g_actions = QGroupBox("Actions")
        h_actions = QHBoxLayout(g_actions)

        self.btn_export = QPushButton("Export")
        self.btn_import = QPushButton("Import")
        self.btn_clear = QPushButton("Clear")
        self.btn_delete = QPushButton("Delete")

        h_actions.addWidget(self.btn_export)
        h_actions.addWidget(self.btn_import)
        h_actions.addWidget(self.btn_clear)
        h_actions.addWidget(self.btn_delete)
        h_actions.addStretch(1)
        root.addWidget(g_actions)

        # presets
        g_preset = QGroupBox("Drawing Presets")
        f_preset = QFormLayout(g_preset)

        self.cb_mode = QComboBox()
        self.cb_mode.addItem("BBox", "bbox")
        self.cb_mode.addItem("Polyline", "polyline")
        self.cb_mode.addItem("Brush", "brush")

        self.cb_label = QComboBox()
        self.cb_label.setEditable(True)
        self.cb_label.setInsertPolicy(QComboBox.NoInsert)

        self.ck_confirm_each = QCheckBox("Confirm each")

        f_preset.addRow("Mode:", self.cb_mode)
        f_preset.addRow("Label:", self.cb_label)
        f_preset.addRow("", self.ck_confirm_each)
        root.addWidget(g_preset)

        # list
        root.addWidget(QLabel("Annotation List:"))
        self.list_widget = QListWidget()
        self.list_widget.setObjectName("annList")
        self.list_widget.setAlternatingRowColors(True)
        root.addWidget(self.list_widget, stretch=1)

        # editor
        self.editor_stack = QStackedWidget()
        self.editor_stack.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)

        self.empty_edit = QLabel("Select an item to edit")
        self.empty_edit.setAlignment(Qt.AlignCenter)

        self.bbox_edit = BBoxEditWidget()
        self.poly_info = PolylineInfoWidget()

        self.editor_stack.addWidget(self.empty_edit)  # 0
        self.editor_stack.addWidget(self.bbox_edit)   # 1
        self.editor_stack.addWidget(self.poly_info)   # 2

        root.addWidget(QLabel("Fine-tune:"))
        root.addWidget(self.editor_stack)

    # -------- Wiring --------
    def _wire_events(self):
        # actions
        self.btn_export.clicked.connect(self._on_export_clicked)
        self.btn_import.clicked.connect(self._on_import_clicked)
        self.btn_clear.clicked.connect(self._on_clear_clicked)
        self.btn_delete.clicked.connect(self._on_delete_clicked)

        # intent
        self.cb_mode.currentIndexChanged.connect(self._emit_intent)
        self.cb_label.currentTextChanged.connect(self._emit_intent)
        self.ck_confirm_each.toggled.connect(self._emit_intent)

        # selection
        self.list_widget.itemSelectionChanged.connect(self._on_selection_changed)

        # editors
        self.bbox_edit.changed.connect(self._on_bbox_patch)

    # -------- Public API --------
    def commit_geometry(self, anno_type: str, data: dict):
        """
        Entry point for DisplayModel -> Panel:
        bbox_finished / polyline_finished should call this.
        data may contain "modifiers" list.
        """
        if self._suspended:
            return

        anno_type = str(anno_type)
        if anno_type not in ("bbox", "polyline", "brush"):
            return

        intent = self.get_intent()
        preset_label = intent["preset_label"]
        confirm_each = intent["confirm_each"]

        mods = data.get("modifiers", []) or []
        force_dialog = ("Shift" in mods) or ("SHIFT" in mods)
        need_dialog = bool(confirm_each) or bool(force_dialog)

        label = preset_label
        if need_dialog:
            dlg = LabelDialog(self.store.labels(), parent=self)
            dlg.edit_label.setText(preset_label)

            code = dlg.exec()
            if code != QDialog.DialogCode.Accepted:
                return

            label = (getattr(dlg, "result_label", "") or "").strip()
            if not label:
                label, _note = dlg.get_data()
                label = (label or preset_label).strip() or "event"

        geom = dict(data)
        geom.pop("modifiers", None)

        it = self.store.add(label=label, anno_type=anno_type, data=geom)

        self.set_labels(self.store.labels(), keep_current=True)
        self._ui_add_or_update_item(it, select=True)

        self.overlay_add.emit(it.uid, it.anno_type, dict(it.data))

    def suspend(self, flag: bool):
        self._suspended = bool(flag)

    # -------- Intent helpers --------
    def get_intent(self) -> dict:
        mode = self.cb_mode.currentData() or "bbox"
        preset_label = (self.cb_label.currentText() or "").strip() or "event"
        confirm_each = bool(self.ck_confirm_each.isChecked())
        return {"mode": mode, "preset_label": preset_label, "confirm_each": confirm_each}

    def set_labels(self, labels: List[str], *, keep_current: bool = True):
        cur = (self.cb_label.currentText() or "").strip()
        self.cb_label.blockSignals(True)
        self.cb_label.clear()
        self.cb_label.addItems(sorted(set([x for x in labels if str(x).strip()])))
        self.cb_label.blockSignals(False)
        if keep_current and cur:
            self.cb_label.setCurrentText(cur)

    # -------- Export / Import --------
    def _suggest_export_path(self) -> str:
        base_dir = str(self._settings.value(self._settings_key_export_dir, "") or "")
        if base_dir and os.path.isdir(base_dir):
            return os.path.join(base_dir, "annotations.json")
        return os.path.join(os.path.expanduser("~"), "annotations.json")

    def _on_export_clicked(self):
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

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to write file:\n{path}\n\n{e}")
            return

        # remember directory
        self._settings.setValue(self._settings_key_export_dir, os.path.dirname(path))
        QMessageBox.information(self, "Export", f"Exported to:\n{path}")

    def _on_import_clicked(self):
        # 按你要求：先不要实现任何 import 功能
        QMessageBox.information(
            self,
            "Import (Not Implemented)",
            "Import is not implemented yet.\n"
            "You can implement it later (open JSON and load into store).",
        )

    # -------- Internal UI ops --------
    def _emit_intent(self, *args):
        self.intent_changed.emit(self.get_intent())

    def _current_selected_uid(self) -> Optional[int]:
        sel = self.list_widget.selectedItems()
        if not sel:
            return None
        return int(sel[0].data(Qt.UserRole))

    def _ui_add_or_update_item(self, it: AnnotationItem, *, select: bool):
        text = self._format_item_text(it)
        lwi = self._uid_to_lwi.get(it.uid)
        if lwi is None:
            lwi = QListWidgetItem(text)
            lwi.setData(Qt.UserRole, it.uid)
            self._uid_to_lwi[it.uid] = lwi
            self.list_widget.addItem(lwi)
        else:
            lwi.setText(text)
        if select:
            self.list_widget.setCurrentItem(lwi)

    def _ui_remove_item(self, uid: int):
        lwi = self._uid_to_lwi.pop(uid, None)
        if lwi is None:
            return
        row = self.list_widget.row(lwi)
        self.list_widget.takeItem(row)

    def _on_selection_changed(self):
        uid = self._current_selected_uid()
        if uid is None:
            self.editor_stack.setCurrentIndex(0)
            self.selection_changed.emit(None)
            return

        it = self.store.get(uid)
        if it is None:
            self.editor_stack.setCurrentIndex(0)
            self.selection_changed.emit(None)
            return

        self._sync_editor_for_item(it)
        self.selection_changed.emit(uid)

    def _sync_editor_for_item(self, it: AnnotationItem):
        if it.anno_type == "bbox":
            self.editor_stack.setCurrentIndex(1)
            self.bbox_edit.set_data(it.data)
        elif it.anno_type == "polyline":
            self.editor_stack.setCurrentIndex(2)
            self.poly_info.set_data(it.data)
        else:
            self.editor_stack.setCurrentIndex(0)

    def _on_bbox_patch(self, patch: dict):
        uid = self._current_selected_uid()
        if uid is None:
            return
        it = self.store.update(uid, patch)
        if it is None:
            return
        self._ui_add_or_update_item(it, select=False)
        self.overlay_update.emit(uid, dict(patch))

    def _on_delete_clicked(self):
        uid = self._current_selected_uid()
        if uid is None:
            return
        it = self.store.delete(uid)
        if it is None:
            return
        self._ui_remove_item(uid)
        self.editor_stack.setCurrentIndex(0)
        self.selection_changed.emit(None)
        self.overlay_delete.emit(uid)

    def _on_clear_clicked(self):
        if self.list_widget.count() <= 0:
            return
        code = QMessageBox.question(
            self,
            "Clear All?",
            "This will remove ALL annotations in the panel and overlay.\nContinue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if code != QMessageBox.Yes:
            return

        self.store.clear()
        self._uid_to_lwi.clear()
        self.list_widget.clear()
        self.editor_stack.setCurrentIndex(0)
        self.selection_changed.emit(None)
        self.overlay_clear.emit()

    @staticmethod
    def _format_item_text(it: AnnotationItem) -> str:
        if it.anno_type == "bbox":
            t0 = int(round(float(it.data.get("t0", 0))))
            c0 = int(round(float(it.data.get("c0", 0))))
            t1 = int(round(float(it.data.get("t1", 0))))
            c1 = int(round(float(it.data.get("c1", 0))))
            return f"{it.label} [bbox] ({t0},{c0})→({t1},{c1})  #{it.uid}"
        if it.anno_type == "polyline":
            pts = it.data.get("points", []) or []
            return f"{it.label} [poly] n={len(pts)}  #{it.uid}"
        return f"{it.label} [{it.anno_type}]  #{it.uid}"
