"""Comparison source controller for the main GUI window."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtWidgets import QMessageBox

from filark._exceptions import print_exception
from filark.io.tapeio import NpyTape


def _print_exception(context: str) -> None:
    print_exception(context, prefix="CompareController")


class CompareControllerMixin:
    """Own comparison source loading, switching, and cleanup."""

    def _init_compare_state(self) -> None:
        self._compare_sources: list[dict] = []
        self._source_bar_initialized = False

    def _on_denoise_apply_finished(self, save_path: str) -> None:
        reply = QMessageBox.question(
            self,
            "Denoising Complete",
            f"File saved to:\n{save_path}\n\n"
            "Load the denoised file for side-by-side comparison?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._add_compare_source(save_path)

    def _on_compare_requested(self, npy_path: str) -> None:
        self._add_compare_source(npy_path)

    def _add_compare_source(self, npy_path: str) -> None:
        p = Path(npy_path)
        if not p.exists() or not p.is_file():
            QMessageBox.warning(self, "Compare", f"File not found:\n{npy_path}")
            return
        if p.suffix.lower() != ".npy":
            QMessageBox.warning(self, "Compare", "Only .npy files are supported for comparison.")
            return
        meta_path = p.with_name(p.stem + "_meta.json")

        dims = "nt_nc"
        fs = 1.0
        dx = 1.0
        dx_unit = "m"
        gauge_length = None

        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                dims = meta.get("dims", "nt_nc")
                fs = float(meta.get("fs", 1.0))
                dx = float(meta.get("dx", 1.0))
                dx_unit = meta.get("dx_unit", "m")
                raw_gauge = meta.get("gauge_length", None)
                gauge_length = None if raw_gauge is None else float(raw_gauge)
            except Exception:
                _print_exception(f"Failed to parse comparison metadata file: {meta_path}")

        try:
            tape = NpyTape(str(p), dims=dims, fs=fs, dx=dx, dx_unit=dx_unit, gauge_length=gauge_length)
        except Exception as exc:
            _print_exception(f"Failed to load comparison source: {p}")
            QMessageBox.warning(self, "Load Failed", f"Could not load {p.name}:\n{exc}")
            return

        self._add_compare_source_tape(tape, name=p.stem)

    def _add_compare_source_tape(self, tape, *, name: str | None = None) -> None:
        if self._source is None:
            QMessageBox.information(
                self,
                "Compare",
                "Please load a primary data source first before adding a comparison source.",
            )
            return

        src = self._source
        warnings = []
        if hasattr(tape, "shape") and hasattr(src, "shape"):
            ts, ss = tape.shape, src.shape
            if ts != ss:
                warnings.append(f"Shape mismatch: {ts} vs {ss}")
        for attr, label in (("fs", "fs"), ("dx", "dx")):
            tv, sv = getattr(tape, attr, None), getattr(src, attr, None)
            if tv is not None and sv is not None:
                if abs(float(tv) - float(sv)) / (abs(float(sv)) + 1e-12) > 0.01:
                    warnings.append(f"{label} mismatch: {tv} vs {sv}")
        if warnings:
            reply = QMessageBox.warning(
                self,
                "Compare - Compatibility Warning",
                "The source may not be directly comparable to the primary:\n\n"
                + "\n".join(f"  - {w}" for w in warnings)
                + "\n\nAdd anyway?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        if not self._source_bar_initialized:
            orig_name = self._original_source_name()
            self.source_bar.add_source(orig_name, self._source, active=True)
            self._compare_sources.append({"name": orig_name, "source": self._source})
            self._source_bar_initialized = True

        if name is None:
            for attr in ("path", "p"):
                p = getattr(tape, attr, None)
                if p is not None:
                    name = Path(str(p)).stem
                    break
            if name is None:
                name = f"Source {len(self._compare_sources) + 1}"

        existing_names = {e["name"] for e in self._compare_sources}
        base, suffix = name, 1
        while name in existing_names:
            name = f"{base}_{suffix}"
            suffix += 1

        self.source_bar.add_source(name, tape, active=False)
        self._compare_sources.append({"name": name, "source": tape})

    def _original_source_name(self) -> str:
        src = self._source
        for attr in ("path", "p"):
            p = getattr(src, attr, None)
            if p is not None:
                return Path(str(p)).stem
        return "Original"

    def _on_source_activated(self, source) -> None:
        if source is self._source:
            return
        hook = getattr(self.cfg, "on_source_loaded", None)
        if callable(hook):
            try:
                result = hook(source)
                if result is not None:
                    source = result
            except Exception:
                _print_exception("on_source_loaded hook raised an exception.")
        self._source = source
        self.ann_panel.set_source_context(source)
        self._apply_stream_processing_for_source(source)
        self.disp_model.switch_source(source)
        self.load_panel.set_current_source(source)
        self.viz_panel.set_source_meta(getattr(source, "start_time", None))
        self.viz_panel.reapply_to_canvas()

    def _cycle_compare_source(self, step: int) -> None:
        if len(self._compare_sources) < 2:
            return

        curr = -1
        for i, e in enumerate(self._compare_sources):
            if e["source"] is self._source:
                curr = i
                break
        if curr < 0:
            curr = 0

        nxt = (curr + int(step)) % len(self._compare_sources)
        target = self._compare_sources[nxt]

        self.source_bar.set_active_name(target["name"])
        self._on_source_activated(target["source"])
        self.canvas.native.setFocus()

    def _clear_compare_sources(self) -> None:
        for entry in self._compare_sources:
            src = entry.get("source")
            if src is not None and src is not self._source:
                try:
                    src.close()
                except Exception:
                    _print_exception("Failed to close comparison source while clearing compare set.")
        self._compare_sources.clear()
        self._source_bar_initialized = False
        self.source_bar.clear()
