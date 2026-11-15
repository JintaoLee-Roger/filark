"""Go-To dialog for jumping to a source sample/time."""

from __future__ import annotations

import datetime as _dt

from filark.utils.time import tape_sample_to_unix, tape_unix_to_sample

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
)


class GoToDialog(QDialog):
    """Jump to a position by sample index, elapsed seconds, or UTC time."""

    def __init__(self, nt: int, fs: float, start_time, parent=None, *, tape=None):
        super().__init__(parent)
        self.setWindowTitle("Go to position")
        self._nt = int(nt)
        self._fs = float(fs) or 1.0
        self._start_time = start_time
        # Optional source used for gap-aware sample <-> UTC conversion.
        self._tape = tape

        layout = QFormLayout(self)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(10)

        self._sp_sample = QSpinBox()
        self._sp_sample.setRange(0, max(0, self._nt - 1))
        self._sp_sample.setSingleStep(max(1, self._nt // 1000))
        layout.addRow("Sample index:", self._sp_sample)

        self._sp_elapsed = QDoubleSpinBox()
        self._sp_elapsed.setRange(0.0, (self._nt - 1) / self._fs)
        self._sp_elapsed.setDecimals(3)
        self._sp_elapsed.setSingleStep(1.0)
        self._sp_elapsed.setSuffix(" s")
        layout.addRow("Elapsed time:", self._sp_elapsed)

        if start_time is not None:
            self._dt_mod = _dt
            self._le_abs = QLineEdit()
            self._le_abs.setPlaceholderText("YYYY-MM-DD HH:MM:SS.mmm")
            self._le_abs.setText(self._fmt_utc(float(start_time)))
            layout.addRow("UTC time:", self._le_abs)
            self._le_abs.editingFinished.connect(self._on_abs_text_changed)
        else:
            self._le_abs = None

        total_s = (self._nt - 1) / self._fs
        hint = QLabel(f"Total: {self._nt} samples | {total_s:.1f} s | fs={self._fs:.1f} Hz")
        hint.setEnabled(False)
        layout.addRow(hint)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

        self._sp_sample.valueChanged.connect(self._on_sample_changed)
        self._sp_elapsed.valueChanged.connect(self._on_elapsed_changed)
        self._updating = False

    def _abs_of_sample(self, sample: int) -> float:
        t = tape_sample_to_unix(self._tape, sample) if self._tape is not None else None
        if t is None:
            return float(self._start_time) + sample / self._fs
        return float(t)

    def _sample_of_abs(self, unix_ts: float) -> int:
        n = tape_unix_to_sample(self._tape, unix_ts, nt=self._nt) if self._tape is not None else None
        if n is None:
            n = int(round((unix_ts - float(self._start_time)) * self._fs))
        return max(0, min(self._nt - 1, int(n)))

    def _fmt_utc(self, unix_ts: float) -> str:
        dt = self._dt_mod.datetime.fromtimestamp(unix_ts, tz=self._dt_mod.timezone.utc)
        ms = dt.microsecond // 1000
        return dt.strftime(f"%Y-%m-%d %H:%M:%S.{ms:03d}")

    def _parse_utc(self, text: str) -> float | None:
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = self._dt_mod.datetime.strptime(text.strip(), fmt)
            except ValueError:
                continue
            if not (0 <= dt.hour <= 23 and 0 <= dt.minute <= 59 and 0 <= dt.second <= 59):
                return None
            dt = dt.replace(tzinfo=self._dt_mod.timezone.utc)
            return dt.timestamp()
        return None

    def _on_sample_changed(self, v: int):
        if self._updating:
            return
        self._updating = True
        elapsed = v / self._fs
        self._sp_elapsed.setValue(elapsed)
        if self._le_abs is not None and self._start_time is not None:
            self._le_abs.setText(self._fmt_utc(self._abs_of_sample(v)))
        self._updating = False

    def _on_elapsed_changed(self, v: float):
        if self._updating:
            return
        self._updating = True
        s = max(0, min(self._nt - 1, int(round(v * self._fs))))
        self._sp_sample.setValue(s)
        if self._le_abs is not None and self._start_time is not None:
            self._le_abs.setText(self._fmt_utc(self._abs_of_sample(s)))
        self._updating = False

    def _on_abs_text_changed(self):
        if self._updating or self._start_time is None or self._le_abs is None:
            return
        unix_ts = self._parse_utc(self._le_abs.text())
        if unix_ts is None:
            self._le_abs.setText(self._fmt_utc(self._abs_of_sample(self._sp_sample.value())))
            return
        self._updating = True
        s = self._sample_of_abs(unix_ts)
        self._sp_sample.setValue(s)
        self._sp_elapsed.setValue(s / self._fs)
        self._le_abs.setText(self._fmt_utc(self._abs_of_sample(s)))
        self._updating = False

    def result_sample(self) -> int | None:
        return int(self._sp_sample.value())
