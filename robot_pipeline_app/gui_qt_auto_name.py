from __future__ import annotations

from PySide6.QtWidgets import QLineEdit


class AutoNameController:
    def __init__(self, line_edit: QLineEdit) -> None:
        self._line_edit = line_edit
        self._mode = "auto"
        self._suppress_manual_tracking = False
        self._line_edit.textEdited.connect(self._on_text_edited)

    def _on_text_edited(self, _text: str) -> None:
        if self._suppress_manual_tracking:
            return
        self._mode = "manual"

    def is_auto(self) -> bool:
        return self._mode == "auto"

    def is_manual(self) -> bool:
        return self._mode == "manual"

    def mode(self) -> str:
        return self._mode

    def text(self) -> str:
        return self._line_edit.text().strip()

    def set_text(self, value: str, *, mode: str | None = None) -> None:
        self._suppress_manual_tracking = True
        try:
            self._line_edit.setText(str(value or ""))
        finally:
            self._suppress_manual_tracking = False
        if mode is not None:
            self._mode = "manual" if mode == "manual" else "auto"

    def reseed(self, value: str) -> None:
        self.set_text(value, mode="auto")

    def mark_auto(self) -> None:
        self._mode = "auto"

    def mark_manual(self) -> None:
        self._mode = "manual"

    def snapshot(self) -> dict[str, str]:
        return {
            "value": self.text(),
            "mode": self.mode(),
        }
