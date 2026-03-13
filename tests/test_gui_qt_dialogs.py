from __future__ import annotations

import os
import unittest

from robot_pipeline_app.gui_qt_app import ensure_qt_application, qt_available
from robot_pipeline_app.gui_qt_dialogs import QtActionChoiceDialog, QtEditableCommandDialog, QtTextDialog
from robot_pipeline_app.gui_qt_runtime_helpers import QtRunHelperDialog

try:
    from PySide6.QtWidgets import QFrame
except Exception:  # pragma: no cover - exercised only when Qt imports fail
    QFrame = None  # type: ignore[assignment]

_QT_AVAILABLE, _QT_REASON = qt_available()


@unittest.skipUnless(_QT_AVAILABLE, _QT_REASON or "PySide6 unavailable")
class GuiQtDialogsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            cls.app, _ = ensure_qt_application(["robot_pipeline.py", "gui"])
        except RuntimeError as exc:
            raise unittest.SkipTest(str(exc)) from exc

    def test_text_dialog_copy_uses_override_payload(self) -> None:
        dialog = QtTextDialog(parent=None, title="Preview", text="display", copy_text="copy me")
        self.addCleanup(dialog.close)

        self.assertEqual(dialog.objectName(), "AppDialog")
        dialog.copy_current_text()
        clipboard = self.app.clipboard()
        assert clipboard is not None
        self.assertEqual(clipboard.text(), "copy me")

    def test_editable_command_dialog_parses_multiline_command(self) -> None:
        dialog = QtEditableCommandDialog(
            parent=None,
            title="Confirm Command",
            command_argv=["python3", "-m", "lerobot", "record"],
            intro_text="Review the command.",
            confirm_label="Run",
        )
        self.addCleanup(dialog.close)

        self.assertEqual(dialog.objectName(), "AppDialog")
        self.assertEqual(dialog.error_label.styleSheet(), "")
        dialog.editor.setPlainText("python3 -m lerobot\nrecord\n--dataset.repo_id=alice/demo")
        dialog.confirm_dialog()

        self.assertEqual(dialog.result_argv, ["python3", "-m", "lerobot", "record", "--dataset.repo_id=alice/demo"])

    def test_action_choice_dialog_tracks_chosen_action(self) -> None:
        dialog = QtActionChoiceDialog(
            parent=None,
            title="Quick Fixes",
            text="Choose an action",
            actions=[("fix_ports", "Apply Ports")],
        )
        self.addCleanup(dialog.close)

        self.assertEqual(dialog.objectName(), "AppDialog")
        dialog.choose_action("fix_ports")
        self.assertEqual(dialog.result_choice, "fix_ports")

    def test_run_helper_dialog_uses_shared_dialog_panel(self) -> None:
        dialog = QtRunHelperDialog(parent=None, mode_title="Deploy")
        self.addCleanup(dialog.close)

        self.assertEqual(dialog.objectName(), "AppDialog")
        self.assertIsNotNone(dialog.findChild(QFrame, "DialogPanel"))
        self.assertEqual(dialog.status_chip.text(), "Idle")


if __name__ == "__main__":
    unittest.main()
