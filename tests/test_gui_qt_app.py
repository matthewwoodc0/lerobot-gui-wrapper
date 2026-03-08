from __future__ import annotations

import os
import unittest

from robot_pipeline_app.gui_qt_app import (
    create_qt_preview_window,
    ensure_qt_application,
    qt_available,
    qt_preview_sections,
)

_QT_AVAILABLE, _QT_REASON = qt_available()


@unittest.skipUnless(_QT_AVAILABLE, _QT_REASON or "PySide6 unavailable")
class GuiQtAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.app, _ = ensure_qt_application(["robot_pipeline.py", "gui-qt"])

    def test_preview_sections_cover_all_primary_workflows(self) -> None:
        section_ids = [section.id for section in qt_preview_sections()]
        self.assertEqual(
            section_ids,
            ["record", "deploy", "teleop", "config", "training", "visualizer", "history"],
        )

    def test_preview_window_exposes_navigation_and_log_panel(self) -> None:
        window = create_qt_preview_window({"hf_username": "alice", "ui_theme_mode": "dark"})
        self.addCleanup(window.close)

        self.assertEqual(window.current_section_id(), "record")
        self.assertIn("Record", window.section_titles())
        self.assertIn("Qt preview shell initialized.", window.log_contents())

    def test_toggle_theme_mode_updates_theme_and_allows_navigation(self) -> None:
        window = create_qt_preview_window({"ui_theme_mode": "dark"})
        self.addCleanup(window.close)

        self.assertEqual(window.theme_mode, "dark")
        window.toggle_theme_mode()
        self.assertEqual(window.theme_mode, "light")

        window.select_section("history")
        self.assertEqual(window.current_section_id(), "history")
        self.assertIn("Switched to history preview.".lower(), window.log_contents().lower())


if __name__ == "__main__":
    unittest.main()
