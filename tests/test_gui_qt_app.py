from __future__ import annotations

import os
import threading
import time
import unittest

try:
    from robot_pipeline_app.gui_qt_app import (
        _QtAfterAdapter,
        create_qt_preview_window,
        ensure_qt_application,
        qt_available,
        qt_preview_sections,
    )
except Exception as exc:  # pragma: no cover - exercised only when Qt imports fail
    _QtAfterAdapter = None  # type: ignore[assignment]
    create_qt_preview_window = None  # type: ignore[assignment]
    ensure_qt_application = None  # type: ignore[assignment]
    qt_preview_sections = None  # type: ignore[assignment]
    _QT_AVAILABLE, _QT_REASON = False, str(exc)
else:
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
            ["record", "deploy", "teleop", "config", "visualizer", "history"],
        )

    def test_preview_window_exposes_navigation_and_log_panel(self) -> None:
        window = create_qt_preview_window({"hf_username": "alice", "ui_theme_mode": "dark"})
        self.addCleanup(window.close)

        self.assertEqual(window.current_section_id(), "record")
        self.assertIn("Record", window.section_titles())
        self.assertIn("LeRobot GUI initialized.", window.log_contents())
        self.assertFalse(hasattr(window, "latest_artifact_button"))

    def test_toggle_theme_mode_updates_theme_and_allows_navigation(self) -> None:
        window = create_qt_preview_window({"ui_theme_mode": "dark"})
        self.addCleanup(window.close)

        self.assertEqual(window.theme_mode, "dark")
        window.toggle_theme_mode()
        self.assertEqual(window.theme_mode, "light")

        window.select_section("history")
        self.assertEqual(window.current_section_id(), "history")
        self.assertIn("Switched to history.".lower(), window.log_contents().lower())

    def test_terminal_toggle_button_is_visible_and_updates_state(self) -> None:
        window = create_qt_preview_window({"ui_theme_mode": "dark"})
        self.addCleanup(window.close)

        self.assertTrue(window.terminal_visible())
        self.assertEqual(window.terminal_button.text(), "Hide Terminal")
        self.assertIs(window.sidebar.layout().itemAt(window.sidebar.layout().count() - 1).widget(), window.terminal_button)

        window.toggle_terminal_panel()
        self.assertFalse(window.terminal_visible())
        self.assertEqual(window.terminal_button.text(), "Show Terminal")

        window.toggle_terminal_panel()
        self.assertTrue(window.terminal_visible())
        self.assertEqual(window.terminal_button.text(), "Hide Terminal")

    def test_qt_after_adapter_delivers_callbacks_from_worker_thread(self) -> None:
        adapter = _QtAfterAdapter()
        received: list[str] = []

        worker = threading.Thread(target=lambda: adapter.after(0, received.append, "ok"))
        worker.start()
        worker.join(timeout=2)

        deadline = time.time() + 2
        while time.time() < deadline and not received:
            self.app.processEvents()
            time.sleep(0.01)

        self.assertEqual(received, ["ok"])


if __name__ == "__main__":
    unittest.main()
