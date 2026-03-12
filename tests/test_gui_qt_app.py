from __future__ import annotations

import os
import threading
import time
import unittest
from unittest.mock import patch

try:
    from PySide6.QtCore import Qt

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
        with patch("robot_pipeline_app.gui_qt_app._has_huggingface_auth_token", return_value=True):
            window = create_qt_preview_window({"hf_username": "alice", "ui_theme_mode": "dark"})
            self.addCleanup(window.close)

        self.assertEqual(window.current_section_id(), "record")
        self.assertIn("Record", window.section_titles())
        self.assertIn("LeRobot GUI initialized.", window.log_contents())
        self.assertEqual(window.terminal_session_count(), 1)
        self.assertEqual(window.workspace_title_label.text(), "Record")
        self.assertFalse(window.workspace_window.isHidden())
        self.assertEqual(window.hf_status_title_label.text(), "Hugging Face")
        self.assertIn("alice", window.hf_status_label.text())
        self.assertFalse(hasattr(window, "latest_artifact_button"))

    def test_preview_window_shows_login_steps_when_hf_auth_missing(self) -> None:
        with patch("robot_pipeline_app.gui_qt_app._has_huggingface_auth_token", return_value=False):
            window = create_qt_preview_window({"ui_theme_mode": "dark", "hf_username": ""})
            self.addCleanup(window.close)

        self.assertIn("hf auth login", window.hf_status_label.text())
        self.assertIn("Config", window.hf_status_label.text())

    def test_page_scroll_wrappers_hide_horizontal_scrollbars(self) -> None:
        with patch("robot_pipeline_app.gui_qt_app._has_huggingface_auth_token", return_value=True):
            window = create_qt_preview_window({"ui_theme_mode": "dark", "hf_username": "alice"})
            self.addCleanup(window.close)

        for index in range(window.page_stack.count()):
            page = window.page_stack.widget(index)
            self.assertEqual(page.horizontalScrollBarPolicy(), Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

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
        self.assertFalse(window.terminal_window.isHidden())

        window.toggle_terminal_panel()
        self.assertFalse(window.terminal_visible())
        self.assertEqual(window.terminal_button.text(), "Show Terminal")
        self.assertTrue(window.terminal_window.isHidden())

        window.toggle_terminal_panel()
        self.assertTrue(window.terminal_visible())
        self.assertEqual(window.terminal_button.text(), "Hide Terminal")
        self.assertFalse(window.terminal_window.isHidden())

    def test_terminal_visibility_restores_from_config(self) -> None:
        window = create_qt_preview_window({"ui_theme_mode": "dark", "ui_terminal_visible": False})
        self.addCleanup(window.close)

        self.assertFalse(window.terminal_visible())
        self.assertEqual(window.terminal_button.text(), "Show Terminal")
        self.assertTrue(window.terminal_window.isHidden())
        self.assertTrue(window.terminal_tabs.isHidden())

    def test_terminal_toggle_persists_hidden_ui_preference(self) -> None:
        with patch("robot_pipeline_app.gui_qt_app.save_config") as mocked_save_config:
            window = create_qt_preview_window({"ui_theme_mode": "dark"})
            self.addCleanup(window.close)

            window.toggle_terminal_panel()

        self.assertFalse(window.terminal_visible())
        self.assertEqual(window.config["ui_terminal_visible"], False)
        mocked_save_config.assert_called_once()
        self.assertIs(mocked_save_config.call_args.args[0], window.config)
        self.assertEqual(mocked_save_config.call_args.kwargs, {"quiet": True})

    def test_new_terminal_session_adds_a_tab(self) -> None:
        window = create_qt_preview_window({"ui_theme_mode": "dark"})
        self.addCleanup(window.close)

        self.assertEqual(window.terminal_session_count(), 1)
        self.assertEqual(window.new_terminal_tab_button.text(), "+")
        window.new_terminal_session()

        self.assertEqual(window.terminal_session_count(), 2)
        self.assertEqual(window.terminal_tabs.count(), 2)
        self.assertEqual(window.terminal_tabs.tabText(1), "Terminal 2")

    def test_closing_last_terminal_session_creates_a_replacement(self) -> None:
        window = create_qt_preview_window({"ui_theme_mode": "dark"})
        self.addCleanup(window.close)

        window.close_terminal_session_at(0)

        self.assertEqual(window.terminal_session_count(), 1)
        self.assertEqual(window.terminal_tabs.count(), 1)

    def test_sidebar_toggle_updates_collapsed_state(self) -> None:
        window = create_qt_preview_window({"ui_theme_mode": "dark"})
        self.addCleanup(window.close)

        self.assertFalse(window.sidebar_collapsed())
        self.assertFalse(window.sidebar.isHidden())
        self.assertTrue(window.sidebar_rail.isHidden())

        window.toggle_sidebar()
        self.assertTrue(window.sidebar_collapsed())
        self.assertTrue(window.sidebar.isHidden())
        self.assertFalse(window.sidebar_rail.isHidden())

        window.toggle_sidebar()
        self.assertFalse(window.sidebar_collapsed())
        self.assertFalse(window.sidebar.isHidden())
        self.assertTrue(window.sidebar_rail.isHidden())

    def test_sidebar_collapsed_state_restores_from_config(self) -> None:
        window = create_qt_preview_window({"ui_theme_mode": "dark", "ui_sidebar_collapsed": True})
        self.addCleanup(window.close)

        self.assertTrue(window.sidebar_collapsed())
        self.assertTrue(window.sidebar.isHidden())
        self.assertFalse(window.sidebar_rail.isHidden())

    def test_sidebar_toggle_persists_collapsed_ui_preference(self) -> None:
        with patch("robot_pipeline_app.gui_qt_app.save_config") as mocked_save_config:
            window = create_qt_preview_window({"ui_theme_mode": "dark"})
            self.addCleanup(window.close)

            window.toggle_sidebar()

        self.assertTrue(window.sidebar_collapsed())
        self.assertEqual(window.config["ui_sidebar_collapsed"], True)
        mocked_save_config.assert_called_once()
        self.assertIs(mocked_save_config.call_args.args[0], window.config)
        self.assertEqual(mocked_save_config.call_args.kwargs, {"quiet": True})

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
