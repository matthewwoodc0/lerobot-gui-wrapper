from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robot_pipeline_app.constants import DEFAULT_CONFIG_VALUES

try:
    from robot_pipeline_app.gui_qt_app import ensure_qt_application, qt_available
    from robot_pipeline_app.gui_qt_visualizer_page import QtVisualizerPage
except Exception as exc:  # pragma: no cover - exercised only when Qt imports fail
    ensure_qt_application = None  # type: ignore[assignment]
    QtVisualizerPage = None  # type: ignore[assignment]
    _QT_AVAILABLE, _QT_REASON = False, str(exc)
else:
    _qt_ok, _qt_reason = qt_available()
    if not _qt_ok:
        _QT_AVAILABLE, _QT_REASON = False, _qt_reason or "PySide6 unavailable"
    else:
        probe_env = dict(os.environ)
        probe_env.setdefault("QT_QPA_PLATFORM", "offscreen")
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "from PySide6.QtWidgets import QApplication; app = QApplication(['qt-smoke']); print('ok')",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=probe_env,
        )
        _QT_AVAILABLE = probe.returncode == 0 and probe.stdout.strip() == "ok"
        _QT_REASON = None if _QT_AVAILABLE else (probe.stderr.strip() or probe.stdout.strip() or "Qt offscreen smoke check failed")


class _FakeRunController:
    def run_process_async(self, **_kwargs):  # type: ignore[no-untyped-def]
        return True, ""

    def cancel_active_run(self) -> tuple[bool, str]:
        return False, "No active run."


@unittest.skipUnless(_QT_AVAILABLE, _QT_REASON or "PySide6 unavailable")
class GuiQtVisualizerToolsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.app, _ = ensure_qt_application(["robot_pipeline.py", "gui"])

    def test_dataset_tools_card_renders_with_episode_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "data"
            episodes_path = dataset_root / "demo_dataset" / "meta" / "episodes.jsonl"
            episodes_path.parent.mkdir(parents=True, exist_ok=True)
            episodes_path.write_text("{}\n{}\n{}\n", encoding="utf-8")

            config = dict(DEFAULT_CONFIG_VALUES)
            config["record_data_dir"] = str(dataset_root)
            config["last_dataset_repo_id"] = "alice/demo_dataset"

            with (
                patch("robot_pipeline_app.gui_qt_visualizer_page._collect_sources_for_refresh", return_value=([], None, "deployment runs")),
                patch("robot_pipeline_app.gui_qt_visualizer_page.save_config"),
            ):
                page = QtVisualizerPage(
                    config=config,
                    append_log=lambda _msg: None,
                    run_controller=_FakeRunController(),
                )
                self.addCleanup(page.close)
                page.refresh_dataset_episodes()

        self.assertEqual(page.dataset_tools_input.text(), "alice/demo_dataset")
        self.assertEqual(page.dataset_episodes_table.rowCount(), 3)
        self.assertEqual(page.dataset_episodes_table.columnCount(), 2)
        self.assertEqual(page.dataset_episodes_table.horizontalHeaderItem(0).text(), "Episode")
        self.assertEqual(page.dataset_episodes_table.horizontalHeaderItem(1).text(), "Select")
        self.assertEqual(page.dataset_tools_status.text(), "Loaded 3 episode(s) from alice/demo_dataset.")
        self.assertEqual(page.merge_output_dataset_input.placeholderText(), "owner/merged_dataset")
        self.assertEqual(page.merge_datasets_button.text(), "Merge Datasets")
        self.assertEqual(page.merge_source_datasets_input.placeholderText(), "owner/dataset_a\nowner/dataset_b")

    def test_dataset_visualization_card_renders_with_rerun_button(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["last_dataset_repo_id"] = "alice/demo_dataset"

        with (
            patch("robot_pipeline_app.gui_qt_visualizer_page._collect_sources_for_refresh", return_value=([], None, "deployment runs")),
            patch("robot_pipeline_app.gui_qt_visualizer_page.save_config"),
        ):
            page = QtVisualizerPage(
                config=config,
                append_log=lambda _msg: None,
                run_controller=_FakeRunController(),
            )
            self.addCleanup(page.close)

        self.assertEqual(page.visualize_dataset_input.text(), "alice/demo_dataset")
        self.assertEqual(page.visualize_episode_input.minimum(), 0)
        self.assertEqual(page.visualize_episode_input.maximum(), 9999)
        self.assertEqual(page.visualize_open_button.text(), "Open in Rerun")
        self.assertEqual(page.visualize_cancel_button.text(), "Cancel")


if __name__ == "__main__":
    unittest.main()
