from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robot_pipeline_app.config_store import DEFAULT_CONFIG_VALUES
from robot_pipeline_app.gui_qt_app import ensure_qt_application, qt_available
from robot_pipeline_app.gui_qt_secondary_pages import QtHistoryPage, QtVisualizerPage

_QT_AVAILABLE, _QT_REASON = qt_available()


class _FakeRunController:
    def run_process_async(self, **_kwargs):  # type: ignore[no-untyped-def]
        return False, "not used"


@unittest.skipUnless(_QT_AVAILABLE, _QT_REASON or "PySide6 unavailable")
class GuiQtSecondaryPagesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.app, _ = ensure_qt_application(["robot_pipeline.py", "gui"])

    def test_history_page_saves_deploy_notes_and_episode_edits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir) / "runs"
            run_dir = runs_dir / "deploy_20260307_120000"
            run_dir.mkdir(parents=True)
            metadata_path = run_dir / "metadata.json"
            metadata = {
                "run_id": run_dir.name,
                "mode": "deploy",
                "status": "success",
                "started_at_iso": "2026-03-07T12:00:00+00:00",
                "ended_at_iso": "2026-03-07T12:10:00+00:00",
                "duration_s": 600,
                "command": "python3 -m lerobot deploy",
                "command_argv": ["python3", "-m", "lerobot", "deploy"],
                "cwd": str(Path(tmpdir)),
                "deploy_episode_outcomes": {
                    "enabled": True,
                    "total_episodes": 2,
                    "episode_outcomes": [
                        {"episode": 1, "result": "success", "tags": [], "note": ""},
                        {"episode": 2, "result": "unmarked", "tags": [], "note": ""},
                    ],
                },
            }
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

            config = dict(DEFAULT_CONFIG_VALUES)
            config["runs_dir"] = str(runs_dir)
            page = QtHistoryPage(config=config, append_log=lambda _msg: None, run_controller=_FakeRunController())
            self.addCleanup(page.close)

            self.assertEqual(page.run_table.rowCount(), 1)
            self.assertFalse(page.deploy_editor_card.isHidden())

            page.episode_combo.setCurrentText("2")
            page.outcome_combo.setCurrentText("failed")
            page.tags_input.setText("camera, drift")
            page.episode_note_input.setText("Needs recalibration")
            page.save_episode_edit()

            page.overall_notes.setPlainText("Overall deploy notes")
            page.save_deployment_notes()

            updated = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(updated["deploy_notes_summary"], "Overall deploy notes")
            outcomes = updated["deploy_episode_outcomes"]["episode_outcomes"]
            second = next(entry for entry in outcomes if int(entry["episode"]) == 2)
            self.assertEqual(second["result"], "failed")
            self.assertEqual(second["tags"], ["camera", "drift"])
            self.assertEqual(second["note"], "Needs recalibration")
            self.assertTrue((run_dir / "notes.md").exists())
            self.assertTrue((run_dir / "episode_outcomes.csv").exists())
            self.assertTrue((run_dir / "episode_outcomes_summary.csv").exists())

    def test_visualizer_auto_selects_and_previews_first_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "deploy_run"
            source_dir.mkdir(parents=True)
            video_path = source_dir / "episode_001.mp4"
            video_path.write_bytes(b"fake-video")

            source_item = {
                "scope": "local",
                "kind": "deployment",
                "name": "deploy_run",
                "path": str(source_dir),
                "run_path": str(source_dir),
            }
            video_item = {
                "path": str(video_path),
                "relative_path": "episode_001.mp4",
                "size_text": "10 B",
            }

            with patch(
                "robot_pipeline_app.gui_qt_secondary_pages._collect_sources_for_refresh",
                return_value=([source_item], None, "deployments"),
            ), patch(
                "robot_pipeline_app.gui_qt_secondary_pages._collect_videos_for_source",
                return_value=[video_item],
            ):
                page = QtVisualizerPage(config=dict(DEFAULT_CONFIG_VALUES), append_log=lambda _msg: None)
                self.addCleanup(page.close)

            self.app.processEvents()
            self.assertEqual(page.source_table.rowCount(), 1)
            self.assertEqual(page.video_table.rowCount(), 1)
            self.assertEqual(page.video_table.currentRow(), 0)
            self.assertIn("episode_001.mp4", page.player_status.text())


if __name__ == "__main__":
    unittest.main()
