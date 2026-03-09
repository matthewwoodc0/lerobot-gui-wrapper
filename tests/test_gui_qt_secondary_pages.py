from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robot_pipeline_app.config_store import DEFAULT_CONFIG_VALUES
from robot_pipeline_app.gui_qt_app import ensure_qt_application, qt_available

_QT_AVAILABLE, _QT_REASON = qt_available()

if _QT_AVAILABLE:
    import numpy as np

    from robot_pipeline_app.gui_qt_secondary_pages import QtHistoryPage, QtVisualizerPage, _VideoGalleryTile
else:  # pragma: no cover - exercised only when Qt is unavailable
    QtHistoryPage = object  # type: ignore[assignment]
    QtVisualizerPage = object  # type: ignore[assignment]
    _VideoGalleryTile = object  # type: ignore[assignment]


class _FakeCapture:
    def __init__(self) -> None:
        self._fps = 20.0
        self._frame_count = 200.0
        self._position_frame = 0.0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, object]:
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        frame[:, :, 0] = int(self._position_frame) % 255
        self._position_frame += 1.0
        return True, frame

    def release(self) -> None:
        self.released = True

    def get(self, prop: float) -> float:
        if prop == _FakeCv2.CAP_PROP_FPS:
            return self._fps
        if prop == _FakeCv2.CAP_PROP_FRAME_COUNT:
            return self._frame_count
        if prop == _FakeCv2.CAP_PROP_POS_FRAMES:
            return self._position_frame
        if prop == _FakeCv2.CAP_PROP_POS_MSEC:
            return (self._position_frame / self._fps) * 1000.0
        return 0.0

    def set(self, prop: float, value: float) -> bool:
        if prop == _FakeCv2.CAP_PROP_POS_MSEC:
            self._position_frame = max(0.0, (float(value) / 1000.0) * self._fps)
            return True
        if prop == _FakeCv2.CAP_PROP_POS_FRAMES:
            self._position_frame = max(0.0, float(value))
            return True
        return True


class _FakeCv2:
    CAP_PROP_FPS = 5.0
    CAP_PROP_FRAME_COUNT = 7.0
    CAP_PROP_POS_FRAMES = 1.0
    CAP_PROP_POS_MSEC = 2.0
    COLOR_BGR2RGB = 4

    def __init__(self) -> None:
        self.capture = _FakeCapture()

    def VideoCapture(self, _source: str) -> _FakeCapture:
        return self.capture

    def cvtColor(self, frame: object, _code: int) -> object:
        return frame


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

            self.assertTrue(page.output_card.isHidden())
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

    def test_visualizer_auto_selects_and_renders_video_gallery(self) -> None:
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
            self.assertTrue(page.output_card.isHidden())
            self.assertEqual(page.source_table.rowCount(), 1)
            self.assertEqual(len(page._video_tiles), 1)
            self.assertEqual(page._video_tiles[0].title_label.text(), "episode_001.mp4")
            self.assertEqual(page.video_status.text(), "Displaying 1 video from the selected source.")
            self.assertIs(page.video_scroll.widget(), page.video_grid_host)
            self.assertGreaterEqual(page.video_scroll.minimumHeight(), 360)
            self.assertFalse(hasattr(page, "video_table"))

    def test_visualizer_hides_deployment_insights_for_dataset_source_mode(self) -> None:
        source_item = {
            "scope": "local",
            "kind": "dataset",
            "name": "dataset_one",
            "path": "/tmp/dataset_one",
        }

        with patch(
            "robot_pipeline_app.gui_qt_secondary_pages._collect_sources_for_refresh",
            return_value=([source_item], None, "datasets"),
        ), patch(
            "robot_pipeline_app.gui_qt_secondary_pages._collect_videos_for_source",
            return_value=[],
        ):
            page = QtVisualizerPage(config=dict(DEFAULT_CONFIG_VALUES), append_log=lambda _msg: None)
            self.addCleanup(page.close)
            page.source_combo.setCurrentIndex(1)

        self.app.processEvents()
        self.assertEqual(page._active_source(), "datasets")
        self.assertTrue(page.insights_card.isHidden())

    def test_visualizer_refreshes_when_source_mode_changes(self) -> None:
        calls: list[object] = []

        def _fake_collect(_config: dict[str, object], snapshot: object) -> tuple[list[dict[str, str]], None, str]:
            calls.append(snapshot)
            source_name = getattr(snapshot, "source", "deployments")
            return ([], None, str(source_name))

        with patch(
            "robot_pipeline_app.gui_qt_secondary_pages._collect_sources_for_refresh",
            side_effect=_fake_collect,
        ):
            page = QtVisualizerPage(config=dict(DEFAULT_CONFIG_VALUES), append_log=lambda _msg: None)
            self.addCleanup(page.close)
            page.source_combo.setCurrentIndex(2)

        self.app.processEvents()
        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual(getattr(calls[-1], "source", None), "models")
        self.assertEqual(page.root_input.text(), str(DEFAULT_CONFIG_VALUES["trained_models_dir"]))

    def test_video_gallery_tile_pause_and_seek_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "clip.mp4"
            video_path.write_bytes(b"fake-video")
            fake_cv2 = _FakeCv2()
            tile = _VideoGalleryTile(
                item={"path": str(video_path), "relative_path": "clip.mp4"},
                cv2_module=fake_cv2,
            )
            self.addCleanup(tile.close)

            tile.start()
            self.assertEqual(tile.play_pause_button.text(), "Pause")
            self.assertIn("/", tile.time_label.text())

            before_seek = fake_cv2.capture.get(_FakeCv2.CAP_PROP_POS_MSEC)
            tile.seek_seconds(5.0)
            after_seek = fake_cv2.capture.get(_FakeCv2.CAP_PROP_POS_MSEC)
            self.assertGreater(after_seek, before_seek)

            tile.toggle_pause()
            self.assertEqual(tile.play_pause_button.text(), "Play")

            paused_position = fake_cv2.capture.get(_FakeCv2.CAP_PROP_POS_MSEC)
            tile.seek_seconds(-5.0)
            rewound_position = fake_cv2.capture.get(_FakeCv2.CAP_PROP_POS_MSEC)
            self.assertLess(rewound_position, paused_position)

            tile.toggle_pause()
            self.assertEqual(tile.play_pause_button.text(), "Pause")


if __name__ == "__main__":
    unittest.main()
