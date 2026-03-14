from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robot_pipeline_app.config_store import DEFAULT_CONFIG_VALUES
from robot_pipeline_app.gui_qt_app import ensure_qt_application, qt_available
from robot_pipeline_app.qt_bootstrap import probe_qt_platform_support

_qt_ok, _qt_reason = qt_available()
if not _qt_ok:
    _QT_AVAILABLE, _QT_REASON = False, _qt_reason or "PySide6 unavailable"
else:
    _QT_AVAILABLE, _QT_REASON = probe_qt_platform_support(platform_name="offscreen")

if _QT_AVAILABLE:
    import numpy as np
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QHeaderView, QLabel, QPushButton

    from robot_pipeline_app.gui_qt_page_base import _CameraSchemaEditor
    from robot_pipeline_app.gui_qt_secondary_pages import QtConfigPage, QtHistoryPage, QtVisualizerPage, _VideoGalleryTile
else:  # pragma: no cover - exercised only when Qt is unavailable
    Qt = object  # type: ignore[assignment]
    QHeaderView = object  # type: ignore[assignment]
    QLabel = object  # type: ignore[assignment]
    QPushButton = object  # type: ignore[assignment]
    _CameraSchemaEditor = object  # type: ignore[assignment]
    QtConfigPage = object  # type: ignore[assignment]
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

    def cancel_active_run(self) -> tuple[bool, str]:
        return False, "No active run."


@unittest.skipUnless(_QT_AVAILABLE, _QT_REASON or "PySide6 unavailable")
class GuiQtSecondaryPagesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            cls.app, _ = ensure_qt_application(["robot_pipeline.py", "gui"])
        except RuntimeError as exc:
            raise unittest.SkipTest(str(exc)) from exc

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

            self.assertFalse(page.output_card.isHidden())
            self.assertEqual(page.run_table.rowCount(), 1)
            self.assertFalse(page.deploy_editor_card.isHidden())
            self.assertIn("Status: Success", page.output.toPlainText())
            self.assertIn("Raw transcript is missing", page.raw_output.toPlainText())
            self.assertFalse(page.output_panel.explain_button.isEnabled())

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

    def test_history_page_enables_failure_explanation_for_failed_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir) / "runs"
            run_dir = runs_dir / "record_20260307_120000"
            run_dir.mkdir(parents=True)
            metadata_path = run_dir / "metadata.json"
            metadata = {
                "run_id": run_dir.name,
                "mode": "record",
                "status": "failed",
                "started_at_iso": "2026-03-07T12:00:00+00:00",
                "ended_at_iso": "2026-03-07T12:01:00+00:00",
                "duration_s": 60,
                "command": "python3 -m lerobot record",
                "command_argv": ["python3", "-m", "lerobot", "record"],
                "cwd": str(Path(tmpdir)),
                "runtime_diagnostics": [
                    {
                        "level": "FAIL",
                        "code": "MODEL-GPU_OOM",
                        "name": "GPU memory",
                        "detail": "GPU memory exhausted during run.",
                        "fix": "Reduce camera load or use a smaller checkpoint.",
                        "docs_ref": "Resources/error-catalog.md#model",
                        "attribution": "model",
                    }
                ],
                "first_failure_code": "MODEL-GPU_OOM",
                "first_failure_name": "GPU memory",
                "first_failure_detail": "GPU memory exhausted during run.",
                "first_failure_attribution": "model",
            }
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            (run_dir / "command.log").write_text("RuntimeError: CUDA out of memory.\n", encoding="utf-8")

            config = dict(DEFAULT_CONFIG_VALUES)
            config["runs_dir"] = str(runs_dir)
            page = QtHistoryPage(config=config, append_log=lambda _msg: None, run_controller=_FakeRunController())
            self.addCleanup(page.close)

            self.assertIn("Likely source: Model runtime", page.output.toPlainText())
            assert page.raw_output is not None
            self.assertIn("CUDA out of memory", page.raw_output.toPlainText())
            self.assertTrue(page.output_panel.explain_button.isEnabled())

    def test_config_page_snapshot_includes_runtime_snapshot(self) -> None:
        with patch(
            "robot_pipeline_app.gui_qt_config_page.build_compat_snapshot",
            return_value={"lerobot_version": "0.5.0", "validated_track": {"version_spec": "0.5.x"}},
        ), patch(
            "robot_pipeline_app.gui_qt_config_page.build_setup_status_summary",
            return_value="setup ok",
        ), patch(
            "robot_pipeline_app.gui_qt_config_page.probe_setup_wizard_status",
        ) as mocked_status:
            mocked_status.return_value = type("Status", (), {"venv_dir": Path("/tmp/env")})()
            page = QtConfigPage(config=dict(DEFAULT_CONFIG_VALUES), append_log=lambda _msg: None)
            self.addCleanup(page.close)
            payload = json.loads(page.output.toPlainText())

        self.assertIn("runtime_snapshot", payload)
        self.assertEqual(payload["runtime_snapshot"]["lerobot_version"], "0.5.0")

    def test_camera_schema_editor_uses_form_label_for_count_control(self) -> None:
        editor = _CameraSchemaEditor(config=dict(DEFAULT_CONFIG_VALUES))
        self.addCleanup(editor.close)

        labels = [label for label in editor.findChildren(QLabel) if label.text() == "Camera count"]

        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0].objectName(), "FormLabel")

    def test_camera_schema_editor_uses_responsive_column_modes(self) -> None:
        editor = _CameraSchemaEditor(config=dict(DEFAULT_CONFIG_VALUES))
        self.addCleanup(editor.close)

        header = editor.table.horizontalHeader()

        self.assertEqual(editor.table.horizontalScrollBarPolicy(), Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.assertEqual(header.sectionResizeMode(0), QHeaderView.ResizeMode.Stretch)
        self.assertEqual(header.sectionResizeMode(1), QHeaderView.ResizeMode.Stretch)
        self.assertEqual(header.sectionResizeMode(3), QHeaderView.ResizeMode.ResizeToContents)
        self.assertEqual(header.minimumSectionSize(), 56)

    def test_config_page_robot_preset_prefills_editable_fields(self) -> None:
        with patch(
            "robot_pipeline_app.gui_qt_config_page.build_compat_snapshot",
            return_value={"lerobot_version": "0.5.0"},
        ), patch(
            "robot_pipeline_app.gui_qt_config_page.build_setup_status_summary",
            return_value="setup ok",
        ), patch(
            "robot_pipeline_app.gui_qt_config_page.probe_setup_wizard_status",
        ) as mocked_status:
            mocked_status.return_value = type("Status", (), {"venv_dir": Path("/tmp/env")})()
            page = QtConfigPage(config=dict(DEFAULT_CONFIG_VALUES), append_log=lambda _msg: None)
            self.addCleanup(page.close)
            page.robot_preset_combo.setCurrentText("Unitree G1 (29 DOF)")
            page.apply_robot_preset()

        self.assertEqual(page._inputs["follower_robot_type"].text(), "unitree_g1_29dof")
        self.assertEqual(page._inputs["leader_robot_type"].text(), "unitree_g1_29dof")
        self.assertEqual(page._inputs["follower_robot_action_dim"].value(), 29)

    def test_config_page_applies_portable_profile_preset(self) -> None:
        with patch(
            "robot_pipeline_app.gui_qt_config_page.build_compat_snapshot",
            return_value={"lerobot_version": "0.5.0"},
        ), patch(
            "robot_pipeline_app.gui_qt_config_page.build_setup_status_summary",
            return_value="setup ok",
        ), patch(
            "robot_pipeline_app.gui_qt_config_page.probe_setup_wizard_status",
        ) as mocked_status:
            mocked_status.return_value = type("Status", (), {"venv_dir": Path("/tmp/env")})()
            page = QtConfigPage(config=dict(DEFAULT_CONFIG_VALUES), append_log=lambda _msg: None)
            self.addCleanup(page.close)
            page.profile_preset_combo.setCurrentText("SO-101 Lab Dual Cam")
            page.apply_profile_preset()

        self.assertIn('"wrist"', page.config["camera_schema_json"])
        self.assertEqual(page.config["active_profile_name"], "SO-101 Lab Dual Cam")

    def test_config_page_imports_and_exports_profiles_from_gui(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "robot_pipeline_app.gui_qt_config_page.build_compat_snapshot",
            return_value={"lerobot_version": "0.5.0"},
        ), patch(
            "robot_pipeline_app.gui_qt_config_page.build_setup_status_summary",
            return_value="setup ok",
        ), patch(
            "robot_pipeline_app.gui_qt_config_page.probe_setup_wizard_status",
        ) as mocked_status:
            mocked_status.return_value = type("Status", (), {"venv_dir": Path("/tmp/env")})()
            profile_path = Path(tmpdir) / "profile.yaml"
            profile_path.write_text(
                json.dumps(
                    {
                        "schema_version": "community_profile.v1",
                        "name": "Imported Profile",
                        "robot": {"follower": {"type": "so101_follower", "action_dim": 6}},
                        "camera": {"schema_json": {"front": {"index_or_path": 0}}},
                    }
                ),
                encoding="utf-8",
            )
            export_path = Path(tmpdir) / "exported_profile.yaml"
            page = QtConfigPage(config=dict(DEFAULT_CONFIG_VALUES), append_log=lambda _msg: None)
            self.addCleanup(page.close)

            with patch("robot_pipeline_app.gui_qt_config_page.QFileDialog.getOpenFileName", return_value=(str(profile_path), "yaml")):
                page.import_profile_from_file()
            with patch("robot_pipeline_app.gui_qt_config_page.QFileDialog.getSaveFileName", return_value=(str(export_path), "yaml")):
                page.export_profile_to_file()

        self.assertEqual(page.config["active_profile_name"], "profile")
        self.assertTrue(export_path.exists())

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
            with patch(
                "robot_pipeline_app.gui_qt_visualizer_page._collect_sources_for_refresh",
                return_value=([source_item], None, "deployments"),
            ):
                page = QtVisualizerPage(
                    config=dict(DEFAULT_CONFIG_VALUES),
                    append_log=lambda _msg: None,
                    run_controller=_FakeRunController(),
                )
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
            "robot_pipeline_app.gui_qt_visualizer_page._collect_sources_for_refresh",
            return_value=([source_item], None, "datasets"),
        ):
            page = QtVisualizerPage(
                config=dict(DEFAULT_CONFIG_VALUES),
                append_log=lambda _msg: None,
                run_controller=_FakeRunController(),
            )
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
            "robot_pipeline_app.gui_qt_visualizer_page._collect_sources_for_refresh",
            side_effect=_fake_collect,
        ):
            page = QtVisualizerPage(
                config=dict(DEFAULT_CONFIG_VALUES),
                append_log=lambda _msg: None,
                run_controller=_FakeRunController(),
            )
            self.addCleanup(page.close)
            page.source_combo.setCurrentIndex(2)

        self.app.processEvents()
        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual(getattr(calls[-1], "source", None), "models")
        self.assertEqual(page.root_input.text(), str(DEFAULT_CONFIG_VALUES["trained_models_dir"]))

    def test_visualizer_source_browser_keeps_root_browse_and_refresh_controls(self) -> None:
        with patch(
            "robot_pipeline_app.gui_qt_visualizer_page._collect_sources_for_refresh",
            return_value=([], None, "deployments"),
        ):
            page = QtVisualizerPage(
                config=dict(DEFAULT_CONFIG_VALUES),
                append_log=lambda _msg: None,
                run_controller=_FakeRunController(),
            )
            self.addCleanup(page.close)

        root_actions = page.root_input.parentWidget().findChildren(QPushButton)
        self.assertIn("Browse Root", [button.text() for button in root_actions])
        self.assertEqual(page.refresh_button.text(), "Refresh")
        self.assertEqual(page.refresh_button.objectName(), "AccentButton")

    def test_visualizer_restores_persisted_source_mode(self) -> None:
        calls: list[object] = []

        def _fake_collect(_config: dict[str, object], snapshot: object) -> tuple[list[dict[str, str]], None, str]:
            calls.append(snapshot)
            return ([], None, str(getattr(snapshot, "source", "deployments")))

        config = dict(DEFAULT_CONFIG_VALUES)
        config["ui_visualizer_source_kind"] = "datasets"

        with patch(
            "robot_pipeline_app.gui_qt_visualizer_page._collect_sources_for_refresh",
            side_effect=_fake_collect,
        ):
            page = QtVisualizerPage(
                config=config,
                append_log=lambda _msg: None,
                run_controller=_FakeRunController(),
            )
            self.addCleanup(page.close)

        self.app.processEvents()
        self.assertEqual(page._active_source(), "datasets")
        self.assertEqual(getattr(calls[-1], "source", None), "datasets")
        self.assertEqual(page.root_input.text(), str(DEFAULT_CONFIG_VALUES["record_data_dir"]))

    def test_visualizer_refresh_preserves_selected_source_identity(self) -> None:
        sources = [
            {"scope": "local", "kind": "dataset", "name": "dataset_a", "path": "/tmp/dataset_a"},
            {"scope": "local", "kind": "dataset", "name": "dataset_b", "path": "/tmp/dataset_b"},
        ]

        with patch(
            "robot_pipeline_app.gui_qt_visualizer_page._collect_sources_for_refresh",
            return_value=(sources, None, "datasets"),
        ):
            config = dict(DEFAULT_CONFIG_VALUES)
            config["ui_visualizer_source_kind"] = "datasets"
            page = QtVisualizerPage(
                config=config,
                append_log=lambda _msg: None,
                run_controller=_FakeRunController(),
            )
            self.addCleanup(page.close)
            page.source_table.selectRow(1)
            page.refresh_sources()

        self.app.processEvents()
        self.assertEqual(page.source_table.currentRow(), 1)
        self.assertEqual(config["ui_visualizer_selected_name"], "dataset_b")

    def test_visualizer_refresh_from_config_restores_source_once(self) -> None:
        calls: list[object] = []

        def _fake_collect(_config: dict[str, object], snapshot: object) -> tuple[list[dict[str, str]], None, str]:
            calls.append(snapshot)
            return ([], None, str(getattr(snapshot, "source", "deployments")))

        config = dict(DEFAULT_CONFIG_VALUES)

        with patch(
            "robot_pipeline_app.gui_qt_visualizer_page._collect_sources_for_refresh",
            side_effect=_fake_collect,
        ):
            page = QtVisualizerPage(
                config=config,
                append_log=lambda _msg: None,
                run_controller=_FakeRunController(),
            )
            self.addCleanup(page.close)
            calls.clear()
            config["ui_visualizer_source_kind"] = "datasets"
            page.refresh_from_config()

        self.app.processEvents()
        self.assertEqual(len(calls), 1)
        self.assertEqual(page._active_source(), "datasets")
        self.assertEqual(getattr(calls[0], "source", None), "datasets")

    def test_history_restores_persisted_filters(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["ui_history_mode_filter"] = "deploy"
        config["ui_history_status_filter"] = "failed"
        config["ui_history_query"] = "camera"

        page = QtHistoryPage(config=config, append_log=lambda _msg: None, run_controller=_FakeRunController())
        self.addCleanup(page.close)

        self.assertEqual(page.mode_combo.currentData(), "deploy")
        self.assertEqual(page.status_combo.currentData(), "failed")
        self.assertEqual(page.query_input.text(), "camera")

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
