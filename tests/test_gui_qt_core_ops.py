from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from robot_pipeline_app.config_store import DEFAULT_CONFIG_VALUES
from robot_pipeline_app.hardware_workflows import MotorSetupRequest, MotorSetupSupport, ReplayRequest, ReplaySupport
from robot_pipeline_app.workspace_provenance import read_workspace_provenance, write_workspace_provenance

try:
    from PySide6.QtWidgets import QFrame, QSizePolicy
    from robot_pipeline_app.gui_qt_app import ensure_qt_application, qt_available
    from robot_pipeline_app.gui_qt_core_ops import (
        DeployOpsPanel,
        MotorSetupOpsPanel,
        RecordOpsPanel,
        ReplayOpsPanel,
        TeleopOpsPanel,
        _QtDatasetUploadDialog,
        _QtModelUploadDialog,
    )
except Exception as exc:  # pragma: no cover - exercised only when Qt imports fail
    ensure_qt_application = None  # type: ignore[assignment]
    DeployOpsPanel = None  # type: ignore[assignment]
    MotorSetupOpsPanel = None  # type: ignore[assignment]
    RecordOpsPanel = None  # type: ignore[assignment]
    ReplayOpsPanel = None  # type: ignore[assignment]
    TeleopOpsPanel = None  # type: ignore[assignment]
    _QtDatasetUploadDialog = None  # type: ignore[assignment]
    _QtModelUploadDialog = None  # type: ignore[assignment]
    QFrame = None  # type: ignore[assignment]
    QSizePolicy = None  # type: ignore[assignment]
    _QT_AVAILABLE, _QT_REASON = False, str(exc)
else:
    _QT_AVAILABLE, _QT_REASON = qt_available()


class _FakeRunController:
    def __init__(self) -> None:
        self.last_cmd: list[str] | None = None
        self.last_cwd = None
        self.last_kwargs: dict[str, object] | None = None
        self.last_complete_callback = None
        self.cancel_calls = 0
        self.cancel_result: tuple[bool, str] = (False, "No active run.")

    def cancel_active_run(self) -> tuple[bool, str]:
        self.cancel_calls += 1
        return self.cancel_result

    def send_arrow_key(self, direction: str) -> tuple[bool, str]:
        return True, f"Sent {direction} arrow key"

    def run_process_async(self, *, cmd, cwd, hooks, complete_callback, **kwargs):  # type: ignore[no-untyped-def]
        self.last_cmd = list(cmd)
        self.last_cwd = cwd
        self.last_kwargs = dict(kwargs)
        self.last_complete_callback = complete_callback
        _ = hooks
        return True, ""


@unittest.skipUnless(_QT_AVAILABLE, _QT_REASON or "PySide6 unavailable")
class GuiQtCoreOpsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            cls.app, _ = ensure_qt_application(["robot_pipeline.py", "gui"])
        except RuntimeError as exc:
            raise unittest.SkipTest(str(exc)) from exc

    def test_record_preview_opens_modal_dialog(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        panel = RecordOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
        self.addCleanup(panel.close)

        self.assertTrue(panel.output_card.isHidden())

        with tempfile.TemporaryDirectory() as tmpdir:
            panel.dataset_input.setText("alice/demo")
            panel.dataset_root_input.setText(tmpdir)

            with patch("robot_pipeline_app.gui_qt_ops_base.show_text_dialog") as mocked_dialog:
                panel.preview_command()

        mocked_dialog.assert_called_once()
        self.assertEqual(panel.output.toPlainText(), "")

    def test_record_run_uses_editable_command_before_launch(self) -> None:
        controller = _FakeRunController()
        logs: list[str] = []
        config = dict(DEFAULT_CONFIG_VALUES)
        panel = RecordOpsPanel(config=config, append_log=logs.append, run_controller=controller)
        self.addCleanup(panel.close)

        with tempfile.TemporaryDirectory() as tmpdir:
            panel.dataset_input.setText("alice/demo")
            panel.dataset_root_input.setText(tmpdir)
            req, cmd, error = panel._build()

            self.assertIsNone(error)
            assert req is not None and cmd is not None

            with (
                patch("robot_pipeline_app.gui_qt_ops_base.ask_editable_command_dialog", return_value=list(cmd)) as mocked_edit,
                patch("robot_pipeline_app.gui_qt_ops_base.ask_text_dialog", return_value=True) as mocked_preflight,
                patch(
                    "robot_pipeline_app.gui_qt_record.run_preflight_for_record",
                    return_value=[("PASS", "Environment", "Ready.")],
                ) as mocked_checks,
            ):
                panel.run_record()

        mocked_edit.assert_called_once()
        mocked_preflight.assert_called_once()
        mocked_checks.assert_called_once()
        self.assertEqual(controller.last_cmd, cmd)

    def test_record_cancel_advances_to_next_dataset_name(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        config["hf_username"] = "alice"

        with tempfile.TemporaryDirectory() as tmpdir:
            config["record_data_dir"] = tmpdir
            config["last_dataset_name"] = "demo_1"
            with patch("robot_pipeline_app.repo_utils.dataset_exists_on_hf", return_value=False):
                panel = RecordOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
                self.addCleanup(panel.close)

            panel.dataset_input.setText("alice/demo_1")
            panel.dataset_root_input.setText(tmpdir)

            with (
                patch("robot_pipeline_app.gui_qt_ops_base.ask_editable_command_dialog", side_effect=lambda **kwargs: list(kwargs["command_argv"])),
                patch("robot_pipeline_app.gui_qt_ops_base.ask_text_dialog", return_value=True),
                patch("robot_pipeline_app.gui_qt_record.run_preflight_for_record", return_value=[("PASS", "Environment", "Ready.")]),
            ):
                panel.run_record()

            assert controller.last_complete_callback is not None
            controller.last_complete_callback(0, True)

            self.assertEqual(panel.dataset_input.text(), "alice/demo_2")

    def test_record_scan_ports_applies_detected_defaults(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        panel = RecordOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
        self.addCleanup(panel.close)

        scan_entries = [
            {
                "path": "/dev/cu.usbmodem1",
                "by_id": [],
                "readable": True,
                "writable": True,
                "busy": False,
                "busy_detail": "",
                "manufacturer": "",
                "product": "",
                "likely_motor_controller": True,
            }
        ]
        with (
            patch("robot_pipeline_app.gui_qt_ops_base.scan_robot_serial_ports", return_value=scan_entries),
            patch(
                "robot_pipeline_app.gui_qt_ops_base.suggest_follower_leader_ports",
                return_value=("/dev/cu.usbmodem2", "/dev/cu.usbmodem1"),
            ),
            patch("robot_pipeline_app.gui_qt_ops_base.ask_text_dialog_with_actions", return_value="apply_ports"),
            patch("robot_pipeline_app.gui_qt_record.save_config"),
        ):
            panel.scan_robot_ports()

        self.assertEqual(config["follower_port"], "/dev/cu.usbmodem2")
        self.assertEqual(config["leader_port"], "/dev/cu.usbmodem1")

    def test_record_action_row_makes_run_record_first_and_primary(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        panel = RecordOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
        self.addCleanup(panel.close)

        texts = [button.text() for button in panel._action_buttons]

        self.assertGreaterEqual(len(panel._action_buttons), 4)
        self.assertEqual(texts[0], "Run Record")
        self.assertEqual(panel._action_buttons[0].objectName(), "AccentButton")
        self.assertEqual(texts[1], "Preview Command")
        self.assertNotEqual(panel._action_buttons[1].objectName(), "AccentButton")

    def test_record_panel_advances_from_last_numbered_dataset_name(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        config["hf_username"] = "alice"

        with tempfile.TemporaryDirectory() as tmpdir:
            config["record_data_dir"] = tmpdir
            config["last_dataset_name"] = "demo_5"
            Path(tmpdir, "demo_5").mkdir(parents=True, exist_ok=True)

            with patch("robot_pipeline_app.repo_utils.dataset_exists_on_hf", return_value=False):
                panel = RecordOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
                self.addCleanup(panel.close)

                self.assertEqual(panel.dataset_input.text(), "alice/demo_6")

    def test_record_local_dataset_tree_populates_from_record_root(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "alice" / "demo_local"
            (dataset_dir / "meta").mkdir(parents=True, exist_ok=True)
            (dataset_dir / "meta" / "episodes.jsonl").write_text("{}\n", encoding="utf-8")
            config["record_data_dir"] = tmpdir

            panel = RecordOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
            self.addCleanup(panel.close)

            self.assertEqual(panel.local_dataset_tree.topLevelItemCount(), 1)
            owner_item = panel.local_dataset_tree.topLevelItem(0)
            self.assertEqual(owner_item.text(0), "alice")
            self.assertEqual(owner_item.childCount(), 1)
            self.assertEqual(owner_item.child(0).text(0), "demo_local")
            self.assertEqual(owner_item.child(0).text(1), "Dataset")

    def test_record_hf_dataset_owner_defaults_from_saved_username(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        config["hf_username"] = "alice"
        panel = RecordOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
        self.addCleanup(panel.close)

        self.assertEqual(panel.local_hf_owner_input.text(), "alice")

    def test_use_selected_dataset_in_record_prefers_local_provenance_then_hf_selection(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        config["hf_username"] = "alice"

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "demo_local"
            (dataset_dir / "meta").mkdir(parents=True, exist_ok=True)
            (dataset_dir / "meta" / "episodes.jsonl").write_text("{}\n", encoding="utf-8")
            write_workspace_provenance(dataset_dir, payload={"source": "huggingface", "repo_id": "org/demo_remote"})
            config["record_data_dir"] = tmpdir

            panel = RecordOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
            self.addCleanup(panel.close)

            dataset_item = panel.local_dataset_tree.topLevelItem(0)
            panel.local_dataset_tree.setCurrentItem(dataset_item)
            panel.use_selected_dataset_in_record()
            self.assertEqual(panel.dataset_input.text(), "org/demo_remote")

            panel.local_dataset_tree.clearSelection()
            panel._apply_hf_dataset_rows(([{"repo_id": "alice/hf_dataset", "downloads": 5, "likes": 2}], None))
            panel.hf_dataset_table.selectRow(0)
            panel.use_selected_dataset_in_record()
            self.assertEqual(panel.dataset_input.text(), "alice/hf_dataset")

    def test_record_dataset_upload_blocks_when_hf_auth_missing(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        panel = RecordOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
        self.addCleanup(panel.close)

        with patch("robot_pipeline_app.gui_qt_record.has_huggingface_auth_token", return_value=False):
            panel.open_dataset_upload_dialog()

        self.assertIsNone(controller.last_cmd)
        self.assertIn("hf auth login", panel.output.toPlainText())

    def test_record_dataset_upload_warns_when_remote_repo_exists(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        panel = RecordOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
        self.addCleanup(panel.close)

        fake_dialog = SimpleNamespace(
            result_request={
                "repo_id": "alice/demo_remote",
                "local_dataset": Path("/tmp/demo_local"),
                "remote_exists": True,
                "upload_cmd": ["huggingface-cli", "upload", "alice/demo_remote", "/tmp/demo_local", "--repo-type", "dataset"],
                "checks": [("PASS", "Local dataset folder", "/tmp/demo_local")],
                "provenance_repo_id": "",
                "provenance_matches_target": False,
            },
            result_settings={"local_dataset": "/tmp/demo_local", "owner": "alice", "repo_name": "demo_remote"},
            exec=lambda: None,
        )

        with (
            patch("robot_pipeline_app.gui_qt_record.has_huggingface_auth_token", return_value=True),
            patch("robot_pipeline_app.gui_qt_record._QtDatasetUploadDialog", return_value=fake_dialog),
            patch.object(panel, "_confirm_preflight_review", return_value=True),
            patch("robot_pipeline_app.gui_qt_record.ask_text_dialog", return_value=True) as mocked_confirm,
            patch("robot_pipeline_app.gui_qt_record.save_config"),
        ):
            panel.open_dataset_upload_dialog()

        self.assertEqual(controller.last_cmd, fake_dialog.result_request["upload_cmd"])
        self.assertTrue(any(call.kwargs.get("title") == "Remote Dataset Exists" for call in mocked_confirm.call_args_list))

    def test_record_dataset_upload_warns_when_local_provenance_matches_target(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        panel = RecordOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
        self.addCleanup(panel.close)

        fake_dialog = SimpleNamespace(
            result_request={
                "repo_id": "alice/demo_remote",
                "local_dataset": Path("/tmp/demo_local"),
                "remote_exists": False,
                "upload_cmd": ["huggingface-cli", "upload", "alice/demo_remote", "/tmp/demo_local", "--repo-type", "dataset"],
                "checks": [("PASS", "Local dataset folder", "/tmp/demo_local")],
                "provenance_repo_id": "alice/demo_remote",
                "provenance_matches_target": True,
            },
            result_settings={"local_dataset": "/tmp/demo_local", "owner": "alice", "repo_name": "demo_remote"},
            exec=lambda: None,
        )

        with (
            patch("robot_pipeline_app.gui_qt_record.has_huggingface_auth_token", return_value=True),
            patch("robot_pipeline_app.gui_qt_record._QtDatasetUploadDialog", return_value=fake_dialog),
            patch.object(panel, "_confirm_preflight_review", return_value=True),
            patch("robot_pipeline_app.gui_qt_record.ask_text_dialog", return_value=True) as mocked_confirm,
            patch("robot_pipeline_app.gui_qt_record.save_config"),
        ):
            panel.open_dataset_upload_dialog()

        self.assertEqual(controller.last_cmd, fake_dialog.result_request["upload_cmd"])
        self.assertTrue(any(call.kwargs.get("title") == "Dataset Already Linked" for call in mocked_confirm.call_args_list))

    def test_record_dataset_upload_launches_expected_command(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        panel = RecordOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
        self.addCleanup(panel.close)

        fake_dialog = SimpleNamespace(
            result_request={
                "repo_id": "alice/demo_remote",
                "local_dataset": Path("/tmp/demo_local"),
                "remote_exists": False,
                "upload_cmd": ["huggingface-cli", "upload", "alice/demo_remote", "/tmp/demo_local", "--repo-type", "dataset"],
                "checks": [("PASS", "Local dataset folder", "/tmp/demo_local")],
                "provenance_repo_id": "",
                "provenance_matches_target": False,
            },
            result_settings={"local_dataset": "/tmp/demo_local", "owner": "alice", "repo_name": "demo_remote"},
            exec=lambda: None,
        )

        with (
            patch("robot_pipeline_app.gui_qt_record.has_huggingface_auth_token", return_value=True),
            patch("robot_pipeline_app.gui_qt_record._QtDatasetUploadDialog", return_value=fake_dialog),
            patch.object(panel, "_confirm_preflight_review", return_value=True),
            patch("robot_pipeline_app.gui_qt_record.ask_text_dialog", return_value=True),
            patch("robot_pipeline_app.gui_qt_record.save_config"),
        ):
            panel.open_dataset_upload_dialog()

        self.assertEqual(controller.last_cmd, fake_dialog.result_request["upload_cmd"])
        assert controller.last_kwargs is not None
        self.assertEqual(controller.last_kwargs["run_mode"], "upload")
        self.assertEqual(controller.last_kwargs["preflight_checks"], fake_dialog.result_request["checks"])
        self.assertEqual(controller.last_kwargs["artifact_context"], {"dataset_repo_id": "alice/demo_remote"})

    def test_record_dataset_upload_keeps_global_hf_username_unchanged(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        config["hf_username"] = "alice"
        panel = RecordOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
        self.addCleanup(panel.close)

        fake_dialog = SimpleNamespace(
            result_request={
                "repo_id": "robot-lab/demo_remote",
                "local_dataset": Path("/tmp/demo_local"),
                "remote_exists": False,
                "upload_cmd": ["huggingface-cli", "upload", "robot-lab/demo_remote", "/tmp/demo_local", "--repo-type", "dataset"],
                "checks": [("PASS", "Local dataset folder", "/tmp/demo_local")],
                "provenance_repo_id": "",
                "provenance_matches_target": False,
            },
            result_settings={"local_dataset": "/tmp/demo_local", "owner": "robot-lab", "repo_name": "demo_remote"},
            exec=lambda: None,
        )

        with (
            patch("robot_pipeline_app.gui_qt_record.has_huggingface_auth_token", return_value=True),
            patch("robot_pipeline_app.gui_qt_record._QtDatasetUploadDialog", return_value=fake_dialog),
            patch.object(panel, "_confirm_preflight_review", return_value=True),
            patch("robot_pipeline_app.gui_qt_record.ask_text_dialog", return_value=True),
            patch("robot_pipeline_app.gui_qt_record.save_config"),
        ):
            panel.open_dataset_upload_dialog()

        self.assertEqual(config["hf_username"], "alice")
        self.assertEqual(config["record_hf_dataset_owner"], "robot-lab")
        self.assertEqual(panel.local_hf_owner_input.text(), "robot-lab")

    def test_record_dataset_upload_writes_local_provenance_after_success(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_dataset = Path(tmpdir) / "demo_local"
            (local_dataset / "meta").mkdir(parents=True, exist_ok=True)
            (local_dataset / "meta" / "episodes.jsonl").write_text("{}\n", encoding="utf-8")

            panel = RecordOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
            self.addCleanup(panel.close)

            fake_dialog = SimpleNamespace(
                result_request={
                    "repo_id": "alice/demo_remote",
                    "local_dataset": local_dataset,
                    "remote_exists": False,
                    "upload_cmd": ["huggingface-cli", "upload", "alice/demo_remote", str(local_dataset), "--repo-type", "dataset"],
                    "checks": [("PASS", "Local dataset folder", str(local_dataset))],
                    "provenance_repo_id": "",
                    "provenance_matches_target": False,
                },
                result_settings={"local_dataset": str(local_dataset), "owner": "alice", "repo_name": "demo_remote"},
                exec=lambda: None,
            )

            with (
                patch("robot_pipeline_app.gui_qt_record.has_huggingface_auth_token", return_value=True),
                patch("robot_pipeline_app.gui_qt_record._QtDatasetUploadDialog", return_value=fake_dialog),
                patch.object(panel, "_confirm_preflight_review", return_value=True),
                patch("robot_pipeline_app.gui_qt_record.ask_text_dialog", return_value=True),
                patch("robot_pipeline_app.gui_qt_record.save_config"),
                patch.object(panel, "refresh_hf_datasets"),
            ):
                panel.open_dataset_upload_dialog()

            assert controller.last_complete_callback is not None
            controller.last_complete_callback(0, False)
            assert controller.last_complete_callback is not None
            controller.last_complete_callback(0, False)

            provenance = read_workspace_provenance(local_dataset)
            assert provenance is not None
            self.assertEqual(provenance["repo_id"], "alice/demo_remote")
            self.assertEqual(provenance["asset_kind"], "dataset")

    def test_record_dataset_upload_starts_tagging_after_successful_upload(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        panel = RecordOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
        self.addCleanup(panel.close)

        fake_dialog = SimpleNamespace(
            result_request={
                "repo_id": "alice/demo_remote",
                "local_dataset": Path("/tmp/demo_local"),
                "remote_exists": False,
                "upload_cmd": ["huggingface-cli", "upload", "alice/demo_remote", "/tmp/demo_local", "--repo-type", "dataset"],
                "checks": [("PASS", "Local dataset folder", "/tmp/demo_local")],
                "provenance_repo_id": "",
                "provenance_matches_target": False,
            },
            result_settings={"local_dataset": "/tmp/demo_local", "owner": "alice", "repo_name": "demo_remote"},
            exec=lambda: None,
        )

        with (
            patch("robot_pipeline_app.gui_qt_record.has_huggingface_auth_token", return_value=True),
            patch("robot_pipeline_app.gui_qt_record._QtDatasetUploadDialog", return_value=fake_dialog),
            patch.object(panel, "_confirm_preflight_review", return_value=True),
            patch("robot_pipeline_app.gui_qt_record.ask_text_dialog", return_value=True),
            patch("robot_pipeline_app.gui_qt_record.save_config"),
            patch.object(panel, "refresh_hf_datasets"),
        ):
            panel.open_dataset_upload_dialog()

        assert controller.last_complete_callback is not None
        controller.last_complete_callback(0, False)

        assert controller.last_cmd is not None
        self.assertEqual(controller.last_cmd[:3], ["huggingface-cli", "upload", "alice/demo_remote"])
        self.assertEqual(controller.last_cmd[4:], ["README.md", "--repo-type", "dataset"])

    def test_record_post_upload_starts_tagging_after_successful_upload(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        config["hf_username"] = "alice"

        with tempfile.TemporaryDirectory() as tmpdir:
            record_root = Path(tmpdir) / "record"
            lerobot_dir = Path(tmpdir) / "lerobot"
            config["record_data_dir"] = str(record_root)
            config["lerobot_dir"] = str(lerobot_dir)
            source_dataset = lerobot_dir / "data" / "demo_local"
            (source_dataset / "meta").mkdir(parents=True, exist_ok=True)
            (source_dataset / "meta" / "episodes.jsonl").write_text("{}\n", encoding="utf-8")

            with patch("robot_pipeline_app.repo_utils.dataset_exists_on_hf", return_value=False):
                panel = RecordOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
                self.addCleanup(panel.close)

            panel.dataset_input.setText("alice/demo_local")
            panel.dataset_input.textEdited.emit("alice/demo_local")  # mark as manual so _advance_dataset_name doesn't increment
            panel.dataset_root_input.setText(str(record_root))
            panel.upload_checkbox.setChecked(True)

            with (
                patch("robot_pipeline_app.gui_qt_ops_base.ask_editable_command_dialog", side_effect=lambda **kwargs: list(kwargs["command_argv"])),
                patch("robot_pipeline_app.gui_qt_ops_base.ask_text_dialog", return_value=True),
                patch("robot_pipeline_app.gui_qt_record.run_preflight_for_record", return_value=[("PASS", "Environment", "Ready.")]),
                patch.object(panel, "refresh_hf_datasets"),
                patch.object(panel, "_ensure_dataset_name_available"),
            ):
                panel.run_record()

            assert controller.last_complete_callback is not None
            controller.last_complete_callback(0, False)

            assert controller.last_cmd is not None
            self.assertEqual(controller.last_cmd[:3], ["huggingface-cli", "upload", "alice/demo_local"])
            self.assertEqual(controller.last_cmd[4:], ["--repo-type", "dataset"])

            assert controller.last_complete_callback is not None
            controller.last_complete_callback(0, False)

            assert controller.last_cmd is not None
            self.assertEqual(controller.last_cmd[:3], ["huggingface-cli", "upload", "alice/demo_local"])
            self.assertEqual(controller.last_cmd[4:], ["README.md", "--repo-type", "dataset"])

    def test_record_success_refreshes_local_dataset_browser(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        config["hf_username"] = "alice"

        with tempfile.TemporaryDirectory() as tmpdir:
            record_root = Path(tmpdir) / "record"
            lerobot_dir = Path(tmpdir) / "lerobot"
            config["record_data_dir"] = str(record_root)
            config["lerobot_dir"] = str(lerobot_dir)
            source_dataset = lerobot_dir / "data" / "demo_local"
            (source_dataset / "meta").mkdir(parents=True, exist_ok=True)
            (source_dataset / "meta" / "episodes.jsonl").write_text("{}\n", encoding="utf-8")

            with patch("robot_pipeline_app.repo_utils.dataset_exists_on_hf", return_value=False):
                panel = RecordOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
                self.addCleanup(panel.close)

            panel.dataset_input.setText("alice/demo_local")
            panel.dataset_input.textEdited.emit("alice/demo_local")  # mark as manual so _advance_dataset_name doesn't increment
            panel.dataset_root_input.setText(str(record_root))

            with (
                patch("robot_pipeline_app.gui_qt_ops_base.ask_editable_command_dialog", side_effect=lambda **kwargs: list(kwargs["command_argv"])),
                patch("robot_pipeline_app.gui_qt_ops_base.ask_text_dialog", return_value=True),
                patch("robot_pipeline_app.gui_qt_record.run_preflight_for_record", return_value=[("PASS", "Environment", "Ready.")]),
                patch.object(panel, "_ensure_dataset_name_available"),
            ):
                panel.run_record()

            assert controller.last_complete_callback is not None
            controller.last_complete_callback(0, False)

            self.assertEqual(panel.local_dataset_tree.topLevelItemCount(), 1)
            dataset_item = panel.local_dataset_tree.topLevelItem(0)
            self.assertEqual(dataset_item.text(0), "demo_local")

    def test_model_upload_dialog_uses_shared_dialog_panel(self) -> None:
        dialog = _QtModelUploadDialog(
            parent=None,
            default_local_model="",
            default_owner="alice",
            default_repo_name="demo-model",
            model_options=["/tmp/model-a"],
        )
        self.addCleanup(dialog.close)

        self.assertEqual(dialog.objectName(), "AppDialog")
        self.assertIsNotNone(dialog.findChild(QFrame, "DialogPanel"))

    def test_dataset_upload_dialog_uses_shared_dialog_panel(self) -> None:
        dialog = _QtDatasetUploadDialog(
            parent=None,
            default_local_dataset="",
            default_owner="alice",
            default_repo_name="demo-dataset",
            dataset_options=["/tmp/dataset-a"],
        )
        self.addCleanup(dialog.close)

        self.assertEqual(dialog.objectName(), "AppDialog")
        self.assertIsNotNone(dialog.findChild(QFrame, "DialogPanel"))

    def test_deploy_run_applies_eval_prefix_quick_fix_before_launch(self) -> None:
        controller = _FakeRunController()
        logs: list[str] = []
        config = dict(DEFAULT_CONFIG_VALUES)
        config["hf_username"] = "alice"
        panel = DeployOpsPanel(config=config, append_log=logs.append, run_controller=controller)
        self.addCleanup(panel.close)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "model")
            os.mkdir(model_dir)
            with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as handle:
                handle.write("{}")
            with open(os.path.join(model_dir, "model.safetensors"), "w", encoding="utf-8") as handle:
                handle.write("stub")

            panel.models_root_input.setText(tmpdir)
            panel.model_path_input.setText("model")
            panel.eval_dataset_input.setText("demo")
            panel.eval_dataset_input.textEdited.emit("demo")  # mark as manual so _advance_eval_name doesn't overwrite

            req, cmd, updated, error = panel._build()
            self.assertIsNone(error)
            assert req is not None and cmd is not None and updated is not None

            checks_side_effect = [
                [("FAIL", "Eval dataset naming", "Suggested quick fix: alice/eval_demo")],
                [("PASS", "Environment", "Ready.")],
                [("PASS", "Environment", "Ready.")],
            ]

            with (
                patch("robot_pipeline_app.gui_qt_deploy.run_preflight_for_deploy", side_effect=checks_side_effect) as mocked_checks,
                patch("robot_pipeline_app.gui_qt_ops_base.ask_text_dialog_with_actions", return_value="fix_eval_prefix") as mocked_actions,
                patch(
                    "robot_pipeline_app.gui_qt_ops_base.ask_editable_command_dialog",
                    side_effect=lambda **kwargs: list(kwargs["command_argv"]),
                ) as mocked_edit,
                patch("robot_pipeline_app.gui_qt_ops_base.ask_text_dialog", return_value=True) as mocked_confirm,
            ):
                panel.run_deploy()

        mocked_actions.assert_called_once()
        mocked_edit.assert_called_once()
        self.assertGreaterEqual(mocked_checks.call_count, 2)
        self.assertGreaterEqual(mocked_confirm.call_count, 1)
        self.assertEqual(panel.eval_dataset_input.text(), "alice/eval_demo")
        self.assertIsNotNone(controller.last_cmd)

    def test_deploy_completion_callback_resets_running_state(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        panel = DeployOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
        self.addCleanup(panel.close)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "model")
            os.mkdir(model_dir)
            with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as handle:
                handle.write("{}")
            with open(os.path.join(model_dir, "model.safetensors"), "w", encoding="utf-8") as handle:
                handle.write("stub")

            panel.models_root_input.setText(tmpdir)
            panel.model_path_input.setText("model")
            panel.eval_dataset_input.setText("alice/eval_demo")

            with (
                patch(
                    "robot_pipeline_app.gui_qt_deploy.run_preflight_for_deploy",
                    return_value=[("PASS", "Environment", "Ready.")],
                ),
                patch(
                    "robot_pipeline_app.gui_qt_ops_base.ask_editable_command_dialog",
                    side_effect=lambda **kwargs: list(kwargs["command_argv"]),
                ),
                patch("robot_pipeline_app.gui_qt_ops_base.ask_text_dialog", return_value=True),
            ):
                panel.run_deploy()

        self.assertIsNotNone(controller.last_complete_callback)
        panel._set_running(True, "Running command...", False)
        self.assertFalse(panel.run_button.isEnabled())
        self.assertTrue(panel.cancel_button.isEnabled())

        controller.last_complete_callback(0, False)

        self.assertTrue(panel.run_button.isEnabled())
        self.assertFalse(panel.cancel_button.isEnabled())
        self.assertEqual(panel.run_helper_dialog.status_chip.text(), "Deploy completed.")

    def test_deploy_model_browser_selection_updates_model_path(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        panel = DeployOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
        self.addCleanup(panel.close)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "policy_a")
            os.mkdir(model_dir)
            with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as handle:
                handle.write(
                    '{"policy_family":"sarm","policy_class":"vendor_pkg.sarm.SarmPolicy","plugin_package":"vendor_pkg","camera_keys":["front"],"output_shapes":{"action":{"shape":[6]}}}'
                )
            with open(os.path.join(model_dir, "model.safetensors"), "w", encoding="utf-8") as handle:
                handle.write("stub")

            panel.models_root_input.setText(tmpdir)
            panel.refresh_model_browser()

            self.assertGreater(panel.model_tree.topLevelItemCount(), 0)
            with patch("robot_pipeline_app.gui_qt_deploy.save_config"):
                self.assertTrue(panel._select_tree_item_for_path(Path(model_dir)))
            self.assertEqual(panel.model_path_input.text(), model_dir)
            self.assertIn("Selected:", panel.selected_model_label.text())
            self.assertIn("Policy family/class: SARM / vendor_pkg.sarm.SarmPolicy", panel.model_info.toPlainText())

    def test_deploy_model_selection_generates_eval_prefixed_auto_name(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        config["hf_username"] = "alice"

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "policy_a"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
            (model_dir / "model.safetensors").write_text("stub\n", encoding="utf-8")
            config["trained_models_dir"] = tmpdir

            with patch("robot_pipeline_app.repo_utils.dataset_exists_on_hf", return_value=False):
                panel = DeployOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
                self.addCleanup(panel.close)
                with patch("robot_pipeline_app.gui_qt_deploy.save_config"):
                    panel._apply_model_selection(model_dir)

                self.assertEqual(panel.eval_dataset_input.text(), "alice/eval_policy_a_1")

    def test_deploy_manual_eval_name_is_preserved_on_model_selection(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        config["hf_username"] = "alice"

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "policy_a"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
            (model_dir / "model.safetensors").write_text("stub\n", encoding="utf-8")
            config["trained_models_dir"] = tmpdir

            with patch("robot_pipeline_app.repo_utils.dataset_exists_on_hf", return_value=False):
                panel = DeployOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
                self.addCleanup(panel.close)
                panel._eval_name_controller.set_text("alice/eval_manual_9", mode="manual")
                with patch("robot_pipeline_app.gui_qt_deploy.save_config"):
                    panel._apply_model_selection(model_dir)

                self.assertEqual(panel.eval_dataset_input.text(), "alice/eval_manual_9")

    def test_deploy_refresh_from_config_updates_auto_eval_name(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        config["hf_username"] = "alice"
        config["last_eval_dataset_name"] = "eval_old_1"

        with patch("robot_pipeline_app.repo_utils.dataset_exists_on_hf", return_value=False):
            panel = DeployOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
            self.addCleanup(panel.close)

            config["hf_username"] = "bob"
            config["last_eval_dataset_name"] = "eval_new_3"
            panel.refresh_from_config()

            self.assertEqual(panel.eval_dataset_input.text(), "bob/eval_new_3")

    def test_deploy_refresh_from_config_preserves_manual_eval_name(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        config["hf_username"] = "alice"
        config["last_eval_dataset_name"] = "eval_old_1"

        with patch("robot_pipeline_app.repo_utils.dataset_exists_on_hf", return_value=False):
            panel = DeployOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
            self.addCleanup(panel.close)

            panel._eval_name_controller.set_text("alice/eval_manual_9", mode="manual")
            config["hf_username"] = "bob"
            config["last_eval_dataset_name"] = "eval_new_3"
            panel.refresh_from_config()

            self.assertEqual(panel.eval_dataset_input.text(), "alice/eval_manual_9")

    def test_deploy_action_row_makes_run_deploy_first_and_primary(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        panel = DeployOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
        self.addCleanup(panel.close)

        texts = [button.text() for button in panel._action_buttons]

        self.assertGreaterEqual(len(panel._action_buttons), 5)
        self.assertEqual(texts[0], "Run Deploy")
        self.assertEqual(panel._action_buttons[0].objectName(), "AccentButton")
        self.assertEqual(texts[1], "Preview Command")
        self.assertNotEqual(panel._action_buttons[1].objectName(), "AccentButton")

    def test_deploy_cancel_requests_run_cancellation(self) -> None:
        controller = _FakeRunController()
        controller.cancel_result = (True, "Cancel requested.")
        config = dict(DEFAULT_CONFIG_VALUES)
        panel = DeployOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
        self.addCleanup(panel.close)

        panel._cancel_run()

        self.assertEqual(controller.cancel_calls, 1)
        self.assertNotIn("Cancel Unavailable", panel.output.toPlainText())

    def test_replay_run_passes_dataset_context_into_artifact_payload(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        panel = ReplayOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
        self.addCleanup(panel.close)

        request = ReplayRequest(
            dataset_repo_id="alice/demo",
            dataset_path=Path("/tmp/datasets/alice/demo"),
            episode_index=4,
            robot_type="so100_follower",
            robot_port="/dev/ttyUSB0",
            robot_id="arm_follower",
            calibration_dir="/tmp/calibration",
        )
        support = ReplaySupport(
            available=True,
            entrypoint="lerobot.replay",
            detail="Replay entrypoint detected.",
            supported_flags=(),
            dataset_flag=None,
            dataset_root_flag=None,
            dataset_path_flag="dataset.path",
            episode_flag="dataset.episode",
            robot_type_flag="robot.type",
            robot_port_flag="robot.port",
            robot_id_flag="robot.id",
            calibration_dir_flag="robot.calibration_dir",
        )
        cmd = ["python3", "-m", "lerobot.replay", "--dataset.path=/tmp/datasets/alice/demo", "--dataset.episode=4"]

        with patch.object(panel, "_build", return_value=(request, cmd, support, None)), patch.object(
            panel,
            "_ask_editable_command_dialog",
            return_value=list(cmd),
        ), patch.object(
            panel,
            "_confirm_preflight_review",
            return_value=True,
        ), patch(
            "robot_pipeline_app.gui_qt_replay.save_config",
        ):
            panel.run_replay()

        assert controller.last_kwargs is not None
        self.assertEqual(controller.last_kwargs["run_mode"], "replay")
        self.assertEqual(
            controller.last_kwargs["artifact_context"],
            {
                "dataset_repo_id": "alice/demo",
                "dataset_path": "/tmp/datasets/alice/demo",
                "replay_episode": 4,
            },
        )

    def test_replay_panel_populates_discovered_episode_choices(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        panel = ReplayOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
        self.addCleanup(panel.close)

        with patch(
            "robot_pipeline_app.gui_qt_replay.discover_replay_episodes",
            return_value=type("Discovery", (), {"episode_indices": (2, 4, 9), "scan_error": None, "manual_entry_only": False})(),
        ), patch(
            "robot_pipeline_app.gui_qt_replay.build_replay_request_and_command",
            return_value=(None, None, ReplaySupport(False, "", "Replay unavailable.", (), None, None, None, None, None, None, None, None), "Replay unavailable."),
        ):
            panel.dataset_input.setText("alice/demo")
            panel._refresh_episode_state()

        self.assertEqual(panel.episode_combo.count(), 3)
        self.assertEqual(panel.episode_combo.itemText(0), "2")
        self.assertFalse(panel.episode_manual_input.isEnabled())

    def test_replay_panel_enables_manual_fallback_when_discovery_fails(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        panel = ReplayOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
        self.addCleanup(panel.close)

        with patch(
            "robot_pipeline_app.gui_qt_replay.discover_replay_episodes",
            return_value=type("Discovery", (), {"episode_indices": (), "scan_error": "episodes.jsonl missing", "manual_entry_only": True})(),
        ), patch(
            "robot_pipeline_app.gui_qt_replay.build_replay_request_and_command",
            return_value=(None, None, ReplaySupport(False, "", "Replay unavailable.", (), None, None, None, None, None, None, None, None), "Replay unavailable."),
        ):
            panel.dataset_input.setText("alice/demo")
            panel._refresh_episode_state()

        self.assertTrue(panel.episode_manual_input.isEnabled())
        self.assertIn("episodes.jsonl missing", panel.readiness_label.text())

    def test_replay_panel_uses_selected_local_dataset_and_discovers_episodes(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "demo"
            episodes_path = dataset_path / "meta" / "episodes.jsonl"
            episodes_path.parent.mkdir(parents=True)
            episodes_path.write_text("{}\n{}\n{}\n", encoding="utf-8")
            write_workspace_provenance(dataset_path, {"repo_id": "alice/demo"})
            config["record_data_dir"] = tmpdir

            panel = ReplayOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
            self.addCleanup(panel.close)

            unavailable_support = ReplaySupport(
                False,
                "",
                "Replay unavailable.",
                (),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            with patch(
                "robot_pipeline_app.gui_qt_replay.build_replay_request_and_command",
                return_value=(None, None, unavailable_support, "Replay unavailable."),
            ):
                panel.refresh_local_dataset_browser()
                item = panel.local_dataset_tree.topLevelItem(0)
                assert item is not None
                panel.local_dataset_tree.setCurrentItem(item)
                panel.use_selected_dataset_in_replay()

            self.assertEqual(panel.dataset_input.text(), "alice/demo")
            self.assertEqual(panel.dataset_path_input.text(), str(dataset_path))
            self.assertEqual(panel.episode_combo.count(), 3)
            self.assertEqual(panel.episode_combo.itemText(0), "0")
            self.assertEqual(panel.episode_combo.itemText(2), "2")

    def test_replay_panel_uses_selected_hf_dataset_without_forcing_local_path(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "demo"
            episodes_path = dataset_path / "meta" / "episodes.jsonl"
            episodes_path.parent.mkdir(parents=True)
            episodes_path.write_text("{}\n", encoding="utf-8")
            write_workspace_provenance(dataset_path, {"repo_id": "alice/demo"})
            config["record_data_dir"] = tmpdir

            panel = ReplayOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
            self.addCleanup(panel.close)

            unavailable_support = ReplaySupport(
                False,
                "",
                "Replay unavailable.",
                (),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            with patch(
                "robot_pipeline_app.gui_qt_replay.build_replay_request_and_command",
                return_value=(None, None, unavailable_support, "Replay unavailable."),
            ):
                panel.refresh_local_dataset_browser()
                item = panel.local_dataset_tree.topLevelItem(0)
                assert item is not None
                panel.local_dataset_tree.setCurrentItem(item)
                panel._mark_dataset_selection_source("local")
                panel.use_selected_dataset_in_replay()

                panel._apply_hf_dataset_rows(([{"repo_id": "alice/remote-demo", "downloads": 12, "likes": 4}], None))
                panel._mark_dataset_selection_source("hf")
                panel.hf_dataset_table.selectRow(0)
                panel.use_selected_dataset_in_replay()

        self.assertEqual(panel.dataset_input.text(), "alice/remote-demo")
        self.assertEqual(panel.dataset_path_input.text(), "")
        self.assertTrue(panel.episode_manual_input.isEnabled())

    def test_motor_setup_run_stores_motor_metadata_and_updates_config_on_success(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        panel = MotorSetupOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
        self.addCleanup(panel.close)

        request = MotorSetupRequest(
            role="leader",
            robot_type="so101_leader",
            port="/dev/ttyUSB9",
            robot_id="leader_old",
            new_id="leader_new",
            baudrate=1_000_000,
        )
        support = MotorSetupSupport(
            available=True,
            entrypoint="lerobot.setup_motors",
            detail="Motor setup entrypoint detected.",
            supported_flags=(),
            role_flag="robot.role",
            type_flag="robot.type",
            port_flag="robot.port",
            id_flag="robot.id",
            new_id_flag="robot.new_id",
            baudrate_flag="robot.baudrate",
            uses_calibrate_fallback=False,
        )
        cmd = ["python3", "-m", "lerobot.setup_motors", "--robot.port=/dev/ttyUSB9"]

        with patch.object(panel, "_build", return_value=(request, cmd, support, None)), patch.object(
            panel,
            "_ask_editable_command_dialog",
            return_value=list(cmd),
        ), patch.object(
            panel,
            "_confirm_preflight_review",
            return_value=True,
        ), patch(
            "robot_pipeline_app.gui_qt_motor_setup.save_config",
        ) as mocked_save_config:
            panel.run_motor_setup()
            assert controller.last_complete_callback is not None
            controller.last_complete_callback(0, False)

        assert controller.last_kwargs is not None
        self.assertEqual(controller.last_kwargs["run_mode"], "motor_setup")
        artifact_context = controller.last_kwargs["artifact_context"]
        assert isinstance(artifact_context, dict)
        self.assertEqual(artifact_context["motor_setup"]["role"], "leader")
        self.assertEqual(artifact_context["motor_setup"]["new_id"], "leader_new")
        self.assertEqual(config["leader_port"], "/dev/ttyUSB9")
        self.assertEqual(config["leader_robot_id"], "leader_new")
        self.assertEqual(config["leader_robot_type"], "so101_leader")
        self.assertIn("Motor id update: Applied by runtime flags.", panel.output.toPlainText())
        mocked_save_config.assert_called()

    def test_motor_setup_result_mentions_active_rig_divergence_only_when_needed(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        config["saved_rigs"] = [{"name": "Bench A", "description": "", "snapshot": {"leader_port": "/dev/ttyUSB1"}}]
        config["active_rig_name"] = "Bench A"
        panel = MotorSetupOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
        self.addCleanup(panel.close)

        request = MotorSetupRequest(
            role="leader",
            robot_type="so101_leader",
            port="/dev/ttyUSB9",
            robot_id="leader_old",
            new_id="leader_new",
            baudrate=1_000_000,
        )
        support = MotorSetupSupport(
            available=True,
            entrypoint="lerobot.setup_motors",
            detail="Motor setup entrypoint detected.",
            supported_flags=(),
            role_flag="robot.role",
            type_flag="robot.type",
            port_flag="robot.port",
            id_flag="robot.id",
            new_id_flag="robot.new_id",
            baudrate_flag="robot.baudrate",
            uses_calibrate_fallback=False,
        )
        cmd = ["python3", "-m", "lerobot.setup_motors", "--robot.port=/dev/ttyUSB9"]

        with patch.object(panel, "_build", return_value=(request, cmd, support, None)), patch.object(
            panel,
            "_ask_editable_command_dialog",
            return_value=list(cmd),
        ), patch.object(
            panel,
            "_confirm_preflight_review",
            return_value=True,
        ), patch(
            "robot_pipeline_app.gui_qt_motor_setup.save_config",
        ):
            panel.run_motor_setup()
            assert controller.last_complete_callback is not None
            controller.last_complete_callback(0, False)

        self.assertIn("Active rig 'Bench A' now differs from its saved snapshot", panel.output.toPlainText())

    def test_teleop_scan_ports_updates_visible_fields(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        panel = TeleopOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
        self.addCleanup(panel.close)

        scan_entries = [
            {
                "path": "/dev/cu.usbmodem1",
                "by_id": [],
                "readable": True,
                "writable": True,
                "busy": False,
                "busy_detail": "",
                "manufacturer": "",
                "product": "",
                "likely_motor_controller": True,
            }
        ]
        with (
            patch("robot_pipeline_app.gui_qt_ops_base.scan_robot_serial_ports", return_value=scan_entries),
            patch(
                "robot_pipeline_app.gui_qt_ops_base.suggest_follower_leader_ports",
                return_value=("/dev/cu.usbmodem2", "/dev/cu.usbmodem1"),
            ),
            patch("robot_pipeline_app.gui_qt_ops_base.ask_text_dialog_with_actions", return_value="apply_ports"),
            patch("robot_pipeline_app.gui_qt_teleop.save_config"),
        ):
            panel.scan_robot_ports()

        self.assertEqual(panel.follower_port_input.text(), "/dev/cu.usbmodem2")
        self.assertEqual(panel.leader_port_input.text(), "/dev/cu.usbmodem1")

    def test_teleop_action_row_makes_run_teleop_first_and_primary(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        panel = TeleopOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
        self.addCleanup(panel.close)

        texts = [button.text() for button in panel._action_buttons]

        self.assertGreaterEqual(len(panel._action_buttons), 5)
        self.assertEqual(texts[0], "Run Teleop")
        self.assertEqual(panel._action_buttons[0].objectName(), "AccentButton")
        self.assertEqual(texts[1], "Preview Command")
        self.assertNotEqual(panel._action_buttons[1].objectName(), "AccentButton")

    def test_teleop_helper_hides_episode_step_controls(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        panel = TeleopOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
        self.addCleanup(panel.close)

        self.assertFalse(hasattr(panel, "reset_episode_button"))
        self.assertFalse(hasattr(panel, "next_episode_button"))
        self.assertIsNone(panel.run_helper_dialog.reset_episode_button)
        self.assertIsNone(panel.run_helper_dialog.next_episode_button)

    def test_teleop_helper_uses_runtime_log_view_instead_of_episode_tracker(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        panel = TeleopOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
        self.addCleanup(panel.close)

        dialog = panel.run_helper_dialog
        dialog.start_run(run_mode="teleop")
        dialog.handle_output_line("Teleop running and connected")

        self.assertEqual(dialog.cancel_button.text(), "End Teleop")
        self.assertTrue(dialog.outcomes_wrap.isHidden())
        self.assertTrue(dialog.outcome_table.isHidden())
        self.assertIn("Teleop running and connected", dialog.runtime_log_output.toPlainText())
        self.assertEqual(dialog.outcome_table.rowCount(), 0)
        self.assertTrue(dialog.elapsed_label.text().startswith("Elapsed: "))

    def test_teleop_panel_exposes_snapshot_and_camera_preview(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        panel = TeleopOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
        self.addCleanup(panel.close)

        panel.follower_port_input.setText("/dev/follower")
        panel.leader_port_input.setText("/dev/leader")
        panel.follower_id_input.setText("red4")
        panel.leader_id_input.setText("white")
        panel.control_fps_input.setText("30")

        self.assertTrue(hasattr(panel, "camera_preview"))
        self.assertIn("/dev/follower", panel.connection_summary_label.text())
        self.assertIn("/dev/leader", panel.connection_summary_label.text())
        self.assertIn("Control FPS: 30", panel.command_summary_label.text())

    def test_core_ops_cards_keep_vertical_layout_snug(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        panel = RecordOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
        self.addCleanup(panel.close)

        layout = panel.layout()
        self.assertIsNotNone(layout)
        assert layout is not None

        self.assertEqual(panel.form_card.sizePolicy().verticalPolicy(), QSizePolicy.Policy.Maximum)
        self.assertEqual(panel.output_card.sizePolicy().verticalPolicy(), QSizePolicy.Policy.Maximum)
        self.assertEqual(panel.camera_preview.sizePolicy().verticalPolicy(), QSizePolicy.Policy.Maximum)
        self.assertIsNotNone(layout.itemAt(layout.count() - 1).spacerItem())


if __name__ == "__main__":
    unittest.main()
