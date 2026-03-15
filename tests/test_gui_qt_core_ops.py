from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robot_pipeline_app.config_store import DEFAULT_CONFIG_VALUES
from robot_pipeline_app.hardware_workflows import MotorSetupRequest, MotorSetupSupport, ReplayRequest, ReplaySupport

try:
    from PySide6.QtWidgets import QFrame, QSizePolicy
    from robot_pipeline_app.gui_qt_app import ensure_qt_application, qt_available
    from robot_pipeline_app.gui_qt_core_ops import (
        DeployOpsPanel,
        MotorSetupOpsPanel,
        RecordOpsPanel,
        ReplayOpsPanel,
        TeleopOpsPanel,
        _QtModelUploadDialog,
    )
except Exception as exc:  # pragma: no cover - exercised only when Qt imports fail
    ensure_qt_application = None  # type: ignore[assignment]
    DeployOpsPanel = None  # type: ignore[assignment]
    MotorSetupOpsPanel = None  # type: ignore[assignment]
    RecordOpsPanel = None  # type: ignore[assignment]
    ReplayOpsPanel = None  # type: ignore[assignment]
    TeleopOpsPanel = None  # type: ignore[assignment]
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
        self.assertIsNone(panel.run_helper_dialog.reset_button)
        self.assertIsNone(panel.run_helper_dialog.next_button)

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
