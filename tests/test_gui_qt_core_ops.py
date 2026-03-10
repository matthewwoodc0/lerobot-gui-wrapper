from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robot_pipeline_app.config_store import DEFAULT_CONFIG_VALUES

try:
    from PySide6.QtWidgets import QSizePolicy
    from robot_pipeline_app.gui_qt_app import ensure_qt_application, qt_available
    from robot_pipeline_app.gui_qt_core_ops import DeployOpsPanel, RecordOpsPanel, TeleopOpsPanel
except Exception as exc:  # pragma: no cover - exercised only when Qt imports fail
    ensure_qt_application = None  # type: ignore[assignment]
    DeployOpsPanel = None  # type: ignore[assignment]
    RecordOpsPanel = None  # type: ignore[assignment]
    TeleopOpsPanel = None  # type: ignore[assignment]
    QSizePolicy = None  # type: ignore[assignment]
    _QT_AVAILABLE, _QT_REASON = False, str(exc)
else:
    _QT_AVAILABLE, _QT_REASON = qt_available()


class _FakeRunController:
    def __init__(self) -> None:
        self.last_cmd: list[str] | None = None
        self.last_cwd = None
        self.cancel_calls = 0
        self.cancel_result: tuple[bool, str] = (False, "No active run.")

    def cancel_active_run(self) -> tuple[bool, str]:
        self.cancel_calls += 1
        return self.cancel_result

    def run_process_async(self, *, cmd, cwd, hooks, complete_callback, **kwargs):  # type: ignore[no-untyped-def]
        self.last_cmd = list(cmd)
        self.last_cwd = cwd
        _ = (hooks, complete_callback, kwargs)
        return True, ""


@unittest.skipUnless(_QT_AVAILABLE, _QT_REASON or "PySide6 unavailable")
class GuiQtCoreOpsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.app, _ = ensure_qt_application(["robot_pipeline.py", "gui"])

    def test_record_preview_opens_modal_dialog(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        panel = RecordOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
        self.addCleanup(panel.close)

        self.assertTrue(panel.output_card.isHidden())

        with tempfile.TemporaryDirectory() as tmpdir:
            panel.dataset_input.setText("alice/demo")
            panel.dataset_root_input.setText(tmpdir)

            with patch("robot_pipeline_app.gui_qt_core_ops.show_text_dialog") as mocked_dialog:
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
                patch("robot_pipeline_app.gui_qt_core_ops.ask_editable_command_dialog", return_value=list(cmd)) as mocked_edit,
                patch("robot_pipeline_app.gui_qt_core_ops.ask_text_dialog", return_value=True) as mocked_preflight,
                patch(
                    "robot_pipeline_app.gui_qt_core_ops.run_preflight_for_record",
                    return_value=[("PASS", "Environment", "Ready.")],
                ) as mocked_checks,
            ):
                panel.run_record()

        mocked_edit.assert_called_once()
        mocked_preflight.assert_called_once()
        mocked_checks.assert_called_once()
        self.assertEqual(controller.last_cmd, cmd)

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
            patch("robot_pipeline_app.gui_qt_core_ops.scan_robot_serial_ports", return_value=scan_entries),
            patch(
                "robot_pipeline_app.gui_qt_core_ops.suggest_follower_leader_ports",
                return_value=("/dev/cu.usbmodem2", "/dev/cu.usbmodem1"),
            ),
            patch("robot_pipeline_app.gui_qt_core_ops.ask_text_dialog_with_actions", return_value="apply_ports"),
            patch("robot_pipeline_app.gui_qt_core_ops.save_config"),
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
                patch("robot_pipeline_app.gui_qt_core_ops.run_preflight_for_deploy", side_effect=checks_side_effect) as mocked_checks,
                patch("robot_pipeline_app.gui_qt_core_ops.ask_text_dialog_with_actions", return_value="fix_eval_prefix") as mocked_actions,
                patch(
                    "robot_pipeline_app.gui_qt_core_ops.ask_editable_command_dialog",
                    side_effect=lambda **kwargs: list(kwargs["command_argv"]),
                ) as mocked_edit,
                patch("robot_pipeline_app.gui_qt_core_ops.ask_text_dialog", return_value=True) as mocked_confirm,
            ):
                panel.run_deploy()

        mocked_actions.assert_called_once()
        mocked_edit.assert_called_once()
        self.assertGreaterEqual(mocked_checks.call_count, 2)
        self.assertGreaterEqual(mocked_confirm.call_count, 1)
        self.assertEqual(panel.eval_dataset_input.text(), "alice/eval_demo")
        self.assertIsNotNone(controller.last_cmd)

    def test_deploy_model_browser_selection_updates_model_path(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        panel = DeployOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
        self.addCleanup(panel.close)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "policy_a")
            os.mkdir(model_dir)
            with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as handle:
                handle.write("{}")
            with open(os.path.join(model_dir, "model.safetensors"), "w", encoding="utf-8") as handle:
                handle.write("stub")

            panel.models_root_input.setText(tmpdir)
            panel.refresh_model_browser()

            self.assertGreater(panel.model_tree.topLevelItemCount(), 0)
            with patch("robot_pipeline_app.gui_qt_core_ops.save_config"):
                self.assertTrue(panel._select_tree_item_for_path(Path(model_dir)))
            self.assertEqual(panel.model_path_input.text(), model_dir)
            self.assertIn("Selected:", panel.selected_model_label.text())

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
            patch("robot_pipeline_app.gui_qt_core_ops.scan_robot_serial_ports", return_value=scan_entries),
            patch(
                "robot_pipeline_app.gui_qt_core_ops.suggest_follower_leader_ports",
                return_value=("/dev/cu.usbmodem2", "/dev/cu.usbmodem1"),
            ),
            patch("robot_pipeline_app.gui_qt_core_ops.ask_text_dialog_with_actions", return_value="apply_ports"),
            patch("robot_pipeline_app.gui_qt_core_ops.save_config"),
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
