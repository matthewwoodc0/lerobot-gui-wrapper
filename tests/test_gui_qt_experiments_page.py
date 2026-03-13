from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robot_pipeline_app.constants import DEFAULT_CONFIG_VALUES
from robot_pipeline_app.experiments_service import discover_checkpoint_artifacts

try:
    from robot_pipeline_app.gui_qt_app import ensure_qt_application, qt_available
    from robot_pipeline_app.gui_qt_experiments_page import QtExperimentsPage
    from robot_pipeline_app.qt_bootstrap import probe_qt_platform_support
except Exception as exc:  # pragma: no cover - exercised only when Qt imports fail
    ensure_qt_application = None  # type: ignore[assignment]
    QtExperimentsPage = None  # type: ignore[assignment]
    _QT_AVAILABLE, _QT_REASON = False, str(exc)
else:
    _qt_ok, _qt_reason = qt_available()
    if not _qt_ok:
        _QT_AVAILABLE, _QT_REASON = False, _qt_reason or "PySide6 unavailable"
    else:
        _QT_AVAILABLE, _QT_REASON = probe_qt_platform_support(platform_name="offscreen")


class _FakeRunController:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, object] | None = None

    def cancel_active_run(self) -> tuple[bool, str]:
        return False, "No active run."

    def run_process_async(self, **kwargs):  # type: ignore[no-untyped-def]
        self.last_kwargs = dict(kwargs)
        return True, ""


@unittest.skipUnless(_QT_AVAILABLE, _QT_REASON or "PySide6 unavailable")
class GuiQtExperimentsPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            cls.app, _ = ensure_qt_application(["robot_pipeline.py", "gui"])
        except RuntimeError as exc:
            raise unittest.SkipTest(str(exc)) from exc

    def test_checkpoint_to_deploy_handoff_runs_through_shared_controller(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir) / "runs"
            train_run = runs_dir / "train_20260312_130000"
            train_run.mkdir(parents=True, exist_ok=True)
            output_dir = Path(tmpdir) / "outputs" / "train_1"
            payload = output_dir / "checkpoints" / "checkpoint-010000" / "pretrained_model"
            payload.mkdir(parents=True, exist_ok=True)
            (payload / "config.json").write_text("{}\n", encoding="utf-8")
            (payload / "model.safetensors").write_text("weights\n", encoding="utf-8")
            (payload.parent / "train_config.json").write_text("{}\n", encoding="utf-8")
            metadata = {
                "run_id": train_run.name,
                "mode": "train",
                "status": "success",
                "started_at_iso": "2026-03-12T13:00:00+00:00",
                "duration_s": 60.0,
                "command": "python -m lerobot.train",
                "command_argv": ["python", "-m", "lerobot.train", "--policy.type=act"],
                "dataset_repo_id": "alice/demo_train",
                "policy_type": "act",
                "output_dir": str(output_dir),
                "output_dir_resolved": str(output_dir),
                "checkpoint_artifacts": discover_checkpoint_artifacts(output_dir),
                "train_metrics": {"step": 10000, "loss": 0.2},
                "wandb": {"enabled": False},
            }
            (train_run / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            (train_run / "command.log").write_text("step: 10000 loss: 0.2\n", encoding="utf-8")

            config = dict(DEFAULT_CONFIG_VALUES)
            config["runs_dir"] = str(runs_dir)
            config["lerobot_dir"] = tmpdir
            config["compat_probe_enabled"] = False
            controller = _FakeRunController()

            with patch("robot_pipeline_app.gui_qt_experiments_page.save_config"), patch(
                "robot_pipeline_app.gui_qt_experiments_page.run_preflight_for_deploy",
                return_value=[],
            ):
                page = QtExperimentsPage(config=config, append_log=lambda _msg: None, run_controller=controller)
                self.addCleanup(page.close)
                page.run_table.selectRow(0)
                self.app.processEvents()
                page.checkpoint_table.selectRow(0)
                self.app.processEvents()
                page.deploy_dataset_input.setText("alice/eval_demo")
                page.deploy_episodes_input.setText("4")
                page.deploy_duration_input.setText("20")
                page.deploy_task_input.setText("pick and place")
                page.launch_deploy_from_checkpoint()

        assert controller.last_kwargs is not None
        self.assertEqual(controller.last_kwargs["run_mode"], "deploy")
        self.assertTrue(any(str(payload) in str(part) for part in controller.last_kwargs["cmd"]))
        self.assertEqual(controller.last_kwargs["artifact_context"]["model_path"], str(payload))

    def test_sim_eval_checkpoint_controls_keep_expected_defaults(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["ui_sim_eval_env_type"] = "pusht"
        config["ui_sim_eval_output_dir"] = "outputs/eval"
        controller = _FakeRunController()

        with patch("robot_pipeline_app.gui_qt_experiments_page.save_config"):
            page = QtExperimentsPage(config=config, append_log=lambda _msg: None, run_controller=controller)
            self.addCleanup(page.close)

        self.assertEqual(page.sim_env_type_input.text(), "pusht")
        self.assertEqual(page.sim_output_dir_input.text(), "outputs/eval")
        self.assertEqual(page.sim_custom_args_input.placeholderText(), "optional extra flags")
        self.assertEqual(page.sim_eval_checkpoint_button.text(), "Launch Sim Eval")
        self.assertEqual(page.sim_eval_checkpoint_button.objectName(), "AccentButton")


if __name__ == "__main__":
    unittest.main()
