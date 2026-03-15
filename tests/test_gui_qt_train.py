from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from robot_pipeline_app.checks import run_preflight_for_train
from robot_pipeline_app.commands import build_lerobot_train_command
from robot_pipeline_app.constants import DEFAULT_CONFIG_VALUES
from robot_pipeline_app.gui_forms import build_train_request_and_command

try:
    from robot_pipeline_app.gui_qt_app import ensure_qt_application, qt_available
    from robot_pipeline_app.qt_bootstrap import probe_qt_platform_support
    from robot_pipeline_app.gui_qt_train import TrainOpsPanel
except Exception as exc:  # pragma: no cover - exercised only when Qt imports fail
    ensure_qt_application = None  # type: ignore[assignment]
    TrainOpsPanel = None  # type: ignore[assignment]
    _QT_AVAILABLE, _QT_REASON = False, str(exc)
else:
    _qt_ok, _qt_reason = qt_available()
    if not _qt_ok:
        _QT_AVAILABLE, _QT_REASON = False, _qt_reason or "PySide6 unavailable"
    else:
        _QT_AVAILABLE, _QT_REASON = probe_qt_platform_support(platform_name="offscreen")


class _FakeRunController:
    def __init__(self) -> None:
        self.last_cmd: list[str] | None = None

    def cancel_active_run(self) -> tuple[bool, str]:
        return False, "No active run."

    def run_process_async(self, *, cmd, cwd, hooks, complete_callback, **kwargs):  # type: ignore[no-untyped-def]
        self.last_cmd = list(cmd)
        _ = (cwd, hooks, complete_callback, kwargs)
        return True, ""


class TrainCommandBuilderTests(unittest.TestCase):
    def test_build_lerobot_train_command_includes_requested_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_dir = Path(tmpdir) / "lerobot_env"
            python_path = venv_dir / "bin" / "python3"
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.write_text("", encoding="utf-8")

            config = dict(DEFAULT_CONFIG_VALUES)
            config["lerobot_venv_dir"] = str(venv_dir)

            fake_capabilities = SimpleNamespace(
                train_resume_path_flag="config_path",
                train_resume_toggle_flag="resume",
                train_resume_detail="Checkpoint resume is supported via --resume and --config_path.",
            )

            with patch("robot_pipeline_app.commands.resolve_train_entrypoint", return_value="lerobot.scripts.lerobot_train"), patch(
                "robot_pipeline_app.commands.probe_lerobot_capabilities",
                return_value=fake_capabilities,
            ):
                cmd = build_lerobot_train_command(
                    config,
                    {
                        "dataset_repo_id": "alice/demo_train",
                        "policy_type": "diffusion",
                        "output_dir": "/tmp/train_outputs",
                        "device": "cuda",
                        "dataset_episodes": "1,2,3",
                        "wandb_enabled": True,
                        "wandb_project": "research",
                        "job_name": "nightly-train",
                        "resume_from": "/tmp/checkpoints/last",
                    },
                )

        self.assertEqual(cmd[0], str(python_path))
        self.assertEqual(cmd[1:3], ["-m", "lerobot.scripts.lerobot_train"])
        self.assertIn("--dataset.repo_id=alice/demo_train", cmd)
        self.assertIn("--policy.type=diffusion", cmd)
        self.assertIn("--output_dir=/tmp/train_outputs", cmd)
        self.assertIn("--policy.device=cuda", cmd)
        self.assertIn("--dataset.episodes=1,2,3", cmd)
        self.assertIn("--wandb.enable=true", cmd)
        self.assertIn("--wandb.project=research", cmd)
        self.assertIn("--job_name=nightly-train", cmd)
        self.assertIn("--resume=true", cmd)
        self.assertIn("--config_path=/tmp/checkpoints/last", cmd)

    def test_build_train_request_and_command_rejects_empty_dataset(self) -> None:
        req, cmd, error = build_train_request_and_command(
            form_values={"dataset_repo_id": "", "policy_type": "act"},
            config=dict(DEFAULT_CONFIG_VALUES),
        )

        self.assertIsNone(req)
        self.assertIsNone(cmd)
        self.assertEqual(error, "Dataset name is required.")

    def test_build_train_request_and_command_rejects_resume_when_runtime_lacks_path_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            checkpoint_path.write_text("checkpoint\n", encoding="utf-8")
            fake_capabilities = SimpleNamespace(
                supports_train_resume=False,
                train_resume_path_flag=None,
                train_resume_toggle_flag=None,
                train_resume_detail="Detected train entrypoint does not expose a checkpoint/config resume path flag.",
            )
            with patch("robot_pipeline_app.gui_forms.probe_lerobot_capabilities", return_value=fake_capabilities):
                req, cmd, error = build_train_request_and_command(
                    form_values={
                        "dataset_repo_id": "alice/demo_train",
                        "policy_type": "act",
                        "resume_from": str(checkpoint_path),
                    },
                    config=dict(DEFAULT_CONFIG_VALUES),
                )

        self.assertIsNone(req)
        self.assertIsNone(cmd)
        self.assertIsNotNone(error)

    def test_build_train_request_and_command_resolves_checkpoint_folder_to_train_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints" / "last"
            config_path = checkpoint_dir / "train_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text("{}\n", encoding="utf-8")
            fake_capabilities = SimpleNamespace(
                supports_train_resume=True,
                train_resume_path_flag="config_path",
                train_resume_toggle_flag="resume",
                train_resume_detail="Checkpoint resume is supported via --resume and --config_path.",
            )
            with patch("robot_pipeline_app.gui_forms.probe_lerobot_capabilities", return_value=fake_capabilities), patch(
                "robot_pipeline_app.commands.probe_lerobot_capabilities",
                return_value=fake_capabilities,
            ):
                req, cmd, error = build_train_request_and_command(
                    form_values={
                        "dataset_repo_id": "alice/demo_train",
                        "policy_type": "act",
                        "resume_from": str(checkpoint_dir),
                    },
                    config=dict(DEFAULT_CONFIG_VALUES),
                )

        self.assertIsNone(error)
        assert req is not None and cmd is not None
        self.assertEqual(req.resume_from, str(config_path))
        self.assertIn(f"--config_path={config_path}", cmd)

    def test_run_preflight_for_train_returns_standard_check_tuples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            lerobot_dir = base / "lerobot"
            venv_dir = base / "lerobot_env"
            dataset_dir = base / "datasets" / "demo_train"
            output_dir = base / "outputs" / "train_run"
            checkpoint_path = base / "checkpoints" / "last.ckpt"

            (lerobot_dir / "scripts").mkdir(parents=True, exist_ok=True)
            (lerobot_dir / "scripts" / "lerobot_train.py").write_text("print('train')\n", encoding="utf-8")
            (venv_dir / "bin").mkdir(parents=True, exist_ok=True)
            (venv_dir / "bin" / "activate").write_text("", encoding="utf-8")
            (venv_dir / "bin" / "python3").write_text("", encoding="utf-8")
            dataset_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text("checkpoint\n", encoding="utf-8")

            config = dict(DEFAULT_CONFIG_VALUES)
            config.update(
                {
                    "lerobot_dir": str(lerobot_dir),
                    "lerobot_venv_dir": str(venv_dir),
                    "hf_username": "alice",
                }
            )
            fake_capabilities = SimpleNamespace(
                supports_train_resume=True,
                train_resume_path_flag="config_path",
                train_resume_toggle_flag="resume",
                train_resume_detail="Checkpoint resume is supported via --resume and --config_path.",
            )

            with (
                patch("robot_pipeline_app.checks_train.resolve_train_entrypoint", return_value="scripts.lerobot_train"),
                patch("robot_pipeline_app.checks.probe_module_import", return_value=(True, "")),
                patch("robot_pipeline_app.checks._probe_torch_accelerator", return_value=("cuda", "CUDA available")),
                patch("robot_pipeline_app.checks_train.probe_lerobot_capabilities", return_value=fake_capabilities),
                patch("robot_pipeline_app.checks_train.os.access", return_value=True),
            ):
                checks = run_preflight_for_train(
                    config,
                    {
                        "dataset_repo_id": str(dataset_dir),
                        "policy_type": "act",
                        "output_dir": str(output_dir),
                        "device": "cuda",
                        "resume_from": str(checkpoint_path),
                    },
                )

        self.assertTrue(checks)
        self.assertTrue(all(isinstance(item, tuple) and len(item) == 3 for item in checks))
        levels_by_name = {name: level for level, name, _detail in checks}
        self.assertEqual(levels_by_name["Environment activation"], "PASS")
        self.assertEqual(levels_by_name["LeRobot venv folder"], "PASS")
        self.assertEqual(levels_by_name["Python module: lerobot"], "PASS")
        self.assertEqual(levels_by_name["Train entrypoint"], "PASS")
        self.assertEqual(levels_by_name["Dataset"], "PASS")
        self.assertEqual(levels_by_name["Output directory"], "PASS")
        self.assertEqual(levels_by_name["Training device"], "PASS")
        self.assertEqual(levels_by_name["Resume checkpoint/config"], "PASS")
        self.assertEqual(levels_by_name["Resume support"], "PASS")
        self.assertEqual(levels_by_name["Policy type"], "PASS")


@unittest.skipUnless(_QT_AVAILABLE, _QT_REASON or "PySide6 unavailable")
class TrainOpsPanelQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            cls.app, _ = ensure_qt_application(["robot_pipeline.py", "gui"])
        except RuntimeError as exc:
            raise unittest.SkipTest(str(exc)) from exc

    def test_train_panel_can_be_instantiated(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        config["last_dataset_repo_id"] = "alice/demo_train"
        config["last_train_policy_type"] = "diffusion"

        panel = TrainOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
        self.addCleanup(panel.close)

        self.assertEqual(panel.dataset_input.text(), "alice/demo_train")
        self.assertEqual(panel.policy_type_combo.currentText(), "diffusion")
        self.assertEqual(panel.output_dir_input.text(), str(config["trained_models_dir"]))
        self.assertFalse(panel.wandb_checkbox.isChecked())
        self.assertTrue(panel.output_card.isHidden())
        self.assertEqual([button.text() for button in panel._action_buttons], ["Run Preflight", "Start Training", "Cancel"])

    def test_train_panel_auto_generates_monotonic_job_name(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        config["last_dataset_repo_id"] = "alice/demo-train"
        config["last_train_policy_type"] = "diffusion"
        config["last_train_job_name"] = "demo_train_diffusion_2"

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "demo_train_diffusion_2"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "checkpoint.txt").write_text("occupied\n", encoding="utf-8")
            config["trained_models_dir"] = tmpdir

            with patch("robot_pipeline_app.repo_utils.model_exists_on_hf", return_value=False):
                panel = TrainOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
                self.addCleanup(panel.close)

                self.assertEqual(panel.job_name_input.text(), "demo_train_diffusion_3")

    def test_train_manual_job_name_is_preserved_when_dependencies_change(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        config["last_dataset_repo_id"] = "alice/demo-train"
        config["last_train_policy_type"] = "act"

        with patch("robot_pipeline_app.repo_utils.model_exists_on_hf", return_value=False):
            panel = TrainOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
            self.addCleanup(panel.close)

            panel._job_name_controller.set_text("custom_run_9", mode="manual")
            panel.policy_type_combo.setCurrentText("diffusion")
            panel._sync_job_name_from_dependencies()

            self.assertEqual(panel.job_name_input.text(), "custom_run_9")

    def test_train_auto_job_name_reseeds_when_policy_changes(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        config["last_dataset_repo_id"] = "alice/demo-train"
        config["last_train_policy_type"] = "act"
        config["last_train_job_name"] = "demo_train_act_1"

        with patch("robot_pipeline_app.repo_utils.model_exists_on_hf", return_value=False):
            panel = TrainOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
            self.addCleanup(panel.close)

            self.assertEqual(panel.job_name_input.text(), "demo_train_act_1")
            panel.policy_type_combo.setCurrentText("diffusion")
            panel._sync_job_name_from_dependencies()

            self.assertEqual(panel.job_name_input.text(), "demo_train_diffusion_1")

    def test_train_refresh_from_config_preserves_in_progress_inputs(self) -> None:
        controller = _FakeRunController()
        config = dict(DEFAULT_CONFIG_VALUES)
        config["last_dataset_repo_id"] = "alice/demo-train"
        config["last_train_policy_type"] = "act"
        config["trained_models_dir"] = "outputs/train"

        with patch("robot_pipeline_app.repo_utils.model_exists_on_hf", return_value=False):
            panel = TrainOpsPanel(config=config, append_log=lambda _msg: None, run_controller=controller)
            self.addCleanup(panel.close)

            panel.dataset_input.setText("alice/custom-dataset")
            panel.policy_type_combo.setCurrentText("diffusion")
            panel.output_dir_input.setText("/tmp/custom-train")

            config["last_dataset_repo_id"] = "alice/config-dataset"
            config["last_train_policy_type"] = "tdmpc"
            config["trained_models_dir"] = "/tmp/from-config"
            panel.refresh_from_config()

            self.assertEqual(panel.dataset_input.text(), "alice/custom-dataset")
            self.assertEqual(panel.policy_type_combo.currentText(), "diffusion")
            self.assertEqual(panel.output_dir_input.text(), "/tmp/custom-train")


if __name__ == "__main__":
    unittest.main()
