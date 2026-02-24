from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robot_pipeline_app.checks import (
    _run_common_preflight_checks,
    collect_doctor_checks,
    run_preflight_for_deploy,
    run_preflight_for_record,
)
from robot_pipeline_app.constants import DEFAULT_CONFIG_VALUES


class ChecksDoctorTest(unittest.TestCase):
    def test_collect_doctor_checks_warns_on_next_dataset_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            lerobot_dir = base / "lerobot"
            record_dir = base / "record_data"
            models_dir = base / "trained_models"
            runs_dir = base / "runs"

            (lerobot_dir / "data").mkdir(parents=True, exist_ok=True)
            record_dir.mkdir(parents=True, exist_ok=True)
            models_dir.mkdir(parents=True, exist_ok=True)
            runs_dir.mkdir(parents=True, exist_ok=True)

            (record_dir / "demo_2").mkdir(parents=True, exist_ok=True)
            (lerobot_dir / "data" / "eval_run_2").mkdir(parents=True, exist_ok=True)

            config = dict(DEFAULT_CONFIG_VALUES)
            config.update(
                {
                    "lerobot_dir": str(lerobot_dir),
                    "record_data_dir": str(record_dir),
                    "trained_models_dir": str(models_dir),
                    "runs_dir": str(runs_dir),
                    "last_dataset_name": "demo_1",
                    "last_eval_dataset_name": "eval_run_1",
                }
            )

            with patch("robot_pipeline_app.checks.probe_module_import", return_value=(True, "")), patch(
                "robot_pipeline_app.checks.probe_camera_capture",
                return_value=(True, "frame=640x480"),
            ), patch(
                "robot_pipeline_app.checks.os.access",
                return_value=True,
            ), patch(
                "robot_pipeline_app.checks.serial_port_fingerprint",
                return_value="serial_fp",
            ), patch(
                "robot_pipeline_app.checks.camera_fingerprint",
                return_value="camera_fp",
            ), patch(
                "robot_pipeline_app.checks._serial_lock_check",
                return_value=("PASS", "Serial port lock", "ok"),
            ):
                checks = collect_doctor_checks(config)

        self.assertTrue(any(level == "WARN" and name == "Next record dataset collision" for level, name, _ in checks))
        self.assertTrue(any(level == "WARN" and name == "Next eval dataset collision" for level, name, _ in checks))

    def test_collect_doctor_checks_marks_missing_ports_and_cameras_as_fail(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["follower_port"] = ""
        config["leader_port"] = ""

        with patch("robot_pipeline_app.checks.get_lerobot_dir", return_value=Path("/tmp")), patch(
            "robot_pipeline_app.checks.Path.exists",
            return_value=True,
        ), patch(
            "robot_pipeline_app.checks.probe_module_import",
            return_value=(True, ""),
        ), patch(
            "robot_pipeline_app.checks.probe_camera_capture",
            return_value=(False, "camera not opened"),
        ), patch(
            "robot_pipeline_app.checks.camera_fingerprint",
            return_value=None,
        ), patch(
            "robot_pipeline_app.checks.serial_port_fingerprint",
            return_value=None,
        ), patch(
            "robot_pipeline_app.checks._serial_lock_check",
            return_value=("WARN", "Serial port lock", "skipped"),
        ):
            checks = collect_doctor_checks(config)

        self.assertTrue(any(level == "FAIL" and name == "Follower port" for level, name, _ in checks))
        self.assertTrue(any(level == "FAIL" and "Camera" in name and "probe" in name for level, name, _ in checks))

    def test_common_preflight_warns_when_camera_resolution_mismatch_detected(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["camera_laptop_index"] = 0
        config["camera_phone_index"] = 6

        with patch("robot_pipeline_app.checks.get_lerobot_dir", return_value=Path("/tmp")), patch(
            "robot_pipeline_app.checks.Path.exists",
            return_value=True,
        ), patch(
            "robot_pipeline_app.checks.probe_module_import",
            return_value=(True, ""),
        ), patch(
            "robot_pipeline_app.checks.probe_camera_capture",
            side_effect=[(True, "frame=640x480"), (True, "frame=640x360")],
        ), patch(
            "robot_pipeline_app.checks.os.access",
            return_value=True,
        ), patch(
            "robot_pipeline_app.checks.serial_port_fingerprint",
            side_effect=["follower", "leader"],
        ), patch(
            "robot_pipeline_app.checks.camera_fingerprint",
            side_effect=["camera_laptop", "camera_phone"],
        ), patch(
            "robot_pipeline_app.checks._serial_lock_check",
            return_value=("PASS", "Serial port lock", "ok"),
        ):
            checks = _run_common_preflight_checks(config)

        self.assertTrue(
            any(
                level == "WARN" and name == "Laptop camera resolution" and "detected=640x480" in detail
                for level, name, detail in checks
            )
        )

    def test_common_preflight_fails_when_ports_are_identical(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["follower_port"] = "/dev/ttyACM0"
        config["leader_port"] = "/dev/ttyACM0"

        with patch("robot_pipeline_app.checks.get_lerobot_dir", return_value=Path("/tmp")), patch(
            "robot_pipeline_app.checks.Path.exists",
            return_value=True,
        ), patch(
            "robot_pipeline_app.checks.probe_module_import",
            return_value=(True, ""),
        ), patch(
            "robot_pipeline_app.checks.probe_camera_capture",
            return_value=(True, "frame=640x480"),
        ), patch(
            "robot_pipeline_app.checks.os.access",
            return_value=True,
        ), patch(
            "robot_pipeline_app.checks.serial_port_fingerprint",
            return_value="same_port_fp",
        ), patch(
            "robot_pipeline_app.checks.camera_fingerprint",
            side_effect=["camera_laptop", "camera_phone"],
        ), patch(
            "robot_pipeline_app.checks._serial_lock_check",
            return_value=("PASS", "Serial port lock", "ok"),
        ):
            checks = _run_common_preflight_checks(config)

        self.assertTrue(
            any(level == "FAIL" and name == "Leader/Follower port uniqueness" for level, name, _ in checks)
        )
        self.assertTrue(any(level == "FAIL" and name == "Leader/Follower port identity" for level, name, _ in checks))

    def test_common_preflight_fails_when_dialout_membership_missing(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["follower_port"] = "/dev/ttyACM0"
        config["leader_port"] = "/dev/ttyACM1"

        with patch("robot_pipeline_app.checks.get_lerobot_dir", return_value=Path("/tmp")), patch(
            "robot_pipeline_app.checks.Path.exists",
            return_value=True,
        ), patch(
            "robot_pipeline_app.checks.probe_module_import",
            return_value=(True, ""),
        ), patch(
            "robot_pipeline_app.checks.probe_camera_capture",
            return_value=(True, "frame=640x480"),
        ), patch(
            "robot_pipeline_app.checks.os.access",
            return_value=True,
        ), patch(
            "robot_pipeline_app.checks.serial_port_fingerprint",
            side_effect=["follower_fp", "leader_fp"],
        ), patch(
            "robot_pipeline_app.checks.camera_fingerprint",
            side_effect=["camera_laptop", "camera_phone"],
        ), patch(
            "robot_pipeline_app.checks._serial_lock_check",
            return_value=("PASS", "Serial port lock", "ok"),
        ), patch(
            "robot_pipeline_app.checks._dialout_membership",
            return_value=(False, "missing dialout"),
        ):
            checks = _run_common_preflight_checks(config)

        self.assertTrue(any(level == "FAIL" and name == "dialout group" for level, name, _ in checks))

    def test_run_preflight_for_record_warns_for_short_episode_and_typo(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            dataset_root = base / "record_data"
            lerobot_dir = base / "lerobot"
            (dataset_root / "demo_5").mkdir(parents=True, exist_ok=True)
            (lerobot_dir / "data" / "demo_6").mkdir(parents=True, exist_ok=True)

            config = dict(DEFAULT_CONFIG_VALUES)
            config["lerobot_dir"] = str(lerobot_dir)
            config["last_dataset_name"] = "demo_5"

            checks = run_preflight_for_record(
                config=config,
                dataset_root=dataset_root,
                upload_enabled=False,
                episode_time_s=5,
                dataset_repo_id="alice/dem0_5",
                common_checks_fn=lambda _: [],
            )

        self.assertTrue(any(level == "WARN" and name == "Episode duration" for level, name, _ in checks))
        self.assertTrue(any(level == "WARN" and name == "Dataset repo typo risk" for level, name, _ in checks))

    def test_run_preflight_for_deploy_fails_on_model_camera_key_mismatch(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model_ok"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "config.json").write_text('{"camera_keys": ["wrist", "overhead"]}\n', encoding="utf-8")
            (model_dir / "model.safetensors").write_text("weights\n", encoding="utf-8")

            with patch("robot_pipeline_app.checks._probe_policy_path_support", return_value=("PASS", "policy", "ok")):
                checks = run_preflight_for_deploy(
                    config=config,
                    model_path=model_dir,
                    eval_repo_id="alice/eval_run_1",
                    common_checks_fn=lambda _: [],
                )

        self.assertTrue(any(level == "FAIL" and name == "Model camera keys" for level, name, _ in checks))

    def test_run_preflight_for_deploy_warns_for_cpu_only_high_fps(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["camera_fps"] = 30
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model_ok"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
            (model_dir / "model.safetensors").write_text("weights\n", encoding="utf-8")

            with patch("robot_pipeline_app.checks._probe_policy_path_support", return_value=("PASS", "policy", "ok")), patch(
                "robot_pipeline_app.checks._probe_torch_accelerator",
                return_value=("cpu", "CPU-only runtime"),
            ):
                checks = run_preflight_for_deploy(
                    config=config,
                    model_path=model_dir,
                    eval_repo_id="alice/eval_run_1",
                    common_checks_fn=lambda _: [],
                )

        self.assertTrue(any(level == "WARN" and name == "Deploy loop performance risk" for level, name, _ in checks))


if __name__ == "__main__":
    unittest.main()
