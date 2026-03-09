from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robot_pipeline_app.checks import (
    _extract_model_camera_keys,
    _find_robot_calibration_path,
    _is_suspicious_float,
    _run_common_preflight_checks,
    _validate_calibration_values,
    collect_doctor_checks,
    diagnostics_from_checks,
    run_preflight_for_deploy,
    run_preflight_for_record,
    run_preflight_for_teleop,
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
                level == "WARN" and name == "Camera 'phone' resolution" and "detected=640x360" in detail
                for level, name, detail in checks
            )
        )

    def test_common_preflight_reports_when_compat_probe_disabled(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["compat_probe_enabled"] = False

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
            side_effect=["follower", "leader"],
        ), patch(
            "robot_pipeline_app.checks.camera_fingerprint",
            side_effect=["camera_laptop", "camera_phone"],
        ), patch(
            "robot_pipeline_app.checks._serial_lock_check",
            return_value=("PASS", "Serial port lock", "ok"),
        ):
            checks = _run_common_preflight_checks(config)

        self.assertTrue(any(level == "WARN" and name == "Compatibility probe" for level, name, _ in checks))

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

    def test_common_preflight_supports_three_camera_schema(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["camera_schema_json"] = (
            '{"camera1":{"index_or_path":0},"camera2":{"index_or_path":1},"camera3":{"index_or_path":2}}'
        )
        with patch("robot_pipeline_app.checks.get_lerobot_dir", return_value=Path("/tmp")), patch(
            "robot_pipeline_app.checks.Path.exists",
            return_value=True,
        ), patch(
            "robot_pipeline_app.checks.probe_module_import",
            return_value=(True, ""),
        ), patch(
            "robot_pipeline_app.checks.probe_camera_capture",
            return_value=(True, "frame=640x360"),
        ), patch(
            "robot_pipeline_app.checks.os.access",
            return_value=True,
        ), patch(
            "robot_pipeline_app.checks.serial_port_fingerprint",
            side_effect=["follower_fp", "leader_fp"],
        ), patch(
            "robot_pipeline_app.checks.camera_fingerprint",
            side_effect=["cam1_fp", "cam2_fp", "cam3_fp"],
        ), patch(
            "robot_pipeline_app.checks._serial_lock_check",
            return_value=("PASS", "Serial port lock", "ok"),
        ):
            checks = _run_common_preflight_checks(config)

        self.assertTrue(
            any(
                level == "PASS" and name == "Camera schema" and "3 camera(s)" in detail
                for level, name, detail in checks
            )
        )
        self.assertFalse(any(level == "FAIL" and name == "Camera source uniqueness" for level, name, _ in checks))

    def test_common_preflight_supports_single_camera_schema(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["camera_schema_json"] = '{"wrist":{"index_or_path":0}}'
        with patch("robot_pipeline_app.checks.get_lerobot_dir", return_value=Path("/tmp")), patch(
            "robot_pipeline_app.checks.Path.exists",
            return_value=True,
        ), patch(
            "robot_pipeline_app.checks.probe_module_import",
            return_value=(True, ""),
        ), patch(
            "robot_pipeline_app.checks.probe_camera_capture",
            return_value=(True, "frame=640x360"),
        ), patch(
            "robot_pipeline_app.checks.os.access",
            return_value=True,
        ), patch(
            "robot_pipeline_app.checks.serial_port_fingerprint",
            side_effect=["follower_fp", "leader_fp"],
        ), patch(
            "robot_pipeline_app.checks.camera_fingerprint",
            return_value="cam_wrist",
        ), patch(
            "robot_pipeline_app.checks._serial_lock_check",
            return_value=("PASS", "Serial port lock", "ok"),
        ):
            checks = _run_common_preflight_checks(config)

        self.assertTrue(any(level == "PASS" and name == "Camera schema" for level, name, _ in checks))

    def test_common_preflight_warns_dialout_when_ports_accessible_via_udev(self) -> None:
        """Not in dialout group but ports are R/W accessible (e.g. udev rule) → WARN, not FAIL."""
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
            return_value=(False, "User 'testuser' is not in dialout."),
        ):
            checks = _run_common_preflight_checks(config)

        dialout_checks = [(l, n, d) for l, n, d in checks if n == "dialout group"]
        self.assertEqual(len(dialout_checks), 1)
        level, _, detail = dialout_checks[0]
        self.assertEqual(level, "WARN")
        self.assertIn("udev", detail.lower())

    def test_common_preflight_fails_dialout_when_ports_not_accessible(self) -> None:
        """Not in dialout AND ports are NOT accessible → FAIL."""
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
            return_value=False,
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
            return_value=(False, "User 'testuser' is not in dialout."),
        ):
            checks = _run_common_preflight_checks(config)

        self.assertTrue(any(level == "FAIL" and name == "dialout group" for level, name, _ in checks))

    def test_common_preflight_checks_scservo_sdk_module(self) -> None:
        """scservo_sdk import check appears in common preflight."""
        config = dict(DEFAULT_CONFIG_VALUES)

        def _mock_import(module_name: str) -> tuple[bool, str]:
            if module_name == "scservo_sdk":
                return (False, "ModuleNotFoundError: No module named 'scservo_sdk'")
            return (True, "")

        with patch("robot_pipeline_app.checks.get_lerobot_dir", return_value=Path("/tmp")), patch(
            "robot_pipeline_app.checks.Path.exists",
            return_value=True,
        ), patch(
            "robot_pipeline_app.checks.probe_module_import",
            side_effect=_mock_import,
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
        ):
            checks = _run_common_preflight_checks(config)

        scs_checks = [(l, n, d) for l, n, d in checks if n == "Python module: scservo_sdk"]
        self.assertEqual(len(scs_checks), 1)
        level, _, detail = scs_checks[0]
        self.assertEqual(level, "FAIL")
        self.assertIn("pip install feetech-servo-sdk", detail)

    def test_common_preflight_passes_activation_when_custom_command_configured(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["setup_venv_activate_cmd"] = "conda activate lerobot"
        config["lerobot_venv_dir"] = "/definitely/missing/env"

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
        ):
            checks = _run_common_preflight_checks(config)

        activation_checks = [(l, n, d) for l, n, d in checks if n == "Environment activation"]
        self.assertEqual(len(activation_checks), 1)
        self.assertEqual(activation_checks[0][0], "PASS")
        self.assertIn("custom activation command", activation_checks[0][2])

    def test_common_preflight_passes_activation_for_conda_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            conda_env = Path(tmpdir) / "miniforge3" / "envs" / "lerobot"
            (conda_env / "conda-meta").mkdir(parents=True, exist_ok=True)

            config = dict(DEFAULT_CONFIG_VALUES)
            config["lerobot_venv_dir"] = str(conda_env)

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
            ):
                checks = _run_common_preflight_checks(config)

        activation_checks = [(l, n, d) for l, n, d in checks if n == "Environment activation"]
        self.assertEqual(len(activation_checks), 1)
        self.assertEqual(activation_checks[0][0], "PASS")
        self.assertIn("conda environment folder detected", activation_checks[0][2])

    def test_teleop_preflight_uses_configured_robot_ids_for_calibration_checks(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["follower_robot_id"] = "arm_alpha"
        config["leader_robot_id"] = "operator_beta"

        with patch("robot_pipeline_app.checks._check_robot_calibration", return_value=[] ) as check_robot_calibration:
            run_preflight_for_teleop(config=config, common_checks_fn=lambda _: [])

        self.assertEqual(check_robot_calibration.call_count, 2)
        first_call = check_robot_calibration.call_args_list[0]
        second_call = check_robot_calibration.call_args_list[1]
        self.assertEqual(first_call.kwargs.get("robot_id"), "arm_alpha")
        self.assertEqual(second_call.kwargs.get("robot_id"), "operator_beta")

    def test_teleop_preflight_infers_robot_ids_from_selected_calibration_files(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["follower_robot_id"] = "red4"
        config["leader_robot_id"] = "white"
        config["follower_calibration_path"] = "/tmp/calibration/arm_alpha.json"
        config["leader_calibration_path"] = "/tmp/calibration/operator_beta.json"

        with patch("robot_pipeline_app.checks._check_robot_calibration", return_value=[] ) as check_robot_calibration:
            run_preflight_for_teleop(config=config, common_checks_fn=lambda _: [])

        self.assertEqual(check_robot_calibration.call_count, 2)
        first_call = check_robot_calibration.call_args_list[0]
        second_call = check_robot_calibration.call_args_list[1]
        self.assertEqual(first_call.kwargs.get("robot_id"), "arm_alpha")
        self.assertEqual(second_call.kwargs.get("robot_id"), "operator_beta")

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

    def test_run_preflight_for_deploy_fails_on_camera_count_mismatch(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model_ok"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "config.json").write_text('{"camera_keys": ["camera1", "camera2", "camera3"]}\n', encoding="utf-8")
            (model_dir / "model.safetensors").write_text("weights\n", encoding="utf-8")

            with patch("robot_pipeline_app.checks._probe_policy_path_support", return_value=("PASS", "policy", "ok")):
                checks = run_preflight_for_deploy(
                    config=config,
                    model_path=model_dir,
                    eval_repo_id="alice/eval_run_1",
                    common_checks_fn=lambda _: [],
                )

        self.assertTrue(any(level == "FAIL" and name == "Model camera keys" and "camera count mismatch" in detail for level, name, detail in checks))

    def test_run_preflight_for_deploy_passes_with_matching_rename_map_override(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model_ok"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "config.json").write_text('{"camera_keys": ["camera1", "camera2"]}\n', encoding="utf-8")
            (model_dir / "model.safetensors").write_text("weights\n", encoding="utf-8")
            cmd = [
                "python3",
                "-m",
                "lerobot.scripts.lerobot_record",
                '--rename_map={"observation.images.laptop":"observation.images.camera1","observation.images.phone":"observation.images.camera2"}',
            ]

            with patch("robot_pipeline_app.checks._probe_policy_path_support", return_value=("PASS", "policy", "ok")):
                checks = run_preflight_for_deploy(
                    config=config,
                    model_path=model_dir,
                    eval_repo_id="alice/eval_run_1",
                    command=cmd,
                    common_checks_fn=lambda _: [],
                )

        self.assertTrue(any(level == "PASS" and name == "Camera rename map" for level, name, _ in checks))
        self.assertTrue(any(level == "PASS" and name == "Model camera keys" for level, name, _ in checks))
        self.assertFalse(any(level == "FAIL" and name == "Model camera keys" for level, name, _ in checks))

    def test_run_preflight_for_deploy_suggests_rename_map_for_weird_runtime_camera_names(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["camera_schema_json"] = (
            '{"wrist":{"index_or_path":0},"overhead":{"index_or_path":1},"side":{"index_or_path":2}}'
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model_ok"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "config.json").write_text('{"camera_keys": ["camera1", "camera2", "camera3"]}\n', encoding="utf-8")
            (model_dir / "model.safetensors").write_text("weights\n", encoding="utf-8")

            with patch("robot_pipeline_app.checks._probe_policy_path_support", return_value=("PASS", "policy", "ok")):
                checks = run_preflight_for_deploy(
                    config=config,
                    model_path=model_dir,
                    eval_repo_id="alice/eval_run_1",
                    common_checks_fn=lambda _: [],
                )

        self.assertTrue(any(level == "FAIL" and name == "Model camera keys" for level, name, _ in checks))
        suggestion = next(
            (detail for level, name, detail in checks if level == "WARN" and name == "Camera rename map suggestion"),
            "",
        )
        self.assertIn('"observation.images.wrist":"observation.images.camera', suggestion)

    def test_run_preflight_for_deploy_warns_when_compat_probe_disabled(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["compat_probe_enabled"] = False
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model_ok"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
            (model_dir / "model.safetensors").write_text("weights\n", encoding="utf-8")

            checks = run_preflight_for_deploy(
                config=config,
                model_path=model_dir,
                eval_repo_id="alice/eval_run_1",
                common_checks_fn=lambda _: [],
            )

        self.assertTrue(any(level == "WARN" and name == "Deploy compatibility" for level, name, _ in checks))

    def test_extract_model_camera_keys_handles_permission_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir(parents=True, exist_ok=True)
            with patch("pathlib.Path.iterdir", side_effect=PermissionError("denied")):
                keys, detail = _extract_model_camera_keys(model_dir)
        self.assertIsNone(keys)
        self.assertIn("permission denied", detail.lower())

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

    def test_run_preflight_for_teleop_warns_for_extreme_fps(self) -> None:
        checks = run_preflight_for_teleop(
            config=dict(DEFAULT_CONFIG_VALUES),
            control_fps=240,
            common_checks_fn=lambda _: [],
        )
        self.assertTrue(any(level == "WARN" and name == "Teleop control FPS" for level, name, _ in checks))

    # ------------------------------------------------------------------ #
    # Training config vs. deploy config checks                            #
    # ------------------------------------------------------------------ #

    def test_deploy_preflight_passes_fps_when_training_matches_runtime(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["camera_fps"] = 30
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text('{"fps": 30, "robot_type": "so101_follower"}\n', encoding="utf-8")
            (model_dir / "model.safetensors").write_text("w\n", encoding="utf-8")

            with patch("robot_pipeline_app.checks._probe_policy_path_support", return_value=("PASS", "p", "ok")), \
                 patch("robot_pipeline_app.checks._probe_torch_accelerator", return_value=("cuda", "CUDA")), \
                 patch("robot_pipeline_app.checks._find_robot_calibration_path", return_value=None):
                checks = run_preflight_for_deploy(
                    config=config, model_path=model_dir,
                    eval_repo_id="alice/eval_run_1", common_checks_fn=lambda _: [],
                )

        self.assertTrue(any(level == "PASS" and name == "Training vs deploy FPS" for level, name, _ in checks))

    def test_deploy_preflight_fails_fps_when_training_differs_from_runtime(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["camera_fps"] = 15
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text('{"fps": 30}\n', encoding="utf-8")
            (model_dir / "model.safetensors").write_text("w\n", encoding="utf-8")

            with patch("robot_pipeline_app.checks._probe_policy_path_support", return_value=("PASS", "p", "ok")), \
                 patch("robot_pipeline_app.checks._probe_torch_accelerator", return_value=("cuda", "CUDA")), \
                 patch("robot_pipeline_app.checks._find_robot_calibration_path", return_value=None):
                checks = run_preflight_for_deploy(
                    config=config, model_path=model_dir,
                    eval_repo_id="alice/eval_run_1", common_checks_fn=lambda _: [],
                )

        self.assertTrue(any(level == "FAIL" and name == "Training vs deploy FPS" for level, name, _ in checks))
        fail_detail = next(detail for level, name, detail in checks if name == "Training vs deploy FPS" and level == "FAIL")
        self.assertIn("30", fail_detail)
        self.assertIn("15", fail_detail)

    def test_deploy_preflight_passes_robot_type_when_training_matches(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text('{"robot_type": "so101_follower"}\n', encoding="utf-8")
            (model_dir / "model.safetensors").write_text("w\n", encoding="utf-8")

            with patch("robot_pipeline_app.checks._probe_policy_path_support", return_value=("PASS", "p", "ok")), \
                 patch("robot_pipeline_app.checks._probe_torch_accelerator", return_value=("cuda", "CUDA")), \
                 patch("robot_pipeline_app.checks._find_robot_calibration_path", return_value=None):
                checks = run_preflight_for_deploy(
                    config=config, model_path=model_dir,
                    eval_repo_id="alice/eval_run_1", common_checks_fn=lambda _: [],
                )

        self.assertTrue(any(level == "PASS" and name == "Training vs deploy robot type" for level, name, _ in checks))

    def test_deploy_preflight_fails_robot_type_when_training_differs(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text('{"robot_type": "lekiwi"}\n', encoding="utf-8")
            (model_dir / "model.safetensors").write_text("w\n", encoding="utf-8")

            with patch("robot_pipeline_app.checks._probe_policy_path_support", return_value=("PASS", "p", "ok")), \
                 patch("robot_pipeline_app.checks._probe_torch_accelerator", return_value=("cuda", "CUDA")), \
                 patch("robot_pipeline_app.checks._find_robot_calibration_path", return_value=None):
                checks = run_preflight_for_deploy(
                    config=config, model_path=model_dir,
                    eval_repo_id="alice/eval_run_1", common_checks_fn=lambda _: [],
                )

        self.assertTrue(any(level == "FAIL" and name == "Training vs deploy robot type" for level, name, _ in checks))

    def test_deploy_preflight_fails_action_dim_when_model_has_wrong_dof(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            payload = '{"output_shapes": {"action": {"shape": [7]}}}\n'
            (model_dir / "config.json").write_text(payload, encoding="utf-8")
            (model_dir / "model.safetensors").write_text("w\n", encoding="utf-8")

            with patch("robot_pipeline_app.checks._probe_policy_path_support", return_value=("PASS", "p", "ok")), \
                 patch("robot_pipeline_app.checks._probe_torch_accelerator", return_value=("cuda", "CUDA")), \
                 patch("robot_pipeline_app.checks._find_robot_calibration_path", return_value=None):
                checks = run_preflight_for_deploy(
                    config=config, model_path=model_dir,
                    eval_repo_id="alice/eval_run_1", common_checks_fn=lambda _: [],
                )

        self.assertTrue(any(level == "FAIL" and name == "Training vs deploy action dim" for level, name, _ in checks))

    def test_deploy_preflight_passes_action_dim_when_model_has_correct_dof(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            payload = '{"output_shapes": {"action": {"shape": [6]}}}\n'
            (model_dir / "config.json").write_text(payload, encoding="utf-8")
            (model_dir / "model.safetensors").write_text("w\n", encoding="utf-8")

            with patch("robot_pipeline_app.checks._probe_policy_path_support", return_value=("PASS", "p", "ok")), \
                 patch("robot_pipeline_app.checks._probe_torch_accelerator", return_value=("cuda", "CUDA")), \
                 patch("robot_pipeline_app.checks._find_robot_calibration_path", return_value=None):
                checks = run_preflight_for_deploy(
                    config=config, model_path=model_dir,
                    eval_repo_id="alice/eval_run_1", common_checks_fn=lambda _: [],
                )

        self.assertTrue(any(level == "PASS" and name == "Training vs deploy action dim" for level, name, _ in checks))

    def test_deploy_preflight_warns_when_calibration_file_missing(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
            (model_dir / "model.safetensors").write_text("w\n", encoding="utf-8")

            with patch("robot_pipeline_app.checks._probe_policy_path_support", return_value=("PASS", "p", "ok")), \
                 patch("robot_pipeline_app.checks._probe_torch_accelerator", return_value=("cuda", "CUDA")), \
                 patch("robot_pipeline_app.checks._find_robot_calibration_path", return_value=None):
                checks = run_preflight_for_deploy(
                    config=config, model_path=model_dir,
                    eval_repo_id="alice/eval_run_1", common_checks_fn=lambda _: [],
                )

        self.assertTrue(any(level == "WARN" and name == "Follower calibration file" for level, name, _ in checks))

    def test_deploy_preflight_fails_motor_mismatch_between_model_and_calibration(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            calib_dir = Path(tmpdir) / "calibration"
            model_dir.mkdir()
            calib_dir.mkdir()

            model_payload = '{"motor_names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]}\n'
            (model_dir / "config.json").write_text(model_payload, encoding="utf-8")
            (model_dir / "model.safetensors").write_text("w\n", encoding="utf-8")

            # Calibration has a different joint ("claw" instead of "gripper")
            calib_payload = '{"shoulder_pan": {}, "shoulder_lift": {}, "elbow_flex": {}, "wrist_flex": {}, "wrist_roll": {}, "claw": {}}'
            calib_file = calib_dir / "red4.json"
            calib_file.write_text(calib_payload, encoding="utf-8")

            with patch("robot_pipeline_app.checks._probe_policy_path_support", return_value=("PASS", "p", "ok")), \
                 patch("robot_pipeline_app.checks._probe_torch_accelerator", return_value=("cuda", "CUDA")), \
                 patch("robot_pipeline_app.checks._find_robot_calibration_path", return_value=calib_file):
                checks = run_preflight_for_deploy(
                    config=config, model_path=model_dir,
                    eval_repo_id="alice/eval_run_1", common_checks_fn=lambda _: [],
                )

        self.assertTrue(any(level == "FAIL" and name == "Follower model vs calibration motors" for level, name, _ in checks))

    def test_deploy_preflight_passes_motor_match_between_model_and_calibration(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            calib_dir = Path(tmpdir) / "calibration"
            model_dir.mkdir()
            calib_dir.mkdir()

            import json as _json
            (model_dir / "config.json").write_text(
                _json.dumps({"motor_names": joints}), encoding="utf-8"
            )
            (model_dir / "model.safetensors").write_text("w\n", encoding="utf-8")

            calib_payload = _json.dumps({j: {} for j in joints})
            calib_file = calib_dir / "red4.json"
            calib_file.write_text(calib_payload, encoding="utf-8")

            with patch("robot_pipeline_app.checks._probe_policy_path_support", return_value=("PASS", "p", "ok")), \
                 patch("robot_pipeline_app.checks._probe_torch_accelerator", return_value=("cuda", "CUDA")), \
                 patch("robot_pipeline_app.checks._find_robot_calibration_path", return_value=calib_file):
                checks = run_preflight_for_deploy(
                    config=config, model_path=model_dir,
                    eval_repo_id="alice/eval_run_1", common_checks_fn=lambda _: [],
                )

        self.assertTrue(any(level == "PASS" and name == "Follower model vs calibration motors" for level, name, _ in checks))


class CalibrationValidationTest(unittest.TestCase):
    """Tests for _is_suspicious_float and _validate_calibration_values."""

    # ------------------------------------------------------------------ #
    # _is_suspicious_float                                                 #
    # ------------------------------------------------------------------ #

    def test_suspicious_float_detects_inf(self) -> None:
        self.assertTrue(_is_suspicious_float(float("inf")))
        self.assertTrue(_is_suspicious_float(float("-inf")))

    def test_suspicious_float_detects_nan(self) -> None:
        self.assertTrue(_is_suspicious_float(float("nan")))

    def test_suspicious_float_passes_normal_numbers(self) -> None:
        self.assertFalse(_is_suspicious_float(0))
        self.assertFalse(_is_suspicious_float(-2145))
        self.assertFalse(_is_suspicious_float(4095))
        self.assertFalse(_is_suspicious_float(3.14))

    def test_suspicious_float_handles_non_numeric(self) -> None:
        self.assertFalse(_is_suspicious_float("hello"))
        self.assertFalse(_is_suspicious_float(None))

    # ------------------------------------------------------------------ #
    # Old array-format calibration                                         #
    # ------------------------------------------------------------------ #

    def _write_calib(self, tmpdir: Path, payload: object) -> Path:
        import json as _json
        calib = tmpdir / "red4.json"
        calib.write_text(_json.dumps(payload), encoding="utf-8")
        return calib

    def test_old_format_valid_calibration_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            calib = self._write_calib(Path(tmpdir), {
                "homing_offset": [-2145, 4, -1401, -3126, 2970, -1769],
                "drive_mode": [0, 1, 0, 0, 1, 0],
                "start_pos": [2085, 56, 1285, 3237, 3080, 1983],
                "end_pos": [3169, 1020, 2425, 4150, 946, 2793],
                "calib_mode": ["DEGREE"] * 5 + ["LINEAR"],
                "motor_names": ["shoulder_pan", "shoulder_lift", "elbow_flex",
                                "wrist_flex", "wrist_roll", "gripper"],
            })
            checks = _validate_calibration_values(calib)

        self.assertTrue(any(level == "PASS" and "homing offsets" in name for level, name, _ in checks))
        self.assertTrue(any(level == "PASS" and "drive modes" in name for level, name, _ in checks))
        self.assertFalse(any(level == "FAIL" for level, _, _ in checks))

    def test_old_format_inf_homing_offset_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            calib = self._write_calib(Path(tmpdir), {
                "homing_offset": [float("inf"), 4, -1401, -3126, 2970, -1769],
                "drive_mode": [0, 1, 0, 0, 1, 0],
                "motor_names": ["shoulder_pan", "shoulder_lift", "elbow_flex",
                                "wrist_flex", "wrist_roll", "gripper"],
            })
            checks = _validate_calibration_values(calib)

        self.assertTrue(any(level == "FAIL" and "inf" in name.lower() for level, name, _ in checks))

    def test_old_format_invalid_drive_mode_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            calib = self._write_calib(Path(tmpdir), {
                "homing_offset": [-2145, 4, -1401, -3126, 2970, -1769],
                "drive_mode": [0, 2, 0, 0, 1, 0],  # 2 is invalid
                "motor_names": ["shoulder_pan", "shoulder_lift", "elbow_flex",
                                "wrist_flex", "wrist_roll", "gripper"],
            })
            checks = _validate_calibration_values(calib)

        self.assertTrue(any(level == "FAIL" and "drive mode" in name.lower() for level, name, _ in checks))

    # ------------------------------------------------------------------ #
    # New per-motor-object calibration                                     #
    # ------------------------------------------------------------------ #

    def _new_format_payload(self, overrides: dict | None = None) -> dict:
        base = {
            "shoulder_pan":   {"id": 1, "drive_mode": 0, "homing_offset": -2145, "range_min": 1000, "range_max": 3200},
            "shoulder_lift":  {"id": 2, "drive_mode": 1, "homing_offset":     4, "range_min":   50, "range_max": 2000},
            "elbow_flex":     {"id": 3, "drive_mode": 0, "homing_offset": -1401, "range_min":  800, "range_max": 3000},
            "wrist_flex":     {"id": 4, "drive_mode": 0, "homing_offset": -3126, "range_min":  600, "range_max": 2800},
            "wrist_roll":     {"id": 5, "drive_mode": 1, "homing_offset":  2970, "range_min":  200, "range_max": 3900},
            "gripper":        {"id": 6, "drive_mode": 0, "homing_offset": -1769, "range_min":  900, "range_max": 2500},
        }
        if overrides:
            for motor, fields in overrides.items():
                if motor in base:
                    base[motor] = {**base[motor], **fields}
                else:
                    base[motor] = fields
        return base

    def test_new_format_valid_calibration_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            calib = self._write_calib(Path(tmpdir), self._new_format_payload())
            checks = _validate_calibration_values(calib)

        self.assertFalse(any(level == "FAIL" for level, _, _ in checks))
        self.assertTrue(any(level == "PASS" and "drive modes" in name for level, name, _ in checks))
        self.assertTrue(any(level == "PASS" and "homing offsets" in name for level, name, _ in checks))
        self.assertTrue(any(level == "PASS" and "motor ranges" in name for level, name, _ in checks))

    def test_new_format_inf_in_homing_offset_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            calib = self._write_calib(
                Path(tmpdir),
                self._new_format_payload({"wrist_roll": {"homing_offset": float("inf")}}),
            )
            checks = _validate_calibration_values(calib)

        self.assertTrue(any(level == "FAIL" and "inf" in name.lower() for level, name, _ in checks))

    def test_new_format_invalid_drive_mode_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            calib = self._write_calib(
                Path(tmpdir),
                self._new_format_payload({"shoulder_lift": {"drive_mode": 3}}),
            )
            checks = _validate_calibration_values(calib)

        self.assertTrue(any(level == "FAIL" and "drive mode" in name.lower() for level, name, _ in checks))

    def test_new_format_inverted_range_fails(self) -> None:
        # range_min >= range_max → normalisation is undefined → robot outputs NaN actions
        with tempfile.TemporaryDirectory() as tmpdir:
            calib = self._write_calib(
                Path(tmpdir),
                self._new_format_payload({"elbow_flex": {"range_min": 3000, "range_max": 800}}),
            )
            checks = _validate_calibration_values(calib)

        self.assertTrue(any(level == "FAIL" and "motor range" in name.lower() for level, name, _ in checks))

    def test_new_format_narrow_range_warns(self) -> None:
        # Very narrow range (< 200 ticks) indicates bad calibration sweep
        with tempfile.TemporaryDirectory() as tmpdir:
            calib = self._write_calib(
                Path(tmpdir),
                self._new_format_payload({"gripper": {"range_min": 900, "range_max": 950}}),  # 50 ticks
            )
            checks = _validate_calibration_values(calib)

        self.assertTrue(any(level == "WARN" and "motor range" in name.lower() for level, name, _ in checks))

    def test_new_format_duplicate_motor_ids_fails(self) -> None:
        payload = self._new_format_payload()
        payload["gripper"]["id"] = 1  # duplicate of shoulder_pan
        with tempfile.TemporaryDirectory() as tmpdir:
            calib = self._write_calib(Path(tmpdir), payload)
            checks = _validate_calibration_values(calib)

        self.assertTrue(any(level == "FAIL" and "motor id" in name.lower() for level, name, _ in checks))

    def test_new_format_large_homing_offset_warns(self) -> None:
        # Values > 8192 suggest uint16 wrap-around or motor slip
        with tempfile.TemporaryDirectory() as tmpdir:
            calib = self._write_calib(
                Path(tmpdir),
                self._new_format_payload({"shoulder_pan": {"homing_offset": 10000}}),
            )
            checks = _validate_calibration_values(calib)

        self.assertTrue(any(level == "WARN" and "homing offset" in name.lower() for level, name, _ in checks))

    def test_deploy_preflight_includes_deep_calibration_checks_when_calib_found(self) -> None:
        """Integration test: bad calibration (inverted range) FAIL appears in deploy preflight."""
        config = dict(DEFAULT_CONFIG_VALUES)
        import json as _json

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            calib_dir = Path(tmpdir) / "calibration"
            model_dir.mkdir()
            calib_dir.mkdir()

            (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
            (model_dir / "model.safetensors").write_text("w\n", encoding="utf-8")

            # Bad calibration: shoulder_pan has inverted range
            bad_payload = {
                "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": -2145,
                                 "range_min": 3200, "range_max": 1000},  # INVERTED
            }
            calib_file = calib_dir / "red4.json"
            calib_file.write_text(_json.dumps(bad_payload), encoding="utf-8")

            with patch("robot_pipeline_app.checks._probe_policy_path_support", return_value=("PASS", "p", "ok")), \
                 patch("robot_pipeline_app.checks._probe_torch_accelerator", return_value=("cuda", "CUDA")), \
                 patch("robot_pipeline_app.checks._find_robot_calibration_path", return_value=calib_file):
                checks = run_preflight_for_deploy(
                    config=config, model_path=model_dir,
                    eval_repo_id="alice/eval_run_1", common_checks_fn=lambda _: [],
                )

        self.assertTrue(any(level == "FAIL" and "motor range" in name.lower() for level, name, _ in checks))

    def test_new_format_warns_when_normalization_stats_absent_from_model(self) -> None:
        """When calibration looks valid but model has no normalization_stats, emit a WARN
        so the user knows we cannot verify calibration drift from recalibration."""
        model_config_fields: dict = {
            "motor_names": ["shoulder_pan", "shoulder_lift", "elbow_flex",
                            "wrist_flex", "wrist_roll", "gripper"],
            # Deliberately omit "normalization_stats" — simulates a checkpoint that
            # does not embed training-time observation stats.
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            calib = self._write_calib(Path(tmpdir), self._new_format_payload())
            checks = _validate_calibration_values(calib, model_config_fields)

        warn_names = [name for level, name, _ in checks if level == "WARN"]
        # Should warn that we cannot verify normalization
        self.assertTrue(
            any("normalization" in n.lower() for n in warn_names),
            f"Expected a normalization WARN but got: {warn_names}",
        )
        # Should NOT be a FAIL — calibration values themselves are fine
        self.assertFalse(any(level == "FAIL" for level, _, _ in checks))

    def test_new_format_normalization_stats_present_passes_when_consistent(self) -> None:
        """When model embeds matching normalization stats, no drift WARN is raised."""
        # Calibration ranges: shoulder_pan width=2200, shoulder_lift width=1950, etc.
        # Model stats min/max widths approximate same values.
        model_config_fields: dict = {
            "motor_names": ["shoulder_pan", "shoulder_lift", "elbow_flex",
                            "wrist_flex", "wrist_roll", "gripper"],
            "normalization_stats": {
                "min": [1000.0,  50.0, 800.0, 600.0, 200.0, 900.0],
                "max": [3200.0, 2000.0, 3000.0, 2800.0, 3900.0, 2500.0],
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            calib = self._write_calib(Path(tmpdir), self._new_format_payload())
            checks = _validate_calibration_values(calib, model_config_fields)

        norm_checks = [(level, name, detail) for level, name, detail in checks
                       if "normalization" in name.lower()]
        self.assertTrue(norm_checks, "Expected at least one normalization check")
        self.assertTrue(
            any(level == "PASS" for level, _, _ in norm_checks),
            f"Expected PASS normalization check, got: {norm_checks}",
        )
        self.assertFalse(any(level in ("FAIL", "WARN") for level, _, _ in norm_checks))


class CalibrationPathOverrideTest(unittest.TestCase):
    """Tests for follower/leader calibration_path config overrides."""

    def test_follower_path_takes_precedence(self) -> None:
        """When follower_calibration_path is set to a valid file, it is returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            calib_file = Path(tmpdir) / "my_calibration.json"
            calib_file.write_text('{"shoulder_pan": {"id": 1}}', encoding="utf-8")

            config = dict(DEFAULT_CONFIG_VALUES)
            config["follower_calibration_path"] = str(calib_file)

            result = _find_robot_calibration_path(config, config_key="follower_calibration_path")
            self.assertEqual(result, calib_file)

    def test_leader_path_takes_precedence(self) -> None:
        """When leader_calibration_path is set to a valid file, it is returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            calib_file = Path(tmpdir) / "leader_calib.json"
            calib_file.write_text('{"shoulder_pan": {"id": 1}}', encoding="utf-8")

            config = dict(DEFAULT_CONFIG_VALUES)
            config["leader_calibration_path"] = str(calib_file)

            result = _find_robot_calibration_path(
                config, robot_id="white", robot_type="so101_leader",
                config_key="leader_calibration_path",
            )
            self.assertEqual(result, calib_file)

    def test_legacy_calibration_path_fallback_for_follower(self) -> None:
        """Old single calibration_path is still picked up for follower."""
        with tempfile.TemporaryDirectory() as tmpdir:
            calib_file = Path(tmpdir) / "legacy.json"
            calib_file.write_text('{"shoulder_pan": {"id": 1}}', encoding="utf-8")

            config = dict(DEFAULT_CONFIG_VALUES)
            config["calibration_path"] = str(calib_file)
            config["follower_calibration_path"] = ""

            result = _find_robot_calibration_path(config, config_key="follower_calibration_path")
            self.assertEqual(result, calib_file)

    def test_empty_path_falls_back_to_autodiscovery(self) -> None:
        """Empty calibration path means auto-discover (returns None when no candidates exist)."""
        config = dict(DEFAULT_CONFIG_VALUES)
        config["follower_calibration_path"] = ""
        config["lerobot_dir"] = "/nonexistent/lerobot"

        result = _find_robot_calibration_path(config, config_key="follower_calibration_path")
        self.assertIsNone(result)

    def test_nonexistent_path_falls_back_to_autodiscovery(self) -> None:
        """Non-existent path gracefully falls through to auto-discovery."""
        config = dict(DEFAULT_CONFIG_VALUES)
        config["follower_calibration_path"] = "/nonexistent/path/calib.json"
        config["lerobot_dir"] = "/nonexistent/lerobot"

        result = _find_robot_calibration_path(config, config_key="follower_calibration_path")
        self.assertIsNone(result)

    def test_directory_override_resolves_robot_id_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            calib_dir = Path(tmpdir) / "calibration"
            calib_dir.mkdir(parents=True, exist_ok=True)
            follower_file = calib_dir / "arm_alpha.json"
            follower_file.write_text('{"joint": {"id": 1}}', encoding="utf-8")
            config = dict(DEFAULT_CONFIG_VALUES)
            config["follower_calibration_path"] = str(calib_dir)
            result = _find_robot_calibration_path(
                config,
                robot_id="arm_alpha",
                config_key="follower_calibration_path",
            )
            self.assertEqual(result, follower_file)

    def test_deploy_preflight_checks_both_follower_and_leader(self) -> None:
        """Deploy preflight includes calibration checks for both robots."""
        import json as _json

        config = dict(DEFAULT_CONFIG_VALUES)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
            (model_dir / "model.safetensors").write_text("w\n", encoding="utf-8")

            # Follower calibration file
            follower_calib = Path(tmpdir) / "follower_calib.json"
            follower_calib.write_text(_json.dumps({
                "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": -2145,
                                 "range_min": 1000, "range_max": 3200},
            }), encoding="utf-8")
            config["follower_calibration_path"] = str(follower_calib)

            # Leader calibration file
            leader_calib = Path(tmpdir) / "leader_calib.json"
            leader_calib.write_text(_json.dumps({
                "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": -1000,
                                 "range_min": 500, "range_max": 3500},
            }), encoding="utf-8")
            config["leader_calibration_path"] = str(leader_calib)

            with patch("robot_pipeline_app.checks._probe_policy_path_support",
                       return_value=("PASS", "p", "ok")), \
                 patch("robot_pipeline_app.checks._probe_torch_accelerator",
                       return_value=("cuda", "CUDA")):
                checks = run_preflight_for_deploy(
                    config=config, model_path=model_dir,
                    eval_repo_id="alice/eval_run_1", common_checks_fn=lambda _: [],
                )

        check_names = [name.lower() for _, name, _ in checks]
        self.assertTrue(any("follower calibration file" in n for n in check_names),
                        f"Expected Follower calibration file check, got: {check_names}")
        self.assertTrue(any("leader calibration file" in n for n in check_names),
                        f"Expected Leader calibration file check, got: {check_names}")

    def test_record_preflight_checks_both_calibrations(self) -> None:
        """Record preflight includes calibration checks for both robots."""
        config = dict(DEFAULT_CONFIG_VALUES)
        config["lerobot_dir"] = "/nonexistent/lerobot"

        with tempfile.TemporaryDirectory() as tmpdir:
            checks = run_preflight_for_record(
                config=config, dataset_root=Path(tmpdir),
                upload_enabled=False, common_checks_fn=lambda _: [],
            )

        check_names = [name.lower() for _, name, _ in checks]
        self.assertTrue(any("follower calibration" in n for n in check_names),
                        f"Expected Follower calibration check in record preflight, got: {check_names}")
        self.assertTrue(any("leader calibration" in n for n in check_names),
                        f"Expected Leader calibration check in record preflight, got: {check_names}")

    def test_teleop_preflight_checks_both_calibrations(self) -> None:
        """Teleop preflight includes calibration checks for both robots."""
        config = dict(DEFAULT_CONFIG_VALUES)
        config["lerobot_dir"] = "/nonexistent/lerobot"

        checks = run_preflight_for_teleop(
            config=config, common_checks_fn=lambda _: [],
        )

        check_names = [name.lower() for _, name, _ in checks]
        self.assertTrue(any("follower calibration" in n for n in check_names),
                        f"Expected Follower calibration check in teleop preflight, got: {check_names}")
        self.assertTrue(any("leader calibration" in n for n in check_names),
                        f"Expected Leader calibration check in teleop preflight, got: {check_names}")

    def test_diagnostics_from_checks_produces_code_fix_docs_for_failures(self) -> None:
        checks = [
            ("FAIL", "Eval dataset naming", "Suggested quick fix: alice/eval_run_9"),
            ("FAIL", "Training vs deploy FPS", "model trained at 15 Hz but camera_fps=30"),
            ("WARN", "Model payload candidates", "/tmp/model_a, /tmp/model_b"),
        ]
        events = diagnostics_from_checks(checks)
        fail_events = [event for event in events if event.level == "FAIL"]
        self.assertTrue(fail_events)
        for event in fail_events:
            self.assertTrue(event.code)
            self.assertTrue(event.fix)
            self.assertTrue(event.docs_ref)


if __name__ == "__main__":
    unittest.main()
