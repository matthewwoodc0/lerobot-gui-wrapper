from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robot_pipeline_app.checks import _run_common_preflight_checks, collect_doctor_checks
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
            ):
                checks = collect_doctor_checks(config)

        self.assertTrue(any(level == "WARN" and name == "Next record dataset collision" for level, name, _ in checks))
        self.assertTrue(any(level == "WARN" and name == "Next eval dataset collision" for level, name, _ in checks))

    def test_common_preflight_warns_when_camera_resolution_mismatch_detected(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["camera_laptop_index"] = 0
        config["camera_phone_index"] = 6
        config["camera_laptop_width"] = 640
        config["camera_laptop_height"] = 360
        config["camera_phone_width"] = 640
        config["camera_phone_height"] = 360

        with patch("robot_pipeline_app.checks.get_lerobot_dir", return_value=Path("/tmp")), patch(
            "robot_pipeline_app.checks.Path.exists",
            return_value=True,
        ), patch(
            "robot_pipeline_app.checks.probe_module_import",
            return_value=(True, ""),
        ), patch(
            "robot_pipeline_app.checks.probe_camera_capture",
            side_effect=[(True, "frame=640x480"), (True, "frame=640x360")],
        ):
            checks = _run_common_preflight_checks(config)

        self.assertTrue(
            any(
                level == "WARN" and name == "Laptop camera resolution" and "detected=640x480" in detail
                for level, name, detail in checks
            )
        )


if __name__ == "__main__":
    unittest.main()
