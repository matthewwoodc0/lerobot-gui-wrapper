from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from robot_pipeline_app.compat import (
    compatibility_checks,
    probe_lerobot_capabilities,
    resolve_record_entrypoint,
    resolve_teleop_entrypoint,
    resolve_train_entrypoint,
)
from robot_pipeline_app.compat_policy import TRAINING_COMMAND_NOTE, WORKFLOW_PASS_GATE_NOTE
from robot_pipeline_app.compat_snapshot import build_compat_snapshot
from robot_pipeline_app.constants import DEFAULT_CONFIG_VALUES


class CompatTest(unittest.TestCase):
    def test_probe_capabilities_detects_flag_fallbacks(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["camera_rename_flag"] = "rename_map"

        class _Result:
            returncode = 0
            stdout = "--policy --dataset.rename_map --dataset.repo_id"
            stderr = ""

        def _module_available(name: str) -> bool:
            return name in {
                "lerobot.record",
                "lerobot.train",
                "lerobot.teleoperate",
                "lerobot.calibrate",
            }

        with patch("robot_pipeline_app.compat._module_available", side_effect=_module_available), patch(
            "robot_pipeline_app.compat.subprocess.run",
            return_value=_Result(),
        ):
            caps = probe_lerobot_capabilities(config, include_flag_probe=True, force_refresh=True)

        self.assertEqual(caps.record_entrypoint, "lerobot.record")
        self.assertEqual(caps.train_entrypoint, "lerobot.train")
        self.assertEqual(caps.policy_path_flag, "policy")
        self.assertEqual(caps.active_rename_flag, "dataset.rename_map")
        self.assertTrue(caps.train_help_available)
        self.assertTrue(any("unsupported" in note.lower() for note in caps.fallback_notes))

    def test_probe_capabilities_cache_hit(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)

        with patch("robot_pipeline_app.compat._probe_record_help_flags", return_value=({"policy.path"}, "")), patch(
            "robot_pipeline_app.compat._probe_train_help_flags",
            return_value=(
                {
                    "policy.path",
                    "policy.input_features",
                    "policy.output_features",
                    "dataset.repo_id",
                    "batch_size",
                    "steps",
                    "output_dir",
                    "job_name",
                    "policy.device",
                    "wandb.enable",
                    "policy.push_to_hub",
                    "save_freq",
                },
                "",
            ),
        ):
            first = probe_lerobot_capabilities(config, include_flag_probe=True, force_refresh=True)
            second = probe_lerobot_capabilities(config, include_flag_probe=True, force_refresh=False)

        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)

    def test_probe_capabilities_cache_invalidates_when_lerobot_version_changes(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)

        with patch(
            "robot_pipeline_app.compat._detect_lerobot_version",
            side_effect=["0.4.3", "0.4.3", "0.4.4"],
        ), patch(
            "robot_pipeline_app.compat._probe_record_help_flags",
            return_value=({"policy.path"}, ""),
        ) as mocked_record_probe, patch(
            "robot_pipeline_app.compat._probe_train_help_flags",
            return_value=(
                {
                    "policy.path",
                    "policy.input_features",
                    "policy.output_features",
                    "dataset.repo_id",
                    "batch_size",
                    "steps",
                    "output_dir",
                    "job_name",
                    "policy.device",
                    "wandb.enable",
                    "policy.push_to_hub",
                    "save_freq",
                },
                "",
            ),
        ) as mocked_train_probe:
            first = probe_lerobot_capabilities(config, include_flag_probe=True, force_refresh=True)
            second = probe_lerobot_capabilities(config, include_flag_probe=True, force_refresh=False)
            third = probe_lerobot_capabilities(config, include_flag_probe=True, force_refresh=False)

        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertFalse(third.cache_hit)
        self.assertEqual(mocked_record_probe.call_count, 2)
        self.assertEqual(mocked_train_probe.call_count, 2)

    def test_resolve_train_entrypoint_prefers_installed_train_module(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)

        def _module_available(name: str) -> bool:
            return name == "lerobot.train"

        with patch("robot_pipeline_app.compat._module_available", side_effect=_module_available):
            entrypoint = resolve_train_entrypoint(config)

        self.assertEqual(entrypoint, "lerobot.train")

    def test_resolve_current_source_layout_entrypoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "lerobot" / "scripts").mkdir(parents=True, exist_ok=True)
            (root / "lerobot" / "scripts" / "record.py").write_text("", encoding="utf-8")
            (root / "lerobot" / "scripts" / "train.py").write_text("", encoding="utf-8")
            (root / "lerobot" / "scripts" / "teleoperate.py").write_text("", encoding="utf-8")

            config = dict(DEFAULT_CONFIG_VALUES)
            config["lerobot_dir"] = str(root)

            with patch("robot_pipeline_app.compat._module_available", return_value=False):
                self.assertEqual(resolve_record_entrypoint(config), "lerobot.scripts.record")
                self.assertEqual(resolve_train_entrypoint(config), "lerobot.scripts.train")
                teleop_entrypoint, uses_legacy = resolve_teleop_entrypoint(config)

        self.assertEqual(teleop_entrypoint, "lerobot.scripts.teleoperate")
        self.assertFalse(uses_legacy)

    def test_compatibility_checks_emit_entrypoint_and_policy_rows(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        checks = compatibility_checks(config, include_flag_probe=False)
        names = {name for _, name, _ in checks}
        self.assertIn("Compatibility policy", names)
        self.assertIn("Validated tracks", names)
        self.assertIn("Workflow validation gate", names)
        self.assertIn("Record entrypoint", names)
        self.assertIn("Train entrypoint", names)
        self.assertIn("Train flags", names)
        self.assertIn("Policy path flag", names)

    def test_compatibility_checks_include_manual_gate_note(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        checks = compatibility_checks(config, include_flag_probe=False)
        details = {name: detail for _, name, detail in checks}
        self.assertEqual(details["Workflow validation gate"], WORKFLOW_PASS_GATE_NOTE)

    def test_build_compat_snapshot_includes_train_probe_fields(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        fake_capabilities = SimpleNamespace(
            record_entrypoint="lerobot.record",
            train_entrypoint="lerobot.train",
            teleop_entrypoint="lerobot.teleoperate",
            calibrate_entrypoint="lerobot.calibrate",
            record_help_available=True,
            train_help_available=True,
            active_rename_flag="rename_map",
            supported_record_flags=("dataset.repo_id", "policy.path"),
            supported_train_flags=("dataset.repo_id", "policy.path", "steps"),
            missing_train_flags=("output_dir",),
            policy_path_flag="policy.path",
            fallback_notes=("note",),
            lerobot_version="0.4.4",
        )

        with patch("robot_pipeline_app.compat_snapshot.probe_lerobot_capabilities", return_value=fake_capabilities):
            snapshot = build_compat_snapshot(config)

        self.assertEqual(snapshot["train_entrypoint"], "lerobot.train")
        self.assertTrue(snapshot["train_help_available"])
        self.assertEqual(snapshot["missing_train_flags"], ["output_dir"])
        self.assertIn("validated_tracks", snapshot)
        self.assertEqual(snapshot["workflow_pass_gate_note"], WORKFLOW_PASS_GATE_NOTE)
        self.assertEqual(TRAINING_COMMAND_NOTE, "The generated command uses the detected LeRobot train entrypoint for your environment.")


if __name__ == "__main__":
    unittest.main()
