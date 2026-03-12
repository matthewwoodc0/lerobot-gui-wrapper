from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from robot_pipeline_app.compat import (
    _CAP_CACHE,
    _choose_policy_path_flag,
    _choose_train_resume_path_flag,
    _missing_required_train_flags,
    _parse_help_flags,
    compatibility_checks,
    probe_lerobot_capabilities,
    resolve_calibrate_entrypoint,
    resolve_record_entrypoint,
    resolve_teleop_entrypoint,
    resolve_train_entrypoint,
)
from robot_pipeline_app.compat_policy import (
    TRAINING_COMMAND_NOTE,
    WORKFLOW_PASS_GATE_NOTE,
    evaluate_python_compatibility,
    match_validated_track,
)
from robot_pipeline_app.compat_snapshot import build_compat_snapshot
from robot_pipeline_app.constants import DEFAULT_CONFIG_VALUES


class CompatTest(unittest.TestCase):
    def setUp(self) -> None:
        _CAP_CACHE.clear()

    def test_validated_tracks_match_0_5_current_and_0_4_n_minus_1(self) -> None:
        current = match_validated_track("0.5.0")
        n_minus_1 = match_validated_track("0.4.4")

        assert current is not None
        assert n_minus_1 is not None
        self.assertEqual(current.key, "current")
        self.assertEqual(current.version_spec, "0.5.x")
        self.assertEqual(n_minus_1.key, "n_minus_1")
        self.assertEqual(n_minus_1.version_spec, "0.4.x")

    def test_python_compatibility_fails_for_lerobot_0_5_on_python_3_11(self) -> None:
        result = evaluate_python_compatibility("0.5.0", (3, 11, 9))

        self.assertEqual(result.status, "FAIL")
        self.assertTrue(result.hard_fail)
        self.assertIn("requires Python 3.12+", result.detail)

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

        with patch("robot_pipeline_app.compat._lerobot_module_available", side_effect=lambda _config, name: _module_available(name)), patch(
            "robot_pipeline_app.compat.subprocess.run",
            return_value=_Result(),
        ):
            caps = probe_lerobot_capabilities(config, include_flag_probe=True, force_refresh=True)

        self.assertEqual(caps.record_entrypoint, "lerobot.record")
        self.assertEqual(caps.train_entrypoint, "lerobot.train")
        self.assertEqual(caps.policy_path_flag, "policy")
        self.assertEqual(caps.active_rename_flag, "dataset.rename_map")
        self.assertTrue(caps.train_help_available)
        self.assertFalse(caps.supports_train_resume)
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

        with patch("robot_pipeline_app.compat._lerobot_module_available", side_effect=lambda _config, name: _module_available(name)):
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

            with patch("robot_pipeline_app.compat._lerobot_module_available", return_value=False):
                self.assertEqual(resolve_record_entrypoint(config), "lerobot.scripts.record")
                self.assertEqual(resolve_train_entrypoint(config), "lerobot.scripts.train")
                teleop_entrypoint, uses_legacy = resolve_teleop_entrypoint(config)

        self.assertEqual(teleop_entrypoint, "lerobot.scripts.teleoperate")
        self.assertFalse(uses_legacy)

    def test_resolve_entrypoints_respects_configured_checkout_over_installed_modules(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "lerobot").mkdir(parents=True, exist_ok=True)
            (root / "lerobot" / "record.py").write_text("", encoding="utf-8")
            (root / "lerobot" / "train.py").write_text("", encoding="utf-8")
            (root / "lerobot" / "teleoperate.py").write_text("", encoding="utf-8")
            (root / "lerobot" / "calibrate.py").write_text("", encoding="utf-8")

            config = dict(DEFAULT_CONFIG_VALUES)
            config["lerobot_dir"] = str(root)
            config["teleop_av1_fallback"] = False

            with patch("robot_pipeline_app.compat._lerobot_module_available", return_value=True):
                self.assertEqual(resolve_record_entrypoint(config), "lerobot.record")
                self.assertEqual(resolve_train_entrypoint(config), "lerobot.train")
                self.assertEqual(resolve_calibrate_entrypoint(config), "lerobot.calibrate")
                teleop_entrypoint, uses_legacy = resolve_teleop_entrypoint(config)

        self.assertEqual(teleop_entrypoint, "lerobot.teleoperate")
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
            supports_train_resume=False,
            train_resume_path_flag=None,
            train_resume_toggle_flag=None,
            train_resume_detail="Train help output did not expose a checkpoint/config-path resume flag.",
            policy_path_flag="policy.path",
            python_requirement="Python 3.12+",
            python_compatibility_status="PASS",
            python_compatibility_detail="Active Python 3.12.2 satisfies the wrapper baseline (3.12).",
            python_hard_compat_fail=False,
            fallback_notes=("note",),
            lerobot_version="0.4.4",
            python_version="3.12.2",
        )

        with patch("robot_pipeline_app.compat_snapshot.probe_lerobot_capabilities", return_value=fake_capabilities):
            snapshot = build_compat_snapshot(config)

        self.assertEqual(snapshot["train_entrypoint"], "lerobot.train")
        self.assertTrue(snapshot["train_help_available"])
        self.assertEqual(snapshot["missing_train_flags"], ["output_dir"])
        self.assertFalse(snapshot["supports_train_resume"])
        self.assertIn("validated_tracks", snapshot)
        self.assertEqual(snapshot["python_requirement"], "Python 3.12+")
        self.assertEqual(snapshot["python_compatibility_status"], "PASS")
        self.assertEqual(snapshot["workflow_pass_gate_note"], WORKFLOW_PASS_GATE_NOTE)
        self.assertEqual(
            TRAINING_COMMAND_NOTE,
            "The generated command uses the configured LeRobot runtime and detected train entrypoint for your environment.",
        )

    def test_compatibility_checks_fail_when_lerobot_0_5_runs_on_python_below_3_12(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        with patch("robot_pipeline_app.compat._detect_lerobot_version", return_value="0.5.0"), patch(
            "robot_pipeline_app.compat.detect_runtime_python_version",
            return_value="3.11.9",
        ):
            checks = compatibility_checks(config, include_flag_probe=False)

        detail_by_name = {name: (level, detail) for level, name, detail in checks}
        self.assertEqual(detail_by_name["Python compatibility"][0], "FAIL")
        self.assertIn("requires Python 3.12+", detail_by_name["Python compatibility"][1])

    def test_parse_lerobot_0_5_help_fixtures_cover_record_and_train_flags(self) -> None:
        fixture_root = Path(__file__).parent / "fixtures" / "compat"
        record_help = (fixture_root / "lerobot_0_5_record_help.txt").read_text(encoding="utf-8")
        train_help = (fixture_root / "lerobot_0_5_train_help.txt").read_text(encoding="utf-8")

        record_flags = _parse_help_flags(record_help)
        train_flags = _parse_help_flags(train_help)

        self.assertEqual(_choose_policy_path_flag(record_flags), "policy.path")
        self.assertIn("dataset.rename_map", record_flags)
        self.assertEqual(_missing_required_train_flags(train_flags), [])
        self.assertIsNone(_choose_train_resume_path_flag(train_flags))

    def test_choose_train_resume_flag_prefers_explicit_config_path(self) -> None:
        flags = {"dataset.repo_id", "resume", "config_path", "output_dir"}

        self.assertEqual(_choose_train_resume_path_flag(flags), "config_path")

    def test_probe_capabilities_with_lerobot_0_5_help_fixtures(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        fixture_root = Path(__file__).parent / "fixtures" / "compat"
        record_help = (fixture_root / "lerobot_0_5_record_help.txt").read_text(encoding="utf-8")
        train_help = (fixture_root / "lerobot_0_5_train_help.txt").read_text(encoding="utf-8")

        class _Result:
            def __init__(self, stdout: str) -> None:
                self.returncode = 0
                self.stdout = stdout
                self.stderr = ""

        def _subprocess_run(cmd: list[str], **_kwargs: object) -> _Result:
            module_name = cmd[2]
            if module_name == "lerobot.record":
                return _Result(record_help)
            if module_name == "lerobot.train":
                return _Result(train_help)
            return _Result(record_help)

        def _module_available(name: str) -> bool:
            return name in {
                "lerobot.record",
                "lerobot.train",
                "lerobot.teleoperate",
                "lerobot.calibrate",
            }

        with patch("robot_pipeline_app.compat._detect_lerobot_version", return_value="0.5.0"), patch(
            "robot_pipeline_app.compat._lerobot_module_available",
            side_effect=lambda _config, name: _module_available(name),
        ), patch(
            "robot_pipeline_app.compat.subprocess.run",
            side_effect=_subprocess_run,
        ):
            caps = probe_lerobot_capabilities(config, include_flag_probe=True, force_refresh=True)

        self.assertEqual(caps.record_entrypoint, "lerobot.record")
        self.assertEqual(caps.train_entrypoint, "lerobot.train")
        self.assertEqual(caps.policy_path_flag, "policy.path")
        self.assertEqual(caps.active_rename_flag, "dataset.rename_map")
        self.assertEqual(caps.missing_train_flags, ())
        self.assertFalse(caps.supports_train_resume)


if __name__ == "__main__":
    unittest.main()
