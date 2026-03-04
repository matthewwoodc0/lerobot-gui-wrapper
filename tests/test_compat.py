from __future__ import annotations

import unittest
from unittest.mock import patch

from robot_pipeline_app.compat import compatibility_checks, probe_lerobot_capabilities
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
                "lerobot.teleoperate",
                "lerobot.calibrate",
            }

        with patch("robot_pipeline_app.compat._module_available", side_effect=_module_available), patch(
            "robot_pipeline_app.compat.subprocess.run",
            return_value=_Result(),
        ):
            caps = probe_lerobot_capabilities(config, include_flag_probe=True, force_refresh=True)

        self.assertEqual(caps.record_entrypoint, "lerobot.record")
        self.assertEqual(caps.policy_path_flag, "policy")
        self.assertEqual(caps.active_rename_flag, "dataset.rename_map")
        self.assertTrue(any("unsupported" in note.lower() for note in caps.fallback_notes))

    def test_probe_capabilities_cache_hit(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)

        with patch("robot_pipeline_app.compat._probe_record_help_flags", return_value=({"policy.path"}, "")):
            first = probe_lerobot_capabilities(config, include_flag_probe=True, force_refresh=True)
            second = probe_lerobot_capabilities(config, include_flag_probe=True, force_refresh=False)

        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)

    def test_compatibility_checks_emit_entrypoint_and_policy_rows(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        checks = compatibility_checks(config, include_flag_probe=False)
        names = {name for _, name, _ in checks}
        self.assertIn("Compatibility policy", names)
        self.assertIn("Record entrypoint", names)
        self.assertIn("Policy path flag", names)


if __name__ == "__main__":
    unittest.main()
