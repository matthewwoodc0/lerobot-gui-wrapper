from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robot_pipeline_app.auto_names import (
    build_train_job_name_base,
    resolve_available_name,
    resolve_deploy_eval_name,
    resolve_train_job_name,
)
from robot_pipeline_app.constants import DEFAULT_CONFIG_VALUES


class AutoNamesTests(unittest.TestCase):
    def test_resolve_available_name_advances_monotonically_from_numbered_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "demo_5").mkdir(parents=True, exist_ok=True)

            resolution = resolve_available_name(
                "demo_5",
                prefer_owner_display=False,
                local_exists_fn=lambda name: (root / name).exists(),
            )

        self.assertTrue(resolution.occupied)
        self.assertEqual(resolution.occupied_sources, ("local",))
        self.assertEqual(resolution.resolved_name, "demo_6")

    def test_resolve_available_name_applies_required_prefix_before_collision_check(self) -> None:
        seen: list[str] = []

        def fake_exists(repo_id: str) -> bool | None:
            seen.append(repo_id)
            return repo_id in {"alice/eval_run_1"}

        resolution = resolve_available_name(
            "run_1",
            default_owner="alice",
            required_prefix="eval_",
            remote_exists_fn=fake_exists,
        )

        self.assertTrue(resolution.prefix_applied)
        self.assertEqual(resolution.repo_id, "alice/eval_run_2")
        self.assertIn("alice/eval_run_1", seen)

    def test_resolve_deploy_eval_name_preserves_explicit_owner(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["hf_username"] = "alice"

        with patch("robot_pipeline_app.repo_utils.dataset_exists_on_hf", side_effect=lambda repo_id: repo_id == "org/eval_run_1"):
            resolution = resolve_deploy_eval_name("org/run_1", config=config)

        self.assertEqual(resolution.owner, "org")
        self.assertEqual(resolution.repo_id, "org/eval_run_2")

    def test_build_train_job_name_base_uses_dataset_leaf_and_policy(self) -> None:
        self.assertEqual(build_train_job_name_base("alice/demo-train", "diffusion"), "demo_train_diffusion")

    def test_resolve_train_job_name_checks_model_repo_collisions(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["hf_username"] = "alice"

        with patch("robot_pipeline_app.repo_utils.model_exists_on_hf", side_effect=lambda repo_id: repo_id == "alice/demo_train_act_1"):
            resolution = resolve_train_job_name(
                "",
                config=config,
                dataset_input="alice/demo-train",
                policy_type="act",
                output_dir_raw="outputs/train",
            )

        self.assertEqual(resolution.resolved_name, "demo_train_act_2")
        self.assertEqual(resolution.repo_id, "alice/demo_train_act_2")


if __name__ == "__main__":
    unittest.main()
