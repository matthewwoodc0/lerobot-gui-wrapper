from __future__ import annotations

import unittest

from robot_pipeline_app.training_profiles import load_training_profiles, save_training_profiles
from robot_pipeline_app.types import TrainingProfile


class TrainingProfilesTest(unittest.TestCase):
    def test_load_training_profiles_defaults_when_missing(self) -> None:
        profiles, active_id = load_training_profiles({})
        self.assertTrue(profiles)
        self.assertEqual(active_id, profiles[0].id)
        self.assertEqual(profiles[0].id, "olympus")
        self.assertEqual(profiles[0].remote_project_root, "~/lerobot/src")
        self.assertEqual(profiles[0].default_tmux_session, "train")
        self.assertIn("srun -p gpu-research", profiles[0].default_srun_prefix)

    def test_load_training_profiles_normalizes_and_dedupes(self) -> None:
        config = {
            "training_profiles": [
                {
                    "id": "p1",
                    "name": "One",
                    "host": "example.org",
                    "username": "alice",
                    "auth_mode": "PASSWORD",
                },
                {
                    "id": "p1",
                    "name": "Duplicate",
                    "host": "dupe.example.org",
                    "username": "bob",
                },
            ],
            "training_active_profile_id": "missing",
        }
        profiles, active_id = load_training_profiles(config)
        self.assertEqual(len(profiles), 1)
        self.assertEqual(profiles[0].id, "p1")
        self.assertEqual(profiles[0].auth_mode, "password")
        self.assertEqual(active_id, "p1")

    def test_save_training_profiles_falls_back_to_first_active(self) -> None:
        profile = TrainingProfile(
            id="demo",
            name="Demo",
            host="host",
            port=22,
            username="user",
            auth_mode="password",
            identity_file="",
            remote_models_root="~/models",
            remote_project_root="~/proj",
            env_activate_cmd="source env/bin/activate",
            default_tmux_session="train",
            default_srun_prefix="srun --pty bash -lc",
        )
        config: dict[str, object] = {}
        save_training_profiles(config, [profile], active_profile_id="missing")
        self.assertEqual(config["training_active_profile_id"], "demo")
        self.assertEqual(len(config["training_profiles"]), 1)


if __name__ == "__main__":
    unittest.main()
