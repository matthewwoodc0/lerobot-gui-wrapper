from __future__ import annotations

import tempfile
import unittest

from robot_pipeline_app.constants import CONFIG_FIELDS, DEFAULT_CONFIG_VALUES
from robot_pipeline_app.gui_forms import (
    build_deploy_request_and_command,
    build_record_request_and_command,
    coerce_config_from_vars,
)


class FakeVar:
    def __init__(self, value: str) -> None:
        self._value = value

    def get(self) -> str:
        return self._value


class GuiFormsTest(unittest.TestCase):
    def test_build_record_request_and_command(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        req, cmd, error = build_record_request_and_command(
            config=config,
            dataset_input="alice/demo_5",
            episodes_raw="3",
            duration_raw="15",
            task_raw="Move the cube",
            dataset_dir_raw="/tmp/datasets",
            upload_enabled=True,
        )
        self.assertIsNone(error)
        assert req is not None and cmd is not None
        self.assertEqual(req.dataset_repo_id, "alice/demo_5")
        self.assertEqual(req.num_episodes, 3)
        self.assertTrue(req.upload_after_record)
        self.assertIn("--dataset.repo_id=alice/demo_5", cmd)

    def test_build_deploy_request_missing_model(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        req, cmd, updated, error = build_deploy_request_and_command(
            config=config,
            deploy_root_raw="/tmp/models",
            deploy_model_raw="/tmp/does_not_exist",
            eval_dataset_raw="alice/eval_1",
            eval_episodes_raw="2",
            eval_duration_raw="20",
            eval_task_raw="Test",
        )
        self.assertIsNone(req)
        self.assertIsNone(cmd)
        self.assertIsNone(updated)
        self.assertIsNotNone(error)

    def test_coerce_config_from_vars(self) -> None:
        base = dict(DEFAULT_CONFIG_VALUES)
        vars_map = {field["key"]: FakeVar(str(base[field["key"]])) for field in CONFIG_FIELDS}
        vars_map["camera_laptop_index"] = FakeVar("9")
        parsed, error = coerce_config_from_vars(base, vars_map, CONFIG_FIELDS)
        self.assertIsNone(error)
        assert parsed is not None
        self.assertEqual(parsed["camera_laptop_index"], 9)

    def test_build_deploy_request_success(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = f"{tmpdir}/model_a"
            import os

            os.makedirs(model_dir, exist_ok=True)
            req, cmd, updated, error = build_deploy_request_and_command(
                config=config,
                deploy_root_raw=tmpdir,
                deploy_model_raw=model_dir,
                eval_dataset_raw="alice/eval_7",
                eval_episodes_raw="4",
                eval_duration_raw="25",
                eval_task_raw="Pick and place",
            )

        self.assertIsNone(error)
        assert req is not None and cmd is not None and updated is not None
        self.assertEqual(req.eval_repo_id, "alice/eval_7")
        self.assertEqual(updated["last_model_name"], "model_a")
        self.assertIn("--policy.path=", " ".join(cmd))


if __name__ == "__main__":
    unittest.main()
