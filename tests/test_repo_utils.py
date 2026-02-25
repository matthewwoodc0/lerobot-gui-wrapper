from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from urllib.error import URLError
from unittest.mock import patch

from robot_pipeline_app.repo_utils import (
    extract_dataset_repo_id_arg,
    get_hf_dataset_info,
    get_hf_model_info,
    has_eval_prefix,
    list_hf_datasets,
    list_hf_models,
    model_exists_on_hf,
    normalize_deploy_rerun_command,
    replace_dataset_repo_id_arg,
    resolve_unique_repo_id,
    suggest_eval_prefixed_repo_id,
)


class RepoUtilsTest(unittest.TestCase):
    def test_suggest_eval_prefixed_repo_id_adds_prefix_for_bare_name(self) -> None:
        repo_id, changed = suggest_eval_prefixed_repo_id("alice", "run_1")
        self.assertEqual(repo_id, "eval_run_1")
        self.assertTrue(changed)

    def test_suggest_eval_prefixed_repo_id_preserves_explicit_owner(self) -> None:
        repo_id, changed = suggest_eval_prefixed_repo_id("alice", "org/run_1")
        self.assertEqual(repo_id, "org/eval_run_1")
        self.assertTrue(changed)

    def test_suggest_eval_prefixed_repo_id_noop_when_prefixed(self) -> None:
        repo_id, changed = suggest_eval_prefixed_repo_id("alice", "org/eval_run_1")
        self.assertEqual(repo_id, "org/eval_run_1")
        self.assertFalse(changed)
        self.assertTrue(has_eval_prefix(repo_id))

    def test_resolve_unique_repo_id_increments_for_remote_collision(self) -> None:
        seen: list[str] = []

        def fake_exists(repo_id: str) -> bool | None:
            seen.append(repo_id)
            return repo_id.endswith("run_1")

        resolved, adjusted, checked_remote = resolve_unique_repo_id(
            username="alice",
            dataset_name_or_repo_id="run_1",
            local_roots=[],
            exists_fn=fake_exists,
        )
        self.assertEqual(resolved, "alice/run_2")
        self.assertTrue(adjusted)
        self.assertTrue(checked_remote)
        self.assertGreaterEqual(len(seen), 2)

    def test_resolve_unique_repo_id_increments_for_local_collision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "eval_1").mkdir(parents=True, exist_ok=True)
            resolved, adjusted, checked_remote = resolve_unique_repo_id(
                username="alice",
                dataset_name_or_repo_id="eval_1",
                local_roots=[root],
                exists_fn=lambda _: False,
            )
        self.assertEqual(resolved, "alice/eval_2")
        self.assertTrue(adjusted)
        self.assertTrue(checked_remote)

    def test_extract_and_replace_dataset_repo_id_arg(self) -> None:
        cmd = ["python3", "-m", "foo", "--dataset.repo_id=alice/eval_run_1"]
        match = extract_dataset_repo_id_arg(cmd)
        self.assertEqual(match, (3, "alice/eval_run_1"))
        updated = replace_dataset_repo_id_arg(cmd, "alice/eval_run_2")
        self.assertEqual(updated[3], "--dataset.repo_id=alice/eval_run_2")

    def test_normalize_deploy_rerun_command_iterates_for_collision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir)
            (data_root / "eval_run_1").mkdir(parents=True, exist_ok=True)
            cmd = ["python3", "-m", "lerobot.scripts.lerobot_record", "--dataset.repo_id=alice/eval_run_1"]
            updated, message = normalize_deploy_rerun_command(
                command_argv=cmd,
                username="alice",
                local_roots=[data_root],
                exists_fn=lambda _: False,
            )

        self.assertEqual(updated[-1], "--dataset.repo_id=alice/eval_run_2")
        self.assertIsNotNone(message)
        assert message is not None
        self.assertIn("iterated", message)

    def test_normalize_deploy_rerun_command_adds_eval_prefix(self) -> None:
        cmd = ["python3", "-m", "lerobot.scripts.lerobot_record", "--dataset.repo_id=alice/run_1"]
        updated, message = normalize_deploy_rerun_command(
            command_argv=cmd,
            username="alice",
            local_roots=[],
            exists_fn=lambda _: False,
        )
        self.assertEqual(updated[-1], "--dataset.repo_id=alice/eval_run_1")
        self.assertIsNotNone(message)
        assert message is not None
        self.assertIn("eval_", message)

    def test_list_hf_datasets_from_cached_payload(self) -> None:
        payload = [{"id": "alice/demo_1", "downloads": 12, "likes": 3}]
        with patch("robot_pipeline_app.repo_utils._hf_get_json", return_value=(payload, 200)):
            rows, error_text = list_hf_datasets("alice", limit=10)
        self.assertIsNone(error_text)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["repo_id"], "alice/demo_1")
        self.assertEqual(rows[0]["name"], "demo_1")

    def test_list_hf_models_from_cached_payload(self) -> None:
        payload = [{"id": "alice/model_a", "downloads": 10}]
        with patch("robot_pipeline_app.repo_utils._hf_get_json", return_value=(payload, 200)):
            rows, error_text = list_hf_models("alice", limit=10)
        self.assertIsNone(error_text)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["repo_id"], "alice/model_a")
        self.assertEqual(rows[0]["name"], "model_a")

    def test_get_hf_dataset_info_404(self) -> None:
        with patch("robot_pipeline_app.repo_utils._hf_get_json", return_value=(None, 404)):
            info, error_text = get_hf_dataset_info("alice/demo_1")
        self.assertIsNone(info)
        self.assertIn("not found", str(error_text).lower())

    def test_get_hf_model_info_404(self) -> None:
        with patch("robot_pipeline_app.repo_utils._hf_get_json", return_value=(None, 404)):
            info, error_text = get_hf_model_info("alice/model_1")
        self.assertIsNone(info)
        self.assertIn("not found", str(error_text).lower())

    def test_model_exists_on_hf_handles_http_errors(self) -> None:
        with patch("robot_pipeline_app.repo_utils.request.urlopen", side_effect=URLError("offline")):
            self.assertIsNone(model_exists_on_hf("alice/model_1"))


if __name__ == "__main__":
    unittest.main()
