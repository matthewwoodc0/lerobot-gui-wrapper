from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from robot_pipeline_app.gui_deploy_tab import (
    _default_pull_destination,
    _first_model_payload_candidate,
    _increment_pull_destination,
    _model_tree_node_kind,
    _needs_eval_prefix_quick_fix,
    _remote_path_name,
    _resolve_payload_path,
)


class GuiDeployTabHelpersTest(unittest.TestCase):
    def test_first_model_payload_candidate_returns_first(self) -> None:
        checks = [
            ("PASS", "Model payload", "ok"),
            ("WARN", "Model payload candidates", "/tmp/model_a, /tmp/model_b"),
        ]
        self.assertEqual(_first_model_payload_candidate(checks), "/tmp/model_a")

    def test_model_tree_node_kind_for_checkpoint_with_nested_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "checkpoint-020000"
            payload = checkpoint / "pretrained_model"
            payload.mkdir(parents=True, exist_ok=True)
            (payload / "config.json").write_text("{}\n", encoding="utf-8")
            (payload / "model.safetensors").write_text("x\n", encoding="utf-8")

            label, tag = _model_tree_node_kind(checkpoint)

        self.assertEqual(label, "Checkpoint -> model")
        self.assertEqual(tag, "resolved")

    def test_resolve_payload_path_returns_nested_pretrained_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "020000"
            payload = checkpoint / "pretrained_model"
            payload.mkdir(parents=True, exist_ok=True)
            (payload / "policy_config.json").write_text("{}\n", encoding="utf-8")
            (payload / "model.safetensors").write_text("x\n", encoding="utf-8")

            resolved = _resolve_payload_path(checkpoint)

        self.assertEqual(resolved, payload)

    def test_resolve_payload_path_prefers_latest_checkpoint_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_root = Path(tmpdir) / "model_a"
            older = model_root / "checkpoints" / "checkpoint-010000" / "pretrained_model"
            newer = model_root / "checkpoints" / "checkpoint-020000" / "pretrained_model"
            older.mkdir(parents=True, exist_ok=True)
            newer.mkdir(parents=True, exist_ok=True)
            (older / "config.json").write_text("{}\n", encoding="utf-8")
            (older / "model.safetensors").write_text("x\n", encoding="utf-8")
            (newer / "config.json").write_text("{}\n", encoding="utf-8")
            (newer / "model.safetensors").write_text("x\n", encoding="utf-8")

            resolved = _resolve_payload_path(model_root)

        self.assertEqual(resolved, newer)

    def test_needs_eval_prefix_quick_fix_for_bare_name_without_prefix(self) -> None:
        self.assertTrue(_needs_eval_prefix_quick_fix("alice", "run_1"))

    def test_needs_eval_prefix_quick_fix_for_repo_id_without_prefix(self) -> None:
        self.assertTrue(_needs_eval_prefix_quick_fix("alice", "alice/run_1"))

    def test_needs_eval_prefix_quick_fix_false_for_prefixed_bare_name(self) -> None:
        self.assertFalse(_needs_eval_prefix_quick_fix("alice", "eval_run_1"))

    def test_needs_eval_prefix_quick_fix_false_for_prefixed_repo_id(self) -> None:
        self.assertFalse(_needs_eval_prefix_quick_fix("alice", "alice/eval_run_1"))

    def test_remote_path_name_defaults_for_empty(self) -> None:
        self.assertEqual(_remote_path_name(""), "remote_model")

    def test_default_pull_destination_uses_trained_models_dir(self) -> None:
        destination = _default_pull_destination("/tmp/trained_models", "~/checkpoints/model_a")
        self.assertEqual(destination, Path("/tmp/trained_models/model_a"))

    def test_increment_pull_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "model"
            base.mkdir()
            second = Path(tmpdir) / "model_2"
            second.mkdir()
            candidate = _increment_pull_destination(base)
        self.assertEqual(candidate.name, "model_3")


if __name__ == "__main__":
    unittest.main()
