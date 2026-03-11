from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robot_pipeline_app.deploy_workflow_helpers import build_model_browser_tree, build_model_upload_request, summarize_model_info


class DeployWorkflowHelpersTests(unittest.TestCase):
    def test_build_model_browser_tree_includes_nested_checkpoint_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            top_model = root / "policy_a"
            top_model.mkdir()
            (top_model / "config.json").write_text("{}", encoding="utf-8")
            (top_model / "model.safetensors").write_text("stub", encoding="utf-8")

            nested_root = root / "policy_b"
            nested_root.mkdir()
            checkpoint = nested_root / "checkpoint_100"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text("{}", encoding="utf-8")
            (checkpoint / "model.safetensors").write_text("stub", encoding="utf-8")

            nodes = build_model_browser_tree(root)

        self.assertEqual([node.label for node in nodes], ["policy_a", "policy_b"])
        self.assertEqual(nodes[0].kind, "Model")
        self.assertEqual(nodes[1].children[0].kind, "Model")

    def test_build_model_upload_request_builds_cli_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            local_model = Path(tmpdir)
            with (
                patch("robot_pipeline_app.deploy_workflow_helpers.shutil.which", return_value="/usr/bin/huggingface-cli"),
                patch("robot_pipeline_app.deploy_workflow_helpers.model_exists_on_hf", return_value=False),
            ):
                request, error = build_model_upload_request(
                    local_model_raw=str(local_model),
                    owner_raw="alice",
                    repo_name_raw="demo-model",
                )

        self.assertIsNone(error)
        assert request is not None
        self.assertEqual(request["repo_id"], "alice/demo-model")
        self.assertEqual(
            request["upload_cmd"],
            ["huggingface-cli", "upload", "alice/demo-model", str(local_model), "--repo-type", "model"],
        )

    def test_summarize_model_info_includes_structured_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload = root / "checkpoint_200" / "pretrained_model"
            payload.mkdir(parents=True)
            (payload / "config.json").write_text(
                '{"policy_family":"wall-x","policy_class":"vendor.wallx.WallXPolicy","plugin_package":"vendor","fps":20,"robot_type":"so101_follower","camera_keys":["front","side"],"output_shapes":{"action":{"shape":[6]}}}\n',
                encoding="utf-8",
            )
            (payload / "model.safetensors").write_text("stub", encoding="utf-8")

            summary = summarize_model_info(root / "checkpoint_200")

        self.assertIn("Policy family/class: Wall-X / vendor.wallx.WallXPolicy", summary)
        self.assertIn("Plugin package: vendor", summary)
        self.assertIn("Cameras: ['front', 'side']", summary)


if __name__ == "__main__":
    unittest.main()
