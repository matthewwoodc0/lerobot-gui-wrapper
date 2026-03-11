from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from robot_pipeline_app.model_metadata import extract_model_metadata, format_model_metadata_summary


class ModelMetadataTests(unittest.TestCase):
    def test_extract_model_metadata_reads_policy_plugin_runtime_labels_and_rtc(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "config.json").write_text(
                """
                {
                  "fps": 30,
                  "robot_type": "unitree_g1",
                  "policy_family": "pi0-fast",
                  "policy_class": "acme_pi0.policy.Pi0FastPolicy",
                  "plugin_package": "acme_pi0",
                  "camera_keys": ["front", "wrist"],
                  "stats": {"action": {"min": [0, 0], "max": [1, 1]}},
                  "runtime": {"env": "isaaclab", "supports_rtc": true},
                  "output_shapes": {"action": {"shape": [29]}}
                }
                """.strip()
                + "\n",
                encoding="utf-8",
            )

            metadata = extract_model_metadata(model_dir)

        self.assertEqual(metadata.policy_family, "Pi0-FAST")
        self.assertEqual(metadata.plugin_package, "acme_pi0")
        self.assertEqual(metadata.robot_type, "unitree_g1")
        self.assertEqual(metadata.action_dim, 29)
        self.assertEqual(metadata.camera_keys, ("front", "wrist"))
        self.assertTrue(metadata.supports_rtc)
        self.assertTrue(metadata.normalization_present)
        self.assertIn("IsaacLab", metadata.runtime_labels)
        self.assertEqual(metadata.metadata_source, "detected via: config.json")

    def test_format_model_metadata_summary_surfaces_structured_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload = root / "pretrained_model"
            payload.mkdir()
            (payload / "config.json").write_text(
                '{"policy_family":"x-vla","policy_class":"vendor_pkg.xvla.XVlaPolicy","plugin_package":"vendor_pkg","fps":15,"robot_type":"so101_follower","camera_keys":["front"],"output_shapes":{"action":{"shape":[6]}}}\n',
                encoding="utf-8",
            )
            (payload / "model.safetensors").write_text("stub\n", encoding="utf-8")

            summary = format_model_metadata_summary(root, deploy_payload=payload)

        self.assertIn("Policy family/class: X-VLA / vendor_pkg.xvla.XVlaPolicy", summary)
        self.assertIn("Plugin package: vendor_pkg", summary)
        self.assertIn("Robot type: so101_follower", summary)
        self.assertIn("Action dim: 6", summary)
        self.assertIn("Cameras: ['front']", summary)


if __name__ == "__main__":
    unittest.main()
