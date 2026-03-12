from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from robot_pipeline_app.workspace_lineage import build_lineage_graph, lineage_rows_for_selection
from robot_pipeline_app.workspace_provenance import write_workspace_provenance


class WorkspaceLineageTest(unittest.TestCase):
    def test_build_lineage_graph_connects_dataset_train_checkpoint_and_deploy(self) -> None:
        runs = [
            {
                "run_id": "train_1",
                "mode": "train",
                "dataset_repo_id": "alice/demo",
                "output_dir_resolved": "/tmp/train_outputs",
                "checkpoint_artifacts": [{"path": "/tmp/train_outputs/checkpoints/ckpt-1", "label": "ckpt-1"}],
            },
            {
                "run_id": "deploy_1",
                "mode": "deploy",
                "dataset_repo_id": "alice/eval_demo",
                "model_path": "/tmp/train_outputs/checkpoints/ckpt-1",
            },
        ]
        graph = build_lineage_graph(runs)
        self.assertTrue(any(node["kind"] == "dataset" and node["label"] == "alice/demo" for node in graph["nodes"]))
        self.assertTrue(any(edge["relation"] == "produces_checkpoint" for edge in graph["edges"]))
        self.assertTrue(any(edge["relation"] == "used_by_run" for edge in graph["edges"]))

    def test_lineage_rows_for_local_model_include_hf_source_and_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model_a"
            model_dir.mkdir(parents=True)
            write_workspace_provenance(model_dir, payload={"repo_id": "alice/model_a", "source": "huggingface"})
            rows = lineage_rows_for_selection(
                selection={"kind": "model", "scope": "local", "path": str(model_dir)},
                runs=[{"run_id": "deploy_1", "mode": "deploy", "model_path": str(model_dir), "_run_path": "/tmp/runs/deploy_1"}],
            )
        self.assertTrue(any(row["relation"] == "HF source" for row in rows))
        self.assertTrue(any(row["relation"] == "deploy run" for row in rows))
