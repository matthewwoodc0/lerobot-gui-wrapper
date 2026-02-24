from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from robot_pipeline_app.artifacts import (
    build_deploy_notes_markdown,
    write_deploy_episode_spreadsheet,
    write_run_artifacts,
)
from robot_pipeline_app.constants import DEFAULT_CONFIG_VALUES


class ArtifactsTest(unittest.TestCase):
    def test_build_deploy_notes_markdown_includes_summary_and_episode_rows(self) -> None:
        markdown = build_deploy_notes_markdown(
            {
                "run_id": "deploy_1",
                "status": "success",
                "exit_code": 0,
                "model_path": "/tmp/model_a",
                "dataset_repo_id": "alice/eval_demo",
                "started_at_iso": "2026-02-24T12:00:00+00:00",
                "ended_at_iso": "2026-02-24T12:10:00+00:00",
                "duration_s": 600,
                "command": "python -m lerobot.scripts.lerobot_record --policy.path=/tmp/model_a",
                "deploy_episode_outcomes": {
                    "total_episodes": 2,
                    "episode_outcomes": [
                        {"episode": 1, "result": "success", "tags": ["left"], "note": "good pickup"},
                        {"episode": 2, "result": "unmarked", "tags": ["right"], "note": "interesting drift"},
                    ],
                },
                "deploy_notes_summary": "Overall rollout was stable.",
            }
        )
        self.assertIn("# Deployment Notes", markdown)
        self.assertIn("## Deployment Summary", markdown)
        self.assertIn("| 1 | Success | left | good pickup |", markdown)
        self.assertIn("| 2 | Unmarked | right | interesting drift |", markdown)
        self.assertIn("Overall rollout was stable.", markdown)

    def test_write_run_artifacts_deploy_writes_notes_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = dict(DEFAULT_CONFIG_VALUES)
            config["runs_dir"] = tmpdir
            started = datetime(2026, 2, 24, 12, 0, 0, tzinfo=timezone.utc)
            ended = datetime(2026, 2, 24, 12, 1, 0, tzinfo=timezone.utc)
            run_path = write_run_artifacts(
                config=config,
                mode="deploy",
                command=["python3", "-m", "lerobot.scripts.lerobot_record"],
                cwd=Path("/tmp"),
                started_at=started,
                ended_at=ended,
                exit_code=0,
                canceled=False,
                preflight_checks=[],
                output_lines=["ok"],
                dataset_repo_id="alice/eval_demo",
                model_path=Path("/tmp/model_a"),
                metadata_extra={
                    "deploy_episode_outcomes": {
                        "total_episodes": 2,
                        "episode_outcomes": [{"episode": 1, "result": "success", "tags": ["baseline"]}],
                    }
                },
            )
            self.assertIsNotNone(run_path)
            assert run_path is not None
            notes_path = run_path / "notes.md"
            self.assertTrue(notes_path.exists())
            text = notes_path.read_text(encoding="utf-8")
            self.assertIn("Deployment Notes", text)
            self.assertIn("Episode Outcomes", text)
            episode_csv = run_path / "episode_outcomes.csv"
            summary_csv = run_path / "episode_outcomes_summary.csv"
            self.assertTrue(episode_csv.exists())
            self.assertTrue(summary_csv.exists())
            episode_text = episode_csv.read_text(encoding="utf-8")
            self.assertIn("episode,status,is_success,is_failed,is_unmarked,is_pending", episode_text)
            self.assertIn("tag__baseline", episode_text)
            summary_text = summary_csv.read_text(encoding="utf-8")
            self.assertIn("metric,value", summary_text)
            self.assertIn("success_count", summary_text)
            self.assertIn("unmarked_count", summary_text)
            metadata = (run_path / "metadata.json").read_text(encoding="utf-8")
            self.assertIn("\"episode\": 2", metadata)
            self.assertIn("\"result\": \"unmarked\"", metadata)

    def test_write_deploy_episode_spreadsheet_includes_tag_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir)
            csv_path, summary_path = write_deploy_episode_spreadsheet(
                run_path,
                {
                    "deploy_episode_outcomes": {
                        "total_episodes": 2,
                        "episode_outcomes": [
                            {"episode": 1, "result": "success", "tags": ["left zone", "easy"]},
                            {"episode": 2, "result": "failed", "tags": ["hard"]},
                        ],
                    }
                },
            )
            self.assertIsNotNone(csv_path)
            self.assertIsNotNone(summary_path)
            assert csv_path is not None
            text = csv_path.read_text(encoding="utf-8")
            self.assertIn("tag__left_zone", text)
            self.assertIn("tag__easy", text)
            self.assertIn("tag__hard", text)


if __name__ == "__main__":
    unittest.main()
