from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from robot_pipeline_app.repo_utils import resolve_unique_repo_id


class RepoUtilsTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
