from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from robot_pipeline_app.gui_terminal_shell import _ActiveCommand, GuiTerminalShell, is_sensitive_command


class _RootStub:
    def after(self, _delay: int, callback: object, *args: object) -> None:
        callback(*args)  # type: ignore[misc]


class GuiTerminalShellTest(unittest.TestCase):
    def test_is_sensitive_command_detects_secret_patterns(self) -> None:
        self.assertTrue(is_sensitive_command("export HF_TOKEN=abcd"))
        self.assertTrue(is_sensitive_command("python train.py --api-key abc"))
        self.assertTrue(is_sensitive_command("my_password=abc"))

    def test_is_sensitive_command_allows_regular_commands(self) -> None:
        self.assertFalse(is_sensitive_command("ls -la"))
        self.assertFalse(is_sensitive_command("python3 robot_pipeline.py history"))

    def _build_shell(self, logs: list[str]) -> GuiTerminalShell:
        config = {"lerobot_dir": "/tmp", "runs_dir": "/tmp/robot_pipeline_test_runs"}
        return GuiTerminalShell(
            root=_RootStub(),
            config=config,
            append_log=logs.append,
            is_pipeline_active=lambda: False,
            send_pipeline_stdin=lambda _text: (True, ""),
            on_artifact_written=None,
        )

    @patch("robot_pipeline_app.gui_terminal_shell.write_run_artifacts")
    def test_abort_active_command_persists_history_for_non_sensitive_command(self, write_artifacts: object) -> None:
        logs: list[str] = []
        shell = self._build_shell(logs)
        shell._last_known_cwd = Path("/tmp")
        shell._active_command = _ActiveCommand(
            command="echo hello",
            started_at=datetime.now(timezone.utc),
            output_lines=["$ echo hello"],
            persist_history=True,
            command_id=1,
        )

        shell._abort_active_command("Shell died unexpectedly.", exit_code=1)

        self.assertIsNone(shell._active_command)
        self.assertIn("Shell died unexpectedly.", logs)
        self.assertTrue(write_artifacts.called)  # type: ignore[attr-defined]
        called_kwargs = write_artifacts.call_args.kwargs  # type: ignore[attr-defined]
        self.assertEqual("shell", called_kwargs["mode"])
        self.assertEqual(1, called_kwargs["exit_code"])

    @patch("robot_pipeline_app.gui_terminal_shell.write_run_artifacts")
    def test_abort_active_command_skips_history_for_sensitive_command(self, write_artifacts: object) -> None:
        logs: list[str] = []
        shell = self._build_shell(logs)
        shell._active_command = _ActiveCommand(
            command="export HF_TOKEN=secret",
            started_at=datetime.now(timezone.utc),
            output_lines=["$ export HF_TOKEN=secret"],
            persist_history=False,
            command_id=2,
        )

        shell._abort_active_command("Shell exited.", exit_code=1)

        self.assertIsNone(shell._active_command)
        self.assertIn("Shell exited.", logs)
        self.assertIn("history persistence skipped", " ".join(logs).lower())
        write_artifacts.assert_not_called()  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()
