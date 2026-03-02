from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from robot_pipeline_app.gui_terminal_shell import GuiTerminalShell


class _RootStub:
    def after(self, _delay: int, callback: object, *args: object) -> None:
        callback(*args)  # type: ignore[misc]


class _FakeShellProcess:
    def __init__(self, *, pid: int = 2222, poll_values: list[int | None] | None = None) -> None:
        self.pid = pid
        self._poll_values = list(poll_values or [None])
        self.wait_called = False

    def poll(self) -> int | None:
        if len(self._poll_values) > 1:
            return self._poll_values.pop(0)
        return self._poll_values[0]

    def wait(self, timeout: float | None = None) -> int:
        self.wait_called = True
        return 0


class GuiTerminalShellTest(unittest.TestCase):
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

    # ------------------------------------------------------------------
    # Raw-terminal mode: is_busy() must never block pipelines
    # ------------------------------------------------------------------

    def test_is_busy_always_returns_false(self) -> None:
        """Raw terminal never blocks pipeline runs with a busy flag."""
        logs: list[str] = []
        shell = self._build_shell(logs)
        self.assertFalse(shell.is_busy())

    def test_is_busy_still_false_when_process_is_alive(self) -> None:
        """Even with an active PTY process is_busy() returns False."""
        logs: list[str] = []
        shell = self._build_shell(logs)
        shell._process = _FakeShellProcess(poll_values=[None])  # type: ignore[assignment]
        self.assertFalse(shell.is_busy())

    # ------------------------------------------------------------------
    # handle_terminal_submit: forwards to pipeline stdin when active
    # ------------------------------------------------------------------

    def test_handle_terminal_submit_forwards_to_pipeline_when_active(self) -> None:
        forwarded: list[str] = []
        config = {"lerobot_dir": "/tmp"}
        shell = GuiTerminalShell(
            root=_RootStub(),
            config=config,
            append_log=[].append,
            is_pipeline_active=lambda: True,
            send_pipeline_stdin=lambda text: (forwarded.append(text), (True, ""))[1],
        )
        ok, _ = shell.handle_terminal_submit("hello")
        self.assertTrue(ok)
        self.assertIn("hello\n", forwarded)

    # ------------------------------------------------------------------
    # send_interrupt: delegates to pipeline when active
    # ------------------------------------------------------------------

    def test_send_interrupt_delegates_to_pipeline_when_active(self) -> None:
        sent: list[str] = []
        config = {"lerobot_dir": "/tmp"}
        shell = GuiTerminalShell(
            root=_RootStub(),
            config=config,
            append_log=[].append,
            is_pipeline_active=lambda: True,
            send_pipeline_stdin=lambda text: (sent.append(text), (True, ""))[1],
        )
        ok, _ = shell.send_interrupt()
        self.assertTrue(ok)
        self.assertIn("\x03", sent)

    # ------------------------------------------------------------------
    # Venv auto-activation path resolution
    # ------------------------------------------------------------------

    def test_get_venv_dir_uses_config_key(self) -> None:
        logs: list[str] = []
        config = {
            "lerobot_dir": "/tmp",
            "lerobot_venv_dir": "/opt/my_venv",
        }
        shell = GuiTerminalShell(
            root=_RootStub(),
            config=config,
            append_log=logs.append,
            is_pipeline_active=lambda: False,
            send_pipeline_stdin=lambda _text: (True, ""),
        )
        self.assertEqual(shell._get_venv_dir(), Path("/opt/my_venv"))

    def test_get_venv_dir_falls_back_to_default(self) -> None:
        logs: list[str] = []
        config = {"lerobot_dir": "/home/user/lerobot"}
        shell = GuiTerminalShell(
            root=_RootStub(),
            config=config,
            append_log=logs.append,
            is_pipeline_active=lambda: False,
            send_pipeline_stdin=lambda _text: (True, ""),
        )
        venv_dir = shell._get_venv_dir()
        self.assertEqual(venv_dir, Path("/home/user/lerobot/lerobot_env"))

    # ------------------------------------------------------------------
    # Shell process termination helpers
    # ------------------------------------------------------------------

    @patch("robot_pipeline_app.gui_terminal_shell.kill_process_tree")
    @patch("robot_pipeline_app.gui_terminal_shell.terminate_process_tree")
    def test_terminate_shell_process_skips_force_kill_when_process_exits(
        self, terminate_tree: object, kill_tree: object
    ) -> None:
        logs: list[str] = []
        shell = self._build_shell(logs)
        process = _FakeShellProcess(poll_values=[0])

        shell._terminate_shell_process(process, reason="Shell shutdown requested.")

        terminate_tree.assert_called_once()  # type: ignore[attr-defined]
        kill_tree.assert_not_called()  # type: ignore[attr-defined]
        self.assertFalse(process.wait_called)

    @patch("robot_pipeline_app.gui_terminal_shell.time.sleep", return_value=None)
    @patch("robot_pipeline_app.gui_terminal_shell.kill_process_tree")
    @patch("robot_pipeline_app.gui_terminal_shell.terminate_process_tree")
    @patch("robot_pipeline_app.gui_terminal_shell._CANCEL_TIMEOUT_SECONDS", 0.01)
    def test_terminate_shell_process_force_kills_after_timeout(
        self,
        terminate_tree: object,
        kill_tree: object,
        _sleep: object,
    ) -> None:
        logs: list[str] = []
        shell = self._build_shell(logs)
        process = _FakeShellProcess(poll_values=[None, None, None])

        ticks = {"value": -1}

        def _fake_monotonic() -> float:
            ticks["value"] += 1
            return float(ticks["value"]) * 0.02

        with patch("robot_pipeline_app.gui_terminal_shell.time.monotonic", side_effect=_fake_monotonic):
            shell._terminate_shell_process(process, reason="Shell shutdown requested.")

        terminate_tree.assert_called_once()  # type: ignore[attr-defined]
        kill_tree.assert_called_once()  # type: ignore[attr-defined]
        self.assertTrue(process.wait_called)

    # ------------------------------------------------------------------
    # Ready-marker gating: startup noise suppression
    # ------------------------------------------------------------------

    def test_output_suppressed_before_ready_marker(self) -> None:
        """Lines emitted before the ready marker must be silently dropped."""
        logs: list[str] = []
        shell = self._build_shell(logs)
        shell._shell_ready = False
        shell._handle_output_line("zsh: some startup noise")
        shell._handle_output_line("% matthewwoodcock@host lerobot %")
        self.assertEqual(logs, [])

    def test_ready_marker_switches_output_on(self) -> None:
        """Seeing the marker enables output; the marker line itself is not logged."""
        from robot_pipeline_app.gui_terminal_shell import _SHELL_READY_MARKER

        logs: list[str] = []
        shell = self._build_shell(logs)
        shell._shell_ready = False
        shell._handle_output_line(_SHELL_READY_MARKER)
        self.assertTrue(shell._shell_ready)
        self.assertEqual(logs, [])  # marker line itself must not be logged

    def test_output_forwarded_after_ready_marker(self) -> None:
        """Lines after the marker must appear in the log."""
        logs: list[str] = []
        shell = self._build_shell(logs)
        shell._shell_ready = True
        shell._handle_output_line("hello from shell")
        self.assertIn("hello from shell", logs)

    # ------------------------------------------------------------------
    # ANSI stripping
    # ------------------------------------------------------------------

    def test_clean_output_line_strips_ansi_codes(self) -> None:
        logs: list[str] = []
        shell = self._build_shell(logs)
        raw = "\x1b[32mhello\x1b[0m world"
        self.assertEqual(shell._clean_output_line(raw), "hello world")

    def test_clean_output_line_strips_carriage_returns(self) -> None:
        logs: list[str] = []
        shell = self._build_shell(logs)
        self.assertEqual(shell._clean_output_line("line\r\n"), "line")


if __name__ == "__main__":
    unittest.main()
