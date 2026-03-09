from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch
import tempfile

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
    def _build_shell(self, logs: list[str], *, terminal_chunks: list[str] | None = None) -> GuiTerminalShell:
        config = {"lerobot_dir": "/tmp", "runs_dir": "/tmp/robot_pipeline_test_runs"}
        return GuiTerminalShell(
            root=_RootStub(),
            config=config,
            append_log=logs.append,
            is_pipeline_active=lambda: False,
            send_pipeline_stdin=lambda _text: (True, ""),
            append_terminal_output=(terminal_chunks.append if terminal_chunks is not None else None),
            on_artifact_written=None,
        )

    # ------------------------------------------------------------------
    # Raw-terminal mode: is_busy() must never block pipelines
    # ------------------------------------------------------------------

    def test_is_busy_always_returns_false(self) -> None:
        logs: list[str] = []
        shell = self._build_shell(logs)
        self.assertFalse(shell.is_busy())

    def test_is_busy_still_false_when_process_is_alive(self) -> None:
        logs: list[str] = []
        shell = self._build_shell(logs)
        shell._process = _FakeShellProcess(poll_values=[None])  # type: ignore[assignment]
        self.assertFalse(shell.is_busy())

    # ------------------------------------------------------------------
    # handle_terminal_input / submit routing
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

    def test_handle_terminal_input_uses_shell_writer_when_idle(self) -> None:
        logs: list[str] = []
        shell = self._build_shell(logs)
        with patch.object(shell, "start", return_value=(True, "")), patch.object(
            shell, "_write_payload", return_value=True
        ) as write_payload:
            ok, message = shell.handle_terminal_input(b"ls\n")
        self.assertTrue(ok)
        self.assertEqual(message, "")
        write_payload.assert_called_once_with(b"ls\n")

    def test_handle_terminal_submit_adds_newline(self) -> None:
        logs: list[str] = []
        shell = self._build_shell(logs)
        captured: list[bytes] = []

        def _capture(payload: bytes) -> tuple[bool, str]:
            captured.append(payload)
            return True, ""

        with patch.object(shell, "handle_terminal_input", side_effect=_capture):
            ok, _ = shell.handle_terminal_submit("echo hi")
        self.assertTrue(ok)
        self.assertEqual(captured, [b"echo hi\n"])

    def test_shell_environment_defaults_term_when_missing(self) -> None:
        logs: list[str] = []
        shell = self._build_shell(logs)

        with patch.dict(os.environ, {}, clear=True):
            env = shell._shell_environment()

        self.assertEqual(env["TERM"], "dumb")

    def test_shell_environment_preserves_existing_term(self) -> None:
        logs: list[str] = []
        shell = self._build_shell(logs)

        with patch.dict(os.environ, {"TERM": "xterm-256color"}, clear=True):
            env = shell._shell_environment()

        self.assertEqual(env["TERM"], "xterm-256color")

    @patch("robot_pipeline_app.gui_terminal_shell.fcntl")
    @patch("robot_pipeline_app.gui_terminal_shell.struct")
    @patch("robot_pipeline_app.gui_terminal_shell.termios")
    def test_resize_terminal_applies_pty_window_size(self, termios_mod: object, struct_mod: object, fcntl_mod: object) -> None:
        logs: list[str] = []
        shell = self._build_shell(logs)
        shell._master_fd = 42
        struct_mod.pack.return_value = b"winsize"  # type: ignore[attr-defined]
        termios_mod.TIOCSWINSZ = 99  # type: ignore[attr-defined]

        shell.resize_terminal(120, 40)

        struct_mod.pack.assert_called_once_with("HHHH", 40, 120, 0, 0)  # type: ignore[attr-defined]
        fcntl_mod.ioctl.assert_called_once_with(42, 99, b"winsize")  # type: ignore[attr-defined]

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

    def test_activation_command_prefers_custom_saved_command(self) -> None:
        logs: list[str] = []
        config = {
            "lerobot_dir": "/home/user/lerobot",
            "setup_venv_activate_cmd": "conda activate lerobot",
            "lerobot_venv_dir": "/home/user/lerobot/lerobot_env",
        }
        shell = GuiTerminalShell(
            root=_RootStub(),
            config=config,
            append_log=logs.append,
            is_pipeline_active=lambda: False,
            send_pipeline_stdin=lambda _text: (True, ""),
        )
        command, source = shell._activation_command()
        self.assertEqual(command, "conda activate lerobot")
        self.assertEqual(source, "config:setup_venv_activate_cmd")

    def test_activation_command_reports_missing_when_no_activate_script(self) -> None:
        logs: list[str] = []
        config = {
            "lerobot_dir": "/tmp",
            "lerobot_venv_dir": "/definitely/missing/env",
        }
        shell = GuiTerminalShell(
            root=_RootStub(),
            config=config,
            append_log=logs.append,
            is_pipeline_active=lambda: False,
            send_pipeline_stdin=lambda _text: (True, ""),
        )
        command, source = shell._activation_command()
        self.assertIsNone(command)
        self.assertIn("missing activate script", source)

    def test_activation_command_uses_conda_base_activate_for_prefix(self) -> None:
        logs: list[str] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            conda_env = base / "miniforge3" / "envs" / "lerobot"
            (conda_env / "conda-meta").mkdir(parents=True, exist_ok=True)
            base_activate = base / "miniforge3" / "bin" / "activate"
            base_activate.parent.mkdir(parents=True, exist_ok=True)
            base_activate.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

            config = {
                "lerobot_dir": "/tmp",
                "lerobot_venv_dir": str(conda_env),
            }
            shell = GuiTerminalShell(
                root=_RootStub(),
                config=config,
                append_log=logs.append,
                is_pipeline_active=lambda: False,
                send_pipeline_stdin=lambda _text: (True, ""),
            )
            command, source = shell._activation_command()

        self.assertIsNotNone(command)
        assert command is not None
        self.assertIn(str(base_activate), command)
        self.assertIn(str(conda_env), command)
        self.assertEqual(source, "config:lerobot_venv_dir(conda-prefix)")

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
    # ANSI stripping fallback helper
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
