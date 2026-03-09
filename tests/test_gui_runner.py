from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from robot_pipeline_app.gui_runner import create_run_controller
from robot_pipeline_app.types import DiagnosticEvent


class _RootStub:
    def after(self, _delay: int, callback: object, *args: object) -> str:
        callback(*args)  # type: ignore[misc]
        return "after-id"


class _StatusVar:
    def __init__(self) -> None:
        self.value = ""

    def set(self, value: str) -> None:
        self.value = value


class _LogPanelStub:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.running_states: list[bool] = []

    def append_log(self, line: str) -> None:
        self.lines.append(str(line))

    def set_running_state(self, active: bool) -> None:
        self.running_states.append(bool(active))


class _MessageBoxStub:
    def __init__(self) -> None:
        self.info_calls: list[tuple[str, str]] = []
        self.error_calls: list[tuple[str, str]] = []

    def showinfo(self, title: str, message: str) -> None:
        self.info_calls.append((title, message))

    def showerror(self, title: str, message: str) -> None:
        self.error_calls.append((title, message))


class _FakeRunControlPopout:
    configure_calls: list[dict[str, Any]] = []
    theme_calls: list[dict[str, str]] = []

    def __init__(self, *, root: Any, colors: dict[str, str], on_send_key: Any, on_cancel: Any) -> None:
        self.on_cancel = on_cancel

    def configure_record_camera_feed(
        self,
        *,
        config: dict[str, Any],
        cv2_probe_ok: bool,
        cv2_probe_error: str,
        append_log: Any,
        background_jobs: Any | None = None,
    ) -> None:
        type(self).configure_calls.append(
            {
                "config": config,
                "cv2_probe_ok": cv2_probe_ok,
                "cv2_probe_error": cv2_probe_error,
                "append_log": append_log,
                "background_jobs": background_jobs,
            }
        )

    def hide(self) -> None:
        return

    def start_run(self, run_mode: str, expected_episodes: int | None, expected_seconds: int | None) -> None:
        return

    def handle_output_line(self, _line: str) -> None:
        return

    def get_episode_outcome_summary(self) -> None:
        return None

    def apply_theme(self, colors: dict[str, str]) -> None:
        type(self).theme_calls.append(colors)


class _FakeTeleopRunPopout:
    theme_calls: list[dict[str, str]] = []

    def __init__(self, *, root: Any, colors: dict[str, str], on_cancel: Any) -> None:
        self.on_cancel = on_cancel

    def hide(self) -> None:
        return

    def start_run(self, **_: Any) -> None:
        return

    def mark_startup_complete(self) -> None:
        return

    def apply_theme(self, colors: dict[str, str]) -> None:
        type(self).theme_calls.append(colors)


class _StdinRecorder:
    def __init__(self) -> None:
        self.writes: list[str] = []

    def write(self, data: object) -> int:
        text = str(data)
        self.writes.append(text)
        return len(text)

    def flush(self) -> None:
        return


class _ProcessStub:
    def __init__(self, stdin: _StdinRecorder | None = None) -> None:
        self._poll_value: int | None = None
        self.stdin = stdin

    def poll(self) -> int | None:
        return self._poll_value


class _ThreadStub:
    pass


class GuiRunnerCancelSemanticsTests(unittest.TestCase):
    def _build_controller(self, *, on_run_failure: Any = None) -> tuple[Any, dict[str, Any], _StatusVar, _LogPanelStub, list[str], _MessageBoxStub]:
        root = _RootStub()
        status_var = _StatusVar()
        log_panel = _LogPanelStub()
        messagebox = _MessageBoxStub()
        dot_colors: list[str] = []
        colors = {"running": "#f0a500", "error": "#ef4444", "ready": "#22c55e"}

        controller = create_run_controller(
            root=root,
            config={},
            colors=colors,
            status_var=status_var,
            set_status_dot=dot_colors.append,
            log_panel=log_panel,
            messagebox=messagebox,
            action_buttons=[],
            last_command_state={"value": ""},
            on_run_failure=on_run_failure,
        )
        return controller, colors, status_var, log_panel, dot_colors, messagebox

    @patch("robot_pipeline_app.gui_runner.TeleopRunPopout", _FakeTeleopRunPopout)
    @patch("robot_pipeline_app.gui_runner.RunControlPopout", _FakeRunControlPopout)
    def test_configure_record_camera_feed_delegates_to_run_popout(self) -> None:
        _FakeRunControlPopout.configure_calls.clear()
        controller, _colors, _status_var, log_panel, _dot_colors, _messagebox = self._build_controller()
        background_jobs = object()

        controller.configure_record_camera_feed(
            {"camera_fps": 15},
            True,
            "",
            log_panel.append_log,
            background_jobs,
        )

        self.assertEqual(len(_FakeRunControlPopout.configure_calls), 1)
        call = _FakeRunControlPopout.configure_calls[0]
        self.assertEqual(call["config"]["camera_fps"], 15)
        self.assertTrue(call["cv2_probe_ok"])
        self.assertIs(getattr(call["append_log"], "__self__", None), log_panel)
        self.assertIs(call["background_jobs"], background_jobs)

    @patch("robot_pipeline_app.gui_runner.TeleopRunPopout", _FakeTeleopRunPopout)
    @patch("robot_pipeline_app.gui_runner.RunControlPopout", _FakeRunControlPopout)
    def test_apply_theme_delegates_to_both_popouts(self) -> None:
        _FakeRunControlPopout.theme_calls.clear()
        _FakeTeleopRunPopout.theme_calls.clear()
        controller, _colors, _status_var, _log_panel, _dot_colors, _messagebox = self._build_controller()
        updated = {"theme_mode": "light", "panel": "#ffffff"}

        controller.apply_theme(updated)

        self.assertEqual(_FakeRunControlPopout.theme_calls, [updated])
        self.assertEqual(_FakeTeleopRunPopout.theme_calls, [updated])

    @patch("robot_pipeline_app.gui_runner.explain_runtime_slowdown", return_value=[])
    @patch("robot_pipeline_app.gui_runner.diagnose_deploy_failure_events", return_value=[])
    @patch("robot_pipeline_app.gui_runner.write_run_artifacts", return_value=None)
    @patch("robot_pipeline_app.gui_runner.TeleopRunPopout", _FakeTeleopRunPopout)
    @patch("robot_pipeline_app.gui_runner.RunControlPopout", _FakeRunControlPopout)
    def test_cancel_zero_exit_reports_canceled_in_callback_and_artifacts(self, write_run_artifacts: object, _deploy_diag: object, _runtime_diag: object) -> None:
        controller, _, _status_var, log_panel, _dot_colors, _messagebox = self._build_controller()
        hooks: dict[str, Any] = {}

        def fake_run_process_streaming(**kwargs: Any) -> _ThreadStub:
            process = _ProcessStub()
            kwargs["on_process_started"](process)
            hooks["on_complete"] = kwargs["on_complete"]
            return _ThreadStub()

        with patch("robot_pipeline_app.gui_runner.run_process_streaming", side_effect=fake_run_process_streaming):
            completion: list[tuple[int, bool]] = []
            controller.run_process_async(
                cmd=["python3", "-c", "print('ok')"],
                cwd=Path("/tmp"),
                complete_callback=lambda return_code, canceled: completion.append((return_code, canceled)),
                run_mode="run",
            )
            controller.cancel_active_run()
            hooks["on_complete"](0)

        self.assertEqual(completion, [(0, True)])
        self.assertTrue(any("Command canceled by user." in line for line in log_panel.lines))
        kwargs = write_run_artifacts.call_args.kwargs  # type: ignore[attr-defined]
        self.assertEqual(kwargs["exit_code"], 0)
        self.assertTrue(kwargs["canceled"])

    @patch("robot_pipeline_app.gui_runner.explain_runtime_slowdown", return_value=[])
    @patch("robot_pipeline_app.gui_runner.diagnose_deploy_failure_events", return_value=[])
    @patch("robot_pipeline_app.gui_runner.write_run_artifacts", return_value=None)
    @patch("robot_pipeline_app.gui_runner.TeleopRunPopout", _FakeTeleopRunPopout)
    @patch("robot_pipeline_app.gui_runner.RunControlPopout", _FakeRunControlPopout)
    def test_cancel_nonzero_exit_uses_canceled_status_and_skips_failure_callback(self, _write_run_artifacts: object, _deploy_diag: object, _runtime_diag: object) -> None:
        on_failure_calls: list[bool] = []
        controller, colors, status_var, _log_panel, dot_colors, _messagebox = self._build_controller(
            on_run_failure=lambda: on_failure_calls.append(True)
        )
        hooks: dict[str, Any] = {}

        def fake_run_process_streaming(**kwargs: Any) -> _ThreadStub:
            process = _ProcessStub()
            kwargs["on_process_started"](process)
            hooks["on_complete"] = kwargs["on_complete"]
            return _ThreadStub()

        with patch("robot_pipeline_app.gui_runner.run_process_streaming", side_effect=fake_run_process_streaming):
            controller.run_process_async(
                cmd=["python3", "-c", "print('ok')"],
                cwd=Path("/tmp"),
                complete_callback=None,
                run_mode="run",
            )
            controller.cancel_active_run()
            hooks["on_complete"](1)

        self.assertEqual(status_var.value, "Command canceled.")
        self.assertEqual(on_failure_calls, [])
        self.assertTrue(dot_colors)
        self.assertEqual(dot_colors[-1], colors["ready"])

    @patch("robot_pipeline_app.gui_runner.explain_runtime_slowdown", return_value=[])
    @patch("robot_pipeline_app.gui_runner.diagnose_deploy_failure_events", return_value=[])
    @patch("robot_pipeline_app.gui_runner.write_run_artifacts", return_value=None)
    @patch("robot_pipeline_app.gui_runner.TeleopRunPopout", _FakeTeleopRunPopout)
    @patch("robot_pipeline_app.gui_runner.RunControlPopout", _FakeRunControlPopout)
    def test_teleop_av1_decode_spam_is_collapsed_to_single_notice(self, _write_run_artifacts: object, _deploy_diag: object, _runtime_diag: object) -> None:
        controller, _colors, _status_var, log_panel, _dot_colors, _messagebox = self._build_controller()

        def fake_run_process_streaming(**kwargs: Any) -> _ThreadStub:
            process = _ProcessStub()
            kwargs["on_process_started"](process)
            kwargs["on_line"]("Failed to get pixel format for AV1 stream")
            kwargs["on_line"]("Missing sequence header while decoding AV1")
            kwargs["on_line"]("teleop still running")
            kwargs["on_complete"](0)
            return _ThreadStub()

        with patch("robot_pipeline_app.gui_runner.run_process_streaming", side_effect=fake_run_process_streaming):
            controller.run_process_async(
                cmd=["python3", "-m", "lerobot.teleoperate"],
                cwd=Path("/tmp"),
                complete_callback=None,
                run_mode="teleop",
            )

        warning_lines = [line for line in log_panel.lines if "Teleop media decode fallback" in line]
        self.assertEqual(len(warning_lines), 1)
        self.assertFalse(any("failed to get pixel format" in line.lower() for line in log_panel.lines))
        self.assertFalse(any("missing sequence header" in line.lower() for line in log_panel.lines))
        self.assertTrue(any("teleop still running" in line for line in log_panel.lines))

    @patch("robot_pipeline_app.gui_runner.explain_runtime_slowdown", return_value=[])
    @patch("robot_pipeline_app.gui_runner.diagnose_deploy_failure_events", return_value=[])
    @patch("robot_pipeline_app.gui_runner.write_run_artifacts", return_value=None)
    @patch("robot_pipeline_app.gui_runner.TeleopRunPopout", _FakeTeleopRunPopout)
    @patch("robot_pipeline_app.gui_runner.RunControlPopout", _FakeRunControlPopout)
    def test_teleop_auto_accepts_saved_calibration_prompt_once_per_id(self, _write_run_artifacts: object, _deploy_diag: object, _runtime_diag: object) -> None:
        controller, _colors, _status_var, log_panel, _dot_colors, _messagebox = self._build_controller()
        stdin_recorder = _StdinRecorder()
        calibration_prompt = (
            "Press ENTER to use provided calibration file associated with the id grey_follower, "
            "or type 'c' and press ENTER to run calibration: "
        )

        def fake_run_process_streaming(**kwargs: Any) -> _ThreadStub:
            process = _ProcessStub(stdin=stdin_recorder)
            kwargs["on_process_started"](process)
            kwargs["on_line"](calibration_prompt)
            kwargs["on_line"](calibration_prompt)
            kwargs["on_complete"](0)
            return _ThreadStub()

        with patch("robot_pipeline_app.gui_runner.run_process_streaming", side_effect=fake_run_process_streaming):
            controller.run_process_async(
                cmd=[
                    "python3",
                    "-m",
                    "lerobot.teleoperate",
                    "--robot.calibration_dir=/tmp/calibration",
                    "--teleop.calibration_dir=/tmp/calibration",
                ],
                cwd=Path("/tmp"),
                complete_callback=None,
                run_mode="teleop",
            )

        self.assertEqual("".join(stdin_recorder.writes).count("\n"), 1)
        self.assertTrue(
            any("auto-sent ENTER to use the selected saved calibration file" in line for line in log_panel.lines)
        )

    @patch("robot_pipeline_app.gui_runner.explain_runtime_slowdown", return_value=[])
    @patch(
        "robot_pipeline_app.gui_runner.diagnose_runtime_failure_events",
        return_value=[
            DiagnosticEvent(
                level="FAIL",
                code="SER-PORT_ACCESS",
                name="Serial access",
                detail="check ports and calibration",
                fix="",
                docs_ref="docs/error-catalog.md#ser",
            )
        ],
    )
    @patch("robot_pipeline_app.gui_runner.diagnose_deploy_failure_events", return_value=[])
    @patch("robot_pipeline_app.gui_runner.write_run_artifacts", return_value=None)
    @patch("robot_pipeline_app.gui_runner.TeleopRunPopout", _FakeTeleopRunPopout)
    @patch("robot_pipeline_app.gui_runner.RunControlPopout", _FakeRunControlPopout)
    def test_teleop_does_not_auto_accept_calibration_prompt_without_explicit_calibration_dir(self, _write_run_artifacts: object, _deploy_diag: object, _runtime_failure: object, _runtime_diag: object) -> None:
        controller, _colors, _status_var, _log_panel, _dot_colors, _messagebox = self._build_controller()
        stdin_recorder = _StdinRecorder()
        calibration_prompt = (
            "Press ENTER to use provided calibration file associated with the id grey_follower, "
            "or type 'c' and press ENTER to run calibration: "
        )

        def fake_run_process_streaming(**kwargs: Any) -> _ThreadStub:
            process = _ProcessStub(stdin=stdin_recorder)
            kwargs["on_process_started"](process)
            kwargs["on_line"](calibration_prompt)
            kwargs["on_complete"](0)
            return _ThreadStub()

        with patch("robot_pipeline_app.gui_runner.run_process_streaming", side_effect=fake_run_process_streaming):
            controller.run_process_async(
                cmd=["python3", "-m", "lerobot.teleoperate"],
                cwd=Path("/tmp"),
                complete_callback=None,
                run_mode="teleop",
            )

        self.assertEqual("".join(stdin_recorder.writes), "")

    @patch("robot_pipeline_app.gui_runner.explain_runtime_slowdown", return_value=[])
    @patch(
        "robot_pipeline_app.gui_runner.diagnose_runtime_failure_events",
        return_value=[
            DiagnosticEvent(
                level="FAIL",
                code="SER-PORT_ACCESS",
                name="Serial access",
                detail="check ports and calibration",
                fix="",
                docs_ref="docs/error-catalog.md#ser",
            )
        ],
    )
    @patch("robot_pipeline_app.gui_runner.diagnose_deploy_failure_events", return_value=[])
    @patch("robot_pipeline_app.gui_runner.write_run_artifacts", return_value=None)
    @patch("robot_pipeline_app.gui_runner.TeleopRunPopout", _FakeTeleopRunPopout)
    @patch("robot_pipeline_app.gui_runner.RunControlPopout", _FakeRunControlPopout)
    def test_runtime_diagnostics_are_logged_for_nonzero_noncanceled_runs(self, _write_run_artifacts: object, _deploy_diag: object, _runtime_failure: object, _runtime_diag: object) -> None:
        controller, _colors, _status_var, log_panel, _dot_colors, _messagebox = self._build_controller()

        def fake_run_process_streaming(**kwargs: Any) -> _ThreadStub:
            process = _ProcessStub()
            kwargs["on_process_started"](process)
            kwargs["on_line"]("ConnectionError: [TxRxResult] There is no status packet!")
            kwargs["on_complete"](1)
            return _ThreadStub()

        with patch("robot_pipeline_app.gui_runner.run_process_streaming", side_effect=fake_run_process_streaming):
            controller.run_process_async(
                cmd=["python3", "-m", "lerobot.teleoperate", "--robot.port=/dev/ttyACM0", "--teleop.port=/dev/ttyACM1"],
                cwd=Path("/tmp"),
                complete_callback=None,
                run_mode="teleop",
            )

        self.assertTrue(any("Runtime diagnostics [SER-PORT_ACCESS]: check ports and calibration" in line for line in log_panel.lines))


if __name__ == "__main__":
    unittest.main()
