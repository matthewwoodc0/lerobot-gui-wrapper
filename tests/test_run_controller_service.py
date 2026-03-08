from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from robot_pipeline_app.run_controller_service import ManagedRunController, RunUiHooks
from robot_pipeline_app.types import DiagnosticEvent


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


class ManagedRunControllerTests(unittest.TestCase):
    def _build_controller(self, *, on_run_failure: Any = None) -> tuple[ManagedRunController, list[str], list[bool]]:
        logs: list[str] = []
        running_states: list[bool] = []
        controller = ManagedRunController(
            config={},
            schedule_ui=lambda callback, *args: callback(*args),
            append_log=lambda line: logs.append(str(line)),
            on_run_failure=on_run_failure,
            on_running_state_change=lambda active: running_states.append(bool(active)),
        )
        return controller, logs, running_states

    def _build_hooks(self) -> tuple[RunUiHooks, list[str], list[tuple[bool, str | None, bool]]]:
        output_lines: list[str] = []
        state_updates: list[tuple[bool, str | None, bool]] = []
        hooks = RunUiHooks(
            set_running=lambda active, status, is_error: state_updates.append((bool(active), status, bool(is_error))),
            append_output_line=lambda line: output_lines.append(str(line)),
        )
        return hooks, output_lines, state_updates

    @patch("robot_pipeline_app.run_controller_service.explain_runtime_slowdown", return_value=[])
    @patch("robot_pipeline_app.run_controller_service.diagnose_deploy_failure_events", return_value=[])
    @patch("robot_pipeline_app.run_controller_service.write_run_artifacts", return_value=None)
    def test_cancel_zero_exit_reports_canceled_and_releases_controller(self, write_run_artifacts: object, _deploy_diag: object, _runtime_diag: object) -> None:
        controller, logs, running_states = self._build_controller()
        hooks, output_lines, state_updates = self._build_hooks()
        callbacks: dict[str, Any] = {}

        def fake_run_process_streaming(**kwargs: Any) -> _ThreadStub:
            process = _ProcessStub()
            kwargs["on_process_started"](process)
            callbacks["on_complete"] = kwargs["on_complete"]
            return _ThreadStub()

        with patch("robot_pipeline_app.run_controller_service.run_process_streaming", side_effect=fake_run_process_streaming):
            completion: list[tuple[int, bool]] = []
            ok, message = controller.run_process_async(
                cmd=["python3", "-c", "print('ok')"],
                cwd=Path("/tmp"),
                hooks=hooks,
                complete_callback=lambda return_code, canceled: completion.append((return_code, canceled)),
            )
            self.assertTrue(ok)
            self.assertIsNone(message)

            cancel_ok, _cancel_message = controller.cancel_active_run()
            self.assertTrue(cancel_ok)
            callbacks["on_complete"](0)

            second_ok, second_message = controller.run_process_async(
                cmd=["python3", "-c", "print('again')"],
                cwd=Path("/tmp"),
                hooks=hooks,
            )

        self.assertEqual(completion, [(0, True)])
        self.assertTrue(second_ok)
        self.assertIsNone(second_message)
        self.assertTrue(any("Command canceled by user." in line for line in logs))
        self.assertTrue(any("Command canceled by user." in line for line in output_lines))
        self.assertEqual(running_states[:2], [True, False])
        self.assertTrue(state_updates)
        kwargs = write_run_artifacts.call_args.kwargs  # type: ignore[attr-defined]
        self.assertEqual(kwargs["exit_code"], 0)
        self.assertTrue(kwargs["canceled"])

    @patch("robot_pipeline_app.run_controller_service.explain_runtime_slowdown", return_value=[])
    @patch("robot_pipeline_app.run_controller_service.diagnose_deploy_failure_events", return_value=[])
    @patch("robot_pipeline_app.run_controller_service.write_run_artifacts", return_value=None)
    def test_teleop_av1_decode_spam_is_collapsed_to_single_notice(self, _write_run_artifacts: object, _deploy_diag: object, _runtime_diag: object) -> None:
        controller, logs, _running_states = self._build_controller()
        hooks, output_lines, _state_updates = self._build_hooks()

        def fake_run_process_streaming(**kwargs: Any) -> _ThreadStub:
            process = _ProcessStub()
            kwargs["on_process_started"](process)
            kwargs["on_line"]("Failed to get pixel format for AV1 stream")
            kwargs["on_line"]("Missing sequence header while decoding AV1")
            kwargs["on_line"]("teleop still running")
            kwargs["on_complete"](0)
            return _ThreadStub()

        with patch("robot_pipeline_app.run_controller_service.run_process_streaming", side_effect=fake_run_process_streaming):
            controller.run_process_async(
                cmd=["python3", "-m", "lerobot.teleoperate"],
                cwd=Path("/tmp"),
                hooks=hooks,
                run_mode="teleop",
            )

        warning_lines = [line for line in logs if "Teleop media decode fallback" in line]
        self.assertEqual(len(warning_lines), 1)
        self.assertFalse(any("failed to get pixel format" in line.lower() for line in logs))
        self.assertFalse(any("missing sequence header" in line.lower() for line in output_lines))
        self.assertTrue(any("teleop still running" in line for line in output_lines))

    @patch("robot_pipeline_app.run_controller_service.explain_runtime_slowdown", return_value=[])
    @patch("robot_pipeline_app.run_controller_service.diagnose_deploy_failure_events", return_value=[])
    @patch("robot_pipeline_app.run_controller_service.write_run_artifacts", return_value=None)
    def test_teleop_auto_accepts_saved_calibration_prompt_once_per_id(self, _write_run_artifacts: object, _deploy_diag: object, _runtime_diag: object) -> None:
        controller, logs, _running_states = self._build_controller()
        hooks, _output_lines, _state_updates = self._build_hooks()
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

        with patch("robot_pipeline_app.run_controller_service.run_process_streaming", side_effect=fake_run_process_streaming):
            controller.run_process_async(
                cmd=[
                    "python3",
                    "-m",
                    "lerobot.teleoperate",
                    "--robot.calibration_dir=/tmp/calibration",
                    "--teleop.calibration_dir=/tmp/calibration",
                ],
                cwd=Path("/tmp"),
                hooks=hooks,
                run_mode="teleop",
            )

        self.assertEqual("".join(stdin_recorder.writes).count("\n"), 1)
        self.assertTrue(any("auto-sent ENTER to use the selected saved calibration file" in line for line in logs))

    @patch("robot_pipeline_app.run_controller_service.explain_runtime_slowdown", return_value=[])
    @patch(
        "robot_pipeline_app.run_controller_service.diagnose_runtime_failure_events",
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
    @patch("robot_pipeline_app.run_controller_service.diagnose_deploy_failure_events", return_value=[])
    @patch("robot_pipeline_app.run_controller_service.write_run_artifacts", return_value=None)
    def test_runtime_diagnostics_are_logged_for_nonzero_runs(self, _write_run_artifacts: object, _deploy_diag: object, _runtime_failure: object, _runtime_diag: object) -> None:
        controller, logs, _running_states = self._build_controller()
        hooks, output_lines, _state_updates = self._build_hooks()

        def fake_run_process_streaming(**kwargs: Any) -> _ThreadStub:
            process = _ProcessStub()
            kwargs["on_process_started"](process)
            kwargs["on_line"]("ConnectionError: [TxRxResult] There is no status packet!")
            kwargs["on_complete"](1)
            return _ThreadStub()

        with patch("robot_pipeline_app.run_controller_service.run_process_streaming", side_effect=fake_run_process_streaming):
            controller.run_process_async(
                cmd=["python3", "-m", "lerobot.teleoperate", "--robot.port=/dev/ttyACM0", "--teleop.port=/dev/ttyACM1"],
                cwd=Path("/tmp"),
                hooks=hooks,
                run_mode="teleop",
            )

        self.assertTrue(any("Runtime diagnostics [SER-PORT_ACCESS]: check ports and calibration" in line for line in logs))
        self.assertTrue(any("Runtime diagnostics [SER-PORT_ACCESS]: check ports and calibration" in line for line in output_lines))


if __name__ == "__main__":
    unittest.main()
