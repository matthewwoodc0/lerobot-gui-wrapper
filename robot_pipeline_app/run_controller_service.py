from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from .artifacts import build_run_id, write_run_artifacts
from .command_overrides import get_policy_path_value
from .deploy_diagnostics import (
    diagnose_deploy_failure_events,
    diagnose_runtime_failure_events,
    explain_runtime_slowdown,
    summarize_camera_command_load,
)
from .run_state import (
    ProcessSessionState,
    command_has_explicit_calibration_dir,
    extract_calibration_prompt_id,
    is_saved_calibration_prompt,
    is_teleop_av1_decode_error,
    is_teleop_ready_line,
)
from .runner import format_command, is_huggingface_cli_command_missing, run_process_streaming


RunCompleteCallback = Callable[[int, bool], None]
SetRunningCallback = Callable[[bool, Optional[str], bool], None]


@dataclass(frozen=True)
class RunUiHooks:
    set_running: SetRunningCallback
    append_output_line: Callable[[str], None] | None = None
    append_output_chunk: Callable[[str], None] | None = None
    on_teleop_ready: Callable[[], None] | None = None
    on_artifact_written: Callable[[Path], None] | None = None


class ManagedRunController:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        schedule_ui: Callable[..., None],
        append_log: Callable[[str], None],
        external_busy: Callable[[], bool] | None = None,
        on_run_failure: Callable[[], None] | None = None,
        on_running_state_change: Callable[[bool], None] | None = None,
    ) -> None:
        self._config = config
        self._schedule_ui = schedule_ui
        self._append_log = append_log
        self._external_busy = external_busy
        self._on_run_failure = on_run_failure
        self._on_running_state_change = on_running_state_change
        self._session = ProcessSessionState()
        self._active_hooks: RunUiHooks | None = None

    def has_active_process(self) -> bool:
        return self._session.has_active_process()

    def is_running(self) -> bool:
        return self._session.is_running()

    def send_stdin(self, text: str) -> tuple[bool, str]:
        return self._session.send_input(str(text))

    def send_arrow_key(self, direction: str) -> tuple[bool, str]:
        ok, message = self._session.send_arrow_key(direction)
        if not ok:
            return False, message
        detail = f"Arrow key dispatched ({str(message).split(':', 1)[0].lower()}). Waiting for process response..."
        self._emit_line(detail)
        return True, message

    def cancel_active_run(self) -> tuple[bool, str]:
        if not self._session.active:
            return False, "No active command is running."
        if self._session.process is None or self._session.process.poll() is not None:
            detail = "Cancel requested, but no active process handle was available."
            self._emit_line(detail)
            return False, detail
        self._session.request_cancel()
        detail = "Cancel requested. Initiating graceful shutdown of the process tree..."
        self._emit_line(detail)
        return True, detail

    def run_process_async(
        self,
        *,
        cmd: list[str],
        cwd: Path | None,
        hooks: RunUiHooks,
        complete_callback: RunCompleteCallback | None = None,
        expected_episodes: int | None = None,
        expected_seconds: int | None = None,
        run_mode: str = "run",
        preflight_checks: list[tuple[str, str, str]] | None = None,
        artifact_context: dict[str, Any] | None = None,
        start_error_callback: Callable[[Exception], None] | None = None,
    ) -> tuple[bool, str | None]:
        if self._session.active:
            return False, "Another command is already running."
        if self._external_busy is not None and self._external_busy():
            return False, "A terminal shell command is currently running. Wait for it to finish first."

        checks = preflight_checks or []
        context = artifact_context or {}
        self._session.cancel_requested = False
        self._session.cancel_outcome = False
        self._active_hooks = hooks
        self._set_running(True, "Running command...", False)

        command_text = format_command(cmd)
        self._emit_line("▶ " + command_text)
        run_id = build_run_id(run_mode)
        run_started = datetime.now(timezone.utc)
        run_output_lines: list[str] = [f"$ {command_text}"]
        teleop_av1_warning = {"shown": False}
        auto_accept_calibration_prompt = (
            run_mode in {"record", "deploy", "teleop"} and command_has_explicit_calibration_dir(cmd)
        )
        handled_calibration_prompt_keys: set[str] = set()
        calibration_prompt_sequence = 0
        calibration_chunk_tail = ""
        runtime_diagnostics: list[dict[str, Any]] = []

        if run_mode == "record":
            camera_load_summary = summarize_camera_command_load(cmd)
            if camera_load_summary:
                self._emit_line(camera_load_summary)
                run_output_lines.append(camera_load_summary)

        def maybe_auto_accept_calibration_prompt(text: str) -> None:
            nonlocal calibration_prompt_sequence
            if not auto_accept_calibration_prompt:
                return
            if not is_saved_calibration_prompt(text):
                return

            prompt_id = extract_calibration_prompt_id(text)
            if prompt_id:
                prompt_key = f"id:{prompt_id}"
            else:
                calibration_prompt_sequence += 1
                prompt_key = f"sequence:{calibration_prompt_sequence}"

            if prompt_key in handled_calibration_prompt_keys:
                return
            handled_calibration_prompt_keys.add(prompt_key)

            ok, detail = self._session.send_input("\n")
            if prompt_id:
                message = (
                    f"Calibration prompt detected for id '{prompt_id}'; "
                    "auto-sent ENTER to use the selected saved calibration file."
                )
            else:
                message = "Calibration prompt detected; auto-sent ENTER to use the selected saved calibration file."
            if not ok:
                message += f" Input dispatch failed: {detail}"
            run_output_lines.append(message)
            self._emit_line(message)

        def persist_artifacts(exit_code: int | None, canceled: bool) -> None:
            run_ended = datetime.now(timezone.utc)
            metadata_extra: dict[str, Any] = {}
            if runtime_diagnostics:
                metadata_extra["runtime_diagnostics"] = list(runtime_diagnostics)
            artifact_path = write_run_artifacts(
                config=self._config,
                mode=run_mode,
                command=cmd,
                cwd=cwd,
                started_at=run_started,
                ended_at=run_ended,
                exit_code=exit_code,
                canceled=canceled,
                preflight_checks=checks,
                output_lines=run_output_lines,
                dataset_repo_id=context.get("dataset_repo_id"),
                model_path=context.get("model_path"),
                run_id=run_id,
                source="pipeline",
                metadata_extra=metadata_extra or None,
            )
            if artifact_path is None:
                return
            self._emit_line(f"Run artifacts saved: {artifact_path}")
            if self._active_hooks is not None and self._active_hooks.on_artifact_written is not None:
                self._dispatch(self._active_hooks.on_artifact_written, artifact_path)

        def on_line(line: str) -> None:
            if run_mode == "teleop" and is_teleop_av1_decode_error(line):
                if not teleop_av1_warning["shown"]:
                    teleop_av1_warning["shown"] = True
                    summary = (
                        "Teleop media decode fallback: AV1 hardware decode is unavailable on this platform; "
                        "using compatibility path and suppressing repeated decoder spam."
                    )
                    run_output_lines.append(summary)
                    self._emit_line(summary)
                return

            run_output_lines.append(line)
            maybe_auto_accept_calibration_prompt(line)
            self._emit_line(line)

            if run_mode == "teleop" and is_teleop_ready_line(line):
                if self._active_hooks is not None and self._active_hooks.on_teleop_ready is not None:
                    self._dispatch(self._active_hooks.on_teleop_ready)

        def on_chunk(chunk: str) -> None:
            nonlocal calibration_chunk_tail
            if auto_accept_calibration_prompt and chunk:
                calibration_chunk_tail = (calibration_chunk_tail + chunk)[-1200:]
                maybe_auto_accept_calibration_prompt(calibration_chunk_tail)
            if not chunk or self._active_hooks is None or self._active_hooks.append_output_chunk is None:
                return
            self._dispatch(self._active_hooks.append_output_chunk, chunk)

        def on_start_error(exc: Exception) -> None:
            if start_error_callback is not None:
                try:
                    start_error_callback(exc)
                except Exception as callback_exc:
                    self._emit_line(f"Start-error callback failed: {callback_exc}")

            if isinstance(exc, FileNotFoundError):
                message = f"Command not found: {cmd[0]}"
                self._emit_line(message)
                run_output_lines.append(message)
                if is_huggingface_cli_command_missing(cmd, exc):
                    hint = "Make sure you're in your lerobot env: source ~/lerobot/lerobot_env/bin/activate"
                    self._emit_line(hint)
                    run_output_lines.append(hint)
            else:
                message = f"Failed to start command ({exc.__class__.__name__}): {exc}"
                self._emit_line(message)
                run_output_lines.append(message)

            persist_artifacts(exit_code=-1, canceled=False)
            self._set_running(False, "Command failed to start.", True)
            if self._on_run_failure is not None:
                self._dispatch(self._on_run_failure)

        def on_complete(return_code: int) -> None:
            canceled = bool(self._session.cancel_requested)
            self._session.set_cancel_outcome(canceled)
            if canceled:
                self._emit_line("Command canceled by user.")
                run_output_lines.append("Command canceled by user.")

            exit_summary = f"[exit code {return_code}]"
            self._emit_line(exit_summary)
            run_output_lines.append(exit_summary)

            if return_code != 0 and not canceled:
                runtime_events = diagnose_runtime_failure_events(run_output_lines, cmd, run_mode)
                for event in runtime_events:
                    runtime_diagnostics.append(event.to_dict())
                    self._emit_line(f"Runtime diagnostics [{event.code}]: {event.detail}")
                    run_output_lines.append(f"Runtime diagnostics [{event.code}]: {event.detail}")
                    if event.fix:
                        self._emit_line(f"Fix: {event.fix}")
                        run_output_lines.append(f"Fix: {event.fix}")

                is_deploy = bool(run_mode == "deploy" or get_policy_path_value(cmd) is not None)
                if is_deploy:
                    model_path_raw = context.get("model_path")
                    model_path = Path(str(model_path_raw)) if model_path_raw else None
                    deploy_events = diagnose_deploy_failure_events(run_output_lines, model_path)
                    for event in deploy_events:
                        runtime_diagnostics.append(event.to_dict())
                        self._emit_line(f"Deploy diagnostics [{event.code}]: {event.detail}")
                        run_output_lines.append(f"Deploy diagnostics [{event.code}]: {event.detail}")
                        if event.fix:
                            self._emit_line(f"Fix: {event.fix}")
                            run_output_lines.append(f"Fix: {event.fix}")

            if run_mode in {"record", "deploy"} and not canceled:
                for hint in explain_runtime_slowdown(run_output_lines, cmd):
                    detail = f"Performance diagnostics: {hint}"
                    self._emit_line(detail)
                    run_output_lines.append(detail)

            persist_artifacts(exit_code=return_code, canceled=canceled)
            if complete_callback is not None:
                self._release_run()
                if self._on_running_state_change is not None:
                    self._dispatch(self._on_running_state_change, False)
                self._dispatch(complete_callback, return_code, canceled)
            else:
                if canceled:
                    self._set_running(False, "Command canceled.", False)
                else:
                    self._set_running(False, "Ready." if return_code == 0 else "Command failed.", return_code != 0)

            if return_code != 0 and not canceled and self._on_run_failure is not None:
                self._dispatch(self._on_run_failure)

        def on_process_started(process: subprocess.Popen[str]) -> None:
            self._session.attach_process(process)

        self._session.attach_thread(
            run_process_streaming(
                cmd=cmd,
                cwd=cwd,
                on_line=on_line,
                on_chunk=on_chunk if self._active_hooks.append_output_chunk is not None else None,
                on_complete=on_complete,
                on_start_error=on_start_error,
                cancel_requested=lambda: bool(self._session.cancel_requested),
                on_process_started=on_process_started,
                use_pty=run_mode in {"record", "deploy", "teleop", "train_attach"},
                suppress_carriage_updates=run_mode in {"train_attach"},
            )
        )
        return True, None

    def _dispatch(self, callback: Callable[..., None], *args: Any) -> None:
        try:
            self._schedule_ui(callback, *args)
        except Exception:
            pass

    def _emit_line(self, line: str) -> None:
        self._dispatch(self._append_log, str(line))
        if self._active_hooks is not None and self._active_hooks.append_output_line is not None:
            self._dispatch(self._active_hooks.append_output_line, str(line))

    def _set_running(self, active: bool, status_text: str | None, is_error: bool) -> None:
        active_hooks = self._active_hooks
        if active:
            self._session.mark_active()
        else:
            self._release_run()
        if active_hooks is not None:
            self._dispatch(active_hooks.set_running, active, status_text, is_error)
        if self._on_running_state_change is not None:
            self._dispatch(self._on_running_state_change, active)

    def _release_run(self) -> None:
        self._session.mark_idle()
        self._active_hooks = None
