from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .artifacts import build_run_id, write_run_artifacts
from .deploy_diagnostics import explain_deploy_failure, explain_runtime_slowdown, summarize_camera_command_load
from .gui_log import GuiLogPanel
from .gui_run_popout import RunControlPopout, TeleopRunPopout
from .runner import format_command, is_huggingface_cli_command_missing, run_process_streaming


RunCompleteCallback = Callable[[int, bool], None]
_TELEOP_AV1_ERROR_MARKERS = (
    "failed to get pixel format",
    "missing sequence header",
    "hardware accelerated av1 decoding",
)


def _is_teleop_av1_decode_error(line: str) -> bool:
    text = str(line or "").strip().lower()
    if "av1" not in text:
        return False
    if any(marker in text for marker in _TELEOP_AV1_ERROR_MARKERS):
        return True
    return "decode" in text and "hardware" in text


_TELEOP_READY_MARKERS = (
    "teleoperate",
    "teleop started",
    "start teleoperation",
    "teleop running",
    "connected",
    "ready",
    "torque enabled",
    "fps:",
)


def _is_teleop_ready_line(line: str) -> bool:
    """Return True when a log line indicates the teleop process is fully running."""
    text = str(line or "").strip().lower()
    return any(marker in text for marker in _TELEOP_READY_MARKERS)


@dataclass
class GuiRunController:
    cancel_active_run: Callable[[], None]
    run_process_async: Callable[
        [
            list[str],
            Path | None,
            RunCompleteCallback | None,
            int | None,
            int | None,
            str,
            list[tuple[str, str, str]] | None,
            dict[str, Any] | None,
            Callable[[Exception], None] | None,
        ],
        None,
    ]
    set_running: Callable[[bool, str | None, bool], None]
    is_running: Callable[[], bool]
    send_stdin: Callable[[str], tuple[bool, str]]
    has_active_process: Callable[[], bool]


def create_run_controller(
    *,
    root: Any,
    config: dict[str, Any],
    colors: dict[str, str],
    status_var: Any,
    set_status_dot: Callable[[str], None],
    log_panel: GuiLogPanel,
    messagebox: Any,
    action_buttons: list[Any],
    last_command_state: dict[str, str],
    external_busy: Callable[[], bool] | None = None,
    on_run_failure: Callable[[], None] | None = None,
    on_artifact_written: Callable[[], None] | None = None,
) -> GuiRunController:
    running_state: dict[str, Any] = {
        "active": False,
        "process": None,
        "cancel_requested": False,
        "cancel_outcome": False,
        "thread": None,
    }

    def has_active_process() -> bool:
        process = running_state.get("process")
        if process is not None and process.poll() is None:
            return True
        return bool(running_state.get("active"))

    def is_running() -> bool:
        return bool(running_state["active"])

    def _write_process_input_bytes(payload: bytes) -> tuple[bool, str]:
        process = running_state.get("process")
        if process is None or process.poll() is not None:
            return False, "No active record/deploy process to receive input."

        pty_error: str | None = None
        master_fd = getattr(process, "_rp_master_fd", None)
        if isinstance(master_fd, int):
            try:
                os.write(master_fd, payload)
                return True, "Input sent to active process."
            except Exception as exc:
                # Fall through to stdin handle as backup.
                pty_error = f"Failed to send PTY input ({exc})."

        stdin_handle = getattr(process, "stdin", None)
        if stdin_handle is not None:
            try:
                try:
                    stdin_handle.write(payload.decode("utf-8", errors="ignore"))
                except TypeError:
                    stdin_handle.write(payload)
                stdin_handle.flush()
                return True, "Input sent to active process."
            except Exception as exc:
                if pty_error:
                    return False, f"{pty_error} Failed to send stdin ({exc})."
                return False, f"Failed to send stdin ({exc})."

        return False, "Active process stdin is unavailable."

    def _write_process_input(payload: str) -> tuple[bool, str]:
        return _write_process_input_bytes(str(payload).encode("utf-8", errors="ignore"))

    def send_stdin(text: str) -> tuple[bool, str]:
        return _write_process_input(str(text))

    def send_arrow_key(direction: str) -> tuple[bool, str]:
        action_label = "Reset episode" if direction == "left" else "Start next episode"
        seq = b"\x1b[D" if direction == "left" else b"\x1b[C"
        ok, message = _write_process_input_bytes(seq)
        if not ok:
            return False, f"{action_label}: {message}"

        log_panel.append_log(f"Arrow key dispatched ({action_label.lower()}). Waiting for process response...")
        return True, f"{action_label}: key sent."

    run_popout = RunControlPopout(
        root=root,
        colors=colors,
        on_send_key=send_arrow_key,
        on_cancel=lambda: None,
    )

    teleop_popout = TeleopRunPopout(
        root=root,
        colors=colors,
        on_cancel=lambda: None,
    )

    def set_running(active: bool, status_text: str | None = None, is_error: bool = False) -> None:
        running_state["active"] = active
        if active:
            status_var.set(status_text or "Running command...")
            set_status_dot(colors["running"])
        else:
            if is_error:
                status_var.set(status_text or "Last command failed.")
                set_status_dot(colors["error"])
            else:
                status_var.set(status_text or "Ready.")
                set_status_dot(colors["ready"])
            running_state["process"] = None
            running_state["thread"] = None
            running_state["cancel_requested"] = False
            running_state["cancel_outcome"] = False
            run_popout.hide()
            teleop_popout.hide()

        log_panel.set_running_state(active)
        for button in action_buttons:
            button.configure(state="disabled" if active else "normal")

    def cancel_active_run() -> None:
        if not running_state.get("active"):
            messagebox.showinfo("Cancel", "No active command is running.")
            return
        process = running_state.get("process")
        if process is None or process.poll() is not None:
            log_panel.append_log("Cancel requested, but no active process handle was available.")
            return
        running_state["cancel_requested"] = True
        log_panel.append_log("Cancel requested. Initiating graceful shutdown of the process tree...")

    run_popout.on_cancel = cancel_active_run
    teleop_popout.on_cancel = cancel_active_run

    def run_process_async(
        cmd: list[str],
        cwd: Path | None,
        complete_callback: RunCompleteCallback | None = None,
        expected_episodes: int | None = None,
        expected_seconds: int | None = None,
        run_mode: str = "run",
        preflight_checks: list[tuple[str, str, str]] | None = None,
        artifact_context: dict[str, Any] | None = None,
        start_error_callback: Callable[[Exception], None] | None = None,
    ) -> None:
        if running_state["active"]:
            messagebox.showinfo("Busy", "Another command is already running.")
            return
        if external_busy is not None and external_busy():
            messagebox.showinfo("Busy", "A terminal shell command is currently running. Wait for it to finish first.")
            return

        checks = preflight_checks or []
        context = artifact_context or {}
        running_state["cancel_requested"] = False
        running_state["cancel_outcome"] = False
        set_running(True, "Running command...")
        command_text = format_command(cmd)
        last_command_state["value"] = command_text
        log_panel.append_log("▶ " + command_text)
        run_id = build_run_id(run_mode)
        run_started = datetime.now(timezone.utc)
        run_output_lines: list[str] = [f"$ {command_text}"]
        teleop_av1_warning: dict[str, bool] = {"shown": False}
        if run_mode == "record":
            camera_load_summary = summarize_camera_command_load(cmd)
            if camera_load_summary:
                log_panel.append_log(camera_load_summary)
                run_output_lines.append(camera_load_summary)

        if run_mode in {"record", "deploy"}:
            run_popout.start_run(run_mode, expected_episodes, expected_seconds)
            teleop_popout.hide()
        elif run_mode == "teleop":
            run_popout.hide()
            teleop_popout.start_run(
                follower_port=str(context.get("follower_port", "") or ""),
                follower_id=str(context.get("follower_id", "") or ""),
                leader_port=str(context.get("leader_port", "") or ""),
                leader_id=str(context.get("leader_id", "") or ""),
            )
        else:
            run_popout.hide()
            teleop_popout.hide()

        def persist_artifacts(exit_code: int | None, canceled: bool) -> None:
            run_ended = datetime.now(timezone.utc)
            metadata_extra: dict[str, Any] = {}
            outcome_summary = run_popout.get_episode_outcome_summary()
            if run_mode == "deploy" and outcome_summary is not None:
                metadata_extra["deploy_episode_outcomes"] = outcome_summary
            artifact_path = write_run_artifacts(
                config=config,
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
            if artifact_path is not None:
                try:
                    root.after(0, log_panel.append_log, f"Run artifacts saved: {artifact_path}")
                except Exception:
                    pass
                if on_artifact_written is not None:
                    try:
                        root.after(0, on_artifact_written)
                    except Exception:
                        pass

        def on_line(line: str) -> None:
            if run_mode == "teleop" and _is_teleop_av1_decode_error(line):
                if not teleop_av1_warning["shown"]:
                    teleop_av1_warning["shown"] = True
                    summary = (
                        "Teleop media decode fallback: AV1 hardware decode is unavailable on this platform; "
                        "using compatibility path and suppressing repeated decoder spam."
                    )
                    run_output_lines.append(summary)
                    try:
                        root.after(0, log_panel.append_log, summary)
                    except Exception:
                        pass
                return
            run_output_lines.append(line)
            try:
                root.after(0, log_panel.append_log, line)
            except Exception:
                pass
            try:
                root.after(0, run_popout.handle_output_line, line)
            except Exception:
                pass
            # Mark teleop startup complete when the process signals it is ready
            if run_mode == "teleop" and _is_teleop_ready_line(line):
                try:
                    root.after(0, teleop_popout.mark_startup_complete)
                except Exception:
                    pass

        def on_start_error(exc: Exception) -> None:
            if start_error_callback is not None:
                try:
                    start_error_callback(exc)
                except Exception as callback_exc:
                    try:
                        root.after(0, log_panel.append_log, f"Start-error callback failed: {callback_exc}")
                    except Exception:
                        pass

            if isinstance(exc, FileNotFoundError):
                message = f"Command not found: {cmd[0]}"
                try:
                    root.after(0, log_panel.append_log, message)
                except Exception:
                    pass
                run_output_lines.append(message)
                if is_huggingface_cli_command_missing(cmd, exc):
                    hint = "Make sure you're in your lerobot env: source ~/lerobot/lerobot_env/bin/activate"
                    try:
                        root.after(0, log_panel.append_log, hint)
                    except Exception:
                        pass
                    run_output_lines.append(hint)
            else:
                message = f"Failed to start command ({exc.__class__.__name__}): {exc}"
                try:
                    root.after(0, log_panel.append_log, message)
                except Exception:
                    pass
                run_output_lines.append(message)

            persist_artifacts(exit_code=-1, canceled=False)

            def notify() -> None:
                set_running(False, "Command failed to start.", True)
                if on_run_failure is not None:
                    on_run_failure()
                messagebox.showerror("Command Error", str(exc))

            try:
                root.after(0, notify)
            except Exception:
                pass

        def on_complete(return_code: int) -> None:
            canceled = bool(running_state.get("cancel_requested"))
            running_state["cancel_outcome"] = canceled
            if canceled:
                try:
                    root.after(0, log_panel.append_log, "Command canceled by user.")
                except Exception:
                    pass
                run_output_lines.append("Command canceled by user.")

            try:
                root.after(0, log_panel.append_log, f"[exit code {return_code}]")
            except Exception:
                pass
            run_output_lines.append(f"[exit code {return_code}]")

            if return_code != 0:
                is_deploy = bool(run_mode == "deploy" or any(arg.startswith("--policy.path=") for arg in cmd))
                if is_deploy:
                    model_path_raw = context.get("model_path")
                    model_path = Path(str(model_path_raw)) if model_path_raw else None
                    for hint in explain_deploy_failure(run_output_lines, model_path):
                        try:
                            root.after(0, log_panel.append_log, f"Deploy diagnostics: {hint}")
                        except Exception:
                            pass
                        run_output_lines.append(f"Deploy diagnostics: {hint}")

            if run_mode in {"record", "deploy"}:
                for hint in explain_runtime_slowdown(run_output_lines, cmd):
                    try:
                        root.after(0, log_panel.append_log, f"Performance diagnostics: {hint}")
                    except Exception:
                        pass
                    run_output_lines.append(f"Performance diagnostics: {hint}")

            persist_artifacts(exit_code=return_code, canceled=canceled)

            def complete() -> None:
                if complete_callback is not None:
                    complete_callback(return_code, canceled)
                else:
                    if canceled:
                        set_running(False, "Command canceled.", False)
                    else:
                        set_running(False, "Ready." if return_code == 0 else "Command failed.", return_code != 0)

                if return_code != 0 and not canceled and on_run_failure is not None:
                    on_run_failure()

            try:
                root.after(0, complete)
            except Exception:
                pass

        def on_process_started(process: subprocess.Popen[str]) -> None:
            running_state["process"] = process

        running_state["thread"] = run_process_streaming(
            cmd=cmd,
            cwd=cwd,
            on_line=on_line,
            on_complete=on_complete,
            on_start_error=on_start_error,
            cancel_requested=lambda: bool(running_state.get("cancel_requested")),
            on_process_started=on_process_started,
            use_pty=run_mode in {"record", "deploy", "train_attach"},
        )

    return GuiRunController(
        cancel_active_run=cancel_active_run,
        run_process_async=run_process_async,
        set_running=set_running,
        is_running=is_running,
        send_stdin=send_stdin,
        has_active_process=has_active_process,
    )
