from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .artifacts import build_run_id, write_run_artifacts
from .deploy_diagnostics import explain_deploy_failure
from .gui_log import GuiLogPanel
from .gui_run_popout import RunControlPopout
from .runner import format_command, is_huggingface_cli_command_missing, run_process_streaming


RunCompleteCallback = Callable[[int, bool], None]


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
        "thread": None,
    }

    def has_active_process() -> bool:
        process = running_state.get("process")
        if process is not None and process.poll() is None:
            return True
        return bool(running_state.get("active"))

    def is_running() -> bool:
        return bool(running_state["active"])

    def _write_process_input(payload: str) -> tuple[bool, str]:
        process = running_state.get("process")
        if process is None or process.poll() is not None:
            return False, "No active record/deploy process to receive input."

        stdin_handle = getattr(process, "stdin", None)
        if stdin_handle is not None:
            try:
                stdin_handle.write(str(payload))
                stdin_handle.flush()
                return True, "Input sent to active process."
            except Exception as exc:
                return False, f"Failed to send stdin ({exc})."

        master_fd = getattr(process, "_rp_master_fd", None)
        if isinstance(master_fd, int):
            try:
                os.write(master_fd, str(payload).encode("utf-8", errors="ignore"))
                return True, "Input sent to active process."
            except Exception as exc:
                return False, f"Failed to send PTY input ({exc})."

        return False, "Active process stdin is unavailable."

    def send_stdin(text: str) -> tuple[bool, str]:
        return _write_process_input(str(text))

    def send_arrow_key(direction: str) -> tuple[bool, str]:
        action_label = "Redo run" if direction == "left" else "Start next run"
        seq = "\x1b[D" if direction == "left" else "\x1b[C"
        ok, message = _write_process_input(seq)
        if not ok:
            return False, f"{action_label}: {message}"

        log_panel.append_log(f"Sent {'left' if direction == 'left' else 'right'} arrow key ({action_label.lower()}).")
        return True, f"{action_label}: key sent."

    run_popout = RunControlPopout(
        root=root,
        colors=colors,
        on_send_key=send_arrow_key,
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
            run_popout.hide()

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
        log_panel.append_log("Cancel requested. Sending terminate signal...")
        try:
            process.terminate()
        except Exception as exc:
            log_panel.append_log(f"Terminate failed: {exc}")

    run_popout.on_cancel = cancel_active_run

    def run_process_async(
        cmd: list[str],
        cwd: Path | None,
        complete_callback: RunCompleteCallback | None = None,
        expected_episodes: int | None = None,
        expected_seconds: int | None = None,
        run_mode: str = "run",
        preflight_checks: list[tuple[str, str, str]] | None = None,
        artifact_context: dict[str, Any] | None = None,
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
        set_running(True, "Running command...")
        command_text = format_command(cmd)
        last_command_state["value"] = command_text
        log_panel.append_log("▶ " + command_text)
        run_id = build_run_id(run_mode)
        run_started = datetime.now(timezone.utc)
        run_output_lines: list[str] = [f"$ {command_text}"]

        if run_mode in {"record", "deploy"}:
            run_popout.start_run(run_mode, expected_episodes, expected_seconds)
        else:
            run_popout.hide()

        def persist_artifacts(exit_code: int | None, canceled: bool) -> None:
            run_ended = datetime.now(timezone.utc)
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
            )
            if artifact_path is not None:
                root.after(0, log_panel.append_log, f"Run artifacts saved: {artifact_path}")
                if on_artifact_written is not None:
                    root.after(0, on_artifact_written)

        def on_line(line: str) -> None:
            run_output_lines.append(line)
            root.after(0, log_panel.append_log, line)
            root.after(0, run_popout.handle_output_line, line)

        def on_start_error(exc: Exception) -> None:
            if isinstance(exc, FileNotFoundError):
                message = f"Command not found: {cmd[0]}"
                root.after(0, log_panel.append_log, message)
                run_output_lines.append(message)
                if is_huggingface_cli_command_missing(cmd, exc):
                    hint = "Make sure you're in your lerobot env: source ~/lerobot/lerobot_env/bin/activate"
                    root.after(0, log_panel.append_log, hint)
                    run_output_lines.append(hint)
            else:
                message = f"Failed to start command ({exc.__class__.__name__}): {exc}"
                root.after(0, log_panel.append_log, message)
                run_output_lines.append(message)

            persist_artifacts(exit_code=-1, canceled=False)

            def notify() -> None:
                set_running(False, "Command failed to start.", True)
                if on_run_failure is not None:
                    on_run_failure()
                messagebox.showerror("Command Error", str(exc))

            root.after(0, notify)

        def on_complete(return_code: int) -> None:
            canceled = bool(running_state.get("cancel_requested") and return_code != 0)
            if canceled:
                root.after(0, log_panel.append_log, "Command canceled by user.")
                run_output_lines.append("Command canceled by user.")

            root.after(0, log_panel.append_log, f"[exit code {return_code}]")
            run_output_lines.append(f"[exit code {return_code}]")

            if return_code != 0:
                is_deploy = bool(run_mode == "deploy" or any(arg.startswith("--policy.path=") for arg in cmd))
                if is_deploy:
                    model_path_raw = context.get("model_path")
                    model_path = Path(str(model_path_raw)) if model_path_raw else None
                    for hint in explain_deploy_failure(run_output_lines, model_path):
                        root.after(0, log_panel.append_log, f"Deploy diagnostics: {hint}")
                        run_output_lines.append(f"Deploy diagnostics: {hint}")

            persist_artifacts(exit_code=return_code, canceled=canceled)

            def complete() -> None:
                if complete_callback is not None:
                    complete_callback(return_code, canceled)
                else:
                    set_running(False, "Ready." if return_code == 0 else "Command failed.", return_code != 0)

                if return_code != 0 and on_run_failure is not None:
                    on_run_failure()

            root.after(0, complete)

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
            use_pty=run_mode in {"record", "deploy"},
        )

    return GuiRunController(
        cancel_active_run=cancel_active_run,
        run_process_async=run_process_async,
        set_running=set_running,
        is_running=is_running,
        send_stdin=send_stdin,
        has_active_process=has_active_process,
    )
