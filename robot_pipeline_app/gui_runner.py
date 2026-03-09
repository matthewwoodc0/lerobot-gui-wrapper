from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .artifacts import build_run_id, write_run_artifacts
from .command_overrides import get_policy_path_value
from .deploy_diagnostics import (
    diagnose_deploy_failure_events,
    diagnose_runtime_failure_events,
    explain_runtime_slowdown,
    summarize_camera_command_load,
)
from .gui_log import GuiLogPanel
from .gui_run_popout import RunControlPopout, TeleopRunPopout
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
_is_teleop_av1_decode_error = is_teleop_av1_decode_error
_is_teleop_ready_line = is_teleop_ready_line
_is_saved_calibration_prompt = is_saved_calibration_prompt
_extract_calibration_prompt_id = extract_calibration_prompt_id
_command_has_explicit_calibration_dir = command_has_explicit_calibration_dir


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
    configure_record_camera_feed: Callable[[dict[str, Any], bool, str, Callable[[str], None], Any | None], None]
    apply_theme: Callable[[dict[str, str]], None]


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
    on_running_state_change: Callable[[bool], None] | None = None,
) -> GuiRunController:
    session = ProcessSessionState()

    def has_active_process() -> bool:
        return session.has_active_process()

    def is_running() -> bool:
        return session.is_running()

    def send_stdin(text: str) -> tuple[bool, str]:
        return session.send_input(str(text))

    def send_arrow_key(direction: str) -> tuple[bool, str]:
        ok, message = session.send_arrow_key(direction)
        if not ok:
            return False, message
        log_panel.append_log(f"Arrow key dispatched ({str(message).split(':', 1)[0].lower()}). Waiting for process response...")
        return True, message

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
        if active:
            session.mark_active()
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
            session.mark_idle()
            run_popout.hide()
            teleop_popout.hide()

        log_panel.set_running_state(active)
        if on_running_state_change is not None:
            try:
                on_running_state_change(active)
            except Exception:
                pass
        for button in action_buttons:
            button.configure(state="disabled" if active else "normal")

    def cancel_active_run() -> None:
        if not session.active:
            messagebox.showinfo("Cancel", "No active command is running.")
            return
        if session.process is None or session.process.poll() is not None:
            log_panel.append_log("Cancel requested, but no active process handle was available.")
            return
        session.request_cancel()
        log_panel.append_log("Cancel requested. Initiating graceful shutdown of the process tree...")

    run_popout.on_cancel = cancel_active_run
    teleop_popout.on_cancel = cancel_active_run

    def configure_record_camera_feed(
        config_payload: dict[str, Any],
        cv2_probe_ok: bool,
        cv2_probe_error: str,
        append_log: Callable[[str], None],
        background_jobs: Any | None = None,
    ) -> None:
        run_popout.configure_record_camera_feed(
            config=config_payload,
            cv2_probe_ok=cv2_probe_ok,
            cv2_probe_error=cv2_probe_error,
            append_log=append_log,
            background_jobs=background_jobs,
        )

    def apply_theme(updated_colors: dict[str, str]) -> None:
        run_popout.apply_theme(updated_colors)
        teleop_popout.apply_theme(updated_colors)

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
        if session.active:
            messagebox.showinfo("Busy", "Another command is already running.")
            return
        if external_busy is not None and external_busy():
            messagebox.showinfo("Busy", "A terminal shell command is currently running. Wait for it to finish first.")
            return

        checks = preflight_checks or []
        context = artifact_context or {}
        session.cancel_requested = False
        session.cancel_outcome = False
        set_running(True, "Running command...")
        command_text = format_command(cmd)
        last_command_state["value"] = command_text
        log_panel.append_log("▶ " + command_text)
        run_id = build_run_id(run_mode)
        run_started = datetime.now(timezone.utc)
        run_output_lines: list[str] = [f"$ {command_text}"]
        teleop_av1_warning: dict[str, bool] = {"shown": False}
        auto_accept_calibration_prompt = (
            run_mode in {"record", "deploy", "teleop"} and _command_has_explicit_calibration_dir(cmd)
        )
        handled_calibration_prompt_keys: set[str] = set()
        calibration_prompt_sequence = 0
        calibration_chunk_tail = ""
        runtime_diagnostics: list[dict[str, Any]] = []
        if run_mode == "record":
            camera_load_summary = summarize_camera_command_load(cmd)
            if camera_load_summary:
                log_panel.append_log(camera_load_summary)
                run_output_lines.append(camera_load_summary)
        feed_terminal = getattr(log_panel, "feed_terminal_output", None)
        stream_terminal_output = callable(feed_terminal)

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

            ok, detail = session.send_input("\n")
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
            try:
                root.after(0, log_panel.append_log, message)
            except Exception:
                pass

        def persist_artifacts(exit_code: int | None, canceled: bool) -> None:
            run_ended = datetime.now(timezone.utc)
            metadata_extra: dict[str, Any] = {}
            outcome_summary = run_popout.get_episode_outcome_summary()
            if run_mode == "deploy" and outcome_summary is not None:
                metadata_extra["deploy_episode_outcomes"] = outcome_summary
            if runtime_diagnostics:
                metadata_extra["runtime_diagnostics"] = list(runtime_diagnostics)
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
            maybe_auto_accept_calibration_prompt(line)
            if not stream_terminal_output:
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

        def on_chunk(chunk: str) -> None:
            nonlocal calibration_chunk_tail
            if auto_accept_calibration_prompt and chunk:
                calibration_chunk_tail = (calibration_chunk_tail + chunk)[-1200:]
                maybe_auto_accept_calibration_prompt(calibration_chunk_tail)
            if not stream_terminal_output or not chunk:
                return
            try:
                root.after(0, feed_terminal, chunk)
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
            canceled = bool(session.cancel_requested)
            session.set_cancel_outcome(canceled)
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

            if return_code != 0 and not canceled:
                runtime_events = diagnose_runtime_failure_events(run_output_lines, cmd, run_mode)
                for event in runtime_events:
                    runtime_diagnostics.append(event.to_dict())
                    try:
                        root.after(0, log_panel.append_log, f"Runtime diagnostics [{event.code}]: {event.detail}")
                    except Exception:
                        pass
                    run_output_lines.append(f"Runtime diagnostics [{event.code}]: {event.detail}")
                    if event.fix:
                        try:
                            root.after(0, log_panel.append_log, f"Fix: {event.fix}")
                        except Exception:
                            pass
                        run_output_lines.append(f"Fix: {event.fix}")

                is_deploy = bool(run_mode == "deploy" or get_policy_path_value(cmd) is not None)
                if is_deploy:
                    model_path_raw = context.get("model_path")
                    model_path = Path(str(model_path_raw)) if model_path_raw else None
                    deploy_events = diagnose_deploy_failure_events(run_output_lines, model_path)
                    for event in deploy_events:
                        runtime_diagnostics.append(event.to_dict())
                        try:
                            root.after(0, log_panel.append_log, f"Deploy diagnostics [{event.code}]: {event.detail}")
                        except Exception:
                            pass
                        run_output_lines.append(f"Deploy diagnostics [{event.code}]: {event.detail}")
                        if event.fix:
                            try:
                                root.after(0, log_panel.append_log, f"Fix: {event.fix}")
                            except Exception:
                                pass
                            run_output_lines.append(f"Fix: {event.fix}")

            if run_mode in {"record", "deploy"} and not canceled:
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
            session.attach_process(process)

        session.attach_thread(run_process_streaming(
            cmd=cmd,
            cwd=cwd,
            on_line=on_line,
            on_chunk=on_chunk if stream_terminal_output else None,
            on_complete=on_complete,
            on_start_error=on_start_error,
            cancel_requested=lambda: bool(session.cancel_requested),
            on_process_started=on_process_started,
            use_pty=run_mode in {"record", "deploy", "teleop"},
            suppress_carriage_updates=False,
        ))

    return GuiRunController(
        cancel_active_run=cancel_active_run,
        run_process_async=run_process_async,
        set_running=set_running,
        is_running=is_running,
        send_stdin=send_stdin,
        has_active_process=has_active_process,
        configure_record_camera_feed=configure_record_camera_feed,
        apply_theme=apply_theme,
    )
