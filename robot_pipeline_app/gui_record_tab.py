from __future__ import annotations

import shlex
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .command_overrides import get_flag_value
from .checks import run_preflight_for_record
from .config_store import get_lerobot_dir, normalize_path, save_config
from .constants import DEFAULT_TASK
from .gui_camera import DualCameraPreview
from .gui_dialogs import ask_text_dialog, format_command_for_dialog, show_text_dialog
from .gui_file_dialogs import ask_directory_dialog
from .gui_forms import build_record_request_and_command
from .gui_log import GuiLogPanel
from .repo_utils import dataset_exists_on_hf, repo_name_from_repo_id, resolve_unique_repo_id, suggest_dataset_name
from .runner import format_command
from .training_profiles import load_training_profiles
from .training_remote import ensure_host_trusted
from .types import GuiRunProcessAsync, TrainingProfile
from .workflows import move_recorded_dataset


@dataclass
class RecordTabHandles:
    record_dir_var: Any
    record_camera_preview: DualCameraPreview
    refresh_summary: Callable[[], None]
    action_buttons: list[Any]


def _default_olympus_profile() -> TrainingProfile:
    return TrainingProfile(
        id="olympus",
        name="Olympus",
        host="olympus.ece.tamu.edu",
        port=22,
        username="",
        auth_mode="password",
        identity_file="",
        remote_models_root="~/lerobot/trained_models",
        remote_project_root="~/lerobot",
        env_activate_cmd="source ~/lerobot/lerobot_env/bin/activate",
        default_tmux_session="",
        default_srun_prefix="",
    )


def _pick_olympus_profile(config: dict[str, Any]) -> TrainingProfile:
    profiles, _ = load_training_profiles(config)
    for profile in profiles:
        if profile.id == "olympus":
            return profile
    for profile in profiles:
        if str(profile.host).strip().lower() == "olympus.ece.tamu.edu":
            return profile
    return profiles[0] if profiles else _default_olympus_profile()


def _build_ssh_profile(host: str, port: int, username: str) -> TrainingProfile:
    return TrainingProfile(
        id="record_sync",
        name="Record Sync",
        host=host,
        port=port,
        username=username,
        auth_mode="password",
        identity_file="",
        remote_models_root="",
        remote_project_root="",
        env_activate_cmd="",
        default_tmux_session="",
        default_srun_prefix="",
    )


def _list_local_dataset_dirs(record_data_dir: Path, lerobot_dir: Path) -> list[Path]:
    roots = [
        Path(normalize_path(str(record_data_dir))),
        Path(normalize_path(str(lerobot_dir / "data"))),
    ]
    seen: set[str] = set()
    datasets: list[Path] = []
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        try:
            children = sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))
        except OSError:
            continue
        for child in children:
            key = str(child.resolve())
            if key in seen:
                continue
            seen.add(key)
            datasets.append(child)
    return sorted(datasets, key=lambda path: path.name.lower())


def _compose_repo_id(owner: str, dataset_name: str) -> str | None:
    clean_owner = str(owner).strip().strip("/")
    clean_name = str(dataset_name).strip().strip("/")
    if not clean_owner or not clean_name:
        return None
    return f"{clean_owner}/{clean_name}"


def _hf_parity_detail(exists: bool | None, repo_id: str) -> tuple[str, str]:
    if exists is True:
        return "WARN", f"Remote dataset already exists: {repo_id}"
    if exists is False:
        return "PASS", f"Remote dataset not found yet: {repo_id}"
    return "WARN", f"Unable to confirm if remote dataset exists: {repo_id}"


def _build_v30_convert_command(repo_id: str, python_bin: str | None = None) -> list[str]:
    py = python_bin or sys.executable or shutil.which("python3") or shutil.which("python") or "python3"
    return [
        py,
        "-m",
        "lerobot.datasets.v30.convert_dataset_v21_to_v30",
        f"--repo-id={repo_id}",
    ]


def _build_rsync_push_command(profile: TrainingProfile, local_dataset: Path, remote_parent: str) -> list[str]:
    known_hosts = str((Path.home() / ".ssh" / "known_hosts").resolve())
    ssh_parts = [
        "ssh",
        "-p",
        str(profile.port),
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
    ]
    remote_target = f"{profile.username}@{profile.host}:{str(remote_parent).rstrip('/')}/"
    return [
        "rsync",
        "-az",
        "--partial",
        "--info=progress2",
        "-e",
        shlex.join(ssh_parts),
        str(local_dataset),
        remote_target,
    ]


def _build_sftp_push_command(
    profile: TrainingProfile,
    local_dataset: Path,
    remote_parent: str,
) -> tuple[list[str], Path]:
    batch_file = tempfile.NamedTemporaryFile("w", encoding="utf-8", prefix="lerobot_sftp_push_", suffix=".txt", delete=False)
    batch_path = Path(batch_file.name)
    try:
        batch_file.write(f"lcd {str(local_dataset.parent)}\n")
        batch_file.write(f"cd {str(remote_parent)}\n")
        batch_file.write(f"put -r {local_dataset.name}\n")
    finally:
        batch_file.close()

    known_hosts = str((Path.home() / ".ssh" / "known_hosts").resolve())
    cmd = [
        "sftp",
        "-P",
        str(profile.port),
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        "-b",
        str(batch_path),
        f"{profile.username}@{profile.host}",
    ]
    return cmd, batch_path


def setup_record_tab(
    *,
    root: Any,
    record_tab: Any,
    config: dict[str, Any],
    colors: dict[str, str],
    cv2_probe_ok: bool,
    cv2_probe_error: str,
    choose_folder: Callable[[Any], None],
    log_panel: GuiLogPanel,
    messagebox: Any,
    set_running: Callable[[bool, str | None, bool], None],
    run_process_async: GuiRunProcessAsync,
    on_camera_indices_changed: Callable[[int, int], None],
    refresh_header_subtitle: Callable[[], None],
    last_command_state: dict[str, str],
    confirm_preflight_in_gui: Callable[[str, list[tuple[str, str, str]]], bool],
) -> RecordTabHandles:
    import tkinter as tk
    from tkinter import ttk

    record_container = ttk.Frame(record_tab, style="Panel.TFrame")
    record_container.pack(fill="both", expand=True)

    suggested_dataset, _ = suggest_dataset_name(config)
    record_dataset_var = tk.StringVar(value=suggested_dataset)
    record_episodes_var = tk.StringVar(value="20")
    record_duration_var = tk.StringVar(value="20")
    record_task_var = tk.StringVar(value=DEFAULT_TASK)
    record_dir_var = tk.StringVar(value=str(config["record_data_dir"]))
    record_upload_var = tk.BooleanVar(value=False)
    record_hf_username_var = tk.StringVar(value=str(config.get("hf_username", "")).strip())
    record_hf_repo_name_var = tk.StringVar(value=repo_name_from_repo_id(record_dataset_var.get().strip()))
    record_tag_after_upload_var = tk.BooleanVar(value=bool(config.get("record_tag_after_upload", True)))
    record_advanced_enabled_var = tk.BooleanVar(value=False)
    record_custom_args_var = tk.StringVar(value="")
    lerobot_dir = get_lerobot_dir(config)

    record_form = ttk.LabelFrame(record_container, text="Recording Setup", style="Section.TLabelframe", padding=12)
    record_form.pack(fill="x")
    record_form.columnconfigure(1, weight=1)

    ttk.Label(record_form, text="Dataset name (or repo id)", style="Field.TLabel").grid(
        row=0,
        column=0,
        sticky="w",
        padx=(0, 6),
        pady=4,
    )
    ttk.Entry(record_form, textvariable=record_dataset_var, width=52).grid(row=0, column=1, sticky="ew", pady=4)
    ttk.Button(
        record_form,
        text="Suggest Next",
        command=lambda: record_dataset_var.set(suggest_dataset_name(config)[0]),
    ).grid(row=0, column=2, sticky="w", padx=(6, 0), pady=4)

    ttk.Label(record_form, text="Local dataset save folder", style="Field.TLabel").grid(
        row=1,
        column=0,
        sticky="w",
        padx=(0, 6),
        pady=4,
    )
    ttk.Entry(record_form, textvariable=record_dir_var, width=52).grid(row=1, column=1, sticky="ew", pady=4)
    ttk.Button(record_form, text="Browse", command=lambda: choose_folder(record_dir_var)).grid(
        row=1,
        column=2,
        sticky="w",
        padx=(6, 0),
        pady=4,
    )

    ttk.Label(record_form, text="Episodes", style="Field.TLabel").grid(row=2, column=0, sticky="w", padx=(0, 6), pady=4)
    ttk.Entry(record_form, textvariable=record_episodes_var, width=20).grid(row=2, column=1, sticky="w", pady=4)

    ttk.Label(record_form, text="Episode time (seconds)", style="Field.TLabel").grid(
        row=3,
        column=0,
        sticky="w",
        padx=(0, 6),
        pady=4,
    )
    ttk.Entry(record_form, textvariable=record_duration_var, width=20).grid(row=3, column=1, sticky="w", pady=4)

    ttk.Label(record_form, text="Task description", style="Field.TLabel").grid(row=4, column=0, sticky="w", padx=(0, 6), pady=4)
    ttk.Entry(record_form, textvariable=record_task_var, width=52).grid(row=4, column=1, sticky="ew", pady=4)

    ttk.Checkbutton(record_form, text="Upload to Hugging Face after recording", variable=record_upload_var).grid(
        row=5,
        column=1,
        sticky="w",
        pady=(8, 8),
    )

    upload_options = ttk.LabelFrame(record_form, text="Upload Options", style="Section.TLabelframe", padding=10)
    upload_options.columnconfigure(1, weight=1)
    ttk.Label(upload_options, text="Hugging Face username", style="Field.TLabel").grid(
        row=0,
        column=0,
        sticky="w",
        padx=(0, 6),
        pady=4,
    )
    ttk.Entry(upload_options, textvariable=record_hf_username_var, width=24).grid(
        row=0,
        column=1,
        sticky="w",
        pady=4,
    )

    ttk.Label(upload_options, text="Dataset name on Hugging Face", style="Field.TLabel").grid(
        row=1,
        column=0,
        sticky="w",
        padx=(0, 6),
        pady=4,
    )
    ttk.Entry(upload_options, textvariable=record_hf_repo_name_var, width=40).grid(
        row=1,
        column=1,
        sticky="ew",
        pady=4,
    )
    ttk.Button(
        upload_options,
        text="Use Dataset Field",
        command=lambda: record_hf_repo_name_var.set(repo_name_from_repo_id(record_dataset_var.get().strip())),
    ).grid(row=1, column=2, sticky="w", padx=(6, 0), pady=4)

    ttk.Checkbutton(
        upload_options,
        text="Run LeRobot v3.0 conversion/tagging after upload",
        variable=record_tag_after_upload_var,
    ).grid(row=2, column=1, sticky="w", pady=(6, 2))

    ttk.Label(
        upload_options,
        text="Runs: python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 --repo-id=<repo>",
        style="Muted.TLabel",
    ).grid(row=3, column=1, sticky="w", pady=(0, 2))

    record_advanced_fields = [
        ("robot.type", "Robot type"),
        ("robot.port", "Follower port"),
        ("robot.id", "Follower robot id"),
        ("robot.cameras", "Robot cameras JSON"),
        ("teleop.type", "Teleop type"),
        ("teleop.port", "Leader port"),
        ("teleop.id", "Leader robot id"),
        ("dataset.repo_id", "Dataset repo id"),
        ("dataset.num_episodes", "Dataset episodes"),
        ("dataset.single_task", "Dataset task"),
        ("dataset.episode_time_s", "Episode time (s)"),
    ]
    record_advanced_vars: dict[str, Any] = {
        key: tk.StringVar(value="")
        for key, _ in record_advanced_fields
    }

    ttk.Checkbutton(
        record_form,
        text="Advanced command options",
        variable=record_advanced_enabled_var,
    ).grid(row=7, column=1, sticky="w", pady=(6, 0))

    record_advanced_frame = ttk.LabelFrame(
        record_form,
        text="Advanced Record Options",
        style="Section.TLabelframe",
        padding=10,
    )
    record_advanced_frame.columnconfigure(1, weight=1)
    ttk.Label(
        record_advanced_frame,
        text="Fields are prefilled from the setup panel. Edit to override, or clear a field to use setup defaults.",
        style="Muted.TLabel",
    ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

    for idx, (key, label) in enumerate(record_advanced_fields, start=1):
        ttk.Label(record_advanced_frame, text=f"{label} (--{key})", style="Field.TLabel").grid(
            row=idx,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=3,
        )
        ttk.Entry(record_advanced_frame, textvariable=record_advanced_vars[key], width=58).grid(
            row=idx,
            column=1,
            sticky="ew",
            pady=3,
        )

    custom_row = len(record_advanced_fields) + 1
    ttk.Label(record_advanced_frame, text="Custom args (raw)", style="Field.TLabel").grid(
        row=custom_row,
        column=0,
        sticky="w",
        padx=(0, 8),
        pady=(8, 3),
    )
    ttk.Entry(record_advanced_frame, textvariable=record_custom_args_var, width=58).grid(
        row=custom_row,
        column=1,
        sticky="ew",
        pady=(8, 3),
    )

    record_buttons = ttk.Frame(record_form, style="Panel.TFrame")
    record_buttons.grid(row=9, column=1, sticky="w", pady=(8, 0))
    preview_record_button = ttk.Button(record_buttons, text="Preview Command")
    preview_record_button.pack(side="left")
    run_record_button = ttk.Button(record_buttons, text="Run Record", style="Accent.TButton")
    run_record_button.pack(side="left", padx=(10, 0))
    push_olympus_button = ttk.Button(record_buttons, text="Upload Dataset to Olympus...")
    push_olympus_button.pack(side="left", padx=(10, 0))
    sync_hf_button = ttk.Button(record_buttons, text="Deploy Dataset to Hugging Face...")
    sync_hf_button.pack(side="left", padx=(10, 0))

    record_summary_var = tk.StringVar(value="")
    record_summary_panel = ttk.LabelFrame(record_container, text="Current Robot Snapshot", style="Section.TLabelframe", padding=10)
    record_summary_panel.pack(fill="x", pady=(10, 0))
    ttk.Label(record_summary_panel, textvariable=record_summary_var, style="Muted.TLabel", justify="left").pack(anchor="w")

    record_camera_preview = DualCameraPreview(
        root=root,
        parent=record_container,
        title="Record Camera Preview",
        config=config,
        colors=colors,
        cv2_probe_ok=cv2_probe_ok,
        cv2_probe_error=cv2_probe_error,
        append_log=log_panel.append_log,
        on_camera_indices_changed=on_camera_indices_changed,
    )

    def refresh_record_summary() -> None:
        record_summary_var.set(
            "Follower port: {follower} | Leader port: {leader}\n"
            "Laptop camera idx: {laptop} | Phone camera idx: {phone}\n"
            "Camera stream size: auto-detected at runtime | FPS: {fps} (warmup {warmup}s)".format(
                follower=config["follower_port"],
                leader=config["leader_port"],
                laptop=config["camera_laptop_index"],
                phone=config["camera_phone_index"],
                fps=config.get("camera_fps", 30),
                warmup=config["camera_warmup_s"],
            )
        )

    def _local_dataset_candidates() -> list[Path]:
        root_value = record_dir_var.get().strip() or str(config["record_data_dir"])
        return _list_local_dataset_dirs(Path(normalize_path(root_value)), lerobot_dir)

    def _cleanup_temp_file(path: Path | None) -> None:
        if path is None:
            return
        try:
            path.unlink()
        except OSError:
            pass

    push_popup_state: dict[str, Any] = {"window": None}
    hf_sync_popup_state: dict[str, Any] = {"window": None}

    def open_push_to_olympus_popup() -> None:
        popup = push_popup_state.get("window")
        if popup is not None and bool(popup.winfo_exists()):
            popup.deiconify()
            popup.lift()
            popup.focus_force()
            return

        from tkinter import filedialog as _fd

        olympus_profile = _pick_olympus_profile(config)
        dataset_candidates = _local_dataset_candidates()

        default_local_dataset = str(config.get("record_push_local_dataset", "")).strip()
        if not default_local_dataset and dataset_candidates:
            default_local_dataset = str(dataset_candidates[0])

        host_var = tk.StringVar(value=str(config.get("record_push_host", "")).strip() or olympus_profile.host)
        port_var = tk.StringVar(value=str(config.get("record_push_port", "")).strip() or str(olympus_profile.port))
        username_var = tk.StringVar(value=str(config.get("record_push_username", "")).strip() or olympus_profile.username)
        remote_parent_var = tk.StringVar(value=str(config.get("record_push_remote_parent", "")).strip() or "~/lerobot/data")
        local_dataset_var = tk.StringVar(value=default_local_dataset)
        prefer_rsync_var = tk.BooleanVar(value=bool(config.get("record_push_prefer_rsync", True)))
        attach_terminal_var = tk.BooleanVar(value=bool(config.get("record_push_attach_terminal", True)))
        status_var = tk.StringVar(value="Upload local dataset folder to Olympus over SSH.")

        popup = tk.Toplevel(root)
        popup.title("Upload Dataset to Olympus")
        popup.geometry("920x420")
        popup.minsize(820, 360)
        popup.configure(bg=colors.get("panel", "#111111"))
        popup.transient(root)
        push_popup_state["window"] = popup

        def _on_close() -> None:
            push_popup_state["window"] = None
            popup.destroy()

        popup.protocol("WM_DELETE_WINDOW", _on_close)

        container = ttk.Frame(popup, style="Panel.TFrame", padding=12)
        container.pack(fill="both", expand=True)
        container.columnconfigure(1, weight=1)
        container.columnconfigure(3, weight=1)

        ttk.Label(container, text="Host", style="Field.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=4)
        ttk.Entry(container, textvariable=host_var, width=30).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Label(container, text="Port", style="Field.TLabel").grid(row=0, column=2, sticky="w", padx=(12, 6), pady=4)
        ttk.Entry(container, textvariable=port_var, width=12).grid(row=0, column=3, sticky="w", pady=4)

        ttk.Label(container, text="Username", style="Field.TLabel").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=4)
        ttk.Entry(container, textvariable=username_var, width=30).grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(container, text="Remote dataset parent", style="Field.TLabel").grid(
            row=1, column=2, sticky="w", padx=(12, 6), pady=4,
        )
        ttk.Entry(container, textvariable=remote_parent_var, width=34).grid(row=1, column=3, sticky="ew", pady=4)

        ttk.Label(container, text="Local dataset folder", style="Field.TLabel").grid(
            row=2, column=0, sticky="w", padx=(0, 6), pady=4,
        )
        ttk.Entry(container, textvariable=local_dataset_var, width=66).grid(
            row=2, column=1, columnspan=2, sticky="ew", pady=4,
        )

        browse_local_button = ttk.Button(container, text="Browse...")
        browse_local_button.grid(row=2, column=3, sticky="w", pady=4)

        ttk.Label(container, text="Detected local datasets", style="Field.TLabel").grid(
            row=3, column=0, sticky="w", padx=(0, 6), pady=4,
        )
        dataset_combo = ttk.Combobox(container, values=[str(path) for path in dataset_candidates], state="readonly", width=66)
        dataset_combo.grid(row=3, column=1, columnspan=2, sticky="ew", pady=4)
        refresh_datasets_button = ttk.Button(container, text="Refresh Datasets")
        refresh_datasets_button.grid(row=3, column=3, sticky="w", pady=4)

        controls_row = ttk.Frame(container, style="Panel.TFrame")
        controls_row.grid(row=4, column=1, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Checkbutton(
            controls_row,
            text="Prefer rsync (fallback to sftp if needed)",
            variable=prefer_rsync_var,
        ).pack(anchor="w")
        ttk.Checkbutton(
            controls_row,
            text="Attach terminal input for SSH password prompts",
            variable=attach_terminal_var,
        ).pack(anchor="w")

        buttons_row = ttk.Frame(container, style="Panel.TFrame")
        buttons_row.grid(row=5, column=1, columnspan=3, sticky="w", pady=(10, 0))
        preview_push_button = ttk.Button(buttons_row, text="Preview Upload Command")
        preview_push_button.pack(side="left")
        run_push_button = ttk.Button(buttons_row, text="Upload to Olympus", style="Accent.TButton")
        run_push_button.pack(side="left", padx=(8, 0))

        ttk.Label(container, textvariable=status_var, style="Muted.TLabel", justify="left").grid(
            row=6,
            column=1,
            columnspan=3,
            sticky="w",
            pady=(8, 0),
        )

        def _refresh_dataset_options() -> None:
            options = [str(path) for path in _local_dataset_candidates()]
            dataset_combo.configure(values=options)
            current = local_dataset_var.get().strip()
            if not current and options:
                local_dataset_var.set(options[0])
            if current and current in options:
                dataset_combo.set(current)

        def _choose_local_dataset() -> None:
            current = local_dataset_var.get().strip()
            initial_dir = current or str(record_dir_var.get().strip() or config["record_data_dir"])
            selected = ask_directory_dialog(
                root=root,
                filedialog=_fd,
                initial_dir=initial_dir,
                title="Select local dataset folder",
            )
            if selected:
                local_dataset_var.set(selected)

        def _save_push_settings() -> None:
            config["record_push_host"] = host_var.get().strip()
            config["record_push_port"] = port_var.get().strip()
            config["record_push_username"] = username_var.get().strip()
            config["record_push_remote_parent"] = remote_parent_var.get().strip()
            config["record_push_local_dataset"] = local_dataset_var.get().strip()
            config["record_push_prefer_rsync"] = bool(prefer_rsync_var.get())
            config["record_push_attach_terminal"] = bool(attach_terminal_var.get())
            save_config(config, quiet=True)

        def _build_push_request() -> tuple[dict[str, Any] | None, str | None]:
            host = host_var.get().strip()
            username = username_var.get().strip()
            remote_parent = remote_parent_var.get().strip()
            local_dataset = Path(normalize_path(local_dataset_var.get().strip()))
            if not host:
                return None, "Host is required."
            if not username:
                return None, "Username is required."
            if not remote_parent:
                return None, "Remote dataset parent path is required."
            if not local_dataset.exists() or not local_dataset.is_dir():
                return None, f"Local dataset folder not found: {local_dataset}"

            try:
                port = int(port_var.get().strip() or "22")
            except ValueError:
                return None, "Port must be an integer."
            if port <= 0 or port > 65535:
                return None, "Port must be between 1 and 65535."

            profile = _build_ssh_profile(host=host, port=port, username=username)
            ssh_path = shutil.which("ssh")
            sftp_path = shutil.which("sftp")
            rsync_path = shutil.which("rsync")
            if not ssh_path:
                return None, "ssh binary not found in PATH."
            if not sftp_path and not rsync_path:
                return None, "Neither sftp nor rsync is available in PATH."

            if prefer_rsync_var.get() and rsync_path:
                primary_cmd = _build_rsync_push_command(profile, local_dataset, remote_parent)
                primary_mode = "rsync"
                primary_batch = None
            else:
                primary_cmd, primary_batch = _build_sftp_push_command(profile, local_dataset, remote_parent)
                primary_mode = "sftp"

            checks: list[tuple[str, str, str]] = [
                ("PASS", "Local dataset", str(local_dataset)),
                ("PASS", "SSH target", f"{username}@{host}:{port}"),
                ("PASS", "Remote dataset parent", remote_parent),
                ("PASS" if ssh_path else "FAIL", "ssh binary", ssh_path or "not found"),
                ("PASS" if sftp_path else "FAIL", "sftp binary", sftp_path or "not found"),
                ("PASS" if rsync_path else "WARN", "rsync binary", rsync_path or "not found"),
            ]
            if primary_mode == "rsync" and sftp_path:
                checks.append(("PASS", "Fallback", "sftp fallback available if rsync fails"))
            elif primary_mode == "rsync":
                checks.append(("WARN", "Fallback", "sftp fallback unavailable"))

            return {
                "profile": profile,
                "local_dataset": local_dataset,
                "remote_parent": remote_parent,
                "primary_cmd": primary_cmd,
                "primary_mode": primary_mode,
                "primary_batch": primary_batch,
                "allow_fallback": bool(primary_mode == "rsync" and sftp_path),
                "checks": checks,
            }, None

        def _run_push_command(
            *,
            request: dict[str, Any],
            command: list[str],
            mode: str,
            batch_file: Path | None,
            allow_fallback: bool,
        ) -> None:
            profile = request["profile"]
            local_dataset = request["local_dataset"]
            remote_parent = request["remote_parent"]
            run_mode = "train_attach" if attach_terminal_var.get() else "run"

            def on_start_error(_exc: Exception) -> None:
                _cleanup_temp_file(batch_file)

            def on_complete(return_code: int, was_canceled: bool) -> None:
                _cleanup_temp_file(batch_file)
                if was_canceled:
                    set_running(False, "Olympus upload canceled.")
                    messagebox.showinfo("Canceled", "Olympus upload was canceled.")
                    return
                if return_code == 0:
                    _save_push_settings()
                    set_running(False, f"Olympus upload completed ({mode}).")
                    messagebox.showinfo(
                        "Upload Complete",
                        (
                            "Dataset upload to Olympus completed.\n\n"
                            f"Local dataset: {local_dataset}\n"
                            f"Remote destination parent: {remote_parent}"
                        ),
                    )
                    return

                if mode == "rsync" and allow_fallback:
                    retry = messagebox.askyesno(
                        "rsync failed",
                        "rsync upload failed. Retry once with sftp fallback?",
                    )
                    if retry:
                        fallback_cmd, fallback_batch = _build_sftp_push_command(profile, local_dataset, remote_parent)
                        set_running(False, "Retrying upload with sftp...")
                        _run_push_command(
                            request=request,
                            command=fallback_cmd,
                            mode="sftp",
                            batch_file=fallback_batch,
                            allow_fallback=False,
                        )
                        return

                set_running(False, "Olympus upload failed.", True)
                messagebox.showerror("Upload Failed", f"Olympus upload failed with exit code {return_code}.")

            run_process_async(
                command,
                None,
                on_complete,
                None,
                None,
                run_mode,
                request["checks"],
                {
                    "training_profile": "Record Olympus Upload",
                    "remote_host": f"{profile.username}@{profile.host}:{profile.port}",
                    "remote_path": remote_parent,
                    "local_path": str(local_dataset),
                    "template_name": "record_push_olympus",
                    "training_transport": mode,
                },
                on_start_error,
            )

        def preview_push_command() -> None:
            request, error_text = _build_push_request()
            if error_text or request is None:
                messagebox.showerror("Upload to Olympus", error_text or "Unable to build upload command.")
                return
            try:
                last_command_state["value"] = format_command(request["primary_cmd"])
                details = [
                    "Primary upload command:",
                    format_command_for_dialog(request["primary_cmd"]),
                ]
                if request["allow_fallback"]:
                    details.append("\nFallback: if rsync fails, the app retries once with sftp.")
                show_text_dialog(
                    root=root,
                    title="Olympus Upload Command",
                    text="\n".join(details),
                    copy_text=last_command_state["value"],
                    wrap_mode="word",
                )
            finally:
                _cleanup_temp_file(request["primary_batch"])

        def run_push_command() -> None:
            request, error_text = _build_push_request()
            if error_text or request is None:
                messagebox.showerror("Upload to Olympus", error_text or "Unable to build upload command.")
                return

            if not confirm_preflight_in_gui("Olympus Dataset Upload Preflight", request["checks"]):
                _cleanup_temp_file(request["primary_batch"])
                return

            if not ask_text_dialog(
                root=root,
                title="Confirm Olympus Upload",
                text=(
                    "Review the upload command below.\n"
                    "Click Confirm to run it, or Cancel to stop.\n\n"
                    + format_command_for_dialog(request["primary_cmd"])
                ),
                copy_text=format_command(request["primary_cmd"]),
                confirm_label="Confirm",
                cancel_label="Cancel",
                wrap_mode="char",
            ):
                _cleanup_temp_file(request["primary_batch"])
                return

            trusted, trust_detail = ensure_host_trusted(request["profile"], messagebox)
            if not trusted:
                _cleanup_temp_file(request["primary_batch"])
                messagebox.showerror("SSH Trust", trust_detail)
                return
            log_panel.append_log(f"SSH trust check: {trust_detail}")

            _save_push_settings()
            _run_push_command(
                request=request,
                command=request["primary_cmd"],
                mode=request["primary_mode"],
                batch_file=request["primary_batch"],
                allow_fallback=bool(request["allow_fallback"]),
            )

        def _on_dataset_combo_selected(_: Any) -> None:
            selected = dataset_combo.get().strip()
            if selected:
                local_dataset_var.set(selected)

        browse_local_button.configure(command=_choose_local_dataset)
        refresh_datasets_button.configure(command=_refresh_dataset_options)
        dataset_combo.bind("<<ComboboxSelected>>", _on_dataset_combo_selected)
        preview_push_button.configure(command=preview_push_command)
        run_push_button.configure(command=run_push_command)
        _refresh_dataset_options()

    def open_sync_to_hf_popup() -> None:
        popup = hf_sync_popup_state.get("window")
        if popup is not None and bool(popup.winfo_exists()):
            popup.deiconify()
            popup.lift()
            popup.focus_force()
            return

        from tkinter import filedialog as _fd

        dataset_candidates = _local_dataset_candidates()
        default_local_dataset = str(config.get("record_hf_sync_local_dataset", "")).strip()
        if not default_local_dataset and dataset_candidates:
            default_local_dataset = str(dataset_candidates[0])

        default_owner = str(config.get("record_hf_sync_owner", "")).strip() or str(config.get("hf_username", "")).strip()
        default_dataset_name = str(config.get("record_hf_sync_repo_name", "")).strip()
        if not default_dataset_name and default_local_dataset:
            default_dataset_name = Path(default_local_dataset).name

        local_dataset_var = tk.StringVar(value=default_local_dataset)
        owner_var = tk.StringVar(value=default_owner)
        repo_name_var = tk.StringVar(value=default_dataset_name)
        convert_var = tk.BooleanVar(value=bool(config.get("record_hf_convert_after_upload", record_tag_after_upload_var.get())))
        skip_if_exists_var = tk.BooleanVar(value=bool(config.get("record_hf_sync_skip_if_exists", True)))
        status_var = tk.StringVar(value="Upload local dataset to Hugging Face with local/remote parity checks.")

        popup = tk.Toplevel(root)
        popup.title("Deploy Dataset to Hugging Face")
        popup.geometry("940x430")
        popup.minsize(840, 370)
        popup.configure(bg=colors.get("panel", "#111111"))
        popup.transient(root)
        hf_sync_popup_state["window"] = popup

        def _on_close() -> None:
            hf_sync_popup_state["window"] = None
            popup.destroy()

        popup.protocol("WM_DELETE_WINDOW", _on_close)

        container = ttk.Frame(popup, style="Panel.TFrame", padding=12)
        container.pack(fill="both", expand=True)
        container.columnconfigure(1, weight=1)
        container.columnconfigure(3, weight=1)

        ttk.Label(container, text="Local dataset folder", style="Field.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 6), pady=4,
        )
        ttk.Entry(container, textvariable=local_dataset_var, width=68).grid(
            row=0, column=1, columnspan=2, sticky="ew", pady=4,
        )
        browse_local_button = ttk.Button(container, text="Browse...")
        browse_local_button.grid(row=0, column=3, sticky="w", pady=4)

        ttk.Label(container, text="Detected local datasets", style="Field.TLabel").grid(
            row=1, column=0, sticky="w", padx=(0, 6), pady=4,
        )
        dataset_combo = ttk.Combobox(container, values=[str(path) for path in dataset_candidates], state="readonly", width=68)
        dataset_combo.grid(row=1, column=1, columnspan=2, sticky="ew", pady=4)
        refresh_datasets_button = ttk.Button(container, text="Refresh Datasets")
        refresh_datasets_button.grid(row=1, column=3, sticky="w", pady=4)

        ttk.Label(container, text="HF owner", style="Field.TLabel").grid(
            row=2, column=0, sticky="w", padx=(0, 6), pady=4,
        )
        ttk.Entry(container, textvariable=owner_var, width=28).grid(row=2, column=1, sticky="ew", pady=4)
        ttk.Label(container, text="HF dataset name", style="Field.TLabel").grid(
            row=2, column=2, sticky="w", padx=(12, 6), pady=4,
        )
        ttk.Entry(container, textvariable=repo_name_var, width=30).grid(row=2, column=3, sticky="ew", pady=4)

        parity_row = ttk.Frame(container, style="Panel.TFrame")
        parity_row.grid(row=3, column=1, columnspan=3, sticky="w", pady=(6, 0))
        check_parity_button = ttk.Button(parity_row, text="Check Local/Remote Parity")
        check_parity_button.pack(side="left")

        controls_row = ttk.Frame(container, style="Panel.TFrame")
        controls_row.grid(row=4, column=1, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Checkbutton(
            controls_row,
            text="Skip upload when remote dataset already exists",
            variable=skip_if_exists_var,
        ).pack(anchor="w")
        ttk.Checkbutton(
            controls_row,
            text="Run LeRobot v3.0 conversion/tagging after upload",
            variable=convert_var,
        ).pack(anchor="w")

        buttons_row = ttk.Frame(container, style="Panel.TFrame")
        buttons_row.grid(row=5, column=1, columnspan=3, sticky="w", pady=(10, 0))
        preview_sync_button = ttk.Button(buttons_row, text="Preview HF Upload Command")
        preview_sync_button.pack(side="left")
        run_sync_button = ttk.Button(buttons_row, text="Deploy to Hugging Face", style="Accent.TButton")
        run_sync_button.pack(side="left", padx=(8, 0))

        ttk.Label(container, textvariable=status_var, style="Muted.TLabel", justify="left").grid(
            row=6,
            column=1,
            columnspan=3,
            sticky="w",
            pady=(8, 0),
        )

        def _refresh_dataset_options() -> None:
            options = [str(path) for path in _local_dataset_candidates()]
            dataset_combo.configure(values=options)
            current = local_dataset_var.get().strip()
            if not current and options:
                local_dataset_var.set(options[0])
            if current and current in options:
                dataset_combo.set(current)

        def _choose_local_dataset() -> None:
            current = local_dataset_var.get().strip()
            initial_dir = current or str(record_dir_var.get().strip() or config["record_data_dir"])
            selected = ask_directory_dialog(
                root=root,
                filedialog=_fd,
                initial_dir=initial_dir,
                title="Select local dataset folder",
            )
            if selected:
                local_dataset_var.set(selected)
                if not repo_name_var.get().strip():
                    repo_name_var.set(Path(selected).name)

        def _save_hf_sync_settings() -> None:
            config["record_hf_sync_local_dataset"] = local_dataset_var.get().strip()
            config["record_hf_sync_owner"] = owner_var.get().strip()
            config["record_hf_sync_repo_name"] = repo_name_var.get().strip()
            config["record_hf_convert_after_upload"] = bool(convert_var.get())
            config["record_hf_sync_skip_if_exists"] = bool(skip_if_exists_var.get())
            save_config(config, quiet=True)

        def _build_hf_sync_request() -> tuple[dict[str, Any] | None, str | None]:
            local_dataset = Path(normalize_path(local_dataset_var.get().strip()))
            if not local_dataset.exists() or not local_dataset.is_dir():
                return None, f"Local dataset folder not found: {local_dataset}"

            repo_id = _compose_repo_id(owner_var.get(), repo_name_var.get())
            if repo_id is None:
                return None, "Hugging Face owner and dataset name are required."

            hf_cli_path = shutil.which("huggingface-cli")
            if hf_cli_path is None:
                return None, "huggingface-cli not found in PATH."

            upload_cmd = [
                "huggingface-cli",
                "upload",
                repo_id,
                str(local_dataset),
                "--repo-type",
                "dataset",
            ]

            exists = dataset_exists_on_hf(repo_id)
            parity_level, parity_detail = _hf_parity_detail(exists, repo_id)
            checks: list[tuple[str, str, str]] = [
                ("PASS", "Local dataset", str(local_dataset)),
                ("PASS", "Target repo", repo_id),
                ("PASS", "huggingface-cli", hf_cli_path),
                (parity_level, "Parity", parity_detail),
            ]

            convert_cmd = _build_v30_convert_command(repo_id) if convert_var.get() else None
            if convert_cmd is not None:
                checks.append(("PASS", "v3.0 convert command", format_command(convert_cmd)))

            return {
                "local_dataset": local_dataset,
                "repo_id": repo_id,
                "upload_cmd": upload_cmd,
                "convert_cmd": convert_cmd,
                "remote_exists": exists,
                "parity_detail": parity_detail,
                "checks": checks,
            }, None

        def _run_hf_convert_step(*, repo_id: str, convert_cmd: list[str], local_dataset: Path) -> None:
            set_running(False, "Upload completed. Running v3.0 conversion/tagging...")

            def after_convert(convert_code: int, convert_canceled: bool) -> None:
                if convert_canceled:
                    set_running(False, "HF upload completed. v3.0 conversion canceled.")
                    messagebox.showwarning(
                        "Upload Completed",
                        (
                            "Dataset upload completed, but v3.0 conversion was canceled.\n\n"
                            f"Local dataset: {local_dataset}\n"
                            f"Hugging Face repo: {repo_id}"
                        ),
                    )
                    return
                if convert_code != 0:
                    set_running(False, "HF upload completed. v3.0 conversion failed.", True)
                    messagebox.showwarning(
                        "Upload Completed (Conversion Warning)",
                        (
                            "Dataset upload completed, but v3.0 conversion/tagging failed.\n\n"
                            f"Local dataset: {local_dataset}\n"
                            f"Hugging Face repo: {repo_id}\n"
                            f"Conversion exit code: {convert_code}"
                        ),
                    )
                    return
                set_running(False, "HF upload + v3.0 conversion completed.")
                messagebox.showinfo(
                    "Done",
                    (
                        "Dataset upload and v3.0 conversion/tagging completed.\n\n"
                        f"Local dataset: {local_dataset}\n"
                        f"Hugging Face repo: {repo_id}"
                    ),
                )

            run_process_async(
                convert_cmd,
                lerobot_dir,
                after_convert,
                None,
                None,
                "upload",
                None,
                {"dataset_repo_id": repo_id},
            )

        def check_parity_now() -> None:
            request, error_text = _build_hf_sync_request()
            if error_text or request is None:
                messagebox.showerror("HF Parity Check", error_text or "Unable to build parity check.")
                return
            status_var.set(str(request.get("parity_detail", "Parity check complete.")))

        def preview_sync_command() -> None:
            request, error_text = _build_hf_sync_request()
            if error_text or request is None:
                messagebox.showerror("Deploy to Hugging Face", error_text or "Unable to build upload command.")
                return
            last_command_state["value"] = format_command(request["upload_cmd"])
            details = [
                "Upload command:",
                format_command_for_dialog(request["upload_cmd"]),
            ]
            convert_cmd = request["convert_cmd"]
            if convert_cmd is not None:
                details.append("\nThen run:")
                details.append(format_command_for_dialog(convert_cmd))
            show_text_dialog(
                root=root,
                title="HF Upload Command",
                text="\n".join(details),
                copy_text=last_command_state["value"],
                wrap_mode="word",
            )

        def run_sync_command() -> None:
            request, error_text = _build_hf_sync_request()
            if error_text or request is None:
                messagebox.showerror("Deploy to Hugging Face", error_text or "Unable to build upload command.")
                return

            repo_id = request["repo_id"]
            local_dataset = request["local_dataset"]
            remote_exists = request["remote_exists"]
            if remote_exists is True and skip_if_exists_var.get():
                messagebox.showinfo(
                    "Skipped",
                    f"Remote dataset already exists. Upload skipped:\n{repo_id}",
                )
                return
            if remote_exists is True and not messagebox.askyesno(
                "Remote Dataset Exists",
                f"{repo_id} already exists on Hugging Face.\nContinue upload anyway?",
            ):
                return
            if remote_exists is None and not messagebox.askyesno(
                "Parity Unknown",
                (
                    f"Could not verify remote parity for {repo_id}.\n"
                    "Continue upload anyway?"
                ),
            ):
                return

            if not confirm_preflight_in_gui("HF Dataset Deploy Preflight", request["checks"]):
                return

            if not ask_text_dialog(
                root=root,
                title="Confirm HF Dataset Deploy",
                text=(
                    "Review the upload command below.\n"
                    "Click Confirm to run it, or Cancel to stop.\n\n"
                    + format_command_for_dialog(request["upload_cmd"])
                ),
                copy_text=format_command(request["upload_cmd"]),
                confirm_label="Confirm",
                cancel_label="Cancel",
                wrap_mode="char",
            ):
                return

            _save_hf_sync_settings()
            config["hf_username"] = str(owner_var.get()).strip().strip("/") or str(config.get("hf_username", ""))
            config["record_tag_after_upload"] = bool(convert_var.get())
            save_config(config, quiet=True)
            refresh_header_subtitle()

            def after_upload(upload_code: int, upload_canceled: bool) -> None:
                if upload_canceled:
                    set_running(False, "HF upload canceled.")
                    messagebox.showinfo("Canceled", "Hugging Face upload was canceled.")
                    return
                if upload_code != 0:
                    set_running(False, "HF upload failed.", True)
                    messagebox.showerror("Upload Failed", f"Hugging Face upload failed with exit code {upload_code}.")
                    return

                convert_cmd = request["convert_cmd"]
                if convert_cmd is None:
                    set_running(False, "HF upload completed.")
                    messagebox.showinfo(
                        "Done",
                        (
                            "Dataset upload to Hugging Face completed.\n\n"
                            f"Local dataset: {local_dataset}\n"
                            f"Hugging Face repo: {repo_id}\n"
                            "v3.0 conversion/tagging: skipped"
                        ),
                    )
                    return

                _run_hf_convert_step(repo_id=repo_id, convert_cmd=convert_cmd, local_dataset=local_dataset)

            run_process_async(
                request["upload_cmd"],
                lerobot_dir,
                after_upload,
                None,
                None,
                "upload",
                request["checks"],
                {"dataset_repo_id": repo_id},
            )

        def _on_dataset_combo_selected(_: Any) -> None:
            selected = dataset_combo.get().strip()
            if selected:
                local_dataset_var.set(selected)
                if not repo_name_var.get().strip():
                    repo_name_var.set(Path(selected).name)

        browse_local_button.configure(command=_choose_local_dataset)
        refresh_datasets_button.configure(command=_refresh_dataset_options)
        dataset_combo.bind("<<ComboboxSelected>>", _on_dataset_combo_selected)
        check_parity_button.configure(command=check_parity_now)
        preview_sync_button.configure(command=preview_sync_command)
        run_sync_button.configure(command=run_sync_command)
        _refresh_dataset_options()

    def _seed_record_advanced_from_current() -> None:
        dataset_input, dataset_error = _record_dataset_input_from_ui()
        if dataset_error is not None or dataset_input is None:
            return
        req, cmd, error_text = build_record_request_and_command(
            config=config,
            dataset_input=dataset_input,
            episodes_raw=record_episodes_var.get(),
            duration_raw=record_duration_var.get(),
            task_raw=record_task_var.get(),
            dataset_dir_raw=record_dir_var.get(),
            upload_enabled=record_upload_var.get(),
        )
        if error_text or req is None or cmd is None:
            return
        for key, _ in record_advanced_fields:
            value = get_flag_value(cmd, key)
            if value is not None:
                record_advanced_vars[key].set(value)

    def _refresh_record_advanced_visibility(*_: Any) -> None:
        if record_advanced_enabled_var.get():
            _seed_record_advanced_from_current()
            record_advanced_frame.grid(row=8, column=1, columnspan=2, sticky="ew", pady=(2, 8))
        else:
            record_advanced_frame.grid_remove()

    def preview_record() -> None:
        req, cmd, error_text = build_current_record_from_ui()
        if error_text or req is None or cmd is None:
            messagebox.showerror("Validation Error", error_text or "Unable to build command.")
            return
        last_command_state["value"] = format_command(cmd)
        command_for_dialog = format_command_for_dialog(cmd)
        log_panel.append_log("Preview record command:")
        log_panel.append_log(last_command_state["value"])
        show_text_dialog(
            root=root,
            title="Record Command",
            text=command_for_dialog,
            wrap_mode="word",
        )

    def run_record_from_gui() -> None:
        req, cmd, error_text = build_current_record_from_ui()
        if error_text or req is None or cmd is None:
            messagebox.showerror("Validation Error", error_text or "Unable to build command.")
            return

        config["record_data_dir"] = str(req.dataset_root)
        resolved_repo_id, adjusted, _ = resolve_unique_repo_id(
            username=str(config["hf_username"]),
            dataset_name_or_repo_id=req.dataset_repo_id,
            local_roots=[req.dataset_root, lerobot_dir / "data"],
        )
        if adjusted:
            if record_upload_var.get():
                owner, name = resolved_repo_id.split("/", 1)
                record_hf_username_var.set(owner)
                record_hf_repo_name_var.set(name)
                log_panel.append_log(f"Auto-iterated Hugging Face dataset to avoid existing target: {resolved_repo_id}")
            else:
                record_dataset_var.set(resolved_repo_id)
                log_panel.append_log(f"Auto-iterated dataset to avoid existing target: {resolved_repo_id}")

            req, cmd, error_text = build_current_record_from_ui()
            if error_text or req is None or cmd is None:
                messagebox.showerror("Validation Error", error_text or "Unable to build command.")
                return

        exists = dataset_exists_on_hf(req.dataset_repo_id)
        if exists is True:
            proceed = messagebox.askyesno(
                "Dataset Exists",
                f"{req.dataset_repo_id} already exists on Hugging Face.\nContinue anyway?",
            )
            if not proceed:
                return

        if not ask_text_dialog(
            root=root,
            title="Confirm Record",
            text=(
                "Review the record command below.\n"
                "Click Confirm to run it, or Cancel to stop.\n\n"
                + format_command_for_dialog(cmd)
            ),
            confirm_label="Confirm",
            cancel_label="Cancel",
            wrap_mode="char",
        ):
            return

        preflight_checks = run_preflight_for_record(
            config=config,
            dataset_root=req.dataset_root,
            upload_enabled=req.upload_after_record,
            episode_time_s=req.episode_time_s,
            dataset_repo_id=req.dataset_repo_id,
        )
        if not confirm_preflight_in_gui("Record Preflight", preflight_checks):
            return

        def after_record(return_code: int, was_canceled: bool) -> None:
            if return_code != 0:
                if was_canceled:
                    set_running(False, "Record canceled.")
                    messagebox.showinfo("Canceled", "Record command was canceled.")
                else:
                    set_running(False, "Recording failed.", True)
                    messagebox.showerror("Record Failed", f"Recording failed with exit code {return_code}.")
                return

            active_dataset = move_recorded_dataset(
                lerobot_dir=lerobot_dir,
                dataset_name=req.dataset_name,
                dataset_root=req.dataset_root,
                log=log_panel.append_log,
            )

            config["last_dataset_name"] = req.dataset_name
            if record_upload_var.get():
                config["hf_username"] = str(record_hf_username_var.get()).strip().strip("/") or str(config.get("hf_username", ""))
            config["record_tag_after_upload"] = bool(record_tag_after_upload_var.get())
            save_config(config)
            next_dataset_name, _ = suggest_dataset_name(config)
            record_dataset_var.set(next_dataset_name)
            if record_upload_var.get():
                record_hf_repo_name_var.set(next_dataset_name)
            refresh_record_summary()
            refresh_header_subtitle()

            if was_canceled:
                set_running(False, "Record canceled.")
                messagebox.showinfo("Canceled", "Record command was canceled. Upload was skipped.")
                return

            if not req.upload_after_record:
                set_running(False, "Record completed.")
                messagebox.showinfo("Done", "Recording completed.")
                return

            upload_cmd = [
                "huggingface-cli",
                "upload",
                req.dataset_repo_id,
                str(active_dataset),
                "--repo-type",
                "dataset",
            ]

            set_running(False, "Record completed. Starting upload...")

            def after_upload(upload_code: int, upload_canceled: bool) -> None:
                if upload_canceled:
                    set_running(False, "Upload canceled.")
                    messagebox.showinfo("Canceled", "Upload command was canceled.")
                elif upload_code != 0:
                    set_running(False, "Upload failed.", True)
                    messagebox.showerror("Upload Failed", f"Upload failed with exit code {upload_code}.")
                else:
                    if not record_tag_after_upload_var.get():
                        set_running(False, "Record + upload completed.")
                        messagebox.showinfo(
                            "Done",
                            "Recording and upload completed.\n\n"
                            f"Hugging Face account: {req.dataset_repo_id.split('/', 1)[0]}\n"
                            f"Uploaded dataset: {req.dataset_name}\n"
                            f"Hugging Face repo: {req.dataset_repo_id}\n"
                            "v3.0 conversion/tagging: skipped",
                        )
                        return

                    convert_cmd = _build_v30_convert_command(req.dataset_repo_id)

                    set_running(False, "Upload completed. Running v3.0 conversion/tagging...")

                    def after_convert(convert_code: int, convert_canceled: bool) -> None:
                        base_details = (
                            "Recording and upload completed.\n\n"
                            f"Hugging Face account: {req.dataset_repo_id.split('/', 1)[0]}\n"
                            f"Uploaded name: {req.dataset_name}\n"
                            f"Hugging Face repo: {req.dataset_repo_id}\n"
                        )
                        if convert_canceled:
                            set_running(False, "Upload completed. v3.0 conversion canceled.")
                            messagebox.showwarning(
                                "Upload Completed",
                                base_details + "v3.0 conversion/tagging: canceled",
                            )
                            return
                        if convert_code != 0:
                            set_running(False, "Upload completed. v3.0 conversion failed.", True)
                            messagebox.showwarning(
                                "Upload Completed (Conversion Warning)",
                                base_details + f"v3.0 conversion/tagging: failed (exit code {convert_code})",
                            )
                            return

                        set_running(False, "Record + upload + v3.0 conversion completed.")
                        messagebox.showinfo(
                            "Done",
                            base_details + "v3.0 conversion/tagging: success",
                        )

                    run_process_async(
                        cmd=convert_cmd,
                        cwd=lerobot_dir,
                        complete_callback=after_convert,
                        expected_episodes=None,
                        expected_seconds=None,
                        run_mode="upload",
                        preflight_checks=None,
                        artifact_context={"dataset_repo_id": req.dataset_repo_id},
                    )

            run_process_async(
                upload_cmd,
                get_lerobot_dir(config),
                after_upload,
                None,
                None,
                "upload",
                None,
                {"dataset_repo_id": req.dataset_repo_id},
            )

        run_process_async(
            cmd,
            get_lerobot_dir(config),
            after_record,
            req.num_episodes,
            req.num_episodes * req.episode_time_s,
            "record",
            preflight_checks,
            {"dataset_repo_id": req.dataset_repo_id},
        )

    def _record_dataset_input_from_ui() -> tuple[str | None, str | None]:
        dataset_input = record_dataset_var.get()
        if record_upload_var.get():
            hf_username = str(record_hf_username_var.get()).strip().strip("/")
            hf_dataset_name = str(record_hf_repo_name_var.get()).strip().strip("/")
            if not hf_username:
                return None, "Hugging Face username is required when upload is enabled."
            if not hf_dataset_name:
                return None, "Hugging Face dataset name is required when upload is enabled."
            dataset_input = f"{hf_username}/{hf_dataset_name}"
        return dataset_input, None

    def _record_advanced_overrides_from_ui() -> tuple[dict[str, str] | None, str]:
        if not record_advanced_enabled_var.get():
            return None, ""
        overrides: dict[str, str] = {}
        for key, _ in record_advanced_fields:
            value = str(record_advanced_vars[key].get()).strip()
            if value:
                overrides[key] = value
        return overrides or None, str(record_custom_args_var.get())

    def build_current_record_from_ui() -> tuple[Any, Any, str | None]:
        dataset_input, dataset_error = _record_dataset_input_from_ui()
        if dataset_error is not None or dataset_input is None:
            return None, None, dataset_error or "Dataset name is required."
        arg_overrides, custom_args_raw = _record_advanced_overrides_from_ui()
        return build_record_request_and_command(
            config=config,
            dataset_input=dataset_input,
            episodes_raw=record_episodes_var.get(),
            duration_raw=record_duration_var.get(),
            task_raw=record_task_var.get(),
            dataset_dir_raw=record_dir_var.get(),
            upload_enabled=record_upload_var.get(),
            arg_overrides=arg_overrides,
            custom_args_raw=custom_args_raw,
        )

    def refresh_upload_options_visibility(*_: Any) -> None:
        if record_upload_var.get():
            if not str(record_hf_username_var.get()).strip():
                record_hf_username_var.set(str(config.get("hf_username", "")).strip())
            if not str(record_hf_repo_name_var.get()).strip():
                record_hf_repo_name_var.set(repo_name_from_repo_id(record_dataset_var.get().strip()))
            upload_options.grid(row=6, column=1, columnspan=2, sticky="ew", pady=(2, 8))
        else:
            upload_options.grid_remove()

    refresh_upload_options_visibility()
    record_upload_var.trace_add("write", refresh_upload_options_visibility)
    _refresh_record_advanced_visibility()
    record_advanced_enabled_var.trace_add("write", _refresh_record_advanced_visibility)

    preview_record_button.configure(command=preview_record)
    run_record_button.configure(command=run_record_from_gui)
    push_olympus_button.configure(command=open_push_to_olympus_popup)
    sync_hf_button.configure(command=open_sync_to_hf_popup)

    return RecordTabHandles(
        record_dir_var=record_dir_var,
        record_camera_preview=record_camera_preview,
        refresh_summary=refresh_record_summary,
        action_buttons=[preview_record_button, run_record_button, push_olympus_button, sync_hf_button],
    )
