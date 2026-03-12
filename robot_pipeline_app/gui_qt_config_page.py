from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLayout,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    import cv2 as _cv2_module  # type: ignore[import-not-found]

    _CV2_AVAILABLE = True
except Exception:  # pragma: no cover - fallback for minimal installs
    _cv2_module = None  # type: ignore[assignment]
    _CV2_AVAILABLE = False

from .artifacts import (
    _normalize_deploy_episode_outcomes,
    list_runs,
    normalize_deploy_result,
    write_deploy_episode_spreadsheet,
    write_deploy_notes_file,
)
from .camera_schema import apply_camera_schema_entries_to_config, camera_schema_entries_for_editor
from .checks import collect_doctor_checks, summarize_checks
from .compat_snapshot import build_compat_snapshot
from .config_store import _atomic_write, get_deploy_data_dir, get_lerobot_dir, normalize_config_without_prompts, save_config
from .constants import CONFIG_FIELDS
from .desktop_launcher import install_desktop_launcher
from .history_utils import (
    HISTORY_MODE_VALUES,
    _build_history_refresh_payload_from_runs,
    _command_from_item,
    open_path_in_file_manager,
)
from .gui_qt_dialogs import ask_text_dialog_with_actions
from .visualizer_utils import (
    _VisualizerRefreshSnapshot,
    _build_selection_payload,
    _collect_sources_for_refresh,
    _open_path,
    _visualizer_source_row_values,
)
from .repo_utils import normalize_deploy_rerun_command
from .run_controller_service import ManagedRunController, RunUiHooks
from .setup_wizard import build_setup_status_summary, build_setup_wizard_guide, probe_setup_wizard_status

from .gui_qt_page_base import (
    _CameraSchemaEditor,
    _InputGrid,
    _PageWithOutput,
    _VideoFrameLabel,
    _VideoGalleryTile,
    _build_card,
    _json_text,
    _quiet_cv2_logging,
    _set_readonly_table,
    _set_table_headers,
)

class QtConfigPage(_PageWithOutput):
    _ROBOT_PRESETS: tuple[tuple[str, dict[str, Any]], ...] = (
        (
            "SO-100",
            {
                "follower_robot_type": "so100_follower",
                "leader_robot_type": "so100_leader",
                "follower_robot_action_dim": 6,
            },
        ),
        (
            "SO-101",
            {
                "follower_robot_type": "so101_follower",
                "leader_robot_type": "so101_leader",
                "follower_robot_action_dim": 6,
            },
        ),
        (
            "Unitree G1 (29 DOF)",
            {
                "follower_robot_type": "unitree_g1_29dof",
                "leader_robot_type": "unitree_g1_29dof",
                "follower_robot_action_dim": 29,
            },
        ),
        (
            "Unitree G1 (23 DOF)",
            {
                "follower_robot_type": "unitree_g1_23dof",
                "leader_robot_type": "unitree_g1_23dof",
                "follower_robot_action_dim": 23,
            },
        ),
    )
    _GROUPS = (
        ("Paths", ["lerobot_dir", "lerobot_venv_dir", "runs_dir", "record_data_dir", "deploy_data_dir", "trained_models_dir"]),
        ("Ports + IDs", ["follower_port", "leader_port", "follower_robot_id", "leader_robot_id"]),
        ("Robot Defaults", ["follower_robot_type", "leader_robot_type", "follower_robot_action_dim"]),
        ("Deploy Defaults", ["record_target_hz", "deploy_target_hz", "eval_num_episodes", "eval_duration_s", "eval_task"]),
        ("Calibration + Hub", ["follower_calibration_path", "leader_calibration_path", "hf_username"]),
    )

    def __init__(
        self,
        *,
        config: dict[str, Any],
        append_log: Callable[[str], None],
        run_terminal_command: Callable[[str], tuple[bool, str]] | None = None,
        update_and_restart_app: Callable[[], tuple[bool, str]] | None = None,
    ) -> None:
        super().__init__(
            title="Config",
            subtitle="Edit the most important runtime paths, robot defaults, diagnostics, and launcher settings.",
            append_log=append_log,
        )
        self.config = config
        self._field_lookup = {field["key"]: field for field in CONFIG_FIELDS}
        self._inputs: dict[str, Any] = {}
        self._run_terminal_command = run_terminal_command
        self._update_and_restart_app = update_and_restart_app

        for title, keys in self._GROUPS:
            card, card_layout = _build_card(title)
            form = _InputGrid(card_layout)
            for key in keys:
                field = self._field_lookup[key]
                prompt = field["prompt"]
                current = str(self.config.get(key, "")).strip()
                if field["type"] == "int":
                    widget = QSpinBox()
                    widget.setRange(-1_000_000, 1_000_000)
                    try:
                        widget.setValue(int(self.config.get(key, 0) or 0))
                    except (TypeError, ValueError):
                        widget.setValue(0)
                elif field["type"] == "path":
                    widget = self._build_path_row(key, current)
                else:
                    widget = QLineEdit(current)
                form.add_field(prompt, widget)
                self._inputs[key] = self._input_target(widget)
            self.content_layout.addWidget(card)

        camera_card, camera_layout = _build_card("Camera Setup")
        self.camera_schema_editor = _CameraSchemaEditor(config=self.config)
        camera_layout.addWidget(self.camera_schema_editor)
        self.content_layout.addWidget(camera_card)

        preset_card, preset_layout = _build_card("Robot Presets")
        preset_row = QHBoxLayout()
        self.robot_preset_combo = QComboBox()
        self.robot_preset_combo.addItems([label for label, _values in self._ROBOT_PRESETS])
        preset_row.addWidget(self.robot_preset_combo, 1)
        apply_preset_button = QPushButton("Apply Preset")
        apply_preset_button.clicked.connect(self.apply_robot_preset)
        preset_row.addWidget(apply_preset_button)
        preset_layout.addLayout(preset_row)
        preset_note = QLabel("Prefills the existing robot type and action-dim fields. Everything remains editable afterwards.")
        preset_note.setWordWrap(True)
        preset_note.setObjectName("MutedLabel")
        preset_layout.addWidget(preset_note)
        self.content_layout.addWidget(preset_card)

        actions_card, actions_layout = _build_card("Actions")
        row = QHBoxLayout()
        save_button = QPushButton("Save Config")
        save_button.setObjectName("AccentButton")
        save_button.clicked.connect(self.save_config_values)
        row.addWidget(save_button)

        doctor_button = QPushButton("Run Doctor")
        doctor_button.clicked.connect(self.run_doctor)
        row.addWidget(doctor_button)

        setup_check_button = QPushButton("Run Setup Check")
        setup_check_button.clicked.connect(self.run_setup_check)
        row.addWidget(setup_check_button)

        guide_button = QPushButton("Open Setup Wizard")
        guide_button.clicked.connect(self.show_setup_guide)
        row.addWidget(guide_button)

        launcher_button = QPushButton("Install Launcher")
        launcher_button.clicked.connect(self.install_launcher)
        row.addWidget(launcher_button)
        row.addStretch(1)
        actions_layout.addLayout(row)
        self.content_layout.addWidget(actions_card)
        self.show_snapshot()

    def _build_path_row(self, key: str, current: str) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        line = QLineEdit(current)
        layout.addWidget(line, 1)
        browse = QPushButton("Browse")
        browse.clicked.connect(lambda: self._browse_path_for_field(key, line))
        layout.addWidget(browse)
        return row

    def _input_target(self, widget: QWidget) -> Any:
        if not isinstance(widget, (QLineEdit, QComboBox, QSpinBox)):
            target = widget.findChild(QLineEdit)
            if target is not None:
                return target
        return widget

    def _browse_path_for_field(self, key: str, target: QLineEdit) -> None:
        current = target.text().strip() or str(Path.home())
        if "calibration_path" in key:
            selected, _ = QFileDialog.getOpenFileName(
                self,
                "Select file",
                str(Path(current).expanduser().parent if current else Path.home()),
                "JSON files (*.json);;All files (*)",
            )
        else:
            selected = QFileDialog.getExistingDirectory(self, "Select folder", current)
        if selected:
            target.setText(selected)

    def _read_form(self) -> dict[str, Any]:
        updated = dict(self.config)
        for key, widget in self._inputs.items():
            if isinstance(widget, QComboBox):
                updated[key] = widget.currentText().strip()
            elif isinstance(widget, QSpinBox):
                updated[key] = int(widget.value())
            else:
                updated[key] = widget.text().strip()
        updated = self.camera_schema_editor.apply_to_config(updated)
        return normalize_config_without_prompts(updated)

    def show_snapshot(self) -> None:
        preview = self._read_form()
        status = probe_setup_wizard_status(preview)
        payload = {
            "config_preview": preview,
            "camera_schema_entries": self.camera_schema_editor.entries(),
            "runtime_snapshot": build_compat_snapshot(preview),
            "setup_status": build_setup_status_summary(status),
        }
        self._set_output(title="Config Snapshot", text=_json_text(payload), log_message="Config snapshot refreshed.")

    def apply_robot_preset(self) -> None:
        selected_label = self.robot_preset_combo.currentText().strip()
        for label, values in self._ROBOT_PRESETS:
            if label != selected_label:
                continue
            for key, value in values.items():
                widget = self._inputs.get(key)
                if isinstance(widget, QSpinBox):
                    widget.setValue(int(value))
                elif isinstance(widget, QComboBox):
                    widget.setCurrentText(str(value))
                elif isinstance(widget, QLineEdit):
                    widget.setText(str(value))
            self.show_snapshot()
            self._append_log(f"Applied robot preset: {label}")
            return

    def save_config_values(self) -> None:
        updated = self._read_form()
        self.config.clear()
        self.config.update(updated)
        save_config(self.config, quiet=True)
        self.camera_schema_editor.reload_from_config(self.config)
        self._set_output(title="Config Saved", text=_json_text(updated), log_message="Config values saved.")

    def run_doctor(self) -> None:
        checks = collect_doctor_checks(self._read_form())
        self._set_output(
            title="Doctor",
            text=summarize_checks(checks, title="Doctor"),
            log_message="Doctor run completed.",
        )

    def run_setup_check(self) -> None:
        status = probe_setup_wizard_status(self._read_form())
        self._set_output(
            title="Setup Check",
            text=build_setup_status_summary(status),
            log_message="Setup wizard environment check completed.",
        )

    def _default_activate_command(self) -> str:
        custom = str(self.config.get("setup_venv_activate_cmd", "")).strip()
        if custom:
            return custom
        preview = self._read_form()
        venv_dir = Path(str(preview.get("lerobot_venv_dir", "")).strip()).expanduser()
        return f'source "{venv_dir / "bin" / "activate"}"'

    def _send_activate_command(self, command: str, *, remember: bool) -> tuple[bool, str]:
        if remember:
            self.config["setup_venv_activate_cmd"] = command
            save_config(self.config, quiet=True)
        if self._run_terminal_command is None:
            return False, "Terminal shell is unavailable in this window."
        return self._run_terminal_command(command)

    def show_setup_guide(self) -> None:
        while True:
            status = probe_setup_wizard_status(self._read_form())
            summary = build_setup_status_summary(status)
            guide = build_setup_wizard_guide(status)
            actions: list[tuple[str, str]] = [
                ("activate_venv", "Activate Venv"),
                ("custom_activate", "Custom Source"),
                ("recheck", "Re-check Environment"),
            ]
            if self._update_and_restart_app is not None:
                actions.insert(0, ("update_restart", "Update and Restart"))
            action = ask_text_dialog_with_actions(
                parent=self.window() if isinstance(self.window(), QWidget) else None,
                title="LeRobot Setup Wizard",
                text=f"{summary}\n\n{guide}",
                actions=actions,
                confirm_label="Done",
                cancel_label="Close",
                wrap_mode="word",
            )
            if action == "update_restart" and self._update_and_restart_app is not None:
                ok, message = self._update_and_restart_app()
                self._set_output(
                    title="Update and Restart" if ok else "Update Failed",
                    text=message,
                    log_message="Setup wizard update requested." if ok else "Setup wizard update failed.",
                )
                if ok:
                    return
                continue
            if action == "activate_venv":
                ok, message = self._send_activate_command(self._default_activate_command(), remember=False)
                self._set_output(
                    title="Activation Sent" if ok else "Activation Failed",
                    text=message or self._default_activate_command(),
                    log_message="Setup wizard sent activation command." if ok else "Setup wizard activation failed.",
                )
                continue
            if action == "custom_activate":
                default = self._default_activate_command()
                custom, accepted = QInputDialog.getText(
                    self,
                    "Custom Venv Source",
                    "Enter the venv activation command to run in terminal.",
                    text=default,
                )
                if not accepted:
                    continue
                custom_command = str(custom).strip()
                if not custom_command:
                    continue
                ok, message = self._send_activate_command(custom_command, remember=True)
                self._set_output(
                    title="Activation Sent" if ok else "Activation Failed",
                    text=message or custom_command,
                    log_message="Setup wizard sent custom activation command." if ok else "Setup wizard custom activation failed.",
                )
                continue
            if action == "recheck":
                self.run_setup_check()
                continue
            self._append_log("Setup guide rendered.")
            self.status_label.setText("Setup Guide")
            return

    def install_launcher(self) -> None:
        updated = self._read_form()
        self.config.clear()
        self.config.update(updated)
        result = install_desktop_launcher(
            app_dir=Path(__file__).resolve().parents[1],
            python_executable=Path(sys.executable),
            venv_dir=Path(str(self.config.get("lerobot_venv_dir", "") or "")).expanduser(),
        )
        text = result.message
        if result.script_path is not None:
            text += f"\n- launcher script: {result.script_path}"
        if result.desktop_entry_path is not None:
            text += f"\n- desktop entry: {result.desktop_entry_path}"
        if result.icon_path is not None:
            text += f"\n- icon: {result.icon_path}"
        self._set_output(
            title="Launcher Installed" if result.ok else "Launcher Failed",
            text=text,
            log_message="Launcher install completed." if result.ok else "Launcher install failed.",
        )
