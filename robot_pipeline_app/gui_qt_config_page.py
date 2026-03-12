from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
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
from .profile_io import apply_profile_payload, export_profile, import_profile, profile_preset_payloads
from .rig_manager import active_rig_name, apply_named_rig, delete_named_rig, list_named_rigs, save_named_rig
from .robot_presets import robot_preset_labels, robot_preset_payload
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
        on_config_changed: Callable[[], None] | None = None,
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
        self._on_config_changed = on_config_changed

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
        self.robot_preset_combo.addItems(robot_preset_labels())
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

        profile_card, profile_layout = _build_card("Profiles + Portable Presets")
        profile_row = QHBoxLayout()
        self.profile_preset_combo = QComboBox()
        self.profile_preset_combo.addItems(sorted(profile_preset_payloads().keys()))
        profile_row.addWidget(self.profile_preset_combo, 1)
        apply_profile_preset_button = QPushButton("Apply Portable Preset")
        apply_profile_preset_button.clicked.connect(self.apply_profile_preset)
        profile_row.addWidget(apply_profile_preset_button)
        profile_layout.addLayout(profile_row)

        profile_actions = QHBoxLayout()
        self.profile_apply_paths_check = QCheckBox("Apply path fields on import")
        profile_actions.addWidget(self.profile_apply_paths_check)
        import_profile_button = QPushButton("Import Profile")
        import_profile_button.clicked.connect(self.import_profile_from_file)
        profile_actions.addWidget(import_profile_button)
        export_profile_button = QPushButton("Export Profile")
        export_profile_button.clicked.connect(self.export_profile_to_file)
        profile_actions.addWidget(export_profile_button)
        profile_actions.addStretch(1)
        profile_layout.addLayout(profile_actions)
        profile_note = QLabel("Profiles update robot defaults, camera schema, rename-map hints, and setup guidance from the GUI.")
        profile_note.setWordWrap(True)
        profile_note.setObjectName("MutedLabel")
        profile_layout.addWidget(profile_note)
        self.content_layout.addWidget(profile_card)

        rig_card, rig_layout = _build_card("Named Rigs")
        rig_row = QHBoxLayout()
        self.rig_combo = QComboBox()
        self.rig_combo.setEditable(True)
        rig_row.addWidget(self.rig_combo, 1)
        apply_rig_button = QPushButton("Switch Rig")
        apply_rig_button.clicked.connect(self.apply_selected_rig)
        rig_row.addWidget(apply_rig_button)
        save_rig_button = QPushButton("Save Rig")
        save_rig_button.clicked.connect(self.save_named_rig_from_form)
        rig_row.addWidget(save_rig_button)
        delete_rig_button = QPushButton("Delete Rig")
        delete_rig_button.clicked.connect(self.delete_selected_rig)
        rig_row.addWidget(delete_rig_button)
        rig_layout.addLayout(rig_row)
        self.rig_status_label = QLabel("Save the current hardware/config state as a named rig, then switch quickly from the GUI.")
        self.rig_status_label.setWordWrap(True)
        self.rig_status_label.setObjectName("MutedLabel")
        rig_layout.addWidget(self.rig_status_label)
        self.content_layout.addWidget(rig_card)

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
        self._refresh_rig_controls()
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

    def _notify_config_changed(self) -> None:
        if self._on_config_changed is not None:
            self._on_config_changed()

    def _populate_form_from_config(self) -> None:
        for key, widget in self._inputs.items():
            value = self.config.get(key, "")
            if isinstance(widget, QSpinBox):
                widget.setValue(int(value or 0))
            elif isinstance(widget, QComboBox):
                widget.setCurrentText(str(value))
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))
        self.camera_schema_editor.reload_from_config(self.config)
        self._refresh_rig_controls()

    def _refresh_rig_controls(self) -> None:
        rigs = list_named_rigs(self.config)
        names = [str(item.get("name", "")).strip() for item in rigs]
        current_name = active_rig_name(self.config)
        self.rig_combo.blockSignals(True)
        self.rig_combo.clear()
        self.rig_combo.addItems(names)
        if current_name:
            if self.rig_combo.findText(current_name) < 0:
                self.rig_combo.addItem(current_name)
            self.rig_combo.setCurrentText(current_name)
        self.rig_combo.blockSignals(False)
        if current_name:
            self.rig_status_label.setText(f"Active rig: {current_name}")
        elif names:
            self.rig_status_label.setText("Select a saved rig to switch hardware context quickly.")
        else:
            self.rig_status_label.setText("No named rigs saved yet. Save the current form as a rig to enable fast switching.")

    def show_snapshot(self) -> None:
        preview = self._read_form()
        status = probe_setup_wizard_status(preview)
        active_profile = str(preview.get("active_profile_name", "")).strip() or None
        payload = {
            "config_preview": preview,
            "camera_schema_entries": self.camera_schema_editor.entries(),
            "runtime_snapshot": build_compat_snapshot(preview),
            "setup_status": build_setup_status_summary(status),
            "active_profile": active_profile,
        }
        self._set_output(title="Config Snapshot", text=_json_text(payload), log_message="Config snapshot refreshed.")

    def apply_robot_preset(self) -> None:
        selected_label = self.robot_preset_combo.currentText().strip()
        values = robot_preset_payload(selected_label)
        if values is None:
            self._set_output(title="Preset Missing", text=f"Unknown preset: {selected_label}", log_message="Config robot preset missing.")
            return
        for key, value in values.items():
            widget = self._inputs.get(key)
            if isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QComboBox):
                widget.setCurrentText(str(value))
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))
        self.show_snapshot()
        self._append_log(f"Applied robot preset: {selected_label}")
        self._notify_config_changed()

    def save_config_values(self) -> None:
        updated = self._read_form()
        self.config.clear()
        self.config.update(updated)
        save_config(self.config, quiet=True)
        self.camera_schema_editor.reload_from_config(self.config)
        self._refresh_rig_controls()
        self._set_output(title="Config Saved", text=_json_text(updated), log_message="Config values saved.")
        self._notify_config_changed()

    def _apply_import_result(self, result: Any, *, source_label: str) -> None:
        if not result.ok or result.updated_config is None:
            self._set_output(title="Profile Failed", text=result.message, log_message=f"{source_label} failed.")
            return
        self.config.clear()
        self.config.update(result.updated_config)
        save_config(self.config, quiet=True)
        self._populate_form_from_config()
        self.show_snapshot()
        self._set_output(
            title="Profile Applied",
            text=result.message + "\n\n" + _json_text({"applied_keys": list(result.applied_keys), "skipped_keys": list(result.skipped_keys)}),
            log_message=f"{source_label} applied.",
        )
        self._notify_config_changed()

    def apply_profile_preset(self) -> None:
        preset_name = self.profile_preset_combo.currentText().strip()
        payload = profile_preset_payloads().get(preset_name)
        if not isinstance(payload, dict):
            self._set_output(title="Preset Missing", text=f"Unknown preset: {preset_name}", log_message="Config profile preset missing.")
            return
        preview = self._read_form()
        preview["active_profile_name"] = preset_name
        result = apply_profile_payload(
            preview,
            payload={**payload, "name": preset_name},
            apply_paths=self.profile_apply_paths_check.isChecked(),
        )
        self._apply_import_result(result, source_label=f"Portable preset '{preset_name}'")

    def import_profile_from_file(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Import Community Profile",
            str(Path.home()),
            "Profile files (*.yaml *.yml *.json);;All files (*)",
        )
        if not selected:
            return
        result = import_profile(
            self._read_form(),
            input_path=Path(selected),
            apply_paths=self.profile_apply_paths_check.isChecked(),
        )
        if result.ok and result.updated_config is not None:
            result.updated_config["active_profile_name"] = Path(selected).stem
        self._apply_import_result(result, source_label=f"Profile import '{selected}'")

    def export_profile_to_file(self) -> None:
        selected, _filter = QFileDialog.getSaveFileName(
            self,
            "Export Community Profile",
            str(Path.home() / "lab-profile.yaml"),
            "Profile files (*.yaml *.yml *.json);;All files (*)",
        )
        if not selected:
            return
        name = str(self._read_form().get("active_profile_name", "")).strip()
        result = export_profile(
            self._read_form(),
            output_path=Path(selected),
            name=name,
            include_paths=self.profile_apply_paths_check.isChecked(),
        )
        self._set_output(
            title="Profile Exported" if result.ok else "Profile Export Failed",
            text=result.message + (f"\n- path: {result.output_path}" if result.output_path is not None else ""),
            log_message="Config profile export completed." if result.ok else "Config profile export failed.",
        )

    def save_named_rig_from_form(self) -> None:
        rig_name = self.rig_combo.currentText().strip()
        if not rig_name:
            self._set_output(title="Rig Name Required", text="Enter a rig name before saving.", log_message="Config rig save skipped with no name.")
            return
        updated = save_named_rig(self._read_form(), name=rig_name)
        self.config.clear()
        self.config.update(updated)
        save_config(self.config, quiet=True)
        self._populate_form_from_config()
        self.show_snapshot()
        self._set_output(title="Rig Saved", text=f"Saved named rig '{rig_name}'.", log_message=f"Saved named rig: {rig_name}")
        self._notify_config_changed()

    def apply_selected_rig(self) -> None:
        rig_name = self.rig_combo.currentText().strip()
        updated, error = apply_named_rig(self.config, name=rig_name)
        if error or updated is None:
            self._set_output(title="Rig Switch Failed", text=error or "Unable to apply the selected rig.", log_message="Config rig switch failed.")
            return
        self.config.clear()
        self.config.update(updated)
        save_config(self.config, quiet=True)
        self._populate_form_from_config()
        self.show_snapshot()
        self._set_output(title="Rig Switched", text=f"Active rig: {rig_name}", log_message=f"Switched to rig: {rig_name}")
        self._notify_config_changed()

    def delete_selected_rig(self) -> None:
        rig_name = self.rig_combo.currentText().strip()
        if not rig_name:
            self._set_output(title="Rig Name Required", text="Select or enter a rig name before deleting.", log_message="Config rig delete skipped with no name.")
            return
        updated = delete_named_rig(self.config, name=rig_name)
        self.config.clear()
        self.config.update(updated)
        save_config(self.config, quiet=True)
        self._populate_form_from_config()
        self.show_snapshot()
        self._set_output(title="Rig Deleted", text=f"Deleted rig '{rig_name}'.", log_message=f"Deleted rig: {rig_name}")
        self._notify_config_changed()

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
            active_profile = str(self._read_form().get("active_profile_name", "")).strip()
            if active_profile:
                guide = f"Active profile: {active_profile}\n\n{guide}"
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

    def refresh_from_config(self) -> None:
        self._populate_form_from_config()
        self.show_snapshot()
