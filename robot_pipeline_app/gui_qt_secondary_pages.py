from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
    from PySide6.QtMultimediaWidgets import QVideoWidget

    _QT_MULTIMEDIA_AVAILABLE = True
except Exception:  # pragma: no cover - fallback for minimal Qt installs
    QAudioOutput = None  # type: ignore[assignment]
    QMediaPlayer = None  # type: ignore[assignment]
    QVideoWidget = None  # type: ignore[assignment]
    _QT_MULTIMEDIA_AVAILABLE = False

from .artifacts import (
    _normalize_deploy_episode_outcomes,
    list_runs,
    normalize_deploy_result,
    write_deploy_episode_spreadsheet,
    write_deploy_notes_file,
)
from .camera_schema import apply_camera_schema_entries_to_config, camera_schema_entries_for_editor
from .checks import collect_doctor_checks, summarize_checks
from .config_store import _atomic_write, get_deploy_data_dir, get_lerobot_dir, normalize_config_without_prompts, save_config
from .constants import CONFIG_FIELDS
from .desktop_launcher import install_desktop_launcher
from .gui_history_tab import (
    HISTORY_MODE_VALUES,
    _build_history_refresh_payload_from_runs,
    _command_from_item,
    open_path_in_file_manager,
)
from .gui_training_tab import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_ENV_ACTIVATE,
    DEFAULT_POLICY_INPUT_FEATURES,
    DEFAULT_POLICY_OUTPUT_FEATURES,
    DEFAULT_POLICY_PATH,
    DEFAULT_PROJECT_ROOT,
    DEFAULT_PYTHON_BIN,
    DEFAULT_SAVE_FREQ,
    DEFAULT_SRUN_CPUS_PER_TASK,
    DEFAULT_SRUN_GRES,
    DEFAULT_SRUN_PARTITION,
    DEFAULT_SRUN_QUEUE,
    DEFAULT_STEPS,
    _build_generated_train_command,
    _build_hil_workflow_text,
    _default_dataset_repo_id,
    _default_output_name,
    _expected_pretrained_model_path,
    _with_hil_suffix,
)
from .gui_qt_dialogs import ask_editable_command_dialog, ask_text_dialog_with_actions, show_text_dialog
from .gui_visualizer_tab import (
    _VisualizerRefreshSnapshot,
    _collect_sources_for_refresh,
    _collect_videos_for_source,
    _local_path_overview,
    _open_path,
    _visualizer_insights_section,
    _visualizer_source_row_values,
)
from .repo_utils import get_hf_dataset_info, get_hf_model_info, normalize_deploy_rerun_command
from .run_controller_service import ManagedRunController, RunUiHooks
from .setup_wizard import build_setup_status_summary, build_setup_wizard_guide, probe_setup_wizard_status
from .compat import resolve_train_entrypoint


def _build_card(title: str) -> tuple[QFrame, QVBoxLayout]:
    card = QFrame()
    card.setObjectName("SectionCard")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(18, 18, 18, 18)
    layout.setSpacing(12)
    header = QLabel(title)
    header.setObjectName("SectionMeta")
    layout.addWidget(header)
    return card, layout


def _set_readonly_table(table: QTableWidget) -> None:
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.verticalHeader().setVisible(False)
    table.setAlternatingRowColors(True)
    table.setWordWrap(False)


def _set_table_headers(table: QTableWidget, headers: list[str]) -> None:
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)
    header = table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    header.setStretchLastSection(True)


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, default=str)


class _InputGrid:
    def __init__(self, layout: QVBoxLayout) -> None:
        self._grid = QGridLayout()
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(14)
        self._grid.setVerticalSpacing(10)
        self._grid.setColumnStretch(1, 1)
        self._grid.setColumnStretch(3, 1)
        self._index = 0
        layout.addLayout(self._grid)

    def add_field(self, label_text: str, widget: QWidget) -> None:
        row = self._index // 2
        pair = self._index % 2
        label_col = pair * 2
        widget_col = label_col + 1
        label = QLabel(label_text)
        label.setObjectName("FormLabel")
        self._grid.addWidget(label, row, label_col)
        self._grid.addWidget(widget, row, widget_col)
        self._index += 1


class _CameraSchemaEditor(QFrame):
    _HEADERS = ["Name", "Source", "Type", "Width", "Height", "FPS", "Warmup"]

    def __init__(self, *, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self._syncing_count = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        note = QLabel(
            "Configure all camera runtime behavior here. "
            "Each row defines one named camera and its source index or path."
        )
        note.setWordWrap(True)
        note.setObjectName("MutedLabel")
        layout.addWidget(note)

        defaults_wrap = QWidget()
        defaults_layout = QVBoxLayout(defaults_wrap)
        defaults_layout.setContentsMargins(0, 0, 0, 0)
        defaults_layout.setSpacing(10)

        defaults_form = _InputGrid(defaults_layout)

        self.default_fps_input = QSpinBox()
        self.default_fps_input.setRange(1, 240)
        self.default_fps_input.setValue(int(self.config.get("camera_fps", 30) or 30))
        defaults_form.add_field("Default camera FPS", self.default_fps_input)

        self.default_warmup_input = QSpinBox()
        self.default_warmup_input.setRange(0, 120)
        self.default_warmup_input.setValue(int(self.config.get("camera_warmup_s", 5) or 5))
        defaults_form.add_field("Default camera warmup (s)", self.default_warmup_input)

        self.rename_flag_input = QLineEdit(str(self.config.get("camera_rename_flag", "rename_map")).strip() or "rename_map")
        defaults_form.add_field("Deploy rename-map flag", self.rename_flag_input)

        self.policy_map_input = QLineEdit(str(self.config.get("camera_policy_feature_map_json", "")).strip())
        self.policy_map_input.setPlaceholderText('optional: {"wrist":"camera1","overhead":"camera2"}')
        defaults_form.add_field("Policy feature map", self.policy_map_input)

        defaults_note = QLabel(
            "Default FPS and warmup are used as the baseline for new rows and as fallbacks at runtime. "
            "Deploy preflight uses the runtime camera names below and will suggest a rename map if a model expects different names."
        )
        defaults_note.setWordWrap(True)
        defaults_note.setObjectName("MutedLabel")
        defaults_layout.addWidget(defaults_note)
        layout.addWidget(defaults_wrap)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)
        controls.addWidget(QLabel("Camera count"))

        self.count_input = QSpinBox()
        self.count_input.setRange(1, 16)
        self.count_input.valueChanged.connect(self._handle_count_changed)
        controls.addWidget(self.count_input)

        add_button = QPushButton("Add Camera")
        add_button.clicked.connect(self.add_camera)
        controls.addWidget(add_button)

        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self.remove_selected_camera)
        controls.addWidget(remove_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.table = QTableWidget(0, len(self._HEADERS))
        self.table.setHorizontalHeaderLabels(self._HEADERS)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)

        self.summary_label = QLabel("")
        self.summary_label.setObjectName("MutedLabel")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.set_entries(camera_schema_entries_for_editor(config))

    def _default_row(self, index: int) -> dict[str, Any]:
        width = int(self.config.get("camera_default_width", 640) or 640)
        height = int(self.config.get("camera_default_height", 480) or 480)
        fps = int(self.default_fps_input.value())
        warmup_s = int(self.default_warmup_input.value())
        return {
            "name": f"camera{index}",
            "source": index - 1,
            "camera_type": "opencv",
            "width": width,
            "height": height,
            "fps": fps,
            "warmup_s": warmup_s,
        }

    def _set_row_payload(self, row: int, payload: dict[str, Any]) -> None:
        values = [
            str(payload.get("name", "")),
            str(payload.get("source", "")),
            str(payload.get("camera_type", payload.get("type", "opencv"))),
            str(payload.get("width", "")),
            str(payload.get("height", "")),
            str(payload.get("fps", "")),
            str(payload.get("warmup_s", "")),
        ]
        for column, value in enumerate(values):
            self.table.setItem(row, column, QTableWidgetItem(value))

    def _sync_summary(self) -> None:
        entries = self.entries()
        summary = ", ".join(f"{entry['name']}={entry['source']}" for entry in entries)
        self.summary_label.setText(
            f"Runtime camera names: {summary or '(none)'}\n"
            "Deploy preflight compares these names against trained model camera keys and will suggest a rename map when needed."
        )

    def _sync_count(self) -> None:
        self._syncing_count = True
        try:
            self.count_input.setValue(max(1, self.table.rowCount()))
        finally:
            self._syncing_count = False
        self._sync_summary()

    def _handle_count_changed(self, value: int) -> None:
        if self._syncing_count:
            return
        current = self.table.rowCount()
        if value > current:
            for index in range(current + 1, value + 1):
                self.add_camera(payload=self._default_row(index), sync_count=False)
        elif value < current:
            while self.table.rowCount() > value:
                self.table.removeRow(self.table.rowCount() - 1)
        self._sync_count()

    def set_entries(self, entries: list[Any]) -> None:
        self.table.setRowCount(0)
        for row_index, entry in enumerate(entries):
            payload = {
                "name": getattr(entry, "name", None) if not isinstance(entry, dict) else entry.get("name"),
                "source": getattr(entry, "source", None) if not isinstance(entry, dict) else entry.get("source"),
                "camera_type": getattr(entry, "camera_type", None) if not isinstance(entry, dict) else entry.get("camera_type"),
                "width": getattr(entry, "width", None) if not isinstance(entry, dict) else entry.get("width"),
                "height": getattr(entry, "height", None) if not isinstance(entry, dict) else entry.get("height"),
                "fps": getattr(entry, "fps", None) if not isinstance(entry, dict) else entry.get("fps"),
                "warmup_s": getattr(entry, "warmup_s", None) if not isinstance(entry, dict) else entry.get("warmup_s"),
            }
            self.table.insertRow(row_index)
            self._set_row_payload(row_index, payload)
        if self.table.rowCount() == 0:
            self.add_camera(payload=self._default_row(1), sync_count=False)
        self._sync_count()

    def entries(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in range(self.table.rowCount()):
            def _text(column: int, default: str = "") -> str:
                item = self.table.item(row, column)
                return item.text().strip() if item is not None else default

            rows.append(
                {
                    "name": _text(0, f"camera{row + 1}"),
                    "source": _text(1, str(row)),
                    "camera_type": _text(2, "opencv") or "opencv",
                    "width": _text(3, str(self.config.get("camera_default_width", 640))),
                    "height": _text(4, str(self.config.get("camera_default_height", 480))),
                    "fps": _text(5, str(self.default_fps_input.value())),
                    "warmup_s": _text(6, str(self.default_warmup_input.value())),
                }
            )
        return rows

    def add_camera(self, *, payload: dict[str, Any] | None = None, sync_count: bool = True) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self._set_row_payload(row, payload or self._default_row(row + 1))
        if sync_count:
            self._sync_count()

    def remove_selected_camera(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            row = self.table.rowCount() - 1
        if self.table.rowCount() <= 1 or row < 0:
            return
        self.table.removeRow(row)
        self._sync_count()

    def apply_to_config(self, config: dict[str, Any]) -> dict[str, Any]:
        updated = dict(config)
        updated["camera_fps"] = int(self.default_fps_input.value())
        updated["camera_warmup_s"] = int(self.default_warmup_input.value())
        updated["camera_rename_flag"] = self.rename_flag_input.text().strip() or "rename_map"
        updated["camera_policy_feature_map_json"] = self.policy_map_input.text().strip()
        return apply_camera_schema_entries_to_config(updated, self.entries())

    def reload_from_config(self, config: dict[str, Any]) -> None:
        self.config = config
        self.default_fps_input.setValue(int(config.get("camera_fps", 30) or 30))
        self.default_warmup_input.setValue(int(config.get("camera_warmup_s", 5) or 5))
        self.rename_flag_input.setText(str(config.get("camera_rename_flag", "rename_map")).strip() or "rename_map")
        self.policy_map_input.setText(str(config.get("camera_policy_feature_map_json", "")).strip())
        self.set_entries(camera_schema_entries_for_editor(config))


class _PageWithOutput(QWidget):
    def __init__(self, *, title: str, subtitle: str, append_log: Callable[[str], None]) -> None:
        super().__init__()
        self._append_log = append_log

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(18)

        hero, hero_layout = _build_card(title)
        hero.setObjectName("SectionHero")
        title_label = QLabel(title)
        title_label.setObjectName("PageTitle")
        hero_layout.addWidget(title_label)

        subtitle_label = QLabel(subtitle)
        subtitle_label.setWordWrap(True)
        subtitle_label.setObjectName("MutedLabel")
        hero_layout.addWidget(subtitle_label)
        layout.addWidget(hero)

        self.content_layout = QVBoxLayout()
        self.content_layout.setSpacing(18)
        layout.addLayout(self.content_layout)

        output_card, output_layout = _build_card("Output")
        self.status_label = QLabel("Ready.")
        self.status_label.setObjectName("StatusChip")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setMaximumWidth(280)
        output_layout.addWidget(self.status_label)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMinimumHeight(220)
        self.output.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        output_layout.addWidget(self.output)
        layout.addWidget(output_card, 1)

    def _set_output(self, *, title: str, text: str, log_message: str | None = None) -> None:
        self.status_label.setText(title)
        self.output.setPlainText(text)
        if log_message:
            self._append_log(log_message)

    def _append_output_line(self, line: str) -> None:
        self.output.appendPlainText(str(line))

    def _append_output_and_log(self, line: str) -> None:
        self._append_output_line(line)
        self._append_log(line)


class QtConfigPage(_PageWithOutput):
    _GROUPS = (
        ("Paths", ["lerobot_dir", "lerobot_venv_dir", "runs_dir", "record_data_dir", "deploy_data_dir", "trained_models_dir"]),
        ("Ports + IDs", ["follower_port", "leader_port", "follower_robot_id", "leader_robot_id"]),
        ("Robot Defaults", ["follower_robot_type", "leader_robot_type", "follower_robot_action_dim"]),
        ("Deploy Defaults", ["record_target_hz", "deploy_target_hz", "eval_num_episodes", "eval_duration_s", "eval_task"]),
        ("Calibration + Hub", ["follower_calibration_path", "leader_calibration_path", "hf_username", "ui_theme_mode"]),
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
                if key == "ui_theme_mode":
                    widget = QComboBox()
                    widget.addItems(["dark", "light"])
                    widget.setCurrentText(current or "dark")
                elif field["type"] == "int":
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
            "setup_status": build_setup_status_summary(status),
        }
        self._set_output(title="Config Snapshot", text=_json_text(payload), log_message="Config snapshot refreshed.")

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


class QtTrainingPage(_PageWithOutput):
    def __init__(self, *, config: dict[str, Any], append_log: Callable[[str], None]) -> None:
        super().__init__(
            title="Training",
            subtitle="Generate editable LeRobot commands and short HIL adaptation workflows from saved defaults.",
            append_log=append_log,
        )
        self.config = config
        self._last_command = ""

        card, card_layout = _build_card("HIL Command Builder")
        form = _InputGrid(card_layout)

        default_name = _default_output_name(config)
        default_output_dir = f"outputs/train/{default_name}"

        self.python_input = QLineEdit(str(config.get("training_gen_python_bin", DEFAULT_PYTHON_BIN)).strip() or DEFAULT_PYTHON_BIN)
        form.add_field("Python binary", self.python_input)

        self.policy_path_input = QLineEdit(str(config.get("training_gen_policy_path", DEFAULT_POLICY_PATH)).strip() or DEFAULT_POLICY_PATH)
        form.add_field("Policy path", self.policy_path_input)

        self.dataset_input = QLineEdit(str(config.get("training_gen_dataset_repo_id", _default_dataset_repo_id(config))).strip())
        form.add_field("Dataset repo id", self.dataset_input)

        self.output_dir_input = QLineEdit(str(config.get("training_gen_output_dir", default_output_dir)).strip() or default_output_dir)
        form.add_field("Output dir", self.output_dir_input)

        self.job_name_input = QLineEdit(str(config.get("training_gen_job_name", default_name)).strip() or default_name)
        form.add_field("Job name", self.job_name_input)

        self.device_input = QLineEdit(str(config.get("training_gen_device", "cuda")).strip() or "cuda")
        form.add_field("Device", self.device_input)

        self.batch_input = QSpinBox()
        self.batch_input.setRange(1, 1_000_000)
        self.batch_input.setValue(int(config.get("training_gen_batch_size", DEFAULT_BATCH_SIZE) or DEFAULT_BATCH_SIZE))
        form.add_field("Batch size", self.batch_input)

        self.steps_input = QSpinBox()
        self.steps_input.setRange(1, 100_000_000)
        self.steps_input.setValue(int(config.get("training_gen_steps", DEFAULT_STEPS) or DEFAULT_STEPS))
        form.add_field("Steps", self.steps_input)

        self.save_freq_input = QSpinBox()
        self.save_freq_input.setRange(1, 100_000_000)
        self.save_freq_input.setValue(int(config.get("training_gen_save_freq", DEFAULT_SAVE_FREQ) or DEFAULT_SAVE_FREQ))
        form.add_field("Save freq", self.save_freq_input)

        self.extra_args_input = QLineEdit(str(config.get("training_gen_extra_args", "")))
        form.add_field("Extra args", self.extra_args_input)

        self.use_srun_checkbox = QCheckBox("Wrap with srun")
        self.use_srun_checkbox.setChecked(bool(config.get("training_gen_use_srun", True)))
        card_layout.addWidget(self.use_srun_checkbox)

        self.srun_partition_input = QLineEdit(str(config.get("training_gen_srun_partition", DEFAULT_SRUN_PARTITION)).strip() or DEFAULT_SRUN_PARTITION)
        form.add_field("srun partition", self.srun_partition_input)

        self.srun_queue_input = QLineEdit(str(config.get("training_gen_srun_queue", DEFAULT_SRUN_QUEUE)).strip() or DEFAULT_SRUN_QUEUE)
        form.add_field("srun queue", self.srun_queue_input)

        self.srun_gres_input = QLineEdit(str(config.get("training_gen_srun_gres", DEFAULT_SRUN_GRES)).strip() or DEFAULT_SRUN_GRES)
        form.add_field("srun gres", self.srun_gres_input)

        self.srun_job_input = QLineEdit(str(config.get("training_gen_srun_job_name", default_name)).strip() or default_name)
        form.add_field("srun job name", self.srun_job_input)

        self.srun_cpus_input = QSpinBox()
        self.srun_cpus_input.setRange(1, 4096)
        self.srun_cpus_input.setValue(int(config.get("training_gen_srun_cpus_per_task", DEFAULT_SRUN_CPUS_PER_TASK) or DEFAULT_SRUN_CPUS_PER_TASK))
        form.add_field("srun cpus/task", self.srun_cpus_input)

        self.srun_extra_input = QLineEdit(str(config.get("training_gen_srun_extra_args", "")))
        form.add_field("srun extra args", self.srun_extra_input)

        self.project_root_input = QLineEdit(str(config.get("training_gen_project_root", "")).strip() or DEFAULT_PROJECT_ROOT)
        form.add_field("Project root", self.project_root_input)

        self.env_activate_input = QLineEdit(str(config.get("training_gen_env_activate_cmd", "")).strip() or DEFAULT_ENV_ACTIVATE)
        form.add_field("Env activate cmd", self.env_activate_input)

        self.hil_repo_input = QLineEdit(str(config.get("training_gen_hil_intervention_repo_id", "")))
        form.add_field("HIL intervention repo", self.hil_repo_input)

        self.hil_model_input = QLineEdit(str(config.get("training_gen_hil_base_model_path", "")))
        form.add_field("HIL base model path", self.hil_model_input)

        self.wandb_checkbox = QCheckBox("W&B enabled")
        self.wandb_checkbox.setChecked(bool(config.get("training_gen_wandb_enable", True)))
        card_layout.addWidget(self.wandb_checkbox)

        self.push_hub_checkbox = QCheckBox("Push to hub")
        self.push_hub_checkbox.setChecked(bool(config.get("training_gen_push_to_hub", False)))
        card_layout.addWidget(self.push_hub_checkbox)

        actions = QHBoxLayout()
        generate_button = QPushButton("Generate Command")
        generate_button.setObjectName("AccentButton")
        generate_button.clicked.connect(self.generate_command)
        actions.addWidget(generate_button)

        copy_button = QPushButton("Copy Command")
        copy_button.clicked.connect(self.copy_command)
        actions.addWidget(copy_button)

        edit_button = QPushButton("Edit Command")
        edit_button.clicked.connect(self.edit_command)
        actions.addWidget(edit_button)

        save_button = QPushButton("Save Defaults")
        save_button.clicked.connect(self.save_defaults)
        actions.addWidget(save_button)

        guide_button = QPushButton("Build HIL Guide")
        guide_button.clicked.connect(self.build_hil_guide)
        actions.addWidget(guide_button)
        actions.addStretch(1)
        card_layout.addLayout(actions)
        self.content_layout.addWidget(card)
        self._set_output(
            title="Training Ready",
            text="Generate a LeRobot training command or build a HIL workflow guide from the current form values.",
            log_message="Training page initialized.",
        )

    def _policy_input_features(self) -> str:
        return str(self.config.get("training_gen_policy_input_features", DEFAULT_POLICY_INPUT_FEATURES)).strip() or DEFAULT_POLICY_INPUT_FEATURES

    def _policy_output_features(self) -> str:
        return str(self.config.get("training_gen_policy_output_features", DEFAULT_POLICY_OUTPUT_FEATURES)).strip() or DEFAULT_POLICY_OUTPUT_FEATURES

    def _current_command(self) -> tuple[str | None, str | None]:
        return _build_generated_train_command(
            python_bin=self.python_input.text().strip() or DEFAULT_PYTHON_BIN,
            train_entrypoint=resolve_train_entrypoint(self.config),
            policy_path=self.policy_path_input.text().strip() or DEFAULT_POLICY_PATH,
            policy_input_features=self._policy_input_features(),
            policy_output_features=self._policy_output_features(),
            dataset_repo_id=self.dataset_input.text().strip(),
            output_dir=self.output_dir_input.text().strip(),
            job_name=self.job_name_input.text().strip(),
            device=self.device_input.text().strip() or "cuda",
            batch_size=int(self.batch_input.value()),
            steps=int(self.steps_input.value()),
            save_freq=int(self.save_freq_input.value()),
            wandb_enable=self.wandb_checkbox.isChecked(),
            push_to_hub=self.push_hub_checkbox.isChecked(),
            extra_args=self.extra_args_input.text().strip(),
            use_srun=self.use_srun_checkbox.isChecked(),
            srun_partition=self.srun_partition_input.text().strip() or DEFAULT_SRUN_PARTITION,
            srun_cpus_per_task=int(self.srun_cpus_input.value()),
            srun_gres=self.srun_gres_input.text().strip() or DEFAULT_SRUN_GRES,
            srun_job_name=self.srun_job_input.text().strip() or self.job_name_input.text().strip() or _default_output_name(self.config),
            srun_queue=self.srun_queue_input.text().strip() or DEFAULT_SRUN_QUEUE,
            srun_extra_args=self.srun_extra_input.text().strip(),
        )

    def _persist_defaults(self, *, quiet: bool = True) -> None:
        self.config["training_gen_python_bin"] = self.python_input.text().strip() or DEFAULT_PYTHON_BIN
        self.config["training_gen_policy_path"] = self.policy_path_input.text().strip() or DEFAULT_POLICY_PATH
        self.config["training_gen_dataset_repo_id"] = self.dataset_input.text().strip()
        self.config["training_gen_output_dir"] = self.output_dir_input.text().strip()
        self.config["training_gen_job_name"] = self.job_name_input.text().strip()
        self.config["training_gen_device"] = self.device_input.text().strip() or "cuda"
        self.config["training_gen_batch_size"] = int(self.batch_input.value())
        self.config["training_gen_steps"] = int(self.steps_input.value())
        self.config["training_gen_save_freq"] = int(self.save_freq_input.value())
        self.config["training_gen_extra_args"] = self.extra_args_input.text().strip()
        self.config["training_gen_use_srun"] = self.use_srun_checkbox.isChecked()
        self.config["training_gen_srun_partition"] = self.srun_partition_input.text().strip() or DEFAULT_SRUN_PARTITION
        self.config["training_gen_srun_queue"] = self.srun_queue_input.text().strip() or DEFAULT_SRUN_QUEUE
        self.config["training_gen_srun_gres"] = self.srun_gres_input.text().strip() or DEFAULT_SRUN_GRES
        self.config["training_gen_srun_job_name"] = self.srun_job_input.text().strip()
        self.config["training_gen_srun_cpus_per_task"] = int(self.srun_cpus_input.value())
        self.config["training_gen_srun_extra_args"] = self.srun_extra_input.text().strip()
        self.config["training_gen_project_root"] = self.project_root_input.text().strip() or DEFAULT_PROJECT_ROOT
        self.config["training_gen_env_activate_cmd"] = self.env_activate_input.text().strip() or DEFAULT_ENV_ACTIVATE
        self.config["training_gen_hil_intervention_repo_id"] = self.hil_repo_input.text().strip()
        self.config["training_gen_hil_base_model_path"] = self.hil_model_input.text().strip()
        self.config["training_gen_wandb_enable"] = self.wandb_checkbox.isChecked()
        self.config["training_gen_push_to_hub"] = self.push_hub_checkbox.isChecked()
        if self._last_command:
            self.config["training_generated_command"] = self._last_command
        save_config(self.config, quiet=quiet)

    def generate_command(self) -> None:
        command_text, error = self._current_command()
        if command_text is None:
            self._set_output(title="Validation Error", text=error or "Unable to build training command.", log_message="Training command generation failed.")
            return
        self._last_command = command_text
        expected_model_path = _expected_pretrained_model_path(
            self.project_root_input.text().strip() or DEFAULT_PROJECT_ROOT,
            self.output_dir_input.text().strip(),
        )
        self._set_output(
            title="Training Command",
            text=f"{command_text}\n\nExpected updated model path:\n{expected_model_path}",
            log_message="Training command generated.",
        )
        show_text_dialog(
            parent=self.window() if isinstance(self.window(), QWidget) else None,
            title="Training Command",
            text=f"{command_text}\n\nExpected updated model path:\n{expected_model_path}",
            copy_text=command_text,
            wrap_mode="word",
        )

    def edit_command(self) -> None:
        command_text, error = self._current_command()
        if command_text is None:
            self._set_output(title="Validation Error", text=error or "Unable to edit training command.", log_message="Training command edit failed.")
            return
        try:
            command_argv = shlex.split(command_text)
        except ValueError as exc:
            self._set_output(title="Validation Error", text=f"Training command could not be parsed for editing: {exc}", log_message="Training command edit parse failed.")
            return
        edited = ask_editable_command_dialog(
            parent=self.window() if isinstance(self.window(), QWidget) else None,
            title="Edit Training Command",
            command_argv=command_argv,
            intro_text=(
                "Review or edit the generated training command below.\n"
                "Use this to keep a hand-tuned command buffer without leaving the GUI."
            ),
            confirm_label="Use Command",
            cancel_label="Cancel",
        )
        if edited is None:
            return
        self._last_command = shlex.join(edited)
        self._set_output(title="Training Command", text=self._last_command, log_message="Training command edited.")

    def copy_command(self) -> None:
        if not self._last_command:
            self.generate_command()
        if not self._last_command:
            return
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self._last_command)
        self._set_output(title="Command Copied", text=self._last_command, log_message="Training command copied to clipboard.")

    def save_defaults(self) -> None:
        command_text, error = self._current_command()
        if command_text is None:
            self._set_output(title="Validation Error", text=error or "Unable to save training defaults.", log_message="Training defaults save failed.")
            return
        self._last_command = command_text
        self._persist_defaults()
        self._set_output(title="Defaults Saved", text=command_text, log_message="Training defaults saved.")

    def build_hil_guide(self) -> None:
        command_text, error = self._current_command()
        if command_text is None:
            self._set_output(title="Validation Error", text=error or "Unable to build HIL guide.", log_message="HIL guide generation failed.")
            return
        self._last_command = command_text
        expected_model_path = _expected_pretrained_model_path(
            self.project_root_input.text().strip() or DEFAULT_PROJECT_ROOT,
            self.output_dir_input.text().strip(),
        )
        guide = _build_hil_workflow_text(
            project_root=self.project_root_input.text().strip() or DEFAULT_PROJECT_ROOT,
            env_activate_cmd=self.env_activate_input.text().strip() or DEFAULT_ENV_ACTIVATE,
            intervention_repo_id=self.hil_repo_input.text().strip() or _with_hil_suffix(self.dataset_input.text().strip()),
            base_model_path=self.hil_model_input.text().strip() or self.policy_path_input.text().strip(),
            command=command_text,
            expected_model_path=expected_model_path,
        )
        self._set_output(title="HIL Guide", text=guide, log_message="HIL guide generated.")
        show_text_dialog(
            parent=self.window() if isinstance(self.window(), QWidget) else None,
            title="HIL Workflow Guide",
            text=guide,
            copy_text=guide,
            wrap_mode="word",
        )


class QtVisualizerPage(_PageWithOutput):
    def __init__(self, *, config: dict[str, Any], append_log: Callable[[str], None]) -> None:
        super().__init__(
            title="Visualizer",
            subtitle="Browse local deployment runs, datasets, models, and discovered video assets without the old Tk tree views.",
            append_log=append_log,
        )
        self.config = config
        self._sources: list[dict[str, Any]] = []
        self._videos: list[dict[str, Any]] = []
        self._video_target_text = ""
        self._video_player_error = ""

        controls_card, controls_layout = _build_card("Source Browser")
        top_row = QHBoxLayout()
        self.source_combo = QComboBox()
        self.source_combo.addItem("Deployments", "deployments")
        self.source_combo.addItem("Datasets", "datasets")
        self.source_combo.addItem("Models", "models")
        top_row.addWidget(QLabel("Source"))
        top_row.addWidget(self.source_combo)

        self.root_input = QLineEdit(str(config.get("deploy_data_dir", get_deploy_data_dir(config))))
        top_row.addWidget(QLabel("Root"))
        top_row.addWidget(self.root_input, 1)

        self.hf_owner_input = QLineEdit(str(config.get("hf_username", "")).strip())
        top_row.addWidget(QLabel("HF owner"))
        top_row.addWidget(self.hf_owner_input)

        browse_root_button = QPushButton("Browse Root")
        browse_root_button.clicked.connect(self.browse_root)
        top_row.addWidget(browse_root_button)

        refresh_button = QPushButton("Refresh")
        refresh_button.setObjectName("AccentButton")
        refresh_button.clicked.connect(self.refresh_sources)
        top_row.addWidget(refresh_button)
        controls_layout.addLayout(top_row)

        actions = QHBoxLayout()
        open_source_button = QPushButton("Open Source")
        open_source_button.clicked.connect(self.open_selected_source)
        actions.addWidget(open_source_button)

        open_video_button = QPushButton("Open Video")
        open_video_button.clicked.connect(self.open_selected_video)
        actions.addWidget(open_video_button)
        actions.addStretch(1)
        controls_layout.addLayout(actions)
        self.content_layout.addWidget(controls_card)

        sources_card, sources_layout = _build_card("Sources")
        self.source_table = QTableWidget(0, 2)
        _set_table_headers(self.source_table, ["Source", "Name"])
        _set_readonly_table(self.source_table)
        self.source_table.itemSelectionChanged.connect(self._on_source_selection_changed)
        sources_layout.addWidget(self.source_table)
        self.content_layout.addWidget(sources_card)

        details_card, details_layout = _build_card("Selection Details")
        self.meta_view = QPlainTextEdit()
        self.meta_view.setReadOnly(True)
        self.meta_view.setMinimumHeight(180)
        self.meta_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        details_layout.addWidget(self.meta_view)
        self.content_layout.addWidget(details_card)

        insights_card, insights_layout = _build_card("Deployment Insights")
        self.insights_table = QTableWidget(0, 4)
        _set_table_headers(self.insights_table, ["Episode", "Result", "Notes", "Tags"])
        _set_readonly_table(self.insights_table)
        insights_layout.addWidget(self.insights_table)
        self.content_layout.addWidget(insights_card)

        player_card, player_layout = _build_card("Video Preview")
        self.player_status = QLabel("Select a source to preview discovered videos.")
        self.player_status.setObjectName("MutedLabel")
        self.player_status.setWordWrap(True)
        player_layout.addWidget(self.player_status)

        if _QT_MULTIMEDIA_AVAILABLE:
            self.video_preview = QVideoWidget()
            self.video_preview.setMinimumHeight(260)
            player_layout.addWidget(self.video_preview)

            player_actions = QHBoxLayout()
            self.play_button = QPushButton("Play")
            self.play_button.clicked.connect(self.play_current_video)
            player_actions.addWidget(self.play_button)

            self.pause_button = QPushButton("Pause")
            self.pause_button.clicked.connect(self.pause_current_video)
            player_actions.addWidget(self.pause_button)

            self.stop_button = QPushButton("Stop")
            self.stop_button.clicked.connect(self.stop_current_video)
            player_actions.addWidget(self.stop_button)
            player_actions.addStretch(1)
            player_layout.addLayout(player_actions)

            self._audio_output = QAudioOutput(self)
            self._audio_output.setVolume(0.5)
            self._media_player = QMediaPlayer(self)
            self._media_player.setAudioOutput(self._audio_output)
            self._media_player.setVideoOutput(self.video_preview)
            self._media_player.errorOccurred.connect(self._on_video_error)
        else:
            self.video_preview = None
            self.play_button = None
            self.pause_button = None
            self.stop_button = None
            self._audio_output = None
            self._media_player = None
            fallback = QLabel("Embedded video preview is unavailable in this Qt install. Use Open Video for external playback.")
            fallback.setObjectName("MutedLabel")
            fallback.setWordWrap(True)
            player_layout.addWidget(fallback)
        self.content_layout.addWidget(player_card)

        videos_card, videos_layout = _build_card("Videos")
        self.video_table = QTableWidget(0, 2)
        _set_table_headers(self.video_table, ["Video", "Size"])
        _set_readonly_table(self.video_table)
        self.video_table.itemSelectionChanged.connect(self._on_video_selection_changed)
        videos_layout.addWidget(self.video_table)
        self.content_layout.addWidget(videos_card)

        self.source_combo.currentIndexChanged.connect(self._sync_root_placeholder)
        self._sync_root_placeholder()
        self.refresh_sources()

    def browse_root(self) -> None:
        current = self.root_input.text().strip() or str(Path.home())
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select Visualizer Root",
            current,
        )
        if not selected:
            return
        self.root_input.setText(selected)
        self.refresh_sources()

    def _active_source(self) -> str:
        return str(self.source_combo.currentData() or "deployments")

    def _sync_root_placeholder(self) -> None:
        source = self._active_source()
        if source == "deployments":
            self.root_input.setText(str(self.config.get("deploy_data_dir", get_deploy_data_dir(self.config))))
        elif source == "datasets":
            self.root_input.setText(str(self.config.get("record_data_dir", get_lerobot_dir(self.config) / "data")))
        else:
            self.root_input.setText(str(self.config.get("trained_models_dir", get_lerobot_dir(self.config) / "trained_models")))

    def _snapshot(self) -> _VisualizerRefreshSnapshot:
        source = self._active_source()
        root_text = self.root_input.text().strip()
        if source == "deployments":
            deploy_root = root_text or str(get_deploy_data_dir(self.config))
            dataset_root = str(self.config.get("record_data_dir", get_lerobot_dir(self.config) / "data"))
            model_root = str(self.config.get("trained_models_dir", get_lerobot_dir(self.config) / "trained_models"))
        elif source == "datasets":
            deploy_root = str(self.config.get("deploy_data_dir", get_deploy_data_dir(self.config)))
            dataset_root = root_text or str(self.config.get("record_data_dir", get_lerobot_dir(self.config) / "data"))
            model_root = str(self.config.get("trained_models_dir", get_lerobot_dir(self.config) / "trained_models"))
        else:
            deploy_root = str(self.config.get("deploy_data_dir", get_deploy_data_dir(self.config)))
            dataset_root = str(self.config.get("record_data_dir", get_lerobot_dir(self.config) / "data"))
            model_root = root_text or str(self.config.get("trained_models_dir", get_lerobot_dir(self.config) / "trained_models"))
        return _VisualizerRefreshSnapshot(
            source=source,
            deploy_root=deploy_root,
            dataset_root=dataset_root,
            model_root=model_root,
            hf_owner=self.hf_owner_input.text().strip(),
        )

    def refresh_sources(self) -> None:
        snapshot = self._snapshot()
        sources, error_text, source_kind = _collect_sources_for_refresh(self.config, snapshot)
        self._sources = list(sources)
        self.source_table.setRowCount(len(self._sources))
        for row, source in enumerate(self._sources):
            scope_text, name_text = _visualizer_source_row_values(source)
            self.source_table.setItem(row, 0, QTableWidgetItem(scope_text))
            self.source_table.setItem(row, 1, QTableWidgetItem(name_text))
        self.video_table.setRowCount(0)
        self.insights_table.setRowCount(0)
        self.meta_view.setPlainText("")
        self._videos = []
        self._clear_video_preview("Select a source to preview discovered videos.")
        if self._sources:
            self.source_table.selectRow(0)
            self._set_output(
                title="Sources Loaded",
                text=f"Loaded {len(self._sources)} {source_kind}.",
                log_message=f"Visualizer refreshed {len(self._sources)} {source_kind}.",
            )
        else:
            detail = error_text or f"No {source_kind} were found."
            self._set_output(title="No Sources", text=detail, log_message="Visualizer refresh returned no sources.")

    def _current_source(self) -> dict[str, Any] | None:
        row = self.source_table.currentRow()
        if row < 0 or row >= len(self._sources):
            return None
        return self._sources[row]

    def _build_selection_payload(self, source: dict[str, Any]) -> dict[str, Any]:
        metadata = source.get("metadata", {}) if isinstance(source.get("metadata"), dict) else {}
        resolved_metadata: dict[str, Any] = dict(metadata)
        metadata_error: str | None = None

        if str(source.get("scope", "local")) == "huggingface":
            repo_id = str(source.get("repo_id", "")).strip()
            if str(source.get("kind", "")) == "dataset":
                resolved, metadata_error = get_hf_dataset_info(repo_id)
            else:
                resolved, metadata_error = get_hf_model_info(repo_id)
            resolved_metadata = resolved or {}

        source_path_raw = source.get("path")
        source_path = Path(source_path_raw) if source_path_raw else None
        scope = str(source.get("scope", "local")).strip() or "local"
        kind = str(source.get("kind", "source")).strip() or "source"
        repo_id = str(source.get("repo_id", "")).strip()

        meta_payload: dict[str, Any] = {
            "scope": scope,
            "kind": kind,
            "name": source.get("name"),
            "path": str(source_path) if source_path is not None else None,
            "repo_id": repo_id or None,
            "run_path": str(source.get("run_path")) if source.get("run_path") else None,
            "data_path": str(source.get("data_path")) if source.get("data_path") else None,
        }
        if scope == "local" and source_path is not None:
            meta_payload["local_overview"] = _local_path_overview(source_path)
        if metadata_error:
            meta_payload["metadata_error"] = metadata_error
        if resolved_metadata:
            meta_payload["metadata"] = resolved_metadata

        insights_visible, insights_header, insights_rows = _visualizer_insights_section(
            kind,
            resolved_metadata if isinstance(resolved_metadata, dict) else {},
        )
        videos = _collect_videos_for_source(source, resolved_metadata if isinstance(resolved_metadata, dict) else None)
        return {
            "meta_payload": meta_payload,
            "insights_visible": insights_visible,
            "insights_header": insights_header,
            "insights_rows": insights_rows,
            "videos": videos,
        }

    def _on_source_selection_changed(self) -> None:
        source = self._current_source()
        if source is None:
            self.meta_view.setPlainText("")
            self.insights_table.setRowCount(0)
            self.video_table.setRowCount(0)
            self._videos = []
            self._clear_video_preview("Select a source to preview discovered videos.")
            return
        payload = self._build_selection_payload(source)
        text = _json_text(payload.get("meta_payload", {}))
        self.meta_view.setPlainText(text)
        self.insights_table.setRowCount(0)
        if payload.get("insights_visible"):
            self.insights_table.setRowCount(len(payload.get("insights_rows", [])))
            for row_index, row in enumerate(payload.get("insights_rows", [])):
                if not isinstance(row, tuple) or len(row) != 4:
                    continue
                for col_index, value in enumerate(row):
                    self.insights_table.setItem(row_index, col_index, QTableWidgetItem(str(value)))
        if payload.get("insights_visible"):
            text += "\n\n" + str(payload.get("insights_header", "Deployment Insights"))
            for row in payload.get("insights_rows", []):
                if isinstance(row, tuple) and len(row) == 4:
                    text += f"\n- Episode {row[0]} | {row[1]} | {row[2]} | {row[3]}"
        self._set_output(title="Selection Details", text=text, log_message=None)
        self._videos = list(payload.get("videos", []))
        self.video_table.setRowCount(len(self._videos))
        for row, item in enumerate(self._videos):
            self.video_table.setItem(row, 0, QTableWidgetItem(str(item.get("relative_path", "-"))))
            self.video_table.setItem(row, 1, QTableWidgetItem(str(item.get("size_text", "-"))))
        if self._videos:
            self.video_table.selectRow(0)
            self._sync_video_preview(auto_play=True)
        else:
            self._clear_video_preview("No videos found for the selected source.")

    def _current_video(self) -> dict[str, Any] | None:
        row = self.video_table.currentRow()
        if row < 0 or row >= len(self._videos):
            return None
        return self._videos[row]

    def _video_url_for_item(self, item: dict[str, Any]) -> QUrl | None:
        raw_path = item.get("path")
        if raw_path:
            path = Path(str(raw_path))
            if path.exists():
                return QUrl.fromLocalFile(str(path))
        raw_url = str(item.get("url", "")).strip()
        if raw_url:
            return QUrl(raw_url)
        return None

    def _clear_video_preview(self, message: str) -> None:
        self._video_target_text = ""
        self._video_player_error = ""
        self.player_status.setText(message)
        if self._media_player is not None:
            self._media_player.stop()
            self._media_player.setSource(QUrl())

    def _sync_video_preview(self, *, auto_play: bool) -> None:
        item = self._current_video()
        if item is None:
            self._clear_video_preview("No video selected.")
            return

        target_text = str(item.get("relative_path") or item.get("path") or item.get("url") or "").strip()
        self._video_target_text = target_text
        self._video_player_error = ""
        self.player_status.setText(f"Previewing {target_text}" if target_text else "Previewing selected video")
        if self._media_player is None:
            return
        source_url = self._video_url_for_item(item)
        if source_url is None or source_url.isEmpty():
            self._clear_video_preview("Selected video could not be resolved for preview.")
            return
        self._media_player.stop()
        self._media_player.setSource(source_url)
        if auto_play:
            self._media_player.play()

    def _on_video_selection_changed(self) -> None:
        if not self._videos:
            self._clear_video_preview("No videos found for the selected source.")
            return
        self._sync_video_preview(auto_play=True)

    def _on_video_error(self, _error: object, error_text: str) -> None:
        self._video_player_error = str(error_text or "").strip()
        if self._video_target_text:
            self.player_status.setText(f"Preview unavailable for {self._video_target_text}: {self._video_player_error or 'unknown media error'}")
        else:
            self.player_status.setText(f"Preview unavailable: {self._video_player_error or 'unknown media error'}")

    def play_current_video(self) -> None:
        if self._media_player is None:
            return
        if self._media_player.source().isEmpty():
            self._sync_video_preview(auto_play=True)
            return
        self._media_player.play()

    def pause_current_video(self) -> None:
        if self._media_player is not None:
            self._media_player.pause()

    def stop_current_video(self) -> None:
        if self._media_player is not None:
            self._media_player.stop()

    def open_selected_source(self) -> None:
        source = self._current_source()
        if source is None:
            self._set_output(title="No Selection", text="Select a source first.", log_message="Visualizer source open skipped with no selection.")
            return
        target = source.get("path")
        if not target and source.get("repo_id"):
            repo_id = str(source.get("repo_id"))
            if str(source.get("kind")) == "dataset":
                target = f"https://huggingface.co/datasets/{repo_id}"
            else:
                target = f"https://huggingface.co/{repo_id}"
        ok, message = _open_path(target or "")
        self._set_output(title="Open Source" if ok else "Open Failed", text=str(target or message), log_message="Visualizer opened source." if ok else "Visualizer source open failed.")

    def open_selected_video(self) -> None:
        row = self.video_table.currentRow()
        if row < 0 or row >= len(self._videos):
            self._set_output(title="No Video", text="Select a video first.", log_message="Visualizer video open skipped with no selection.")
            return
        target = self._videos[row].get("path") or self._videos[row].get("url")
        ok, message = _open_path(target or "")
        self._set_output(title="Open Video" if ok else "Open Failed", text=str(target or message), log_message="Visualizer opened video." if ok else "Visualizer video open failed.")


class QtHistoryPage(_PageWithOutput):
    def __init__(
        self,
        *,
        config: dict[str, Any],
        append_log: Callable[[str], None],
        run_controller: ManagedRunController,
    ) -> None:
        super().__init__(
            title="History",
            subtitle="Browse run artifacts, open logs, and rerun prior commands without the old Tk table widgets.",
            append_log=append_log,
        )
        self.config = config
        self._run_controller = run_controller
        self._rows: list[dict[str, Any]] = []
        self._action_buttons: list[QPushButton] = []

        filters_card, filters_layout = _build_card("Filters")
        row = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("All modes", "all")
        for mode in HISTORY_MODE_VALUES:
            self.mode_combo.addItem(mode, mode)
        row.addWidget(self.mode_combo)

        self.status_combo = QComboBox()
        self.status_combo.addItem("All statuses", "all")
        self.status_combo.addItem("Success", "success")
        self.status_combo.addItem("Failed", "failed")
        self.status_combo.addItem("Canceled", "canceled")
        row.addWidget(self.status_combo)

        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Filter by hint, command text, or timestamp")
        row.addWidget(self.query_input, 1)

        refresh_button = QPushButton("Refresh")
        refresh_button.setObjectName("AccentButton")
        refresh_button.clicked.connect(self.refresh_history)
        row.addWidget(refresh_button)
        self._action_buttons.append(refresh_button)
        filters_layout.addLayout(row)

        actions = QHBoxLayout()
        self.open_run_button = QPushButton("Open Run Folder")
        self.open_run_button.clicked.connect(self.open_run_folder)
        actions.addWidget(self.open_run_button)
        self._action_buttons.append(self.open_run_button)

        self.open_log_button = QPushButton("Open Command Log")
        self.open_log_button.clicked.connect(self.open_command_log)
        actions.addWidget(self.open_log_button)
        self._action_buttons.append(self.open_log_button)

        self.rerun_button = QPushButton("Rerun Selected")
        self.rerun_button.clicked.connect(self.rerun_selected)
        actions.addWidget(self.rerun_button)
        self._action_buttons.append(self.rerun_button)

        actions.addStretch(1)
        filters_layout.addLayout(actions)
        self.content_layout.addWidget(filters_card)

        runs_card, runs_layout = _build_card("Runs")
        self.run_table = QTableWidget(0, 6)
        _set_table_headers(self.run_table, ["Started", "Duration", "Mode", "Status", "Hint", "Command"])
        _set_readonly_table(self.run_table)
        self.run_table.itemSelectionChanged.connect(self._on_selection_changed)
        runs_layout.addWidget(self.run_table)
        self.content_layout.addWidget(runs_card)

        deploy_card, deploy_layout = _build_card("Deploy Outcome + Notes Editor")
        self.deploy_editor_card = deploy_card
        deploy_form = _InputGrid(deploy_layout)

        self.episode_combo = QComboBox()
        self.episode_combo.currentIndexChanged.connect(self._sync_episode_editor_fields)
        deploy_form.add_field("Episode", self.episode_combo)

        self.outcome_combo = QComboBox()
        self.outcome_combo.addItems(["success", "failed", "unmarked"])
        deploy_form.add_field("Status", self.outcome_combo)

        self.tags_input = QLineEdit()
        self.tags_input.setPlaceholderText("comma,separated,tags")
        deploy_form.add_field("Tags", self.tags_input)

        self.episode_note_input = QLineEdit()
        self.episode_note_input.setPlaceholderText("Optional note for this episode")
        deploy_form.add_field("Episode note", self.episode_note_input)

        notes_label = QLabel("Deployment overall notes")
        notes_label.setObjectName("FormLabel")
        deploy_layout.addWidget(notes_label)

        self.overall_notes = QPlainTextEdit()
        self.overall_notes.setMinimumHeight(110)
        deploy_layout.addWidget(self.overall_notes)

        editor_actions = QHBoxLayout()
        self.save_episode_button = QPushButton("Save Episode Edit")
        self.save_episode_button.clicked.connect(self.save_episode_edit)
        editor_actions.addWidget(self.save_episode_button)
        self._action_buttons.append(self.save_episode_button)

        self.save_notes_button = QPushButton("Save Deployment Notes")
        self.save_notes_button.clicked.connect(self.save_deployment_notes)
        editor_actions.addWidget(self.save_notes_button)
        self._action_buttons.append(self.save_notes_button)

        self.open_notes_button = QPushButton("Open notes.md")
        self.open_notes_button.clicked.connect(self.open_deploy_notes_file)
        editor_actions.addWidget(self.open_notes_button)
        self._action_buttons.append(self.open_notes_button)
        editor_actions.addStretch(1)
        deploy_layout.addLayout(editor_actions)

        self.deploy_editor_status = QLabel("")
        self.deploy_editor_status.setObjectName("MutedLabel")
        self.deploy_editor_status.setWordWrap(True)
        deploy_layout.addWidget(self.deploy_editor_status)
        self.content_layout.addWidget(deploy_card)
        self.deploy_editor_card.hide()

        self.mode_combo.currentIndexChanged.connect(self.refresh_history)
        self.status_combo.currentIndexChanged.connect(self.refresh_history)
        self.query_input.textChanged.connect(self.refresh_history)
        self.refresh_history()

    def _current_row(self) -> dict[str, Any] | None:
        row = self.run_table.currentRow()
        if row < 0 or row >= len(self._rows):
            return None
        return self._rows[row]

    def _set_running(self, active: bool, status_text: str | None = None, is_error: bool = False) -> None:
        for button in self._action_buttons:
            button.setEnabled(not active)
        if active:
            self.status_label.setText(status_text or "Running rerun...")
        elif is_error:
            self.status_label.setText(status_text or "Rerun failed.")
        else:
            self.status_label.setText(status_text or "Ready.")

    def refresh_history(self) -> None:
        runs, warning_count = list_runs(config=self.config, limit=5000)
        payload = _build_history_refresh_payload_from_runs(
            runs=runs,
            warning_count=warning_count,
            mode_filter=str(self.mode_combo.currentData() or "all"),
            status_filter=str(self.status_combo.currentData() or "all"),
            query=self.query_input.text().strip().lower(),
        )
        self._rows = list(payload.get("rows", []))
        self.run_table.setRowCount(len(self._rows))
        for row_index, row in enumerate(self._rows):
            values = list(row.get("values", ()))
            while len(values) < 6:
                values.append("")
            for col_index, value in enumerate(values[:6]):
                self.run_table.setItem(row_index, col_index, QTableWidgetItem(str(value)))
        if self._rows:
            self.run_table.selectRow(0)
        stats = payload.get("stats", {})
        self._set_output(
            title="History Refreshed",
            text=_json_text({"stats": stats, "warning_count": warning_count}),
            log_message=f"History refreshed {stats.get('total', 0)} rows.",
        )

    def _on_selection_changed(self) -> None:
        row = self._current_row()
        if row is None:
            self.deploy_editor_card.hide()
            return
        item = row.get("item", {})
        self._set_output(
            title="Run Details",
            text=_json_text(item),
            log_message=None,
        )
        self._populate_deploy_editor()

    def open_run_folder(self) -> None:
        row = self._current_row()
        if row is None:
            self._set_output(title="No Selection", text="Select a run first.", log_message="History open-run skipped with no selection.")
            return
        run_path = Path(str(row.get("item", {}).get("_run_path", "")))
        ok, message = open_path_in_file_manager(run_path)
        self._set_output(title="Open Run" if ok else "Open Failed", text=str(run_path if ok else message), log_message="History opened run folder." if ok else "History failed to open run folder.")

    def open_command_log(self) -> None:
        row = self._current_row()
        if row is None:
            self._set_output(title="No Selection", text="Select a run first.", log_message="History open-log skipped with no selection.")
            return
        run_path = Path(str(row.get("item", {}).get("_run_path", "")))
        log_path = run_path / "command.log"
        ok, message = open_path_in_file_manager(log_path)
        self._set_output(title="Open Log" if ok else "Open Failed", text=str(log_path if ok else message), log_message="History opened command log." if ok else "History failed to open command log.")

    def rerun_selected(self) -> None:
        row = self._current_row()
        if row is None:
            self._set_output(title="No Selection", text="Select a run first.", log_message="History rerun skipped with no selection.")
            return
        item = row.get("item", {})
        cmd, error = _command_from_item(item)
        if cmd is None:
            self._set_output(title="Rerun Failed", text=error or "No command stored in this history entry.", log_message="History rerun failed while parsing command.")
            return

        run_mode = str(item.get("mode", "run")).strip().lower() or "run"
        cwd_raw = str(item.get("cwd", "")).strip()
        cwd = Path(cwd_raw) if cwd_raw else None
        artifact_context: dict[str, Any] = {}
        dataset_repo_id = str(item.get("dataset_repo_id", "")).strip()
        model_path = str(item.get("model_path", "")).strip()
        if dataset_repo_id:
            artifact_context["dataset_repo_id"] = dataset_repo_id
        if model_path:
            artifact_context["model_path"] = model_path

        rerun_cmd = list(cmd)
        if run_mode == "deploy":
            rerun_cmd, rerun_message = normalize_deploy_rerun_command(
                command_argv=rerun_cmd,
                username=str(self.config.get("hf_username", "")),
                local_roots=[get_deploy_data_dir(self.config), get_lerobot_dir(self.config) / "data"],
            )
            if rerun_message:
                self._append_output_and_log(rerun_message)
                for arg in rerun_cmd:
                    if str(arg).startswith("--dataset.repo_id="):
                        artifact_context["dataset_repo_id"] = str(arg).split("=", 1)[1].strip()
                        break

        self.output.setPlainText("Rerunning stored command...\n")
        self._append_output_line(" ".join(rerun_cmd))
        hooks = RunUiHooks(
            set_running=self._set_running,
            append_output_line=self._append_output_line,
        )
        ok, message = self._run_controller.run_process_async(
            cmd=rerun_cmd,
            cwd=cwd,
            hooks=hooks,
            complete_callback=None,
            run_mode=run_mode,
            preflight_checks=None,
            artifact_context=artifact_context,
        )
        if not ok:
            self._set_output(title="Rerun Rejected", text=message or "Unable to rerun selected command.", log_message="History rerun was rejected.")
            return
        self._append_log(f"History rerun started for {run_mode}.")

    def _read_selected_metadata(self) -> tuple[dict[str, Any] | None, Path | None, dict[str, Any] | None]:
        row = self._current_row()
        if row is None:
            return None, None, None
        item = row.get("item", {})
        metadata_path_raw = str(item.get("_metadata_path", "")).strip()
        if not metadata_path_raw:
            return item, None, None
        metadata_path = Path(metadata_path_raw)
        if not metadata_path.exists():
            return item, metadata_path, None
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return item, metadata_path, None
        if not isinstance(data, dict):
            return item, metadata_path, None
        data["_metadata_path"] = str(metadata_path)
        data["_run_path"] = str(metadata_path.parent)
        return item, metadata_path, data

    def _deploy_episode_map(self, item: dict[str, Any]) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
        summary = _normalize_deploy_episode_outcomes(item.get("deploy_episode_outcomes"))
        episode_map: dict[int, dict[str, Any]] = {}
        entries = summary.get("episode_outcomes")
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                try:
                    episode_idx = int(entry.get("episode"))
                except (TypeError, ValueError):
                    continue
                if episode_idx <= 0:
                    continue
                episode_map[episode_idx] = dict(entry)
        return summary, episode_map

    def _episode_choices(self, summary: dict[str, Any], episode_map: dict[int, dict[str, Any]]) -> list[str]:
        total = summary.get("total_episodes")
        if isinstance(total, int) and total > 0:
            return [str(value) for value in range(1, total + 1)]
        if episode_map:
            return [str(value) for value in sorted(episode_map)]
        return ["1"]

    def _populate_deploy_editor(self) -> None:
        row = self._current_row()
        if row is None:
            self.deploy_editor_card.hide()
            return
        item = row.get("item", {})
        if str(item.get("mode", "")).strip().lower() != "deploy":
            self.deploy_editor_card.hide()
            return
        self.deploy_editor_card.show()
        summary, episode_map = self._deploy_episode_map(item)
        choices = self._episode_choices(summary, episode_map)
        self.episode_combo.blockSignals(True)
        self.episode_combo.clear()
        self.episode_combo.addItems(choices)
        self.episode_combo.setCurrentText(choices[0])
        self.episode_combo.blockSignals(False)
        self.overall_notes.setPlainText(str(item.get("deploy_notes_summary", "")).strip())
        self.deploy_editor_status.setText("Edit episode status/tags/note and deployment notes, then save.")
        self._sync_episode_editor_fields()

    def _sync_episode_editor_fields(self) -> None:
        row = self._current_row()
        if row is None:
            return
        item = row.get("item", {})
        summary, episode_map = self._deploy_episode_map(item)
        try:
            current_episode = int(self.episode_combo.currentText().strip() or "1")
        except ValueError:
            current_episode = 1
        if not episode_map and not summary:
            current_entry: dict[str, Any] = {}
        else:
            current_entry = episode_map.get(current_episode, {})
        self.outcome_combo.setCurrentText(normalize_deploy_result(current_entry.get("result", "unmarked")))
        tags = current_entry.get("tags") if isinstance(current_entry.get("tags"), list) else []
        self.tags_input.setText(", ".join(str(tag) for tag in tags))
        self.episode_note_input.setText(str(current_entry.get("note", "")).strip())

    def _persist_deploy_metadata(self, updated_data: dict[str, Any]) -> tuple[bool, str]:
        metadata_path_raw = str(updated_data.get("_metadata_path", "")).strip()
        run_path_raw = str(updated_data.get("_run_path", "")).strip()
        if not metadata_path_raw or not run_path_raw:
            return False, "Selected run is missing metadata/run path references."
        metadata_path = Path(metadata_path_raw)
        run_path = Path(run_path_raw)
        payload = dict(updated_data)
        payload.pop("_metadata_path", None)
        payload.pop("_run_path", None)
        try:
            _atomic_write(json.dumps(payload, indent=2) + "\n", metadata_path)
        except OSError as exc:
            return False, f"Failed to write metadata.json: {exc}"
        notes_path = write_deploy_notes_file(run_path, payload, filename="notes.md")
        if notes_path is None:
            return False, "Saved metadata, but failed to write notes.md"
        csv_path, summary_csv_path = write_deploy_episode_spreadsheet(
            run_path,
            payload,
            filename="episode_outcomes.csv",
            summary_filename="episode_outcomes_summary.csv",
        )
        if csv_path is None or summary_csv_path is None:
            return False, "Saved metadata and notes, but failed to write episode CSV files."
        row = self._current_row()
        if row is not None:
            item = row.get("item", {})
            item.clear()
            item.update(payload)
            item["_metadata_path"] = str(metadata_path)
            item["_run_path"] = str(run_path)
        return True, f"Saved deploy edits: {notes_path.name}, {csv_path.name}, {summary_csv_path.name}"

    def save_episode_edit(self) -> None:
        item, _metadata_path, metadata_data = self._read_selected_metadata()
        if item is None:
            self._set_output(title="No Selection", text="Select a run first.", log_message="History episode save skipped with no selection.")
            return
        if str(item.get("mode", "")).strip().lower() != "deploy":
            self._set_output(title="History", text="Deploy episode editing is only available for deploy runs.", log_message="History episode save skipped for non-deploy run.")
            return
        if metadata_data is None:
            self._set_output(title="History", text="Unable to read metadata.json for this run.", log_message="History episode save failed due to unreadable metadata.")
            return
        try:
            episode_idx = int(self.episode_combo.currentText().strip() or "1")
        except ValueError:
            episode_idx = 1
        summary = _normalize_deploy_episode_outcomes(metadata_data.get("deploy_episode_outcomes"))
        entries = summary.get("episode_outcomes")
        episode_map: dict[int, dict[str, Any]] = {}
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict):
                    try:
                        entry_idx = int(entry.get("episode"))
                    except (TypeError, ValueError):
                        continue
                    episode_map[entry_idx] = dict(entry)
        entry = episode_map.get(episode_idx, {"episode": episode_idx})
        entry["episode"] = episode_idx
        entry["result"] = normalize_deploy_result(self.outcome_combo.currentText())
        tags = [tag.strip() for tag in self.tags_input.text().split(",") if tag.strip()]
        entry["tags"] = tags
        note = self.episode_note_input.text().strip()
        if note:
            entry["note"] = note
        else:
            entry.pop("note", None)
        episode_map[episode_idx] = entry
        total = summary.get("total_episodes")
        if not isinstance(total, int) or total < episode_idx:
            total = episode_idx
        metadata_data["deploy_episode_outcomes"] = _normalize_deploy_episode_outcomes(
            {"enabled": True, "total_episodes": total, "episode_outcomes": [episode_map[idx] for idx in sorted(episode_map)]}
        )
        ok, message = self._persist_deploy_metadata(metadata_data)
        self.deploy_editor_status.setText(message)
        self._set_output(title="Episode Edit Saved" if ok else "Save Failed", text=message, log_message=message)
        if ok:
            self._populate_deploy_editor()

    def save_deployment_notes(self) -> None:
        item, _metadata_path, metadata_data = self._read_selected_metadata()
        if item is None:
            self._set_output(title="No Selection", text="Select a run first.", log_message="History notes save skipped with no selection.")
            return
        if str(item.get("mode", "")).strip().lower() != "deploy":
            self._set_output(title="History", text="Deployment notes are only available for deploy runs.", log_message="History notes save skipped for non-deploy run.")
            return
        if metadata_data is None:
            self._set_output(title="History", text="Unable to read metadata.json for this run.", log_message="History notes save failed due to unreadable metadata.")
            return
        notes = self.overall_notes.toPlainText().strip()
        if notes:
            metadata_data["deploy_notes_summary"] = notes
        else:
            metadata_data.pop("deploy_notes_summary", None)
        metadata_data["deploy_episode_outcomes"] = _normalize_deploy_episode_outcomes(metadata_data.get("deploy_episode_outcomes"))
        ok, message = self._persist_deploy_metadata(metadata_data)
        self.deploy_editor_status.setText(message)
        self._set_output(title="Deployment Notes Saved" if ok else "Save Failed", text=message, log_message=message)

    def open_deploy_notes_file(self) -> None:
        item, _metadata_path, metadata_data = self._read_selected_metadata()
        if item is None:
            self._set_output(title="No Selection", text="Select a run first.", log_message="History notes open skipped with no selection.")
            return
        run_path = Path(str(item.get("_run_path", "")).strip())
        if not run_path.exists():
            self._set_output(title="Open Failed", text="Run folder is missing for this history entry.", log_message="History notes open failed due to missing run path.")
            return
        notes_path = run_path / "notes.md"
        if not notes_path.exists() and metadata_data is not None:
            generated = write_deploy_notes_file(run_path, metadata_data, filename="notes.md")
            if generated is not None:
                notes_path = generated
                write_deploy_episode_spreadsheet(
                    run_path,
                    metadata_data,
                    filename="episode_outcomes.csv",
                    summary_filename="episode_outcomes_summary.csv",
                )
        ok, message = open_path_in_file_manager(notes_path)
        self._set_output(
            title="Open Notes" if ok else "Open Failed",
            text=str(notes_path if ok else message),
            log_message="History opened deploy notes." if ok else "History failed to open deploy notes.",
        )


def build_qt_secondary_panel(
    *,
    section_id: str,
    config: dict[str, Any],
    append_log: Callable[[str], None],
    run_controller: ManagedRunController,
    run_terminal_command: Callable[[str], tuple[bool, str]] | None = None,
    update_and_restart_app: Callable[[], tuple[bool, str]] | None = None,
) -> QWidget | None:
    if section_id == "config":
        return QtConfigPage(
            config=config,
            append_log=append_log,
            run_terminal_command=run_terminal_command,
            update_and_restart_app=update_and_restart_app,
        )
    if section_id == "training":
        return QtTrainingPage(config=config, append_log=append_log)
    if section_id == "visualizer":
        return QtVisualizerPage(config=config, append_log=append_log)
    if section_id == "history":
        return QtHistoryPage(config=config, append_log=append_log, run_controller=run_controller)
    return None
