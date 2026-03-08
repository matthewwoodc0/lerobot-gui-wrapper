from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .checks import run_preflight_for_deploy, run_preflight_for_record, run_preflight_for_teleop, summarize_checks
from .command_text import format_command_for_dialog
from .constants import DEFAULT_TASK
from .gui_forms import (
    build_deploy_request_and_command,
    build_record_request_and_command,
    build_teleop_request_and_command,
)
from .repo_utils import normalize_repo_id


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


class _CoreOpsPanel(QWidget):
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

        self.form_card, self.form_layout = _build_card("Workflow Inputs")
        layout.addWidget(self.form_card)

        self.output_card, output_layout = _build_card("Workflow Output")
        self.status_label = QLabel("Ready.")
        self.status_label.setObjectName("StatusChip")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setMaximumWidth(240)
        output_layout.addWidget(self.status_label)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMinimumHeight(220)
        output_layout.addWidget(self.output)
        layout.addWidget(self.output_card, 1)

    def _set_output(self, *, title: str, text: str, log_message: str) -> None:
        self.status_label.setText(title)
        self.output.setPlainText(text)
        self._append_log(log_message)


class RecordOpsPanel(_CoreOpsPanel):
    def __init__(self, *, config: dict[str, Any], append_log: Callable[[str], None]) -> None:
        super().__init__(
            title="Record",
            subtitle="Qt preview now builds real record commands and runs record preflight against current config.",
            append_log=append_log,
        )
        self.config = config

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)
        self.form_layout.addLayout(form)

        default_dataset = str(config.get("last_dataset_repo_id", "")).strip() or str(config.get("last_dataset_name", "dataset_1"))
        self.dataset_input = QLineEdit(default_dataset)
        self.dataset_input.setPlaceholderText("owner/dataset_name or dataset_name")
        form.addRow("Dataset", self.dataset_input)

        self.dataset_root_input = QLineEdit(str(config.get("record_data_dir", "")))
        form.addRow("Dataset root", self.dataset_root_input)

        self.task_input = QLineEdit(str(config.get("last_task", DEFAULT_TASK)) or DEFAULT_TASK)
        form.addRow("Task", self.task_input)

        self.episodes_input = QSpinBox()
        self.episodes_input.setRange(1, 10000)
        self.episodes_input.setValue(20)
        form.addRow("Episodes", self.episodes_input)

        self.duration_input = QSpinBox()
        self.duration_input.setRange(1, 3600)
        self.duration_input.setValue(20)
        form.addRow("Episode time (s)", self.duration_input)

        self.target_hz_input = QLineEdit(str(config.get("record_target_hz", "")).strip())
        self.target_hz_input.setPlaceholderText("optional")
        form.addRow("Target Hz", self.target_hz_input)

        self.upload_checkbox = QCheckBox("Upload to Hugging Face after record")
        self.upload_checkbox.setChecked(False)
        self.form_layout.addWidget(self.upload_checkbox)

        actions = QHBoxLayout()
        preview_button = QPushButton("Preview Command")
        preview_button.setObjectName("AccentButton")
        preview_button.clicked.connect(self.preview_command)
        actions.addWidget(preview_button)

        preflight_button = QPushButton("Run Preflight")
        preflight_button.clicked.connect(self.run_preflight)
        actions.addWidget(preflight_button)
        actions.addStretch(1)
        self.form_layout.addLayout(actions)

    def _build(self) -> tuple[Any | None, list[str] | None, str | None]:
        return build_record_request_and_command(
            config=self.config,
            dataset_input=self.dataset_input.text(),
            episodes_raw=str(self.episodes_input.value()),
            duration_raw=str(self.duration_input.value()),
            task_raw=self.task_input.text(),
            dataset_dir_raw=self.dataset_root_input.text(),
            upload_enabled=self.upload_checkbox.isChecked(),
            target_hz_raw=self.target_hz_input.text(),
        )

    def preview_command(self) -> None:
        req, cmd, error = self._build()
        if error or req is None or cmd is None:
            self._set_output(title="Validation Error", text=error or "Unable to build record command.", log_message="Qt record preview failed validation.")
            return
        summary = (
            f"Record target: {req.dataset_repo_id}\n"
            f"Episodes: {req.num_episodes}\n"
            f"Episode time: {req.episode_time_s}s\n"
            f"Upload after record: {req.upload_after_record}\n\n"
            f"{format_command_for_dialog(cmd)}"
        )
        self._set_output(title="Record Command", text=summary, log_message=f"Qt record preview built for {req.dataset_repo_id}.")

    def run_preflight(self) -> None:
        req, cmd, error = self._build()
        if error or req is None or cmd is None:
            self._set_output(title="Validation Error", text=error or "Unable to build record command.", log_message="Qt record preflight failed validation.")
            return
        checks = run_preflight_for_record(
            config=self.config,
            dataset_root=req.dataset_root,
            upload_enabled=req.upload_after_record,
            episode_time_s=req.episode_time_s,
            dataset_repo_id=req.dataset_repo_id,
        )
        self._set_output(
            title="Record Preflight",
            text=summarize_checks(checks, title="Record Preflight"),
            log_message=f"Qt record preflight ran for {req.dataset_repo_id}.",
        )


class DeployOpsPanel(_CoreOpsPanel):
    def __init__(self, *, config: dict[str, Any], append_log: Callable[[str], None]) -> None:
        super().__init__(
            title="Deploy",
            subtitle="Qt preview now builds real deploy commands and deploy preflight against the selected model path.",
            append_log=append_log,
        )
        self.config = config

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)
        self.form_layout.addLayout(form)

        models_root = str(config.get("trained_models_dir", ""))
        self.models_root_input = QLineEdit(models_root)
        form.addRow("Models root", self.models_root_input)

        last_model_name = str(config.get("last_model_name", "")).strip()
        self.model_path_input = QLineEdit(last_model_name)
        self.model_path_input.setPlaceholderText("absolute model path or relative folder under models root")
        form.addRow("Model path", self.model_path_input)

        default_eval_name = str(config.get("last_eval_dataset_name", "")).strip() or "eval_run_1"
        default_eval = normalize_repo_id(str(config.get("hf_username", "")), default_eval_name)
        self.eval_dataset_input = QLineEdit(default_eval)
        form.addRow("Eval dataset", self.eval_dataset_input)

        self.task_input = QLineEdit(str(config.get("eval_task", DEFAULT_TASK)) or DEFAULT_TASK)
        form.addRow("Eval task", self.task_input)

        self.episodes_input = QSpinBox()
        self.episodes_input.setRange(1, 10000)
        self.episodes_input.setValue(int(config.get("eval_num_episodes", 10) or 10))
        form.addRow("Eval episodes", self.episodes_input)

        self.duration_input = QSpinBox()
        self.duration_input.setRange(1, 3600)
        self.duration_input.setValue(int(config.get("eval_duration_s", 20) or 20))
        form.addRow("Eval duration (s)", self.duration_input)

        self.target_hz_input = QLineEdit(str(config.get("deploy_target_hz", "")).strip())
        self.target_hz_input.setPlaceholderText("optional")
        form.addRow("Target Hz", self.target_hz_input)

        actions = QHBoxLayout()
        preview_button = QPushButton("Preview Command")
        preview_button.setObjectName("AccentButton")
        preview_button.clicked.connect(self.preview_command)
        actions.addWidget(preview_button)

        preflight_button = QPushButton("Run Preflight")
        preflight_button.clicked.connect(self.run_preflight)
        actions.addWidget(preflight_button)
        actions.addStretch(1)
        self.form_layout.addLayout(actions)

    def _build(self) -> tuple[Any | None, list[str] | None, dict[str, Any] | None, str | None]:
        return build_deploy_request_and_command(
            config=self.config,
            deploy_root_raw=self.models_root_input.text(),
            deploy_model_raw=self.model_path_input.text(),
            eval_dataset_raw=self.eval_dataset_input.text(),
            eval_episodes_raw=str(self.episodes_input.value()),
            eval_duration_raw=str(self.duration_input.value()),
            eval_task_raw=self.task_input.text(),
            target_hz_raw=self.target_hz_input.text(),
        )

    def preview_command(self) -> None:
        req, cmd, _updated, error = self._build()
        if error or req is None or cmd is None:
            self._set_output(title="Validation Error", text=error or "Unable to build deploy command.", log_message="Qt deploy preview failed validation.")
            return
        summary = (
            f"Model path: {req.model_path}\n"
            f"Eval dataset: {req.eval_repo_id}\n"
            f"Episodes: {req.eval_num_episodes}\n"
            f"Duration: {req.eval_duration_s}s\n\n"
            f"{format_command_for_dialog(cmd)}"
        )
        self._set_output(title="Deploy Command", text=summary, log_message=f"Qt deploy preview built for {req.eval_repo_id}.")

    def run_preflight(self) -> None:
        req, cmd, _updated, error = self._build()
        if error or req is None or cmd is None:
            self._set_output(title="Validation Error", text=error or "Unable to build deploy command.", log_message="Qt deploy preflight failed validation.")
            return
        checks = run_preflight_for_deploy(
            config=self.config,
            model_path=req.model_path,
            eval_repo_id=req.eval_repo_id,
            command=cmd,
        )
        self._set_output(
            title="Deploy Preflight",
            text=summarize_checks(checks, title="Deploy Preflight"),
            log_message=f"Qt deploy preflight ran for {req.eval_repo_id}.",
        )


class TeleopOpsPanel(_CoreOpsPanel):
    def __init__(self, *, config: dict[str, Any], append_log: Callable[[str], None]) -> None:
        super().__init__(
            title="Teleop",
            subtitle="Qt preview now builds real teleop commands and teleop preflight from saved robot connection defaults.",
            append_log=append_log,
        )
        self.config = config

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)
        self.form_layout.addLayout(form)

        self.follower_port_input = QLineEdit(str(config.get("follower_port", "")))
        form.addRow("Follower port", self.follower_port_input)

        self.leader_port_input = QLineEdit(str(config.get("leader_port", "")))
        form.addRow("Leader port", self.leader_port_input)

        self.follower_id_input = QLineEdit(str(config.get("follower_robot_id", "red4")).strip() or "red4")
        form.addRow("Follower id", self.follower_id_input)

        self.leader_id_input = QLineEdit(str(config.get("leader_robot_id", "white")).strip() or "white")
        form.addRow("Leader id", self.leader_id_input)

        actions = QHBoxLayout()
        preview_button = QPushButton("Preview Command")
        preview_button.setObjectName("AccentButton")
        preview_button.clicked.connect(self.preview_command)
        actions.addWidget(preview_button)

        preflight_button = QPushButton("Run Preflight")
        preflight_button.clicked.connect(self.run_preflight)
        actions.addWidget(preflight_button)
        actions.addStretch(1)
        self.form_layout.addLayout(actions)

    def _build(self) -> tuple[Any | None, list[str] | None, dict[str, Any] | None, str | None]:
        return build_teleop_request_and_command(
            config=self.config,
            follower_port_raw=self.follower_port_input.text(),
            leader_port_raw=self.leader_port_input.text(),
            follower_id_raw=self.follower_id_input.text(),
            leader_id_raw=self.leader_id_input.text(),
        )

    def preview_command(self) -> None:
        req, cmd, _updated, error = self._build()
        if error or req is None or cmd is None:
            self._set_output(title="Validation Error", text=error or "Unable to build teleop command.", log_message="Qt teleop preview failed validation.")
            return
        summary = (
            f"Follower: {req.follower_port} ({req.follower_id})\n"
            f"Leader: {req.leader_port} ({req.leader_id})\n\n"
            f"{format_command_for_dialog(cmd)}"
        )
        self._set_output(title="Teleop Command", text=summary, log_message="Qt teleop preview built.")

    def run_preflight(self) -> None:
        req, _cmd, updated, error = self._build()
        if error or req is None or updated is None:
            self._set_output(title="Validation Error", text=error or "Unable to build teleop command.", log_message="Qt teleop preflight failed validation.")
            return
        run_config = {**self.config, **updated}
        checks = run_preflight_for_teleop(run_config, control_fps=req.control_fps)
        self._set_output(
            title="Teleop Preflight",
            text=summarize_checks(checks, title="Teleop Preflight"),
            log_message="Qt teleop preflight ran.",
        )


def build_qt_core_ops_panel(*, section_id: str, config: dict[str, Any], append_log: Callable[[str], None]) -> QWidget | None:
    if section_id == "record":
        return RecordOpsPanel(config=config, append_log=append_log)
    if section_id == "deploy":
        return DeployOpsPanel(config=config, append_log=append_log)
    if section_id == "teleop":
        return TeleopOpsPanel(config=config, append_log=append_log)
    return None
