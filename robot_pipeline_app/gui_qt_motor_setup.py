from __future__ import annotations

from typing import Any, Callable

from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QPushButton

from .checks import has_failures
from .config_store import get_lerobot_dir, save_config
from .gui_qt_ops_base import _AdvancedOptionsPanel, _CoreOpsPanel, _InputGrid
from .hardware_workflows import (
    apply_motor_setup_success,
    build_motor_setup_preflight_checks,
    build_motor_setup_request_and_command,
    build_motor_setup_result_summary,
)
from .robot_presets import robot_type_options
from .rig_manager import active_rig_name, build_rig_snapshot, list_named_rigs
from .run_controller_service import ManagedRunController


class MotorSetupOpsPanel(_CoreOpsPanel):
    def __init__(
        self,
        *,
        config: dict[str, Any],
        append_log: Callable[[str], None],
        run_controller: ManagedRunController,
    ) -> None:
        super().__init__(
            title="Motor Setup",
            subtitle="Run first-time motor bring-up and servo configuration with preflight, review, and cancel support.",
            append_log=append_log,
            run_controller=run_controller,
        )
        self.config = config

        form = _InputGrid(self.form_layout)

        self.role_combo = QComboBox()
        self.role_combo.addItem("Follower", "follower")
        self.role_combo.addItem("Leader", "leader")
        self.role_combo.currentIndexChanged.connect(self._load_role_defaults)
        form.add_field("Role", self.role_combo)

        self.robot_type_combo = QComboBox()
        self.robot_type_combo.setEditable(True)
        self.robot_type_combo.addItems(
            robot_type_options(
                str(config.get("follower_robot_type", "")).strip(),
                str(config.get("leader_robot_type", "")).strip(),
            )
        )
        form.add_field("Robot type", self.robot_type_combo)

        self.port_input = QLineEdit("")
        self.port_input.setPlaceholderText("/dev/ttyACM0 or /dev/cu.usbmodem...")
        form.add_field("Port", self.port_input)

        self.robot_id_input = QLineEdit("")
        form.add_field("Current id", self.robot_id_input)

        self.new_id_input = QLineEdit("")
        self.new_id_input.setPlaceholderText("optional")
        form.add_field("New id", self.new_id_input)

        self.baudrate_input = QLineEdit("")
        self.baudrate_input.setPlaceholderText("optional")
        form.add_field("Baudrate", self.baudrate_input)

        self.support_label = QLabel("")
        self.support_label.setWordWrap(True)
        self.support_label.setObjectName("MutedLabel")
        self.form_layout.addWidget(self.support_label)

        self.advanced_toggle = QCheckBox("Advanced command options")
        self.advanced_toggle.toggled.connect(self._toggle_advanced_options)
        self.form_layout.addWidget(self.advanced_toggle)

        self.advanced_panel = _AdvancedOptionsPanel(
            title="Advanced Motor Setup Options",
            fields=[
                ("robot.role", "Role"),
                ("robot.type", "Robot type"),
                ("robot.port", "Port"),
                ("robot.id", "Current id"),
                ("robot.new_id", "New id"),
                ("robot.baudrate", "Baudrate"),
            ],
        )
        self.advanced_panel.hide()
        self.form_layout.addWidget(self.advanced_panel)

        actions = QHBoxLayout()
        run_button = QPushButton("Run Motor Setup")
        run_button.setObjectName("AccentButton")
        run_button.clicked.connect(self.run_motor_setup)
        actions.addWidget(run_button)
        self._register_action_button(run_button)

        preview_button = QPushButton("Preview Command")
        preview_button.clicked.connect(self.preview_command)
        actions.addWidget(preview_button)
        self._register_action_button(preview_button)

        preflight_button = QPushButton("Run Preflight")
        preflight_button.clicked.connect(self.run_preflight)
        actions.addWidget(preflight_button)
        self._register_action_button(preflight_button)

        scan_ports_button = QPushButton("Scan Robot Ports")
        scan_ports_button.clicked.connect(self.scan_robot_ports)
        actions.addWidget(scan_ports_button)
        self._register_action_button(scan_ports_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.setObjectName("DangerButton")
        cancel_button.clicked.connect(self._cancel_run)
        actions.addWidget(cancel_button)
        self._register_action_button(cancel_button, is_cancel=True)
        actions.addStretch(1)
        self.form_layout.addLayout(actions)

        self._load_role_defaults()

    def _selected_role(self) -> str:
        return str(self.role_combo.currentData() or "follower").strip().lower() or "follower"

    def _apply_role_defaults(self) -> None:
        role = self._selected_role()
        if role == "leader":
            self.robot_type_combo.setCurrentText(str(self.config.get("leader_robot_type", "")).strip())
            self.port_input.setText(str(self.config.get("leader_port", "")).strip())
            self.robot_id_input.setText(str(self.config.get("leader_robot_id", "")).strip())
        else:
            self.robot_type_combo.setCurrentText(str(self.config.get("follower_robot_type", "")).strip())
            self.port_input.setText(str(self.config.get("follower_port", "")).strip())
            self.robot_id_input.setText(str(self.config.get("follower_robot_id", "")).strip())

    def _load_role_defaults(self, *_args: object) -> None:
        self._apply_role_defaults()
        self.refresh_from_config()

    def _build(self) -> tuple[Any | None, list[str] | None, Any, str | None]:
        arg_overrides = None
        custom_args_raw = ""
        if self.advanced_toggle.isChecked():
            arg_overrides, custom_args_raw = self.advanced_panel.build_overrides()
        return build_motor_setup_request_and_command(
            config=self.config,
            role=self._selected_role(),
            port_raw=self.port_input.text(),
            robot_id_raw=self.robot_id_input.text(),
            new_id_raw=self.new_id_input.text(),
            baudrate_raw=self.baudrate_input.text(),
            robot_type_raw=self.robot_type_combo.currentText(),
            arg_overrides=arg_overrides,
            custom_args_raw=custom_args_raw,
        )

    def _toggle_advanced_options(self, checked: bool) -> None:
        if checked:
            request, cmd, _support, error = self._build()
            if error is None and request is not None and cmd is not None:
                self.advanced_panel.seed_from_command(cmd)
            self.advanced_panel.show()
        else:
            self.advanced_panel.hide()

    def refresh_from_config(self) -> None:
        self._apply_role_defaults()
        _request, _cmd, support, _error = self._build()
        self.support_label.setText(str(getattr(support, "detail", "Motor setup support status unavailable.")))

    def preview_command(self) -> None:
        request, cmd, support, error = self._build()
        self.support_label.setText(str(support.detail))
        if error or request is None or cmd is None:
            self._set_output(
                title="Validation Error",
                text=error or "Unable to build motor setup command.",
                log_message="Motor setup preview failed validation.",
            )
            return
        self._show_text_dialog(
            title="Motor Setup Command",
            text=(
                f"Role: {request.role}\n"
                f"Robot type: {request.robot_type}\n"
                f"Port: {request.port}\n"
                f"Current id: {request.robot_id}\n"
                f"New id: {request.new_id or '(unchanged)'}\n"
                f"Baudrate: {request.baudrate if request.baudrate is not None else '(unchanged)'}\n\n"
                f"{support.detail}\n\n"
                f"{' '.join(str(part) for part in cmd)}"
            ),
            wrap_mode="word",
        )
        self._append_log(f"Motor setup preview built for {request.role} @ {request.port}.")

    def run_preflight(self) -> None:
        request, cmd, support, error = self._build()
        self.support_label.setText(str(support.detail))
        if error or request is None or cmd is None:
            self._set_output(
                title="Validation Error",
                text=error or "Unable to build motor setup command.",
                log_message="Motor setup preflight failed validation.",
            )
            return
        checks = build_motor_setup_preflight_checks(request=request, support=support)
        self._show_text_dialog(
            title="Motor Setup Preflight",
            text="\n".join(f"[{level}] {name}: {detail}" for level, name, detail in checks),
            wrap_mode="char",
        )
        self._append_log(f"Motor setup preflight ran for {request.role} @ {request.port}.")

    def scan_robot_ports(self) -> None:
        follower_guess, leader_guess = self._run_port_scan_dialog(
            title="Motor Port Scan",
            current_follower=str(self.config.get("follower_port", "")),
            current_leader=str(self.config.get("leader_port", "")),
            apply_scope_label="motor setup",
        )
        role = self._selected_role()
        if role == "leader":
            if leader_guess:
                self.port_input.setText(leader_guess)
        else:
            if follower_guess:
                self.port_input.setText(follower_guess)

    def run_motor_setup(self) -> None:
        request, cmd, support, error = self._build()
        self.support_label.setText(str(support.detail))
        if error or request is None or cmd is None:
            self._set_output(
                title="Validation Error",
                text=error or "Unable to build motor setup command.",
                log_message="Motor setup launch failed validation.",
            )
            return

        editable_cmd = self._ask_editable_command_dialog(
            title="Confirm Motor Setup Command",
            command_argv=cmd,
            intro_text=(
                "Review or edit the motor setup command below.\n"
                "The exact command text here will be executed and saved to run history."
            ),
            confirm_label="Run Motor Setup",
        )
        if editable_cmd is None:
            return
        if editable_cmd != cmd:
            self._append_log("Running edited motor setup command from command editor.")
        cmd = editable_cmd

        checks = build_motor_setup_preflight_checks(request=request, support=support)
        if not self._confirm_preflight_review(title="Motor Setup Preflight", checks=checks):
            self._append_log("Motor setup canceled after preflight review.")
            return

        warning_detail = None
        if has_failures(checks):
            warning_detail = "Motor setup preflight contains FAIL items. Continue only if you intentionally want to override them."
        self._show_launch_summary(
            heading="Launching motor setup...",
            command_label="Motor setup command",
            cmd=cmd,
            preflight_title="Motor Setup Preflight",
            preflight_checks=checks,
            warning_detail=warning_detail,
        )
        self._append_log(f"Motor setup launch starting for {request.role} @ {request.port}.")

        def after_run(return_code: int, was_canceled: bool) -> None:
            if was_canceled:
                self._set_running(False, "Motor setup canceled.", False)
                self._append_output_and_log("Motor setup canceled.")
                return
            if return_code != 0:
                self._set_running(False, "Motor setup failed.", True)
                self._append_output_and_log(f"Motor setup failed with exit code {return_code}.")
                return
            previous_config = dict(self.config)
            updated_config = apply_motor_setup_success(self.config, request=request, support=support)
            self.config.clear()
            self.config.update(updated_config)
            save_config(self.config, quiet=True)
            self._load_role_defaults()
            self._set_running(False, "Motor setup completed.", False)
            summary = build_motor_setup_result_summary(
                previous_config=previous_config,
                updated_config=updated_config,
                request=request,
                support=support,
            )
            rig_notice = self._active_rig_update_notice(updated_config)
            if rig_notice:
                summary += "\n" + rig_notice
            self._set_output(title="Motor Setup Completed", text=summary, log_message=f"Motor setup completed for {request.role} @ {request.port}.")
            self._append_log(f"Motor setup completed for {request.role} @ {request.port}.")

        ok, message = self._run_controller.run_process_async(
            cmd=cmd,
            cwd=get_lerobot_dir(self.config),
            hooks=self._build_hooks(),
            complete_callback=after_run,
            run_mode="motor_setup",
            preflight_checks=checks,
            artifact_context={
                "motor_setup": {
                    "role": request.role,
                    "robot_type": request.robot_type,
                    "port": request.port,
                    "robot_id": request.robot_id,
                    "new_id": request.new_id,
                    "baudrate": request.baudrate,
                    "support_detail": support.detail,
                    "uses_calibrate_fallback": support.uses_calibrate_fallback,
                }
            },
        )
        if not ok and message:
            self._handle_launch_rejection(
                title="Motor Setup Unavailable",
                message=message,
                log_message="Motor setup launch was rejected.",
            )

    def _active_rig_update_notice(self, updated_config: dict[str, Any]) -> str:
        rig_name = active_rig_name(updated_config)
        if not rig_name:
            return ""
        active_rig = next(
            (item for item in list_named_rigs(updated_config) if str(item.get("name", "")).strip().lower() == rig_name.lower()),
            None,
        )
        if active_rig is None:
            return ""
        snapshot = active_rig.get("snapshot")
        if not isinstance(snapshot, dict):
            return ""
        current_snapshot = build_rig_snapshot(updated_config)
        if dict(snapshot) == current_snapshot:
            return ""
        return f"Active rig '{rig_name}' now differs from its saved snapshot. Suggested next action: Save Rig."
