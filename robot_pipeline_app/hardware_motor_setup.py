from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .command_overrides import apply_command_overrides
from .commands import (
    build_lerobot_calibrate_command,
    follower_robot_type,
    leader_robot_type,
    resolve_follower_robot_id,
    resolve_leader_robot_id,
)
from .compat import probe_entrypoint_help_flags, resolve_calibrate_entrypoint, resolve_motor_setup_entrypoint
from .lerobot_runtime import build_lerobot_module_command

_MOTOR_ROLE_FLAG_CANDIDATES: tuple[str, ...] = ("robot.role", "role")
_MOTOR_TYPE_FLAG_CANDIDATES: tuple[str, ...] = ("robot.type", "type")
_MOTOR_PORT_FLAG_CANDIDATES: tuple[str, ...] = ("robot.port", "port")
_MOTOR_ID_FLAG_CANDIDATES: tuple[str, ...] = ("robot.id", "id", "current_id")
_MOTOR_NEW_ID_FLAG_CANDIDATES: tuple[str, ...] = ("robot.new_id", "new_id", "target_id")
_MOTOR_BAUDRATE_FLAG_CANDIDATES: tuple[str, ...] = ("robot.baudrate", "baudrate", "serial.baudrate")


@dataclass(frozen=True)
class MotorSetupSupport:
    available: bool
    entrypoint: str
    detail: str
    supported_flags: tuple[str, ...]
    role_flag: str | None
    type_flag: str | None
    port_flag: str | None
    id_flag: str | None
    new_id_flag: str | None
    baudrate_flag: str | None
    uses_calibrate_fallback: bool = False
    used_fallback_flags: bool = False


@dataclass(frozen=True)
class MotorSetupRequest:
    role: str
    robot_type: str
    port: str
    robot_id: str
    new_id: str
    baudrate: int | None


def _choose_flag(flags: set[str], candidates: tuple[str, ...], *, keywords: tuple[str, ...] = ()) -> str | None:
    for candidate in candidates:
        if candidate in flags:
            return candidate
    for candidate in sorted(flags):
        normalized = candidate.lower()
        if keywords and all(keyword in normalized for keyword in keywords):
            return candidate
    return None


def _parse_positive_int(value: Any) -> int | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def probe_motor_setup_support(config: dict[str, Any]) -> MotorSetupSupport:
    entrypoint = resolve_motor_setup_entrypoint(config)
    if not entrypoint:
        calibrate_entrypoint = resolve_calibrate_entrypoint(config)
        if not calibrate_entrypoint:
            return MotorSetupSupport(
                available=False,
                entrypoint="",
                detail="Motor setup is unavailable: no dedicated setup or calibrate entrypoint was detected.",
                supported_flags=(),
                role_flag=None,
                type_flag=None,
                port_flag=None,
                id_flag=None,
                new_id_flag=None,
                baudrate_flag=None,
            )
        return MotorSetupSupport(
            available=True,
            entrypoint=calibrate_entrypoint,
            detail=(
                "Dedicated motor setup entrypoint was not detected. "
                "The GUI will fall back to calibration-only bring-up, so ID and baudrate edits are informational."
            ),
            supported_flags=(),
            role_flag=None,
            type_flag=None,
            port_flag=None,
            id_flag=None,
            new_id_flag=None,
            baudrate_flag=None,
            uses_calibrate_fallback=True,
        )

    flags, error_text = probe_entrypoint_help_flags(config, entrypoint)
    role_flag = _choose_flag(flags, _MOTOR_ROLE_FLAG_CANDIDATES, keywords=("role",))
    type_flag = _choose_flag(flags, _MOTOR_TYPE_FLAG_CANDIDATES, keywords=("robot", "type"))
    port_flag = _choose_flag(flags, _MOTOR_PORT_FLAG_CANDIDATES, keywords=("port",))
    id_flag = _choose_flag(flags, _MOTOR_ID_FLAG_CANDIDATES, keywords=("id",))
    new_id_flag = _choose_flag(flags, _MOTOR_NEW_ID_FLAG_CANDIDATES, keywords=("new", "id"))
    baudrate_flag = _choose_flag(flags, _MOTOR_BAUDRATE_FLAG_CANDIDATES, keywords=("baud",))

    used_fallback_flags = False
    detail = f"Motor setup entrypoint detected: {entrypoint}."
    if not flags:
        used_fallback_flags = True
        type_flag = type_flag or "robot.type"
        port_flag = port_flag or "robot.port"
        id_flag = id_flag or "robot.id"
        new_id_flag = new_id_flag or "robot.new_id"
        baudrate_flag = baudrate_flag or "robot.baudrate"
        detail = (
            f"Motor setup entrypoint detected: {entrypoint}. Help probing was unavailable"
            + (f" ({error_text})." if error_text else ".")
            + " The GUI is using fallback motor-setup flags; review the command before launch."
        )

    return MotorSetupSupport(
        available=True,
        entrypoint=entrypoint,
        detail=detail,
        supported_flags=tuple(sorted(flags)),
        role_flag=role_flag,
        type_flag=type_flag,
        port_flag=port_flag,
        id_flag=id_flag,
        new_id_flag=new_id_flag,
        baudrate_flag=baudrate_flag,
        uses_calibrate_fallback=False,
        used_fallback_flags=used_fallback_flags,
    )


def build_motor_setup_request_and_command(
    *,
    config: dict[str, Any],
    role: str,
    port_raw: str,
    robot_id_raw: str,
    new_id_raw: str = "",
    baudrate_raw: str = "",
    robot_type_raw: str = "",
    arg_overrides: dict[str, str] | None = None,
    custom_args_raw: str = "",
) -> tuple[MotorSetupRequest | None, list[str] | None, MotorSetupSupport, str | None]:
    selected_role = str(role or "follower").strip().lower()
    if selected_role not in {"follower", "leader"}:
        selected_role = "follower"

    support = probe_motor_setup_support(config)
    if not support.available:
        return None, None, support, support.detail

    port = str(port_raw or "").strip()
    if not port:
        return None, None, support, "Robot port is required for motor bring-up."

    default_robot_type = follower_robot_type(config) if selected_role == "follower" else leader_robot_type(config)
    default_robot_id = resolve_follower_robot_id(config) if selected_role == "follower" else resolve_leader_robot_id(config)
    robot_type = str(robot_type_raw or "").strip() or default_robot_type
    robot_id = str(robot_id_raw or "").strip() or default_robot_id
    new_id = str(new_id_raw or "").strip()
    baudrate = _parse_positive_int(baudrate_raw)
    if str(baudrate_raw or "").strip() and baudrate is None:
        return None, None, support, "Baudrate must be a positive integer."

    request = MotorSetupRequest(
        role=selected_role,
        robot_type=robot_type,
        port=port,
        robot_id=robot_id,
        new_id=new_id,
        baudrate=baudrate,
    )

    if support.uses_calibrate_fallback:
        runtime_config = dict(config)
        if selected_role == "leader":
            runtime_config["leader_port"] = port
            runtime_config["leader_robot_id"] = robot_id
            runtime_config["leader_robot_type"] = robot_type
        else:
            runtime_config["follower_port"] = port
            runtime_config["follower_robot_id"] = robot_id
            runtime_config["follower_robot_type"] = robot_type
        cmd = build_lerobot_calibrate_command(runtime_config, role=selected_role)
    else:
        cmd = [*build_lerobot_module_command(config, support.entrypoint)]
        if support.role_flag:
            cmd.append(f"--{support.role_flag}={selected_role}")
        if support.type_flag:
            cmd.append(f"--{support.type_flag}={robot_type}")
        if support.port_flag:
            cmd.append(f"--{support.port_flag}={port}")
        if support.id_flag and robot_id:
            cmd.append(f"--{support.id_flag}={robot_id}")
        if support.new_id_flag and new_id:
            cmd.append(f"--{support.new_id_flag}={new_id}")
        if support.baudrate_flag and baudrate is not None:
            cmd.append(f"--{support.baudrate_flag}={baudrate}")

    cmd, override_error = apply_command_overrides(
        base_cmd=cmd,
        overrides=arg_overrides,
        custom_args_raw=custom_args_raw,
    )
    if override_error or cmd is None:
        return None, None, support, override_error or "Unable to apply motor setup overrides."
    return request, cmd, support, None


def build_motor_setup_preflight_checks(
    *,
    request: MotorSetupRequest,
    support: MotorSetupSupport,
) -> list[tuple[str, str, str]]:
    checks: list[tuple[str, str, str]] = []
    level = "WARN" if support.uses_calibrate_fallback else "PASS"
    checks.append((level, "Motor setup entrypoint", support.detail))
    checks.append(("PASS" if request.port else "FAIL", "Robot port", request.port or "Robot port is missing."))
    checks.append(("PASS" if request.robot_id else "WARN", "Robot id", request.robot_id or "Robot id is not set."))
    if request.new_id:
        checks.append(
            (
                "PASS" if bool(support.new_id_flag and not support.uses_calibrate_fallback) else "WARN",
                "Motor id reassignment",
                (
                    f"Will request new motor id '{request.new_id}'."
                    if support.new_id_flag and not support.uses_calibrate_fallback
                    else "Current runtime does not expose a dedicated new-id flag; the new id will not be applied automatically."
                ),
            )
        )
    if request.baudrate is not None:
        checks.append(
            (
                "PASS" if bool(support.baudrate_flag and not support.uses_calibrate_fallback) else "WARN",
                "Motor baudrate",
                (
                    f"Will request baudrate {request.baudrate}."
                    if support.baudrate_flag and not support.uses_calibrate_fallback
                    else "Current runtime does not expose a dedicated baudrate flag; the baudrate will not be applied automatically."
                ),
            )
        )
    if support.used_fallback_flags:
        checks.append(("WARN", "Motor setup flag probe", "Help probing was unavailable, so the GUI is using fallback motor-setup flags."))
    return checks


def apply_motor_setup_success(
    config: dict[str, Any],
    *,
    request: MotorSetupRequest,
    support: MotorSetupSupport,
) -> dict[str, Any]:
    updated = dict(config)
    id_value = request.robot_id
    if request.new_id and support.new_id_flag and not support.uses_calibrate_fallback:
        id_value = request.new_id
    if request.role == "leader":
        updated["leader_port"] = request.port
        updated["leader_robot_id"] = id_value
        updated["leader_robot_type"] = request.robot_type
    else:
        updated["follower_port"] = request.port
        updated["follower_robot_id"] = id_value
        updated["follower_robot_type"] = request.robot_type
    return updated


def build_motor_setup_result_summary(
    *,
    previous_config: dict[str, Any],
    updated_config: dict[str, Any],
    request: MotorSetupRequest,
    support: MotorSetupSupport,
) -> str:
    changed_fields: list[str] = []
    for key in ("leader_port", "leader_robot_id", "leader_robot_type", "follower_port", "follower_robot_id", "follower_robot_type"):
        before = str(previous_config.get(key, "")).strip()
        after = str(updated_config.get(key, "")).strip()
        if before != after:
            changed_fields.append(f"{key}: {before or '(empty)'} -> {after or '(empty)'}")

    changed = bool(changed_fields)
    if not changed:
        changed_fields.append("No persisted config fields changed.")

    id_result = "Applied by runtime flags." if request.new_id and support.new_id_flag and not support.uses_calibrate_fallback else (
        "Informational only." if request.new_id else "Not requested."
    )
    baudrate_result = "Applied by runtime flags." if request.baudrate is not None and support.baudrate_flag and not support.uses_calibrate_fallback else (
        "Informational only." if request.baudrate is not None else "Not requested."
    )

    lines = [
        f"Role: {request.role}",
        "Changed config fields:",
        *[f"- {line}" for line in changed_fields],
        f"Motor id update: {id_result}",
        f"Baudrate update: {baudrate_result}",
        f"Suggested next action: {'Save Rig' if changed else 'Open Teleop'}",
    ]
    if support.uses_calibrate_fallback:
        lines.append("Calibration-only fallback was used, so ID and baudrate values were recorded for guidance but not applied by flags.")
    return "\n".join(lines)
