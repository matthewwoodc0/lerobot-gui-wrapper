from __future__ import annotations

import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .constants import DEFAULT_LEROBOT_DIR
from .config_store import normalize_path
from .probes import in_virtual_env, probe_module_import, summarize_probe_error

ModuleProbeFn = Callable[[str], tuple[bool, str]]


@dataclass(frozen=True)
class SetupWizardStatus:
    lerobot_dir: Path
    lerobot_dir_exists: bool
    venv_dir: Path
    venv_dir_exists: bool
    virtual_env_active: bool
    python_executable: str
    lerobot_import_ok: bool
    lerobot_import_detail: str

    @property
    def ready(self) -> bool:
        # Gate only on lerobot being importable; virtual_env_active is a warning
        # indicator only.  When the launcher embeds the correct Python binary the
        # env probe may not fire, but imports still work fine.
        return self.lerobot_import_ok

    @property
    def needs_bootstrap(self) -> bool:
        return not self.virtual_env_active and not self.lerobot_import_ok



def probe_setup_wizard_status(
    config: dict[str, Any],
    *,
    module_probe_fn: ModuleProbeFn = probe_module_import,
) -> SetupWizardStatus:
    lerobot_dir = Path(normalize_path(config.get("lerobot_dir", str(DEFAULT_LEROBOT_DIR))))
    venv_dir = Path(normalize_path(config.get("lerobot_venv_dir", str(lerobot_dir / "lerobot_env"))))
    lerobot_ok, lerobot_raw = module_probe_fn("lerobot")
    lerobot_detail = "import ok" if lerobot_ok else summarize_probe_error(lerobot_raw)
    return SetupWizardStatus(
        lerobot_dir=lerobot_dir,
        lerobot_dir_exists=lerobot_dir.exists(),
        venv_dir=venv_dir,
        venv_dir_exists=venv_dir.exists(),
        virtual_env_active=in_virtual_env(),
        python_executable=sys.executable,
        lerobot_import_ok=lerobot_ok,
        lerobot_import_detail=lerobot_detail,
    )


def _env_type_label() -> str:
    """Return a human-readable label for the active environment type."""
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        env_name = os.environ.get("CONDA_DEFAULT_ENV", "")
        return f"conda ({env_name})" if env_name else f"conda ({conda_prefix})"
    if os.environ.get("VIRTUAL_ENV"):
        return "venv"
    base_prefix = getattr(import_sys := __import__("sys"), "base_prefix", import_sys.prefix)
    if import_sys.prefix != base_prefix:
        return "venv (non-standard)"
    return "none"


def build_setup_status_summary(status: SetupWizardStatus) -> str:
    env_label = _env_type_label()
    conda_active = bool(os.environ.get("CONDA_PREFIX"))
    lines = [
        f"[{'PASS' if status.virtual_env_active else 'FAIL'}] Environment active: {status.virtual_env_active} ({env_label})",
        f"[{'PASS' if status.lerobot_import_ok else 'FAIL'}] Python module: lerobot ({status.lerobot_import_detail})",
        f"[{'PASS' if status.lerobot_dir_exists else 'WARN'}] LeRobot folder: {status.lerobot_dir}",
    ]
    if not conda_active:
        lines.append(f"[{'PASS' if status.venv_dir_exists else 'WARN'}] Expected venv folder: {status.venv_dir}")
    lines.append(f"[INFO] Python executable: {status.python_executable}")
    if status.ready:
        lines.append("[READY] Environment looks good for LeRobot record/deploy.")
    elif status.needs_bootstrap:
        lines.append("[ACTION] Neither virtual env nor lerobot import is working. Run guided setup.")
    else:
        lines.append("[ACTION] Finish remaining setup items, then re-check.")
    return "\n".join(lines)


def build_setup_wizard_commands(status: SetupWizardStatus) -> str:
    lerobot_dir = status.lerobot_dir
    venv_dir = status.venv_dir
    activate_script = venv_dir / "bin" / "activate"
    parent_dir = lerobot_dir.parent
    quoted_parent = shlex.quote(str(parent_dir))
    quoted_lerobot = shlex.quote(str(lerobot_dir))
    quoted_venv = shlex.quote(str(venv_dir))
    quoted_activate = shlex.quote(str(activate_script))
    lines = [
        "# 1) Clone LeRobot source (skip if already cloned)",
        f"mkdir -p {quoted_parent}",
        f"if [ ! -d {quoted_lerobot}/.git ]; then git clone https://github.com/huggingface/lerobot {quoted_lerobot}; fi",
        "",
        "# 2) Create and activate virtual environment",
        f"python3 -m venv {quoted_venv}",
        f"source {quoted_activate}",
        "",
        "# 3) Install LeRobot in editable mode",
        f"cd {quoted_lerobot}",
        "python3 -m pip install --upgrade pip",
        "python3 -m pip install -e .",
        "",
        "# 4) Verify",
        "python3 -c \"import lerobot,sys; print('LeRobot OK:', lerobot.__file__); print('Python:', sys.executable)\"",
    ]
    return "\n".join(lines)


def build_setup_wizard_guide(status: SetupWizardStatus) -> str:
    conda_active = bool(os.environ.get("CONDA_PREFIX"))
    lines = [
        "LeRobot Setup Wizard (Popout)",
        "",
        "Environment check:",
        build_setup_status_summary(status),
        "",
    ]
    if status.ready:
        lines.extend(
            [
                "Everything required is available.",
                "You can proceed to Record/Deploy now.",
            ]
        )
    elif conda_active and not status.lerobot_import_ok:
        # Conda environment is active but lerobot is not installed
        lines.extend(
            [
                "A conda environment is active but lerobot is not importable.",
                "Make sure lerobot is installed inside your active conda environment:",
                "",
                f"  cd {status.lerobot_dir}",
                "  pip install -e .",
                "",
                "Then click 'Re-check Environment' in this wizard.",
            ]
        )
    elif not status.virtual_env_active and not status.lerobot_import_ok:
        lines.append(
            "Detected first-time bootstrap state: no active environment and lerobot is not importable."
        )
        lines.append("")
        lines.extend(
            [
                "Option A — venv (recommended for new setups):",
                "Run the setup commands below in your terminal, then click 'Re-check Environment' in this wizard.",
                "",
                "Suggested commands:",
                build_setup_wizard_commands(status),
                "",
                "After setup, keep this environment active before launching the app:",
                f"  source {status.venv_dir / 'bin' / 'activate'}",
                "",
                "Option B — conda (if you already have a conda environment with lerobot):",
                "  conda activate lerobot",
                f"  cd {status.lerobot_dir.parent}",
                f"  python3 {status.lerobot_dir.parent / 'lerobot-gui-wrapper' / 'robot_pipeline.py'} gui",
                "",
                "Note: The desktop launcher works best when installed while your environment is active.",
            ]
        )
    else:
        lines.extend(
            [
                "Run the setup commands below in your terminal, then click 'Re-check Environment' in this wizard.",
                "",
                "Suggested commands:",
                build_setup_wizard_commands(status),
                "",
                "After setup, keep this environment active before launching the app:",
                f"  source {status.venv_dir / 'bin' / 'activate'}",
            ]
        )
    return "\n".join(lines)
