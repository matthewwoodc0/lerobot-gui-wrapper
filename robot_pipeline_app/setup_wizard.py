from __future__ import annotations

import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .constants import DEFAULT_LEROBOT_DIR
from .config_store import normalize_path
from .probes import probe_module_import, summarize_probe_error

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
        return self.virtual_env_active and self.lerobot_import_ok

    @property
    def needs_bootstrap(self) -> bool:
        return not self.virtual_env_active and not self.lerobot_import_ok


def _in_virtual_env() -> bool:
    if os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX"):
        return True
    base_prefix = getattr(sys, "base_prefix", sys.prefix)
    return sys.prefix != base_prefix


def probe_setup_wizard_status(
    config: dict[str, Any],
    *,
    module_probe_fn: ModuleProbeFn = probe_module_import,
) -> SetupWizardStatus:
    lerobot_dir = Path(normalize_path(config.get("lerobot_dir", str(DEFAULT_LEROBOT_DIR))))
    venv_dir = lerobot_dir / "lerobot_env"
    lerobot_ok, lerobot_raw = module_probe_fn("lerobot")
    lerobot_detail = "import ok" if lerobot_ok else summarize_probe_error(lerobot_raw)
    return SetupWizardStatus(
        lerobot_dir=lerobot_dir,
        lerobot_dir_exists=lerobot_dir.exists(),
        venv_dir=venv_dir,
        venv_dir_exists=venv_dir.exists(),
        virtual_env_active=_in_virtual_env(),
        python_executable=sys.executable,
        lerobot_import_ok=lerobot_ok,
        lerobot_import_detail=lerobot_detail,
    )


def build_setup_status_summary(status: SetupWizardStatus) -> str:
    lines = [
        f"[{'PASS' if status.virtual_env_active else 'FAIL'}] Virtual env active: {status.virtual_env_active}",
        f"[{'PASS' if status.lerobot_import_ok else 'FAIL'}] Python module: lerobot ({status.lerobot_import_detail})",
        f"[{'PASS' if status.lerobot_dir_exists else 'WARN'}] LeRobot folder: {status.lerobot_dir}",
        f"[{'PASS' if status.venv_dir_exists else 'WARN'}] Expected venv folder: {status.venv_dir}",
        f"[INFO] Python executable: {status.python_executable}",
    ]
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
    else:
        if status.needs_bootstrap:
            lines.append(
                "Detected first-time bootstrap state: no active virtual environment and lerobot is not importable."
            )
            lines.append("")
        lines.extend(
            [
                "Run the setup commands below in your terminal, then click 'Re-check Environment' in this wizard.",
                "",
                "Suggested commands:",
                build_setup_wizard_commands(status),
                "",
                "After setup, keep this environment active before launching the app:",
                f"source {status.venv_dir / 'bin' / 'activate'}",
            ]
        )
    return "\n".join(lines)
