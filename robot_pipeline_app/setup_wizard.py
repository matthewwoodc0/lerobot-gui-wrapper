from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .constants import DEFAULT_LEROBOT_DIR
from .config_store import normalize_path
from .probes import in_virtual_env, probe_module_import, summarize_probe_error

ModuleProbeFn = Callable[[str], tuple[bool, str]]
UpdateProbeFn = Callable[[Path], tuple[str, str]]


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
    app_update_state: str
    app_update_detail: str

    @property
    def ready(self) -> bool:
        # Setup is only considered ready when both:
        # 1) the runtime is inside an active venv/conda environment, and
        # 2) lerobot is importable in that environment.
        return self.virtual_env_active and self.lerobot_import_ok

    @property
    def needs_bootstrap(self) -> bool:
        return not self.virtual_env_active and not self.lerobot_import_ok



def probe_setup_wizard_status(
    config: dict[str, Any],
    *,
    module_probe_fn: ModuleProbeFn = probe_module_import,
    update_probe_fn: UpdateProbeFn | None = None,
) -> SetupWizardStatus:
    lerobot_dir = Path(normalize_path(config.get("lerobot_dir", str(DEFAULT_LEROBOT_DIR))))
    venv_dir = Path(normalize_path(config.get("lerobot_venv_dir", str(lerobot_dir / "lerobot_env"))))
    lerobot_ok, lerobot_raw = module_probe_fn("lerobot")
    lerobot_detail = "import ok" if lerobot_ok else summarize_probe_error(lerobot_raw)
    app_dir = Path(__file__).resolve().parents[1]
    probe_fn = update_probe_fn or _probe_wrapper_update_status
    app_update_state, app_update_detail = probe_fn(app_dir)
    return SetupWizardStatus(
        lerobot_dir=lerobot_dir,
        lerobot_dir_exists=lerobot_dir.exists(),
        venv_dir=venv_dir,
        venv_dir_exists=venv_dir.exists(),
        virtual_env_active=in_virtual_env(),
        python_executable=sys.executable,
        lerobot_import_ok=lerobot_ok,
        lerobot_import_detail=lerobot_detail,
        app_update_state=app_update_state,
        app_update_detail=app_update_detail,
    )


def _probe_wrapper_update_status(app_dir: Path) -> tuple[str, str]:
    """Return (state, detail) for wrapper Git update status.

    State values:
    - ``up_to_date``: local checkout is current with upstream
    - ``update_available``: local checkout is behind upstream
    - ``unknown``: unable to determine status (offline, no git, no upstream, etc.)
    """
    git_dir = app_dir / ".git"
    if not git_dir.exists():
        return "unknown", f"{app_dir} is not a git checkout; update check skipped"

    def _run_git(args: list[str], *, timeout_s: float = 5.0) -> subprocess.CompletedProcess[str] | None:
        try:
            git_env = os.environ.copy()
            # Never block UI waiting for credential prompts during update checks.
            git_env["GIT_TERMINAL_PROMPT"] = "0"
            return subprocess.run(
                ["git", *args],
                cwd=str(app_dir),
                check=False,
                capture_output=True,
                text=True,
                env=git_env,
                timeout=timeout_s,
            )
        except FileNotFoundError:
            return None
        except Exception:
            return None

    branch_result = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    if branch_result is None:
        return "unknown", "git is unavailable; update check skipped"
    if branch_result.returncode != 0:
        return "unknown", "could not determine current git branch"
    branch = str(branch_result.stdout or "").strip() or "HEAD"

    upstream_result = _run_git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    if upstream_result is None:
        return "unknown", "git is unavailable; update check skipped"
    if upstream_result.returncode != 0:
        return "unknown", f"branch '{branch}' has no upstream tracking branch"
    upstream_ref = str(upstream_result.stdout or "").strip()
    remote_name = upstream_ref.split("/", 1)[0]
    if not remote_name:
        return "unknown", f"could not parse upstream ref '{upstream_ref}'"

    fetch_result = _run_git(["fetch", "--quiet", remote_name], timeout_s=8.0)
    if fetch_result is None:
        return "unknown", "git is unavailable; update check skipped"
    if fetch_result.returncode != 0:
        fetch_error = (fetch_result.stderr or fetch_result.stdout or "").strip()
        if fetch_error:
            return "unknown", f"could not reach remote for update check ({fetch_error})"
        return "unknown", "could not reach remote for update check"

    counts_result = _run_git(["rev-list", "--left-right", "--count", "HEAD...@{u}"])
    if counts_result is None or counts_result.returncode != 0:
        return "unknown", "could not compare local branch with upstream"

    counts_text = str(counts_result.stdout or "").strip()
    try:
        ahead_str, behind_str = counts_text.split()
        ahead = int(ahead_str)
        behind = int(behind_str)
    except Exception:
        return "unknown", f"unexpected git compare output: {counts_text or '(empty)'}"

    if behind > 0:
        return "update_available", f"{behind} commit(s) behind {upstream_ref} (ahead {ahead})"
    if ahead > 0:
        return "up_to_date", f"up to date with {upstream_ref}; local branch has {ahead} unpushed commit(s)"
    return "up_to_date", f"up to date with {upstream_ref}"


def _env_type_label() -> str:
    """Return a human-readable label for the active environment type."""
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        env_name = os.environ.get("CONDA_DEFAULT_ENV", "")
        return f"conda ({env_name})" if env_name else f"conda ({conda_prefix})"
    if (Path(sys.prefix) / "conda-meta").is_dir():
        return f"conda ({sys.prefix})"
    if os.environ.get("VIRTUAL_ENV"):
        return "venv"
    base_prefix = getattr(import_sys := __import__("sys"), "base_prefix", import_sys.prefix)
    if import_sys.prefix != base_prefix:
        return "venv (non-standard)"
    return "none"


def _conda_runtime_active() -> bool:
    if os.environ.get("CONDA_PREFIX"):
        return True
    return (Path(sys.prefix) / "conda-meta").is_dir()


def build_setup_status_summary(status: SetupWizardStatus) -> str:
    env_label = _env_type_label()
    conda_active = _conda_runtime_active()
    lines = [
        f"[{'PASS' if status.virtual_env_active else 'FAIL'}] Environment active: {status.virtual_env_active} ({env_label})",
        f"[{'PASS' if status.lerobot_import_ok else 'FAIL'}] Python module: lerobot ({status.lerobot_import_detail})",
        f"[{'PASS' if status.lerobot_dir_exists else 'WARN'}] LeRobot folder: {status.lerobot_dir}",
    ]
    if not conda_active:
        lines.append(f"[{'PASS' if status.venv_dir_exists else 'WARN'}] Expected venv folder: {status.venv_dir}")
    if status.app_update_state == "update_available":
        lines.append(f"[WARN] GUI wrapper updates: {status.app_update_detail}")
        lines.append(
            "[ACTION] Update available. Would you like to update and restart now? "
            "Use 'Update and Restart' in First-Time Setup Wizard."
        )
    elif status.app_update_state == "up_to_date":
        lines.append(f"[PASS] GUI wrapper updates: {status.app_update_detail}")
    else:
        lines.append(f"[WARN] GUI wrapper updates: {status.app_update_detail} (non-blocking)")
    lines.append(f"[INFO] Python executable: {status.python_executable}")
    if status.ready:
        lines.append("[READY] Environment looks good for LeRobot record/deploy.")
    elif not status.virtual_env_active and status.lerobot_import_ok:
        lines.append(
            "[ACTION] LeRobot imports, but no active virtual/conda environment is detected. "
            "Activate your environment and relaunch the GUI."
        )
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
    conda_active = _conda_runtime_active()
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
    elif not status.virtual_env_active and status.lerobot_import_ok:
        lines.extend(
            [
                "LeRobot is importable, but this launch is not inside an active environment.",
                "Activate your intended environment first, then relaunch the app.",
                "",
                "Suggested command:",
                f"  source {status.venv_dir / 'bin' / 'activate'}",
                "",
                "If you use conda instead:",
                "  conda activate <your-env-name>",
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
