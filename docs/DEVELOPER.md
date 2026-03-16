# Developer Guide

This guide is for contributors and agents working on the LeRobot Pipeline Manager codebase. For user documentation see [README.md](../README.md) and [Resources/](../Resources/).

The machine-readable developer contract (subsystems, update matrix, validation commands) lives in [AGENTS.md](../AGENTS.md). This guide provides the architectural context that AGENTS.md presumes.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Key Subsystems](#key-subsystems)
3. [Adding a New Workflow Page](#adding-a-new-workflow-page)
4. [Running Tests](#running-tests)
5. [Code Conventions](#code-conventions)
6. [Definition of Done](#definition-of-done)

---

## Architecture Overview

The app is a PySide6 desktop application that wraps your existing LeRobot installation. It never touches the LeRobot source — it builds commands, runs them in a subprocess, streams their output, and persists the results.

```
robot_pipeline.py               # legacy shim / entry point
robot_pipeline_app/
  gui_qt_app.py                 # QApplication setup, section definitions, theme
  gui_qt_core_ops.py            # dispatcher: routes sidebar nav to workflow pages
  gui_qt_secondary_pages.py     # Config, History, Visualizer, Experiments pages
  gui_qt_runner.py              # Qt bridge to run_controller_service
  run_controller_service.py     # unified run execution lifecycle
  runner.py                     # subprocess, PTY, streaming
  workflow_queue.py             # sequential local recipe queue
  commands.py                   # all command builders
  gui_forms.py                  # form-to-request builders (RecordRequest, etc.)
  config_store.py               # load/save ~/.robot_config.json
  artifacts.py                  # run metadata, lineage, provenance
  compat.py                     # LeRobot version detection and fallback
  checks.py                     # preflight check framework
```

### Data flow for a typical run

1. User fills out a form (e.g. Record tab).
2. `gui_forms.py` validates inputs and builds a typed request (`RecordRequest`).
3. `commands.py` converts the request into a shell command string.
4. `run_controller_service.py` manages the lifecycle: preflight → confirm → execute → artifact write.
5. `runner.py` spawns the subprocess in `lerobot_dir`, streams output to the terminal widget.
6. On completion, `artifacts.py` writes run metadata to `runs_dir`.
7. History and Experiments tabs read from the artifact store.

### Config

Config is a flat JSON dictionary at `~/.robot_config.json`. `config_store.py` handles all load/save/normalize logic. The 47 config keys are defined in `constants.py` with defaults. Config is loaded once at startup and persisted on demand — there is no reactive config binding.

### Compatibility

`compat.py` detects which LeRobot entrypoints and flags are available by probing imports and running `--help` in the configured `lerobot_venv_dir`. Results are cached in `compat_snapshot.py` with a short TTL. All command builders in `commands.py` call through compat to select the right entrypoint and flag set.

---

## Key Subsystems

### Command building

**Files:** `commands.py`, `gui_forms.py`, `hardware_replay.py`, `hardware_motor_setup.py`

`gui_forms.py` validates UI state and builds typed request objects (`RecordRequest`, `DeployRequest`, etc.). `commands.py` converts those requests into runnable shell commands, using compat probes to select the right entrypoint and flags for the installed LeRobot version.

Rule: never build a shell command string anywhere except `commands.py`. GUI files call `build_*_request_and_command()` and treat the result as opaque.

### Run execution

**Files:** `runner.py`, `run_controller_service.py`, `workflow_queue.py`

`runner.py` spawns subprocesses using a PTY for terminal-compatible output streaming. It handles resize events and clean cancellation.

`run_controller_service.py` orchestrates the full run lifecycle: preflight, user confirmation, execution, artifact writing, and result callbacks.

`workflow_queue.py` runs sequential recipes (e.g. Record → Upload). Each step uses the normal run controller. Queue state is persisted at `runs_dir/workflow_state.json` and survives restarts. Interrupted steps never auto-resume — the user must explicitly retry.

### Preflight checks

**Files:** `checks.py`, `checks_record.py`, `checks_deploy.py`, `checks_teleop.py`, `checks_common.py`, `checks_calibration.py`, `checks_train.py`

The check framework in `checks.py` defines `DiagnosticEvent` (level, code, name, detail, fix, quick_action_id). Each workflow module defines its own check set. `PreflightReport` aggregates pass/warn/fail counts.

Quick actions are small callables attached to a `DiagnosticEvent` that can auto-fix the issue (e.g. prepend `eval_` to a dataset name). They are surfaced in the preflight Fix Center dialog.

### Artifacts and history

**Files:** `artifacts.py`, `history_utils.py`, `workspace_lineage.py`, `workspace_provenance.py`

Every completed run writes a JSON artifact to `runs_dir/<run_id>/`. Metadata keys are defined in `artifacts.py`. Lineage links each run to its source dataset, model, and downstream artifacts. The History tab reads artifacts directly from disk — there is no database.

Metadata conventions (from AGENTS.md):
- Replay metadata must carry `dataset_repo_id`, resolved `dataset_path`, and `replay_episode`
- Queue metadata must carry queue id, recipe type, step index, step label, and prior run linkage
- Rig metadata stays in named snapshots only; UI-only keys never belong in snapshots

### Compatibility and probes

**Files:** `compat.py`, `compat_snapshot.py`, `workspace_compatibility.py`, `probes.py`

`compat.py` detects LeRobot capabilities by probing the configured venv. Results are cached with a short TTL via `compat_snapshot.py`.

`probes.py` handles runtime hardware probes (camera availability, serial ports). These run on-demand, not at startup.

`workspace_compatibility.py` checks whether the current workspace state (installed packages, LeRobot version) is compatible with the wrapper's expectations and emits warnings when the user is on an untested track.

---

## Adding a New Workflow Page

1. **Create the page class** in a new `gui_qt_<name>.py` module. Subclass `QWidget`. Expose a `build_panel()` or `__init__(parent, config, ...)` constructor that matches the pattern in existing pages.

2. **Register it in `gui_qt_app.py`** by adding a `QtSectionDefinition` entry with `id`, `title`, `subtitle`, and `stage` fields.

3. **Wire it in `gui_qt_core_ops.py`** — add a branch in the section dispatcher that maps the section id to your new panel class.

4. **Add command building in `commands.py`** if the page launches a LeRobot command. Define a typed request in `types.py` first, then a `build_<workflow>_request_and_command()` function.

5. **Add preflight checks** in a new `checks_<workflow>.py` module. Import and run them through the shared preflight framework.

6. **Write tests** covering at minimum:
   - panel construction doesn't raise
   - command building produces expected flag set
   - preflight passes on valid input and fails on invalid
   - artifacts contain required metadata keys

7. **Update docs** per the matrix in AGENTS.md.

---

## Running Tests

The project uses pytest with pytest-qt.

```bash
# Full suite
.venv/bin/python -m pytest

# Specific subsystems
.venv/bin/python -m pytest tests/test_workflow_queue.py -q
.venv/bin/python -m pytest tests/test_hardware_workflows.py tests/test_gui_qt_core_ops.py -q
.venv/bin/python -m pytest tests/test_rig_manager.py tests/test_profile_io.py -q
.venv/bin/python -m pytest tests/test_compat.py tests/test_run_controller_service.py -q
```

Tests that require a display (Qt widget tests) use the pytest-qt `qtbot` fixture. They run headlessly on CI via `QT_QPA_PLATFORM=offscreen`.

---

## Code Conventions

- **Python 3.12+**, typed where practical.
- **Line length:** 120 characters (ruff).
- **No emojis** in code, log output, or UI text.
- Use `python3` (not `python`) in all shell invocations.
- Never build shell command strings outside `commands.py`.
- Never import from `robot_pipeline.py` — that file is a shim for external callers. Import from `robot_pipeline_app.*` directly.
- Config keys: use `snake_case` strings. Never invent new keys without adding them to `constants.py` with a default.
- Qt thread safety: all UI updates must happen on the main thread. Use signals or `QTimer.singleShot` to schedule updates from background threads.

---

## Definition of Done

From AGENTS.md — a change is done when:

1. Existing external page constructors and launch entry points still work unchanged.
2. New or changed behavior has targeted tests.
3. Required docs from the update matrix are updated.
4. Validation commands were run and pass.
