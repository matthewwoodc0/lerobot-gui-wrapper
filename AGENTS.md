# AGENTS.md

Developer contract for this repo. Keep it short, current, and complementary to [`README.md`](/Users/matthewwoodcock/Desktop/Projects/LeRobot%20GUI%20Wrapper/README.md).

## Subsystems

- Command building: `robot_pipeline_app/commands.py`, `robot_pipeline_app/gui_forms.py`, `robot_pipeline_app/hardware_replay.py`, `robot_pipeline_app/hardware_motor_setup.py`
- Run execution and lifecycle: `robot_pipeline_app/runner.py`, `robot_pipeline_app/run_controller_service.py`, `robot_pipeline_app/workflow_queue.py`
- Queue recipes and recovery state: `robot_pipeline_app/workflow_queue_recipes.py`, `robot_pipeline_app/workflow_queue_models.py`, `robot_pipeline_app/workflow_queue.py`
- Artifacts and history: `robot_pipeline_app/artifacts.py`, `robot_pipeline_app/history_utils.py`, `robot_pipeline_app/gui_qt_history_page.py`
- Compatibility and probes: `robot_pipeline_app/compat.py`, `robot_pipeline_app/compat_snapshot.py`, `robot_pipeline_app/workspace_compatibility.py`
- Profile and rig IO: `robot_pipeline_app/profile_io.py`, `robot_pipeline_app/rig_manager.py`, `robot_pipeline_app/gui_qt_config_page.py`
- Docs and guides: `README.md`, `Resources/*.md`, `docs/community-profiles.md`

## Docs Update Matrix

- Replay, motor setup, queue, rig workflows: update `Resources/hardware-operations-guide.md`
- Config, profiles, rigs: update `Resources/config-tab-guide.md` and `docs/community-profiles.md`
- History or visualizer launch behavior: update `Resources/history-tab-guide.md`
- User-visible surface-area changes: update `README.md`

## When You Touch X, Update Y

- Command flag changes: update the matching preview/preflight tests and the relevant guide page
- Queue state, restart behavior, or recipe sequencing: update `tests/test_workflow_queue.py` and rerun full `pytest`
- Artifact metadata keys, lineage, or provenance: update `tests/test_artifacts.py`, `tests/test_run_controller_service.py`, and history docs
- Rig snapshot rules or config persistence: update `tests/test_rig_manager.py`, `tests/test_profile_io.py`, and config/profile docs
- Replay or motor setup UX: update `tests/test_hardware_workflows.py`, `tests/test_gui_qt_core_ops.py`, and the hardware/history guides

## Validation

- Queue or run-controller changes: `.venv/bin/python -m pytest`
- Replay and motor setup changes: `.venv/bin/python -m pytest tests/test_hardware_workflows.py tests/test_gui_qt_core_ops.py -q`
- Queue changes: `.venv/bin/python -m pytest tests/test_workflow_queue.py -q`
- Rig/profile changes: `.venv/bin/python -m pytest tests/test_rig_manager.py tests/test_profile_io.py -q`
- Compatibility changes: `.venv/bin/python -m pytest tests/test_compat.py tests/test_run_controller_service.py -q`

## Extraction Rule

- If a helper module is already broad and you add a second unrelated responsibility, extract first and keep compatibility wrappers thin.

## Metadata Conventions

- Replay metadata must carry `dataset_repo_id`, resolved `dataset_path` when known, and `replay_episode`
- Queue metadata must carry queue id, recipe type, step index, step label, and prior run linkage when available
- Rig metadata should stay in named snapshots only; UI-only keys never belong in snapshots
- Provenance and lineage metadata should prefer stable path/id fields over UI labels

## Definition Of Done

- Existing external page constructors and launch entry points still work unchanged
- New or changed behavior has targeted tests
- Required docs from the matrix are updated
- Validation commands were run and recorded in the final handoff
