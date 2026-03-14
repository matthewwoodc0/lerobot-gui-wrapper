# LeRobot GUI Wrapper

Qt desktop wrapper for local LeRobot record, replay, deploy, teleop, motor setup, train, workflows, experiments, config, history, and visualizer tooling.

Recent hardware-ops hardening:
- workflow state persists to `runs_dir/workflow_state.json` and restart recovery marks in-flight work as `interrupted`
- replay now prefers discovered local episodes, keeps a manual fallback, and shows readiness guidance before launch
- motor setup now reports exactly what changed and when rig snapshots should be refreshed

Recent UI polish:
- the shell now uses tighter card spacing and less rounded surfaces
- the workspace header shows a compact Hugging Face auth status with tooltip detail

## What This Project Is

This app sits on top of an existing LeRobot checkout or install. It does not replace LeRobot. It gives you a local GUI shell for:

- hardware bring-up and diagnostics
- record, replay, teleop, and deploy/eval workflows
- train, sim-eval, and experiment comparison
- config management, named rigs, and portable profiles
- run history, lineage, and support-bundle export

Validated tracks:

- current: LeRobot `0.5.x`
- N-1: LeRobot `0.4.x`

Recommended wrapper runtime:

- Python `3.12+`

## Quick Start

If you already have a working LeRobot environment:

```bash
git clone <this-repo>
cd <this-repo>
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[qt]"
pip install opencv-python pyyaml
python robot_pipeline.py gui
```

Then open `Config` and point `lerobot_dir` and `lerobot_venv_dir` at your existing LeRobot checkout/runtime.

Prompt yourself during setup:
- login with Hugging Face?
  - if yes, run `hf auth login` in Terminal and then set `hf_username` in `Config`
  - if no, skip it and continue with local-only workflows

## Install Guide

### Option A: You already have LeRobot

Use this when LeRobot already runs correctly in a shell and you only need the wrapper.

1. Create a separate wrapper env or use your existing GUI-friendly env.
2. Install the wrapper plus Qt.
3. Install OpenCV for camera previews.
4. Launch the GUI.
5. In `Config`, set:
   - `lerobot_dir`
   - `lerobot_venv_dir`
   - robot ports, IDs, and paths

Example:

```bash
git clone <this-repo>
cd <this-repo>
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[qt]"
pip install opencv-python pyyaml
python robot_pipeline.py gui
```

Notes:

- The GUI runtime and the LeRobot runtime do not have to be the same interpreter.
- Generated LeRobot commands and compatibility probes use the configured `lerobot_venv_dir` interpreter when it exists.

### Option B: You need both LeRobot and the wrapper

Use this when you are setting up a new machine from scratch.

1. Install LeRobot first and verify it runs in a terminal.
2. Create either:
   - one shared env for both LeRobot and the wrapper, or
   - a dedicated wrapper env plus a dedicated LeRobot env
3. Install this wrapper.
4. Launch the GUI.
5. Use the first-time setup flow below.

The wrapper assumes LeRobot itself is already installable and working on the target machine. This README does not duplicate upstream LeRobot install steps because those can change independently of this repo. What matters here is:

- LeRobot import works in the runtime you point `lerobot_venv_dir` at
- the relevant LeRobot entrypoints are available
- the wrapper env can import `PySide6`
- the wrapper env can import `cv2`

## Platform Notes

### Linux

- Prefer `/dev/serial/by-id/...` robot paths when available.
- You may need serial permissions such as membership in the `dialout` group.
- Multi-camera labs are a primary validation target.
- Shared-machine note: the launcher now sanitizes Qt plugin paths in user space so PySide6 does not pick up OpenCV's `cv2/qt` plugins by mistake.
- If you still hit a Qt `xcb` bootstrap error in your wrapper env, prefer `opencv-python-headless` over `opencv-python` for the wrapper runtime and relaunch with `python robot_pipeline.py gui`.

### macOS

- Expect robot devices like `/dev/tty.*` or `/dev/cu.*`.
- macOS is a validated hardware target for one-camera and two-camera lab setups.

## Dependency Checks

The wrapper package metadata is intentionally minimal, so treat these as the practical runtime requirements:

- Python `3.12+`
- `PySide6`
- `opencv-python` or `opencv-python-headless` for `cv2`
- `pyyaml` if you want full YAML profile import/export
- a working LeRobot runtime in the configured `lerobot_venv_dir`

Quick checks:

```bash
python -c "import PySide6; print('PySide6 OK')"
python -c "import cv2; print('cv2 OK')"
python -c "import yaml; print('PyYAML OK')"
python -c "import robot_pipeline_app; print('wrapper OK')"
```

If you want dev tooling too:

```bash
pip install -e ".[qt,dev]"
```

## Launch

```bash
python robot_pipeline.py gui
```

You can also use `python -m robot_pipeline gui`.

`gui` and `gui-qt` both launch the same Qt application.

## First-Time Setup

Recommended order on a new machine, new robot pair, or after USB/calibration changes:

1. Launch the app from a Python `3.12+` environment with wrapper dependencies installed.
2. Open `Config`.
3. Click `Run Setup Check`.
4. Fix any `FAIL` items.
5. Save the core machine defaults:
   - `lerobot_dir`
   - `lerobot_venv_dir`
   - `follower_port`
   - `leader_port`
   - `follower_robot_id`
   - `leader_robot_id`
   - camera config
   - `record_data_dir`
   - `deploy_data_dir`
   - `trained_models_dir`
6. Click `Run Doctor`.
7. Fix any remaining `FAIL` items.
8. Use `Teleop` as the first real hardware test.
9. Verify cameras separately in `Record` or `Deploy`.
10. Do one short record before trying a real deploy/eval run.

Readiness checklist:

- `Run Setup Check` has no unresolved `FAIL`
- `Run Doctor` has no unresolved `FAIL`
- `Teleop` starts successfully
- calibration files match the intended robot IDs
- camera preview is correct
- one short record completes successfully

## Compatibility

Current policy:

| Track | Status | Notes |
| --- | --- | --- |
| `0.5.x` | validated current track | primary validation target |
| `0.4.x` | validated N-1 track | supported with entrypoint/flag fallback logic |

Important behavior:

- compatibility is probe-driven, not hardcoded to one LeRobot build
- replay, sim-eval, resume, and rename-map behavior depend on detected upstream entrypoints and `--help` flags
- CI verifies probe/tooling behavior, not full hardware PASS status
- workflow PASS still requires real hardware validation

Useful commands:

```bash
python3 robot_pipeline.py doctor
python3 robot_pipeline.py compat
python3 robot_pipeline.py compat --json
python3 robot_pipeline.py doctor --json
```

If you are outside the validated tracks:

1. Run `doctor`.
2. Run `compat`.
3. Use command preview.
4. Compare generated flags against your installed LeRobot `--help` output before running hardware workflows.

## Current UI Surface

- `Record`
- `Replay`
- `Deploy`
- `Teleop`
- `Motor Setup`
- `Train`
- `Workflows`
- `Experiments`
- `Config`
- `Visualizer`
- `History`

## Operational Notes

- If Qt imports fail, verify the active environment can import `PySide6`.
- On Linux, the app strips OpenCV Qt plugin paths before startup so shared-machine wrapper envs do not depend on `sudo` fixes just to launch the GUI.
- If you still see a Qt `xcb` plugin error on Linux, switch the wrapper env from `opencv-python` to `opencv-python-headless` and relaunch with `python robot_pipeline.py gui`.
- If camera previews or video tiles fail, verify the active environment can import `cv2`.
- `Visualizer` is the research workspace surface: local assets plus HF browse/search, dataset QA, in-context compatibility warnings, sync-to-local roots, and lineage links.
- `History` exposes lineage links and compatibility context alongside rerun and deploy-note editing.
- `History` and `Visualizer` can launch hardware replay directly against local dataset episodes when the configured LeRobot runtime exposes a replay entrypoint.
- `Config` supports GUI profile import/export, portable lab presets, and named rig save/switch flows.
- `Motor Setup` handles first-time servo bring-up from the GUI with port scan, command review, preflight, live output, and result logging.
- `Workflows` stays intentionally local and sequential: `record -> upload`, `train -> sim-eval`, and `train -> deploy eval`.
- Replay and motor setup remain upstream-entrypoint-driven. If the configured LeRobot runtime does not expose those commands, the UI explains that clearly instead of inventing a second hardware backend.
- Training resume is only accepted when the detected train entrypoint exposes a real checkpoint/config-path flag.
- `Experiments` turns saved train, deploy, and sim-eval runs into one comparison surface with checkpoint discovery, parsed metrics, deploy/sim-eval handoff, and optional WandB links.
- Simulation eval is compatibility-driven and only enabled when the target runtime exposes a supported eval entrypoint and matching flags.

## Guides

Start here:

- [Resources Index](/Users/matthewwoodcock/Desktop/Projects/LeRobot%20GUI%20Wrapper/Resources/resources-index.md)
- [First-Time Setup](/Users/matthewwoodcock/Desktop/Projects/LeRobot%20GUI%20Wrapper/Resources/first-time-setup.md)
- [Compatibility Matrix](/Users/matthewwoodcock/Desktop/Projects/LeRobot%20GUI%20Wrapper/Resources/compatibility-matrix.md)
- [Hardware Operations Guide](/Users/matthewwoodcock/Desktop/Projects/LeRobot%20GUI%20Wrapper/Resources/hardware-operations-guide.md)

Workflow guides:

- [Config Tab Guide](/Users/matthewwoodcock/Desktop/Projects/LeRobot%20GUI%20Wrapper/Resources/config-tab-guide.md)
- [Record Tab Guide](/Users/matthewwoodcock/Desktop/Projects/LeRobot%20GUI%20Wrapper/Resources/record-tab-guide.md)
- [Deploy Tab Guide](/Users/matthewwoodcock/Desktop/Projects/LeRobot%20GUI%20Wrapper/Resources/deploy-tab-guide.md)
- [Teleop Tab Guide](/Users/matthewwoodcock/Desktop/Projects/LeRobot%20GUI%20Wrapper/Resources/teleop-tab-guide.md)
- [Training Tab Guide](/Users/matthewwoodcock/Desktop/Projects/LeRobot%20GUI%20Wrapper/Resources/training-tab-guide.md)
- [Experiments Tab Guide](/Users/matthewwoodcock/Desktop/Projects/LeRobot%20GUI%20Wrapper/Resources/experiments-tab-guide.md)
- [History Tab Guide](/Users/matthewwoodcock/Desktop/Projects/LeRobot%20GUI%20Wrapper/Resources/history-tab-guide.md)

Reference guides:

- [Community Profiles](/Users/matthewwoodcock/Desktop/Projects/LeRobot%20GUI%20Wrapper/Resources/community-profiles.md)
- [Error Catalog](/Users/matthewwoodcock/Desktop/Projects/LeRobot%20GUI%20Wrapper/Resources/error-catalog.md)
- [Support Bundle Guide](/Users/matthewwoodcock/Desktop/Projects/LeRobot%20GUI%20Wrapper/Resources/support-bundle.md)
- [Upstream Bridge Guide](/Users/matthewwoodcock/Desktop/Projects/LeRobot%20GUI%20Wrapper/Resources/upstream-bridge.md)
- [GA Validation Guide](/Users/matthewwoodcock/Desktop/Projects/LeRobot%20GUI%20Wrapper/Resources/ga-validation.md)
- [Qt UI Layout and Style Standard](/Users/matthewwoodcock/Desktop/Projects/LeRobot%20GUI%20Wrapper/Resources/ui-layout-style-standard.md)
