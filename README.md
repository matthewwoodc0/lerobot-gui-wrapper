# LeRobot Pipeline Manager

A local desktop control center for LeRobot hardware bring-up, teleoperation, recording, training, deployment, and experiment review.

[![Version](https://img.shields.io/badge/version-0.1.0-6f5cff)](https://github.com/matthewwoodc0/lerobot-gui-wrapper)
[![Python](https://img.shields.io/badge/python-3.12%2B-3776ab)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-2ea44f)](LICENSE)

![LeRobot Pipeline Manager record workspace](docs/media/pipeline-manager-record.png)

The app keeps LeRobot as the execution layer. It adds saved rig profiles, preflight checks, recoverable workflow queues, run history, artifact lineage, support bundles, and one interface for the full local workflow.

![Config to teleoperation and recording demo](docs/media/pipeline-manager-demo.gif)

The demo uses a safe example configuration. It does not start a robot command.

## Why I built this

LeRobot has a strong command-line workflow, but repeated lab work also needs configuration recall, clear preflight failures, run recovery, and a record of what produced each artifact. I built LeRobot Pipeline Manager to make those operator tasks visible and repeatable without hiding the commands that run underneath.

## Where it fits now

[LeLab](https://huggingface.co/docs/lerobot/lelab) is Hugging Face's first-party GUI and is the best default for SO-ARM101 onboarding. LeRobot Pipeline Manager is a local-first research and operations companion when you need:

- repeatable rig configuration
- preflight diagnostics and Doctor
- safe command preview
- recoverable workflow queues
- hardware replay
- experiment comparison, run history, and artifact lineage
- support bundles

This `0.1.0` release is a **software-validated release candidate**. Automated tests cover LeRobot `0.6.x` (current) and `0.5.x` (N-1). Real-robot PASS still requires the manual hardware gate in `Resources/ga-validation.md`.

## Built with Codex

I used Codex as an engineering collaborator for architecture reviews, implementation, refactoring, test coverage, and documentation. I set the product direction, reviewed the changes, and validated the workflows. Codex is a development tool for this repository; it is not a runtime dependency.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation and dependency policy](#installation-and-dependency-policy)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Platform Setup](#platform-setup)
6. [Your First Session](#your-first-session)
7. [Tabs at a Glance](#tabs-at-a-glance)
8. [Resources and Guides](#resources-and-guides)
9. [Getting Help](#getting-help)
10. [License](#license)

---

## Quick Start

Copy these commands for a first working install.

```bash
# 1) Create and activate a Python 3.12 environment
conda create -n lerobot python=3.12 -y
conda activate lerobot

# 2) Install current LeRobot with common operator extras
pip install 'lerobot[core_scripts,training,feetech]'
# core_scripts = record/replay/calibrate/teleoperate + dataset + hardware + viz
# training     = train policies
# feetech      = SO-101 motor support (omit if you do not use Feetech)

# 3) Install this package (editable for development, or wheel for users)
pip install -e ".[gui]"
# or: pip install "lerobot-gui-wrapper[gui]"

# 4) Launch the desktop app
lerobot-pipeline-manager gui
```

Then in the app:

1. Open **Config** and set rig ports, robot IDs, and paths.
2. Click **Save Config**.
3. Run **Doctor** until FAIL count is zero.

Useful CLI commands after install:

```bash
lerobot-pipeline-manager doctor
lerobot-pipeline-manager compat
lerobot-pipeline-manager support-bundle --run-id latest --output ~/Desktop/support.zip
lerobot-pipeline-manager install-launcher
```

Developer source fallback (from a git checkout only):

```bash
python3 robot_pipeline.py gui
# or
python3 -m robot_pipeline_app gui
```

## Installation and dependency policy

The base package has no required runtime dependencies. LeRobot environments can supply their own Qt and OpenCV builds. For normal desktop use, install the `gui` extra:

```bash
pip install -e ".[gui]"
# or: pip install "lerobot-gui-wrapper[gui]"
```

Extras:

- `.[qt]` installs PySide6
- `.[media]` installs headless OpenCV
- `.[gui]` installs both
- `.[dev]` installs test and lint tools

Prefer `opencv-python-headless` over `opencv-python`. This reduces Qt plugin conflicts.

### Software-validated LeRobot versions

| Track | Version | Evidence |
|---|---|---|
| Current | `0.6.x` | Automated compat smoke + unit/integration tests |
| N-1 | `0.5.x` | Automated compat smoke + unit/integration tests |

Hardware evidence is separate. See `Resources/ga-validation.md`.

Deploy path policy:

- Prefer `lerobot-rollout` when the installed runtime provides it (LeRobot 0.6+)
- Fall back to record with `--policy.path` on supported older runtimes

---

## Prerequisites

| Requirement | Notes |
|---|---|
| LeRobot `0.6.x` or `0.5.x` | Install with the extras model above |
| Python 3.12+ | Required by this package and current LeRobot |
| SO-101 arms or compatible hardware | Optional for software-only use; required for hardware gate |
| `ffmpeg` | Recommended for recording; `conda install ffmpeg -y` |
| Serial port access | On Linux: membership in the `dialout` group |

---

## Installation

### Step 1 — Install LeRobot (current)

```bash
conda create -n lerobot python=3.12 -y
conda activate lerobot
pip install 'lerobot[core_scripts,training,feetech]'
conda install ffmpeg -y
pip show lerobot | grep Version
```

### Step 2 — Install LeRobot Pipeline Manager

From a source checkout:

```bash
git clone https://github.com/matthewwoodc0/lerobot-gui-wrapper.git
cd lerobot-gui-wrapper
pip install -e ".[gui]"
lerobot-pipeline-manager gui
```

From a built wheel:

```bash
pip install "path/to/lerobot_gui_wrapper-*.whl[gui]"
lerobot-pipeline-manager gui
```
---

## Platform Setup

### macOS

No extra steps needed. Robot devices appear as `/dev/tty.*` or `/dev/cu.*`. Launch with:

```bash
conda activate lerobot
python3 robot_pipeline.py gui
```

---

### Linux

**Serial port permissions** — add your user to `dialout` and log out/back in:

```bash
sudo usermod -aG dialout $USER
# then log out and back in
```

**If the GUI fails to start with an `xcb-cursor` error:**

```bash
conda install -c conda-forge xcb-util-cursor -y
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

Make that export permanent so you never see it again:

```bash
echo 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc
source ~/.bashrc
```

Launch:

```bash
python3 robot_pipeline.py gui
```

If xcb errors persist, try the Wayland backend:

```bash
QT_QPA_PLATFORM=wayland python3 robot_pipeline.py gui
```

**Stable device paths on Linux** — Use `/dev/serial/by-id/...` paths instead of `/dev/ttyACM0` etc. These stay constant across reboots even when USB port order changes:

```bash
ls /dev/serial/by-id/
```

---

### Linux — Shared or Lab Machine

Everything above applies, plus:

- The wrapper sanitizes Qt plugin paths at startup to avoid conflicts with other users' OpenCV environments — no `sudo` fix required just to launch.
- Add `LD_LIBRARY_PATH` to `~/.bashrc` (not system-wide).
- Prefer `opencv-python-headless` to avoid plugin conflicts with other users' envs.

---

## Your First Session

A complete zero-to-working walkthrough for a new machine or a new robot pair. Budget about 10–15 minutes.

### 1. Launch and open Config

```bash
conda activate lerobot
python3 robot_pipeline.py gui
```

Open the **Config** tab.

### 2. Set your paths

Fill in at minimum:

| Field | What to set |
|---|---|
| `lerobot_dir` | Path to your LeRobot checkout, e.g. `~/lerobot` |
| `lerobot_venv_dir` | Auto-fills after you set `lerobot_dir` — or set manually |
| `follower_port` | Serial device for the follower arm |
| `leader_port` | Serial device for the leader arm |
| `follower_robot_id` | e.g. `red4` |
| `leader_robot_id` | e.g. `white` |
| `camera_laptop_index` | USB camera index, typically `0` or `2` |
| `camera_phone_index` | Second camera index if present |
| `record_data_dir` | Where recorded datasets are saved locally |
| `trained_models_dir` | Where trained model folders live |
| `name_iteration_policy` | How Record, Training, Deploy, Workflows, Experiments checkpoint deploy, and History reruns handle name collisions: `manual`, `auto`, or `always` |

Click **Apply Path Defaults** to auto-fill derivative paths from `lerobot_dir`, then click **Save Config**.

`name_iteration_policy` defaults to `auto`:
- `manual` keeps typed names unchanged and surfaces collisions for you to fix manually
- `auto` advances only auto-managed names
- `always` advances colliding names even after you type them explicitly

### 3. Run diagnostics

Click **Run Setup Check**. Fix any `FAIL` items before continuing.

Click **Run Doctor**. This checks hardware, environment, and calibration. Resolve all `FAIL` items.

> **Common first-run failures:** missing `feetech-servo-sdk` (install it via pip), serial port permissions (see Linux section above), ffmpeg not found (install via conda).

### 4. Verify hardware with Teleop first

Open the **Teleop** tab. This is the fastest way to confirm the entire hardware chain is working — ports, IDs, and calibration — before you commit to a full record session.

1. Click **Scan Robot Ports**.
2. Confirm or assign follower/leader ports.
3. Set robot IDs if they don't match.
4. Click **Preview Command** to review the generated command.
5. Click **Run Teleop**.
6. In the Teleop Helper window, watch for the arms to respond. Move the leader arm — the follower should mirror it within a second.

If teleop starts but arms don't respond, a calibration prompt may be waiting in the log. See [Calibration Failures](Resources/first-time-setup.md#if-teleop-fails-on-calibration).

### 5. Verify cameras

Switch to the **Record** tab. Click **Scan Camera Ports**. Assign the correct roles (laptop/phone) to the detected indices. Click **Refresh Camera Preview** and confirm the views look right.

### 6. Record a short test dataset

Still in **Record**:

1. Leave the dataset name on its auto-managed value (e.g. `your_name_1`).
2. Set episodes to `2`, episode time to `10`, and a simple task description.
3. Click **Run Record** and complete two short episodes.

If this completes cleanly, your full stack is working.

### 7. You're ready

The machine is ready for normal use once:

- `Run Setup Check` shows no unresolved `FAIL`
- `Run Doctor` shows no unresolved `FAIL`
- Teleop starts and arms respond
- Camera preview is correct
- One short record run completes

---

## Tabs at a Glance

| Tab | Purpose | When to use |
|---|---|---|
| **Config** | Paths, hardware defaults, named rigs, diagnostics | First-time setup; any time environment or hardware changes |
| **Teleop** | Live teleoperation with the leader arm | Hardware bring-up verification; free-form exploration |
| **Record** | Teleoperated dataset collection + HF upload | Building training datasets |
| **Deploy** | Evaluate a trained model on hardware | Testing a policy after training |
| **Training** | HIL adaptation from a base model + dataset | Fine-tuning with human intervention |
| **Experiments** | Cross-run comparison of train/deploy/sim-eval | Picking the best checkpoint; tracking progress over runs |
| **History** | Full run log with outcome annotation | Reviewing what worked; annotating episode successes |
| **Replay** | Replay recorded episodes on hardware with local/HF dataset picking | Verifying data quality |
| **Motor Setup** | First-time servo bring-up | New robot or new port assignment |
| **Workflows** | Sequential multi-step recipes | Record → Upload, Train → Deploy in one queue |
| **Visualizer** | Browse datasets, videos, and model metadata | Inspecting recorded data; navigating model folders |

Name iteration is controlled globally from **Config**. The same `manual` / `auto` / `always` policy is applied across Record, Training, Deploy, queue launches, checkpoint deploys from Experiments, and deploy reruns from History.

---

## Resources and Guides

Full documentation lives in the [`Resources/`](Resources/) folder. Start here based on what you need:

### Getting started
- [First-Time Setup Walkthrough](Resources/first-time-setup.md) — new machine, new robot pair, or after USB changes
- [Compatibility Matrix](Resources/compatibility-matrix.md) — which LeRobot versions work with which wrapper features

### Core workflow guides
- [Teleop Guide](Resources/teleop-tab-guide.md) — command options, preflight, Teleop Helper
- [Record Guide](Resources/record-tab-guide.md) — dataset naming, camera setup, HF upload, troubleshooting
- [Deploy Guide](Resources/deploy-tab-guide.md) — model selection, eval naming, preflight, results
- [Training Guide](Resources/training-tab-guide.md) — HIL adaptation, srun wrapping, checkpoint discovery
- [Experiments Guide](Resources/experiments-tab-guide.md) — cross-run comparison, WandB integration
- [History Guide](Resources/history-tab-guide.md) — outcome annotation, replay launch, lineage

### Hardware and configuration
- [Hardware Operations Guide](Resources/hardware-operations-guide.md) — Replay, Motor Setup, named rigs, local workflows
- [Config Tab Guide](Resources/config-tab-guide.md) — all config fields, named rigs, community profiles, launcher

### Reference
- [Error Catalog](Resources/error-catalog.md) — preflight error codes and fixes
- [Community Profiles](Resources/community-profiles.md) — portable config sharing
- [Support Bundle Guide](Resources/support-bundle.md) — creating debug exports for bug reports
- [Upstream Bridge Guide](Resources/upstream-bridge.md) — how the wrapper integrates with LeRobot
- [Transition and Community Upgrade Plan](docs/transition-and-upgrade-plan.md) — current-runtime release gates and the community-tool roadmap

---

## Getting Help

**Something failing in preflight?** Check the [Error Catalog](Resources/error-catalog.md) — it maps every diagnostic code to an actionable fix.

**Unexpected behavior after an update?** Check the [Compatibility Matrix](Resources/compatibility-matrix.md) for known version-specific quirks.

**Need to file a bug?** Use **Export Support Bundle** in the Config tab to generate a redacted diagnostics archive, then open an issue at [github.com/matthewwoodc0/lerobot-gui-wrapper/issues](https://github.com/matthewwoodc0/lerobot-gui-wrapper/issues) and attach the bundle.

**Building on or contributing to this project?** See the [Developer Guide](docs/DEVELOPER.md).

## License

Licensed under the [Apache License 2.0](LICENSE).
