# LeRobot GUI Wrapper

A Qt desktop GUI for LeRobot record, teleop, deploy, train, and experiment workflows. Runs on top of your existing LeRobot install and does not replace it.

Validated tracks:

| LeRobot Version | Status |
| --- | --- |
| `0.5.x` | primary validated track |
| `0.4.x` | supported with entrypoint/flag fallback |

---

## Table of Contents

1. [What This Project Is](#what-this-project-is)
2. [Step 1 — Install LeRobot 0.5.0](#step-1--install-lerobot-050)
3. [Step 2 — Install the GUI Wrapper](#step-2--install-the-gui-wrapper)
4. [Platform Notes](#platform-notes)
5. [Launching the App](#launching-the-app)
6. [First-Time Setup](#first-time-setup)
7. [Feature Guide](#feature-guide)

---

## What This Project Is

This app sits on top of an existing LeRobot checkout. It gives you a local GUI for:

- hardware bring-up and diagnostics (motor setup, port scan, camera preview)
- record, replay, teleop, and deploy/eval workflows
- train, sim-eval, and experiment comparison
- config management, named rigs, and portable profiles
- run history, lineage, and support-bundle export

---

## Step 1 — Install LeRobot 0.5.0

These steps set up a fresh conda environment and install LeRobot. Run them in your terminal.

**If you have an old `lerobot` conda environment, remove it first:**

```bash
conda deactivate
conda remove -n lerobot --all -y
```

**Create a fresh environment:**

```bash
conda create -n lerobot python=3.12 -y
conda activate lerobot
```

**Install LeRobot (with feetech support for SO-101):**

```bash
cd ~/lerobot
pip install -e ".[feetech]"
```

> If you are not using SO-101 / feetech motors, use `pip install -e "."` instead.

**Verify the install:**

```bash
pip show lerobot | grep Version
```

**Install ffmpeg (required for dataset recording):**

```bash
conda install ffmpeg -y
```

---

## Step 2 — Install the GUI Wrapper

**Clone the repository:**

```bash
git clone <this-repo-url>
cd <this-repo-folder>
```

**Install the wrapper:**

```bash
pip install -e .
pip install opencv-python-headless
```

> Use `opencv-python-headless` instead of `opencv-python` to avoid Qt plugin conflicts on Linux. On macOS either works, but headless is still recommended for the wrapper environment.

---

## Platform Notes

### macOS

- Robot devices appear as `/dev/tty.*` or `/dev/cu.*`.
- No extra steps are needed for Qt startup on macOS.
- Both one-camera and two-camera lab setups are validated.
- Run the app with:

```bash
conda activate lerobot
python3 robot_pipeline.py gui
```

### Linux

- Prefer `/dev/serial/by-id/...` device paths when available — they stay stable across reboots.
- You may need serial port permissions. Add your user to the `dialout` group:

```bash
sudo usermod -aG dialout $USER
# then log out and back in
```

- If the GUI fails to start with an `xcb-cursor` error, install the missing library:

```bash
conda install -c conda-forge xcb-util-cursor -y
```

- Then export the conda library path so Qt can find it:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

- Make that export persistent so you do not have to type it every session:

```bash
echo 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc
```

- Then launch the app:

```bash
python3 robot_pipeline.py gui
```

- If the `xcb` error persists, try the Wayland backend:

```bash
QT_QPA_PLATFORM=wayland python3 robot_pipeline.py gui
```

### Shared Linux Machine (lab / server)

Same steps as Linux above, but note:

- The wrapper sanitizes Qt plugin paths at startup so PySide6 does not accidentally pick up OpenCV's bundled Qt plugins. No `sudo` fix is required just to launch the GUI.
- Use `opencv-python-headless` in your wrapper environment to avoid plugin conflicts with other users' envs.
- The `LD_LIBRARY_PATH` export from conda is especially important on shared machines where system libraries may be older.
- Prefer `~/.bashrc` (or `~/.bash_profile`) additions over system-wide changes.

---

## Launching the App

From inside the cloned wrapper folder, with the conda env active:

```bash
python3 robot_pipeline.py gui
```

You can also run:

```bash
python3 -m robot_pipeline gui
```

Both commands launch the same Qt application.

---

## First-Time Setup

Recommended order on a new machine, new robot pair, or after USB/calibration changes:

1. Launch the app.
2. Open the **Config** tab.
3. Set `lerobot_dir` to your LeRobot checkout path (e.g. `~/lerobot`).
4. Click **Run Setup Check**.
5. If setup is not ready, click **Open Setup Wizard** and follow the prompts.
6. Set these fields at minimum:
   - `lerobot_dir`
   - `lerobot_venv_dir`
   - `follower_port` and `leader_port`
   - `follower_robot_id` and `leader_robot_id`
   - camera indices
   - `record_data_dir`, `deploy_data_dir`, `trained_models_dir`
7. Click **Apply Path Defaults** to auto-fill paths from `lerobot_dir`.
8. Click **Save Config**.
9. Click **Run Doctor** and resolve any `FAIL` items.
10. Open **Teleop** as your first real hardware test — confirm arms respond before trying Record or Deploy.
11. Verify cameras in **Record** or **Deploy**.
12. Do one short record run before attempting a deploy/eval run.

**Readiness checklist before running hardware workflows:**

- `Run Setup Check` shows no unresolved `FAIL`
- `Run Doctor` shows no unresolved `FAIL`
- `Teleop` starts and arms respond
- Camera preview looks correct in Record or Deploy
- One short record run completes successfully

---

## Feature Guide

Each tab in the app corresponds to a specific workflow. Brief descriptions are below; full guides are linked in the [Resources Index](Resources/resources-index.md).

---

### Config

Control center for all paths, hardware defaults, named rigs, diagnostics, and setup.

- Set and persist all runtime config (`lerobot_dir`, ports, cameras, HF username, paths).
- Run the setup wizard and doctor diagnostics from one place.
- Save and switch **named rigs** — snapshots of your full hardware config — for fast switching between multiple robots on one machine.
- Import and export portable community profiles.
- Install or update the desktop launcher.

Full guide: [Config Tab Guide](Resources/config-tab-guide.md)

---

### Teleop

Lightweight tab for launching teleoperation with minimal setup.

- Set follower/leader ports and robot IDs.
- Preview the generated command before running.
- Hardware preflight runs before launch.
- The **Teleop Helper** shows elapsed session time, live console output, and a clean stop action.
- Use this tab first on a new machine — before Record or Deploy.

Full guide: [Teleop Tab Guide](Resources/teleop-tab-guide.md)

---

### Record

Teleoperated data collection with optional Hugging Face upload.

- Configure dataset name, episodes, episode time, and task description.
- Camera preview and port assignment built into the tab.
- Auto-managed dataset naming advances monotonically and detects local/HF collisions before launch.
- Optional post-record upload to Hugging Face, including v3.0 dataset conversion.
- Dataset browser shows both local and remote HF datasets in one panel.

Full guide: [Record Tab Guide](Resources/record-tab-guide.md)

---

### Deploy

Run a trained policy on hardware for evaluation.

- Browse and select model/checkpoint folders in a tree view.
- Generates and runs a `lerobot_record` command with `--policy.path`.
- Enforces `eval_` dataset naming with a one-click quick-fix.
- Preflight checks validate model payload, compute device, camera keys, and policy flag support.
- Results feed directly into **History** for outcome annotation and **Experiments** for cross-run comparison.

Full guide: [Deploy Tab Guide](Resources/deploy-tab-guide.md)

---

### Training

Human Intervention Learning (HIL) adaptation workflow.

- Build an incremental HIL adaptation command from a base model and intervention dataset.
- Supports `srun` wrapping for cluster-based training.
- `Apply HIL Preset` sets short adaptation defaults (8 batch, 3000 steps, 300 save freq) and opens a step-by-step HIL dialog.
- Generated command is editable before copy/paste into your terminal.
- Results feed into **Experiments** with parsed metrics and discovered checkpoints.

Full guide: [Training Tab Guide](Resources/training-tab-guide.md)

---

### Experiments

Cross-run comparison console for train, deploy, and sim-eval runs.

- Filter and compare training, deploy, and simulation eval runs side by side.
- Inspect parsed metrics from stdout, `trainer_state.json`, `wandb-summary.json`, and `eval_info.json`.
- Browse discovered checkpoints for each training run.
- Launch **Deploy Eval** or **Sim Eval** directly from a selected checkpoint.
- Optional WandB integration: deep-links to remote runs when credentials are available.

Full guide: [Resources/experiments-tab-guide.md](Resources/experiments-tab-guide.md)

---

### History

Run log browser, replay launcher, and deploy outcome editor.

- Filter runs by mode, status, and free-text search.
- View the full command, metadata, and raw transcript for any past run.
- Rerun or replay any past dataset-backed run directly from the table.
- Edit deploy episode outcomes (success / failed / unmarked), add tags and notes, and export `episode_outcomes.csv` and `notes.md`.
- Lineage panel links each run to its source dataset, model/checkpoint, and downstream artifacts.

Full guide: [History Tab Guide](Resources/history-tab-guide.md)

---

### Additional Resources

- [Resources Index](Resources/resources-index.md)
- [First-Time Setup Guide](Resources/first-time-setup.md)
- [Compatibility Matrix](Resources/compatibility-matrix.md)
- [Hardware Operations Guide](Resources/hardware-operations-guide.md)
- [Community Profiles](Resources/community-profiles.md)
- [Error Catalog](Resources/error-catalog.md)
- [Support Bundle Guide](Resources/support-bundle.md)
- [Upstream Bridge Guide](Resources/upstream-bridge.md)
