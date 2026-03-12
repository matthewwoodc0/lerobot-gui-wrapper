# Config Tab Guide

This tab is the control center for paths, hardware defaults, named rigs, setup checks, diagnostics, launcher install, and GUI-first profile flows.

## What This Tab Is For

Use `Config` to:
- Set all persistent runtime defaults.
- Run first-time setup checks/wizard.
- Run doctor diagnostics.
- Import/export community profiles from the GUI.
- Apply built-in portable lab/hardware presets.
- Save and switch named rigs.
- Save config to disk.
- Install desktop launcher.

Config is persisted to:
- `~/.robot_config.json`
- `<lerobot_dir>/.robot_config.json` (secondary mirror)

## Main UI Areas

## 1) Grouped Config Fields

Sections:
- `Paths`
  - `lerobot_dir`
  - `runs_dir`
  - `record_data_dir`
  - `deploy_data_dir`
  - `trained_models_dir`
- `Robot Ports`
  - `follower_port`
  - `leader_port`
- `Cameras`
  - `camera_laptop_index`
  - `camera_phone_index`
  - `camera_warmup_s`
  - `camera_fps`
- `Hugging Face + Defaults`
  - `hf_username`
  - `eval_num_episodes`
  - `eval_duration_s`
  - `eval_task`

Path fields have `Browse` buttons.

## 2) First-Time Setup Wizard

Buttons:
- `Run Setup Check`
- `Open Setup Wizard`
- `Copy Setup Commands`
- `Apply Path Defaults`

What setup check validates:
- virtual env active or not
- `lerobot` importable or not
- expected LeRobot and venv folder presence
- current Python executable

Wizard popout actions:
- `Activate Venv`
  - default command: `source ~/lerobot/lerobot_env/bin/activate`
- `Enter Custom Venv Source`
  - lets you provide and remember custom activation command
- `Copy Setup Commands`
- `Apply Path Defaults`
- `Re-check Environment`

Auto behavior:
- if no venv is active, wizard can auto-open from config tab.

## 3) Profiles + Portable Presets

Buttons:
- `Import Profile`
- `Export Profile`
- `Apply Portable Preset`

Behavior:
- `Import Profile` loads a community YAML/JSON profile into the current GUI config.
- `Export Profile` writes the current GUI config back out as a portable profile.
- `Apply path fields on import` controls whether imported path values overwrite the current machine-local paths.
- Portable presets update robot defaults, camera schema, rename-map hints, and setup guidance together.

## 4) Named Rigs

Buttons:
- `Save Rig`
- `Switch Rig`
- `Delete Rig`

Behavior:
- Saves the current non-UI config as a named rig snapshot.
- Restores a saved rig snapshot back into the active config.
- Keeps the active rig name visible in both `Config` and the main window header.
- Supports fast switching across multiple robots or benches on one machine.
- Remains the place to refresh the active rig snapshot after motor setup changes the live hardware config.

What rig snapshots exclude:
- UI-only keys such as collapsed/terminal/theme state
- the saved rig list itself

## 5) Diagnostics (Doctor)

Buttons:
- `Run Doctor`
- `Copy Doctor Report`

Doctor checks include:
- Python executable/version
- LeRobot folder + important paths
- next dataset collision checks
- `huggingface-cli`
- common preflight checks (ports, camera probes, module imports, etc.)

## 6) Desktop App Launcher

Button:
- `Install Desktop Launcher`

Behavior:
- Installs/updates app launcher scripts when supported.
- If unsupported on your platform, install reports failure message.

## 7) Save Config

Button:
- `Save Config`

On save, app also refreshes related tab state:
- record dataset root field
- replay dataset defaults
- deploy model root and eval defaults
- model list refresh
- camera preview labels
- rig header state

## Example First-Time Workflow

1. Open `Config`.
2. Set `lerobot_dir` and `hf_username`.
3. Click `Run Setup Check`.
4. If setup is not ready, click `Open Setup Wizard`.
5. In wizard:
   - try `Activate Venv`
   - if needed, use `Enter Custom Venv Source`
   - use `Copy Setup Commands` for first-time bootstrap
6. Click `Apply Path Defaults`.
7. Click `Save Config`.
8. Run `Run Doctor` and resolve `FAIL` items.
9. Save the current hardware bench as a named rig.
10. Switch to `Teleop` or `Motor Setup` and verify ports / calibration before trying `Record`, `Replay`, or `Deploy`.

## What You Might See

Setup status text examples:

```text
[FAIL] Virtual env active: False
[FAIL] Python module: lerobot (ModuleNotFoundError: No module named 'lerobot')
[PASS] LeRobot folder: /Users/you/lerobot
[WARN] Expected venv folder: /Users/you/lerobot/lerobot_env
[INFO] Python executable: /usr/bin/python3
[ACTION] Neither virtual env nor lerobot import is working. Run guided setup.
```

Wizard failure prompt example:
- `Unable to run default activation command. ... Do you want to enter a custom venv source command now?`

Doctor style output:
- `[PASS] ...`
- `[WARN] ...`
- `[FAIL] ...`

## Notes

- `Apply Path Defaults` derives path values from current `lerobot_dir` + `hf_username`.
- Setup command copy is useful when onboarding from zero and ensures users run the right bootstrap sequence in terminal.
- Named rigs are the safest way to move between multiple robots on one machine because they update the whole saved hardware context in one step.
- Community profiles remain the shareable layer; named rigs remain the local bench snapshot layer. See `docs/community-profiles.md`.
- After Config is clean, the recommended bring-up order is: `Motor Setup` for first-time servo work, then `Teleop`, then camera verification in `Record`, then `Replay` or `Deploy`.
