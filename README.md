# LeRobot GUI Wrapper

Desktop GUI for a practical LeRobot workflow:

- Record datasets with camera preview and preflight checks
- Teleoperate with live status and hardware diagnostics
- Generate and edit training commands
- Deploy and evaluate local models
- Inspect runs, datasets, and videos in the Visualizer
- Browse full run history with logs and artifacts

---

## Platform Support

This app runs on **macOS** and **Linux**. The core workflow is the same on both, but there are hardware, permission, and launcher differences you need to know. These are called out explicitly throughout this README and collected in the [Platform Differences Reference](#platform-differences-reference) at the bottom.

---

## Environment Rules (Read This First)

1. Use a **virtual environment or conda environment** for LeRobot.
2. Install LeRobot **inside** that environment.
3. Keep the environment **activated** when launching this GUI.
4. Run `pip install -e .` in the **LeRobot repo** (`~/lerobot`), not in this wrapper repo.

This wrapper requires no `pip install`. Run it directly:

```bash
python3 robot_pipeline.py gui
```

> **conda users:** The GUI fully supports conda environments. See the [conda Quick Start](#conda-quick-start-already-have-lerobot-in-conda) section below for daily usage. The Setup Wizard correctly detects conda environments — you can ignore any "Expected venv folder" warnings when using conda.

---

## macOS Installation

### 1. Install system prerequisites

```bash
brew install python@3.12 python-tk@3.12 git
```

> **`python-tk@3.12` is required.** Without it, the GUI will not launch. This is a macOS-only requirement — Linux users get Tkinter through the system package manager.

### 2. Create venv and install LeRobot

```bash
mkdir -p ~/lerobot
if [ ! -d ~/lerobot/.git ]; then
  git clone https://github.com/huggingface/lerobot ~/lerobot
fi

/opt/homebrew/bin/python3.12 -m venv ~/lerobot/lerobot_env
source ~/lerobot/lerobot_env/bin/activate

python3 -m pip install --upgrade pip
cd ~/lerobot
python3 -m pip install -e .
```

### 3. Verify

```bash
python3 -c "import sys, tkinter as tk, lerobot; print(sys.executable); print('Tk', tk.TkVersion); print('LeRobot OK')"
```

### 4. Clone and launch

```bash
cd ~
git clone https://github.com/matthewwoodc0/lerobot-gui-wrapper.git
cd lerobot-gui-wrapper
python3 robot_pipeline.py gui
```

### 5. (Optional) Install app launcher

Creates `~/Applications/LeRobot Pipeline Manager.app` so you can open the app from Finder, Spotlight, or drag it to your Dock:

```bash
python3 robot_pipeline.py install-launcher
```

Or use the **Install Desktop Launcher** button in the Config tab.

> Installs a `.app` bundle to `~/Applications/` and a CLI script to `~/.local/bin/lerobot-pipeline-manager`.

---

## Linux Installation

### 1. Install system prerequisites

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-tk python3-pip git
```

### 2. Create venv and install LeRobot

```bash
mkdir -p ~/lerobot
if [ ! -d ~/lerobot/.git ]; then
  git clone https://github.com/huggingface/lerobot ~/lerobot
fi

python3 -m venv ~/lerobot/lerobot_env
source ~/lerobot/lerobot_env/bin/activate

python3 -m pip install --upgrade pip
cd ~/lerobot
python3 -m pip install -e .
```

### 3. Verify

```bash
python3 -c "import sys, tkinter, lerobot; print(sys.executable); print('LeRobot OK')"
```

### 4. Clone and launch

```bash
cd ~
git clone https://github.com/matthewwoodc0/lerobot-gui-wrapper.git
cd lerobot-gui-wrapper
python3 robot_pipeline.py gui
```

### 5. (Optional) Install app launcher

Creates a `.desktop` entry so the app appears in your system app menu:

```bash
python3 robot_pipeline.py install-launcher
```

Or use the **Install Desktop Launcher** button in the Config tab.

> Installs to `~/.local/share/applications/`, `~/.local/bin/`, and copies the icon to `~/.local/share/icons/`.

### 6. Serial port permissions (Linux only)

> **This step is Linux-only and required for hardware access.**

Your user must be in the `dialout` group to access serial ports without `sudo`:

```bash
sudo usermod -a -G dialout $USER
```

Log out and log back in (or run `newgrp dialout` in the same shell). Without this, robot arms and servo controllers will fail to connect.

The **Run Setup Check** in the Config tab flags this automatically.

---

## Fast Start (Already Have LeRobot Working)

```bash
source ~/lerobot/lerobot_env/bin/activate
cd ~/lerobot-gui-wrapper
python3 robot_pipeline.py gui
```

## conda Quick Start (Already Have LeRobot in conda)

If you installed LeRobot using Miniforge, Miniconda, or Anaconda:

```bash
conda activate lerobot
cd ~/lerobot-gui-wrapper
python3 robot_pipeline.py gui
```

> **Desktop launcher with conda:** After installing the desktop launcher (`python3 robot_pipeline.py install-launcher`), the launcher will attempt to activate your conda environment automatically if it was active when the launcher was installed. If it fails to start (Setup Wizard shown at launch), run the GUI from terminal instead. The launcher works most reliably when your conda environment is active during installation.

---

## First Launch Checklist

1. Open the **Config** tab.
2. Click **Run Setup Check** — review all PASS/WARN/FAIL items.
3. If prompted, open **Setup Wizard** and follow the commands it shows.
4. Confirm and save:
   - `lerobot_dir` — path to your LeRobot repo
   - `follower_port` and `leader_port` — serial ports for your arms
   - `camera_laptop_index` and `camera_phone_index` — OpenCV camera indices
   - `trained_models_dir`, `record_data_dir`, `deploy_data_dir`
5. Start with **Record** or **Teleop**.

---

## Serial Port Names by Platform

> **You must set the correct port names for your platform in Config. The defaults are Linux-style and will not work on macOS.**

| Platform | Typical port names |
|----------|--------------------|
| **Linux** | `/dev/ttyACM0`, `/dev/ttyACM1`, `/dev/ttyUSB0` |
| **macOS** | `/dev/tty.usbserial-XXXX`, `/dev/cu.usbserial-XXXX` |

To find your ports:

- **Linux:** `ls /dev/ttyACM* /dev/ttyUSB*` or `dmesg | grep tty`
- **macOS:** `ls /dev/tty.usbserial* /dev/cu.usbserial*` or `ls /dev/tty.*`

---

## Tabs and What They Do

### Record
- Teleop data recording with camera preview and preflight checks
- Set episode count, task description, dataset name, FPS
- Optional upload to Hugging Face after recording
- Browse local and HuggingFace datasets

### Deploy
- Run local model evaluation against the robot
- Two-panel model/checkpoint browser
- Preflight checks and command preview before running
- Pull trained model checkpoints from remote sources

### Teleop
- Lightweight teleop launcher (no dataset recording)
- Camera scan and quick port/id assignment
- Floating Run Controls popout with elapsed timer and stop button

### Training
- Generate an editable `python -m lerobot.train ...` command
- Copy/paste into your own terminal for local or remote training
- Does not run training directly — this is intentional for cluster/remote use

### Visualizer
- Browse local datasets, deployment results, and HuggingFace repos
- View dataset metadata, episode stats, and discovered video files
- Double-click to open videos in your system player

### History
- Browse all past Record, Deploy, and Teleop runs
- View logs, artifacts, command lines, exit codes
- Re-run any previous command with one click

### Config
- All hardware and path settings
- Run Setup Check / Setup Wizard
- Install Desktop Launcher (works on both macOS and Linux)
- Diagnostics text copy

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Cmd+1` / `Ctrl+1` | Switch to Record tab |
| `Cmd+2` / `Ctrl+2` | Switch to Deploy tab |
| `Cmd+3` / `Ctrl+3` | Switch to Teleop tab |
| `Cmd+4` / `Ctrl+4` | Switch to Training tab |
| `Cmd+5` / `Ctrl+5` | Switch to Visualizer tab |
| `Cmd+6` / `Ctrl+6` | Switch to Config tab |
| `Cmd+7` / `Ctrl+7` | Switch to History tab |
| `F2` | Focus terminal input |

> macOS uses `Cmd` (Command key). Linux uses `Ctrl`.

---

## CLI Modes

```bash
python3 robot_pipeline.py gui              # Launch desktop GUI
python3 robot_pipeline.py record           # Record a dataset (CLI)
python3 robot_pipeline.py deploy           # Run deploy/eval (CLI)
python3 robot_pipeline.py config           # Print current config
python3 robot_pipeline.py doctor           # Run diagnostics
python3 robot_pipeline.py history          # Show run history
python3 robot_pipeline.py install-launcher # Install desktop launcher
```

---

## Platform Differences Reference

This section consolidates every meaningful behavioral difference between macOS and Linux.

---

### Python and Tkinter

| | macOS | Linux |
|--|-------|-------|
| **Python source** | Homebrew (`brew install python@3.12`) | System packages (`apt install python3`) |
| **Tkinter** | Must install separately: `brew install python-tk@3.12` | Included via `apt install python3-tk` |
| **Failure mode** | GUI will not start without `python-tk` | GUI will not start without `python3-tk` |

---

### Serial Port Names

> **Critical: the app defaults to Linux-style port names. macOS users must update them in Config.**

| | macOS | Linux |
|--|-------|-------|
| **Port naming** | `/dev/tty.usbserial-*` or `/dev/cu.usbserial-*` | `/dev/ttyACM*` or `/dev/ttyUSB*` |
| **Default in config** | `/dev/ttyACM0`, `/dev/ttyACM1` (**wrong for macOS — must change**) | `/dev/ttyACM0`, `/dev/ttyACM1` (correct) |
| **Find ports** | `ls /dev/tty.*` | `ls /dev/ttyACM* /dev/ttyUSB*` |

---

### Serial Port Permissions

| | macOS | Linux |
|--|-------|-------|
| **Permission model** | macOS grants access by default — no group membership required | User must be in the `dialout` group |
| **Fix if denied** | Not usually needed; try replugging or rebooting | `sudo usermod -a -G dialout $USER` then re-login |
| **Setup Check** | Skips dialout check (not applicable) | Flags missing dialout membership as FAIL |

---

### Desktop Launcher

| | macOS | Linux |
|--|-------|-------|
| **Format** | `.app` bundle in `~/Applications/` | `.desktop` entry in `~/.local/share/applications/` |
| **CLI script** | `~/.local/bin/lerobot-pipeline-manager` | `~/.local/bin/lerobot-pipeline-manager` |
| **After install** | Open from Finder, Spotlight, or drag to Dock | Open from app menu / application launcher |
| **Install command** | `python3 robot_pipeline.py install-launcher` | `python3 robot_pipeline.py install-launcher` |

---

### Teleop Keyboard Input (Next/Reset Episode)

> **This is an area of known friction on macOS.**

| | macOS | Linux |
|--|-------|-------|
| **Permissions needed** | **Yes — System Settings > Privacy & Security > Input Monitoring and Accessibility** | No special permissions on X11; Wayland may block input injection |
| **If keys don't work** | Grant Terminal (or your launcher app) Input Monitoring + Accessibility access, then fully quit and relaunch | Switch from Wayland to X11 session if blocked |
| **Fallback** | Use the terminal and press right/left arrow keys directly from keyboard | Use the terminal and press right/left arrow keys directly from keyboard |
| **Display session** | N/A | Wayland restricts key injection by design; X11 is permissive |

> On macOS, if arrow keys in the Run Controls popout do nothing, open **System Settings → Privacy & Security → Input Monitoring**, find the app running this GUI (Terminal, iTerm2, Python, etc.), and enable it. Do the same under **Accessibility**. You must fully quit and relaunch the approved app for the change to take effect.

---

### Teleop AV1 / Video Codec

| | macOS | Linux |
|--|-------|-------|
| **Default teleop entrypoint** | Prefers legacy `control_robot` entrypoint to avoid AV1 hardware decode issues | Uses the standard `lerobot_teleoperate` entrypoint |
| **Why** | Some macOS systems lack AV1 hardware decode support; the newer entrypoint may require it | No special fallback needed |
| **Config key** | `teleop_av1_fallback` (default `true` on macOS) | Not set / not applicable |

---

### File Manager Integration

| | macOS | Linux |
|--|-------|-------|
| **Open in file manager** | `open <path>` (opens in Finder) | `xdg-open <path>` (opens in default file manager) |
| **Open video files** | `open <file>` (uses default app via Launch Services) | `xdg-open <file>` |

---

### Keyboard Modifier Key

| | macOS | Linux |
|--|-------|-------|
| **Tab shortcuts** | `Cmd+1` through `Cmd+7` | `Ctrl+1` through `Ctrl+7` |

---

## HuggingFace Account (Optional)

A HuggingFace account is **not required** to use this app. You can record, teleoperate, train, and deploy entirely offline without one.

If you do not have a HuggingFace account:

- Leave the `hf_username` field in Config blank or as a placeholder.
- Deploy runs automatically disable dataset upload to HuggingFace (`--dataset.push_to_hub=false`).
- For recording, uncheck the **Upload to HuggingFace** option in the Record tab.
- Dataset names will still work — they just won't be pushed to the Hub.

If you do have a HuggingFace account:

- Set `hf_username` in Config to your HF username.
- Use `huggingface-cli login` in your terminal before uploading datasets.
- The Record tab's upload step will push to `{username}/{dataset_name}`.

---

## Port Fingerprints (What Are They?)

When you run a preflight check, you may see warnings like:

```
[WARN] Follower port fingerprint: no baseline saved yet; scan/assign once to lock mapping.
[WARN] Follower port role inference: could not infer role from serial fingerprint text.
```

**What this means:** A "fingerprint" is the unique serial number or ID string that a USB device exposes. The app uses these to verify that the right physical robot arm is always connected to the right port (follower vs. leader), even if port names change after a reboot.

**What to do:** These are warnings, not errors. Your robot will work fine without them. To lock the mapping and suppress these warnings permanently:

1. Go to the **Teleop** tab.
2. Click **Scan Ports / Assign Roles**.
3. The app will read the serial fingerprint from each connected device and save it as the baseline.

After saving, the preflight check will confirm that the correct hardware is on each port every time.

---

## Common Issues

### Setup Wizard appears at launch when using desktop shortcut (conda users)

The desktop launcher tries to activate your Python environment automatically. If your conda environment was not active when you installed the launcher, it may not be able to find it.

**Fix:** Launch from terminal instead:
```bash
conda activate lerobot
cd ~/lerobot-gui-wrapper
python3 robot_pipeline.py gui
```

Or reinstall the launcher while your conda environment is active:
```bash
conda activate lerobot
cd ~/lerobot-gui-wrapper
python3 robot_pipeline.py install-launcher
```

### Setup Wizard shows `[WARN] Expected venv folder` (conda users)

This warning appears because the Setup Wizard looks for a `venv`-style environment folder. If you're using conda, this is expected and harmless — the wizard now recognizes conda environments correctly and will show `[PASS] Environment active: True (conda)`.

### `does not appear to be a Python project` during `pip install -e .`
You ran this in the wrapper repo. Run it in `~/lerobot` instead.

### macOS: GUI crash or "no display" / Tk version mismatch
Almost always caused by using system Python instead of Homebrew Python.

Fix:
1. `brew install python@3.12 python-tk@3.12`
2. Create your venv with `/opt/homebrew/bin/python3.12 -m venv ...`
3. Activate that venv before launching.

### macOS: Camera preview blank or frozen
Try setting the environment variable before launching:
```bash
LEROBOT_DISABLE_CAMERA_PREVIEW=1 python3 robot_pipeline.py gui
```
Then test cameras separately.

### `No module named lerobot`
Your venv is not activated:
```bash
source ~/lerobot/lerobot_env/bin/activate
```

### `No module named cv2`
```bash
python3 -m pip install opencv-python
```

### Linux: `Permission denied` on serial port
You are not in the `dialout` group:
```bash
sudo usermod -a -G dialout $USER
# Then log out and log back in
```

### macOS: Serial port not found / wrong name
The default config uses Linux port names. Open Config, find `follower_port` and `leader_port`, and set them to your macOS device paths (e.g., `/dev/tty.usbserial-AB1234`). Run `ls /dev/tty.*` with the device plugged in to find the right name.

### macOS: Arrow keys don't work in Run Controls
See [Teleop Keyboard Input](#teleop-keyboard-input-nextreset-episode) above. You need to grant **Input Monitoring** and **Accessibility** access to your terminal app in System Settings.

### Linux: Arrow keys don't work on Wayland
Switch your session to X11 (log out, select "Ubuntu on Xorg" or equivalent at the login screen).

### Deploy crashes with "cannot post" or HuggingFace upload error

Deploy runs automatically disable dataset upload to HuggingFace. If you still see upload errors, add `--dataset.push_to_hub=false` to the **Custom args** field in the Deploy tab's Advanced Options section.

This also applies to smolVLA and other policy types — they all use the same record mechanism for evaluation, and none of them need to push to HuggingFace to function locally.

### `ValueError: not enough values to unpack` during second deploy run

This can happen if a bare dataset name (without `username/` prefix) ends up in the Advanced Options `dataset.repo_id` field. The app now normalizes this automatically. If you see it, check that your `hf_username` is set in Config (it can be any placeholder if you don't have an HF account), and that the Advanced Options `dataset.repo_id` field includes a `/` (e.g., `myusername/eval_dataset_1`).

### Camera or serial ports fail preflight
- Verify camera indices and serial ports in Config.
- Re-run `python3 robot_pipeline.py doctor`.
