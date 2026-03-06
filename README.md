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

### Tkinter Rules (Important)

This is the part that trips people up:

- In a normal **`venv`**, Tkinter does **not** come from `pip`. It comes from the base Python installation used to create the venv.
- If Tkinter is missing in that `venv`, the fix is:
  1. install the system/Homebrew Tk package for that Python
  2. recreate the venv from that Python
  3. reactivate the new venv
  4. rerun `python3 -m pip install -e .` inside `~/lerobot`
- In a **conda/Miniforge** env, Tkinter comes from the `tk` package. You can usually fix an existing env with `mamba install tk` or `conda install tk` and **do not** need to recreate it.
- `pip install tkinter` is not the fix for either path.

This wrapper requires no `pip install`. Run it directly:

```bash
python3 robot_pipeline.py gui
```

> **conda/mamba users:** The GUI supports conda-based environments (including Miniforge + `mamba`). See the [conda Quick Start](#conda-quick-start-already-have-lerobot-in-conda) section below for daily usage and Config setup.

---

## Quick Start by Persona

### Already Have LeRobot (venv)
Start here if you already have a working LeRobot `venv` and just want to launch the GUI:

1. Activate your existing env.
2. Run `python3 -c "import lerobot, tkinter; print('OK')"` and confirm it passes.
3. If it passes, skip the install sections and use [Fast Start (Already Have LeRobot Working)](#fast-start-already-have-lerobot-working).
4. If `import tkinter` fails, do **not** try `pip install tkinter`:
   recreate that `venv` after installing the system/Homebrew Tk package.

### Quick Start (macOS venv)
1. Follow [macOS Installation](#macos-installation).
2. Launch the app: `python3 robot_pipeline.py gui`.
3. Open **Config -> Run Setup Check** and fix any FAILs.
4. Follow the [First-Time Setup Guide](docs/first-time-setup.md).

### Quick Start (Linux sudo)
1. Follow [Linux Installation -> Option A](#option-a--standard-install-you-have-sudo).
2. Ensure serial permissions are applied (`dialout`) and re-login.
3. Launch: `python3 robot_pipeline.py gui`.
4. Follow the [First-Time Setup Guide](docs/first-time-setup.md).

### Quick Start (Conda / No sudo)
1. Follow [Linux Installation -> Option B](#option-b--no-sudo-access-shared-server-lab-machine-hpc).
2. Activate your conda env and verify `import lerobot`.
3. Launch from that shell: `python3 robot_pipeline.py gui`.
4. Use Setup Wizard if environment activation is not detected.
5. Follow the [First-Time Setup Guide](docs/first-time-setup.md).

### Already Have LeRobot (conda)
Start here if LeRobot already works in Miniforge / conda / mamba:

1. Activate that env.
2. Run `python3 -c "import lerobot, tkinter; print('OK')"` and confirm it passes.
3. If it passes, skip the install sections and use [conda Quick Start (Already Have LeRobot in conda)](#conda-quick-start-already-have-lerobot-in-conda).
4. If `import tkinter` fails, install `tk` into the same env with `mamba install tk` or `conda install tk`.

### Lab Maintainer
1. Standardize `follower/leader` IDs, calibration files, and camera schema in Config.
2. Use stable serial paths where possible (`/dev/serial/by-id/...` on Linux).
3. Validate on each workstation with `python3 robot_pipeline.py doctor`.
4. Keep docs and defaults aligned with the code-sync block below.

---

## Known Good Matrix (Latest + N-1 Policy)

| LeRobot version | Record | Deploy | Teleop | Notes |
|---|---|---|---|---|
| `0.3.x` (latest) | PASS | PASS | PASS | Primary target; validated by wrapper test suite + compatibility probes. |
| `0.2.x` (N-1) | PASS | PASS | PASS | Supported with entrypoint/flag fallback handling where needed. |

If your environment is outside these ranges, run `doctor` and verify generated commands against `--help` for your installed LeRobot modules.

---

## Code-Synced Defaults

This block is validated by tests so README defaults stay aligned with `robot_pipeline_app/constants.py`.

<!-- README_DEFAULTS_START -->
```json
{
  "camera_default_width": 640,
  "camera_default_height": 480,
  "camera_fps": 30,
  "camera_warmup_s": 5,
  "follower_robot_id_default": "red4",
  "leader_robot_id_default": "white",
  "ports": {
    "linux": {
      "follower_port": "/dev/ttyACM1",
      "leader_port": "/dev/ttyACM0"
    },
    "darwin": {
      "follower_port": "/dev/cu.usbmodem1",
      "leader_port": "/dev/cu.usbmodem0"
    }
  }
}
```
<!-- README_DEFAULTS_END -->

Additional community docs:
- [First-Time Setup Guide](docs/first-time-setup.md)
- [Compatibility Matrix](docs/compatibility-matrix.md)
- [Error Catalog](docs/error-catalog.md)
- [Support Bundle Guide](docs/support-bundle.md)
- [Community Profiles Guide](docs/community-profiles.md)
- [Upstream Bridge Guide](docs/upstream-bridge.md)
- [GA Validation Guide](docs/ga-validation.md)

---

## macOS Installation

Start here if:

- you are on macOS
- you want a standard Homebrew + `venv` setup
- you do **not** already have a working LeRobot environment with both `import lerobot` and `import tkinter`

If you already have a working macOS LeRobot `venv`, skip to [Fast Start (Already Have LeRobot Working)](#fast-start-already-have-lerobot-working).

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
# Check you are using the Homebrew venv Python (path should contain lerobot_env)
python3 -c "import sys; print('Python:', sys.executable)"

# Check Tkinter and its version — must be from python-tk@3.12, not system Python
python3 -c "import tkinter as tk; print('Tkinter OK, Tk version:', tk.TkVersion)" \
  || echo "FAIL: run  brew install python-tk@3.12  and recreate your venv"

# Check LeRobot
python3 -c "import lerobot; print('LeRobot OK')" \
  || echo "FAIL: activate your venv (source ~/lerobot/lerobot_env/bin/activate) and retry"
```

> **`No module named 'tkinter'` on macOS:** This almost always means your venv was created from the wrong Python — either system Python (`/usr/bin/python3`) or a Homebrew Python without the `python-tk` companion package. `pip install tkinter` will not fix this. Run `brew install python-tk@3.12`, then recreate your venv using `/opt/homebrew/bin/python3.12 -m venv ~/lerobot/lerobot_env`, reactivate it, and rerun `python3 -m pip install -e .` inside `~/lerobot`.

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

Choose the path that matches your machine first:

| Path | Use this when | Best fit | Important note |
|---|---|---|---|
| **Option A** | You have `sudo` and want the standard Linux `python3` + `venv` route | Personal laptop, desktop, owned VM | Tkinter is a system package; if it was missing when the venv was created, recreate the venv after installing it |
| **Option B** | You do **not** have `sudo`, or you already use conda/Miniforge/mamba | Shared lab machine, HPC, locked-down server | Tkinter comes from conda `tk`; you can add it to an existing env without recreating it |
| **Option C** | You want Option A, but `python3 -m venv` itself is broken or missing | Minimal Linux image | This is a repair path, not the default choice |

If you already have LeRobot working:

- existing `venv` + `import lerobot, tkinter` works: skip to [Fast Start (Already Have LeRobot Working)](#fast-start-already-have-lerobot-working)
- existing conda env + `import lerobot, tkinter` works: skip to [conda Quick Start (Already Have LeRobot in conda)](#conda-quick-start-already-have-lerobot-in-conda)

---

### Option A — Standard install (you have sudo)

Start here if:

- this is your own Linux machine or VM
- you have `sudo`
- you want the standard system-Python + `venv` path

Do **not** start here if you already use conda/Miniforge, or if you cannot install `python3-tk` with sudo.

This path uses your system package manager to install Python and Tkinter, then creates a `venv` in your home directory for everything else.

#### 1. Install system prerequisites

Ubuntu / Debian:
```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-tk python3-pip git
```

Fedora / RHEL / Rocky Linux:
```bash
sudo dnf install -y python3 python3-tkinter git
# pip and venv are included with python3 on Fedora
```

Arch Linux:
```bash
sudo pacman -Sy python python-pip tk git
```

> **`python3-tk` (or `python3-tkinter`) is required** — this is the GUI toolkit the app is built on. If it's missing the app will fail to start with a `ModuleNotFoundError: No module named 'tkinter'` error. This package is always a system-level install; it cannot be added via pip.

#### 2. Create a virtual environment and install LeRobot

Everything from here on happens entirely inside your home directory — no sudo needed.

```bash
# Clone the LeRobot repo (skip if you already have it)
mkdir -p ~/lerobot
if [ ! -d ~/lerobot/.git ]; then
  git clone https://github.com/huggingface/lerobot ~/lerobot
fi

# Create a venv inside the repo folder
python3 -m venv ~/lerobot/lerobot_env

# Activate it (you'll need to do this every new terminal session)
source ~/lerobot/lerobot_env/bin/activate

# Upgrade pip, then install LeRobot in editable mode
python3 -m pip install --upgrade pip
cd ~/lerobot
python3 -m pip install -e .
```

> **Why `pip install -e .`?** The `-e` flag installs LeRobot in "editable" mode, meaning it runs from the cloned source directory. This lets you `git pull` to update LeRobot without reinstalling. Always run this inside `~/lerobot`, not inside the GUI wrapper repo.

#### 3. Verify

Run each check individually so you get a specific error message if something is missing:

```bash
# Check you are using the venv Python (path should contain lerobot_env)
python3 -c "import sys; print('Python:', sys.executable)"

# Check Tkinter — must be installed at system level before creating the venv
python3 -c "import tkinter; print('Tkinter OK')" \
  || echo "FAIL: run  sudo apt install python3-tk  then recreate your venv"

# Check LeRobot — fails if venv is not activated or pip install -e . wasn't run
python3 -c "import lerobot; print('LeRobot OK')" \
  || echo "FAIL: activate your venv (source ~/lerobot/lerobot_env/bin/activate) and retry"
```

> **`No module named 'tkinter'` inside a venv:** Tkinter is part of the Python standard library but requires a system-level companion package (`python3-tk`). A venv inherits tkinter from the system Python it was created from. `pip install tkinter` will not fix this. If tkinter is missing, install `python3-tk` first and then **recreate the venv** — the existing venv won't pick it up automatically. After recreating it, reactivate the new venv and rerun `python3 -m pip install -e .` inside `~/lerobot`. Alternatively, use the Miniforge path (Option B) which bundles `tk` with no system install required.

#### 4. Clone the GUI wrapper and launch

```bash
cd ~
git clone https://github.com/matthewwoodc0/lerobot-gui-wrapper.git
cd lerobot-gui-wrapper
python3 robot_pipeline.py gui
```

#### 5. (Optional) Install desktop launcher

Creates a `.desktop` entry so the app appears in your system app menu and can be pinned to a taskbar:

```bash
python3 robot_pipeline.py install-launcher
```

Or use the **Install Desktop Launcher** button in the Config tab.

> Installs to `~/.local/share/applications/`, `~/.local/bin/`, and copies the icon to `~/.local/share/icons/`. No sudo required.

#### 6. Serial port permissions

> **Required for robot hardware access. This is Linux-only.**

By default, serial ports (`/dev/ttyACM*`, `/dev/ttyUSB*`) are only accessible to the `dialout` group. Without this your robot arms will fail with `Permission denied`:

```bash
sudo usermod -a -G dialout $USER
```

Then **log out and log back in** (the group change only takes effect on a fresh login). As a shortcut in the current shell only:

```bash
newgrp dialout
```

The **Run Setup Check** in the Config tab detects and flags this automatically.

---

### Option B — No sudo access (shared server, lab machine, HPC)

Start here if:

- you do not have `sudo`
- you are on a shared machine, HPC cluster, or locked-down server
- you already use conda / mamba / Miniforge and want to stay on that path

This installs everything in your home directory using **Miniforge**. It gives you a full Python + package manager with no system-level permissions needed.

#### 1. Install Miniforge (user-level conda)

First, check your machine architecture — the installer filename depends on it:

```bash
uname -m
# x86_64  → use Miniforge3-Linux-x86_64.sh   (most desktops and servers)
# aarch64 → use Miniforge3-Linux-aarch64.sh  (ARM servers, Raspberry Pi, some cloud VMs)
```

Download the installer using whichever tool is available on your machine:

**If you have `wget`:**
```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
     -O ~/miniforge_installer.sh
```

**If you have `curl`:**
```bash
curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
     -o ~/miniforge_installer.sh
```

**If you have neither curl nor wget — use Python (always available):**
```bash
python3 -c "
import urllib.request
url = 'https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh'
urllib.request.urlretrieve(url, 'miniforge_installer.sh')
print('Downloaded.')
"
```

**If you have no internet access on the server — copy from another machine:**
```bash
# On your local machine (download the .sh file from your browser first), then:
scp Miniforge3-Linux-x86_64.sh youruser@serveraddress:~/miniforge_installer.sh
```
You can download the file directly in your browser from [github.com/conda-forge/miniforge/releases/latest](https://github.com/conda-forge/miniforge/releases/latest).

Once the file is on the server, run it:

```bash
bash ~/miniforge_installer.sh -b -p ~/miniforge3

# Add conda to your shell (appends to ~/.bashrc or ~/.zshrc)
~/miniforge3/bin/conda init

# Apply the change in your current shell
source ~/.bashrc    # or source ~/.zshrc if you use zsh
```

You now have `conda` and `mamba` available. `mamba` is a faster drop-in replacement for `conda` and is preferred for installs.

#### 2. Create a conda environment with Python and Tkinter

```bash
mamba create -n lerobot python=3.10 tk git -y
conda activate lerobot
```

> `tk` here is the conda-forge Tkinter package — no system install needed. This is the key advantage of the conda path when you don't have sudo.

#### 3. Clone and install LeRobot

```bash
mkdir -p ~/lerobot
if [ ! -d ~/lerobot/.git ]; then
  git clone https://github.com/huggingface/lerobot ~/lerobot
fi

cd ~/lerobot
pip install -e .
```

#### 4. Verify

```bash
# Check you are using the conda env Python (path should contain miniforge3/envs/lerobot)
python3 -c "import sys; print('Python:', sys.executable)"

# Check Tkinter — should work if you included tk when creating the environment
python3 -c "import tkinter; print('Tkinter OK')" \
  || echo "FAIL: run  mamba install tk  in your active conda env"

# Check LeRobot
python3 -c "import lerobot; print('LeRobot OK')" \
  || echo "FAIL: run  pip install -e ~/lerobot  with the env active"
```

> **`No module named 'tkinter'` in conda:** You need the `tk` package from conda-forge: `mamba install tk`. Unlike the venv path you do **not** need to recreate the environment — just install `tk` into the existing env and tkinter will be available immediately.

#### 5. Clone the GUI wrapper and launch

```bash
cd ~
git clone https://github.com/matthewwoodc0/lerobot-gui-wrapper.git
cd lerobot-gui-wrapper
python3 robot_pipeline.py gui
```

#### 6. Serial port permissions without sudo

This is the one Linux requirement that genuinely needs root. If you can't run `sudo usermod`, ask your system administrator to add your username to the `dialout` group:

```
# They need to run this once:
sudo usermod -a -G dialout YOUR_USERNAME
```

As a temporary workaround while waiting (does not persist across reboots):
```bash
sudo chmod a+rw /dev/ttyACM0   # replace with your actual port
```

On some systems a `udev` rule can grant persistent access without adding to dialout — ask your sysadmin if that option is available.

---

### Option C — Python is already installed but `venv` module is missing

Start here only if you wanted **Option A** but `python3 -m venv` fails because `venv` or `ensurepip` is missing.

This is not a separate long-term workflow. It is just the fix for a broken/missing `venv` toolchain on minimal Linux installs.

Some minimal Linux installs ship Python without `ensurepip` or `venv`. If `python3 -m venv` fails:

```bash
# Try installing the system venv package
sudo apt install python3-venv python3-full   # Debian/Ubuntu

# Or install virtualenv via pip --user (no sudo)
pip install --user virtualenv
~/.local/bin/virtualenv ~/lerobot/lerobot_env
source ~/lerobot/lerobot_env/bin/activate
```

If `pip` itself is missing, download the bootstrap script — use whichever downloader you have:
```bash
# wget
wget https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py

# curl
curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py

# python only (no curl or wget needed)
python3 -c "import urllib.request; urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', '/tmp/get-pip.py')"

# then run it (no sudo)
python3 /tmp/get-pip.py --user
```

If none of that is available on the machine, fall back to the Miniforge path in Option B — it brings its own Python, pip, and venv tooling.

---

### Which option should I choose?

| Situation | Recommended path |
|-----------|-----------------|
| Your own laptop / desktop | **Option A** (system apt + venv) |
| Shared lab or university server | **Option B** (Miniforge, no sudo) |
| HPC cluster (SLURM, PBS, etc.) | **Option B** (Miniforge is standard on HPC) |
| Cloud VM where you are the owner | **Option A**, or Option B if the distro is stripped |
| `python3 -m venv` fails | **Option C** then Option B as fallback |
| Already have conda/Miniforge | **Option B** (skip the installer step) |
| Already have a working LeRobot `venv` | Skip to **Fast Start** |
| Already have a working LeRobot conda env | Skip to **conda Quick Start** |

### Shared Computer: one env for everyone?

Yes. On a shared lab machine, you do **not** need one conda env per person.

The usual pattern is:

1. One admin-managed shared environment per machine, for example a shared conda env or shared `venv`
2. Each user launches the GUI from that same environment
3. Each user keeps their own config, caches, datasets, runs, and calibration paths in their own home directory

Important tradeoff:

- a package upgrade in the shared env affects everyone
- one person's experimental installs can break other users
- this works best when one person maintains the env and everyone else treats it as read-only

Practical recommendation:

- if the machine is centrally managed, prefer **one stable shared env**
- if someone needs to experiment with package versions, give them a separate env just for that work
- avoid having multiple people run ad-hoc `pip install` commands into the shared env

For this GUI specifically, the Python environment can be shared, while the app config is still per-user (`~/.robot_config.json` by default), so different users can point at different ports, calibration files, dataset folders, and model folders without needing separate envs.

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

> **Desktop launcher with conda:** The launcher now accepts both classic `venv` and conda-prefix Python runtimes. If your configured env is `~/miniforge3/envs/lerobot`, set `LeRobot venv folder path` to that prefix and reinstall launcher once from inside that env:
>
> ```bash
> conda activate lerobot
> python3 robot_pipeline.py install-launcher
> ```
>
> If launch still opens Setup Wizard with "Environment active: False", run the app from terminal once, open Config, and set `setup_venv_activate_cmd` explicitly (for example `conda activate lerobot`).

### Conda/Mamba Environment Setup In Config

If you use conda/mamba, set these in **Config** so setup checks and terminal activation are consistent:

1. `LeRobot venv folder path` = your conda env prefix (example: `~/miniforge3/envs/lerobot`)
2. Keep launching from an activated environment:

```bash
conda activate lerobot
# or
mamba activate lerobot
python3 robot_pipeline.py gui
```

3. Verify the active env prefix (paste this in the terminal view):

```bash
python3 -c "import os,sys; print('CONDA_PREFIX=', os.environ.get('CONDA_PREFIX','')); print('sys.prefix=', sys.prefix)"
```

The `LeRobot venv folder path` should match `CONDA_PREFIX` / `sys.prefix` for conda/mamba workflows.
Custom activation commands entered in Setup Wizard are saved to config as `setup_venv_activate_cmd`.

---

## First Launch Checklist

For the full machine bring-up flow, use the [First-Time Setup Guide](docs/first-time-setup.md).

1. Open the **Config** tab.
2. Click **Run Setup Check** — review all PASS/WARN/FAIL items.
3. If prompted, open **Setup Wizard** and follow the commands it shows.
4. Confirm and save:
   - `lerobot_dir` — path to your LeRobot repo
   - `hf_username` — optional, used for Hub defaults
   - `follower_port` and `leader_port` — serial ports for your arms
   - `follower_robot_id` and `leader_robot_id` — set these if your IDs differ from the defaults shown in Config
   - `follower_calibration_path` and `leader_calibration_path` (optional explicit overrides)
   - `camera_laptop_index` and `camera_phone_index` — OpenCV camera indices
   - `trained_models_dir`, `record_data_dir`, `deploy_data_dir`
   - If a calibration file is selected and IDs are still default, IDs are inferred from filename (`arm_alpha.json` -> `arm_alpha`).
5. In the Output panel:
   - **Terminal view** is the live output and input surface for shell + run commands.
   - Use it for prompts during calibration/teleop/deploy/record (Enter, arrows, tab-complete, Ctrl+C).
6. Start with **Teleop** first, then verify cameras in **Record**.

---

## Serial Port Names by Platform

> **You must set the correct port names for your platform in Config.** Defaults are platform-aware, but you should still confirm actual USB enumerations on each machine.

| Platform | Typical port names |
|----------|--------------------|
| **Linux** | `/dev/ttyACM0`, `/dev/ttyACM1`, `/dev/ttyUSB0` |
| **macOS** | `/dev/tty.usbserial-XXXX`, `/dev/cu.usbserial-XXXX` |

To find your ports:

- **Linux:** `ls /dev/ttyACM* /dev/ttyUSB*` or `dmesg | grep tty`
- **macOS:** `ls /dev/tty.usbserial* /dev/cu.usbserial*` or `ls /dev/tty.*`

### Leader/Follower Port Walkthrough (Recommended)

If your team keeps mixing up `ACM0` vs `ACM1`, use this quick method:

1. Connect both robot USB cables.
2. In the app, click **Scan Robot Ports** (Teleop/Record/Deploy tab).
3. Write down the listed ports (`/dev/ttyACM0`, `/dev/ttyACM1`, etc.).
4. Unplug one robot cable (for example, the one physically labeled "Leader").
5. Click **Scan Robot Ports** again.
6. The missing port is the one for the cable you unplugged.
7. Plug it back in, unplug the other cable, and scan once more to verify.
8. Assign **Leader/Follower** by your physical labels, not by ACM number.

Tip: if available, prefer `/dev/serial/by-id/...` names since they are usually more stable than `ACM0`/`ACM1`.

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
python3 robot_pipeline.py doctor --json    # Run diagnostics as JSON (DiagnosticEvent v2)
python3 robot_pipeline.py compat           # Probe LeRobot entrypoints/flags
python3 robot_pipeline.py compat --json    # Machine-readable compatibility snapshot
python3 robot_pipeline.py profile export --output ./lab-profile.yaml
python3 robot_pipeline.py profile import --input ./lab-profile.yaml
python3 robot_pipeline.py support-bundle --run-id latest --output ./support-bundle.zip
python3 robot_pipeline.py history          # Show run history
python3 robot_pipeline.py install-launcher # Install desktop launcher
```

### Rollout Feature Flags

Set these in `~/.robot_config.json` for staged rollout:

```json
{
  "diagnostics_v2_enabled": true,
  "compat_probe_enabled": true,
  "support_bundle_enabled": true
}
```

### Project Status Quick Check

```bash
# Unit + integration test health
python3 -m unittest discover -s tests -p 'test_*.py' -q

# Machine-readable diagnostics health
python3 robot_pipeline.py doctor --json

# LeRobot compatibility probe status
python3 robot_pipeline.py compat --json

# Recent runtime outcomes
python3 robot_pipeline.py history --limit 10
```

---

## Platform Differences Reference

This section consolidates every meaningful behavioral difference between macOS and Linux.

---

### Python and Tkinter

| | macOS | Linux (with sudo) | Linux (no sudo) |
|--|-------|-------------------|-----------------|
| **Python source** | Homebrew (`brew install python@3.12`) | System packages (`apt install python3`) | Miniforge (`conda create -n lerobot python=3.10`) |
| **Tkinter** | Must install separately: `brew install python-tk@3.12` | `apt install python3-tk` | `mamba install tk` (conda-forge, no sudo) |
| **Failure mode** | GUI will not start without `python-tk` | GUI will not start without `python3-tk` | Use Miniforge Option B — brings its own `tk` |

---

### Serial Port Names

> **Critical: confirm ports from a live scan on each machine.** The app has platform-aware defaults, but USB device naming is still hardware-dependent.

| | macOS | Linux |
|--|-------|-------|
| **Port naming** | `/dev/tty.usbserial-*` or `/dev/cu.usbserial-*` | `/dev/ttyACM*` or `/dev/ttyUSB*` |
| **Default in config** | follower `/dev/cu.usbmodem1`, leader `/dev/cu.usbmodem0` | follower `/dev/ttyACM1`, leader `/dev/ttyACM0` |
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

## First-Time Hardware Calibration

On a new machine, teleop/record/deploy often fail preflight until calibration files are created for the connected arms.

Recommended order:

1. Run **Config -> Run Setup Check**
2. Use **Teleop -> Scan Robot Ports**
3. Try **Run Teleop**
4. If preflight flags calibration, open the terminal view in the output panel and run:

```bash
python3 -m lerobot.calibrate --robot.type=<follower_robot_type> --robot.port=<follower_port> --robot.id=<follower_robot_id>
python3 -m lerobot.calibrate --robot.type=<leader_robot_type>   --robot.port=<leader_port>   --robot.id=<leader_robot_id>
```

Use the exact values shown in your **Config** tab and preflight dialogs for your current machine. After calibration succeeds, rerun **Teleop** before moving on to **Record** or **Deploy**.

For the complete bring-up flow, see [First-Time Setup Guide](docs/first-time-setup.md).

If your LeRobot environment reports:

```text
ModuleNotFoundError: No module named 'scservo_sdk'
```

install the Feetech SDK package in your active environment:

```bash
python3 -m pip install feetech-servo-sdk
```

This is different from Tkinter: for Feetech, you usually **do not** need to recreate the venv. Activate the same existing LeRobot environment first, install `feetech-servo-sdk`, then retry calibration / teleop.

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
2. Click **Scan Robot Ports** and confirm you are using the expected follower/leader devices.
3. Prefer stable device paths such as `/dev/serial/by-id/...` on Linux when available.

Treat the fingerprint checks as an identity hint, not the primary source of truth. The most important thing is that Teleop starts cleanly with the intended follower/leader ports and IDs.

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

### `No module named 'transformers'` when deploying a model

VLM-based policies (SmolVLA, etc.) require the `transformers` library, which is not installed by a base `pip install -e .` of LeRobot. The deploy preflight will catch this as a FAIL before the run starts.

```bash
# Quick fix — install transformers directly:
pip install transformers

# Or install LeRobot with the SmolVLA extras (recommended):
cd ~/lerobot
pip install -e '.[smolvla]'
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

If you don't have sudo, ask your sysadmin to add you to `dialout`. As a temporary workaround (does not persist after reboot):
```bash
sudo chmod a+rw /dev/ttyACM0   # replace with your actual port name
```

### Linux: `ModuleNotFoundError: No module named 'tkinter'`
Tkinter is not included in the Python standard library on most Linux distros — it is a separate system package. If you have sudo:
```bash
sudo apt install python3-tk          # Debian/Ubuntu
sudo dnf install python3-tkinter     # Fedora/RHEL
sudo pacman -Sy tk                   # Arch
```
`pip install tkinter` will not fix this in a plain `venv`. If your current LeRobot venv was created before the system Tk package was installed, recreate that venv and rerun `python3 -m pip install -e .` in `~/lerobot`. If you **don't have sudo**, use the Miniforge path (Option B in the Linux Installation section). Miniforge lets you install `tk` from conda-forge into your user environment with no system permissions required:
```bash
mamba install tk
# or during environment creation:
mamba create -n lerobot python=3.10 tk -y
```

### Linux: No sudo access — can't install system packages
See [Option B — No sudo access](#option-b--no-sudo-access-shared-server-lab-machine-hpc) in the Linux Installation section above. The short version: install Miniforge into your home directory and use `conda`/`mamba` instead of `apt`. Everything, including Python and Tkinter, installs under `~/miniforge3` with no root required.

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
