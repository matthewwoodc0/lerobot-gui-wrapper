# LeRobot GUI Wrapper

Desktop app for running a practical LeRobot workflow:
- Record datasets
- Teleoperate quickly
- Generate training commands
- Deploy/evaluate local models
- Inspect runs/datasets/videos in the visualizer

This README is focused on installation and daily use.

## Step 0: Environment Rules (Important)

1. Use a virtual environment for LeRobot.
2. Install LeRobot inside that environment.
3. Keep that environment active when launching this GUI.
4. Run `pip install -e .` in the **LeRobot repo** (`~/lerobot`), not in this wrapper repo.

This wrapper itself is run as:

```bash
python robot_pipeline.py gui
```

No `pip install -e .` is required in this wrapper repository.

## macOS Installation Guide

### 1) Install system prerequisites

```bash
brew install python@3.12 python-tk@3.12 git
```

`python-tk@3.12` is required for Tkinter GUI support on macOS.

### 2) Create LeRobot env and install LeRobot

```bash
mkdir -p ~/lerobot
if [ ! -d ~/lerobot/.git ]; then
  git clone https://github.com/huggingface/lerobot ~/lerobot
fi

/opt/homebrew/bin/python3.12 -m venv ~/lerobot/lerobot_env
source ~/lerobot/lerobot_env/bin/activate

python -m pip install --upgrade pip
cd ~/lerobot
python -m pip install -e .
```

### 3) Verify environment

```bash
python -c "import sys, tkinter as tk, lerobot; print(sys.executable); print('Tk', tk.TkVersion); print('LeRobot', lerobot.__file__)"
```

### 4) Clone and run this GUI wrapper

```bash
cd ~
git clone https://github.com/matthewwoodc0/lerobot-gui-wrapper.git
cd lerobot-gui-wrapper
python robot_pipeline.py gui
```

## Linux Installation Guide

### 1) Install system prerequisites

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-tk python3-pip git
```

### 2) Create LeRobot env and install LeRobot

```bash
mkdir -p ~/lerobot
if [ ! -d ~/lerobot/.git ]; then
  git clone https://github.com/huggingface/lerobot ~/lerobot
fi

python3 -m venv ~/lerobot/lerobot_env
source ~/lerobot/lerobot_env/bin/activate

python -m pip install --upgrade pip
cd ~/lerobot
python -m pip install -e .
```

### 3) Verify environment

```bash
python -c "import sys, tkinter, lerobot; print(sys.executable); print('LeRobot', lerobot.__file__)"
```

### 4) Clone and run this GUI wrapper

```bash
cd ~
git clone https://github.com/matthewwoodc0/lerobot-gui-wrapper.git
cd lerobot-gui-wrapper
python robot_pipeline.py gui
```

Optional Linux launcher:

```bash
python robot_pipeline.py install-launcher
```

## Fast Start (If You Already Have LeRobot Working)

```bash
source ~/lerobot/lerobot_env/bin/activate
cd ~/lerobot-gui-wrapper
python robot_pipeline.py gui
```

## First Launch Checklist

1. Open `Config` tab.
2. Click `Run Setup Check`.
3. If prompted, open `Setup Wizard` popout and follow its commands.
4. Confirm/save:
   - `lerobot_dir`
   - serial ports (`follower_port`, `leader_port`)
   - camera indices (`camera_laptop_index`, `camera_phone_index`)
   - dataset/model paths
5. Start with `Record` or `Teleop`.

## How Users Interact With The App

- `Record` tab:
  - record teleop data with camera preview and preflight checks
  - optional upload dataset to Hugging Face
  - local vs Hugging Face dataset browsing

- `Deploy` tab:
  - run local model evaluation/deployment
  - preflight checks and command preview
  - pull trained models from remote sources into local model directory

- `Teleop` tab:
  - lightweight teleop launcher
  - camera scan and quick preview

- `Training` tab:
  - generate editable training command text (copy/paste into your own terminal)
  - designed for local use or remote cluster use (manual execution)

- `Visualizer` tab:
  - browse datasets/deployments locally and on Hugging Face
  - inspect metadata and open discovered videos

- `History` tab:
  - inspect prior runs, logs, and artifacts

## CLI Modes

```bash
python robot_pipeline.py gui
python robot_pipeline.py record
python robot_pipeline.py deploy
python robot_pipeline.py config
python robot_pipeline.py doctor
python robot_pipeline.py history
python robot_pipeline.py install-launcher
```

## Common Issues

### `does not appear to be a Python project` while running `pip install -e .`
You ran that command in the wrapper repo. Run it in `~/lerobot` instead.

### macOS GUI crash or version mismatch error
Usually caused by system Python + old/missing Tk bindings.

Fix:
1. Use a venv created with Homebrew Python 3.12.
2. Install `python-tk@3.12`.
3. Launch GUI from that activated venv.

### `No module named lerobot`
Activate the LeRobot venv:

```bash
source ~/lerobot/lerobot_env/bin/activate
```

### `No module named cv2`
Install OpenCV in your active environment:

```bash
python -m pip install opencv-python
```

### Camera or serial ports fail preflight
- Verify camera indices and serial ports in `Config`.
- Re-run `python robot_pipeline.py doctor`.

## Daily Usage Pattern

1. Activate venv.
2. Run `python robot_pipeline.py gui`.
3. Record or teleop.
4. Train via generated command in your terminal.
5. Deploy local model.
6. Inspect results in `Visualizer` and `History`.
