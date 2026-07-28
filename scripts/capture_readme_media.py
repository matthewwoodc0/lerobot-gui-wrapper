#!/usr/bin/env python3
"""Create sanitized README media from the real Qt interface."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/media"),
        help="Directory for the screenshot and GIF.",
    )
    return parser.parse_args()


def _copy_qt_plugins(target_root: Path) -> Path:
    spec = importlib.util.find_spec("PySide6")
    if spec is None or spec.origin is None:
        raise RuntimeError("PySide6 is not installed.")
    source = Path(spec.origin).resolve().parent / "Qt" / "plugins"
    target = target_root / "qt-plugins"
    subprocess.run(["/bin/cp", "-R", str(source), str(target)], check=True)
    return target


def _set_qt_paths(plugin_root: Path) -> None:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["QT_PLUGIN_PATH"] = str(plugin_root)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(plugin_root / "platforms")


def _demo_config() -> dict[str, Any]:
    return {
        "hf_username": "demo",
        "ui_theme_mode": "dark",
        "ui_terminal_visible": False,
        "ui_sidebar_collapsed": False,
        "lerobot_dir": "/Users/demo/lerobot",
        "lerobot_venv_dir": "/Users/demo/lerobot/.venv",
        "follower_port": "/dev/tty.usbmodem-follower",
        "leader_port": "/dev/tty.usbmodem-leader",
        "follower_robot_id": "so101_follower",
        "leader_robot_id": "so101_leader",
        "camera_laptop_index": "0",
        "camera_phone_index": "1",
        "record_data_dir": "/Users/demo/lerobot/data",
        "deploy_data_dir": "/Users/demo/lerobot/cache",
        "trained_models_dir": "/Users/demo/lerobot/models",
        "runs_dir": "/Users/demo/lerobot/runs",
        "last_dataset_name": "pick-and-place-cubes",
        "last_task": "Pick up the red cube and place it in the tray.",
    }


def _capture_pages(output_dir: Path, plugin_root: Path) -> dict[str, Path]:
    from robot_pipeline_app.gui_qt_app import create_qt_preview_window

    _set_qt_paths(plugin_root)

    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication

    QCoreApplication.setLibraryPaths([str(plugin_root)])
    app = QApplication.instance() or QApplication(["capture-readme-media"])
    app.setApplicationName("LeRobot Pipeline Manager")

    with (
        patch("robot_pipeline_app.gui_qt_app.save_config"),
        patch("robot_pipeline_app.gui_qt_visualizer_page.save_config"),
        patch("robot_pipeline_app.gui_qt_app.GuiTerminalShell.start", return_value=(True, "")),
    ):
        window = create_qt_preview_window(_demo_config())
        window.resize(1320, 880)
        window.show()
        app.processEvents()

        captured: dict[str, Path] = {}
        for section_id in ("config", "teleop", "record"):
            window.select_section(section_id)
            app.processEvents()
            path = output_dir / f"capture-{section_id}.png"
            if not window.grab().save(str(path), "PNG"):
                raise RuntimeError(f"Could not save {path}.")
            captured[section_id] = path

        shutil.copy2(captured["record"], output_dir / "pipeline-manager-record.png")
        window.close()
        app.processEvents()
    return captured


def _build_gif(captured: dict[str, Path], output_path: Path) -> None:
    from PIL import Image

    resampling = getattr(Image, "Resampling", Image)
    page_images = [
        Image.open(captured[section_id]).convert("RGB").resize((990, 660), resampling.LANCZOS)
        for section_id in ("config", "teleop", "record")
    ]

    frames: list[Image.Image] = []
    hold_frames = 36
    transition_frames = 6
    for index, image in enumerate(page_images):
        frames.extend([image.copy() for _ in range(hold_frames)])
        if index + 1 < len(page_images):
            next_image = page_images[index + 1]
            for step in range(1, transition_frames + 1):
                frames.append(Image.blend(image, next_image, step / (transition_frames + 1)))

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=170,
        loop=0,
        optimize=True,
    )


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="lerobot-readme-media-", dir="/private/tmp") as temp_dir:
        plugin_root = _copy_qt_plugins(Path(temp_dir))
        _set_qt_paths(plugin_root)
        captured = _capture_pages(output_dir, plugin_root)
        _build_gif(captured, output_dir / "pipeline-manager-demo.gif")

    for path in captured.values():
        path.unlink(missing_ok=True)
    print(f"Created README media in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
