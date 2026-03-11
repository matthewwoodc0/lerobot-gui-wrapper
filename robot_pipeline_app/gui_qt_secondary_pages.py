from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLayout,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    import cv2 as _cv2_module  # type: ignore[import-not-found]

    _CV2_AVAILABLE = True
except Exception:  # pragma: no cover - fallback for minimal installs
    _cv2_module = None  # type: ignore[assignment]
    _CV2_AVAILABLE = False

from .artifacts import (
    _normalize_deploy_episode_outcomes,
    list_runs,
    normalize_deploy_result,
    write_deploy_episode_spreadsheet,
    write_deploy_notes_file,
)
from .camera_schema import apply_camera_schema_entries_to_config, camera_schema_entries_for_editor
from .checks import collect_doctor_checks, summarize_checks
from .compat_snapshot import build_compat_snapshot
from .config_store import _atomic_write, get_deploy_data_dir, get_lerobot_dir, normalize_config_without_prompts, save_config
from .constants import CONFIG_FIELDS
from .desktop_launcher import install_desktop_launcher
from .history_utils import (
    HISTORY_MODE_VALUES,
    _build_history_refresh_payload_from_runs,
    _command_from_item,
    open_path_in_file_manager,
)
from .gui_qt_dialogs import ask_text_dialog_with_actions
from .visualizer_utils import (
    _VisualizerRefreshSnapshot,
    _build_selection_payload,
    _collect_sources_for_refresh,
    _open_path,
    _visualizer_source_row_values,
)
from .repo_utils import normalize_deploy_rerun_command
from .run_controller_service import ManagedRunController, RunUiHooks
from .setup_wizard import build_setup_status_summary, build_setup_wizard_guide, probe_setup_wizard_status


def _build_card(title: str) -> tuple[QFrame, QVBoxLayout]:
    card = QFrame()
    card.setObjectName("SectionCard")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(18, 18, 18, 18)
    layout.setSpacing(12)
    header = QLabel(title)
    header.setObjectName("SectionMeta")
    layout.addWidget(header)
    return card, layout


def _set_readonly_table(table: QTableWidget) -> None:
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.verticalHeader().setVisible(False)
    table.setAlternatingRowColors(True)
    table.setWordWrap(False)


def _set_table_headers(table: QTableWidget, headers: list[str]) -> None:
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)
    header = table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    header.setStretchLastSection(True)


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, default=str)


def _quiet_cv2_logging(cv2_mod: Any) -> None:
    try:
        if hasattr(cv2_mod, "utils") and hasattr(cv2_mod.utils, "logging"):
            logging_mod = cv2_mod.utils.logging
            level = getattr(logging_mod, "LOG_LEVEL_SILENT", getattr(logging_mod, "LOG_LEVEL_ERROR", None))
            if level is not None:
                logging_mod.setLogLevel(level)
                return
        if hasattr(cv2_mod, "setLogLevel"):
            level = getattr(cv2_mod, "LOG_LEVEL_SILENT", getattr(cv2_mod, "LOG_LEVEL_ERROR", None))
            if level is not None:
                cv2_mod.setLogLevel(level)
    except Exception:
        pass


if _CV2_AVAILABLE:
    _quiet_cv2_logging(_cv2_module)


class _VideoFrameLabel(QLabel):
    def __init__(self) -> None:
        super().__init__("Loading video preview...")
        self._pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setMinimumHeight(210)
        self.setObjectName("MutedLabel")

    def set_preview_message(self, message: str) -> None:
        self._pixmap = None
        self.clear()
        self.setText(str(message))

    def set_preview_pixmap(self, pixmap: QPixmap) -> None:
        if pixmap.isNull():
            return
        self._pixmap = pixmap
        self.setText("")
        self._sync_pixmap()

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)  # type: ignore[misc]
        self._sync_pixmap()

    def _sync_pixmap(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        target = self.size()
        if target.width() <= 0 or target.height() <= 0:
            return
        scaled = self._pixmap.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


class _VideoGalleryTile(QFrame):
    _SEEK_SECONDS = 5.0

    def __init__(self, *, item: dict[str, Any], cv2_module: Any | None = None) -> None:
        super().__init__()
        self.setObjectName("SectionCard")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._item: dict[str, Any] = {}
        self._cv2_module = cv2_module if cv2_module is not None else _cv2_module
        self._source_value: str | None = None
        self._capture: Any | None = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_frame)
        self._frame_interval_ms = 66
        self._is_paused = False
        self._duration_ms: float | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        self.preview = _VideoFrameLabel()
        self.preview.setMinimumHeight(240)
        layout.addWidget(self.preview)

        self.title_label = QLabel("")
        self.title_label.setWordWrap(True)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setObjectName("SectionMeta")
        layout.addWidget(self.title_label)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)

        self.back_button = QPushButton("-5s")
        self.back_button.clicked.connect(lambda: self.seek_seconds(-self._SEEK_SECONDS))
        controls.addWidget(self.back_button)

        self.play_pause_button = QPushButton("Pause")
        self.play_pause_button.clicked.connect(self.toggle_pause)
        controls.addWidget(self.play_pause_button)

        self.forward_button = QPushButton("+5s")
        self.forward_button.clicked.connect(lambda: self.seek_seconds(self._SEEK_SECONDS))
        controls.addWidget(self.forward_button)

        layout.addLayout(controls)

        self.time_label = QLabel("00:00 / --:--")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_label.setObjectName("MutedLabel")
        layout.addWidget(self.time_label)

        self.set_item(item)

    def set_item(self, item: dict[str, Any]) -> None:
        self._item = dict(item)
        title = str(item.get("relative_path") or item.get("path") or item.get("url") or "Video").strip()
        self.title_label.setText(title or "Video")
        self._source_value = self._resolve_source_value(item)
        self._is_paused = False
        self._duration_ms = None
        self._refresh_control_labels()
        if self._cv2_module is None:
            self.preview.set_preview_message("Video playback is unavailable because OpenCV is not installed.")
            return
        if not self._source_value:
            self.preview.set_preview_message("This video could not be resolved for playback.")
            return
        self.preview.set_preview_message("Loading video preview...")

    def start(self) -> None:
        if self._cv2_module is None or not self._source_value:
            return
        if not self._ensure_capture():
            return
        self._is_paused = False
        self._refresh_control_labels()
        self._timer.start(self._frame_interval_ms)
        self._advance_frame()

    def stop(self) -> None:
        self._timer.stop()
        self._is_paused = False
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None
        self._refresh_control_labels()

    def closeEvent(self, event: object) -> None:
        self.stop()
        super().closeEvent(event)  # type: ignore[misc]

    def toggle_pause(self) -> None:
        if self._capture is None and not self._ensure_capture():
            return
        if self._is_paused:
            self._is_paused = False
            self._timer.start(self._frame_interval_ms)
            self._advance_frame()
        else:
            self._is_paused = True
            self._timer.stop()
        self._refresh_control_labels()

    def seek_seconds(self, delta_seconds: float) -> None:
        if self._capture is None and not self._ensure_capture():
            return
        if self._capture is None or self._cv2_module is None:
            return
        current_ms = self._current_position_ms()
        duration_ms = self._duration_ms
        target_ms = max(0.0, current_ms + (float(delta_seconds) * 1000.0))
        if duration_ms is not None and duration_ms > 0:
            target_ms = min(target_ms, max(duration_ms - 1.0, 0.0))
        try:
            self._capture.set(self._cv2_module.CAP_PROP_POS_MSEC, target_ms)
        except Exception:
            fps = self._capture_fps()
            if fps > 0:
                try:
                    self._capture.set(self._cv2_module.CAP_PROP_POS_FRAMES, max(0.0, (target_ms / 1000.0) * fps))
                except Exception:
                    pass
        if self._is_paused:
            self._advance_frame()
        else:
            self._timer.start(self._frame_interval_ms)
            self._advance_frame()
        self._refresh_control_labels()

    def _resolve_source_value(self, item: dict[str, Any]) -> str | None:
        raw_path = item.get("path")
        if raw_path:
            path = Path(str(raw_path))
            if path.exists():
                return str(path)
        raw_url = str(item.get("url", "")).strip()
        if raw_url:
            return raw_url
        return None

    def _ensure_capture(self) -> bool:
        if self._capture is not None:
            return True
        if self._cv2_module is None or not self._source_value:
            return False
        capture = self._cv2_module.VideoCapture(self._source_value)
        if capture is None or not capture.isOpened():
            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    pass
            self.preview.set_preview_message("Unable to open this video.")
            return False
        fps = 0.0
        try:
            fps = float(capture.get(self._cv2_module.CAP_PROP_FPS) or 0.0)
        except Exception:
            fps = 0.0
        if fps > 1.0:
            self._frame_interval_ms = max(16, min(250, int(1000.0 / fps)))
        else:
            self._frame_interval_ms = 66
        self._duration_ms = self._calculate_duration_ms(capture, fps)
        self._capture = capture
        self._refresh_time_label()
        return True

    def _advance_frame(self) -> None:
        if self._capture is None and not self._ensure_capture():
            self._timer.stop()
            return
        if self._capture is None or self._cv2_module is None:
            return
        ok, frame_bgr = self._capture.read()
        if not ok or frame_bgr is None:
            try:
                self._capture.set(self._cv2_module.CAP_PROP_POS_FRAMES, 0)
                ok, frame_bgr = self._capture.read()
                if ok and frame_bgr is not None:
                    self._is_paused = False
                    self._refresh_control_labels()
            except Exception:
                ok = False
                frame_bgr = None
        if not ok or frame_bgr is None:
            self.preview.set_preview_message("Unable to decode frames for this video.")
            self.stop()
            return
        self._render_frame(frame_bgr)
        self._refresh_time_label()

    def _render_frame(self, frame_bgr: Any) -> None:
        if self._cv2_module is None:
            return
        try:
            rgb = self._cv2_module.cvtColor(frame_bgr, self._cv2_module.COLOR_BGR2RGB)
        except Exception:
            self.preview.set_preview_message("Unable to convert video frames.")
            self.stop()
            return
        height, width, _channels = rgb.shape
        bytes_per_line = rgb.strides[0]
        image = QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()
        self.preview.set_preview_pixmap(QPixmap.fromImage(image))

    def _capture_fps(self) -> float:
        if self._capture is None or self._cv2_module is None:
            return 0.0
        try:
            return float(self._capture.get(self._cv2_module.CAP_PROP_FPS) or 0.0)
        except Exception:
            return 0.0

    def _current_position_ms(self) -> float:
        if self._capture is None or self._cv2_module is None:
            return 0.0
        try:
            value = float(self._capture.get(self._cv2_module.CAP_PROP_POS_MSEC) or 0.0)
            if value > 0:
                return value
        except Exception:
            pass
        fps = self._capture_fps()
        if fps <= 0:
            return 0.0
        try:
            frame_index = float(self._capture.get(self._cv2_module.CAP_PROP_POS_FRAMES) or 0.0)
        except Exception:
            frame_index = 0.0
        return max(0.0, (frame_index / fps) * 1000.0)

    def _calculate_duration_ms(self, capture: Any, fps: float) -> float | None:
        if self._cv2_module is None:
            return None
        frame_count = 0.0
        try:
            frame_count = float(capture.get(self._cv2_module.CAP_PROP_FRAME_COUNT) or 0.0)
        except Exception:
            frame_count = 0.0
        if fps > 0 and frame_count > 0:
            return max(0.0, (frame_count / fps) * 1000.0)
        return None

    def _refresh_control_labels(self) -> None:
        self.play_pause_button.setText("Play" if self._is_paused else "Pause")

    def _refresh_time_label(self) -> None:
        current_text = self._format_media_ms(self._current_position_ms())
        duration_text = self._format_media_ms(self._duration_ms)
        self.time_label.setText(f"{current_text} / {duration_text}")

    def _format_media_ms(self, value: float | None) -> str:
        if value is None or value < 0:
            return "--:--"
        total_seconds = int(value // 1000.0)
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"


class _InputGrid:
    def __init__(self, layout: QVBoxLayout) -> None:
        self._grid = QGridLayout()
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(14)
        self._grid.setVerticalSpacing(10)
        self._grid.setColumnStretch(1, 1)
        self._grid.setColumnStretch(3, 1)
        self._index = 0
        layout.addLayout(self._grid)

    def add_field(self, label_text: str, widget: QWidget) -> None:
        row = self._index // 2
        pair = self._index % 2
        label_col = pair * 2
        widget_col = label_col + 1
        label = QLabel(label_text)
        label.setObjectName("FormLabel")
        self._grid.addWidget(label, row, label_col)
        self._grid.addWidget(widget, row, widget_col)
        self._index += 1


class _CameraSchemaEditor(QFrame):
    _HEADERS = ["Name", "Source", "Type", "Width", "Height", "FPS", "Warmup"]

    def __init__(self, *, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self._syncing_count = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        note = QLabel(
            "Configure all camera runtime behavior here. "
            "Each row defines one named camera and its source index or path."
        )
        note.setWordWrap(True)
        note.setObjectName("MutedLabel")
        layout.addWidget(note)

        defaults_wrap = QWidget()
        defaults_layout = QVBoxLayout(defaults_wrap)
        defaults_layout.setContentsMargins(0, 0, 0, 0)
        defaults_layout.setSpacing(10)

        defaults_form = _InputGrid(defaults_layout)

        self.default_fps_input = QSpinBox()
        self.default_fps_input.setRange(1, 240)
        self.default_fps_input.setValue(int(self.config.get("camera_fps", 30) or 30))
        defaults_form.add_field("Default camera FPS", self.default_fps_input)

        self.default_warmup_input = QSpinBox()
        self.default_warmup_input.setRange(0, 120)
        self.default_warmup_input.setValue(int(self.config.get("camera_warmup_s", 5) or 5))
        defaults_form.add_field("Default camera warmup (s)", self.default_warmup_input)

        self.rename_flag_input = QLineEdit(str(self.config.get("camera_rename_flag", "rename_map")).strip() or "rename_map")
        defaults_form.add_field("Deploy rename-map flag", self.rename_flag_input)

        self.policy_map_input = QLineEdit(str(self.config.get("camera_policy_feature_map_json", "")).strip())
        self.policy_map_input.setPlaceholderText('optional: {"wrist":"camera1","overhead":"camera2"}')
        defaults_form.add_field("Policy feature map", self.policy_map_input)

        defaults_note = QLabel(
            "Default FPS and warmup are used as the baseline for new rows and as fallbacks at runtime. "
            "Deploy preflight uses the runtime camera names below and will suggest a rename map if a model expects different names."
        )
        defaults_note.setWordWrap(True)
        defaults_note.setObjectName("MutedLabel")
        defaults_layout.addWidget(defaults_note)
        layout.addWidget(defaults_wrap)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)
        controls.addWidget(QLabel("Camera count"))

        self.count_input = QSpinBox()
        self.count_input.setRange(1, 16)
        self.count_input.valueChanged.connect(self._handle_count_changed)
        controls.addWidget(self.count_input)

        add_button = QPushButton("Add Camera")
        add_button.clicked.connect(self.add_camera)
        controls.addWidget(add_button)

        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self.remove_selected_camera)
        controls.addWidget(remove_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.table = QTableWidget(0, len(self._HEADERS))
        self.table.setHorizontalHeaderLabels(self._HEADERS)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)

        self.summary_label = QLabel("")
        self.summary_label.setObjectName("MutedLabel")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.set_entries(camera_schema_entries_for_editor(config))

    def _default_row(self, index: int) -> dict[str, Any]:
        width = int(self.config.get("camera_default_width", 640) or 640)
        height = int(self.config.get("camera_default_height", 480) or 480)
        fps = int(self.default_fps_input.value())
        warmup_s = int(self.default_warmup_input.value())
        return {
            "name": f"camera{index}",
            "source": index - 1,
            "camera_type": "opencv",
            "width": width,
            "height": height,
            "fps": fps,
            "warmup_s": warmup_s,
        }

    def _set_row_payload(self, row: int, payload: dict[str, Any]) -> None:
        values = [
            str(payload.get("name", "")),
            str(payload.get("source", "")),
            str(payload.get("camera_type", payload.get("type", "opencv"))),
            str(payload.get("width", "")),
            str(payload.get("height", "")),
            str(payload.get("fps", "")),
            str(payload.get("warmup_s", "")),
        ]
        for column, value in enumerate(values):
            self.table.setItem(row, column, QTableWidgetItem(value))

    def _sync_summary(self) -> None:
        entries = self.entries()
        summary = ", ".join(f"{entry['name']}={entry['source']}" for entry in entries)
        self.summary_label.setText(
            f"Runtime camera names: {summary or '(none)'}\n"
            "Deploy preflight compares these names against trained model camera keys and will suggest a rename map when needed."
        )

    def _sync_count(self) -> None:
        self._syncing_count = True
        try:
            self.count_input.setValue(max(1, self.table.rowCount()))
        finally:
            self._syncing_count = False
        self._sync_summary()

    def _handle_count_changed(self, value: int) -> None:
        if self._syncing_count:
            return
        current = self.table.rowCount()
        if value > current:
            for index in range(current + 1, value + 1):
                self.add_camera(payload=self._default_row(index), sync_count=False)
        elif value < current:
            while self.table.rowCount() > value:
                self.table.removeRow(self.table.rowCount() - 1)
        self._sync_count()

    def set_entries(self, entries: list[Any]) -> None:
        self.table.setRowCount(0)
        for row_index, entry in enumerate(entries):
            payload = {
                "name": getattr(entry, "name", None) if not isinstance(entry, dict) else entry.get("name"),
                "source": getattr(entry, "source", None) if not isinstance(entry, dict) else entry.get("source"),
                "camera_type": getattr(entry, "camera_type", None) if not isinstance(entry, dict) else entry.get("camera_type"),
                "width": getattr(entry, "width", None) if not isinstance(entry, dict) else entry.get("width"),
                "height": getattr(entry, "height", None) if not isinstance(entry, dict) else entry.get("height"),
                "fps": getattr(entry, "fps", None) if not isinstance(entry, dict) else entry.get("fps"),
                "warmup_s": getattr(entry, "warmup_s", None) if not isinstance(entry, dict) else entry.get("warmup_s"),
            }
            self.table.insertRow(row_index)
            self._set_row_payload(row_index, payload)
        if self.table.rowCount() == 0:
            self.add_camera(payload=self._default_row(1), sync_count=False)
        self._sync_count()

    def entries(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in range(self.table.rowCount()):
            def _text(column: int, default: str = "") -> str:
                item = self.table.item(row, column)
                return item.text().strip() if item is not None else default

            rows.append(
                {
                    "name": _text(0, f"camera{row + 1}"),
                    "source": _text(1, str(row)),
                    "camera_type": _text(2, "opencv") or "opencv",
                    "width": _text(3, str(self.config.get("camera_default_width", 640))),
                    "height": _text(4, str(self.config.get("camera_default_height", 480))),
                    "fps": _text(5, str(self.default_fps_input.value())),
                    "warmup_s": _text(6, str(self.default_warmup_input.value())),
                }
            )
        return rows

    def add_camera(self, *, payload: dict[str, Any] | None = None, sync_count: bool = True) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self._set_row_payload(row, payload or self._default_row(row + 1))
        if sync_count:
            self._sync_count()

    def remove_selected_camera(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            row = self.table.rowCount() - 1
        if self.table.rowCount() <= 1 or row < 0:
            return
        self.table.removeRow(row)
        self._sync_count()

    def apply_to_config(self, config: dict[str, Any]) -> dict[str, Any]:
        updated = dict(config)
        updated["camera_fps"] = int(self.default_fps_input.value())
        updated["camera_warmup_s"] = int(self.default_warmup_input.value())
        updated["camera_rename_flag"] = self.rename_flag_input.text().strip() or "rename_map"
        updated["camera_policy_feature_map_json"] = self.policy_map_input.text().strip()
        return apply_camera_schema_entries_to_config(updated, self.entries())

    def reload_from_config(self, config: dict[str, Any]) -> None:
        self.config = config
        self.default_fps_input.setValue(int(config.get("camera_fps", 30) or 30))
        self.default_warmup_input.setValue(int(config.get("camera_warmup_s", 5) or 5))
        self.rename_flag_input.setText(str(config.get("camera_rename_flag", "rename_map")).strip() or "rename_map")
        self.policy_map_input.setText(str(config.get("camera_policy_feature_map_json", "")).strip())
        self.set_entries(camera_schema_entries_for_editor(config))


class _PageWithOutput(QWidget):
    def __init__(self, *, title: str, subtitle: str, append_log: Callable[[str], None]) -> None:
        super().__init__()
        _ = title, subtitle
        self._append_log = append_log

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(18)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)

        self.content_layout = QVBoxLayout()
        self.content_layout.setSpacing(18)
        layout.addLayout(self.content_layout)

        self.output_card, output_layout = _build_card("Output")
        self.status_label = QLabel("Ready.")
        self.status_label.setObjectName("StatusChip")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setMaximumWidth(280)
        output_layout.addWidget(self.status_label)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMinimumHeight(220)
        self.output.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        output_layout.addWidget(self.output)
        layout.addWidget(self.output_card, 1)
        self.output_card.hide()

    def _set_output(self, *, title: str, text: str, log_message: str | None = None) -> None:
        self.status_label.setText(title)
        self.output.setPlainText(text)
        if log_message:
            self._append_log(log_message)

    def _append_output_line(self, line: str) -> None:
        self.output.appendPlainText(str(line))

    def _append_output_and_log(self, line: str) -> None:
        self._append_output_line(line)
        self._append_log(line)


class QtConfigPage(_PageWithOutput):
    _ROBOT_PRESETS: tuple[tuple[str, dict[str, Any]], ...] = (
        (
            "SO-100",
            {
                "follower_robot_type": "so100_follower",
                "leader_robot_type": "so100_leader",
                "follower_robot_action_dim": 6,
            },
        ),
        (
            "SO-101",
            {
                "follower_robot_type": "so101_follower",
                "leader_robot_type": "so101_leader",
                "follower_robot_action_dim": 6,
            },
        ),
        (
            "Unitree G1 (29 DOF)",
            {
                "follower_robot_type": "unitree_g1_29dof",
                "leader_robot_type": "unitree_g1_29dof",
                "follower_robot_action_dim": 29,
            },
        ),
        (
            "Unitree G1 (23 DOF)",
            {
                "follower_robot_type": "unitree_g1_23dof",
                "leader_robot_type": "unitree_g1_23dof",
                "follower_robot_action_dim": 23,
            },
        ),
    )
    _GROUPS = (
        ("Paths", ["lerobot_dir", "lerobot_venv_dir", "runs_dir", "record_data_dir", "deploy_data_dir", "trained_models_dir"]),
        ("Ports + IDs", ["follower_port", "leader_port", "follower_robot_id", "leader_robot_id"]),
        ("Robot Defaults", ["follower_robot_type", "leader_robot_type", "follower_robot_action_dim"]),
        ("Deploy Defaults", ["record_target_hz", "deploy_target_hz", "eval_num_episodes", "eval_duration_s", "eval_task"]),
        ("Calibration + Hub", ["follower_calibration_path", "leader_calibration_path", "hf_username"]),
    )

    def __init__(
        self,
        *,
        config: dict[str, Any],
        append_log: Callable[[str], None],
        run_terminal_command: Callable[[str], tuple[bool, str]] | None = None,
        update_and_restart_app: Callable[[], tuple[bool, str]] | None = None,
    ) -> None:
        super().__init__(
            title="Config",
            subtitle="Edit the most important runtime paths, robot defaults, diagnostics, and launcher settings.",
            append_log=append_log,
        )
        self.config = config
        self._field_lookup = {field["key"]: field for field in CONFIG_FIELDS}
        self._inputs: dict[str, Any] = {}
        self._run_terminal_command = run_terminal_command
        self._update_and_restart_app = update_and_restart_app

        for title, keys in self._GROUPS:
            card, card_layout = _build_card(title)
            form = _InputGrid(card_layout)
            for key in keys:
                field = self._field_lookup[key]
                prompt = field["prompt"]
                current = str(self.config.get(key, "")).strip()
                if field["type"] == "int":
                    widget = QSpinBox()
                    widget.setRange(-1_000_000, 1_000_000)
                    try:
                        widget.setValue(int(self.config.get(key, 0) or 0))
                    except (TypeError, ValueError):
                        widget.setValue(0)
                elif field["type"] == "path":
                    widget = self._build_path_row(key, current)
                else:
                    widget = QLineEdit(current)
                form.add_field(prompt, widget)
                self._inputs[key] = self._input_target(widget)
            self.content_layout.addWidget(card)

        camera_card, camera_layout = _build_card("Camera Setup")
        self.camera_schema_editor = _CameraSchemaEditor(config=self.config)
        camera_layout.addWidget(self.camera_schema_editor)
        self.content_layout.addWidget(camera_card)

        preset_card, preset_layout = _build_card("Robot Presets")
        preset_row = QHBoxLayout()
        self.robot_preset_combo = QComboBox()
        self.robot_preset_combo.addItems([label for label, _values in self._ROBOT_PRESETS])
        preset_row.addWidget(self.robot_preset_combo, 1)
        apply_preset_button = QPushButton("Apply Preset")
        apply_preset_button.clicked.connect(self.apply_robot_preset)
        preset_row.addWidget(apply_preset_button)
        preset_layout.addLayout(preset_row)
        preset_note = QLabel("Prefills the existing robot type and action-dim fields. Everything remains editable afterwards.")
        preset_note.setWordWrap(True)
        preset_note.setObjectName("MutedLabel")
        preset_layout.addWidget(preset_note)
        self.content_layout.addWidget(preset_card)

        actions_card, actions_layout = _build_card("Actions")
        row = QHBoxLayout()
        save_button = QPushButton("Save Config")
        save_button.setObjectName("AccentButton")
        save_button.clicked.connect(self.save_config_values)
        row.addWidget(save_button)

        doctor_button = QPushButton("Run Doctor")
        doctor_button.clicked.connect(self.run_doctor)
        row.addWidget(doctor_button)

        setup_check_button = QPushButton("Run Setup Check")
        setup_check_button.clicked.connect(self.run_setup_check)
        row.addWidget(setup_check_button)

        guide_button = QPushButton("Open Setup Wizard")
        guide_button.clicked.connect(self.show_setup_guide)
        row.addWidget(guide_button)

        launcher_button = QPushButton("Install Launcher")
        launcher_button.clicked.connect(self.install_launcher)
        row.addWidget(launcher_button)
        row.addStretch(1)
        actions_layout.addLayout(row)
        self.content_layout.addWidget(actions_card)
        self.show_snapshot()

    def _build_path_row(self, key: str, current: str) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        line = QLineEdit(current)
        layout.addWidget(line, 1)
        browse = QPushButton("Browse")
        browse.clicked.connect(lambda: self._browse_path_for_field(key, line))
        layout.addWidget(browse)
        return row

    def _input_target(self, widget: QWidget) -> Any:
        if not isinstance(widget, (QLineEdit, QComboBox, QSpinBox)):
            target = widget.findChild(QLineEdit)
            if target is not None:
                return target
        return widget

    def _browse_path_for_field(self, key: str, target: QLineEdit) -> None:
        current = target.text().strip() or str(Path.home())
        if "calibration_path" in key:
            selected, _ = QFileDialog.getOpenFileName(
                self,
                "Select file",
                str(Path(current).expanduser().parent if current else Path.home()),
                "JSON files (*.json);;All files (*)",
            )
        else:
            selected = QFileDialog.getExistingDirectory(self, "Select folder", current)
        if selected:
            target.setText(selected)

    def _read_form(self) -> dict[str, Any]:
        updated = dict(self.config)
        for key, widget in self._inputs.items():
            if isinstance(widget, QComboBox):
                updated[key] = widget.currentText().strip()
            elif isinstance(widget, QSpinBox):
                updated[key] = int(widget.value())
            else:
                updated[key] = widget.text().strip()
        updated = self.camera_schema_editor.apply_to_config(updated)
        return normalize_config_without_prompts(updated)

    def show_snapshot(self) -> None:
        preview = self._read_form()
        status = probe_setup_wizard_status(preview)
        payload = {
            "config_preview": preview,
            "camera_schema_entries": self.camera_schema_editor.entries(),
            "runtime_snapshot": build_compat_snapshot(preview),
            "setup_status": build_setup_status_summary(status),
        }
        self._set_output(title="Config Snapshot", text=_json_text(payload), log_message="Config snapshot refreshed.")

    def apply_robot_preset(self) -> None:
        selected_label = self.robot_preset_combo.currentText().strip()
        for label, values in self._ROBOT_PRESETS:
            if label != selected_label:
                continue
            for key, value in values.items():
                widget = self._inputs.get(key)
                if isinstance(widget, QSpinBox):
                    widget.setValue(int(value))
                elif isinstance(widget, QComboBox):
                    widget.setCurrentText(str(value))
                elif isinstance(widget, QLineEdit):
                    widget.setText(str(value))
            self.show_snapshot()
            self._append_log(f"Applied robot preset: {label}")
            return

    def save_config_values(self) -> None:
        updated = self._read_form()
        self.config.clear()
        self.config.update(updated)
        save_config(self.config, quiet=True)
        self.camera_schema_editor.reload_from_config(self.config)
        self._set_output(title="Config Saved", text=_json_text(updated), log_message="Config values saved.")

    def run_doctor(self) -> None:
        checks = collect_doctor_checks(self._read_form())
        self._set_output(
            title="Doctor",
            text=summarize_checks(checks, title="Doctor"),
            log_message="Doctor run completed.",
        )

    def run_setup_check(self) -> None:
        status = probe_setup_wizard_status(self._read_form())
        self._set_output(
            title="Setup Check",
            text=build_setup_status_summary(status),
            log_message="Setup wizard environment check completed.",
        )

    def _default_activate_command(self) -> str:
        custom = str(self.config.get("setup_venv_activate_cmd", "")).strip()
        if custom:
            return custom
        preview = self._read_form()
        venv_dir = Path(str(preview.get("lerobot_venv_dir", "")).strip()).expanduser()
        return f'source "{venv_dir / "bin" / "activate"}"'

    def _send_activate_command(self, command: str, *, remember: bool) -> tuple[bool, str]:
        if remember:
            self.config["setup_venv_activate_cmd"] = command
            save_config(self.config, quiet=True)
        if self._run_terminal_command is None:
            return False, "Terminal shell is unavailable in this window."
        return self._run_terminal_command(command)

    def show_setup_guide(self) -> None:
        while True:
            status = probe_setup_wizard_status(self._read_form())
            summary = build_setup_status_summary(status)
            guide = build_setup_wizard_guide(status)
            actions: list[tuple[str, str]] = [
                ("activate_venv", "Activate Venv"),
                ("custom_activate", "Custom Source"),
                ("recheck", "Re-check Environment"),
            ]
            if self._update_and_restart_app is not None:
                actions.insert(0, ("update_restart", "Update and Restart"))
            action = ask_text_dialog_with_actions(
                parent=self.window() if isinstance(self.window(), QWidget) else None,
                title="LeRobot Setup Wizard",
                text=f"{summary}\n\n{guide}",
                actions=actions,
                confirm_label="Done",
                cancel_label="Close",
                wrap_mode="word",
            )
            if action == "update_restart" and self._update_and_restart_app is not None:
                ok, message = self._update_and_restart_app()
                self._set_output(
                    title="Update and Restart" if ok else "Update Failed",
                    text=message,
                    log_message="Setup wizard update requested." if ok else "Setup wizard update failed.",
                )
                if ok:
                    return
                continue
            if action == "activate_venv":
                ok, message = self._send_activate_command(self._default_activate_command(), remember=False)
                self._set_output(
                    title="Activation Sent" if ok else "Activation Failed",
                    text=message or self._default_activate_command(),
                    log_message="Setup wizard sent activation command." if ok else "Setup wizard activation failed.",
                )
                continue
            if action == "custom_activate":
                default = self._default_activate_command()
                custom, accepted = QInputDialog.getText(
                    self,
                    "Custom Venv Source",
                    "Enter the venv activation command to run in terminal.",
                    text=default,
                )
                if not accepted:
                    continue
                custom_command = str(custom).strip()
                if not custom_command:
                    continue
                ok, message = self._send_activate_command(custom_command, remember=True)
                self._set_output(
                    title="Activation Sent" if ok else "Activation Failed",
                    text=message or custom_command,
                    log_message="Setup wizard sent custom activation command." if ok else "Setup wizard custom activation failed.",
                )
                continue
            if action == "recheck":
                self.run_setup_check()
                continue
            self._append_log("Setup guide rendered.")
            self.status_label.setText("Setup Guide")
            return

    def install_launcher(self) -> None:
        updated = self._read_form()
        self.config.clear()
        self.config.update(updated)
        result = install_desktop_launcher(
            app_dir=Path(__file__).resolve().parents[1],
            python_executable=Path(sys.executable),
            venv_dir=Path(str(self.config.get("lerobot_venv_dir", "") or "")).expanduser(),
        )
        text = result.message
        if result.script_path is not None:
            text += f"\n- launcher script: {result.script_path}"
        if result.desktop_entry_path is not None:
            text += f"\n- desktop entry: {result.desktop_entry_path}"
        if result.icon_path is not None:
            text += f"\n- icon: {result.icon_path}"
        self._set_output(
            title="Launcher Installed" if result.ok else "Launcher Failed",
            text=text,
            log_message="Launcher install completed." if result.ok else "Launcher install failed.",
        )


class QtVisualizerPage(_PageWithOutput):
    _ROOT_STATE_KEYS = {
        "deployments": "ui_visualizer_deploy_root",
        "datasets": "ui_visualizer_dataset_root",
        "models": "ui_visualizer_model_root",
    }

    def __init__(self, *, config: dict[str, Any], append_log: Callable[[str], None]) -> None:
        super().__init__(
            title="Visualizer",
            subtitle="Browse local deployment runs, datasets, models, and discovered video assets.",
            append_log=append_log,
        )
        self.config = config
        self._sources: list[dict[str, Any]] = []
        self._video_tiles: list[_VideoGalleryTile] = []
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self._current_source_kind = self._persisted_source_kind()

        self.content_layout.addWidget(self._build_controls_card())
        self.content_layout.addWidget(self._build_video_gallery_card())
        self.content_layout.addWidget(self._build_sources_card())
        self.content_layout.addWidget(self._build_details_card())
        self.content_layout.addWidget(self._build_insights_card())
        self.content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.content_layout.addStretch(1)

        self._restore_persisted_visualizer_state()
        self.source_combo.currentIndexChanged.connect(self._handle_source_changed)
        self.hf_owner_input.editingFinished.connect(self._persist_visualizer_state)
        self.root_input.editingFinished.connect(self._persist_visualizer_state)
        self._handle_source_changed()

    def _build_controls_card(self) -> QFrame:
        card, layout = _build_card("Source Browser")

        top_row = QHBoxLayout()
        self.source_combo = QComboBox()
        self.source_combo.addItem("Deployments", "deployments")
        self.source_combo.addItem("Datasets", "datasets")
        self.source_combo.addItem("Models", "models")
        top_row.addWidget(QLabel("Source"))
        top_row.addWidget(self.source_combo)

        self.root_input = QLineEdit(self._root_text_for_source(self._current_source_kind))
        top_row.addWidget(QLabel("Root"))
        top_row.addWidget(self.root_input, 1)

        self.hf_owner_input = QLineEdit(str(self.config.get("ui_visualizer_hf_owner", self.config.get("hf_username", ""))).strip())
        top_row.addWidget(QLabel("HF owner"))
        top_row.addWidget(self.hf_owner_input)

        browse_root_button = QPushButton("Browse Root")
        browse_root_button.clicked.connect(self.browse_root)
        top_row.addWidget(browse_root_button)

        refresh_button = QPushButton("Refresh")
        refresh_button.setObjectName("AccentButton")
        refresh_button.clicked.connect(self.refresh_sources)
        top_row.addWidget(refresh_button)
        layout.addLayout(top_row)

        actions = QHBoxLayout()
        open_source_button = QPushButton("Open Source")
        open_source_button.clicked.connect(self.open_selected_source)
        actions.addWidget(open_source_button)
        actions.addStretch(1)
        layout.addLayout(actions)
        return card

    def _build_sources_card(self) -> QFrame:
        card, layout = _build_card("Sources")
        self.source_table = QTableWidget(0, 2)
        _set_table_headers(self.source_table, ["Source", "Name"])
        _set_readonly_table(self.source_table)
        self.source_table.setMinimumHeight(180)
        self.source_table.setMaximumHeight(280)
        self.source_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.source_table.itemSelectionChanged.connect(self._on_source_selection_changed)
        layout.addWidget(self.source_table)
        return card

    def _build_details_card(self) -> QFrame:
        card, layout = _build_card("Selection Details")
        self.meta_view = QPlainTextEdit()
        self.meta_view.setReadOnly(True)
        self.meta_view.setMinimumHeight(140)
        self.meta_view.setMaximumHeight(220)
        self.meta_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.meta_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self.meta_view)
        return card

    def _build_insights_card(self) -> QFrame:
        self.insights_card, layout = _build_card("Deployment Insights")
        self.insights_table = QTableWidget(0, 4)
        _set_table_headers(self.insights_table, ["Episode", "Result", "Notes", "Tags"])
        _set_readonly_table(self.insights_table)
        layout.addWidget(self.insights_table)
        self.insights_card.hide()
        return self.insights_card

    def _build_video_gallery_card(self) -> QFrame:
        self.video_gallery_card, layout = _build_card("Video Gallery")
        self.video_status = QLabel("Select a source to display discovered videos.")
        self.video_status.setObjectName("MutedLabel")
        self.video_status.setWordWrap(True)
        layout.addWidget(self.video_status)

        self.video_grid_host = QWidget()
        self.video_grid = QGridLayout(self.video_grid_host)
        self.video_grid.setContentsMargins(0, 0, 0, 0)
        self.video_grid.setHorizontalSpacing(14)
        self.video_grid.setVerticalSpacing(14)
        self.video_grid_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.video_scroll = QScrollArea()
        self.video_scroll.setWidgetResizable(True)
        self.video_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.video_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.video_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_scroll.setMinimumHeight(360)
        self.video_scroll.setWidget(self.video_grid_host)
        layout.addWidget(self.video_scroll, 1)
        return self.video_gallery_card

    def _handle_source_changed(self, *_args: object) -> None:
        new_source = self._active_source()
        if self._current_source_kind != new_source:
            self._persist_visualizer_root(self._current_source_kind, self.root_input.text().strip())
        self._current_source_kind = new_source
        self._sync_root_placeholder()
        self._set_insights_visible(False)
        self._persist_visualizer_state()
        self.refresh_sources()

    def browse_root(self) -> None:
        current = self.root_input.text().strip() or str(Path.home())
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select Visualizer Root",
            current,
        )
        if not selected:
            return
        self.root_input.setText(selected)
        self._persist_visualizer_state()
        self.refresh_sources()

    def _active_source(self) -> str:
        return str(self.source_combo.currentData() or "deployments")

    def _persisted_source_kind(self) -> str:
        value = str(self.config.get("ui_visualizer_source_kind", "deployments")).strip().lower()
        if value in {"deployments", "datasets", "models"}:
            return value
        return "deployments"

    def _default_root_for_source(self, source: str) -> str:
        if source == "deployments":
            return str(self.config.get("deploy_data_dir", get_deploy_data_dir(self.config)))
        if source == "datasets":
            return str(self.config.get("record_data_dir", get_lerobot_dir(self.config) / "data"))
        return str(self.config.get("trained_models_dir", get_lerobot_dir(self.config) / "trained_models"))

    def _root_text_for_source(self, source: str) -> str:
        key = self._ROOT_STATE_KEYS.get(source, "")
        persisted = str(self.config.get(key, "")).strip()
        return persisted or self._default_root_for_source(source)

    def _restore_persisted_visualizer_state(self) -> None:
        source = self._persisted_source_kind()
        index = self.source_combo.findData(source)
        blocked = self.source_combo.blockSignals(True)
        if index >= 0:
            self.source_combo.setCurrentIndex(index)
        self.source_combo.blockSignals(blocked)
        self.root_input.setText(self._root_text_for_source(source))
        self._current_source_kind = source

    def _persist_visualizer_root(self, source: str, root_text: str) -> None:
        key = self._ROOT_STATE_KEYS.get(str(source).strip(), "")
        if not key:
            return
        value = str(root_text).strip()
        if value:
            self.config[key] = value
        else:
            self.config.pop(key, None)

    def _source_identity(self, source: dict[str, Any] | None) -> tuple[str, str, str]:
        if not isinstance(source, dict):
            return "", "", ""
        return (
            str(source.get("scope", "")).strip(),
            str(source.get("kind", "")).strip(),
            str(source.get("name", source.get("repo_id", source.get("path", "")))).strip(),
        )

    def _preferred_source_identity(self) -> tuple[str, str, str]:
        current_source = self._current_source()
        current_identity = self._source_identity(current_source)
        if any(current_identity):
            return current_identity
        return (
            str(self.config.get("ui_visualizer_selected_scope", "")).strip(),
            str(self.config.get("ui_visualizer_selected_kind", "")).strip(),
            str(self.config.get("ui_visualizer_selected_name", "")).strip(),
        )

    def _restore_source_selection(self, preferred_identity: tuple[str, str, str]) -> bool:
        if not any(preferred_identity):
            return False
        for row, source in enumerate(self._sources):
            if self._source_identity(source) == preferred_identity:
                self.source_table.selectRow(row)
                return True
        return False

    def _persist_visualizer_state(self) -> None:
        self.config["ui_visualizer_source_kind"] = self._active_source()
        self.config["ui_visualizer_hf_owner"] = self.hf_owner_input.text().strip()
        self._persist_visualizer_root(self._active_source(), self.root_input.text().strip())
        scope, kind, name = self._source_identity(self._current_source())
        if scope and kind and name:
            self.config["ui_visualizer_selected_scope"] = scope
            self.config["ui_visualizer_selected_kind"] = kind
            self.config["ui_visualizer_selected_name"] = name
        else:
            self.config.pop("ui_visualizer_selected_scope", None)
            self.config.pop("ui_visualizer_selected_kind", None)
            self.config.pop("ui_visualizer_selected_name", None)
        save_config(self.config, quiet=True)

    def _sync_root_placeholder(self) -> None:
        self.root_input.setText(self._root_text_for_source(self._active_source()))

    def _set_insights_visible(self, visible: bool) -> None:
        self.insights_card.setVisible(bool(visible))

    def _clear_video_tiles(self) -> None:
        for tile in self._video_tiles:
            tile.stop()
        self._video_tiles.clear()
        while self.video_grid.count():
            item = self.video_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._refresh_scroll_geometry()

    def _video_gallery_columns(self, video_count: int) -> int:
        if video_count <= 1:
            return 1
        if video_count <= 4:
            return 2
        return 3

    def _start_video_tiles(self) -> None:
        for tile in self._video_tiles:
            tile.start()

    def _stop_video_tiles(self) -> None:
        for tile in self._video_tiles:
            tile.stop()

    def _render_video_gallery(self, videos: list[dict[str, Any]], *, empty_message: str) -> None:
        self._clear_video_tiles()
        video_items = list(videos)
        if not video_items:
            self.video_status.setText(empty_message)
            empty = QLabel(empty_message)
            empty.setWordWrap(True)
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setObjectName("MutedLabel")
            self.video_grid.addWidget(empty, 0, 0)
            self._refresh_scroll_geometry()
            return

        count = len(video_items)
        self.video_status.setText(f"Displaying {count} video{'s' if count != 1 else ''} from the selected source.")
        columns = self._video_gallery_columns(count)
        for column in range(columns):
            self.video_grid.setColumnStretch(column, 1)
        for index, item in enumerate(video_items):
            tile = _VideoGalleryTile(item=item)
            self._video_tiles.append(tile)
            row = index // columns
            column = index % columns
            self.video_grid.addWidget(tile, row, column)
        self._refresh_scroll_geometry()
        if self.isVisible():
            self._start_video_tiles()

    def _refresh_scroll_geometry(self) -> None:
        self.video_grid_host.updateGeometry()
        self.video_grid_host.adjustSize()
        self.video_gallery_card.updateGeometry()
        self.video_gallery_card.adjustSize()
        self.updateGeometry()
        self.adjustSize()
        QTimer.singleShot(0, self._refresh_parent_scroll_area)

    def _refresh_parent_scroll_area(self) -> None:
        parent = self.parentWidget()
        while parent is not None:
            if isinstance(parent, QScrollArea):
                widget = parent.widget()
                if widget is not None:
                    widget.updateGeometry()
                    widget.adjustSize()
                parent.viewport().update()
                break
            parent = parent.parentWidget()

    def _snapshot(self) -> _VisualizerRefreshSnapshot:
        source = self._active_source()
        root_text = self.root_input.text().strip()
        if source == "deployments":
            deploy_root = root_text or str(get_deploy_data_dir(self.config))
            dataset_root = str(self.config.get("record_data_dir", get_lerobot_dir(self.config) / "data"))
            model_root = str(self.config.get("trained_models_dir", get_lerobot_dir(self.config) / "trained_models"))
        elif source == "datasets":
            deploy_root = str(self.config.get("deploy_data_dir", get_deploy_data_dir(self.config)))
            dataset_root = root_text or str(self.config.get("record_data_dir", get_lerobot_dir(self.config) / "data"))
            model_root = str(self.config.get("trained_models_dir", get_lerobot_dir(self.config) / "trained_models"))
        else:
            deploy_root = str(self.config.get("deploy_data_dir", get_deploy_data_dir(self.config)))
            dataset_root = str(self.config.get("record_data_dir", get_lerobot_dir(self.config) / "data"))
            model_root = root_text or str(self.config.get("trained_models_dir", get_lerobot_dir(self.config) / "trained_models"))
        return _VisualizerRefreshSnapshot(
            source=source,
            deploy_root=deploy_root,
            dataset_root=dataset_root,
            model_root=model_root,
            hf_owner=self.hf_owner_input.text().strip(),
        )

    def refresh_sources(self) -> None:
        preferred_identity = self._preferred_source_identity()
        snapshot = self._snapshot()
        sources, error_text, source_kind = _collect_sources_for_refresh(self.config, snapshot)
        self._sources = list(sources)
        self.source_table.setRowCount(len(self._sources))
        for row, source in enumerate(self._sources):
            scope_text, name_text = _visualizer_source_row_values(source)
            self.source_table.setItem(row, 0, QTableWidgetItem(scope_text))
            self.source_table.setItem(row, 1, QTableWidgetItem(name_text))
        self.meta_view.setPlainText("")
        self.insights_table.setRowCount(0)
        self._set_insights_visible(False)
        self._render_video_gallery([], empty_message="Select a source to display discovered videos.")
        if self._sources:
            self._set_output(
                title="Sources Loaded",
                text=f"Loaded {len(self._sources)} {source_kind}.",
                log_message=f"Visualizer refreshed {len(self._sources)} {source_kind}.",
            )
            if not self._restore_source_selection(preferred_identity):
                self.source_table.selectRow(0)
        else:
            detail = error_text or f"No {source_kind} were found."
            self._set_output(title="No Sources", text=detail, log_message="Visualizer refresh returned no sources.")
            self._persist_visualizer_state()

    def _current_source(self) -> dict[str, Any] | None:
        row = self.source_table.currentRow()
        if row < 0 or row >= len(self._sources):
            return None
        return self._sources[row]

    def _on_source_selection_changed(self) -> None:
        source = self._current_source()
        if source is None:
            self.meta_view.setPlainText("")
            self.insights_table.setRowCount(0)
            self._set_insights_visible(False)
            self._render_video_gallery([], empty_message="Select a source to display discovered videos.")
            self._persist_visualizer_state()
            return
        payload = _build_selection_payload(source)
        text = _json_text(payload.get("meta_payload", {}))
        self.meta_view.setPlainText(text)
        self.insights_table.setRowCount(0)
        show_insights = self._active_source() == "deployments" and bool(payload.get("insights_visible"))
        self._set_insights_visible(show_insights)
        if show_insights:
            self.insights_table.setRowCount(len(payload.get("insights_rows", [])))
            for row_index, row in enumerate(payload.get("insights_rows", [])):
                if not isinstance(row, tuple) or len(row) != 4:
                    continue
                for col_index, value in enumerate(row):
                    self.insights_table.setItem(row_index, col_index, QTableWidgetItem(str(value)))
        if show_insights:
            text += "\n\n" + str(payload.get("insights_header", "Deployment Insights"))
            for row in payload.get("insights_rows", []):
                if isinstance(row, tuple) and len(row) == 4:
                    text += f"\n- Episode {row[0]} | {row[1]} | {row[2]} | {row[3]}"
        self._set_output(title="Selection Details", text=text, log_message=None)
        self._render_video_gallery(
            list(payload.get("videos", [])),
            empty_message="No videos found for the selected source.",
        )
        self._persist_visualizer_state()

    def open_selected_source(self) -> None:
        source = self._current_source()
        if source is None:
            self._set_output(title="No Selection", text="Select a source first.", log_message="Visualizer source open skipped with no selection.")
            return
        target = source.get("path")
        if not target and source.get("repo_id"):
            repo_id = str(source.get("repo_id"))
            if str(source.get("kind")) == "dataset":
                target = f"https://huggingface.co/datasets/{repo_id}"
            else:
                target = f"https://huggingface.co/{repo_id}"
        ok, message = _open_path(target or "")
        self._set_output(title="Open Source" if ok else "Open Failed", text=str(target or message), log_message="Visualizer opened source." if ok else "Visualizer source open failed.")

    def showEvent(self, event: object) -> None:
        super().showEvent(event)  # type: ignore[misc]
        self._start_video_tiles()

    def hideEvent(self, event: object) -> None:
        self._stop_video_tiles()
        super().hideEvent(event)  # type: ignore[misc]

    def refresh_from_config(self) -> None:
        self._restore_persisted_visualizer_state()
        self.refresh_sources()


class QtHistoryPage(_PageWithOutput):
    def __init__(
        self,
        *,
        config: dict[str, Any],
        append_log: Callable[[str], None],
        run_controller: ManagedRunController,
    ) -> None:
        super().__init__(
            title="History",
            subtitle="Browse run artifacts, open logs, and rerun prior commands from the main shell.",
            append_log=append_log,
        )
        self.config = config
        self._run_controller = run_controller
        self._rows: list[dict[str, Any]] = []
        self._action_buttons: list[QPushButton] = []

        filters_card, filters_layout = _build_card("Filters")
        row = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("All modes", "all")
        for mode in HISTORY_MODE_VALUES:
            self.mode_combo.addItem(mode, mode)
        row.addWidget(self.mode_combo)

        self.status_combo = QComboBox()
        self.status_combo.addItem("All statuses", "all")
        self.status_combo.addItem("Success", "success")
        self.status_combo.addItem("Failed", "failed")
        self.status_combo.addItem("Canceled", "canceled")
        row.addWidget(self.status_combo)

        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Filter by hint, command text, or timestamp")
        row.addWidget(self.query_input, 1)

        refresh_button = QPushButton("Refresh")
        refresh_button.setObjectName("AccentButton")
        refresh_button.clicked.connect(self.refresh_history)
        row.addWidget(refresh_button)
        self._action_buttons.append(refresh_button)
        filters_layout.addLayout(row)

        actions = QHBoxLayout()
        self.open_run_button = QPushButton("Open Run Folder")
        self.open_run_button.clicked.connect(self.open_run_folder)
        actions.addWidget(self.open_run_button)
        self._action_buttons.append(self.open_run_button)

        self.open_log_button = QPushButton("Open Command Log")
        self.open_log_button.clicked.connect(self.open_command_log)
        actions.addWidget(self.open_log_button)
        self._action_buttons.append(self.open_log_button)

        self.rerun_button = QPushButton("Rerun Selected")
        self.rerun_button.clicked.connect(self.rerun_selected)
        actions.addWidget(self.rerun_button)
        self._action_buttons.append(self.rerun_button)

        actions.addStretch(1)
        filters_layout.addLayout(actions)
        self.content_layout.addWidget(filters_card)

        runs_card, runs_layout = _build_card("Runs")
        self.run_table = QTableWidget(0, 6)
        _set_table_headers(self.run_table, ["Started", "Duration", "Mode", "Status", "Hint", "Command"])
        _set_readonly_table(self.run_table)
        self.run_table.itemSelectionChanged.connect(self._on_selection_changed)
        runs_layout.addWidget(self.run_table)
        self.content_layout.addWidget(runs_card)

        deploy_card, deploy_layout = _build_card("Deploy Outcome + Notes Editor")
        self.deploy_editor_card = deploy_card
        deploy_form = _InputGrid(deploy_layout)

        self.episode_combo = QComboBox()
        self.episode_combo.currentIndexChanged.connect(self._sync_episode_editor_fields)
        deploy_form.add_field("Episode", self.episode_combo)

        self.outcome_combo = QComboBox()
        self.outcome_combo.addItems(["success", "failed", "unmarked"])
        deploy_form.add_field("Status", self.outcome_combo)

        self.tags_input = QLineEdit()
        self.tags_input.setPlaceholderText("comma,separated,tags")
        deploy_form.add_field("Tags", self.tags_input)

        self.episode_note_input = QLineEdit()
        self.episode_note_input.setPlaceholderText("Optional note for this episode")
        deploy_form.add_field("Episode note", self.episode_note_input)

        notes_label = QLabel("Deployment overall notes")
        notes_label.setObjectName("FormLabel")
        deploy_layout.addWidget(notes_label)

        self.overall_notes = QPlainTextEdit()
        self.overall_notes.setMinimumHeight(110)
        deploy_layout.addWidget(self.overall_notes)

        editor_actions = QHBoxLayout()
        self.save_episode_button = QPushButton("Save Episode Edit")
        self.save_episode_button.clicked.connect(self.save_episode_edit)
        editor_actions.addWidget(self.save_episode_button)
        self._action_buttons.append(self.save_episode_button)

        self.save_notes_button = QPushButton("Save Deployment Notes")
        self.save_notes_button.clicked.connect(self.save_deployment_notes)
        editor_actions.addWidget(self.save_notes_button)
        self._action_buttons.append(self.save_notes_button)

        self.open_notes_button = QPushButton("Open notes.md")
        self.open_notes_button.clicked.connect(self.open_deploy_notes_file)
        editor_actions.addWidget(self.open_notes_button)
        self._action_buttons.append(self.open_notes_button)
        editor_actions.addStretch(1)
        deploy_layout.addLayout(editor_actions)

        self.deploy_editor_status = QLabel("")
        self.deploy_editor_status.setObjectName("MutedLabel")
        self.deploy_editor_status.setWordWrap(True)
        deploy_layout.addWidget(self.deploy_editor_status)
        self.content_layout.addWidget(deploy_card)
        self.deploy_editor_card.hide()

        self._restore_history_filters()
        self.mode_combo.currentIndexChanged.connect(self._handle_history_filter_changed)
        self.status_combo.currentIndexChanged.connect(self._handle_history_filter_changed)
        self.query_input.textChanged.connect(self._handle_history_query_changed)
        self.refresh_history()

    def _restore_history_filters(self) -> None:
        mode_value = str(self.config.get("ui_history_mode_filter", "all")).strip()
        status_value = str(self.config.get("ui_history_status_filter", "all")).strip()
        query_value = str(self.config.get("ui_history_query", "")).strip()
        mode_index = self.mode_combo.findData(mode_value)
        status_index = self.status_combo.findData(status_value)
        if mode_index >= 0:
            self.mode_combo.setCurrentIndex(mode_index)
        if status_index >= 0:
            self.status_combo.setCurrentIndex(status_index)
        self.query_input.setText(query_value)

    def _persist_history_filters(self) -> None:
        self.config["ui_history_mode_filter"] = str(self.mode_combo.currentData() or "all")
        self.config["ui_history_status_filter"] = str(self.status_combo.currentData() or "all")
        self.config["ui_history_query"] = self.query_input.text().strip()
        save_config(self.config, quiet=True)

    def _handle_history_filter_changed(self, *_args: object) -> None:
        self._persist_history_filters()
        self.refresh_history()

    def _handle_history_query_changed(self, _text: str) -> None:
        self._persist_history_filters()
        self.refresh_history()

    def _current_row(self) -> dict[str, Any] | None:
        row = self.run_table.currentRow()
        if row < 0 or row >= len(self._rows):
            return None
        return self._rows[row]

    def _set_running(self, active: bool, status_text: str | None = None, is_error: bool = False) -> None:
        for button in self._action_buttons:
            button.setEnabled(not active)
        if active:
            self.status_label.setText(status_text or "Running rerun...")
        elif is_error:
            self.status_label.setText(status_text or "Rerun failed.")
        else:
            self.status_label.setText(status_text or "Ready.")

    def refresh_history(self) -> None:
        runs, warning_count = list_runs(config=self.config, limit=5000)
        payload = _build_history_refresh_payload_from_runs(
            runs=runs,
            warning_count=warning_count,
            mode_filter=str(self.mode_combo.currentData() or "all"),
            status_filter=str(self.status_combo.currentData() or "all"),
            query=self.query_input.text().strip().lower(),
        )
        self._rows = list(payload.get("rows", []))
        self.run_table.setRowCount(len(self._rows))
        for row_index, row in enumerate(self._rows):
            values = list(row.get("values", ()))
            while len(values) < 6:
                values.append("")
            for col_index, value in enumerate(values[:6]):
                self.run_table.setItem(row_index, col_index, QTableWidgetItem(str(value)))
        if self._rows:
            self.run_table.selectRow(0)
        stats = payload.get("stats", {})
        self._set_output(
            title="History Refreshed",
            text=_json_text({"stats": stats, "warning_count": warning_count}),
            log_message=f"History refreshed {stats.get('total', 0)} rows.",
        )

    def _on_selection_changed(self) -> None:
        row = self._current_row()
        if row is None:
            self.deploy_editor_card.hide()
            return
        item = row.get("item", {})
        self._set_output(
            title="Run Details",
            text=_json_text(item),
            log_message=None,
        )
        self._populate_deploy_editor()

    def open_run_folder(self) -> None:
        row = self._current_row()
        if row is None:
            self._set_output(title="No Selection", text="Select a run first.", log_message="History open-run skipped with no selection.")
            return
        run_path = Path(str(row.get("item", {}).get("_run_path", "")))
        ok, message = open_path_in_file_manager(run_path)
        self._set_output(title="Open Run" if ok else "Open Failed", text=str(run_path if ok else message), log_message="History opened run folder." if ok else "History failed to open run folder.")

    def open_command_log(self) -> None:
        row = self._current_row()
        if row is None:
            self._set_output(title="No Selection", text="Select a run first.", log_message="History open-log skipped with no selection.")
            return
        run_path = Path(str(row.get("item", {}).get("_run_path", "")))
        log_path = run_path / "command.log"
        ok, message = open_path_in_file_manager(log_path)
        self._set_output(title="Open Log" if ok else "Open Failed", text=str(log_path if ok else message), log_message="History opened command log." if ok else "History failed to open command log.")

    def rerun_selected(self) -> None:
        row = self._current_row()
        if row is None:
            self._set_output(title="No Selection", text="Select a run first.", log_message="History rerun skipped with no selection.")
            return
        item = row.get("item", {})
        cmd, error = _command_from_item(item)
        if cmd is None:
            self._set_output(title="Rerun Failed", text=error or "No command stored in this history entry.", log_message="History rerun failed while parsing command.")
            return

        run_mode = str(item.get("mode", "run")).strip().lower() or "run"
        cwd_raw = str(item.get("cwd", "")).strip()
        cwd = Path(cwd_raw) if cwd_raw else None
        artifact_context: dict[str, Any] = {}
        dataset_repo_id = str(item.get("dataset_repo_id", "")).strip()
        model_path = str(item.get("model_path", "")).strip()
        if dataset_repo_id:
            artifact_context["dataset_repo_id"] = dataset_repo_id
        if model_path:
            artifact_context["model_path"] = model_path

        rerun_cmd = list(cmd)
        if run_mode == "deploy":
            rerun_cmd, rerun_message = normalize_deploy_rerun_command(
                command_argv=rerun_cmd,
                username=str(self.config.get("hf_username", "")),
                local_roots=[get_deploy_data_dir(self.config), get_lerobot_dir(self.config) / "data"],
            )
            if rerun_message:
                self._append_output_and_log(rerun_message)
                for arg in rerun_cmd:
                    if str(arg).startswith("--dataset.repo_id="):
                        artifact_context["dataset_repo_id"] = str(arg).split("=", 1)[1].strip()
                        break

        self.output.setPlainText("Rerunning stored command...\n")
        self._append_output_line(" ".join(rerun_cmd))
        hooks = RunUiHooks(
            set_running=self._set_running,
            append_output_line=self._append_output_line,
        )
        ok, message = self._run_controller.run_process_async(
            cmd=rerun_cmd,
            cwd=cwd,
            hooks=hooks,
            complete_callback=None,
            run_mode=run_mode,
            preflight_checks=None,
            artifact_context=artifact_context,
        )
        if not ok:
            self._set_output(title="Rerun Rejected", text=message or "Unable to rerun selected command.", log_message="History rerun was rejected.")
            return
        self._append_log(f"History rerun started for {run_mode}.")

    def _read_selected_metadata(self) -> tuple[dict[str, Any] | None, Path | None, dict[str, Any] | None]:
        row = self._current_row()
        if row is None:
            return None, None, None
        item = row.get("item", {})
        metadata_path_raw = str(item.get("_metadata_path", "")).strip()
        if not metadata_path_raw:
            return item, None, None
        metadata_path = Path(metadata_path_raw)
        if not metadata_path.exists():
            return item, metadata_path, None
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return item, metadata_path, None
        if not isinstance(data, dict):
            return item, metadata_path, None
        data["_metadata_path"] = str(metadata_path)
        data["_run_path"] = str(metadata_path.parent)
        return item, metadata_path, data

    def _deploy_episode_map(self, item: dict[str, Any]) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
        summary = _normalize_deploy_episode_outcomes(item.get("deploy_episode_outcomes"))
        episode_map: dict[int, dict[str, Any]] = {}
        entries = summary.get("episode_outcomes")
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                try:
                    episode_idx = int(entry.get("episode"))
                except (TypeError, ValueError):
                    continue
                if episode_idx <= 0:
                    continue
                episode_map[episode_idx] = dict(entry)
        return summary, episode_map

    def _episode_choices(self, summary: dict[str, Any], episode_map: dict[int, dict[str, Any]]) -> list[str]:
        total = summary.get("total_episodes")
        if isinstance(total, int) and total > 0:
            return [str(value) for value in range(1, total + 1)]
        if episode_map:
            return [str(value) for value in sorted(episode_map)]
        return ["1"]

    def _populate_deploy_editor(self) -> None:
        row = self._current_row()
        if row is None:
            self.deploy_editor_card.hide()
            return
        item = row.get("item", {})
        if str(item.get("mode", "")).strip().lower() != "deploy":
            self.deploy_editor_card.hide()
            return
        self.deploy_editor_card.show()
        summary, episode_map = self._deploy_episode_map(item)
        choices = self._episode_choices(summary, episode_map)
        self.episode_combo.blockSignals(True)
        self.episode_combo.clear()
        self.episode_combo.addItems(choices)
        self.episode_combo.setCurrentText(choices[0])
        self.episode_combo.blockSignals(False)
        self.overall_notes.setPlainText(str(item.get("deploy_notes_summary", "")).strip())
        self.deploy_editor_status.setText("Edit episode status/tags/note and deployment notes, then save.")
        self._sync_episode_editor_fields()

    def _sync_episode_editor_fields(self) -> None:
        row = self._current_row()
        if row is None:
            return
        item = row.get("item", {})
        summary, episode_map = self._deploy_episode_map(item)
        try:
            current_episode = int(self.episode_combo.currentText().strip() or "1")
        except ValueError:
            current_episode = 1
        if not episode_map and not summary:
            current_entry: dict[str, Any] = {}
        else:
            current_entry = episode_map.get(current_episode, {})
        self.outcome_combo.setCurrentText(normalize_deploy_result(current_entry.get("result", "unmarked")))
        tags = current_entry.get("tags") if isinstance(current_entry.get("tags"), list) else []
        self.tags_input.setText(", ".join(str(tag) for tag in tags))
        self.episode_note_input.setText(str(current_entry.get("note", "")).strip())

    def _persist_deploy_metadata(self, updated_data: dict[str, Any]) -> tuple[bool, str]:
        metadata_path_raw = str(updated_data.get("_metadata_path", "")).strip()
        run_path_raw = str(updated_data.get("_run_path", "")).strip()
        if not metadata_path_raw or not run_path_raw:
            return False, "Selected run is missing metadata/run path references."
        metadata_path = Path(metadata_path_raw)
        run_path = Path(run_path_raw)
        payload = dict(updated_data)
        payload.pop("_metadata_path", None)
        payload.pop("_run_path", None)
        try:
            _atomic_write(json.dumps(payload, indent=2) + "\n", metadata_path)
        except OSError as exc:
            return False, f"Failed to write metadata.json: {exc}"
        notes_path = write_deploy_notes_file(run_path, payload, filename="notes.md")
        if notes_path is None:
            return False, "Saved metadata, but failed to write notes.md"
        csv_path, summary_csv_path = write_deploy_episode_spreadsheet(
            run_path,
            payload,
            filename="episode_outcomes.csv",
            summary_filename="episode_outcomes_summary.csv",
        )
        if csv_path is None or summary_csv_path is None:
            return False, "Saved metadata and notes, but failed to write episode CSV files."
        row = self._current_row()
        if row is not None:
            item = row.get("item", {})
            item.clear()
            item.update(payload)
            item["_metadata_path"] = str(metadata_path)
            item["_run_path"] = str(run_path)
        return True, f"Saved deploy edits: {notes_path.name}, {csv_path.name}, {summary_csv_path.name}"

    def save_episode_edit(self) -> None:
        item, _metadata_path, metadata_data = self._read_selected_metadata()
        if item is None:
            self._set_output(title="No Selection", text="Select a run first.", log_message="History episode save skipped with no selection.")
            return
        if str(item.get("mode", "")).strip().lower() != "deploy":
            self._set_output(title="History", text="Deploy episode editing is only available for deploy runs.", log_message="History episode save skipped for non-deploy run.")
            return
        if metadata_data is None:
            self._set_output(title="History", text="Unable to read metadata.json for this run.", log_message="History episode save failed due to unreadable metadata.")
            return
        try:
            episode_idx = int(self.episode_combo.currentText().strip() or "1")
        except ValueError:
            episode_idx = 1
        summary = _normalize_deploy_episode_outcomes(metadata_data.get("deploy_episode_outcomes"))
        entries = summary.get("episode_outcomes")
        episode_map: dict[int, dict[str, Any]] = {}
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict):
                    try:
                        entry_idx = int(entry.get("episode"))
                    except (TypeError, ValueError):
                        continue
                    episode_map[entry_idx] = dict(entry)
        entry = episode_map.get(episode_idx, {"episode": episode_idx})
        entry["episode"] = episode_idx
        entry["result"] = normalize_deploy_result(self.outcome_combo.currentText())
        tags = [tag.strip() for tag in self.tags_input.text().split(",") if tag.strip()]
        entry["tags"] = tags
        note = self.episode_note_input.text().strip()
        if note:
            entry["note"] = note
        else:
            entry.pop("note", None)
        episode_map[episode_idx] = entry
        total = summary.get("total_episodes")
        if not isinstance(total, int) or total < episode_idx:
            total = episode_idx
        metadata_data["deploy_episode_outcomes"] = _normalize_deploy_episode_outcomes(
            {"enabled": True, "total_episodes": total, "episode_outcomes": [episode_map[idx] for idx in sorted(episode_map)]}
        )
        ok, message = self._persist_deploy_metadata(metadata_data)
        self.deploy_editor_status.setText(message)
        self._set_output(title="Episode Edit Saved" if ok else "Save Failed", text=message, log_message=message)
        if ok:
            self._populate_deploy_editor()

    def save_deployment_notes(self) -> None:
        item, _metadata_path, metadata_data = self._read_selected_metadata()
        if item is None:
            self._set_output(title="No Selection", text="Select a run first.", log_message="History notes save skipped with no selection.")
            return
        if str(item.get("mode", "")).strip().lower() != "deploy":
            self._set_output(title="History", text="Deployment notes are only available for deploy runs.", log_message="History notes save skipped for non-deploy run.")
            return
        if metadata_data is None:
            self._set_output(title="History", text="Unable to read metadata.json for this run.", log_message="History notes save failed due to unreadable metadata.")
            return
        notes = self.overall_notes.toPlainText().strip()
        if notes:
            metadata_data["deploy_notes_summary"] = notes
        else:
            metadata_data.pop("deploy_notes_summary", None)
        metadata_data["deploy_episode_outcomes"] = _normalize_deploy_episode_outcomes(metadata_data.get("deploy_episode_outcomes"))
        ok, message = self._persist_deploy_metadata(metadata_data)
        self.deploy_editor_status.setText(message)
        self._set_output(title="Deployment Notes Saved" if ok else "Save Failed", text=message, log_message=message)

    def open_deploy_notes_file(self) -> None:
        item, _metadata_path, metadata_data = self._read_selected_metadata()
        if item is None:
            self._set_output(title="No Selection", text="Select a run first.", log_message="History notes open skipped with no selection.")
            return
        run_path = Path(str(item.get("_run_path", "")).strip())
        if not run_path.exists():
            self._set_output(title="Open Failed", text="Run folder is missing for this history entry.", log_message="History notes open failed due to missing run path.")
            return
        notes_path = run_path / "notes.md"
        if not notes_path.exists() and metadata_data is not None:
            generated = write_deploy_notes_file(run_path, metadata_data, filename="notes.md")
            if generated is not None:
                notes_path = generated
                write_deploy_episode_spreadsheet(
                    run_path,
                    metadata_data,
                    filename="episode_outcomes.csv",
                    summary_filename="episode_outcomes_summary.csv",
                )
        ok, message = open_path_in_file_manager(notes_path)
        self._set_output(
            title="Open Notes" if ok else "Open Failed",
            text=str(notes_path if ok else message),
            log_message="History opened deploy notes." if ok else "History failed to open deploy notes.",
        )


def build_qt_secondary_panel(
    *,
    section_id: str,
    config: dict[str, Any],
    append_log: Callable[[str], None],
    run_controller: ManagedRunController,
    run_terminal_command: Callable[[str], tuple[bool, str]] | None = None,
    update_and_restart_app: Callable[[], tuple[bool, str]] | None = None,
) -> QWidget | None:
    if section_id == "config":
        return QtConfigPage(
            config=config,
            append_log=append_log,
            run_terminal_command=run_terminal_command,
            update_and_restart_app=update_and_restart_app,
        )
    if section_id == "visualizer":
        return QtVisualizerPage(config=config, append_log=append_log)
    if section_id == "history":
        return QtHistoryPage(config=config, append_log=append_log, run_controller=run_controller)
    return None
