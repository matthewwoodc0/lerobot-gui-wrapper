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
from .gui_qt_common import _InputGrid, _build_card
from .gui_qt_dialogs import ask_text_dialog_with_actions
from .gui_qt_output import QtRunOutputPanel
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
from .app_theme import SPACING_CARD, SPACING_COMPACT, SPACING_SHELL


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
        layout.setContentsMargins(SPACING_COMPACT, SPACING_COMPACT, SPACING_COMPACT, SPACING_COMPACT)
        layout.setSpacing(SPACING_CARD)

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

        self.count_input = QSpinBox()
        self.count_input.setRange(1, 16)
        self.count_input.valueChanged.connect(self._handle_count_changed)

        controls_wrap = QWidget()
        controls_layout = QVBoxLayout(controls_wrap)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)

        controls_form = _InputGrid(controls_layout)
        controls_form.add_field("Camera count", self.count_input)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)

        add_button = QPushButton("Add Camera")
        add_button.clicked.connect(self.add_camera)
        controls.addWidget(add_button)

        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self.remove_selected_camera)
        controls.addWidget(remove_button)
        controls.addStretch(1)
        controls_layout.addLayout(controls)
        layout.addWidget(controls_wrap)

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
    def __init__(
        self,
        *,
        title: str,
        subtitle: str,
        append_log: Callable[[str], None],
        use_output_tabs: bool = False,
    ) -> None:
        super().__init__()
        _ = title, subtitle
        self._append_log = append_log
        self._use_output_tabs = bool(use_output_tabs)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING_SHELL)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)

        self.content_layout = QVBoxLayout()
        self.content_layout.setSpacing(SPACING_SHELL)
        layout.addLayout(self.content_layout)

        self.output_card, output_layout = _build_card("Output")
        self.output_panel: QtRunOutputPanel | None = None
        self.raw_output: QPlainTextEdit | None = None
        self._explain_callback: Callable[[], None] | None = None
        if self._use_output_tabs:
            self.output_panel = QtRunOutputPanel()
            self.status_label = self.output_panel.status_label
            self.output = self.output_panel.summary_output
            self.raw_output = self.output_panel.raw_output
            output_layout.addWidget(self.output_panel)
        else:
            self.status_label = QLabel("Ready.")
            self.status_label.setObjectName("StatusChip")
            self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.status_label.setMaximumWidth(280)
            output_layout.addWidget(self.status_label)

            self.output = QPlainTextEdit()
            self.output.setReadOnly(True)
            self.output.setMinimumHeight(140)
            self.output.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
            output_layout.addWidget(self.output)
        layout.addWidget(self.output_card, 1)
        self.output_card.hide()

    def _set_output(self, *, title: str, text: str, log_message: str | None = None) -> None:
        self.status_label.setText(title)
        if self.output_panel is not None:
            self.output_panel.set_summary_text(text)
        else:
            self.output.setPlainText(text)
        if log_message:
            self._append_log(log_message)

    def _append_output_line(self, line: str) -> None:
        if self.output_panel is not None:
            self.output_panel.append_summary_line(line)
        else:
            self.output.appendPlainText(str(line))

    def _append_output_chunk(self, chunk: str) -> None:
        if self.output_panel is not None:
            self.output_panel.append_raw_text(chunk)

    def _set_raw_output(self, text: str) -> None:
        if self.output_panel is not None:
            self.output_panel.set_raw_text(text)

    def _show_summary_tab(self) -> None:
        if self.output_panel is not None:
            self.output_panel.show_summary_tab()

    def _show_raw_tab(self) -> None:
        if self.output_panel is not None:
            self.output_panel.show_raw_tab()

    def _set_explain_callback(self, callback: Callable[[], None] | None) -> None:
        if self.output_panel is None:
            return
        if self._explain_callback is not None:
            try:
                self.output_panel.explain_button.clicked.disconnect(self._explain_callback)
            except Exception:
                pass
        self._explain_callback = None
        if callback is None:
            self.output_panel.explain_button.setEnabled(False)
            return
        self._explain_callback = callback
        self.output_panel.explain_button.clicked.connect(callback)
        self.output_panel.explain_button.setEnabled(True)

    def _append_output_and_log(self, line: str) -> None:
        self._append_output_line(line)
        self._append_log(line)
