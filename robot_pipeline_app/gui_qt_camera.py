from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .camera_state import (
    DEFAULT_LIVE_PREVIEW_FPS_CAP,
    assign_camera_role,
    camera_indices,
    camera_mapping_summary,
    compute_capture_fps,
    live_preview_interval_ms,
    normalize_live_preview_fps_cap,
    normalize_scan_limit,
    positive_int,
    sanitize_reported_fps,
)
from .config_store import save_config
from .probes import camera_fingerprint, probe_module_import, summarize_probe_error


class QtDualCameraPreview(QFrame):
    def __init__(
        self,
        *,
        config: dict[str, Any],
        append_log: Callable[[str], None],
        title: str = "Camera Preview",
    ) -> None:
        super().__init__()
        self.setObjectName("SectionCard")
        self.config = config
        self._append_log = append_log
        self._cv2_module: Any | None = None
        self._cv2_probe_ok, self._cv2_probe_error = probe_module_import("cv2")
        self._detected_indices: list[int] = []
        self._detected_cards: dict[int, dict[str, Any]] = {}
        self._capture_lock = threading.Lock()
        self._capture_pool: dict[int, Any] = {}
        self._capture_pool_reported_fps: dict[int, float | None] = {}
        self._capture_pool_timestamps: dict[int, list[float]] = {}
        self._capture_retry_after: dict[int, float] = {}
        self._run_active = False
        self._live_paused_for_run = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        header = QLabel(title)
        header.setObjectName("SectionMeta")
        layout.addWidget(header)

        controls = QHBoxLayout()
        controls.setSpacing(8)

        self.scan_button = QPushButton("Scan Camera Ports")
        self.scan_button.clicked.connect(self.scan_camera_ports)
        controls.addWidget(self.scan_button)

        self.refresh_button = QPushButton("Refresh Camera Preview")
        self.refresh_button.clicked.connect(self.refresh_camera_previews)
        controls.addWidget(self.refresh_button)

        self.live_button = QPushButton("Start Live")
        self.live_button.clicked.connect(self.toggle_live_preview)
        controls.addWidget(self.live_button)

        self.pause_on_run_checkbox = QCheckBox("Pause on run")
        self.pause_on_run_checkbox.setChecked(True)
        self.pause_on_run_checkbox.toggled.connect(self._on_pause_on_run_toggled)
        controls.addWidget(self.pause_on_run_checkbox)

        controls.addWidget(QLabel("Live FPS"))
        self.live_fps_input = QLineEdit(str(DEFAULT_LIVE_PREVIEW_FPS_CAP))
        self.live_fps_input.setMaximumWidth(56)
        controls.addWidget(self.live_fps_input)

        controls.addWidget(QLabel("Max idx"))
        self.scan_limit_input = QLineEdit("14")
        self.scan_limit_input.setMaximumWidth(56)
        controls.addWidget(self.scan_limit_input)

        controls.addStretch(1)

        self.status_label = QLabel("Preview idle.")
        self.status_label.setObjectName("MutedLabel")
        self.status_label.setWordWrap(True)
        controls.addWidget(self.status_label, 1)
        layout.addLayout(controls)

        self.mapping_label = QLabel(camera_mapping_summary(self.config))
        self.mapping_label.setObjectName("MutedLabel")
        self.mapping_label.setWordWrap(True)
        layout.addWidget(self.mapping_label)

        self.detected_ports_label = QLabel("Detected open camera ports: (scan to detect)")
        self.detected_ports_label.setObjectName("MutedLabel")
        self.detected_ports_label.setWordWrap(True)
        layout.addWidget(self.detected_ports_label)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._cards_wrap = QWidget()
        self._cards_layout = QGridLayout(self._cards_wrap)
        self._cards_layout.setContentsMargins(0, 0, 0, 0)
        self._cards_layout.setHorizontalSpacing(12)
        self._cards_layout.setVerticalSpacing(12)
        self._scroll.setWidget(self._cards_wrap)
        layout.addWidget(self._scroll)

        self._live_timer = QTimer(self)
        self._live_timer.timeout.connect(self.refresh_camera_previews)

    def closeEvent(self, event: Any) -> None:
        try:
            self._release_all_pooled_captures()
        finally:
            super().closeEvent(event)

    def set_active_run(self, active: bool) -> None:
        self._run_active = bool(active)
        if not self.pause_on_run_checkbox.isChecked():
            return
        if self._run_active:
            if self._live_timer.isActive():
                self._live_paused_for_run = True
                self._live_timer.stop()
                self._release_all_pooled_captures()
                self.status_label.setText("Live preview auto-paused while a workflow is active.")
                self.live_button.setText("Live Paused")
        else:
            if self._live_paused_for_run:
                self._live_paused_for_run = False
                self.status_label.setText("Workflow ended. Live preview resumed.")
                self._restart_live_timer()

    def scan_camera_ports(self) -> None:
        if not self._ensure_cv2_module():
            self.detected_ports_label.setText("Detected open camera ports: unavailable")
            return
        limit = normalize_scan_limit(self.scan_limit_input.text())
        self.scan_limit_input.setText(str(limit))
        candidates = self._candidate_scan_indices(limit)
        detected: list[int] = []
        for index in candidates:
            cap = self._open_capture(index)
            if cap is None:
                continue
            detected.append(index)
            try:
                cap.release()
            except Exception:
                pass
        self._detected_indices = detected
        if detected:
            self.detected_ports_label.setText(f"Detected open camera ports: {', '.join(str(i) for i in detected)}")
            self._append_log(f"Detected camera ports: {', '.join(str(i) for i in detected)}")
        else:
            self.detected_ports_label.setText("Detected open camera ports: none found")
            self._append_log("No open camera ports detected in scan range.")
        self._rebuild_cards()
        self.refresh_camera_previews()

    def refresh_camera_previews(self) -> None:
        if not self._ensure_cv2_module():
            return
        if not self._detected_indices:
            self.status_label.setText("Scan camera ports first.")
            return
        refreshed = 0
        for index in list(self._detected_indices):
            frame, fps = self._capture_live_frame_with_fps(index) if self._live_timer.isActive() else self._capture_frame_with_fps(index)
            card = self._detected_cards.get(index)
            if card is None:
                continue
            if frame is None:
                card["preview"].setText("No frame")
                continue
            self._render_frame(card["preview"], frame)
            fps_text = f"{fps:.1f} FPS" if fps is not None else "n/a FPS"
            card["fps"].setText(f"Input: {frame.shape[1]}x{frame.shape[0]} @ {fps_text}")
            refreshed += 1
        self.status_label.setText(f"Preview refreshed for {refreshed}/{len(self._detected_indices)} detected ports.")

    def toggle_live_preview(self) -> None:
        if self._live_timer.isActive():
            self._live_timer.stop()
            self._release_all_pooled_captures()
            self.live_button.setText("Start Live")
            self.status_label.setText("Live preview stopped.")
            return
        self._live_paused_for_run = False
        self._restart_live_timer()
        self.live_button.setText("Stop Live")
        self.status_label.setText("Live preview running.")
        self.refresh_camera_previews()

    def _on_pause_on_run_toggled(self, checked: bool) -> None:
        if checked and self._run_active and self._live_timer.isActive():
            self.set_active_run(True)
        elif not checked and self._live_paused_for_run:
            self._live_paused_for_run = False
            self._restart_live_timer()

    def _restart_live_timer(self) -> None:
        if self._run_active and self.pause_on_run_checkbox.isChecked():
            return
        interval = live_preview_interval_ms(normalize_live_preview_fps_cap(self.live_fps_input.text()))
        self.live_fps_input.setText(str(normalize_live_preview_fps_cap(self.live_fps_input.text())))
        self._live_timer.start(interval)
        self.live_button.setText("Stop Live")

    def _candidate_scan_indices(self, limit: int) -> list[int]:
        if os.name != "posix":
            return list(range(limit + 1))
        video_nodes = sorted(Path("/dev").glob("video*"))
        detected: list[int] = []
        for node in video_nodes:
            suffix = node.name.replace("video", "", 1)
            if suffix.isdigit():
                idx = int(suffix)
                if idx <= limit:
                    detected.append(idx)
        return sorted(set(detected)) if detected else list(range(limit + 1))

    def _set_cv2_quiet(self, cv2_mod: Any) -> None:
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

    def _ensure_cv2_module(self) -> bool:
        if self._cv2_module is not None:
            return True
        if not self._cv2_probe_ok:
            self.status_label.setText(f"Camera preview unavailable: {summarize_probe_error(self._cv2_probe_error)}")
            self._append_log(self.status_label.text())
            return False
        try:
            import cv2 as cv2_loaded  # type: ignore[import-not-found]
        except Exception as exc:
            self.status_label.setText(f"Camera preview unavailable: {exc}")
            self._append_log(self.status_label.text())
            return False
        self._set_cv2_quiet(cv2_loaded)
        self._cv2_module = cv2_loaded
        return True

    def _camera_indices(self) -> dict[str, int]:
        return camera_indices(self.config)

    def _open_capture(self, index: int) -> Any | None:
        if self._cv2_module is None:
            return None
        cv2_mod = self._cv2_module
        result: list[Any] = [None]

        def _try_open() -> None:
            result[0] = cv2_mod.VideoCapture(index)

        thread = threading.Thread(target=_try_open, daemon=True)
        thread.start()
        thread.join(timeout=2.5)
        if thread.is_alive():
            return None
        cap = result[0]
        if cap is None or not cap.isOpened():
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            return None
        width = positive_int(self.config.get("camera_default_width"), 640)
        height = positive_int(self.config.get("camera_default_height"), 480)
        fps = positive_int(self.config.get("camera_fps"), 30)
        try:
            cap.set(cv2_mod.CAP_PROP_FRAME_WIDTH, float(width))
            cap.set(cv2_mod.CAP_PROP_FRAME_HEIGHT, float(height))
            cap.set(cv2_mod.CAP_PROP_FPS, float(fps))
            cap.set(cv2_mod.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def _capture_frame_with_fps(self, index: int) -> tuple[Any | None, float | None]:
        cap = self._open_capture(index)
        if cap is None:
            return None, None
        latest_frame = None
        timestamps: list[float] = []
        reported_fps = sanitize_reported_fps(cap.get(self._cv2_module.CAP_PROP_FPS)) if self._cv2_module is not None else None
        try:
            for _ in range(12):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                latest_frame = frame
                timestamps.append(time.monotonic())
                if len(timestamps) >= 2 and (timestamps[-1] - timestamps[0]) >= 0.8:
                    break
        finally:
            try:
                cap.release()
            except Exception:
                pass
        if latest_frame is None:
            return None, None
        return latest_frame, compute_capture_fps(timestamps, reported_fps=reported_fps)

    def _capture_live_frame_with_fps(self, index: int) -> tuple[Any | None, float | None]:
        if self._cv2_module is None:
            return None, None
        with self._capture_lock:
            now = time.monotonic()
            retry_after = self._capture_retry_after.get(index, 0.0)
            if now < retry_after:
                return None, None
            cap = self._capture_pool.get(index)
            if cap is None or not cap.isOpened():
                if cap is not None:
                    self._release_pooled_capture_locked(index)
                cap = self._open_capture(index)
                if cap is None:
                    self._capture_retry_after[index] = now + 1.0
                    return None, None
                self._capture_pool[index] = cap
                self._capture_pool_reported_fps[index] = sanitize_reported_fps(cap.get(self._cv2_module.CAP_PROP_FPS))
                self._capture_pool_timestamps[index] = []
                self._capture_retry_after.pop(index, None)
            ok, frame = cap.read()
            if not ok or frame is None:
                self._release_pooled_capture_locked(index)
                self._capture_retry_after[index] = time.monotonic() + 0.5
                return None, None
            timestamps = self._capture_pool_timestamps.setdefault(index, [])
            timestamps.append(time.monotonic())
            cutoff = timestamps[-1] - 2.0
            while len(timestamps) > 2 and timestamps[0] < cutoff:
                timestamps.pop(0)
            fps = compute_capture_fps(timestamps, reported_fps=self._capture_pool_reported_fps.get(index))
            return frame, fps

    def _release_pooled_capture_locked(self, index: int) -> None:
        cap = self._capture_pool.pop(index, None)
        self._capture_pool_reported_fps.pop(index, None)
        self._capture_pool_timestamps.pop(index, None)
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def _release_all_pooled_captures(self) -> None:
        with self._capture_lock:
            for index in list(self._capture_pool):
                self._release_pooled_capture_locked(index)

    def _render_frame(self, label: QLabel, frame_bgr: Any) -> None:
        if self._cv2_module is None:
            return
        rgb = self._cv2_module.cvtColor(frame_bgr, self._cv2_module.COLOR_BGR2RGB)
        height, width, _channels = rgb.shape
        bytes_per_line = rgb.strides[0]
        image = QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            240,
            180,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(pixmap)
        label.setText("")

    def _clear_cards(self) -> None:
        while self._cards_layout.count():
            item = self._cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._detected_cards.clear()

    def _rebuild_cards(self) -> None:
        self._clear_cards()
        indices = self._camera_indices()
        for idx, index in enumerate(self._detected_indices):
            card = QFrame()
            card.setObjectName("SectionCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(12, 12, 12, 12)
            card_layout.setSpacing(8)

            title = QLabel(f"Port {index}")
            title.setObjectName("FormLabel")
            card_layout.addWidget(title)

            role_label = QLabel(self._role_text(index, indices))
            role_label.setObjectName("MutedLabel")
            role_label.setWordWrap(True)
            card_layout.addWidget(role_label)

            fps_label = QLabel("Input: n/a @ n/a FPS")
            fps_label.setObjectName("MutedLabel")
            card_layout.addWidget(fps_label)

            preview = QLabel("No frame")
            preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            preview.setMinimumSize(240, 180)
            preview.setObjectName("DialogText")
            card_layout.addWidget(preview)

            actions = QHBoxLayout()
            laptop_button = QPushButton("Set Laptop")
            laptop_button.clicked.connect(lambda _checked=False, source=index: self._assign_role("laptop", source))
            actions.addWidget(laptop_button)

            phone_button = QPushButton("Set Phone")
            phone_button.clicked.connect(lambda _checked=False, source=index: self._assign_role("phone", source))
            actions.addWidget(phone_button)
            actions.addStretch(1)
            card_layout.addLayout(actions)

            self._cards_layout.addWidget(card, idx // 2, idx % 2)
            self._detected_cards[index] = {
                "card": card,
                "role": role_label,
                "fps": fps_label,
                "preview": preview,
            }

    def _role_text(self, index: int, indices: dict[str, int]) -> str:
        laptop_idx = indices["laptop"]
        phone_idx = indices["phone"]
        if index == laptop_idx and index == phone_idx:
            return "Role: Laptop + Phone"
        if index == laptop_idx:
            return "Role: Laptop"
        if index == phone_idx:
            return "Role: Phone"
        return "Role: Unassigned"

    def _assign_role(self, role: str, index: int) -> None:
        assignment = assign_camera_role(
            config=self.config,
            detected_indices=self._detected_indices,
            detected_frame_sizes={},
            role=role,
            index=index,
            fingerprint=camera_fingerprint(index),
        )
        if not assignment.ok:
            for message in assignment.messages:
                self._append_log(message)
            return
        self.config.clear()
        self.config.update(assignment.updated_config)
        save_config(self.config, quiet=True)
        self.mapping_label.setText(camera_mapping_summary(self.config))
        indices = self._camera_indices()
        for port, card in self._detected_cards.items():
            cast_label = card.get("role")
            if isinstance(cast_label, QLabel):
                cast_label.setText(self._role_text(port, indices))
        for message in assignment.messages:
            self._append_log(message)
