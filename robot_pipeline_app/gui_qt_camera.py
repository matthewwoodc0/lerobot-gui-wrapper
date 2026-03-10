from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .camera_state import (
    DEFAULT_LIVE_PREVIEW_FPS_CAP,
    assign_named_camera_source,
    camera_source_map,
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


class _QtCallbackDispatcher(QObject):
    invoke = Signal(object, tuple)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.invoke.connect(self._dispatch)

    @Slot(object, tuple)
    def _dispatch(self, callback: object, args: tuple[object, ...]) -> None:
        if not callable(callback):
            return
        callback(*args)

    def schedule(self, callback: Callable[..., None], *args: Any) -> None:
        self.invoke.emit(callback, tuple(args))


class QtDualCameraPreview(QFrame):
    _MACOS_FALLBACK_SCAN_INDICES = tuple(range(7))

    def __init__(
        self,
        *,
        config: dict[str, Any],
        append_log: Callable[[str], None],
        title: str = "Camera Preview",
    ) -> None:
        super().__init__()
        self.setObjectName("SectionCard")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
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
        self._scan_in_progress = False
        self._refresh_in_progress = False
        self._ui_dispatcher = _QtCallbackDispatcher(self)

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

        self._cards_wrap = QWidget()
        self._cards_wrap.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self._cards_layout = QGridLayout(self._cards_wrap)
        self._cards_layout.setContentsMargins(0, 0, 0, 0)
        self._cards_layout.setHorizontalSpacing(12)
        self._cards_layout.setVerticalSpacing(12)
        self._cards_layout.setColumnStretch(0, 1)
        self._cards_layout.setColumnStretch(1, 1)
        layout.addWidget(self._cards_wrap)

        self._live_timer = QTimer(self)
        self._live_timer.timeout.connect(self.refresh_camera_previews)

    def closeEvent(self, event: Any) -> None:
        try:
            self._live_timer.stop()
            self._release_all_pooled_captures()
        finally:
            super().closeEvent(event)

    def set_active_run(self, active: bool) -> None:
        self._run_active = bool(active)
        if self._run_active:
            if self._live_timer.isActive():
                self._live_timer.stop()
                self._release_all_pooled_captures()
                self.status_label.setText("Live preview stopped while a workflow is active.")
                self.live_button.setText("Start Live")

    def scan_camera_ports(self) -> None:
        if self._scan_in_progress:
            self.status_label.setText("Scan already in progress...")
            return
        if not self._ensure_cv2_module():
            self.detected_ports_label.setText("Detected open camera ports: unavailable")
            return
        limit = normalize_scan_limit(self.scan_limit_input.text())
        self.scan_limit_input.setText(str(limit))
        configured_candidates = self._candidate_scan_indices(limit)
        fallback_candidates = self._macos_fallback_scan_indices(limit, configured_candidates)

        self._scan_in_progress = True
        self.scan_button.setEnabled(False)
        if sys.platform == "darwin":
            scan_count = len(configured_candidates) if configured_candidates else len(fallback_candidates)
            if configured_candidates:
                self.status_label.setText(f"Scanning {scan_count} configured camera port(s)...")
            else:
                fallback_text = ", ".join(str(idx) for idx in fallback_candidates)
                self.status_label.setText(f"Scanning fallback camera port(s): {fallback_text}")
                self._append_log(
                    "macOS camera scan found no configured runtime camera indices; trying fallback ports "
                    f"{fallback_text}."
                )
        else:
            self.status_label.setText(f"Scanning {len(configured_candidates)} camera port(s)...")

        thread = threading.Thread(
            target=self._scan_candidates_worker,
            args=(configured_candidates, fallback_candidates),
            daemon=True,
            name="camera-scan",
        )
        thread.start()

    def refresh_camera_previews(self) -> None:
        if self._refresh_in_progress:
            return
        if not self._ensure_cv2_module():
            return
        if not self._detected_indices:
            self.status_label.setText("Scan camera ports first.")
            return
        manual_refresh = not self._live_timer.isActive()
        if manual_refresh:
            self._capture_retry_after.clear()
        self._refresh_in_progress = True
        try:
            refreshed = 0
            for index in list(self._detected_indices):
                frame, fps = (
                    self._capture_preview_snapshot_with_fps(index)
                    if manual_refresh
                    else self._capture_live_frame_with_fps(index)
                )
                card = self._detected_cards.get(index)
                if card is None:
                    continue
                if frame is None:
                    card["preview"].clear()
                    card["preview"].setText("No frame")
                    card["fps"].setText("Input: n/a @ n/a FPS")
                    continue
                self._render_frame(card["preview"], frame)
                fps_text = f"{fps:.1f} FPS" if fps is not None else "n/a FPS"
                card["fps"].setText(f"Input: {frame.shape[1]}x{frame.shape[0]} @ {fps_text}")
                refreshed += 1
            if refreshed:
                self.status_label.setText(f"Preview refreshed for {refreshed}/{len(self._detected_indices)} detected ports.")
            else:
                missing = ", ".join(str(index) for index in self._detected_indices)
                self.status_label.setText(f"No preview frame received from detected ports: {missing}.")
        finally:
            self._refresh_in_progress = False

    def toggle_live_preview(self) -> None:
        if self._live_timer.isActive():
            self._live_timer.stop()
            self._release_all_pooled_captures()
            self.live_button.setText("Start Live")
            self.status_label.setText("Live preview stopped.")
            return
        if self._run_active:
            self.status_label.setText("Live preview unavailable while a workflow is active.")
            self.live_button.setText("Start Live")
            return
        self._restart_live_timer()
        self.live_button.setText("Stop Live")
        self.status_label.setText("Live preview running.")
        self.refresh_camera_previews()

    def _restart_live_timer(self) -> None:
        if self._run_active:
            return
        interval = live_preview_interval_ms(normalize_live_preview_fps_cap(self.live_fps_input.text()))
        self.live_fps_input.setText(str(normalize_live_preview_fps_cap(self.live_fps_input.text())))
        self._live_timer.start(interval)
        self.live_button.setText("Stop Live")

    def _candidate_scan_indices(self, limit: int) -> list[int]:
        if sys.platform == "darwin":
            configured_indices = sorted(
                {
                    int(source)
                    for source in camera_source_map(self.config).values()
                    if type(source) is int and 0 <= int(source) <= limit
                }
            )
            return configured_indices
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

    def _macos_fallback_scan_indices(self, limit: int, configured_candidates: list[int]) -> list[int]:
        if sys.platform != "darwin":
            return []
        configured = set(configured_candidates)
        return [
            idx
            for idx in self._MACOS_FALLBACK_SCAN_INDICES
            if idx <= limit and idx not in configured
        ]

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

    def _camera_assignments(self) -> dict[str, int | str]:
        return camera_source_map(self.config)

    def _scan_candidates_worker(
        self,
        configured_candidates: list[int],
        fallback_candidates: list[int],
    ) -> None:
        detected, failures = self._probe_scan_indices(configured_candidates or fallback_candidates)
        fallback_used = bool(sys.platform == "darwin" and not configured_candidates and fallback_candidates)
        fallback_reason = "no configured runtime camera indices were available" if fallback_used else ""
        if sys.platform == "darwin" and not detected and configured_candidates and fallback_candidates:
            fallback_used = True
            fallback_reason = "configured camera ports failed to open"
            fallback_detected, fallback_failures = self._probe_scan_indices(fallback_candidates)
            detected = fallback_detected
            failures.update(fallback_failures)
        self._ui_dispatcher.schedule(
            self._finish_scan,
            detected,
            failures,
            fallback_used,
            tuple(fallback_candidates),
            bool(configured_candidates),
            fallback_reason,
        )

    def _probe_scan_indices(
        self,
        candidates: list[int],
    ) -> tuple[list[int], dict[int, str]]:
        detected: list[int] = []
        failures: dict[int, str] = {}
        for index in candidates:
            try:
                opened, detail = self._scan_index_with_preview_capture(index)
            except Exception as exc:
                opened = False
                detail = str(exc)
            if opened:
                detected.append(index)
                continue
            summarized = summarize_probe_error(detail)
            if summarized:
                failures[index] = summarized
        return detected, failures

    def _scan_index_with_preview_capture(self, index: int) -> tuple[bool, str]:
        with self._capture_lock:
            self._capture_retry_after.pop(index, None)
        frame, fps = self._capture_preview_snapshot_with_fps(index)
        if frame is None:
            return False, "no preview frame"
        fps_text = f" @ {fps:.1f} FPS" if fps is not None else ""
        return True, f"frame={frame.shape[1]}x{frame.shape[0]}{fps_text}"

    def _capture_preview_snapshot_with_fps(self, index: int) -> tuple[Any | None, float | None]:
        with self._capture_lock:
            self._capture_retry_after.pop(index, None)
        frame, fps = self._capture_live_frame_with_fps(index)
        with self._capture_lock:
            self._release_pooled_capture_locked(index)
            self._capture_retry_after.pop(index, None)
        return frame, fps

    def _finish_scan(
        self,
        detected: list[int],
        failures: dict[int, str],
        fallback_used: bool,
        fallback_candidates: tuple[int, ...],
        had_configured_candidates: bool,
        fallback_reason: str,
    ) -> None:
        self._scan_in_progress = False
        self.scan_button.setEnabled(True)
        self._detected_indices = detected
        if detected:
            self.detected_ports_label.setText(f"Detected open camera ports: {', '.join(str(i) for i in detected)}")
            self._append_log(f"Detected camera ports: {', '.join(str(i) for i in detected)}")
            if fallback_used and fallback_candidates:
                fallback_text = ", ".join(str(idx) for idx in fallback_candidates)
                self.status_label.setText("Scan complete using macOS fallback ports. Click Refresh Camera Preview to open detected cameras safely.")
                self._append_log(f"macOS fallback camera scan succeeded after {fallback_reason}; checked fallback ports {fallback_text}.")
            else:
                self.status_label.setText("Scan complete. Click Refresh Camera Preview to open detected cameras safely.")
        else:
            self.detected_ports_label.setText("Detected open camera ports: none found")
            if failures:
                self.status_label.setText(
                    self._scan_failure_status(
                        failures,
                        used_fallback=fallback_used or (sys.platform == "darwin" and not had_configured_candidates),
                        had_configured_candidates=had_configured_candidates,
                    )
                )
                for index, detail in failures.items():
                    self._append_log(f"Camera scan failed on port {index}: {detail}")
            else:
                self.status_label.setText("Scan complete. No open camera ports found.")
                self._append_log("No open camera ports detected in scan range.")
        self._rebuild_cards()
        self._release_all_pooled_captures()

    def _scan_failure_status(
        self,
        failures: dict[int, str],
        *,
        used_fallback: bool,
        had_configured_candidates: bool,
    ) -> str:
        unique_details: list[str] = []
        for detail in failures.values():
            if detail not in unique_details:
                unique_details.append(detail)
        if used_fallback and had_configured_candidates:
            prefix = "Scan failed on configured and fallback ports"
        elif used_fallback:
            prefix = "Scan failed on macOS fallback ports"
        elif sys.platform == "darwin":
            prefix = "Scan failed on configured ports"
        else:
            prefix = "Scan failed on checked ports"
        if len(unique_details) == 1:
            return f"{prefix}: {unique_details[0]}"
        return f"{prefix}. First error: {unique_details[0]}"

    def _capture_backend_candidates(self, cv2_mod: Any) -> list[int | None]:
        backends: list[int | None] = []
        if sys.platform == "darwin":
            avfoundation = getattr(cv2_mod, "CAP_AVFOUNDATION", None)
            if avfoundation is not None:
                backends.append(int(avfoundation))
        backends.append(None)
        return backends

    def _frame_wait_budget_s(self, *, live: bool) -> float:
        configured_warmup = positive_int(self.config.get("camera_warmup_s"), 1)
        if sys.platform == "darwin":
            cap = 0.8 if live else 1.5
        else:
            cap = 0.5 if live else 0.75
        return min(cap, max(0.35, float(configured_warmup)))

    def _read_frame_with_warmup(self, cap: Any, *, live: bool) -> tuple[Any | None, list[float]]:
        latest_frame = None
        timestamps: list[float] = []
        first_frame_at: float | None = None
        attempts = 0
        deadline = time.monotonic() + self._frame_wait_budget_s(live=live)

        while attempts < 24 and time.monotonic() < deadline:
            attempts += 1
            ok, frame = cap.read()
            now = time.monotonic()
            if ok and frame is not None:
                latest_frame = frame
                timestamps.append(now)
                if live:
                    return latest_frame, timestamps
                if first_frame_at is None:
                    first_frame_at = now
                elif len(timestamps) >= 2 and (now - first_frame_at) >= 0.25:
                    break
                time.sleep(0.03)
                continue
            if latest_frame is None:
                time.sleep(0.05)
                continue
            break

        return latest_frame, timestamps

    def _open_capture(self, index: int) -> Any | None:
        if self._cv2_module is None:
            return None
        cv2_mod = self._cv2_module
        result: list[Any] = [None]
        timed_out = threading.Event()

        def _try_open() -> None:
            for backend in self._capture_backend_candidates(cv2_mod):
                try:
                    cap = cv2_mod.VideoCapture(index) if backend is None else cv2_mod.VideoCapture(index, backend)
                except Exception:
                    cap = None
                if cap is None:
                    continue
                if cap.isOpened():
                    if timed_out.is_set():
                        try:
                            cap.release()
                        except Exception:
                            pass
                        return
                    result[0] = cap
                    return
                try:
                    cap.release()
                except Exception:
                    pass

        thread = threading.Thread(target=_try_open, daemon=True)
        thread.start()
        thread.join(timeout=2.5)
        if thread.is_alive():
            timed_out.set()
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
        reported_fps = sanitize_reported_fps(cap.get(self._cv2_module.CAP_PROP_FPS)) if self._cv2_module is not None else None
        try:
            latest_frame, timestamps = self._read_frame_with_warmup(cap, live=False)
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
            frame, sample_timestamps = self._read_frame_with_warmup(cap, live=True)
            if frame is None:
                self._release_pooled_capture_locked(index)
                self._capture_retry_after[index] = time.monotonic() + 0.5
                return None, None
            timestamps = self._capture_pool_timestamps.setdefault(index, [])
            timestamps.append(sample_timestamps[-1] if sample_timestamps else time.monotonic())
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
            self._capture_retry_after.clear()

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
        self._refresh_cards_geometry()

    def _refresh_cards_geometry(self) -> None:
        self._cards_wrap.adjustSize()
        self._cards_wrap.updateGeometry()
        self.adjustSize()
        self.updateGeometry()

    def _rebuild_cards(self) -> None:
        self._clear_cards()
        assignments = self._camera_assignments()
        for idx, index in enumerate(self._detected_indices):
            card = QFrame()
            card.setObjectName("SectionCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(12, 12, 12, 12)
            card_layout.setSpacing(8)

            title = QLabel(f"Port {index}")
            title.setObjectName("FormLabel")
            card_layout.addWidget(title)

            role_label = QLabel(self._role_text(index, assignments))
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

            actions_wrap = QWidget()
            actions = QGridLayout(actions_wrap)
            actions.setContentsMargins(0, 0, 0, 0)
            actions.setHorizontalSpacing(8)
            actions.setVerticalSpacing(8)
            buttons: dict[str, QPushButton] = {}
            for button_idx, camera_name in enumerate(assignments):
                button = QPushButton(self._assignment_button_text(camera_name, index, assignments))
                button.clicked.connect(
                    lambda _checked=False, camera_name=camera_name, source=index: self._assign_role(camera_name, source)
                )
                actions.addWidget(button, button_idx // 2, button_idx % 2)
                buttons[camera_name] = button
            card_layout.addWidget(actions_wrap)

            self._cards_layout.addWidget(card, idx // 2, idx % 2)
            self._detected_cards[index] = {
                "card": card,
                "role": role_label,
                "fps": fps_label,
                "preview": preview,
                "buttons": buttons,
            }
        self._refresh_cards_geometry()

    def _role_text(self, index: int, assignments: dict[str, int | str]) -> str:
        bound = [name for name, source in assignments.items() if source == index]
        if not bound:
            return "Assigned: Unassigned"
        return "Assigned: " + ", ".join(bound)

    def _assignment_button_text(self, camera_name: str, index: int, assignments: dict[str, int | str]) -> str:
        return f"{camera_name} (Active)" if assignments.get(camera_name) == index else f"Set {camera_name}"

    def _assign_role(self, role: str, index: int) -> None:
        assignment = assign_named_camera_source(
            config=self.config,
            detected_indices=self._detected_indices,
            detected_frame_sizes={},
            camera_name=role,
            index=int(index),
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
        assignments = self._camera_assignments()
        for port, card in self._detected_cards.items():
            cast_label = card.get("role")
            if isinstance(cast_label, QLabel):
                cast_label.setText(self._role_text(port, assignments))
            button_map = card.get("buttons")
            if isinstance(button_map, dict):
                for camera_name, button in button_map.items():
                    if isinstance(button, QPushButton):
                        button.setText(self._assignment_button_text(str(camera_name), port, assignments))
        for message in assignment.messages:
            self._append_log(message)
