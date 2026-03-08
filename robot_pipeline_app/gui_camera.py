from __future__ import annotations

import base64
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from .camera_state import (
    DEFAULT_LIVE_PREVIEW_FPS_CAP,
    CameraPreviewSnapshot,
    assign_camera_role,
    camera_indices,
    camera_input_fps_summary,
    camera_mapping_summary,
    choose_alternative_index,
    compute_capture_fps,
    export_camera_preview_state,
    live_preview_interval_ms,
    normalize_live_preview_fps_cap,
    normalize_scan_limit,
    positive_int,
    restore_camera_preview_state,
    sanitize_reported_fps,
)
from .config_store import save_config
from .gui_async import UiBackgroundJobs
from .probes import camera_fingerprint, summarize_probe_error

_PREVIEW_FPS_SAMPLE_FRAMES = 12
_PREVIEW_FPS_SAMPLE_TIMEOUT_S = 0.8
_PREVIEW_CANVAS_WIDTH = 240
_PREVIEW_CANVAS_HEIGHT = 180

_normalize_scan_limit = normalize_scan_limit
_positive_int = positive_int
_sanitize_reported_fps = sanitize_reported_fps
_compute_capture_fps = compute_capture_fps
_normalize_live_preview_fps_cap = normalize_live_preview_fps_cap
_live_preview_interval_ms = live_preview_interval_ms


class DualCameraPreview:
    def __init__(
        self,
        root: Any,
        parent: Any,
        title: str,
        config: dict[str, Any],
        colors: dict[str, str],
        cv2_probe_ok: bool,
        cv2_probe_error: str,
        append_log: Callable[[str], None],
        on_camera_indices_changed: Callable[[int, int], None] | None = None,
        background_jobs: UiBackgroundJobs | None = None,
    ) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.config = config
        self.colors = colors
        self.cv2_probe_ok = cv2_probe_ok
        self.cv2_probe_error = cv2_probe_error
        self.append_log = append_log
        self.on_camera_indices_changed = on_camera_indices_changed
        self.background_jobs = background_jobs

        self.frame = ttk.LabelFrame(parent, text=title, style="Section.TLabelframe", padding=10)
        self.frame.pack(fill="x", pady=(10, 0))

        self.cv2_module: Any | None = None
        self.detected_indices: list[int] = []

        self.detected_canvases: dict[int, Any] = {}
        self.detected_photos: dict[int, Any] = {}
        self.detected_frame_sizes: dict[int, tuple[int, int]] = {}
        self.detected_input_fps: dict[int, float] = {}
        self.role_label_vars: dict[int, Any] = {}
        self.role_label_widgets: dict[int, Any] = {}
        self.fps_label_vars: dict[int, Any] = {}
        self.role_buttons: dict[int, dict[str, Any]] = {}
        self._detected_cards: list[tuple[int, Any]] = []
        self.detected_ports_canvas: Any | None = None
        self._detected_ports_window_id: int | None = None

        self.status_preview_var = tk.StringVar(value="Preview idle.")
        self.detected_ports_var = tk.StringVar(value="Detected open camera ports: (scan to detect)")
        self.detected_empty_var = tk.StringVar(value="No detected camera previews yet. Click 'Scan Camera Ports'.")
        self.scan_limit_var = tk.StringVar(value="14")
        self._busy_job: str | None = None
        self._busy_ticks = 0
        self.live_fps_cap_var = tk.StringVar(value=str(DEFAULT_LIVE_PREVIEW_FPS_CAP))
        self.live_enabled_var = tk.BooleanVar(value=False)
        self.live_button_text_var = tk.StringVar(value="Start Live")
        self.pause_on_run_var = tk.BooleanVar(value=True)
        self._live_job: str | None = None
        self._live_tick_inflight = False
        self._live_paused_for_run = False
        self._run_active = False
        self._capture_lock = threading.Lock()
        self._capture_pool: dict[int, Any] = {}
        self._capture_pool_reported_fps: dict[int, float | None] = {}
        self._capture_pool_timestamps: dict[int, list[float]] = {}
        self._capture_retry_after: dict[int, float] = {}

        controls = ttk.Frame(self.frame, style="Panel.TFrame")
        controls.pack(fill="x", pady=(0, 8))

        self.scan_button = ttk.Button(controls, text="Scan Camera Ports", command=self.scan_camera_ports)
        self.scan_button.grid(row=0, column=0, sticky="w")

        self.refresh_button = ttk.Button(controls, text="Refresh Camera Preview", command=self.refresh_camera_previews)
        self.refresh_button.grid(row=0, column=1, sticky="w", padx=(8, 0))

        self.live_toggle_button = ttk.Button(
            controls,
            textvariable=self.live_button_text_var,
            style="Secondary.TButton",
            command=self.toggle,
        )
        self.live_toggle_button.grid(row=0, column=2, sticky="w", padx=(8, 0))

        ttk.Label(controls, text="Live FPS cap", style="Muted.TLabel").grid(row=0, column=3, sticky="w", padx=(10, 4))
        ttk.Entry(controls, textvariable=self.live_fps_cap_var, width=4).grid(row=0, column=4, sticky="w")

        ttk.Checkbutton(
            controls,
            text="Pause on run",
            variable=self.pause_on_run_var,
            style="TCheckbutton",
            command=self._on_pause_on_run_toggled,
        ).grid(
            row=0,
            column=5,
            sticky="w",
            padx=(10, 0),
        )

        ttk.Label(controls, text="Max idx", style="Muted.TLabel").grid(row=0, column=6, sticky="w", padx=(12, 4))
        ttk.Entry(controls, textvariable=self.scan_limit_var, width=5).grid(row=0, column=7, sticky="w")

        ttk.Label(controls, textvariable=self.status_preview_var, style="Muted.TLabel").grid(
            row=0,
            column=8,
            sticky="w",
            padx=(10, 0),
        )
        controls.columnconfigure(8, weight=1)

        ttk.Label(self.frame, textvariable=self.detected_ports_var, style="Muted.TLabel").pack(anchor="w", pady=(0, 6))

        detected_wrap = ttk.LabelFrame(self.frame, text="Detected Camera Ports", style="Section.TLabelframe", padding=8)
        detected_wrap.pack(fill="both", pady=(0, 8))
        ttk.Label(detected_wrap, textvariable=self.detected_empty_var, style="Muted.TLabel").pack(anchor="w")

        detected_scroll_wrap = ttk.Frame(detected_wrap, style="Panel.TFrame")
        detected_scroll_wrap.pack(fill="both", expand=True, pady=(6, 0))

        self.detected_ports_canvas = tk.Canvas(
            detected_scroll_wrap,
            height=330,
            bg=self.colors.get("bg", "#070b14"),
            highlightthickness=0,
            bd=0,
            relief="flat",
        )
        self.detected_ports_canvas.pack(side="left", fill="both", expand=True)

        self.detected_ports_frame = ttk.Frame(self.detected_ports_canvas, style="Panel.TFrame")
        self._detected_ports_window_id = self.detected_ports_canvas.create_window((0, 0), window=self.detected_ports_frame, anchor="nw")
        self.detected_ports_frame.bind("<Configure>", lambda _: self._sync_detected_scroll_region())
        self.detected_ports_canvas.bind("<Configure>", self._on_detected_canvas_configure)

    def _sync_detected_scroll_region(self) -> None:
        if self.detected_ports_canvas is None:
            return
        self.detected_ports_canvas.configure(scrollregion=self.detected_ports_canvas.bbox("all"))

    def _on_detected_canvas_configure(self, event: Any) -> None:
        if self.detected_ports_canvas is None or self._detected_ports_window_id is None:
            return
        self.detected_ports_canvas.itemconfigure(self._detected_ports_window_id, width=event.width)
        self._relayout_detected_cards()

    def _detected_card_columns(self) -> int:
        if self.detected_ports_canvas is None:
            return 3
        width = int(self.detected_ports_canvas.winfo_width() or self.detected_ports_canvas.winfo_reqwidth() or 760)
        return max(1, min(4, width // 250))

    def _relayout_detected_cards(self) -> None:
        columns = self._detected_card_columns()
        for col in range(4):
            self.detected_ports_frame.columnconfigure(col, weight=1 if col < columns else 0)
        for i, (_, card) in enumerate(self._detected_cards):
            row = i // columns
            col = i % columns
            card.grid(row=row, column=col, sticky="nw", padx=(0, 10), pady=(0, 10))
        self.root.after_idle(self._sync_detected_scroll_region)

    @contextmanager
    def _suppress_stderr(self) -> Iterator[None]:
        original_fd = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, 2)
            yield
        finally:
            os.dup2(original_fd, 2)
            os.close(original_fd)
            os.close(devnull)

    def _camera_indices(self) -> dict[str, int]:
        return camera_indices(self.config)

    def _preview_target_dimensions(self) -> tuple[int, int]:
        width = _positive_int(self.config.get("camera_default_width"), 640)
        height = _positive_int(self.config.get("camera_default_height"), 480)
        return width, height

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
        if self.cv2_module is not None:
            return True
        try:
            import cv2 as cv2_loaded  # type: ignore[import-not-found]
        except Exception as exc:
            self.status_preview_var.set("OpenCV import failed.")
            self.append_log(f"Camera preview unavailable: {exc}")
            return False
        self._set_cv2_quiet(cv2_loaded)
        self.cv2_module = cv2_loaded
        return True

    def _scan_limit(self) -> int:
        value = _normalize_scan_limit(self.scan_limit_var.get())
        self.scan_limit_var.set(str(value))
        return value

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

        if detected:
            return sorted(set(detected))
        return list(range(limit + 1))

    def _open_capture(self, index: int) -> Any | None:
        if self.cv2_module is None:
            return None
        cv2_mod = self.cv2_module
        result: list[Any] = [None]

        def _try_open() -> None:
            with self._suppress_stderr():
                result[0] = cv2_mod.VideoCapture(index)

        t = threading.Thread(target=_try_open, daemon=True)
        t.start()
        t.join(timeout=3.0)
        if t.is_alive():
            def _cleanup() -> None:
                t.join()
                cap = result[0]
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
            threading.Thread(target=_cleanup, daemon=True).start()
            return None
        if result[0] is None:
            return None
        cap = result[0]
        if not cap.isOpened():
            cap.release()
            return None
        target_width, target_height = self._preview_target_dimensions()
        target_fps = _positive_int(self.config.get("camera_fps"), 30)
        try:
            with self._suppress_stderr():
                cap.set(cv2_mod.CAP_PROP_FRAME_WIDTH, float(target_width))
                cap.set(cv2_mod.CAP_PROP_FRAME_HEIGHT, float(target_height))
                cap.set(cv2_mod.CAP_PROP_FPS, float(target_fps))
        except Exception:
            pass
        return cap

    def _capture_frame(self, index: int) -> Any | None:
        frame, _ = self._capture_frame_with_fps(index)
        return frame

    def _release_pooled_capture_locked(self, index: int) -> None:
        cap = self._capture_pool.pop(index, None)
        self._capture_pool_reported_fps.pop(index, None)
        self._capture_pool_timestamps.pop(index, None)
        if cap is None:
            return
        try:
            cap.release()
        except Exception:
            pass

    def _release_all_pooled_captures(self) -> None:
        with self._capture_lock:
            for index in list(self._capture_pool.keys()):
                self._release_pooled_capture_locked(index)

    def _sync_capture_pool_indices_locked(self, active_indices: set[int]) -> None:
        for index in list(self._capture_pool.keys()):
            if index not in active_indices:
                self._release_pooled_capture_locked(index)
        for index in list(self._capture_retry_after.keys()):
            if index not in active_indices:
                self._capture_retry_after.pop(index, None)

    def _pooled_capture_for_index_locked(self, index: int) -> Any | None:
        now = time.monotonic()
        retry_after = self._capture_retry_after.get(index, 0.0)
        if now < retry_after:
            return None

        cap = self._capture_pool.get(index)
        if cap is not None and cap.isOpened():
            return cap
        if cap is not None:
            self._release_pooled_capture_locked(index)

        cap = self._open_capture(index)
        if cap is None:
            self._capture_retry_after[index] = now + 1.0
            return None

        if self.cv2_module is not None:
            try:
                cap.set(self.cv2_module.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

        reported_fps = _sanitize_reported_fps(cap.get(self.cv2_module.CAP_PROP_FPS)) if self.cv2_module is not None else None
        self._capture_pool[index] = cap
        self._capture_pool_reported_fps[index] = reported_fps
        self._capture_pool_timestamps[index] = []
        self._capture_retry_after.pop(index, None)
        return cap

    def _capture_live_frame_with_fps(self, index: int) -> tuple[Any | None, float | None]:
        if self.cv2_module is None:
            return None, None

        with self._capture_lock:
            cap = self._pooled_capture_for_index_locked(index)
            if cap is None:
                return None, None
            with self._suppress_stderr():
                ok, frame = cap.read()
            if not ok or frame is None:
                self._release_pooled_capture_locked(index)
                self._capture_retry_after[index] = time.monotonic() + 0.5
                return None, None

            now = time.monotonic()
            timestamps = self._capture_pool_timestamps.setdefault(index, [])
            timestamps.append(now)
            cutoff = now - 2.0
            while len(timestamps) > 2 and timestamps[0] < cutoff:
                timestamps.pop(0)
            fps = _compute_capture_fps(timestamps, reported_fps=self._capture_pool_reported_fps.get(index))
            return frame, fps

    def _capture_frame_with_fps(self, index: int) -> tuple[Any | None, float | None]:
        cap = self._open_capture(index)
        if cap is None:
            return None, None

        latest_frame: Any | None = None
        frame_timestamps: list[float] = []
        reported_fps = _sanitize_reported_fps(cap.get(self.cv2_module.CAP_PROP_FPS)) if self.cv2_module is not None else None
        try:
            with self._suppress_stderr():
                while len(frame_timestamps) < _PREVIEW_FPS_SAMPLE_FRAMES:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    latest_frame = frame
                    frame_timestamps.append(time.monotonic())
                    if len(frame_timestamps) >= 2 and (frame_timestamps[-1] - frame_timestamps[0]) >= _PREVIEW_FPS_SAMPLE_TIMEOUT_S:
                        break
        finally:
            cap.release()

        if latest_frame is None:
            return None, None

        fps = _compute_capture_fps(frame_timestamps, reported_fps=reported_fps)
        return latest_frame, fps

    def _draw_placeholder(self, canvas: Any, text: str) -> None:
        width = int(canvas["width"])
        height = int(canvas["height"])
        canvas.delete("all")
        canvas.create_rectangle(0, 0, width, height, fill=self.colors.get("surface", "#111827"), outline="")
        canvas.create_text(width // 2, height // 2, text=text, fill="#9ca3af", width=max(width - 20, 80))

    def _render_detected_preview(self, index: int, frame_bgr: Any) -> None:
        if self.cv2_module is None:
            return
        cv2_mod = self.cv2_module
        canvas = self.detected_canvases.get(index)
        if canvas is None:
            return
        target_width = int(canvas["width"])
        target_height = int(canvas["height"])
        src_h, src_w = frame_bgr.shape[:2]
        if src_w <= 0 or src_h <= 0:
            return
        scale = min(target_width / float(src_w), target_height / float(src_h))
        resized_w = max(1, int(round(float(src_w) * scale)))
        resized_h = max(1, int(round(float(src_h) * scale)))
        interpolation = cv2_mod.INTER_AREA if scale < 1.0 else cv2_mod.INTER_LINEAR
        resized = cv2_mod.resize(frame_bgr, (resized_w, resized_h), interpolation=interpolation)
        if resized_w != target_width or resized_h != target_height:
            top = max(0, (target_height - resized_h) // 2)
            bottom = max(0, target_height - resized_h - top)
            left = max(0, (target_width - resized_w) // 2)
            right = max(0, target_width - resized_w - left)
            frame = cv2_mod.copyMakeBorder(
                resized,
                top,
                bottom,
                left,
                right,
                cv2_mod.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
        else:
            frame = resized
        ok, encoded = cv2_mod.imencode(".ppm", frame)
        if not ok:
            ok, encoded = cv2_mod.imencode(".png", frame)
        if not ok:
            return
        data = base64.b64encode(encoded.tobytes()).decode("ascii")
        import tkinter as tk

        try:
            photo = tk.PhotoImage(data=data)
        except Exception:
            ok_png, encoded_png = cv2_mod.imencode(".png", frame)
            if not ok_png:
                return
            try:
                photo = tk.PhotoImage(data=base64.b64encode(encoded_png.tobytes()).decode("ascii"))
            except Exception:
                return
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=photo)
        self.detected_photos[index] = photo

    def _apply_button_role_style(self, button: Any, active: bool) -> None:
        button.configure(style="Accent.TButton" if active else "Secondary.TButton")

    def _update_role_ui(self) -> None:
        indices = self._camera_indices()
        laptop_idx = indices["laptop"]
        phone_idx = indices["phone"]

        for index in self.detected_indices:
            label_var = self.role_label_vars.get(index)
            if label_var is None:
                continue

            if index == laptop_idx and index == phone_idx:
                label_var.set("Role: Laptop + Phone")
            elif index == laptop_idx:
                label_var.set("Role: Laptop")
            elif index == phone_idx:
                label_var.set("Role: Phone")
            else:
                label_var.set("Role: Unassigned")

            buttons = self.role_buttons.get(index, {})
            laptop_button = buttons.get("laptop")
            phone_button = buttons.get("phone")
            if laptop_button is not None:
                laptop_button.configure(text="Laptop (Active)" if index == laptop_idx else "Set Laptop")
                self._apply_button_role_style(laptop_button, index == laptop_idx)
            if phone_button is not None:
                phone_button.configure(text="Phone (Active)" if index == phone_idx else "Set Phone")
                self._apply_button_role_style(phone_button, index == phone_idx)

    def _build_detected_port_cards(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        for child in self.detected_ports_frame.winfo_children():
            child.destroy()

        self.detected_canvases = {}
        self.detected_photos = {}
        self.detected_frame_sizes = {}
        self.detected_input_fps = {}
        self.role_label_vars = {}
        self.role_label_widgets = {}
        self.fps_label_vars = {}
        self.role_buttons = {}
        self._detected_cards = []

        if not self.detected_indices:
            self.detected_empty_var.set("No camera ports detected in scan range.")
            self._sync_detected_scroll_region()
            return

        self.detected_empty_var.set("Use Laptop/Phone buttons on each port to set camera roles.")

        for index in self.detected_indices:
            card = ttk.Frame(self.detected_ports_frame, style="Panel.TFrame")
            ttk.Label(card, text=f"Port {index}", style="Field.TLabel").pack(anchor="w")
            self._detected_cards.append((index, card))

            role_var = tk.StringVar(value="Role: Unassigned")
            self.role_label_vars[index] = role_var
            role_label = tk.Label(
                card,
                textvariable=role_var,
                bg=self.colors.get("panel", "#111a2e"),
                fg=self.colors.get("accent", "#93c5fd"),
                font=(self.colors.get("font_ui", "TkDefaultFont"), 9, "bold"),
            )
            role_label.pack(anchor="w")
            self.role_label_widgets[index] = role_label

            fps_var = tk.StringVar(value="Input: n/a @ n/a FPS")
            self.fps_label_vars[index] = fps_var
            fps_label = tk.Label(
                card,
                textvariable=fps_var,
                bg=self.colors.get("panel", "#111a2e"),
                fg=self.colors.get("muted", "#9ca3af"),
                font=(self.colors.get("font_ui", "TkDefaultFont"), 8),
            )
            fps_label.pack(anchor="w")

            canvas = tk.Canvas(
                card,
                width=_PREVIEW_CANVAS_WIDTH,
                height=_PREVIEW_CANVAS_HEIGHT,
                bg=self.colors.get("surface", "#111827"),
                highlightthickness=1,
                highlightbackground=self.colors["border"],
            )
            canvas.pack(anchor="w", pady=(4, 4))
            self.detected_canvases[index] = canvas
            self._draw_placeholder(canvas, "No frame")

            actions = ttk.Frame(card, style="Panel.TFrame")
            actions.pack(anchor="w")
            laptop_button = ttk.Button(actions, text="Set Laptop", style="Secondary.TButton", command=lambda idx=index: self._assign_role("laptop", idx))
            laptop_button.pack(side="left")
            phone_button = ttk.Button(actions, text="Set Phone", style="Secondary.TButton", command=lambda idx=index: self._assign_role("phone", idx))
            phone_button.pack(side="left", padx=(6, 0))
            self.role_buttons[index] = {"laptop": laptop_button, "phone": phone_button}

        self._relayout_detected_cards()
        self._update_role_ui()
        self._update_input_fps_ui()

    def _notify_mapping_changed(self) -> None:
        if self.on_camera_indices_changed is None:
            return
        indices = self._camera_indices()
        self.on_camera_indices_changed(indices["laptop"], indices["phone"])

    def _choose_alternative_index(self, disallow_index: int) -> int | None:
        return choose_alternative_index(self.detected_indices, disallow_index)

    def _assign_role(self, role: str, index: int) -> None:
        assignment = assign_camera_role(
            config=self.config,
            detected_indices=self.detected_indices,
            detected_frame_sizes=self.detected_frame_sizes,
            role=role,
            index=index,
            fingerprint=camera_fingerprint(index),
        )
        if not assignment.ok:
            for message in assignment.messages:
                self.append_log(message)
            self._update_role_ui()
            return

        self.config.clear()
        self.config.update(assignment.updated_config)
        save_config(self.config, quiet=True)
        self._update_role_ui()
        self._notify_mapping_changed()
        for message in assignment.messages:
            self.append_log(message)

    def _stop_busy_status(self, final_text: str | None = None) -> None:
        if self._busy_job is not None:
            try:
                self.root.after_cancel(self._busy_job)
            except Exception:
                pass
            self._busy_job = None
        if final_text is not None:
            self.status_preview_var.set(final_text)

    def _start_busy_status(self, base_text: str) -> None:
        self._stop_busy_status()
        self._busy_ticks = 0

        def _tick() -> None:
            dots = "." * ((self._busy_ticks % 3) + 1)
            self.status_preview_var.set(f"{base_text}{dots}")
            self._busy_ticks += 1
            self._busy_job = self.root.after(280, _tick)

        _tick()

    def _scan_ports_worker(self, limit: int) -> tuple[list[int], int]:
        candidates = self._candidate_scan_indices(limit)
        detected: list[int] = []
        for index in candidates:
            cap = self._open_capture(index)
            if cap is None:
                continue
            detected.append(index)
            cap.release()
        return detected, len(candidates)

    def _set_controls_scanning(self, scanning: bool) -> None:
        state = "disabled" if scanning else "normal"
        self.scan_button.configure(state=state)
        self.refresh_button.configure(state=state)

    def _sync_live_cap_var(self) -> int:
        cap = _normalize_live_preview_fps_cap(self.live_fps_cap_var.get())
        self.live_fps_cap_var.set(str(cap))
        return cap

    def _set_live_button_state(self) -> None:
        if self.live_enabled_var.get():
            self.live_button_text_var.set("Live Paused" if self._live_paused_for_run else "Stop Live")
            self.live_toggle_button.configure(style="Accent.TButton")
        else:
            self.live_button_text_var.set("Start Live")
            self.live_toggle_button.configure(style="Secondary.TButton")

    def _cancel_live_job(self) -> None:
        if self._live_job is None:
            return
        try:
            self.root.after_cancel(self._live_job)
        except Exception:
            pass
        self._live_job = None

    def _schedule_live_tick(self, delay_ms: int | None = None) -> None:
        self._cancel_live_job()
        if not self.live_enabled_var.get() or self._live_paused_for_run:
            return
        interval_ms = live_preview_interval_ms(self._sync_live_cap_var())
        delay = interval_ms if delay_ms is None else max(0, int(delay_ms))
        self._live_job = self.root.after(delay, self._live_tick)

    def _live_tick(self) -> None:
        self._live_job = None
        if not self.live_enabled_var.get() or self._live_paused_for_run:
            return
        if self._live_tick_inflight:
            self._schedule_live_tick()
            return
        self._live_tick_inflight = True

        def _on_complete() -> None:
            self._live_tick_inflight = False
            self._schedule_live_tick()

        self.refresh_camera_previews(log_when_empty=False, from_live=True, on_complete=_on_complete)

    def set_live_enabled(self, enabled: bool) -> None:
        enabled_flag = bool(enabled)
        self.live_enabled_var.set(enabled_flag)
        if not enabled_flag:
            self._cancel_live_job()
            self._live_tick_inflight = False
            self._release_all_pooled_captures()
        else:
            self._schedule_live_tick(delay_ms=0)
        self._set_live_button_state()

    def _on_pause_on_run_toggled(self) -> None:
        pause_enabled = bool(self.pause_on_run_var.get())
        if pause_enabled:
            if self._run_active:
                self.set_active_run(True)
            return

        if self._live_paused_for_run:
            self._live_paused_for_run = False
            if self.live_enabled_var.get():
                self.status_preview_var.set("Pause-on-run disabled. Live preview will continue during active runs.")
                self._schedule_live_tick(delay_ms=0)
        self._set_live_button_state()

    def set_active_run(self, active: bool) -> None:
        active_flag = bool(active)
        self._run_active = active_flag
        pause_on_run = bool(self.pause_on_run_var.get())

        if not pause_on_run:
            self._live_paused_for_run = False
            self._set_live_button_state()
            return

        if active_flag:
            if not self._live_paused_for_run:
                self._live_paused_for_run = True
                if self.live_enabled_var.get():
                    self.status_preview_var.set("Live preview auto-paused while robot run is active.")
            self._cancel_live_job()
            self._release_all_pooled_captures()
            self._set_live_button_state()
            return

        was_paused = self._live_paused_for_run
        self._live_paused_for_run = False
        if was_paused and self.live_enabled_var.get():
            self.status_preview_var.set("Robot run ended. Resuming live preview.")
            self._schedule_live_tick(delay_ms=0)
        self._set_live_button_state()

    def scan_camera_ports(self) -> None:
        if not self.cv2_probe_ok:
            reason = summarize_probe_error(self.cv2_probe_error) if self.cv2_probe_error else "incompatible module"
            self.append_log(f"Camera scan unavailable: {reason}")
            self.detected_ports_var.set("Detected open camera ports: unavailable")
            return
        if not self._ensure_cv2_module():
            self.detected_ports_var.set("Detected open camera ports: unavailable")
            return
        self._release_all_pooled_captures()
        scan_limit = self._scan_limit()

        if self.background_jobs is None:
            detected, candidate_count = self._scan_ports_worker(scan_limit)
            self._apply_scan_result(detected, candidate_count)
            return

        self._set_controls_scanning(True)
        self._start_busy_status("Scanning camera ports")

        def _apply(result: tuple[list[int], int]) -> None:
            detected, candidate_count = result
            self._apply_scan_result(detected, candidate_count)

        self.background_jobs.submit(
            "camera-scan",
            lambda: self._scan_ports_worker(scan_limit),
            on_success=_apply,
            on_error=lambda exc: self._stop_busy_status(f"Scan failed: {exc}"),
            on_complete=lambda _: self._set_controls_scanning(False),
        )

    def _apply_scan_result(self, detected: list[int], candidate_count: int) -> None:
        self.detected_indices = detected
        with self._capture_lock:
            self._sync_capture_pool_indices_locked(set(detected))
        if detected:
            self.detected_ports_var.set(f"Detected open camera ports: {', '.join(str(i) for i in detected)}")
        else:
            self.detected_ports_var.set("Detected open camera ports: none found")

        self._build_detected_port_cards()
        self.refresh_camera_previews(log_when_empty=False)
        self._stop_busy_status(f"Scan complete ({len(detected)}/{candidate_count}).")

        if detected:
            self.append_log(f"Detected camera ports: {', '.join(str(i) for i in detected)}")
        else:
            self.append_log("No open camera ports detected in scan range.")

    def _camera_mapping_summary(self) -> str:
        return camera_mapping_summary(self.config)

    def _camera_input_fps_summary(self) -> str:
        return camera_input_fps_summary(self.config, self.detected_input_fps)

    def _update_input_fps_ui(self) -> None:
        for index in self.detected_indices:
            fps_var = self.fps_label_vars.get(index)
            if fps_var is None:
                continue
            fps = self.detected_input_fps.get(index)
            size = self.detected_frame_sizes.get(index)
            if size is None:
                size_text = "n/a"
            else:
                size_text = f"{size[0]}x{size[1]}"
            fps_text = "n/a" if fps is None else f"{fps:.1f}"
            fps_var.set(f"Input: {size_text} @ {fps_text} FPS")

    def refresh_camera_previews(
        self,
        log_when_empty: bool = True,
        *,
        from_live: bool = False,
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        def _finish() -> None:
            if on_complete is not None:
                try:
                    on_complete()
                except Exception:
                    pass

        if not self.detected_indices:
            if log_when_empty:
                self.append_log("No detected camera ports. Click 'Scan Camera Ports' first.")
            self.status_preview_var.set("No detected ports to refresh.")
            _finish()
            return
        if not self._ensure_cv2_module():
            self.status_preview_var.set("Refresh unavailable.")
            _finish()
            return

        if self.background_jobs is None:
            self._refresh_previews_sync(from_live=from_live)
            _finish()
            return

        if not from_live:
            self.refresh_button.configure(state="disabled")
            self._start_busy_status("Refreshing camera previews")

        def _worker() -> dict[int, tuple[Any | None, float | None]]:
            result: dict[int, tuple[Any | None, float | None]] = {}
            for index in self.detected_indices:
                if from_live:
                    result[index] = self._capture_live_frame_with_fps(index)
                else:
                    result[index] = self._capture_frame_with_fps(index)
            return result

        def _apply(frames: dict[int, tuple[Any | None, float | None]]) -> None:
            refreshed = 0
            for index, (frame, fps) in frames.items():
                if frame is None:
                    canvas = self.detected_canvases.get(index)
                    if canvas is not None:
                        self._draw_placeholder(canvas, "Unavailable")
                    self.detected_input_fps.pop(index, None)
                    continue
                try:
                    h, w = frame.shape[:2]
                    self.detected_frame_sizes[index] = (int(w), int(h))
                except Exception:
                    pass
                if fps is not None:
                    self.detected_input_fps[index] = fps
                else:
                    self.detected_input_fps.pop(index, None)
                self._render_detected_preview(index, frame)
                refreshed += 1
            self._update_role_ui()
            self._update_input_fps_ui()
            timestamp = time.strftime("%H:%M:%S")
            if from_live:
                self.status_preview_var.set(
                    f"Live preview {timestamp} ({refreshed}/{len(self.detected_indices)})"
                    f" | {self._camera_mapping_summary()} | {self._camera_input_fps_summary()}"
                )
            else:
                self._stop_busy_status(
                    f"Preview refreshed at {timestamp} ({refreshed}/{len(self.detected_indices)})"
                    f" | {self._camera_mapping_summary()} | {self._camera_input_fps_summary()}"
                )

        self.background_jobs.submit(
            "camera-preview-live" if from_live else "camera-preview-refresh",
            _worker,
            on_success=_apply,
            on_error=(
                (lambda exc: self.status_preview_var.set(f"Live preview failed: {exc}"))
                if from_live
                else (lambda exc: self._stop_busy_status(f"Preview refresh failed: {exc}"))
            ),
            on_complete=(
                (lambda _: _finish())
                if from_live
                else (lambda _: (self.refresh_button.configure(state="normal"), _finish()))
            ),
        )

    def _refresh_previews_sync(self, *, from_live: bool = False) -> None:
        refreshed = 0
        for index in self.detected_indices:
            frame, fps = self._capture_frame_with_fps(index)
            if frame is None:
                canvas = self.detected_canvases.get(index)
                if canvas is not None:
                    self._draw_placeholder(canvas, "Unavailable")
                self.detected_input_fps.pop(index, None)
                continue
            try:
                h, w = frame.shape[:2]
                self.detected_frame_sizes[index] = (int(w), int(h))
            except Exception:
                pass
            if fps is not None:
                self.detected_input_fps[index] = fps
            else:
                self.detected_input_fps.pop(index, None)
            self._render_detected_preview(index, frame)
            refreshed += 1

        self._update_role_ui()
        self._update_input_fps_ui()
        timestamp = time.strftime("%H:%M:%S")
        if from_live:
            self.status_preview_var.set(
                f"Live preview {timestamp} ({refreshed}/{len(self.detected_indices)})"
                f" | {self._camera_mapping_summary()} | {self._camera_input_fps_summary()}"
            )
        else:
            self.status_preview_var.set(
                f"Preview refreshed at {timestamp} ({refreshed}/{len(self.detected_indices)})"
                f" | {self._camera_mapping_summary()} | {self._camera_input_fps_summary()}"
            )

    def apply_theme(self, colors: dict[str, str]) -> None:
        self.colors = colors
        if self.detected_ports_canvas is not None:
            self.detected_ports_canvas.configure(bg=self.colors.get("bg", "#070b14"))
        for index, canvas in self.detected_canvases.items():
            canvas.configure(bg=self.colors.get("surface", "#111827"), highlightbackground=self.colors.get("border", "#2d2d2d"))
            if index not in self.detected_photos:
                self._draw_placeholder(canvas, "No frame")
        for label in self.role_label_widgets.values():
            try:
                label.configure(
                    bg=self.colors.get("panel", "#111a2e"),
                    fg=self.colors.get("accent", "#93c5fd"),
                    font=(self.colors.get("font_ui", "TkDefaultFont"), 9, "bold"),
                )
            except Exception:
                pass

    def refresh_labels(self) -> None:
        # Public hook used by parent UI when config changes externally.
        self._update_role_ui()

    def export_state(self) -> dict[str, Any]:
        return export_camera_preview_state(
            detected_indices=self.detected_indices,
            detected_frame_sizes=self.detected_frame_sizes,
            detected_input_fps=self.detected_input_fps,
            status_text=str(self.status_preview_var.get()),
            detected_ports_text=str(self.detected_ports_var.get()),
            scan_limit=str(self.scan_limit_var.get()),
            live_fps_cap=str(self.live_fps_cap_var.get()),
            live_enabled=bool(self.live_enabled_var.get()),
            pause_on_run=bool(self.pause_on_run_var.get()),
            run_active=bool(self._run_active),
            live_paused_for_run=bool(self._live_paused_for_run),
        )

    def restore_state(self, state: dict[str, Any] | None) -> None:
        snapshot: CameraPreviewSnapshot = restore_camera_preview_state(state)

        self.detected_indices = snapshot.detected_indices
        self.detected_frame_sizes = snapshot.detected_frame_sizes
        self.detected_input_fps = snapshot.detected_input_fps
        self.status_preview_var.set(snapshot.status_text)
        self.detected_ports_var.set(snapshot.detected_ports_text)
        self.scan_limit_var.set(snapshot.scan_limit)
        self.live_fps_cap_var.set(snapshot.live_fps_cap)
        self.live_enabled_var.set(snapshot.live_enabled)
        self.pause_on_run_var.set(snapshot.pause_on_run)
        self._run_active = snapshot.run_active
        self._live_paused_for_run = snapshot.live_paused_for_run
        self._live_tick_inflight = False

        with self._capture_lock:
            self._sync_capture_pool_indices_locked(set(detected))

        self._build_detected_port_cards()
        self._update_role_ui()
        self._update_input_fps_ui()
        self._set_live_button_state()

    def start(self) -> None:
        if self.live_enabled_var.get():
            self._schedule_live_tick(delay_ms=0)
            return
        self.refresh_camera_previews(log_when_empty=False)

    def stop(self) -> None:
        self._cancel_live_job()
        self._live_tick_inflight = False
        self._release_all_pooled_captures()

    def toggle(self) -> None:
        target = not self.live_enabled_var.get()
        self.set_live_enabled(target)
        if target:
            self.status_preview_var.set(
                f"Live preview enabled ({self._sync_live_cap_var()} FPS cap)."
            )
            if not self.detected_indices:
                self.append_log("Live preview enabled. Scan camera ports to start streaming.")
        else:
            self.status_preview_var.set("Live preview stopped.")

    def close(self) -> None:
        self._stop_busy_status()
        self.set_live_enabled(False)
        self.stop()
