from __future__ import annotations

import base64
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from .config_store import save_config
from .gui_async import UiBackgroundJobs
from .probes import camera_fingerprint, summarize_probe_error

_PREVIEW_FPS_SAMPLE_FRAMES = 12
_PREVIEW_FPS_SAMPLE_TIMEOUT_S = 0.8


def _normalize_scan_limit(raw: str) -> int:
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        value = 14
    return max(1, min(value, 64))


def _compute_capture_fps(frame_count: int, elapsed_s: float) -> float | None:
    if frame_count < 2:
        return None
    if elapsed_s <= 0:
        return None
    return frame_count / elapsed_s


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

        controls = ttk.Frame(self.frame, style="Panel.TFrame")
        controls.pack(fill="x", pady=(0, 8))

        self.scan_button = ttk.Button(controls, text="Scan Camera Ports", command=self.scan_camera_ports)
        self.scan_button.grid(row=0, column=0, sticky="w")

        self.refresh_button = ttk.Button(controls, text="Refresh Camera Preview", command=self.refresh_camera_previews)
        self.refresh_button.grid(row=0, column=1, sticky="w", padx=(8, 0))

        ttk.Label(controls, text="Max idx", style="Muted.TLabel").grid(row=0, column=2, sticky="w", padx=(12, 4))
        ttk.Entry(controls, textvariable=self.scan_limit_var, width=5).grid(row=0, column=3, sticky="w")

        ttk.Label(controls, textvariable=self.status_preview_var, style="Muted.TLabel").grid(
            row=0,
            column=4,
            sticky="w",
            padx=(10, 0),
        )
        controls.columnconfigure(4, weight=1)

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
        return {
            "laptop": int(self.config.get("camera_laptop_index", 0)),
            "phone": int(self.config.get("camera_phone_index", 1)),
        }

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
        return cap

    def _capture_frame(self, index: int) -> Any | None:
        frame, _ = self._capture_frame_with_fps(index)
        return frame

    def _capture_frame_with_fps(self, index: int) -> tuple[Any | None, float | None]:
        cap = self._open_capture(index)
        if cap is None:
            return None, None

        frame_count = 0
        latest_frame: Any | None = None
        start_t = time.monotonic()
        try:
            with self._suppress_stderr():
                while frame_count < _PREVIEW_FPS_SAMPLE_FRAMES:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    latest_frame = frame
                    frame_count += 1
                    if (time.monotonic() - start_t) >= _PREVIEW_FPS_SAMPLE_TIMEOUT_S:
                        break
        finally:
            cap.release()

        if latest_frame is None:
            return None, None

        elapsed_s = time.monotonic() - start_t
        fps = _compute_capture_fps(frame_count, elapsed_s)
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
        frame = cv2_mod.resize(frame_bgr, (220, 140), interpolation=cv2_mod.INTER_AREA)
        cv2_mod.putText(
            frame,
            f"index {index}",
            (8, 20),
            cv2_mod.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2_mod.LINE_AA,
        )
        ok, encoded = cv2_mod.imencode(".png", frame)
        if not ok:
            return
        data = base64.b64encode(encoded.tobytes()).decode("ascii")
        import tkinter as tk

        photo = tk.PhotoImage(data=data)
        canvas = self.detected_canvases.get(index)
        if canvas is None:
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

            fps_var = tk.StringVar(value="Input FPS: -")
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
                width=220,
                height=140,
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
        for idx in self.detected_indices:
            if idx != disallow_index:
                return idx
        return None

    def _assign_role(self, role: str, index: int) -> None:
        role_key = f"camera_{role}_index"
        other_role = "phone" if role == "laptop" else "laptop"
        other_key = f"camera_{other_role}_index"

        previous_role_index = int(self.config.get(role_key, -1))
        previous_other_index = int(self.config.get(other_key, -1))

        self.config[role_key] = index

        if previous_other_index == index:
            fallback = self._choose_alternative_index(index)
            if fallback is None and previous_role_index != index:
                fallback = previous_role_index
            if fallback is None:
                self.config[role_key] = previous_role_index
                self.append_log("Could not assign role: laptop/phone must use two different ports.")
                self._update_role_ui()
                return
            self.config[other_key] = fallback

        detected_size = self.detected_frame_sizes.get(index)
        if detected_size is not None:
            width, height = detected_size
            self.append_log(f"Mapped {role} camera {index} (detected frame {width}x{height}).")

        fingerprint = camera_fingerprint(index)
        if fingerprint:
            self.config[f"camera_{role}_fingerprint"] = fingerprint
            self.append_log(f"Saved {role} camera fingerprint.")

        # Persist role mapping immediately.
        save_config(self.config, quiet=True)

        self._update_role_ui()
        self._notify_mapping_changed()
        self.append_log(
            f"Set roles: laptop={self.config.get('camera_laptop_index')}, phone={self.config.get('camera_phone_index')}"
        )

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

    def scan_camera_ports(self) -> None:
        if not self.cv2_probe_ok:
            reason = summarize_probe_error(self.cv2_probe_error) if self.cv2_probe_error else "incompatible module"
            self.append_log(f"Camera scan unavailable: {reason}")
            self.detected_ports_var.set("Detected open camera ports: unavailable")
            return
        if not self._ensure_cv2_module():
            self.detected_ports_var.set("Detected open camera ports: unavailable")
            return
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
        indices = self._camera_indices()
        return f"laptop={indices['laptop']} phone={indices['phone']}"

    def _camera_input_fps_summary(self) -> str:
        indices = self._camera_indices()
        parts: list[str] = []
        for role in ("laptop", "phone"):
            idx = indices[role]
            fps = self.detected_input_fps.get(idx)
            if fps is None:
                parts.append(f"{role}=n/a")
            else:
                parts.append(f"{role}={fps:.1f}")
        return "input fps " + " ".join(parts)

    def _update_input_fps_ui(self) -> None:
        for index in self.detected_indices:
            fps_var = self.fps_label_vars.get(index)
            if fps_var is None:
                continue
            fps = self.detected_input_fps.get(index)
            if fps is None:
                fps_var.set("Input FPS: n/a")
            else:
                fps_var.set(f"Input FPS: {fps:.1f}")

    def refresh_camera_previews(self, log_when_empty: bool = True) -> None:
        if not self.detected_indices:
            if log_when_empty:
                self.append_log("No detected camera ports. Click 'Scan Camera Ports' first.")
            self.status_preview_var.set("No detected ports to refresh.")
            return
        if not self._ensure_cv2_module():
            self.status_preview_var.set("Refresh unavailable.")
            return

        if self.background_jobs is None:
            self._refresh_previews_sync()
            return

        self.refresh_button.configure(state="disabled")
        self._start_busy_status("Refreshing camera previews")

        def _worker() -> dict[int, tuple[Any | None, float | None]]:
            result: dict[int, tuple[Any | None, float | None]] = {}
            for index in self.detected_indices:
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
            self._stop_busy_status(
                f"Preview refreshed at {timestamp} ({refreshed}/{len(self.detected_indices)})"
                f" | {self._camera_mapping_summary()} | {self._camera_input_fps_summary()}"
            )

        self.background_jobs.submit(
            "camera-preview-refresh",
            _worker,
            on_success=_apply,
            on_error=lambda exc: self._stop_busy_status(f"Preview refresh failed: {exc}"),
            on_complete=lambda _: self.refresh_button.configure(state="normal"),
        )

    def _refresh_previews_sync(self) -> None:
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

    def start(self) -> None:
        # Backwards-compatible no-op: widget now uses static snapshots only.
        self.refresh_camera_previews(log_when_empty=False)

    def stop(self) -> None:
        # Backwards-compatible no-op: no live stream loop is active.
        return

    def toggle(self) -> None:
        # Backwards-compatible no-op.
        self.refresh_camera_previews(log_when_empty=False)

    def close(self) -> None:
        self._stop_busy_status()
        self.stop()
