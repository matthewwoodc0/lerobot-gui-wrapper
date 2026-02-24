from __future__ import annotations

import base64
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from .config_store import save_config
from .probes import summarize_probe_error


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

        self.frame = ttk.LabelFrame(parent, text=title, style="Section.TLabelframe", padding=10)
        self.frame.pack(fill="x", pady=(10, 0))

        self.cv2_module: Any | None = None
        self.detected_indices: list[int] = []

        self.detected_canvases: dict[int, Any] = {}
        self.detected_photos: dict[int, Any] = {}
        self.detected_frame_sizes: dict[int, tuple[int, int]] = {}
        self.role_label_vars: dict[int, Any] = {}
        self.role_buttons: dict[int, dict[str, Any]] = {}

        self.status_preview_var = tk.StringVar(value="Preview idle.")
        self.detected_ports_var = tk.StringVar(value="Detected open camera ports: (scan to detect)")
        self.detected_empty_var = tk.StringVar(value="No detected camera previews yet. Click 'Scan Camera Ports'.")
        self.scan_limit_var = tk.StringVar(value="14")

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
        detected_wrap.pack(fill="x", pady=(0, 8))
        ttk.Label(detected_wrap, textvariable=self.detected_empty_var, style="Muted.TLabel").pack(anchor="w")
        self.detected_ports_frame = ttk.Frame(detected_wrap, style="Panel.TFrame")
        self.detected_ports_frame.pack(fill="x", pady=(6, 0))

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
        raw = self.scan_limit_var.get().strip()
        try:
            value = int(raw)
        except ValueError:
            value = 14
        value = max(1, min(value, 64))
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
        with self._suppress_stderr():
            cap = self.cv2_module.VideoCapture(index)
        if cap is None:
            return None
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    def _capture_frame(self, index: int) -> Any | None:
        cap = self._open_capture(index)
        if cap is None:
            return None
        with self._suppress_stderr():
            ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        return frame

    def _draw_placeholder(self, canvas: Any, text: str) -> None:
        width = int(canvas["width"])
        height = int(canvas["height"])
        canvas.delete("all")
        canvas.create_rectangle(0, 0, width, height, fill="#111827", outline="")
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
        if active:
            button.configure(relief="sunken", bg="#334155", fg="#f8fafc", activebackground="#334155", activeforeground="#f8fafc")
        else:
            button.configure(relief="raised", bg="#1f2937", fg="#d4d4d4", activebackground="#374151", activeforeground="#ffffff")

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
        self.role_label_vars = {}
        self.role_buttons = {}

        if not self.detected_indices:
            self.detected_empty_var.set("No camera ports detected in scan range.")
            return

        self.detected_empty_var.set("Use Laptop/Phone buttons on each port to set camera roles.")

        for i, index in enumerate(self.detected_indices):
            card = ttk.Frame(self.detected_ports_frame, style="Panel.TFrame")
            card.grid(row=i // 3, column=i % 3, sticky="nw", padx=(0, 10), pady=(0, 10))
            ttk.Label(card, text=f"Port {index}", style="Field.TLabel").pack(anchor="w")

            role_var = tk.StringVar(value="Role: Unassigned")
            self.role_label_vars[index] = role_var
            tk.Label(card, textvariable=role_var, bg="#111a2e", fg="#93c5fd", font=("Helvetica", 9, "bold")).pack(anchor="w")

            canvas = tk.Canvas(
                card,
                width=220,
                height=140,
                bg="#111827",
                highlightthickness=1,
                highlightbackground=self.colors["border"],
            )
            canvas.pack(anchor="w", pady=(4, 4))
            self.detected_canvases[index] = canvas
            self._draw_placeholder(canvas, "No frame")

            actions = tk.Frame(card, bg="#111a2e")
            actions.pack(anchor="w")
            laptop_button = tk.Button(actions, text="Set Laptop", command=lambda idx=index: self._assign_role("laptop", idx), padx=8)
            laptop_button.pack(side="left")
            phone_button = tk.Button(actions, text="Set Phone", command=lambda idx=index: self._assign_role("phone", idx), padx=8)
            phone_button.pack(side="left", padx=(6, 0))
            self.role_buttons[index] = {"laptop": laptop_button, "phone": phone_button}

        self._update_role_ui()

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
            self.config[f"camera_{role}_width"] = int(width)
            self.config[f"camera_{role}_height"] = int(height)
            self.append_log(f"Mapped {role} camera {index} to {width}x{height}.")

        # Persist role and per-role resolution immediately so command generation remains stable.
        save_config(self.config, quiet=True)

        self._update_role_ui()
        self._notify_mapping_changed()
        self.append_log(
            f"Set roles: laptop={self.config.get('camera_laptop_index')}, phone={self.config.get('camera_phone_index')}"
        )

    def scan_camera_ports(self) -> None:
        if not self.cv2_probe_ok:
            reason = summarize_probe_error(self.cv2_probe_error) if self.cv2_probe_error else "incompatible module"
            self.append_log(f"Camera scan unavailable: {reason}")
            self.detected_ports_var.set("Detected open camera ports: unavailable")
            return
        if not self._ensure_cv2_module():
            self.detected_ports_var.set("Detected open camera ports: unavailable")
            return

        limit = self._scan_limit()
        candidates = self._candidate_scan_indices(limit)
        self.scan_button.configure(state="disabled")
        self.refresh_button.configure(state="disabled")
        self.status_preview_var.set(f"Scanning camera ports ({len(candidates)} candidates)...")
        self.root.update_idletasks()

        detected: list[int] = []
        for index in candidates:
            cap = self._open_capture(index)
            if cap is None:
                continue
            detected.append(index)
            cap.release()

        self.detected_indices = detected
        if detected:
            self.detected_ports_var.set(f"Detected open camera ports: {', '.join(str(i) for i in detected)}")
        else:
            self.detected_ports_var.set("Detected open camera ports: none found")

        self._build_detected_port_cards()
        self.refresh_camera_previews(log_when_empty=False)

        self.scan_button.configure(state="normal")
        self.refresh_button.configure(state="normal")
        self.status_preview_var.set("Scan complete.")

        if detected:
            self.append_log(f"Detected camera ports: {', '.join(str(i) for i in detected)}")
        else:
            self.append_log("No open camera ports detected in scan range.")

    def _camera_mapping_summary(self) -> str:
        indices = self._camera_indices()
        return f"laptop={indices['laptop']} phone={indices['phone']}"

    def refresh_camera_previews(self, log_when_empty: bool = True) -> None:
        if not self.detected_indices:
            if log_when_empty:
                self.append_log("No detected camera ports. Click 'Scan Camera Ports' first.")
            self.status_preview_var.set("No detected ports to refresh.")
            return
        if not self._ensure_cv2_module():
            self.status_preview_var.set("Refresh unavailable.")
            return

        refreshed = 0
        for index in self.detected_indices:
            frame = self._capture_frame(index)
            if frame is None:
                canvas = self.detected_canvases.get(index)
                if canvas is not None:
                    self._draw_placeholder(canvas, "Unavailable")
                continue
            try:
                h, w = frame.shape[:2]
                self.detected_frame_sizes[index] = (int(w), int(h))
            except Exception:
                pass
            self._render_detected_preview(index, frame)
            refreshed += 1

        self._update_role_ui()
        timestamp = time.strftime("%H:%M:%S")
        self.status_preview_var.set(
            f"Preview refreshed at {timestamp} ({refreshed}/{len(self.detected_indices)}) | {self._camera_mapping_summary()}"
        )

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
        self.stop()
