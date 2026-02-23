from __future__ import annotations

import base64
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

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
        self.running = False
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.captures: dict[str, Any] = {"laptop": None, "phone": None}
        self.photos: dict[str, Any] = {}
        self.detected_photos: dict[int, Any] = {}
        self.detected_canvases: dict[int, Any] = {}
        self.status_preview_var = self._stringvar("Preview stopped.")
        self.cv2_module: Any | None = None
        self.detected_indices: list[int] = []
        self.detected_ports_var = self._stringvar("Detected open camera ports: (scan to detect)")
        self.selected_port_var = self._stringvar("")
        self.scan_limit_var = self._stringvar("14")

        controls = ttk.Frame(self.frame, style="Panel.TFrame")
        controls.pack(fill="x", pady=(0, 8))
        self.toggle_button = ttk.Button(controls, text="Preview Cameras", command=self.toggle)
        self.toggle_button.pack(side="left")
        ttk.Label(controls, textvariable=self.status_preview_var, style="Muted.TLabel").pack(side="left", padx=(10, 0))

        mapping = ttk.Frame(self.frame, style="Panel.TFrame")
        mapping.pack(fill="x", pady=(0, 8))
        self.scan_button = ttk.Button(mapping, text="Scan Camera Ports", command=self.scan_camera_ports)
        self.scan_button.grid(row=0, column=0, sticky="w")
        ttk.Label(mapping, text="Max idx", style="Muted.TLabel").grid(row=0, column=1, sticky="w", padx=(10, 4))
        ttk.Entry(mapping, textvariable=self.scan_limit_var, width=5).grid(row=0, column=2, sticky="w")
        ttk.Label(mapping, textvariable=self.detected_ports_var, style="Muted.TLabel").grid(
            row=0,
            column=3,
            sticky="w",
            padx=(10, 0),
        )

        ttk.Label(mapping, text="Detected port", style="Field.TLabel").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.port_combo = ttk.Combobox(mapping, textvariable=self.selected_port_var, state="readonly", width=8)
        self.port_combo.grid(row=1, column=1, sticky="w", pady=(6, 0), padx=(10, 0))
        self.set_laptop_button = ttk.Button(mapping, text="Set as Laptop", command=self.set_selected_as_laptop)
        self.set_laptop_button.grid(row=1, column=2, sticky="w", padx=(10, 0), pady=(6, 0))
        self.set_phone_button = ttk.Button(mapping, text="Set as Phone", command=self.set_selected_as_phone)
        self.set_phone_button.grid(row=1, column=3, sticky="w", padx=(10, 0), pady=(6, 0))
        self.swap_button = ttk.Button(mapping, text="Swap Laptop/Phone", command=self.swap_laptop_and_phone)
        self.swap_button.grid(row=1, column=4, sticky="w", padx=(10, 0), pady=(6, 0))
        mapping.columnconfigure(3, weight=1)

        detected_wrap = ttk.LabelFrame(self.frame, text="Detected Camera Ports", style="Section.TLabelframe", padding=8)
        detected_wrap.pack(fill="x", pady=(0, 8))
        self.detected_empty_var = self._stringvar("No detected camera previews yet. Click 'Scan Camera Ports'.")
        ttk.Label(detected_wrap, textvariable=self.detected_empty_var, style="Muted.TLabel").pack(anchor="w")
        self.detected_ports_frame = ttk.Frame(detected_wrap, style="Panel.TFrame")
        self.detected_ports_frame.pack(fill="x", pady=(6, 0))

        feeds = ttk.Frame(self.frame, style="Panel.TFrame")
        feeds.pack(fill="x")
        self.camera_labels: dict[str, Any] = {}
        self.canvases: dict[str, Any] = {}
        for col, key in enumerate(("laptop", "phone")):
            pane = ttk.Frame(feeds, style="Panel.TFrame")
            pane.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 10, 0))
            feeds.columnconfigure(col, weight=1)
            label_var = self._stringvar("")
            self.camera_labels[key] = label_var
            ttk.Label(pane, textvariable=label_var, style="Field.TLabel").pack(anchor="w", pady=(0, 4))
            canvas = self._canvas(pane, width=320, height=240)
            canvas.pack(anchor="w")
            self.canvases[key] = canvas
            self._draw_placeholder(canvas, "Preview stopped")

        self._refresh_detected_port_widgets()
        self.refresh_labels()
        if self.cv2_probe_ok:
            self.root.after(250, self.scan_camera_ports)

    def _stringvar(self, value: str) -> Any:
        import tkinter as tk

        return tk.StringVar(value=value)

    def _canvas(self, parent: Any, width: int, height: int) -> Any:
        import tkinter as tk

        return tk.Canvas(
            parent,
            width=width,
            height=height,
            bg="#111827",
            highlightthickness=1,
            highlightbackground=self.colors["border"],
        )

    @contextmanager
    def _suppress_stderr(self) -> Iterator[None]:
        # OpenCV/V4L probing emits noisy warnings/errors to stderr for missing indices.
        # Suppress only around probe/open attempts to keep terminal output clean.
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

    def _camera_shape(self) -> tuple[int, int, int]:
        width = max(int(self.config.get("camera_width", 640)), 160)
        height = max(int(self.config.get("camera_height", 360)), 120)
        fps = max(int(self.config.get("camera_fps", 30)), 1)
        return width, height, fps

    def _draw_placeholder(self, canvas: Any, text: str) -> None:
        width = int(canvas["width"])
        height = int(canvas["height"])
        canvas.delete("all")
        canvas.create_rectangle(0, 0, width, height, fill="#111827", outline="")
        canvas.create_text(width // 2, height // 2, text=text, fill="#9ca3af", width=max(width - 20, 80))

    def _draw_placeholder_role(self, key: str, text: str) -> None:
        self._draw_placeholder(self.canvases[key], text)

    def _refresh_detected_port_widgets(self) -> None:
        values = [str(idx) for idx in self.detected_indices]
        self.port_combo.configure(values=values)
        if values:
            selected = self.selected_port_var.get().strip()
            if selected not in values:
                self.selected_port_var.set(values[0])
            self.detected_ports_var.set(f"Detected open camera ports: {', '.join(values)}")
        else:
            self.selected_port_var.set("")
            self.detected_ports_var.set("Detected open camera ports: none found")

    def refresh_labels(self) -> None:
        indices = self._camera_indices()
        self.camera_labels["laptop"].set(f"Laptop camera (index {indices['laptop']})")
        self.camera_labels["phone"].set(f"Phone camera (index {indices['phone']})")

    def _render_frame(self, key: str, frame_rgb: Any) -> None:
        if self.cv2_module is None:
            self._draw_placeholder_role(key, "Preview unavailable")
            return
        cv2_mod = self.cv2_module
        ok, encoded = cv2_mod.imencode(".png", cv2_mod.cvtColor(frame_rgb, cv2_mod.COLOR_RGB2BGR))
        if not ok:
            self._draw_placeholder_role(key, "Frame encode failed")
            return
        data = base64.b64encode(encoded.tobytes()).decode("ascii")
        import tkinter as tk

        photo = tk.PhotoImage(data=data)

        canvas = self.canvases[key]
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=photo)
        self.photos[key] = photo

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

    def _capture_loop(self) -> None:
        if self.cv2_module is None:
            return
        cv2_mod = self.cv2_module
        indices = self._camera_indices()
        while not self.stop_event.is_set():
            saw_frame = False
            for key, cap in self.captures.items():
                if cap is None:
                    continue
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                saw_frame = True
                frame = cv2_mod.resize(frame, (320, 240), interpolation=cv2_mod.INTER_AREA)
                cv2_mod.putText(
                    frame,
                    f"{key.title()} idx {indices[key]}",
                    (8, 20),
                    cv2_mod.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2_mod.LINE_AA,
                )
                frame_rgb = cv2_mod.cvtColor(frame, cv2_mod.COLOR_BGR2RGB)
                self.root.after(0, self._render_frame, key, frame_rgb)
            time.sleep(0.03 if saw_frame else 0.12)

    def _ensure_cv2_module(self) -> bool:
        if self.cv2_module is not None:
            return True
        try:
            import cv2 as cv2_loaded  # type: ignore[import-not-found]
        except Exception as exc:
            self.status_preview_var.set("OpenCV import failed.")
            self._draw_placeholder_role("laptop", "OpenCV import failed")
            self._draw_placeholder_role("phone", "OpenCV import failed")
            self.append_log(f"Camera preview unavailable: {exc}")
            return False
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

    def _build_detected_port_cards(self) -> None:
        from tkinter import ttk

        for child in self.detected_ports_frame.winfo_children():
            child.destroy()

        self.detected_canvases = {}
        self.detected_photos = {}

        if not self.detected_indices:
            self.detected_empty_var.set("No camera ports detected in scan range.")
            return

        self.detected_empty_var.set("Click a card button to map a port to laptop or phone.")

        for i, index in enumerate(self.detected_indices):
            card = ttk.Frame(self.detected_ports_frame, style="Panel.TFrame")
            card.grid(row=i // 3, column=i % 3, sticky="nw", padx=(0, 10), pady=(0, 10))
            ttk.Label(card, text=f"Port {index}", style="Field.TLabel").pack(anchor="w")
            canvas = self._canvas(card, width=220, height=140)
            canvas.pack(anchor="w", pady=(4, 4))
            self.detected_canvases[index] = canvas
            self._draw_placeholder(canvas, "No frame")

            actions = ttk.Frame(card, style="Panel.TFrame")
            actions.pack(anchor="w")
            ttk.Button(actions, text="Laptop", command=lambda idx=index: self._apply_role_index("laptop", idx)).pack(
                side="left"
            )
            ttk.Button(actions, text="Phone", command=lambda idx=index: self._apply_role_index("phone", idx)).pack(
                side="left",
                padx=(6, 0),
            )

    def _refresh_detected_port_previews(self) -> None:
        if not self.detected_indices or self.cv2_module is None:
            return
        cv2_mod = self.cv2_module
        for index in self.detected_indices:
            cap = self._open_capture(index)
            if cap is None:
                canvas = self.detected_canvases.get(index)
                if canvas is not None:
                    self._draw_placeholder(canvas, "Busy/unavailable")
                continue
            with self._suppress_stderr():
                ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                canvas = self.detected_canvases.get(index)
                if canvas is not None:
                    self._draw_placeholder(canvas, "No frame")
                continue
            self._render_detected_preview(index, frame)

    def scan_camera_ports(self) -> None:
        if not self.cv2_probe_ok:
            reason = summarize_probe_error(self.cv2_probe_error) if self.cv2_probe_error else "incompatible module"
            self.append_log(f"Camera scan unavailable: {reason}")
            self.detected_ports_var.set("Detected open camera ports: unavailable")
            return
        if not self._ensure_cv2_module():
            self.detected_ports_var.set("Detected open camera ports: unavailable")
            return

        was_running = self.running
        previous_status = self.status_preview_var.get()
        if was_running:
            self.stop()

        limit = self._scan_limit()
        candidates = self._candidate_scan_indices(limit)
        self.scan_button.configure(state="disabled")
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
        self._refresh_detected_port_widgets()
        self._build_detected_port_cards()
        self._refresh_detected_port_previews()
        self.scan_button.configure(state="normal")

        if detected:
            self.append_log(f"Detected camera ports: {', '.join(str(i) for i in detected)}")
        else:
            self.append_log("No open camera ports detected in scan range.")

        if was_running:
            self.start()
        else:
            self.status_preview_var.set(previous_status if previous_status != "Preview running." else "Preview stopped.")

    def _selected_port_index(self) -> int | None:
        raw = self.selected_port_var.get().strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def _notify_mapping_changed(self) -> None:
        if self.on_camera_indices_changed is None:
            return
        indices = self._camera_indices()
        self.on_camera_indices_changed(indices["laptop"], indices["phone"])

    def _apply_role_index(self, role: str, index: int) -> None:
        key = f"camera_{role}_index"
        current = int(self.config.get(key, -1))
        if current == index:
            self.append_log(f"{role.title()} camera already set to index {index}.")
            return

        self.config[key] = index
        self.refresh_labels()
        self.append_log(f"Set {role} camera index to {index}.")
        self._notify_mapping_changed()

        if self.running:
            self.stop()
            self.start()

    def set_selected_as_laptop(self) -> None:
        selected = self._selected_port_index()
        if selected is None:
            self.append_log("No detected camera port selected for laptop assignment.")
            return
        self._apply_role_index("laptop", selected)

    def set_selected_as_phone(self) -> None:
        selected = self._selected_port_index()
        if selected is None:
            self.append_log("No detected camera port selected for phone assignment.")
            return
        self._apply_role_index("phone", selected)

    def swap_laptop_and_phone(self) -> None:
        indices = self._camera_indices()
        self.config["camera_laptop_index"] = indices["phone"]
        self.config["camera_phone_index"] = indices["laptop"]
        self.refresh_labels()
        self.append_log(
            f"Swapped camera roles: laptop={self.config['camera_laptop_index']}, phone={self.config['camera_phone_index']}."
        )
        self._notify_mapping_changed()

        if self.running:
            self.stop()
            self.start()

    def start(self) -> None:
        if self.running:
            return
        self.refresh_labels()
        if not self.cv2_probe_ok:
            self.status_preview_var.set("OpenCV unavailable for this macOS/Python.")
            self._draw_placeholder_role("laptop", "OpenCV unavailable")
            self._draw_placeholder_role("phone", "OpenCV unavailable")
            reason = summarize_probe_error(self.cv2_probe_error) if self.cv2_probe_error else "incompatible module"
            self.append_log(f"Camera preview disabled: {reason}")
            return

        if not self._ensure_cv2_module():
            return

        cv2_mod = self.cv2_module
        if cv2_mod is None:
            self.status_preview_var.set("OpenCV unavailable.")
            self._draw_placeholder_role("laptop", "OpenCV unavailable")
            self._draw_placeholder_role("phone", "OpenCV unavailable")
            return

        width, height, fps = self._camera_shape()
        indices = self._camera_indices()
        self.stop_event.clear()
        self.running = True
        self.toggle_button.configure(text="Stop Preview")
        self.status_preview_var.set("Preview running.")
        self.captures = {"laptop": None, "phone": None}

        for key, index in indices.items():
            cap = self._open_capture(index)
            if cap is not None:
                cap.set(cv2_mod.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2_mod.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2_mod.CAP_PROP_FPS, fps)
                self.captures[key] = cap
            else:
                self.captures[key] = None
                self._draw_placeholder_role(key, f"Camera index {index} unavailable")

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if not self.running:
            return
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        for cap in self.captures.values():
            if cap is not None:
                cap.release()
        self.captures = {"laptop": None, "phone": None}
        self.thread = None
        self.running = False
        self.toggle_button.configure(text="Preview Cameras")
        self.status_preview_var.set("Preview stopped.")
        self._draw_placeholder_role("laptop", "Preview stopped")
        self._draw_placeholder_role("phone", "Preview stopped")

    def toggle(self) -> None:
        if self.running:
            self.stop()
        else:
            self.start()

    def close(self) -> None:
        self.stop()
