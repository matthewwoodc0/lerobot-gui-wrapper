from __future__ import annotations

import base64
import threading
import time
from typing import Any, Callable

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
    ) -> None:
        from tkinter import ttk

        self.root = root
        self.config = config
        self.colors = colors
        self.cv2_probe_ok = cv2_probe_ok
        self.cv2_probe_error = cv2_probe_error
        self.append_log = append_log

        self.frame = ttk.LabelFrame(parent, text=title, style="Section.TLabelframe", padding=10)
        self.frame.pack(fill="x", pady=(10, 0))
        self.running = False
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.captures: dict[str, Any] = {"laptop": None, "phone": None}
        self.photos: dict[str, Any] = {}
        self.status_preview_var = self._stringvar("Preview stopped.")
        self.cv2_module: Any | None = None

        controls = ttk.Frame(self.frame, style="Panel.TFrame")
        controls.pack(fill="x", pady=(0, 8))
        self.toggle_button = ttk.Button(controls, text="Preview Cameras", command=self.toggle)
        self.toggle_button.pack(side="left")
        ttk.Label(controls, textvariable=self.status_preview_var, style="Muted.TLabel").pack(side="left", padx=(10, 0))

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
            canvas = self._canvas(pane)
            canvas.pack(anchor="w")
            self.canvases[key] = canvas
            self._draw_placeholder(key, "Preview stopped")

        self.refresh_labels()

    def _stringvar(self, value: str) -> Any:
        import tkinter as tk

        return tk.StringVar(value=value)

    def _canvas(self, parent: Any) -> Any:
        import tkinter as tk

        return tk.Canvas(
            parent,
            width=320,
            height=240,
            bg="#111827",
            highlightthickness=1,
            highlightbackground=self.colors["border"],
        )

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

    def _draw_placeholder(self, key: str, text: str) -> None:
        canvas = self.canvases[key]
        canvas.delete("all")
        canvas.create_rectangle(0, 0, 320, 240, fill="#111827", outline="")
        canvas.create_text(160, 120, text=text, fill="#9ca3af", width=290)

    def refresh_labels(self) -> None:
        indices = self._camera_indices()
        self.camera_labels["laptop"].set(f"Laptop camera (index {indices['laptop']})")
        self.camera_labels["phone"].set(f"Phone camera (index {indices['phone']})")

    def _render_frame(self, key: str, frame_rgb: Any) -> None:
        if self.cv2_module is None:
            self._draw_placeholder(key, "Preview unavailable")
            return
        cv2_mod = self.cv2_module
        ok, encoded = cv2_mod.imencode(".png", cv2_mod.cvtColor(frame_rgb, cv2_mod.COLOR_RGB2BGR))
        if not ok:
            self._draw_placeholder(key, "Frame encode failed")
            return
        data = base64.b64encode(encoded.tobytes()).decode("ascii")
        import tkinter as tk

        photo = tk.PhotoImage(data=data)

        canvas = self.canvases[key]
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=photo)
        self.photos[key] = photo

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

    def start(self) -> None:
        if self.running:
            return
        self.refresh_labels()
        if not self.cv2_probe_ok:
            self.status_preview_var.set("OpenCV unavailable for this macOS/Python.")
            self._draw_placeholder("laptop", "OpenCV unavailable")
            self._draw_placeholder("phone", "OpenCV unavailable")
            reason = summarize_probe_error(self.cv2_probe_error) if self.cv2_probe_error else "incompatible module"
            self.append_log(f"Camera preview disabled: {reason}")
            return

        if self.cv2_module is None:
            try:
                import cv2 as cv2_loaded  # type: ignore[import-not-found]
            except Exception as exc:
                self.status_preview_var.set("OpenCV import failed.")
                self._draw_placeholder("laptop", "OpenCV import failed")
                self._draw_placeholder("phone", "OpenCV import failed")
                self.append_log(f"Camera preview unavailable: {exc}")
                return
            self.cv2_module = cv2_loaded

        cv2_mod = self.cv2_module
        if cv2_mod is None:
            self.status_preview_var.set("OpenCV unavailable.")
            self._draw_placeholder("laptop", "OpenCV unavailable")
            self._draw_placeholder("phone", "OpenCV unavailable")
            return

        width, height, fps = self._camera_shape()
        indices = self._camera_indices()
        self.stop_event.clear()
        self.running = True
        self.toggle_button.configure(text="Stop Preview")
        self.status_preview_var.set("Preview running.")
        self.captures = {"laptop": None, "phone": None}

        for key, index in indices.items():
            cap = cv2_mod.VideoCapture(index)
            if cap is not None and cap.isOpened():
                cap.set(cv2_mod.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2_mod.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2_mod.CAP_PROP_FPS, fps)
                self.captures[key] = cap
            else:
                if cap is not None:
                    cap.release()
                self.captures[key] = None
                self._draw_placeholder(key, f"Camera index {index} unavailable")

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
        self._draw_placeholder("laptop", "Preview stopped")
        self._draw_placeholder("phone", "Preview stopped")

    def toggle(self) -> None:
        if self.running:
            self.stop()
        else:
            self.start()

    def close(self) -> None:
        self.stop()
