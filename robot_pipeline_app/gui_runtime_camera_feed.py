from __future__ import annotations

import base64
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from .camera_schema import CameraSpec, resolve_camera_schema
from .camera_state import DEFAULT_LIVE_PREVIEW_FPS_CAP, live_preview_interval_ms, positive_int
from .gui_async import UiBackgroundJobs
from .probes import summarize_probe_error

_CARD_CANVAS_WIDTH = 300
_CARD_CANVAS_HEIGHT = 220


def resolve_runtime_feed_specs(config: dict[str, Any]) -> tuple[list[CameraSpec], list[str], list[str]]:
    resolution = resolve_camera_schema(config)
    return list(resolution.specs), list(resolution.warnings), list(resolution.errors)


class RuntimeCameraFeed:
    def __init__(
        self,
        *,
        root: Any,
        parent: Any,
        config: dict[str, Any],
        colors: dict[str, str],
        cv2_probe_ok: bool,
        cv2_probe_error: str,
        append_log: Callable[[str], None],
        background_jobs: UiBackgroundJobs | None = None,
        title: str = "Live Camera Feed",
    ) -> None:
        self.root = root
        self.config = config
        self.colors = colors
        self.cv2_probe_ok = cv2_probe_ok
        self.cv2_probe_error = cv2_probe_error
        self.append_log = append_log
        self.background_jobs = background_jobs

        self._cards: dict[str, dict[str, Any]] = {}
        self._camera_specs: list[CameraSpec] = []
        self._refresh_job: str | None = None
        self._refresh_inflight = False
        self._active = False
        self._schema_notice: tuple[tuple[str, ...], tuple[str, ...]] | None = None
        self._capture_lock = threading.Lock()
        self._capture_pool: dict[int | str, Any] = {}
        self._capture_retry_after: dict[int | str, float] = {}

        self._build_shell(parent, title)
        self.refresh_schema(log_changes=False)

    def _build_shell(self, parent: Any, title: str) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.frame = ttk.LabelFrame(parent, text=title, style="Section.TLabelframe", padding=10)
        self.frame.columnconfigure(0, weight=1)

        self.status_var = tk.StringVar(value="Live feed idle.")
        self.summary_var = tk.StringVar(value="")
        self.empty_var = tk.StringVar(value="No runtime cameras configured.")

        self._create_info_label(self.frame, textvariable=self.status_var).grid(row=0, column=0, sticky="ew")
        self._create_info_label(
            self.frame,
            textvariable=self.summary_var,
            foreground=self.colors.get("text", "#eeeeee"),
            font=(self.colors.get("font_ui", "TkDefaultFont"), 9, "bold"),
        ).grid(row=1, column=0, sticky="ew", pady=(4, 0))

        self._cards_frame = tk.Frame(self.frame, bg=self.colors.get("panel", "#111111"))
        self._cards_frame.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        self.frame.rowconfigure(2, weight=1)
        self._empty_label = self._create_empty_label()
        self._empty_label.grid(row=0, column=0, sticky="w")

    def _create_info_label(
        self,
        parent: Any,
        *,
        textvariable: Any,
        foreground: str | None = None,
        font: tuple[str, int] | tuple[str, int, str] | None = None,
    ) -> Any:
        import tkinter as tk

        return tk.Label(
            parent,
            textvariable=textvariable,
            bg=self.colors.get("panel", "#111111"),
            fg=foreground or self.colors.get("muted", "#777777"),
            font=font or (self.colors.get("font_ui", "TkDefaultFont"), 9),
            anchor="w",
            justify="left",
            wraplength=420,
        )

    def _create_empty_label(self) -> Any:
        return self._create_info_label(self._cards_frame, textvariable=self.empty_var)

    def _card_surface(self) -> str:
        return self.colors.get("surface", "#1a1a1a")

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

    def _preview_interval_ms(self) -> int:
        configured_fps = positive_int(self.config.get("camera_fps"), DEFAULT_LIVE_PREVIEW_FPS_CAP)
        return live_preview_interval_ms(min(configured_fps, DEFAULT_LIVE_PREVIEW_FPS_CAP))

    def _ensure_cv2_module(self) -> bool:
        if getattr(self, "cv2_module", None) is not None:
            return True
        try:
            import cv2 as cv2_loaded  # type: ignore[import-not-found]
        except Exception as exc:
            self.status_var.set(f"Live feed unavailable: {exc}")
            return False
        self.cv2_module = cv2_loaded
        return True

    def _camera_card_columns(self) -> int:
        return 1 if len(self._camera_specs) <= 1 else 2

    def _camera_summary(self) -> str:
        if not self._camera_specs:
            return "Runtime cameras: (none)"
        parts = [f"{spec.name}={spec.source}" for spec in self._camera_specs]
        return "Runtime cameras: " + "  ".join(parts)

    def _clear_cards(self) -> None:
        for child in list(self._cards_frame.winfo_children()):
            child.destroy()
        self._cards.clear()

    def _build_cards(self) -> None:
        self._clear_cards()
        if not self._camera_specs:
            self._empty_label = self._create_empty_label()
            self._empty_label.grid(row=0, column=0, sticky="w")
            return

        columns = self._camera_card_columns()
        for column in range(columns):
            self._cards_frame.columnconfigure(column, weight=1)

        for idx, spec in enumerate(self._camera_specs):
            row = idx // columns
            column = idx % columns
            self._cards[spec.name] = self._build_card(spec, row=row, column=column, columns=columns)

    def _build_card(self, spec: CameraSpec, *, row: int, column: int, columns: int) -> dict[str, Any]:
        import tkinter as tk

        card = tk.Frame(
            self._cards_frame,
            bg=self._card_surface(),
            highlightthickness=1,
            highlightbackground=self.colors.get("border", "#2d2d2d"),
            padx=8,
            pady=8,
        )
        card.grid(row=row, column=column, sticky="nsew", padx=(0, 10 if column < columns - 1 else 0), pady=(0, 10))

        tk.Label(
            card,
            text=spec.name,
            bg=self._card_surface(),
            fg=self.colors.get("accent", "#f0a500"),
            font=(self.colors.get("font_ui", "TkDefaultFont"), 10, "bold"),
            anchor="w",
        ).pack(fill="x")

        meta_var = tk.StringVar(value=self._card_meta_text(spec))
        tk.Label(
            card,
            textvariable=meta_var,
            bg=self._card_surface(),
            fg=self.colors.get("muted", "#777777"),
            font=(self.colors.get("font_ui", "TkDefaultFont"), 8),
            anchor="w",
            justify="left",
            wraplength=_CARD_CANVAS_WIDTH,
        ).pack(fill="x", pady=(2, 0))

        canvas = tk.Canvas(
            card,
            width=_CARD_CANVAS_WIDTH,
            height=_CARD_CANVAS_HEIGHT,
            bg=self.colors.get("surface_alt", "#252525"),
            highlightthickness=1,
            highlightbackground=self.colors.get("border", "#2d2d2d"),
            bd=0,
        )
        canvas.pack(fill="x", pady=(6, 4))

        state_var = tk.StringVar(value="Waiting for frame...")
        tk.Label(
            card,
            textvariable=state_var,
            bg=self._card_surface(),
            fg=self.colors.get("muted", "#777777"),
            font=(self.colors.get("font_ui", "TkDefaultFont"), 8),
            anchor="w",
            justify="left",
            wraplength=_CARD_CANVAS_WIDTH,
        ).pack(fill="x")

        payload = {
            "canvas": canvas,
            "state_var": state_var,
            "meta_var": meta_var,
            "photo": None,
        }
        self._draw_placeholder(canvas, "Waiting for frame...")
        return payload

    def _card_meta_text(self, spec: CameraSpec) -> str:
        return f"Source: {spec.source}  |  Target: {spec.width}x{spec.height} @ {spec.fps} FPS"

    def refresh_schema(self, *, log_changes: bool = True) -> None:
        specs, warnings, errors = resolve_runtime_feed_specs(self.config)
        self._camera_specs = specs
        self.summary_var.set(self._camera_summary())
        self.empty_var.set("No runtime cameras configured.")
        self._build_cards()

        notice_key = (tuple(warnings), tuple(errors))
        if log_changes and notice_key != self._schema_notice:
            for warning in warnings:
                self.append_log(f"Record live feed warning: {warning}")
            for error in errors:
                self.append_log(f"Record live feed error: {error}")
        self._schema_notice = notice_key

        if errors:
            self.status_var.set("Live feed loaded with camera schema errors. See log for details.")
        elif warnings:
            self.status_var.set("Live feed loaded with camera schema warnings. See log for details.")
        elif specs:
            self.status_var.set("Live feed ready.")
        else:
            self.status_var.set("Live feed unavailable: no runtime cameras configured.")

    def _draw_placeholder(self, canvas: Any, text: str) -> None:
        width = int(canvas["width"])
        height = int(canvas["height"])
        canvas.delete("all")
        canvas.create_rectangle(0, 0, width, height, fill=self.colors.get("surface_alt", "#252525"), outline="")
        canvas.create_text(
            width // 2,
            height // 2,
            text=text,
            fill=self.colors.get("muted", "#777777"),
            width=max(width - 24, 80),
        )

    def _render_frame(self, spec: CameraSpec, frame_bgr: Any) -> None:
        if getattr(self, "cv2_module", None) is None:
            return
        card = self._cards.get(spec.name)
        if card is None:
            return
        canvas = card.get("canvas")
        if canvas is None:
            return
        cv2_mod = self.cv2_module
        target_width = int(canvas["width"])
        target_height = int(canvas["height"])
        rendered_frame = self._fit_frame_to_canvas(frame_bgr, target_width, target_height)
        if rendered_frame is None:
            return
        photo = self._photo_image_from_frame(rendered_frame)
        if photo is None:
            return
        src_h, src_w = frame_bgr.shape[:2]

        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=photo)
        card["photo"] = photo
        card["state_var"].set(f"Live frame: {src_w}x{src_h}")

    def _fit_frame_to_canvas(self, frame_bgr: Any, target_width: int, target_height: int) -> Any | None:
        if getattr(self, "cv2_module", None) is None:
            return None
        cv2_mod = self.cv2_module
        src_h, src_w = frame_bgr.shape[:2]
        if src_w <= 0 or src_h <= 0:
            return None
        scale = min(target_width / float(src_w), target_height / float(src_h))
        resized_w = max(1, int(round(float(src_w) * scale)))
        resized_h = max(1, int(round(float(src_h) * scale)))
        interpolation = cv2_mod.INTER_AREA if scale < 1.0 else cv2_mod.INTER_LINEAR
        resized = cv2_mod.resize(frame_bgr, (resized_w, resized_h), interpolation=interpolation)
        if resized_w == target_width and resized_h == target_height:
            return resized
        top = max(0, (target_height - resized_h) // 2)
        bottom = max(0, target_height - resized_h - top)
        left = max(0, (target_width - resized_w) // 2)
        right = max(0, target_width - resized_w - left)
        return cv2_mod.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2_mod.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

    def _photo_image_from_frame(self, frame: Any) -> Any | None:
        if getattr(self, "cv2_module", None) is None:
            return None
        cv2_mod = self.cv2_module
        ok, encoded = cv2_mod.imencode(".ppm", frame)
        if not ok:
            ok, encoded = cv2_mod.imencode(".png", frame)
        if not ok:
            return None
        data = base64.b64encode(encoded.tobytes()).decode("ascii")
        import tkinter as tk

        try:
            return tk.PhotoImage(data=data)
        except Exception:
            ok_png, encoded_png = cv2_mod.imencode(".png", frame)
            if not ok_png:
                return None
            try:
                return tk.PhotoImage(data=base64.b64encode(encoded_png.tobytes()).decode("ascii"))
            except Exception:
                return None

    def _release_capture_locked(self, source: int | str) -> None:
        capture = self._capture_pool.pop(source, None)
        self._capture_retry_after.pop(source, None)
        if capture is None:
            return
        try:
            capture.release()
        except Exception:
            pass

    def _release_all_captures(self) -> None:
        with self._capture_lock:
            for source in list(self._capture_pool.keys()):
                self._release_capture_locked(source)

    def _sync_capture_pool_locked(self) -> None:
        active_sources = {spec.source for spec in self._camera_specs}
        for source in list(self._capture_pool.keys()):
            if source not in active_sources:
                self._release_capture_locked(source)
        for source in list(self._capture_retry_after.keys()):
            if source not in active_sources:
                self._capture_retry_after.pop(source, None)

    def _open_capture(self, spec: CameraSpec) -> Any | None:
        if getattr(self, "cv2_module", None) is None:
            return None
        result: list[Any] = [None]

        def _try_open() -> None:
            with self._suppress_stderr():
                result[0] = self.cv2_module.VideoCapture(spec.source)

        thread = threading.Thread(target=_try_open, daemon=True)
        thread.start()
        thread.join(timeout=2.0)
        if thread.is_alive():
            return None
        capture = result[0]
        if capture is None or not capture.isOpened():
            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    pass
            return None
        try:
            with self._suppress_stderr():
                capture.set(self.cv2_module.CAP_PROP_FRAME_WIDTH, float(spec.width))
                capture.set(self.cv2_module.CAP_PROP_FRAME_HEIGHT, float(spec.height))
                capture.set(self.cv2_module.CAP_PROP_FPS, float(spec.fps))
        except Exception:
            pass
        return capture

    def _capture_for_spec_locked(self, spec: CameraSpec) -> Any | None:
        retry_after = self._capture_retry_after.get(spec.source, 0.0)
        if time.monotonic() < retry_after:
            return None

        capture = self._capture_pool.get(spec.source)
        if capture is not None and capture.isOpened():
            return capture
        if capture is not None:
            self._release_capture_locked(spec.source)

        capture = self._open_capture(spec)
        if capture is None:
            self._capture_retry_after[spec.source] = time.monotonic() + 1.0
            return None

        self._capture_pool[spec.source] = capture
        self._capture_retry_after.pop(spec.source, None)
        return capture

    def _capture_frame(self, spec: CameraSpec) -> Any | None:
        if not self._ensure_cv2_module():
            return None
        with self._capture_lock:
            self._sync_capture_pool_locked()
            capture = self._capture_for_spec_locked(spec)
            if capture is None:
                return None
            try:
                with self._suppress_stderr():
                    ok, frame = capture.read()
            except Exception:
                ok, frame = False, None
            if ok and frame is not None:
                return frame
            self._release_capture_locked(spec.source)
            self._capture_retry_after[spec.source] = time.monotonic() + 1.0
            return None

    def _capture_all_frames(self) -> dict[str, Any | None]:
        return {spec.name: self._capture_frame(spec) for spec in self._camera_specs}

    def _set_unavailable_state(self, detail: str) -> None:
        self.status_var.set(detail)
        for card in self._cards.values():
            canvas = card.get("canvas")
            if canvas is not None:
                self._draw_placeholder(canvas, "Unavailable")
            card["state_var"].set(detail)

    def _apply_frames(self, frames: dict[str, Any | None]) -> None:
        refreshed = 0
        for spec in self._camera_specs:
            frame = frames.get(spec.name)
            card = self._cards.get(spec.name)
            if card is None:
                continue
            canvas = card.get("canvas")
            if frame is None:
                if canvas is not None:
                    self._draw_placeholder(canvas, "Unavailable")
                card["state_var"].set(f"Unable to open source {spec.source}")
                continue
            self._render_frame(spec, frame)
            refreshed += 1
        timestamp = time.strftime("%H:%M:%S")
        self.status_var.set(f"Live feed {timestamp} ({refreshed}/{len(self._camera_specs)} cameras)")

    def _cancel_refresh_job(self) -> None:
        if self._refresh_job is None:
            return
        try:
            self.root.after_cancel(self._refresh_job)
        except Exception:
            pass
        self._refresh_job = None

    def _schedule_refresh(self, delay_ms: int | None = None) -> None:
        self._cancel_refresh_job()
        if not self._active:
            return
        delay = self._preview_interval_ms() if delay_ms is None else max(0, int(delay_ms))
        self._refresh_job = self.root.after(delay, self._refresh_tick)

    def _refresh_tick(self) -> None:
        self._refresh_job = None
        if not self._active:
            return
        if self._refresh_inflight:
            self._schedule_refresh()
            return
        if not self._camera_specs:
            self.status_var.set("Live feed unavailable: no runtime cameras configured.")
            return
        if not self.cv2_probe_ok:
            detail = summarize_probe_error(self.cv2_probe_error) if self.cv2_probe_error else "OpenCV unavailable"
            self._set_unavailable_state(f"Live feed unavailable: {detail}")
            return

        self._refresh_inflight = True

        def _complete() -> None:
            self._refresh_inflight = False
            self._schedule_refresh()

        if self.background_jobs is None:
            try:
                frames = self._capture_all_frames()
                self._apply_frames(frames)
            finally:
                _complete()
            return

        self.background_jobs.submit(
            "record-runtime-camera-feed",
            self._capture_all_frames,
            on_success=self._apply_frames,
            on_error=lambda exc: self._set_unavailable_state(f"Live feed refresh failed: {exc}"),
            on_complete=lambda _stale: _complete(),
        )

    def start(self) -> None:
        self.refresh_schema()
        self._active = True
        if not self._camera_specs:
            return
        if not self.cv2_probe_ok:
            detail = summarize_probe_error(self.cv2_probe_error) if self.cv2_probe_error else "OpenCV unavailable"
            self._set_unavailable_state(f"Live feed unavailable: {detail}")
            return
        self._schedule_refresh(delay_ms=0)

    def stop(self) -> None:
        self._active = False
        self._cancel_refresh_job()
        self._refresh_inflight = False
        self._release_all_captures()

    def close(self) -> None:
        self.stop()

    def apply_theme(self, colors: dict[str, str]) -> None:
        self.colors = colors
        self.frame.configure(style="Section.TLabelframe")
        try:
            self._cards_frame.configure(bg=self.colors.get("panel", "#111111"))
        except Exception:
            pass
        for card in self._cards.values():
            canvas = card.get("canvas")
            if canvas is not None:
                canvas.configure(
                    bg=self.colors.get("surface_alt", "#252525"),
                    highlightbackground=self.colors.get("border", "#2d2d2d"),
                )
                if card.get("photo") is None:
                    self._draw_placeholder(canvas, "Waiting for frame...")
