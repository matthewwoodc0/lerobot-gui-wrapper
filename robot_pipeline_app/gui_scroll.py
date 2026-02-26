from __future__ import annotations

import sys
from typing import Any


def wheel_units(event: Any) -> int:
    if getattr(event, "num", None) == 4:
        return -1
    if getattr(event, "num", None) == 5:
        return 1
    try:
        delta = float(getattr(event, "delta", 0.0))
    except (TypeError, ValueError):
        return 0
    if delta == 0:
        return 0
    if abs(delta) >= 120:
        units = int(-delta / 120)
        if units != 0:
            return units
    return -1 if delta > 0 else 1


def widget_yview(widget: Any) -> tuple[float, float] | None:
    yview = getattr(widget, "yview", None)
    if not callable(yview):
        return None
    try:
        value = yview()
    except Exception:
        return None
    if not isinstance(value, (tuple, list)) or len(value) < 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def scroll_widget_yview(widget: Any, units: int) -> bool:
    if units == 0:
        return False
    before = widget_yview(widget)
    if before is None:
        return False
    try:
        widget.yview_scroll(units, "units")
    except Exception:
        return False
    after = widget_yview(widget)
    if after is None:
        return False
    return abs(after[0] - before[0]) > 1e-9 or abs(after[1] - before[1]) > 1e-9


def at_scroll_edge(widget: Any, units: int) -> bool:
    view = widget_yview(widget)
    if view is None:
        return False
    epsilon = 1e-6
    if units < 0:
        return view[0] <= epsilon
    if units > 0:
        return view[1] >= (1.0 - epsilon)
    return False


def _find_parent_canvas(widget: Any) -> Any | None:
    """Walk up the widget hierarchy to find the nearest Canvas ancestor."""
    current = widget
    for _ in range(20):
        try:
            parent_name = current.winfo_parent()
        except Exception:
            break
        if not parent_name:
            break
        try:
            parent = current.nametowidget(parent_name)
        except Exception:
            break
        try:
            cls = str(parent.winfo_class()).lower()
        except Exception:
            cls = ""
        if cls == "canvas":
            return parent
        current = parent
    return None


def bind_yview_wheel_scroll(widget: Any) -> None:
    def on_wheel(event: Any) -> str | None:
        units = wheel_units(event)
        if units == 0:
            return None
        if scroll_widget_yview(widget, units):
            return "break"
        # On macOS, ttk class bindings often return "break" before bind_all fires,
        # so the global canvas fallback never gets a chance to run.  Scroll the
        # nearest Canvas ancestor directly and consume the event ourselves.
        if sys.platform == "darwin":
            parent_canvas = _find_parent_canvas(widget)
            if parent_canvas is not None:
                if scroll_widget_yview(parent_canvas, units) or scroll_widget_yview(parent_canvas, -units):
                    return "break"
            return None
        return None

    widget.bind("<MouseWheel>", on_wheel, add="+")
    widget.bind("<Button-4>", on_wheel, add="+")
    widget.bind("<Button-5>", on_wheel, add="+")


def bind_canvas_scroll_recursive(canvas: Any, widget: Any) -> None:
    """Recursively bind <MouseWheel> on non-scrollable descendants to scroll *canvas*.

    On macOS, ttk widget class bindings for <MouseWheel> return "break" before
    bind_all fires, so the global scroll handler never runs when the cursor sits
    over a form field.  Widget-level bindings run *before* class bindings, so
    adding them here ensures scroll events always reach the canvas.
    """
    try:
        cls = str(widget.winfo_class()).lower()
    except Exception:
        cls = ""

    # Leave widgets that have their own meaningful scroll behaviour alone.
    # bind_yview_wheel_scroll already handles the macOS edge-propagation for
    # Treeview / Text / Listbox; nested Canvas widgets are their own scrollers.
    if cls not in ("treeview", "text", "listbox", "canvas"):
        def _fwd(event: Any, _c: Any = canvas) -> str:
            units = wheel_units(event)
            if not units:
                return None
            if scroll_widget_yview(_c, units):
                return "break"
            if sys.platform == "darwin" and scroll_widget_yview(_c, -units):
                return "break"
            return None

        try:
            widget.bind("<MouseWheel>", _fwd, add="+")
            widget.bind("<Button-4>", _fwd, add="+")
            widget.bind("<Button-5>", _fwd, add="+")
        except Exception:
            pass

    try:
        children = widget.winfo_children()
    except Exception:
        return

    for child in children:
        bind_canvas_scroll_recursive(canvas, child)
