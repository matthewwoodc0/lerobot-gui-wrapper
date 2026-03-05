from __future__ import annotations

import sys
from typing import Any

# Tk 9.0+ on macOS sends <TouchpadScroll> (event type 39) for trackpad gestures
# instead of <MouseWheel>. The delta is a 16.16 fixed-point value (divide by
# 65536 to get scroll lines). We keep a small per-widget remainder so momentum
# scrolling stays smooth without leaking leftover fractions between widgets or
# direction changes.
_touchpad_accum: dict[str, float] = {}
_touchpad_last_sign: dict[str, int] = {}

TOUCHPAD_SCROLL_SEQ = "<TouchpadScroll>"


def wheel_units(event: Any) -> int:
    # Tk 9.0 TouchpadScroll events (type 39) encode delta as 16.16 fixed-point.
    event_type = str(getattr(event, "type", ""))
    if event_type in ("39", "TouchpadScroll"):
        try:
            raw = int(getattr(event, "delta", 0))
        except (TypeError, ValueError):
            return 0
        if raw == 0:
            return 0
        widget_key = str(getattr(event, "widget", ""))
        direction = 1 if raw > 0 else -1
        if _touchpad_last_sign.get(widget_key) not in (None, direction):
            _touchpad_accum[widget_key] = 0.0
        _touchpad_last_sign[widget_key] = direction
        lines = raw / 65536.0 * 0.35
        _touchpad_accum[widget_key] = _touchpad_accum.get(widget_key, 0.0) + lines
        units = int(_touchpad_accum[widget_key])
        if units != 0:
            _touchpad_accum[widget_key] -= units
            if abs(_touchpad_accum[widget_key]) < 1e-9:
                _touchpad_accum.pop(widget_key, None)
                _touchpad_last_sign.pop(widget_key, None)
        return units

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
    # On macOS, Tk's Cocoa backend reports event.delta with the opposite sign
    # convention to Windows/Linux.  Positive delta means "scroll toward end of
    # document" (finger dragged upward with natural scrolling), whereas on
    # Windows/Linux a positive delta means "scroll toward the start".
    _macos = sys.platform == "darwin"
    if abs(delta) >= 120:
        units = int(delta / 120) if _macos else int(-delta / 120)
        if units != 0:
            return units
    return (1 if delta > 0 else -1) if _macos else (-1 if delta > 0 else 1)


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
        # Widget is at a scroll edge or can't scroll — propagate to the nearest
        # Canvas ancestor.  On macOS, ttk class bindings may intercept the event
        # before bind_all fires, so this is the only reliable propagation path.
        # On Linux the same logic is a useful direct fallback.
        parent_canvas = _find_parent_canvas(widget)
        if parent_canvas is not None:
            scroll_widget_yview(parent_canvas, units)
            return "break"
        return None

    widget.bind("<MouseWheel>", on_wheel, add="+")
    widget.bind("<Button-4>", on_wheel, add="+")
    widget.bind("<Button-5>", on_wheel, add="+")
    try:
        widget.bind(TOUCHPAD_SCROLL_SEQ, on_wheel, add="+")
    except Exception:
        pass


def bind_canvas_scroll_recursive(canvas: Any, widget: Any) -> None:
    """Recursively bind <MouseWheel> on non-scrollable descendants to scroll *canvas*.

    Widget-level bindings run before class bindings and bind_all, ensuring scroll
    events reach the canvas on all platforms.  On macOS, ttk class bindings can
    consume events before bind_all fires; on Linux the per-widget binding prevents
    double-scrolling that would otherwise occur via bind_all.
    """
    try:
        cls = str(widget.winfo_class()).lower()
    except Exception:
        cls = ""

    # Leave widgets that have their own meaningful scroll behaviour alone.
    # bind_yview_wheel_scroll handles edge-propagation for Treeview / Text / Listbox.
    if cls not in ("treeview", "text", "listbox"):
        def _fwd(event: Any, _c: Any = canvas) -> str | None:
            units = wheel_units(event)
            if not units:
                return None
            scroll_widget_yview(_c, units)
            # Always consume the event: we have handled it (or the canvas is at an
            # edge and there is nothing higher to scroll).  Returning "break" prevents
            # class bindings from re-consuming it — essential on macOS, and also
            # prevents double-scrolling via bind_all on Linux.
            return "break"

        try:
            widget.bind("<MouseWheel>", _fwd, add="+")
            widget.bind("<Button-4>", _fwd, add="+")
            widget.bind("<Button-5>", _fwd, add="+")
        except Exception:
            pass
        try:
            widget.bind(TOUCHPAD_SCROLL_SEQ, _fwd, add="+")
        except Exception:
            pass

    try:
        children = widget.winfo_children()
    except Exception:
        return

    for child in children:
        bind_canvas_scroll_recursive(canvas, child)
