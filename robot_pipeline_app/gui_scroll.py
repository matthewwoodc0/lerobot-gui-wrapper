from __future__ import annotations

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


def bind_yview_wheel_scroll(widget: Any) -> None:
    def on_wheel(event: Any) -> str | None:
        units = wheel_units(event)
        if units == 0:
            return None
        if scroll_widget_yview(widget, units):
            return "break"
        return None

    widget.bind("<MouseWheel>", on_wheel, add="+")
    widget.bind("<Button-4>", on_wheel, add="+")
    widget.bind("<Button-5>", on_wheel, add="+")
