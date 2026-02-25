from __future__ import annotations

from typing import Any


def fit_window_to_screen(
    *,
    window: Any,
    requested_width: int,
    requested_height: int,
    requested_min_width: int,
    requested_min_height: int,
    margin_x: int = 24,
    margin_y: int = 48,
    center: bool = True,
) -> tuple[int, int]:
    screen_w = int(window.winfo_screenwidth() or requested_width)
    screen_h = int(window.winfo_screenheight() or requested_height)

    max_w = max(360, screen_w - margin_x)
    max_h = max(280, screen_h - margin_y)
    final_w = min(requested_width, max_w)
    final_h = min(requested_height, max_h)

    min_w = min(requested_min_width, final_w)
    min_h = min(requested_min_height, final_h)
    window.minsize(min_w, min_h)

    if center:
        x = max((screen_w - final_w) // 2, 8)
        y = max((screen_h - final_h) // 2, 8)
        window.geometry(f"{final_w}x{final_h}+{x}+{y}")
    else:
        window.geometry(f"{final_w}x{final_h}")
    return final_w, final_h
