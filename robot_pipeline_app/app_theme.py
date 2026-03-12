from __future__ import annotations

from typing import Any

_THEME_DARK: dict[str, Any] = {
    "colors": {
        "bg": "#090909",
        "panel": "#121212",
        "surface": "#1b1b1b",
        "surface_alt": "#252525",
        "surface_elevated": "#2a2a2a",
        "header": "#0e0e0e",
        "border": "#303030",
        "border_focus": "#f0a500",
        "text": "#f2f2f2",
        "muted": "#8f8f8f",
        "accent": "#f0a500",
        "accent_dark": "#c88a00",
        "accent_soft": "#4d390e",
        "scrollbar_handle": "#3a3a3a",
        "running": "#f0a500",
        "running_dim": "#7a5200",
        "ready": "#22c55e",
        "error": "#ef4444",
        "success": "#22c55e",
    },
}

_THEME_LIGHT: dict[str, Any] = {
    "colors": {
        "bg": "#f4f5f7",
        "panel": "#ffffff",
        "surface": "#eef1f5",
        "surface_alt": "#e3e8ef",
        "surface_elevated": "#d8e0ea",
        "header": "#e9edf2",
        "border": "#c7d0db",
        "border_focus": "#ca7a00",
        "text": "#12161d",
        "muted": "#526070",
        "accent": "#ca7a00",
        "accent_dark": "#a56200",
        "accent_soft": "#f2dcc1",
        "scrollbar_handle": "#b8c2cf",
        "running": "#ca7a00",
        "running_dim": "#e1b36e",
        "ready": "#1f9d55",
        "error": "#cc3c3c",
        "success": "#1f9d55",
    },
}

# Canonical spacing tiers — referenced by widget code instead of magic numbers.
SPACING_SHELL = 18
SPACING_PANE = 20
SPACING_COMPACT = 14
SPACING_CARD = 12
SPACING_DENSE = 8
SPACING_META = 2

# Canonical border radii.
RADIUS_SHELL = 24
RADIUS_PANE = 20
RADIUS_DIALOG = 22
RADIUS_CARD = 18
RADIUS_NAV = 14
RADIUS_BUTTON = 12
RADIUS_CHIP = 11


def normalize_theme_mode(value: Any) -> str:
    mode = str(value or "dark").strip().lower()
    return "light" if mode == "light" else "dark"


def build_theme_colors(*, ui_font: str, mono_font: str, theme_mode: str = "dark") -> dict[str, str]:
    palette = _THEME_LIGHT if normalize_theme_mode(theme_mode) == "light" else _THEME_DARK
    colors = dict(palette["colors"])
    colors["font_ui"] = ui_font
    colors["font_mono"] = mono_font
    colors["theme_mode"] = normalize_theme_mode(theme_mode)
    return colors
