# Backward-compatibility shim: all symbols have been merged into gui_theme.
# This file may be removed in a future cleanup; import from gui_theme directly.
from .gui_theme import BASE_TOKENS, build_theme_colors, normalize_theme_mode  # noqa: F401
