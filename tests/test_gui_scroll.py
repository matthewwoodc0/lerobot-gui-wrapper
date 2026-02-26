from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from robot_pipeline_app.gui_scroll import (
    at_scroll_edge,
    bind_canvas_scroll_recursive,
    bind_yview_wheel_scroll,
    scroll_widget_yview,
    wheel_units,
    widget_yview,
)


class _FakeEvent:
    def __init__(self, *, num: int | None = None, delta: object = 0.0) -> None:
        self.num = num
        self.delta = delta


class _FakeWidget:
    def __init__(
        self,
        view: tuple[float, float] = (0.0, 1.0),
        *,
        fail_scroll: bool = False,
        class_name: str = "Frame",
        parent: "_FakeWidget | None" = None,
    ) -> None:
        self._view = view
        self.fail_scroll = fail_scroll
        self._class_name = class_name
        self._parent = parent
        self._children: list[_FakeWidget] = []
        if parent is not None:
            parent._children.append(self)
        self.bindings: dict[str, object] = {}

    def yview(self) -> tuple[float, float]:
        return self._view

    def yview_scroll(self, units: int, mode: str) -> None:
        if self.fail_scroll:
            raise RuntimeError("scroll failed")
        if mode != "units":
            raise AssertionError("unexpected scroll mode")
        if self._view[0] <= 0.0 and self._view[1] >= 1.0:
            return
        top, bottom = self._view
        step = 0.1 * units
        new_top = min(max(top + step, 0.0), 1.0)
        new_bottom = min(max(bottom + step, 0.0), 1.0)
        self._view = (new_top, new_bottom)

    def bind(self, event: str, callback, add: str = "") -> None:
        self.bindings[event] = callback

    def winfo_class(self) -> str:
        return self._class_name

    def winfo_children(self) -> list["_FakeWidget"]:
        return list(self._children)

    def winfo_parent(self) -> str:
        if self._parent is None:
            return ""
        return str(id(self._parent))

    def nametowidget(self, name: str) -> "_FakeWidget":
        current = self
        while current._parent is not None:
            if str(id(current._parent)) == name:
                return current._parent
            current = current._parent
        raise KeyError(name)


class GuiScrollTests(unittest.TestCase):
    def test_wheel_units_supports_button_and_delta_sources(self) -> None:
        # Button-4/5 (X11 Linux) are platform-independent.
        self.assertEqual(wheel_units(_FakeEvent(num=4)), -1)
        self.assertEqual(wheel_units(_FakeEvent(num=5)), 1)
        # Zero / bad delta — always 0.
        self.assertEqual(wheel_units(_FakeEvent(delta=0.0)), 0)
        self.assertEqual(wheel_units(_FakeEvent(delta="oops")), 0)

    def test_wheel_units_linux_direction(self) -> None:
        with patch("robot_pipeline_app.gui_scroll.sys.platform", "linux"):
            # Linux/Windows: positive delta → scroll toward start (-1).
            self.assertEqual(wheel_units(_FakeEvent(delta=120.0)), -1)
            self.assertEqual(wheel_units(_FakeEvent(delta=-240.0)), 2)
            self.assertEqual(wheel_units(_FakeEvent(delta=1.0)), -1)
            self.assertEqual(wheel_units(_FakeEvent(delta=-1.0)), 1)

    def test_wheel_units_macos_direction(self) -> None:
        with patch("robot_pipeline_app.gui_scroll.sys.platform", "darwin"):
            # macOS: positive delta → scroll toward end (+1) — inverted vs Linux.
            self.assertEqual(wheel_units(_FakeEvent(delta=120.0)), 1)
            self.assertEqual(wheel_units(_FakeEvent(delta=-240.0)), -2)
            self.assertEqual(wheel_units(_FakeEvent(delta=1.0)), 1)
            self.assertEqual(wheel_units(_FakeEvent(delta=-1.0)), -1)

    def test_widget_yview_returns_none_for_invalid_shapes(self) -> None:
        class _NoYview:
            pass

        class _BadYview:
            def yview(self):
                return (0.1,)

        class _Raises:
            def yview(self):
                raise RuntimeError("boom")

        self.assertEqual(widget_yview(_FakeWidget((0.25, 0.75))), (0.25, 0.75))
        self.assertIsNone(widget_yview(_NoYview()))
        self.assertIsNone(widget_yview(_BadYview()))
        self.assertIsNone(widget_yview(_Raises()))

    def test_scroll_widget_yview_detects_scroll_progress_and_failures(self) -> None:
        widget = _FakeWidget((0.2, 0.8))
        self.assertTrue(scroll_widget_yview(widget, 1))
        self.assertFalse(scroll_widget_yview(widget, 0))
        self.assertFalse(scroll_widget_yview(_FakeWidget((0.0, 1.0)), -1))
        self.assertFalse(scroll_widget_yview(_FakeWidget((0.1, 0.9), fail_scroll=True), 1))

    def test_at_scroll_edge_detects_top_and_bottom(self) -> None:
        self.assertTrue(at_scroll_edge(_FakeWidget((0.0, 0.5)), -1))
        self.assertFalse(at_scroll_edge(_FakeWidget((0.1, 0.6)), -1))
        self.assertTrue(at_scroll_edge(_FakeWidget((0.4, 1.0)), 1))
        self.assertFalse(at_scroll_edge(_FakeWidget((0.3, 0.9)), 1))
        self.assertFalse(at_scroll_edge(_FakeWidget((0.0, 1.0)), 0))

    def test_bind_yview_wheel_scroll_binds_handlers_and_breaks_on_scroll(self) -> None:
        widget = _FakeWidget((0.2, 0.8))
        bind_yview_wheel_scroll(widget)

        self.assertIn("<MouseWheel>", widget.bindings)
        self.assertIn("<Button-4>", widget.bindings)
        self.assertIn("<Button-5>", widget.bindings)

        on_wheel = widget.bindings["<MouseWheel>"]
        with patch("robot_pipeline_app.gui_scroll.scroll_widget_yview", return_value=True):
            self.assertEqual(on_wheel(_FakeEvent(delta=120.0)), "break")
        # No parent canvas — returns None when widget can't scroll.
        with patch("robot_pipeline_app.gui_scroll.scroll_widget_yview", return_value=False):
            self.assertIsNone(on_wheel(_FakeEvent(delta=120.0)))

    def test_bind_yview_wheel_scroll_propagates_to_parent_canvas_at_edge(self) -> None:
        # Works on all platforms: when a scrollable widget is at its edge the
        # nearest Canvas ancestor is scrolled instead.
        canvas = _FakeWidget((0.2, 0.8), class_name="Canvas")
        widget = _FakeWidget((0.0, 1.0), class_name="Treeview", parent=canvas)
        bind_yview_wheel_scroll(widget)
        on_wheel = widget.bindings["<MouseWheel>"]

        result = on_wheel(_FakeEvent(delta=120.0))

        self.assertEqual(result, "break")
        self.assertNotEqual(canvas.yview(), (0.2, 0.8))

    def test_bind_canvas_scroll_recursive_binds_non_scroll_widgets(self) -> None:
        canvas = _FakeWidget((0.2, 0.8), class_name="Canvas")
        content = _FakeWidget((0.0, 1.0), class_name="TFrame", parent=canvas)
        entry = _FakeWidget((0.0, 1.0), class_name="TEntry", parent=content)
        tree = _FakeWidget((0.2, 0.8), class_name="Treeview", parent=content)
        nested_canvas = _FakeWidget((0.0, 1.0), class_name="Canvas", parent=content)

        bind_canvas_scroll_recursive(canvas, content)

        self.assertIn("<MouseWheel>", content.bindings)
        self.assertIn("<MouseWheel>", entry.bindings)
        self.assertNotIn("<MouseWheel>", tree.bindings)
        self.assertIn("<MouseWheel>", nested_canvas.bindings)


if __name__ == "__main__":
    unittest.main()
