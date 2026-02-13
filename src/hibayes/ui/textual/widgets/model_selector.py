"""Model selector bar and dashboard container."""

from __future__ import annotations

import re
from typing import Any

from textual.containers import HorizontalScroll, Vertical
from textual.widgets import Button, ContentSwitcher

from .model_section import ModelSection


def _safe_id(name: str) -> str:
    """Convert a model name to a valid Textual widget ID."""
    return re.sub(r"[^a-zA-Z0-9_-]", "-", name)


class ModelSelectorBar(HorizontalScroll):
    """Row of buttons, one per model + Results. Active button highlighted.

    Uses ``HorizontalScroll`` so the bar scrolls when there are more
    buttons than the window width can display.
    """

    DEFAULT_CSS = """
    ModelSelectorBar {
        height: auto;
        max-height: 5;
        background: $surface;
        border-bottom: solid $primary-background;
        scrollbar-size-horizontal: 1;
    }
    ModelSelectorBar > Button {
        margin: 0 1;
        min-width: 16;
        width: auto;
        height: 3;
    }
    ModelSelectorBar > Button.-active {
        background: $primary;
        color: $text;
        text-style: bold;
    }
    """

    def highlight(self, button_id: str) -> None:
        """Highlight the button with *button_id* and dim all others."""
        for btn in self.query(Button):
            if btn.id == button_id:
                btn.add_class("-active")
                btn.scroll_visible()
            else:
                btn.remove_class("-active")


class ModelDashboard(Vertical):
    """Top-level dashboard: selector bar + content switcher."""

    DEFAULT_CSS = """
    ModelDashboard {
        height: 1fr;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._section_order: list[str] = []  # model names in insertion order

    def compose(self):
        yield ModelSelectorBar(id="model-selector-bar")
        yield ContentSwitcher(id="model-content-switcher")

    # -- public API -----------------------------------------------------------

    def add_model_section(self, model_name: str) -> ModelSection:
        """Create a section + button for *model_name*. Idempotent."""
        sid = _safe_id(model_name)
        switcher = self.query_one(ContentSwitcher)

        # Already exists?
        try:
            existing = switcher.query_one(f"#{sid}", ModelSection)
            return existing
        except Exception:
            pass

        section = ModelSection(model_name, id=sid)
        switcher.mount(section)

        bar = self.query_one(ModelSelectorBar)
        btn = Button(model_name, id=f"btn-{sid}")
        bar.mount(btn)

        self._section_order.append(model_name)
        return section

    def switch_to(self, model_name: str) -> None:
        """Show the section for *model_name* and highlight its button."""
        sid = _safe_id(model_name)
        switcher = self.query_one(ContentSwitcher)
        switcher.current = sid
        bar = self.query_one(ModelSelectorBar)
        bar.highlight(f"btn-{sid}")

    def get_section(self, model_name: str) -> ModelSection:
        """Look up the section widget for *model_name*."""
        sid = _safe_id(model_name)
        switcher = self.query_one(ContentSwitcher)
        return switcher.query_one(f"#{sid}")

    def switch_relative(self, delta: int) -> None:
        """Move *delta* positions (+1 = next, -1 = prev) in the section order."""
        if not self._section_order:
            return
        switcher = self.query_one(ContentSwitcher)
        current = switcher.current
        # Find current index
        current_idx = 0
        for i, name in enumerate(self._section_order):
            if _safe_id(name) == current:
                current_idx = i
                break
        new_idx = (current_idx + delta) % len(self._section_order)
        self.switch_to(self._section_order[new_idx])

    # -- event handling -------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Switch sections when a selector button is clicked."""
        btn_id = event.button.id or ""
        if btn_id.startswith("btn-"):
            sid = btn_id[4:]  # strip "btn-" prefix
            switcher = self.query_one(ContentSwitcher)
            switcher.current = sid
            bar = self.query_one(ModelSelectorBar)
            bar.highlight(btn_id)
            event.stop()
