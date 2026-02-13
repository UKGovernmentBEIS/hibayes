"""Clickable expand/collapse toggle widget."""

from __future__ import annotations

from textual.message import Message
from textual.widgets import Static


class Toggle(Static):
    """A clickable toggle that posts a ``Toggled`` message when clicked."""

    class Toggled(Message):
        """Emitted when the toggle state changes."""

        def __init__(self, toggle: Toggle) -> None:
            self.toggle = toggle
            super().__init__()

    DEFAULT_CSS = """
    Toggle {
        width: 3;
        min-width: 3;
        content-align: center middle;
    }
    """

    on_symbol: str = "\u25bc"   # ▼
    off_symbol: str = "\u25b6"  # ▶

    def __init__(
        self,
        *,
        expanded: bool = False,
        on_symbol: str | None = None,
        off_symbol: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        if on_symbol is not None:
            self.on_symbol = on_symbol
        if off_symbol is not None:
            self.off_symbol = off_symbol
        self._expanded = expanded
        super().__init__(
            self.on_symbol if expanded else self.off_symbol,
            id=id,
            classes=classes,
        )

    @property
    def expanded(self) -> bool:
        return self._expanded

    @expanded.setter
    def expanded(self, value: bool) -> None:
        self._expanded = value
        self.update(self.on_symbol if value else self.off_symbol)

    def on_click(self) -> None:
        self.expanded = not self._expanded
        self.post_message(self.Toggled(self))
