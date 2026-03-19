"""App header widget."""

from __future__ import annotations

from textual.reactive import reactive
from textual.widgets import Static


class AppHeader(Static):
    """Docked-top header showing 'HiBayes - {phase}'."""

    DEFAULT_CSS = """
    AppHeader {
        dock: top;
        height: 1;
        background: $primary;
        color: $text;
        text-style: bold;
        content-align: center middle;
        padding: 0 1;
    }
    """

    title: reactive[str] = reactive("HiBayes")

    def render(self) -> str:
        return self.title
