"""App footer widget."""

from __future__ import annotations

from textual.reactive import reactive
from textual.widgets import Static


class AppFooter(Static):
    """Docked-bottom footer with status text."""

    DEFAULT_CSS = """
    AppFooter {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text-muted;
        content-align: center middle;
        padding: 0 1;
    }
    """

    status: reactive[str] = reactive("Press Ctrl+C to interrupt")

    def render(self) -> str:
        return self.status
