"""Scrollable log viewer widget."""

from __future__ import annotations

from textual.containers import ScrollableContainer
from textual.widgets import Static


class LogsPanel(ScrollableContainer):
    """Displays timestamped log entries and auto-scrolls to the bottom."""

    DEFAULT_CSS = """
    LogsPanel {
        height: 8;
        border: solid $primary-background;
        overflow-y: auto;
    }
    LogsPanel > Static {
        height: auto;
        color: $text-muted;
    }
    """

    def append_log(self, entry: str) -> None:
        """Add a new log entry and scroll to bottom."""
        widget = Static(entry, classes="log-entry", markup=False)
        self.mount(widget)
        widget.scroll_visible()
