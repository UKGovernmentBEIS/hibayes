"""Stats key-value display widget."""

from __future__ import annotations

from typing import Any, Dict

from textual.widgets import Static


def _escape_markup(text: str) -> str:
    """Escape square brackets so Rich markup doesn't interpret them."""
    return text.replace("[", r"\[")


class StatsPanel(Static):
    """Renders key-value statistics as a compact list."""

    DEFAULT_CSS = """
    StatsPanel {
        border: solid $primary-background;
        padding: 0 1;
        height: auto;
        min-height: 5;
    }
    """

    def __init__(
        self,
        *,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(id=id, classes=classes)
        self._stats: Dict[str, Any] = {}

    def update_stat(self, key: str, value: Any) -> None:
        """Update a single statistic and re-render."""
        self._stats[key] = value
        self._refresh_content()

    def update_stats(self, stats: Dict[str, Any]) -> None:
        """Update multiple statistics and re-render."""
        self._stats.update(stats)
        self._refresh_content()

    def _refresh_content(self) -> None:
        lines = []
        for k, v in self._stats.items():
            lines.append(f"[cyan]{_escape_markup(str(k))}[/]: [green]{_escape_markup(str(v))}[/]")
        self.update("\n".join(lines))
