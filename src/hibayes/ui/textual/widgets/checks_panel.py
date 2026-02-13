"""Interactive check list widgets."""

from __future__ import annotations

from typing import Any, Dict, List

from rich.ansi import AnsiDecoder
from rich.console import Group
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.widgets import Static

from .toggle import Toggle


def _escape_markup(text: str) -> str:
    """Escape square brackets so Rich markup doesn't interpret them."""
    return text.replace("[", r"\[")


_STATUS_ICONS = {
    "pass": "[green]\u2714[/]",
    "fail": "[red]\u2718[/]",
    "error": "[yellow]\u26a0[/]",
    "NA": "[dim]\u2022[/]",
}


class CheckPlotPanel(Static):
    """Renders a stored plotext chart inside a check entry (lazy)."""

    DEFAULT_CSS = """
    CheckPlotPanel {
        height: auto;
        min-height: 10;
        max-height: 22;
        padding: 0 2;
    }
    CheckPlotPanel.hidden {
        display: none;
    }
    """

    def __init__(self, plot_data: dict, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._plot_data = plot_data
        self._rendered = False
        self._decoder = AnsiDecoder()

    def render_plot(self) -> None:
        """Generate the plotext chart on first expansion."""
        if self._rendered:
            return
        self._rendered = True
        from ...plot import make_plot

        width = max(self.size.width - 4, 40)
        height = max(self.size.height - 2, 8)
        canvas = make_plot(
            width,
            height,
            self._plot_data["series"],
            title=self._plot_data.get("title", ""),
            xlim=self._plot_data.get("xlim"),
            ylim=self._plot_data.get("ylim"),
        )
        renderable = Group(*self._decoder.decode(canvas))
        self.update(renderable)


class CheckDetailPanel(Static):
    """Collapsible panel showing diagnostic key-value pairs for a check."""

    DEFAULT_CSS = """
    CheckDetailPanel {
        padding: 0 4;
        height: auto;
        color: $text-muted;
    }
    CheckDetailPanel.hidden {
        display: none;
    }
    """

    def __init__(self, details: dict | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._details = details or {}

    def on_mount(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        if not self._details:
            self.update("[dim]No details available[/]")
            return
        lines = []
        for k, v in self._details.items():
            lines.append(f"  [cyan]{_escape_markup(str(k))}[/]: {_escape_markup(str(v))}")
        self.update("\n".join(lines))


class CheckEntry(Vertical):
    """A single check row with toggle to expand/collapse details + plot."""

    _counter: int = 0  # class-level counter for unique IDs

    DEFAULT_CSS = """
    CheckEntry {
        height: auto;
        padding: 0 0;
    }
    CheckEntry > .check-row {
        height: 1;
    }
    CheckEntry > .check-row > Toggle {
        width: 3;
    }
    CheckEntry > .check-row > .check-icon {
        width: 3;
    }
    CheckEntry > .check-row > .check-name {
        width: 1fr;
    }
    CheckEntry > .check-row > .check-result {
        width: 8;
        content-align: right middle;
    }
    """

    def __init__(
        self,
        check_name: str,
        result: str,
        details: dict | None = None,
        plot_data: dict | None = None,
        *,
        id: str | None = None,
    ) -> None:
        CheckEntry._counter += 1
        self._uid = CheckEntry._counter
        super().__init__(id=id)
        self._check_name = check_name
        self._result = result
        self._details = details
        self._plot_data = plot_data

    def compose(self):
        icon = _STATUS_ICONS.get(self._result, "[magenta]?[/]")
        uid = self._uid
        with Horizontal(classes="check-row"):
            yield Toggle(id=f"toggle-{uid}")
            yield Static(icon, classes="check-icon")
            yield Static(self._check_name, classes="check-name", markup=False)
            yield Static(f"[bold]{_escape_markup(self._result)}[/]", classes="check-result")
        if self._plot_data:
            yield CheckPlotPanel(
                self._plot_data, classes="hidden", id=f"chkplot-{uid}"
            )
        yield CheckDetailPanel(
            self._details, classes="hidden", id=f"chkdetail-{uid}"
        )

    def on_toggle_toggled(self, event: Toggle.Toggled) -> None:
        detail = self.query_one(CheckDetailPanel)
        detail.toggle_class("hidden")
        try:
            plot_panel = self.query_one(CheckPlotPanel)
            plot_panel.toggle_class("hidden")
            # Lazy-render on first show
            if not plot_panel.has_class("hidden"):
                plot_panel.render_plot()
        except Exception:
            pass  # no plot for this check


class ChecksSummaryPanel(ScrollableContainer):
    """Compact summary for the Dashboard tab."""

    DEFAULT_CSS = """
    ChecksSummaryPanel {
        border: solid $primary-background;
        height: auto;
        min-height: 3;
        max-height: 8;
    }
    """

    def add_check(self, name: str, result: str, details: dict | None = None) -> None:
        entry = CheckEntry(name, result, details)
        self.mount(entry)


class ChecksDetailView(ScrollableContainer):
    """Full view for checks within a model section."""

    DEFAULT_CSS = """
    ChecksDetailView {
        height: auto;
        min-height: 3;
        border: solid $primary-background;
    }
    """

    BORDER_TITLE = "Checks"

    def add_check(
        self,
        name: str,
        result: str,
        details: dict | None = None,
        plot_data: dict | None = None,
    ) -> None:
        entry = CheckEntry(
            name, result, details, plot_data=plot_data,
        )
        self.mount(entry)
