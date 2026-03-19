"""Interactive communicate result list widget."""

from __future__ import annotations

from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.widgets import Static


def _escape_markup(text: str) -> str:
    """Escape square brackets so Rich markup doesn't interpret them."""
    return text.replace("[", r"\[")

from .toggle import Toggle


_STATUS_ICONS = {
    "pass": "[green]\u2714[/]",
    "fail": "[red]\u2718[/]",
    "error": "[yellow]\u26a0[/]",
    "NA": "[dim]\u2022[/]",
}


class CommDetailPanel(Vertical):
    """Collapsible panel showing details for a communicate result.

    When the details include ``table_data``, the DataFrames are rendered
    as formatted Textual DataTable widgets.
    """

    DEFAULT_CSS = """
    CommDetailPanel {
        padding: 0 4;
        height: auto;
        color: $text-muted;
    }
    CommDetailPanel.hidden {
        display: none;
    }
    """

    def __init__(self, details: dict | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._details = details or {}

    def on_mount(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        table_data = self._details.get("table_data")

        if table_data:
            self._render_tables(table_data)
        elif self._details:
            meta = {k: v for k, v in self._details.items() if k != "table_data"}
            if meta:
                lines = []
                for k, v in meta.items():
                    lines.append(f"  [cyan]{_escape_markup(str(k))}[/]: {_escape_markup(str(v))}")
                self.mount(Static("\n".join(lines)))
            else:
                self.mount(Static("[dim]No details available[/]"))
        else:
            self.mount(Static("[dim]No details available[/]"))

    def _render_tables(self, table_data: dict) -> None:
        from textual.widgets import DataTable

        for table_name, tbl in table_data.items():
            columns = tbl["columns"]
            index = tbl["index"]
            data = tbl["data"]

            dt = DataTable(zebra_stripes=True)
            dt.styles.height = "auto"
            dt.styles.max_height = 30
            dt.add_column("Variable", key="__idx__")
            for col in columns:
                dt.add_column(str(col), key=str(col))
            self.mount(dt)
            for idx_label, row in zip(index, data):
                dt.add_row(idx_label, *row)


class CommEntry(Vertical):
    """A single communicate result row with toggle to expand details."""

    _counter: int = 0  # class-level counter for unique IDs

    DEFAULT_CSS = """
    CommEntry {
        height: auto;
    }
    CommEntry > .comm-row {
        height: 1;
    }
    CommEntry > .comm-row > Toggle {
        width: 3;
    }
    CommEntry > .comm-row > .comm-icon {
        width: 3;
    }
    CommEntry > .comm-row > .comm-name {
        width: 1fr;
    }
    CommEntry > .comm-row > .comm-result {
        width: 8;
        content-align: right middle;
    }
    """

    def __init__(
        self,
        name: str,
        result: str,
        details: dict | None = None,
        *,
        id: str | None = None,
    ) -> None:
        CommEntry._counter += 1
        self._uid = CommEntry._counter
        super().__init__(id=id)
        self._name = name
        self._result = result
        self._details = details

    def compose(self):
        icon = _STATUS_ICONS.get(self._result, "[magenta]?[/]")
        uid = self._uid
        with Horizontal(classes="comm-row"):
            yield Toggle(id=f"toggle-comm-{uid}")
            yield Static(icon, classes="comm-icon")
            yield Static(self._name, classes="comm-name", markup=False)
            yield Static(f"[bold]{_escape_markup(self._result)}[/]", classes="comm-result")
        yield CommDetailPanel(
            self._details, classes="hidden", id=f"comm-detail-{uid}"
        )

    def on_toggle_toggled(self, event: Toggle.Toggled) -> None:
        detail = self.query_one(CommDetailPanel)
        detail.toggle_class("hidden")


class CommsDetailView(ScrollableContainer):
    """Communicate results within a model section."""

    DEFAULT_CSS = """
    CommsDetailView {
        height: auto;
        min-height: 3;
        border: solid $primary-background;
    }
    """

    BORDER_TITLE = "Results"

    def add_communicate_result(
        self, name: str, result: str, details: dict | None = None
    ) -> None:
        entry = CommEntry(name, result, details)
        self.mount(entry)
