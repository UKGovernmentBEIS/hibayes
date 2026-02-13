"""InferenceData summary widget — xarray-style groups, dims, coords, variables."""

from __future__ import annotations

from typing import Any, Dict

from textual.containers import Vertical
from textual.widgets import Static

from .toggle import Toggle


def _esc(text: str) -> str:
    """Escape square brackets for Rich markup."""
    return text.replace("[", r"\[")


class InferenceGroupPanel(Static):
    """Collapsible panel showing xarray-style details for one group."""

    DEFAULT_CSS = """
    InferenceGroupPanel {
        padding: 0 2;
        height: auto;
        color: $text-muted;
    }
    InferenceGroupPanel.hidden {
        display: none;
    }
    """

    def __init__(self, group_info: dict, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._group_info = group_info

    def on_mount(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        lines: list[str] = []
        dims = self._group_info.get("dims", {})
        coords = self._group_info.get("coords", {})
        variables = self._group_info.get("vars", {})

        # Dimensions
        if dims:
            lines.append("[bold]Dimensions:[/]")
            dim_parts = [f"{_esc(k)}: [green]{v}[/]" for k, v in dims.items()]
            lines.append(f"    {', '.join(dim_parts)}")

        # Coordinates
        if coords:
            lines.append("[bold]Coordinates:[/]")
            for coord_name, coord_info in coords.items():
                dim_str = ", ".join(coord_info.get("dims", []))
                dtype = coord_info.get("dtype", "")
                preview = coord_info.get("values")
                detail = f"    [cyan]{_esc(coord_name)}[/]  ({dim_str})  [dim]{dtype}[/]"
                if preview:
                    detail += f"  {' '.join(_esc(v) for v in preview)}"
                lines.append(detail)

        # Data variables
        if variables:
            lines.append("[bold]Data variables:[/]")
            for var_name, var_info in variables.items():
                dim_str = ", ".join(var_info.get("dims", []))
                dtype = var_info.get("dtype", "")
                shape = var_info.get("shape", [])
                total = 1
                for s in shape:
                    total *= s
                lines.append(
                    f"    [green]{_esc(var_name)}[/]  ({dim_str})  "
                    f"[dim]{dtype}[/]  "
                    f"[dim]{_esc(str(shape))}[/]"
                )

        if not lines:
            lines.append("[dim](empty)[/]")

        self.update("\n".join(lines))


class InferenceGroupEntry(Vertical):
    """A single group row with toggle to expand xarray-style details."""

    _counter: int = 0

    DEFAULT_CSS = """
    InferenceGroupEntry {
        height: auto;
    }
    InferenceGroupEntry > .idata-group-row {
        height: 1;
    }
    InferenceGroupEntry > .idata-group-row > Toggle {
        width: 3;
    }
    InferenceGroupEntry > .idata-group-row > .idata-group-name {
        width: auto;
        min-width: 16;
    }
    InferenceGroupEntry > .idata-group-row > .idata-group-meta {
        width: 1fr;
        content-align: right middle;
        color: $text-muted;
    }
    """

    def __init__(self, group_name: str, group_info: dict) -> None:
        InferenceGroupEntry._counter += 1
        self._uid = InferenceGroupEntry._counter
        super().__init__()
        self._group_name = group_name
        self._group_info = group_info

    def compose(self):
        from textual.containers import Horizontal

        uid = self._uid
        dims = self._group_info.get("dims", {})
        num_vars = len(self._group_info.get("vars", {}))
        num_coords = len(self._group_info.get("coords", {}))

        # Compact dimension summary for the header row
        dim_parts = [f"{k}: {v}" for k, v in dims.items()]
        meta_parts = []
        if dim_parts:
            meta_parts.append(" x ".join(dim_parts))
        meta_parts.append(f"{num_vars} vars")
        if num_coords:
            meta_parts.append(f"{num_coords} coords")
        meta_text = ", ".join(meta_parts)

        with Horizontal(classes="idata-group-row"):
            yield Toggle(id=f"idata-toggle-{uid}")
            yield Static(
                f"[bold cyan]{_esc(self._group_name)}[/]",
                classes="idata-group-name",
            )
            yield Static(f"[dim]{meta_text}[/]", classes="idata-group-meta")
        yield InferenceGroupPanel(
            self._group_info, classes="hidden", id=f"idata-vars-{uid}"
        )

    def on_toggle_toggled(self, event: Toggle.Toggled) -> None:
        panel = self.query_one(InferenceGroupPanel)
        panel.toggle_class("hidden")


class InferenceDataPanel(Vertical):
    """Displays an InferenceData summary with xarray-style group details."""

    DEFAULT_CSS = """
    InferenceDataPanel {
        height: auto;
        border: solid $primary-background;
        padding: 0 1;
    }
    InferenceDataPanel.hidden {
        display: none;
    }
    InferenceDataPanel > .idata-title {
        height: 1;
        text-style: bold;
        color: $accent;
    }
    """

    BORDER_TITLE = "InferenceData"

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)

    def compose(self):
        yield Static("", classes="idata-title")

    def on_mount(self) -> None:
        self.add_class("hidden")

    def update_summary(self, summary: Dict[str, Any]) -> None:
        """Replace the displayed summary with new data."""
        # Remove old entries
        for entry in self.query(InferenceGroupEntry):
            entry.remove()

        # Update title with group count
        title = self.query_one(".idata-title", Static)
        title.update(f"[bold]{len(summary)} groups[/]")

        # Add new group entries
        for group_name, group_info in summary.items():
            self.mount(InferenceGroupEntry(group_name, group_info))
        self.remove_class("hidden")
