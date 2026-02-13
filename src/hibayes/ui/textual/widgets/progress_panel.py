"""MCMC chain progress bar widgets."""

from __future__ import annotations

from textual.containers import VerticalScroll
from textual.widgets import ProgressBar, Static


class ProgressEntry(Static):
    """A single progress bar with description and chain label."""

    DEFAULT_CSS = """
    ProgressEntry {
        height: 2;
        layout: horizontal;
        padding: 0 1;
    }
    ProgressEntry > .pe-desc {
        width: 20;
        content-align: left middle;
    }
    ProgressEntry > .pe-info {
        width: 16;
        content-align: left middle;
        color: $text-muted;
    }
    ProgressEntry > ProgressBar {
        width: 1fr;
    }
    """

    def __init__(
        self,
        description: str,
        info: str = "",
        total: float | None = None,
        *,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._description = description
        self._info = info
        self._total = total or 100.0

    def compose(self):
        yield Static(self._description, classes="pe-desc")
        yield Static(self._info, classes="pe-info")
        yield ProgressBar(total=self._total, show_eta=True)

    def advance(self, amount: float = 1) -> None:
        bar = self.query_one(ProgressBar)
        bar.advance(amount)

    def set_progress(self, completed: float, total: float | None = None) -> None:
        bar = self.query_one(ProgressBar)
        if total is not None:
            bar.total = total
        bar.progress = completed

    def set_description(self, description: str) -> None:
        self._description = description
        desc_widget = self.query(".pe-desc").first()
        desc_widget.update(description)

    def set_info(self, info: str) -> None:
        self._info = info
        info_widget = self.query(".pe-info").first()
        info_widget.update(info)


class ProgressPanel(VerticalScroll):
    """Container for MCMC chain progress bars."""

    DEFAULT_CSS = """
    ProgressPanel {
        border: solid $primary-background;
        height: auto;
        min-height: 3;
        max-height: 12;
    }
    """

    def __init__(self, *, id: str | None = None, classes: str | None = None) -> None:
        super().__init__(id=id, classes=classes)
        self._entries: dict[str, ProgressEntry] = {}

    def add_task(
        self,
        key: str,
        description: str,
        chain: str | None = None,
        worker: int | None = None,
        total: int | None = None,
    ) -> str:
        """Add a new progress entry keyed by *key*."""
        info = ""
        if chain is not None:
            info = f"Chain {chain}"
        elif worker is not None:
            info = f"Workers: {worker}"

        entry = ProgressEntry(
            description=description,
            info=info,
            total=total,
        )
        self._entries[key] = entry
        self.mount(entry)
        return key

    def update_task(
        self,
        key: str,
        advance: float = 1,
        new_description: str | None = None,
        **kwargs,
    ) -> None:
        entry = self._entries.get(key)
        if entry is None:
            return
        if "completed" in kwargs and "total" in kwargs:
            entry.set_progress(kwargs["completed"], kwargs["total"])
        else:
            entry.advance(advance)
        if new_description:
            entry.set_description(new_description)
        if "info" in kwargs:
            entry.set_info(kwargs["info"])

    def update_task_description(self, key: str, new_description: str) -> None:
        entry = self._entries.get(key)
        if entry is not None:
            entry.set_description(new_description)
