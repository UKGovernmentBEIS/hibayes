"""Per-model section widget containing progress, plot, and checks."""

from __future__ import annotations

from typing import Any

from textual.containers import VerticalScroll

from .checks_panel import ChecksDetailView
from .comms_panel import CommsDetailView
from .idata_panel import InferenceDataPanel
from .plot_panel import PlotPanel
from .progress_panel import ProgressPanel


class ModelSection(VerticalScroll):
    """A per-model container holding progress bars, a plot panel, and checks."""

    DEFAULT_CSS = """
    ModelSection {
        height: 1fr;
    }
    ModelSection > .model-label {
        height: 1;
        text-style: bold;
        color: $text;
        background: $primary-background;
        content-align: center middle;
        width: 100%;
    }
    """

    def __init__(self, model_name: str, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._model_name = model_name
        self._progress_id = f"progress-{id}" if id else "progress"
        self._plot_id = f"plot-{id}" if id else "plot"
        self._checks_id = f"checks-{id}" if id else "checks"
        self._idata_id = f"idata-{id}" if id else "idata"
        self._comms_id = f"comms-{id}" if id else "comms"

    def compose(self):
        from textual.widgets import Static

        yield Static(self._model_name, classes="model-label")
        yield ProgressPanel(id=self._progress_id)
        yield InferenceDataPanel(id=self._idata_id)
        yield PlotPanel(id=self._plot_id)
        yield ChecksDetailView(id=self._checks_id)
        yield CommsDetailView(id=self._comms_id)

    @property
    def progress_panel(self) -> ProgressPanel:
        return self.query_one(ProgressPanel)

    @property
    def plot_panel(self) -> PlotPanel:
        return self.query_one(PlotPanel)

    @property
    def idata_panel(self) -> InferenceDataPanel:
        return self.query_one(InferenceDataPanel)

    @property
    def checks_view(self) -> ChecksDetailView:
        return self.query_one(ChecksDetailView)

    @property
    def comms_view(self) -> CommsDetailView:
        return self.query_one(CommsDetailView)
