"""Plot panel widget – renders plotext charts with an inline approval prompt."""

from __future__ import annotations

import threading
from typing import Any, Dict, List

from rich.ansi import AnsiDecoder
from rich.console import Group
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Label, Static


class PlotPanel(Vertical):
    """Displays a plotext chart with an inline Yes/No prompt.

    Starts hidden.  Call :meth:`show_plot` to render a chart and
    :meth:`clear_plot` to hide the panel.  Call :meth:`show_prompt` to
    display a question with Yes/No buttons beneath the chart; the caller
    blocks on a :class:`threading.Event` until the user responds.
    """

    DEFAULT_CSS = """
    PlotPanel {
        height: auto;
        min-height: 12;
        max-height: 30;
        border: solid $primary-background;
    }
    PlotPanel.hidden {
        display: none;
    }
    PlotPanel > .plot-canvas {
        height: auto;
        min-height: 10;
    }
    PlotPanel > .plot-prompt-bar {
        height: auto;
        padding: 0 1;
        align: center middle;
        background: $surface;
    }
    PlotPanel > .plot-prompt-bar.hidden {
        display: none;
    }
    PlotPanel > .plot-prompt-bar > .prompt-question {
        width: 1fr;
        content-align: left middle;
        padding: 0 1;
    }
    PlotPanel > .plot-prompt-bar > Button {
        margin: 0 1;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._decoder = AnsiDecoder()
        self._event: threading.Event | None = None
        self._result_holder: list[bool] | None = None

    def compose(self) -> ComposeResult:
        yield Static("", classes="plot-canvas")
        with Horizontal(classes="plot-prompt-bar hidden"):
            yield Label("", classes="prompt-question")
            yield Button("Yes", id=f"plot-yes-{self.id}" if self.id else "plot-yes", variant="success")
            yield Button("No", id=f"plot-no-{self.id}" if self.id else "plot-no", variant="error")

    def on_mount(self) -> None:
        self.add_class("hidden")

    # -- plot -----------------------------------------------------------------

    def show_plot(
        self,
        series: List[Dict[str, Any]] | Dict[str, Any],
        title: str = "",
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
    ) -> None:
        """Generate and display a plotext chart."""
        from ...plot import make_plot

        if not isinstance(series, list):
            series = [series]

        canvas_widget = self.query_one(".plot-canvas", Static)
        # Use the widget's current dimensions (fall back to reasonable defaults)
        width = max(self.size.width - 2, 40)
        height = max(canvas_widget.size.height, 10)

        chart = make_plot(width, height, series, title=title, xlim=xlim, ylim=ylim)
        renderable = Group(*self._decoder.decode(chart))
        canvas_widget.update(renderable)
        self.remove_class("hidden")

    def clear_plot(self) -> None:
        """Hide the plot panel and the prompt bar."""
        canvas_widget = self.query_one(".plot-canvas", Static)
        canvas_widget.update("")
        self.add_class("hidden")
        self._hide_prompt_bar()

    # -- inline prompt --------------------------------------------------------

    def show_prompt(
        self,
        question: str,
        event: threading.Event,
        result_holder: list[bool],
    ) -> None:
        """Show the Yes/No prompt bar beneath the chart."""
        self._event = event
        self._result_holder = result_holder
        label = self.query_one(".prompt-question", Label)
        label.update(question)
        bar = self.query_one(".plot-prompt-bar")
        bar.remove_class("hidden")

    def _hide_prompt_bar(self) -> None:
        bar = self.query_one(".plot-prompt-bar")
        bar.add_class("hidden")
        self._event = None
        self._result_holder = None

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if self._event is None or self._result_holder is None:
            return
        self._result_holder[0] = "plot-yes" in event.button.id
        evt = self._event
        self._hide_prompt_bar()
        evt.set()
