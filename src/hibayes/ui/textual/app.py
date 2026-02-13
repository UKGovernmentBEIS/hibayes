"""HiBayesApp -- Textual TUI application."""

from __future__ import annotations

import os
import threading
from typing import Any, Callable

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal

from .widgets.footer import AppFooter
from .widgets.header import AppHeader
from .widgets.logs_panel import LogsPanel
from .widgets.model_selector import ModelDashboard
from .widgets.stats_panel import StatsPanel


class HiBayesApp(App):
    """Interactive Textual TUI for HiBayes."""

    CSS_PATH = "app.tcss"
    TITLE = "HiBayes"

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True, priority=True),
        Binding("left", "prev_model", "Prev model", show=False),
        Binding("right", "next_model", "Next model", show=False),
    ]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pipeline_fn: Callable | None = None
        self._pipeline_args: Any = None
        self._pipeline_display: Any = None  # TextualDisplay, set later
        self._quit_requested: bool = False

    def compose(self) -> ComposeResult:
        yield AppHeader()
        yield LogsPanel(id="logs-panel")
        with Horizontal(id="main-body"):
            yield ModelDashboard(id="model-dashboard")
            yield StatsPanel(id="stats-panel")
        yield AppFooter()

    def on_mount(self) -> None:
        if self._pipeline_fn is not None:
            self.run_worker(
                self._run_pipeline,
                thread=True,
                exclusive=True,
                name="pipeline",
            )

    async def _run_pipeline(self) -> None:
        """Execute the pipeline function inside a worker thread."""
        try:
            self._pipeline_fn(self._pipeline_args, display=self._pipeline_display)
        except KeyboardInterrupt:
            pass

    def on_worker_state_changed(self, event) -> None:
        """Exit the app when the pipeline worker finishes."""
        from textual.worker import WorkerState
        if event.worker.name == "pipeline" and event.state == WorkerState.SUCCESS:
            footer = self.query_one(AppFooter)
            footer.status = "Pipeline complete. Press Ctrl+C to exit."
            self.exit()

    def action_quit(self) -> None:
        if self._quit_requested:
            # Second Ctrl+C — force exit immediately
            os._exit(0)
        self._quit_requested = True

        # Unblock any pending prompt so the worker thread can finish
        if self._pipeline_display is not None:
            self._pipeline_display._abort_prompts()

        # Cancel the worker and tell Textual to shut down
        for worker in self.workers:
            worker.cancel()
        self.exit()

        # If the worker thread is stuck in computation we can't interrupt
        # it, so schedule a hard exit after giving Textual time to restore
        # the terminal.
        t = threading.Timer(1.0, lambda: os._exit(0))
        t.daemon = True
        t.start()

    def action_prev_model(self) -> None:
        try:
            dashboard = self.query_one(ModelDashboard)
            dashboard.switch_relative(-1)
        except Exception:
            pass

    def action_next_model(self) -> None:
        try:
            dashboard = self.query_one(ModelDashboard)
            dashboard.switch_relative(1)
        except Exception:
            pass


def run_with_tui(pipeline_fn: Callable, args: Any) -> None:
    """Launch the Textual TUI and run *pipeline_fn(args, display=...)* inside it."""
    from .display import TextualDisplay

    app = HiBayesApp()
    display = TextualDisplay(app=app)
    app._pipeline_fn = pipeline_fn
    app._pipeline_args = args
    app._pipeline_display = display
    app.run()
