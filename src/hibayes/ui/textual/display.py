"""TextualDisplay -- implements the Display protocol, bridges pipeline thread to Textual app."""

from __future__ import annotations

import contextlib
import datetime
import logging
import threading
import time
from typing import Any, Dict, List, Optional

from ...utils.logger import init_logger
from ..logger import LogCaptureHandler


class TextualDisplay:
    """Display backend that satisfies the ``Display`` protocol and routes
    all widget updates through ``self._app.call_from_thread(...)`` so that
    the pipeline can run on a worker thread while Textual owns the main
    thread.
    """

    def __init__(
        self,
        app: Any,  # HiBayesApp -- late import to avoid circular deps
        logger: logging.Logger | None = None,
        max_logs: int = 50,
        initial_stats: dict | None = None,
    ) -> None:
        self._app = app
        self.logger = logger or init_logger()
        self._max_logs = max_logs
        self.logs: list[str] = []
        self.all_logs: list[str] = []

        default_stats: dict[str, Any] = {
            "Samples found": 0,
            "Samples processed": 0,
            "Processing speed": "0 samples/sec",
            "AI Models detected": set(),
            "Sample errors": 0,
            "Extractor errors": 0,
            "MCMC samples": 0,
            "Statistical Models": set(),
            "Num divergents": 0,
            "Checks passed": 0,
            "Checks failed": 0,
            "Errors encountered": 0,
        }

        self._loaded_from_state = False
        if initial_stats:
            default_stats.update(initial_stats)
            if "Processing speed" in initial_stats:
                self._loaded_from_state = True

        self.stats: dict[str, Any] = default_stats
        self.start_time = time.time()

        # For modelling / numpyro integration
        self.modelling = False
        self.original_fori_collect = None
        self.chain_method = "parallel"
        self.num_chains = 1

        self._live = False
        self._task_ids: dict[str, int] = {}  # description → tid (compat)
        self._tid_to_key: dict[int, str] = {}  # tid → unique widget key
        self._next_tid: int = 0
        self._pending_plot_data: dict | None = None  # held for next add_check

        # -- model routing state --
        self._current_model: str | None = None
        self._model_sections: dict[str, Any] = {}  # model_name → ModelSection
        self._tid_to_model: dict[int, str] = {}  # task_id → model_name

        # A lightweight proxy so NumPyroRichProgress can call
        # self.display.progress.update() / .add_task() unchanged.
        self.progress = _ProgressProxy(self)

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        self._live = True

    def stop(self) -> None:
        self._live = False

    def _abort_prompts(self) -> None:
        """Unblock any pending prompt_user() call so the worker can exit."""
        # Inline prompt on ALL PlotPanels (one per model section)
        try:
            from .widgets.plot_panel import PlotPanel
            for panel in self._app.query(PlotPanel):
                if panel._event is not None:
                    panel._event.set()
        except Exception:
            pass
        # Modal prompt
        try:
            from .widgets.prompt_modal import PromptModal
            screen = self._app.screen
            if isinstance(screen, PromptModal):
                screen._event.set()
        except Exception:
            pass

    @property
    def is_live(self) -> bool:
        return self._live

    # -- model routing --------------------------------------------------------

    def set_active_model(self, model_name: str) -> None:
        """Create (if needed) and switch to the section for *model_name*."""
        self._current_model = model_name
        try:
            self._app.call_from_thread(self._ensure_and_switch_model, model_name)
        except Exception:
            pass

    def _ensure_and_switch_model(self, model_name: str) -> None:
        from .widgets.model_selector import ModelDashboard

        dashboard = self._app.query_one(ModelDashboard)
        section = dashboard.add_model_section(model_name)
        self._model_sections[model_name] = section
        dashboard.switch_to(model_name)

    def _get_current_section(self):
        """Return the ModelSection for the current model (or None)."""
        if self._current_model is None:
            return None
        return self._model_sections.get(self._current_model)

    def _get_section_for_model(self, model_name: str):
        """Return the ModelSection for a specific model name."""
        return self._model_sections.get(model_name)

    # -- header / logs --------------------------------------------------------

    def update_header(self, text: str) -> None:
        title = f"HiBayes - {text}"
        try:
            self._app.call_from_thread(self._set_header, title)
        except Exception:
            pass

    def _set_header(self, title: str) -> None:
        from .widgets.header import AppHeader
        header = self._app.query_one(AppHeader)
        header.title = title

    def update_logs(self, log_entry: str) -> None:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        entry = f"[{ts}] {log_entry}"
        self.logs.append(entry)
        self.all_logs.append(entry)
        self.logs = self.logs[-self._max_logs:]
        try:
            self._app.call_from_thread(self._append_log, entry)
        except Exception:
            pass

    def _append_log(self, entry: str) -> None:
        from .widgets.logs_panel import LogsPanel
        panel = self._app.query_one(LogsPanel)
        panel.append_log(entry)

    @contextlib.contextmanager
    def capture_logs(self, logger_names: list[str] | None = None):
        """Context manager to capture logs -- same logic as ModellingDisplay."""
        if logger_names is None:
            logger_names = ["jax", "numpyro", "absl", "arviz", None]

        existing_handlers: list[tuple] = []
        for name in logger_names:
            lgr = logging.getLogger(name)
            for handler in lgr.handlers:
                existing_handlers.append((lgr, handler))
                lgr.removeHandler(handler)

        handlers: list[tuple] = []
        for name in logger_names:
            lgr = logging.getLogger(name)
            handler = LogCaptureHandler(self.update_logs)
            lgr.addHandler(handler)
            handlers.append((lgr, handler))

        try:
            yield
        finally:
            for lgr, handler in handlers:
                lgr.removeHandler(handler)
            for lgr, handler in existing_handlers:
                lgr.addHandler(handler)

    # -- statistics -----------------------------------------------------------

    def update_stat(self, key: str, value: Any) -> None:
        self.stats[key] = value
        self._push_stats()

    def update_stats(self, stats: Dict[str, Any]) -> None:
        self.stats.update(stats)
        self._push_stats()

    def _push_stats(self) -> None:
        # Recalculate processing speed if appropriate
        if not self.modelling and self.stats["Samples processed"] > 0:
            if not self._loaded_from_state:
                elapsed = time.time() - self.start_time
                if elapsed > 0:
                    self.stats["Processing speed"] = (
                        f"{self.stats['Samples processed'] / elapsed:.1f} samples/sec"
                    )
        try:
            self._app.call_from_thread(self._set_stats, dict(self.stats))
        except Exception:
            pass

    def _set_stats(self, stats: dict) -> None:
        from .widgets.stats_panel import StatsPanel
        panel = self._app.query_one(StatsPanel)
        panel.update_stats(stats)

    # -- progress / tasks -----------------------------------------------------

    def add_task(
        self,
        description: str,
        chain: Optional[str] = None,
        worker: Optional[int] = None,
        total: Optional[int] = None,
    ) -> int:
        tid = self._next_tid
        self._next_tid += 1
        # Unique key so multiple tasks with the same description don't collide
        unique_key = f"{description}:{tid}"
        self._task_ids[description] = tid
        self._tid_to_key[tid] = unique_key
        # Record which model this task belongs to
        if self._current_model:
            self._tid_to_model[tid] = self._current_model
        try:
            self._app.call_from_thread(
                self._add_task_widget, unique_key, description, chain, worker, total
            )
        except Exception:
            pass
        return tid

    def _add_task_widget(
        self,
        key: str,
        description: str,
        chain: str | None,
        worker: int | None,
        total: int | None,
    ) -> None:
        section = self._get_current_section()
        if section is None:
            return
        # For __results__ section (CommsDetailView), no progress panel
        if not hasattr(section, "progress_panel"):
            return
        section.progress_panel.add_task(key, description, chain=chain, worker=worker, total=total)

    def update_task(self, description: str, advance: int = 1, **kwargs: Any) -> None:
        tid = self._task_ids.get(description)
        if tid is None:
            return
        key = self._tid_to_key.get(tid)
        if key is None:
            return
        model_name = self._tid_to_model.get(tid)
        new_desc = kwargs.pop("description", None)
        try:
            self._app.call_from_thread(
                self._update_task_widget, key, advance, kwargs, new_desc, model_name
            )
        except Exception:
            pass

    def _update_task_widget(
        self, key: str, advance: int, kwargs: dict, new_desc: str | None = None,
        model_name: str | None = None,
    ) -> None:
        section = self._get_section_for_model(model_name) if model_name else self._get_current_section()
        if section is None or not hasattr(section, "progress_panel"):
            return
        section.progress_panel.update_task(key, advance=advance, new_description=new_desc, **kwargs)

    def update_task_description(self, description: str, new_description: str) -> None:
        tid = self._task_ids.get(description)
        if tid is None:
            return
        key = self._tid_to_key.get(tid)
        if key is None:
            return
        model_name = self._tid_to_model.get(tid)
        try:
            self._app.call_from_thread(
                self._update_task_desc_widget, key, new_description, model_name
            )
        except Exception:
            pass
        # Update internal mapping
        self._task_ids[new_description] = self._task_ids.pop(description, tid)

    def _update_task_desc_widget(self, key: str, new_description: str, model_name: str | None = None) -> None:
        section = self._get_section_for_model(model_name) if model_name else self._get_current_section()
        if section is None or not hasattr(section, "progress_panel"):
            return
        section.progress_panel.update_task_description(key, new_description)

    # -- plots ----------------------------------------------------------------

    def add_plot(
        self,
        series: List[Dict[str, list | str]],
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        title: str = "",
    ) -> None:
        self.update_logs(f"Plot: {title}")
        # Stash so the next add_check() can attach it to the check entry
        self._pending_plot_data = {
            "series": series if isinstance(series, list) else [series],
            "title": title,
            "xlim": xlim,
            "ylim": ylim,
        }
        try:
            self._app.call_from_thread(
                self._show_plot, series, title, xlim, ylim
            )
        except Exception:
            pass

    def _show_plot(
        self,
        series: list,
        title: str,
        xlim: tuple | None,
        ylim: tuple | None,
    ) -> None:
        section = self._get_current_section()
        if section is None or not hasattr(section, "plot_panel"):
            return
        section.plot_panel.show_plot(series, title=title, xlim=xlim, ylim=ylim)

    # -- user prompts ---------------------------------------------------------

    def prompt_user(
        self,
        question: str = "Would you like to proceed?",
        options: list | None = None,
    ) -> bool:
        event = threading.Event()
        result_holder: list[bool] = [False]
        self._app.call_from_thread(
            self._show_prompt, question, event, result_holder
        )
        event.wait()
        # After the user responds, clear the plot so it doesn't linger
        try:
            self._app.call_from_thread(self._clear_plot)
        except Exception:
            pass
        return result_holder[0]

    def _show_prompt(
        self, question: str, event: threading.Event, result_holder: list
    ) -> None:
        section = self._get_current_section()
        if section is not None and hasattr(section, "plot_panel"):
            panel = section.plot_panel
            # If a plot is visible, show the prompt inline beneath it
            if not panel.has_class("hidden"):
                panel.show_prompt(question, event, result_holder)
                return
        # No plot on screen — fall back to the modal overlay
        from .widgets.prompt_modal import PromptModal
        self._app.push_screen(PromptModal(question, event, result_holder))

    def _clear_plot(self) -> None:
        section = self._get_current_section()
        if section is not None and hasattr(section, "plot_panel"):
            section.plot_panel.clear_plot()

    # -- checks / communicate -------------------------------------------------

    def add_check(
        self, check_name: str, result: str, details: dict | None = None
    ) -> None:
        if result == "pass":
            self.stats["Checks passed"] = self.stats.get("Checks passed", 0) + 1
        elif result == "fail":
            self.stats["Checks failed"] = self.stats.get("Checks failed", 0) + 1
        elif result == "error":
            self.stats["Checks errored"] = self.stats.get("Checks errored", 0) + 1
        self._push_stats()

        # Grab any plot that was shown for this check
        plot_data = self._pending_plot_data
        self._pending_plot_data = None

        try:
            self._app.call_from_thread(
                self._add_check_widget, check_name, result, details, plot_data
            )
        except Exception:
            pass

    def _add_check_widget(
        self, name: str, result: str, details: dict | None, plot_data: dict | None
    ) -> None:
        section = self._get_current_section()
        if section is None or not hasattr(section, "checks_view"):
            return
        section.checks_view.add_check(name, result, details, plot_data=plot_data)

    def add_communicate_result(
        self, name: str, result: str, details: dict | None = None
    ) -> None:
        try:
            self._app.call_from_thread(
                self._add_comm_widget, name, result, details
            )
        except Exception:
            pass

    def _add_comm_widget(
        self, name: str, result: str, details: dict | None
    ) -> None:
        section = self._get_current_section()
        if section is None or not hasattr(section, "comms_view"):
            return
        section.comms_view.add_communicate_result(name, result, details)

    # -- body content ---------------------------------------------------------

    def update_body_content(self, content: Any) -> None:
        # Rich renderables can't be directly shown in Textual.
        # Log a note instead.
        self.update_logs("[body content updated]")

    # -- persistence / introspection ------------------------------------------

    def get_all_logs(self) -> List[str]:
        return self.all_logs.copy()

    def get_stats_for_persistence(self) -> Dict[str, Any]:
        stats_copy = {}
        for key, value in self.stats.items():
            if isinstance(value, set):
                stats_copy[key] = list(value)
            else:
                stats_copy[key] = value
        return stats_copy

    # -- modelling setup ------------------------------------------------------

    def setupt_for_modelling(self) -> None:
        """Patch NumPyro's fori_collect to integrate with this display."""
        from ..display import patch_fori_collect_with_rich_display
        self.original_fori_collect = patch_fori_collect_with_rich_display(self)
        self.modelling = True

    def set_fit_context(self, chain_method: str, num_chains: int) -> None:
        self.chain_method = chain_method
        self.num_chains = num_chains

    # -- inference data -------------------------------------------------------

    def show_inference_summary(self, summary: dict) -> None:
        try:
            self._app.call_from_thread(self._show_idata_summary, summary)
        except Exception:
            pass

    def _show_idata_summary(self, summary: dict) -> None:
        section = self._get_current_section()
        if section is None or not hasattr(section, "idata_panel"):
            return
        section.idata_panel.update_summary(summary)


class _ProgressProxy:
    """Lightweight proxy that mimics the Rich ``Progress`` API so that
    ``NumPyroRichProgress`` can call ``display.progress.update()`` and
    ``display.progress.add_task()`` unchanged.
    """

    def __init__(self, display: TextualDisplay) -> None:
        self._display = display

    def add_task(
        self, description: str, total: int | None = None, info: str = "", **kw: Any
    ) -> int:
        return self._display.add_task(description, chain=info, total=total)

    def update(self, task_id: int, advance: int = 0, **kwargs: Any) -> None:
        # Resolve task_id → unique widget key
        key = self._display._tid_to_key.get(task_id)
        if key is None:
            return
        # Find which model this task belongs to
        model_name = self._display._tid_to_model.get(task_id)
        # Separate 'description' from kwargs so it doesn't collide with the
        # positional parameter in update_task()
        new_desc = kwargs.pop("description", None)
        try:
            self._display._app.call_from_thread(
                self._display._update_task_widget, key, advance, kwargs, new_desc, model_name
            )
        except Exception:
            pass
