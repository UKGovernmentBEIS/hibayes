"""Display protocol for HiBayes UI backends."""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class Display(Protocol):
    """Protocol capturing the public interface that all display backends must implement.

    The Rich-based ``ModellingDisplay`` and the Textual-based ``TextualDisplay``
    both satisfy this protocol so that callers (checkers, communicators, analysis
    functions) can work with either backend.
    """

    logger: logging.Logger

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None: ...

    def stop(self) -> None: ...

    @property
    def is_live(self) -> bool: ...

    # -- header / logs --------------------------------------------------------

    def update_header(self, text: str) -> None: ...

    def update_logs(self, log_entry: str) -> None: ...

    def capture_logs(
        self, logger_names: list[str] | None = None
    ) -> contextlib.AbstractContextManager: ...

    # -- statistics -----------------------------------------------------------

    def update_stat(self, key: str, value: Any) -> None: ...

    def update_stats(self, stats: Dict[str, Any]) -> None: ...

    # -- progress / tasks -----------------------------------------------------

    def add_task(
        self,
        description: str,
        chain: Optional[str] = None,
        worker: Optional[int] = None,
        total: Optional[int] = None,
    ) -> int: ...

    def update_task(self, description: str, advance: int = 1, **kwargs: Any) -> None: ...

    def update_task_description(
        self, description: str, new_description: str
    ) -> None: ...

    # -- plots ----------------------------------------------------------------

    def add_plot(
        self,
        series: List[Dict[str, list | str]],
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        title: str = "",
    ) -> None: ...

    # -- user prompts ---------------------------------------------------------

    def prompt_user(
        self,
        question: str = "Would you like to proceed?",
        options: list | None = None,
    ) -> bool: ...

    # -- checks / communicate -------------------------------------------------

    def add_check(
        self,
        check_name: str,
        result: str,
        details: dict | None = None,
    ) -> None: ...

    def add_communicate_result(
        self,
        name: str,
        result: str,
        details: dict | None = None,
    ) -> None: ...

    # -- body content ---------------------------------------------------------

    def update_body_content(self, content: Any) -> None: ...

    # -- persistence / introspection ------------------------------------------

    def get_all_logs(self) -> List[str]: ...

    def get_stats_for_persistence(self) -> Dict[str, Any]: ...

    # -- modelling setup ------------------------------------------------------

    def setupt_for_modelling(self) -> None: ...

    def set_fit_context(self, chain_method: str, num_chains: int) -> None: ...

    def set_active_model(self, model_name: str) -> None: ...

    def show_inference_summary(self, summary: Dict[str, Any]) -> None: ...
