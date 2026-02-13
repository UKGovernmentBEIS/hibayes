import contextlib
import datetime
import logging
import time
from functools import partial
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

from rich import box
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.prompt import Confirm
from rich.table import Table
from rich.text import Text

from ..utils.logger import init_logger
from .logger import LogCaptureHandler
from .plot import plotextMixin


class ModellingDisplay:
    """Rich display for fitting and testing statistical models."""

    def __init__(
        self,
        logger: logging.Logger | None = None,
        max_logs: int = 5,
        initial_stats: dict | None = None,
    ):
        self.layout = Layout()
        self.layout.split_column(
            Layout(name="logs", size=8),
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="checks", size=3),
            Layout(name="footer", size=3),
        )

        # Set up logs panel
        self.logs = []
        self.all_logs = []  # Store all logs for saving
        self.max_logs = max_logs
        self.layout["logs"].update(
            Panel("", title="Logger output", border_style="dim white")
        )
        self.logger = logger or init_logger()
        self.package_name = "HiBayes"
        self.header_panel = Panel(
            Text(self.package_name, style="bold blue"),
            box=box.ROUNDED,
            border_style="blue",
        )
        self.check_results = Text("")
        check_panel = Panel(
            self.check_results,
            title="Checker Results",
            border_style="cyan",
            box=box.ROUNDED,
        )
        self.layout["checks"].update(check_panel)

        self.layout["header"].update(self.header_panel)
        self.layout["footer"].update(
            Panel(
                Text("Press Ctrl+C to cancel", style="dim"),
                box=box.ROUNDED,
                border_style="dim",
            )
        )

        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            TextColumn("[green]{task.fields[info]}"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

        self.stats_table = Table(box=box.SIMPLE)
        self.stats_table.add_column("Statistic", style="cyan")
        self.stats_table.add_column("Value", style="green")

        self.body_layout = Layout()
        self.body_layout.split_row(
            Layout(self.progress, name="progress", ratio=2),
            Layout(self.stats_table, name="stats", ratio=1),
        )

        self.layout["body"].update(self.body_layout)

        default_stats = {
            # Processing stats
            "Samples found": 0,
            "Samples processed": 0,
            "Processing speed": "0 samples/sec",
            "AI Models detected": set(),
            "Sample errors": 0,
            "Extractor errors": 0,
            # Modeling stats
            "MCMC samples": 0,
            "Statistical Models": set(),
            "Num divergents": 0,
            "Checks passed": 0,
            "Checks failed": 0,
            "Errors encountered": 0,
        }

        # Track whether we're loading from a saved state
        self._loaded_from_state = False
        if initial_stats:
            # Merge initial_stats into default_stats, preserving any additional keys
            default_stats.update(initial_stats)
            if "Processing speed" in initial_stats:
                self._loaded_from_state = True

        self.stats = default_stats

        self.start_time = time.time()
        self.live = None
        self._task_ids = {}

        self.modelling = False
        self.original_fori_collect = None
        self.chain_method = "parallel"
        self.num_chains = 1

    def setupt_for_modelling(self):
        """Patch numpyro's fori_collect to integrate with a Rich ModellingDisplay."""
        self.original_fori_collect = patch_fori_collect_with_rich_display(self)
        self.modelling = True

    def set_fit_context(self, chain_method: str, num_chains: int):
        """Set the chain method and number of chains for progress bar display."""
        self.chain_method = chain_method
        self.num_chains = num_chains

    def update_logs(self, log_entry):
        """Callback for LogCaptureHandler to update logs panel."""
        timestamped_entry = (
            f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {log_entry}"
        )
        self.logs.append(timestamped_entry)
        self.all_logs.append(timestamped_entry)
        # Keep only the last N logs for display
        self.logs = self.logs[-self.max_logs :]
        # Update the logs panel
        self.layout["logs"].update(
            Panel("\n".join(self.logs), title="Logger output", border_style="dim white")
        )

    @contextlib.contextmanager
    def capture_logs(self, logger_names=None):
        """Context manager to capture logs from specified loggers."""
        if logger_names is None:
            logger_names = ["jax", "numpyro", "absl", "arviz", None]

        # remove existing handers will add them back in at the end
        existing_handlers = []
        for name in logger_names:
            logger = logging.getLogger(name)
            for handler in logger.handlers:
                existing_handlers.append((logger, handler))
                logger.removeHandler(handler)

        # Create and add handlers to all specified loggers
        handlers = []
        for name in logger_names:
            logger = logging.getLogger(name)
            handler = LogCaptureHandler(self.update_logs)
            logger.addHandler(handler)
            handlers.append((logger, handler))

        try:
            yield
        finally:
            # Remove all handlers
            for logger, handler in handlers:
                logger.removeHandler(handler)

            # restore original handlers
            for logger, handler in existing_handlers:
                logger.addHandler(handler)

    def start(self):
        """Start the live display."""
        self.live = Live(self.layout, refresh_per_second=4)
        self.live.start()
        self._update_stats_table()

    def stop(self):
        """Stop the live display."""
        if self.live:
            self.live.stop()

    @property
    def is_live(self):
        """Check if the live display is active."""
        if not self.live:
            return False
        return self.live.is_started

    def update_header(self, text: str):
        """Update the header text."""
        self.header_panel = Panel(
            Text(self.package_name + " - " + text, style="bold blue"),
            box=box.ROUNDED,
            border_style="blue",
        )
        self.layout["header"].update(self.header_panel)

    def update_stat(self, key: str, value: Any):
        """Update a statistic in the stats table."""
        self.stats[key] = value
        self._update_stats_table()

    def update_stats(self, stats: Dict[str, Any]):
        """Update multiple statistics in the stats table."""
        for key, value in stats.items():
            self.stats[key] = value
        self._update_stats_table()

    def _update_stats_table(self):
        """Update the statistics table with current values."""
        if not self.live:
            return

        # Calculate processing speed if loading data
        # Only calculate if we're actively processing (not loading from a saved state)
        if not self.modelling and self.stats["Samples processed"] > 0:
            if not (hasattr(self, "_loaded_from_state") and self._loaded_from_state):
                elapsed = time.time() - self.start_time
                if elapsed > 0:
                    samples_per_sec = self.stats["Samples processed"] / elapsed
                    self.stats["Processing speed"] = (
                        f"{samples_per_sec:.1f} samples/sec"
                    )

        # Rebuild the table
        self.stats_table = Table(box=box.SIMPLE)
        self.stats_table.add_column("Statistic", style="cyan")
        self.stats_table.add_column("Value", style="green")

        for key, value in self.stats.items():
            self.stats_table.add_row(key, str(value))

        self.body_layout["stats"].update(self.stats_table)

    def add_task(
        self,
        description: str,
        chain: Optional[str] = None,
        worker: Optional[int] = None,
        total: Optional[int] = None,
    ) -> int:
        """Add a new task to the progress display."""
        info = ""
        if chain is not None:
            info = f"Chain {chain}"
        elif worker is not None:
            info = f"Workers: {worker}"

        task_id = self.progress.add_task(description, total=total, info=info)
        self._task_ids[description] = task_id
        self.body_layout["progress"].update(self.progress)
        return task_id

    def update_task(self, description: str, advance: int = 1, **kwargs):
        """Update a task by its description."""
        if description in self._task_ids:
            self.progress.update(self._task_ids[description], advance=advance, **kwargs)
            self.body_layout["progress"].update(self.progress)

    def update_task_description(self, description: str, new_description: str):
        """Update the description of a task."""
        if description in self._task_ids:
            self.progress.update(
                self._task_ids[description], description=new_description
            )
            self._task_ids[new_description] = self._task_ids[description]
            del self._task_ids[description]
            self.body_layout["progress"].update(self.progress)

    def add_plot(
        self,
        series: List[Dict[str, list | str]],
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        title="",
    ):
        """Add a plot to the display."""
        plot = plotextMixin(series, title=title, xlim=xlim, ylim=ylim)
        self.body_layout["progress"].update(plot)
        self.live.update(self.layout)

    def prompt_user(
        self,
        question: str = "Would you like to proceed?",
        options: list = ["yes", "no"],
    ) -> bool:
        """Pause the live display and prompt the user for input."""

        self.layout["footer"].update(
            Panel(
                Text("User input needed", style="bold red"),
                box=box.ROUNDED,
                border_style="bold red",
            )
        )
        self.live.stop()

        user_input = Confirm().ask(
            question,
            show_choices=True,
        )

        self.live.start()

        self.layout["footer"].update(
            Panel(
                Text("Press Ctrl+C to cancel", style="dim"),
                box=box.ROUNDED,
                border_style="dim",
            )
        )

        return user_input

    def add_check(self, check_name: str, result: str) -> None:
        """Add a check result to the display."""
        # Update stats
        if result == "pass":
            symbol = Text(".", style="bold green")
            self.stats["Checks passed"] = self.stats.get("Checks passed", 0) + 1
        elif result == "fail":
            symbol = Text("F", style="bold red")
            self.stats["Checks failed"] = self.stats.get("Checks failed", 0) + 1
            # Log failed check
        elif result == "error":
            symbol = Text("E", style="bold yellow")
            self.stats["Checks errored"] = self.stats.get("Checks errored", 0) + 1
        elif result == "NA":
            symbol = Text("•", style="dim")
        else:
            symbol = Text("?", style="bold magenta")

        self._update_stats_table()

        self.check_results.append(symbol)

        check_panel = Panel(
            self.check_results,
            title="Checker Results",
            border_style="cyan",
            box=box.ROUNDED,
        )
        self.layout["checks"].update(check_panel)

    def get_all_logs(self) -> List[str]:
        """Get all captured logs."""
        return self.all_logs.copy()

    def get_stats_for_persistence(self) -> Dict[str, Any]:
        """Get display statistics in a serialisable format for persistence."""
        # Convert sets to lists for JSON serialization
        stats_copy = {}
        for key, value in self.stats.items():
            if isinstance(value, set):
                stats_copy[key] = list(value)
            else:
                stats_copy[key] = value
        return stats_copy


# Create a custom progress tracking integration for NumPyro's MCMC
class NumPyroRichProgress:
    def __init__(
        self,
        display,
        num_samples=0,
        info_label="Chain 0",
        description="Warming up",
    ):
        """
        Initialise the Rich progress integration for a single chain for NumPyro.

        Args:
            display: The ModellingDisplay instance to update
            description: The task description
            num_samples: Number of samples to run per chain
            info_label: Label to display (e.g. "Chain 0" or "Chains 1-4 (vectorized)")
        """
        self.display = display
        self.num_samples = num_samples
        self.info_label = info_label
        self.task_id = None
        self.description = description

        # Create task in the display
        self.task_id = self.display.add_task(
            description=self._process_description(description),
            chain=self.info_label,
            total=self.num_samples,
        )

        self.current_iter = 0

    def update(self, advance=1, description="Warming up"):
        """Update the progress display"""
        self.current_iter += advance

        self.display.progress.update(
            self.task_id,
            description=self._process_description(description),
            info=self.info_label,
            advance=advance,
            completed=min(self.current_iter, self.num_samples),
            total=self.num_samples,
        )

    def _process_description(self, description: str):
        if description == "sample":
            return "Sampling"
        elif description == "warmup":
            return "Warming up"
        else:
            return description


def rich_progress_bar_factory(
    display, num_samples, num_chains, description_fn
) -> Callable:
    """Factory that builds a the rich progress bar decorator along
    with the `set_description` and `close_pbar` functions
    """
    import jax
    import jax.numpy as jnp

    if num_samples > 20:
        print_rate = int(num_samples / 20)
    else:
        print_rate = 1

    remainder = num_samples % print_rate

    idx_counter = 0  # resource counter to assign chains to progress bars
    rich_bars = {}
    # lock serializes access to idx_counter since callbacks are multithreaded
    # this prevents races that assign multiple chains to a progress bar
    lock = Lock()

    for chain in range(num_chains):
        rich_bars[chain] = NumPyroRichProgress(
            display=display,
            description=description_fn(1),
            num_samples=num_samples,
            info_label=f"{chain + 1}/{num_chains}",
        )

    def _update_pbar(increment, iter_num, chain):
        increment = int(increment)
        chain = int(chain)
        description = description_fn(iter_num)
        if chain == -1:
            nonlocal idx_counter
            with lock:
                chain = idx_counter
                idx_counter += 1
        rich_bars[chain].update(increment, description=description)
        return chain

    def _close_pbar(increment, iter_num, chain):
        increment = int(increment)
        chain = int(chain)
        rich_bars[chain].update(increment, description="Completed")

    def _update_progress_bar(iter_num, chain):
        """Updates tqdm progress bar of a JAX loop only if the iteration number is a multiple of the print_rate
        Usage: carry = progress_bar((iter_num, print_rate), carry)
        """

        chain = jax.lax.cond(
            iter_num == 1,
            lambda _: jax.experimental.io_callback(
                _update_pbar, jnp.array(0), 0, iter_num, chain
            ),
            lambda _: chain,
            operand=None,
        )
        chain = jax.lax.cond(
            iter_num % print_rate == 0,
            lambda _: jax.experimental.io_callback(
                _update_pbar, jnp.array(0), print_rate, iter_num, chain
            ),
            lambda _: chain,
            operand=None,
        )
        _ = jax.lax.cond(
            iter_num == num_samples,
            lambda _: jax.experimental.io_callback(
                _close_pbar, None, iter_num, remainder, chain
            ),
            lambda _: None,
            operand=None,
        )
        return chain

    def progress_bar_fori_loop(func: Callable) -> Callable:
        """Decorator that adds a progress bar to `body_fun` used in `lax.fori_loop`.
        Note that `body_fun` must be looping over a tuple who's first element is `np.arange(num_samples)`.
        This means that `iter_num` is the current iteration number
        """

        def wrapper_progress_bar(i, vals):
            (subvals, chain) = vals
            result = func(i, subvals)
            chain = _update_progress_bar(i + 1, chain)
            return (result, chain)

        return wrapper_progress_bar

    return progress_bar_fori_loop


def patch_fori_collect_with_rich_display(modelling_display):
    """
    Patch NumPyro's fori_collect to integrate with a Rich ModellingDisplay

    Args:
        modelling_display: The ModellingDisplay instance to use for progress
    """
    import jax
    import jax.numpy as jnp
    import numpyro

    # Save original fori_collect for restoration later if needed
    original_fori_collect = numpyro.util.fori_collect

    def patched_fori_collect(
        lower: int,
        upper: int,
        body_fun: Callable,
        init_val: Any,
        transform: Callable = numpyro.util.identity,
        progbar: bool = True,
        return_last_val: bool = False,
        collection_size=None,
        thinning=1,
        **progbar_opts,
    ):
        assert lower <= upper
        assert thinning >= 1
        collection_size = (
            (upper - lower) // thinning if collection_size is None else collection_size
        )
        assert collection_size >= (upper - lower) // thinning
        init_val_transformed = transform(init_val)
        start_idx = lower + (upper - lower) % thinning
        num_chains = progbar_opts.pop("num_chains", 1)
        description = progbar_opts.pop("progbar_desc", lambda x: "")

        @partial(numpyro.util.maybe_jit, donate_argnums=2)
        @numpyro.util.cached_by(patched_fori_collect, body_fun, transform)
        def _body_fn(i, val, collection, start_idx, thinning):
            val = body_fun(val)
            idx = (i - start_idx) // thinning

            def update_fn(collect_array, new_val):
                return numpyro.util.cond(
                    idx >= 0,
                    collect_array,
                    lambda x: x.at[idx].set(new_val),
                    collect_array,
                    numpyro.util.identity,
                )

            def update_collection(collection, val):
                return jax.tree.map(update_fn, collection, transform(val))

            collection = update_collection(collection, val)
            return val, collection, start_idx, thinning

        def map_fn(x):
            nx = jnp.asarray(x)
            return (
                jnp.zeros((collection_size, *nx.shape), dtype=nx.dtype) * nx[None, ...]
            )

        collection = jax.tree.map(map_fn, init_val_transformed)

        if not progbar:

            def loop_fn(collection):
                return numpyro.util.fori_loop(
                    0,
                    upper,
                    lambda i, vals: _body_fn(i, *vals),
                    (init_val, collection, start_idx, thinning),
                )

            last_val, collection, _, _ = numpyro.util.maybe_jit(
                loop_fn, donate_argnums=0
            )(collection)

        elif num_chains > 1:
            progress_bar_fori_loop = rich_progress_bar_factory(
                display=modelling_display,
                num_samples=upper,
                num_chains=num_chains,
                description_fn=description,
            )
            _body_fn_pbar = progress_bar_fori_loop(lambda i, vals: _body_fn(i, *vals))

            def loop_fn(collection):
                return numpyro.util.fori_loop(
                    0,
                    upper,
                    _body_fn_pbar,
                    (
                        (init_val, collection, start_idx, thinning),
                        -1,
                    ),  # -1 for chain id
                )[0]

            last_val, collection, _, _ = numpyro.util.maybe_jit(
                loop_fn, donate_argnums=0
            )(collection)

        else:
            # Single chain or vectorized chains
            chain_method = modelling_display.chain_method
            num_actual_chains = modelling_display.num_chains

            if chain_method == "vectorized" and num_actual_chains > 1:
                info_label = f"Chains 1-{num_actual_chains} (vectorized)"
            else:
                info_label = "Chain 0"

            progbar = NumPyroRichProgress(
                modelling_display,
                description=description(1),
                num_samples=upper,
                info_label=info_label,
            )

            vals = (init_val, collection, jnp.asarray(start_idx), jnp.asarray(thinning))

            if upper == 0:
                # special case, only compiling
                val, collection, start_idx, thinning = vals
                _, collection, _, _ = _body_fn(-1, val, collection, start_idx, thinning)
                vals = (val, collection, start_idx, thinning)
            else:
                for i in range(upper):
                    vals = _body_fn(i, *vals)
                    progbar.update(1, description=description(i))

            last_val, collection, _, _ = vals

        return (collection, last_val) if return_last_val else collection

    # Replace the original function with our patched version
    numpyro.infer.mcmc.fori_collect = patched_fori_collect

    # Return the original function incase it needs be restored if needed
    return original_fori_collect
