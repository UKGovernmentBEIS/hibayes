from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd
from rich import box
from rich.table import Table
from rich.text import Text

from ...analysis_state import AnalysisState
from ...ui import ModellingDisplay
from .._communicate import CommunicateResult, communicate


@communicate
def summary_table(
    vars: Optional[List[str]] | None = None,
    *,
    best_model: bool = True,
    round_to: int = 2,
):
    """Create an ArviZ summary table for the selected model(s).

    Uses a cached summary computed by model checkers when available,
    avoiding redundant expensive az.summary() calls.
    """

    def communicate(
        state: AnalysisState,
        display: ModellingDisplay | None = None,
    ) -> Tuple[AnalysisState, CommunicateResult]:
        nonlocal vars

        # Assemble the list of model analyses that will be processed.
        if best_model:
            best = state.get_best_model()
            if best is None:
                raise ValueError(
                    "No best model found – fit a model before calling `summary_table`."
                )
            models_to_run = [best]
        else:
            models_to_run = state.models

        for model_analysis in models_to_run:
            if not model_analysis.is_fitted:
                # Skip unfitted models silently – you may prefer to raise instead.
                continue

            if vars is None:
                vars = model_analysis.model_config.get_plot_params()

            summary_df = model_analysis.get_summary(
                var_names=vars,
                round_to=round_to,
            )

            state.add_table(
                table=summary_df,
                table_name=f"model_{model_analysis.model_name}_summary",
            )

            if display is not None and getattr(display, "is_live", False):
                rich_tbl = _df_to_rich_table(summary_df, round_to=round_to)

                # Replace the entire body area with the table
                display.layout["body"].update(rich_tbl)

        return state, "pass"

    return communicate


def _df_to_rich_table(df: pd.DataFrame, *, round_to: int = 2) -> Table:
    """render the rich table."""
    table = Table(title="Posterior Summary", box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("Variable", style="bold cyan")
    for col in df.columns:
        table.add_column(str(col), style="green", justify="right")

    for idx, row in df.iterrows():
        rendered = [
            f"{val:.{round_to}f}" if isinstance(val, (float, int)) else str(val)
            for val in row
        ]
        table.add_row(Text(str(idx), style="bold cyan"), *rendered)

    return table
