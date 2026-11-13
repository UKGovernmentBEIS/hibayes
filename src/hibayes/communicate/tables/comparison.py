"""Answer-shaped summary for ``accuracy_comparison`` fits."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from ...analysis_state import AnalysisState
from ...ui import Display
from .._communicate import CommunicateResult, Communicator, communicate
from ..utils import format_probability, resolve_models_to_run


def comparison_dims(posterior) -> Tuple[str, str, str | None] | None:
    """
    ``(model_dim, contrast_dim, group_dim)`` of an ``accuracy_comparison``
    posterior, or None if the posterior is not from one.
    """
    if "accuracy_difference" not in posterior or "accuracy" not in posterior:
        return None
    extra = [
        d for d in posterior["accuracy_difference"].dims if d not in ("chain", "draw")
    ]
    contrast = next((d for d in extra if d.endswith("_contrast")), None)
    if contrast is None:
        return None
    model_dim = contrast[: -len("_contrast")]
    if model_dim not in posterior["accuracy"].dims:
        return None
    group = next((d for d in extra if d != contrast), None)
    return model_dim, contrast, group


@communicate
def comparison_summary(
    credible_mass: float = 0.95,
    practical_difference: float = 0.01,
    *,
    best_model: bool = False,
) -> Communicator:
    """
    State a model comparison as an answer.

    Category: Comparison

    For every ``accuracy_comparison`` fit: accuracy of the reference and the
    compared model, the accuracy difference (compared minus reference) with
    an equal-tailed credible interval, the probability the compared model is
    better, and the probability the two are practically equivalent. One row
    per fit, group (e.g. benchmark) and compared model, so fits from
    different steps of a build-up can be read side by side. Grouped fits
    also get an "average over groups" row.

    Probabilities of exactly 0 or 1 among the draws are shown as ``< 0.001``
    or ``> 0.999``: the draws can't resolve anything smaller.

    Args:
        credible_mass: Probability mass of the credible interval.
        practical_difference: Accuracy differences smaller than this in
            absolute value (0.01 = one percentage point) count as practically
            equivalent.
        best_model: Summarise only the best model by WAIC. Defaults to False,
            since comparing fits side by side is the point.
    """
    if not 0 < credible_mass < 1:
        raise ValueError("credible_mass must be between 0 and 1")
    tail = (1 - credible_mass) / 2
    pct = f"{credible_mass:.0%}"

    def communicate(
        state: AnalysisState,
        display: Display | None = None,
    ) -> Tuple[AnalysisState, CommunicateResult]:
        rows = []
        for model_analysis in resolve_models_to_run(state, best_model, display):
            if not model_analysis.is_fitted:
                continue
            post = model_analysis.inference_data.posterior
            found = comparison_dims(post)
            if found is None:
                continue
            model_dim, contrast_dim, group_dim = found
            acc = post["accuracy"].stack(sample=("chain", "draw"))
            diff = post["accuracy_difference"].stack(sample=("chain", "draw"))
            models = [str(m) for m in acc[model_dim].values]
            reference = models[0]
            groups = list(acc[group_dim].values) if group_dim else [None]

            def row(group, other, acc_ref, acc_other, d):
                d = np.asarray(d)
                lo, hi = np.quantile(d, [tail, 1 - tail])
                return {
                    "fit": model_analysis.model_name,
                    "group": "" if group is None else str(group),
                    "reference": reference,
                    "compared": other,
                    "accuracy reference": float(np.mean(acc_ref)),
                    "accuracy compared": float(np.mean(acc_other)),
                    "difference (mean)": float(d.mean()),
                    f"difference {pct} lower": float(lo),
                    f"difference {pct} upper": float(hi),
                    "interval width": float(hi - lo),
                    "P(compared better)": format_probability(d > 0),
                    f"P(|difference| < {practical_difference:g})": format_probability(
                        np.abs(d) < practical_difference
                    ),
                }

            for k, other in enumerate(models[1:]):
                for group in groups:
                    sel = {} if group is None else {group_dim: group}
                    a = acc.sel(sel)
                    rows.append(
                        row(
                            group,
                            other,
                            a.isel({model_dim: 0}),
                            a.isel({model_dim: k + 1}),
                            diff.sel(sel).isel({contrast_dim: k}),
                        )
                    )
                if group_dim is not None:
                    rows.append(
                        row(
                            "average over groups",
                            other,
                            acc.isel({model_dim: 0}).mean(group_dim),
                            acc.isel({model_dim: k + 1}).mean(group_dim),
                            diff.isel({contrast_dim: k}).mean(group_dim),
                        )
                    )

        if not rows:
            if display:
                display.logger.warning(
                    "comparison_summary: no fitted accuracy_comparison model; skipping."
                )
            return state, "NA"

        table = pd.DataFrame(rows).round(4)
        state.add_table(table, "comparison_summary")
        if display:
            for r in rows:
                where = f" on {r['group']}" if r["group"] else ""
                display.logger.info(
                    f"[{r['fit']}]{where}: {r['compared']} minus {r['reference']} = "
                    f"{r['difference (mean)']:+.3f} ({pct} CI "
                    f"{r[f'difference {pct} lower']:+.3f} to "
                    f"{r[f'difference {pct} upper']:+.3f}), "
                    f"P(better) = {r['P(compared better)']}"
                )
        return state, "pass"

    return communicate
