from typing import TYPE_CHECKING, List, Tuple

from arviz import InferenceData

if TYPE_CHECKING:
    from ..analysis_state import AnalysisState, ModelAnalysisState
    from ..ui import Display


def resolve_models_to_run(
    state: "AnalysisState",
    best_model: bool,
    display: "Display | None" = None,
) -> List["ModelAnalysisState"]:
    """
    Resolve which models a communicator should run over.

    When ``best_model`` is False, all models are returned. When True, the best
    model (per ``state.get_best_model()``) is returned. If only one fitted model
    exists it is used directly (best of one is unambiguous). If no model has the
    required diagnostic (e.g. the ``waic`` checker was not run), a warning is
    logged and all models are returned rather than raising.
    """
    if not best_model:
        return state.models

    fitted_models = [model for model in state.models if model.is_fitted]
    if len(fitted_models) == 1:
        return fitted_models

    try:
        return [state.get_best_model()]
    except ValueError:
        if display is not None:
            display.logger.warning(
                "best_model=True but no model has the 'elpd_waic' diagnostic needed "
                "to pick a best model. Enable the 'waic' checker or set "
                "'best_model: false' in your communicator config. Falling back to "
                "running over all models."
            )
        return state.models


def drop_not_present_vars(
    vars: List[str],
    inference_data: InferenceData,
) -> Tuple[List[str], List[str]]:
    """
    Drop variables that are not present in the inference data. Also return the
    list of variables which were dropped.
    """
    present_vars = [var for var in vars if var in inference_data.posterior.data_vars]
    dropped_vars = [var for var in vars if var not in present_vars]
    return present_vars, dropped_vars


def format_probability(events) -> str:
    """
    Share of posterior draws where an event holds, as text.

    With N draws the share can't resolve probabilities below about 1/N, so a
    share of exactly 0 or 1 is reported as ``"< 0.001"`` or ``"> 0.999"``
    (for N >= 1000; ``1/N`` otherwise) rather than as an impossible
    certainty.

    Args:
        events: Boolean array, one entry per posterior draw.
    """
    import numpy as np

    events = np.asarray(events)
    if events.size == 0:
        raise ValueError("format_probability needs at least one draw.")
    p = float(events.mean())
    floor = max(0.001, 1 / events.size)
    if p == 0:
        return f"< {floor:g}"
    if p == 1:
        return f"> {1 - floor:g}"
    return f"{p:.3f}"
