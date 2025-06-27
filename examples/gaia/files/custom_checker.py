from typing import Tuple

import numpy as np

from hibayes.check import Checker, CheckerResult, checker
from hibayes.analysis_state import ModelAnalysisState
from hibayes.ui import ModellingDisplay


@checker
def posterior_mean_positive(
    param: str,
    threshold: float = 0.0,
) -> Checker:
    """
    Passes if the posterior mean of param is strictly greater
    than `threshold` (default 0).
    """

    def check(
        state: ModelAnalysisState,
        display: ModellingDisplay | None = None,
    ) -> Tuple[ModelAnalysisState, CheckerResult]:
        data = state.inference_data.get("posterior")[param].values
        mean_ = float(np.mean(data))
        state.add_diagnostic(f"{param}_posterior_mean", mean_)

        if mean_ > threshold:
            return state, "pass"

        msg = (
            f"Posterior mean of '{param}' = {mean_:.4g} "
            f"does not exceed threshold {threshold}."
        )
        if display:
            display.logger.info(msg)
        else:
            print(msg)
        return state, "fail"

    return check
