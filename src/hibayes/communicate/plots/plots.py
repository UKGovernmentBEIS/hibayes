from typing import Tuple

import arviz as az
import matplotlib.pyplot as plt

from ...analysis_state import AnalysisState
from ...ui import ModellingDisplay
from .._communicate import CommunicateResult, communicate


@communicate
def forest_plot(
    vars: list[str] | None = None,
    best_model: bool = True,
    figsize: tuple[int, int] = (10, 5),
    *args,
    **kwargs,
):
    def communicate(
        state: AnalysisState,
        display: ModellingDisplay | None = None,
    ) -> Tuple[AnalysisState, CommunicateResult]:
        """
        Communicate the results of a model analysis.
        """
        nonlocal vars
        if best_model:
            # Get the model which has the best fit based on the model fit criteria.
            best_model_analysis = state.get_best_model()
            if best_model_analysis is None:
                raise ValueError("No best model found.")
            models_to_run = [best_model_analysis]
        else:
            models_to_run = state.models

        for model_analysis in models_to_run:
            if model_analysis.is_fitted:
                if vars is None:
                    vars = model_analysis.model_builder.config.main_effect_params
                az.plot_forest(
                    model_analysis.inference_data,
                    var_names=vars,
                    figsize=figsize,
                    *args,
                    **kwargs,
                )
                fig = plt.gcf()
                # add plot to analysis state
                state.add_plot(
                    plot=fig,
                    plot_name=f"model_{model_analysis.model_name}_forest",
                )
        return state, "pass"

    return communicate


@communicate
def trace_plot(
    vars: list[str] | None = None,
    best_model: bool = True,
    figsize: tuple[int, int] = (10, 5),
    *args,
    **kwargs,
):
    def communicate(
        state: AnalysisState,
        display: ModellingDisplay | None = None,
    ) -> Tuple[AnalysisState, CommunicateResult]:
        """
        Communicate the trace plots for each model's inference data.
        """
        nonlocal vars
        if best_model:
            best_model_analysis = state.get_best_model()
            if best_model_analysis is None:
                raise ValueError("No best model found.")
            models_to_run = [best_model_analysis]
        else:
            models_to_run = state.models

        for model_analysis in models_to_run:
            if vars is None:
                # best to get all the parameter from the model for the trace plot
                vars = list(model_analysis.inference_data.posterior.data_vars)
            if model_analysis.is_fitted:
                az.plot_trace(
                    model_analysis.inference_data,
                    var_names=vars,
                    figsize=figsize,
                    *args,
                    **kwargs,
                )
                fig = plt.gcf()

                state.add_plot(
                    plot=fig,
                    plot_name=f"model_{model_analysis.model_name}_trace",
                )
        return state, "pass"

    return communicate


@communicate
def pair_plot(
    vars: list[str] | None = None,
    best_model: bool = True,
    figsize: tuple[int, int] = (10, 10),
    *args,
    **kwargs,
):
    def communicate(
        state: AnalysisState,
        display: ModellingDisplay | None = None,
    ) -> Tuple[AnalysisState, CommunicateResult]:
        """
        Communicate pairwise relationships (e.g., KDE) among variables.
        """
        nonlocal vars
        if best_model:
            best_model_analysis = state.get_best_model()
            if best_model_analysis is None:
                raise ValueError("No best model found.")
            models_to_run = [best_model_analysis]
        else:
            models_to_run = state.models

        for model_analysis in models_to_run:
            if model_analysis.is_fitted:
                if vars is None:
                    vars = model_analysis.model_builder.config.main_effect_params
                az.plot_pair(
                    model_analysis.inference_data,
                    var_names=vars,
                    kind="kde",
                    figsize=figsize,
                    *args,
                    **kwargs,
                )
                fig = plt.gcf()
                state.add_plot(
                    plot=fig,
                    plot_name=f"model_{model_analysis.model_name}_pair",
                )
        return state, "pass"

    return communicate


@communicate
def model_comparison_plot(
    method: str = None,
    figsize: tuple[int, int] = (10, 5),
    *args,
    **kwargs,
):
    def communicate(
        state: AnalysisState,
        display: ModellingDisplay | None = None,
    ) -> Tuple[AnalysisState, CommunicateResult]:
        """
        Compare models using specified method (e.g., LOO, WAIC) and plot results.
        """

        # Gather inference data for all fitted models
        data_dict = {
            ma.model_name: ma.inference_data for ma in state.models if ma.is_fitted
        }
        if not data_dict:
            display.logger.warning(
                "No fitted models available for comparison. Please fit models first."
            )
            return state, "Error"

        if len(data_dict) == 1:
            display.logger.warning(
                "Only one model available for comparison. No comparison will be made."
            )
            return state, "Error"

        comparisons = az.compare(
            data_dict,
            method=method,
            *args,
            **kwargs,
        )

        az.plot_compare(comparisons, figsize=figsize, *args, **kwargs)
        fig = plt.gcf()
        state.add_plot(
            plot=fig,
            plot_name="model_comparison",
        )
        plt.savefig(
            "model_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        return state, "pass"

    return communicate
