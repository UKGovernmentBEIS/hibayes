from typing import Tuple

import arviz as az
import matplotlib.pyplot as plt

from ...analysis_state import AnalysisState
from ...ui import ModellingDisplay
from .._communicate import CommunicateResult, communicate
from ..utils import drop_not_present_vars


@communicate
def forest_plot(
    vars: list[str] | None = None,
    vertical_line: float | None = None,
    best_model: bool = True,
    figsize: tuple[int, int] = (10, 5),
    transform: bool = False,
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
            model_vars = vars
            if model_analysis.is_fitted:
                model_vars, dropped = (
                    drop_not_present_vars(model_vars, model_analysis.inference_data)
                    if model_vars
                    else (None, None)
                )
                if dropped and display:
                    display.logger.warning(
                        f"Variables {dropped} were not found in the model {model_analysis.model_name} inference data."
                    )
                if model_vars is None:
                    model_vars = model_analysis.model_config.get_plot_params()

                ax = az.plot_forest(
                    model_analysis.inference_data,
                    var_names=model_vars,
                    figsize=figsize,
                    transform=model_analysis.link_function if transform else None,
                    *args,
                    **kwargs,
                )

                if vertical_line is not None:
                    ax[0].axvline(
                        x=vertical_line,
                        color="red",
                        linestyle="--",
                    )
                fig = plt.gcf()
                # add plot to analysis state
                state.add_plot(
                    plot=fig,
                    plot_name=f"model_{model_analysis.model_name}_{'-'.join(model_vars) if model_vars else ''}_forest",
                )
        return state, "pass"

    return communicate


@communicate
def trace_plot(
    vars: list[str] | None = None,
    best_model: bool = True,
    figsize: tuple[int, int] = (10, 5),
    transform: bool = False,
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
            model_vars = vars
            model_vars, dropped = (
                drop_not_present_vars(model_vars, model_analysis.inference_data)
                if model_vars
                else (None, None)
            )

            if dropped and display:
                display.logger.warning(
                    f"Variables {dropped} were not found in the model {model_analysis.model_name} inference data."
                )
            if model_vars is None:
                # best to get all the parameter from the model for the trace plot
                model_vars = list(model_analysis.inference_data.posterior.data_vars)
            if model_analysis.is_fitted:
                az.plot_trace(
                    model_analysis.inference_data,
                    var_names=model_vars,
                    figsize=figsize,
                    transform=model_analysis.link_function if transform else None,
                    *args,
                    **kwargs,
                )
                fig = plt.gcf()

                state.add_plot(
                    plot=fig,
                    plot_name=f"model_{model_analysis.model_name}_{'-'.join(model_vars) if model_vars else ''}_trace",
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
            model_vars = vars
            if model_analysis.is_fitted:
                model_vars, dropped = (
                    drop_not_present_vars(model_vars, model_analysis.inference_data)
                    if model_vars
                    else (None, None)
                )

                if dropped and display:
                    display.logger.warning(
                        f"Variables {dropped} were not found in the model {model_analysis.model_name} inference data."
                    )
                if model_vars is None:
                    model_vars = model_analysis.model_config.get_plot_params()
                az.plot_pair(
                    model_analysis.inference_data,
                    var_names=model_vars,
                    kind="kde",
                    figsize=figsize,
                    *args,
                    **kwargs,
                )
                fig = plt.gcf()
                state.add_plot(
                    plot=fig,
                    plot_name=f"model_{model_analysis.model_name}_{'-'.join(model_vars) if model_vars else ''}_pair",
                )
        return state, "pass"

    return communicate


@communicate
def model_comparison_plot(
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
            *args,
            **kwargs,
        )
        # remove ic from kwargs
        kwargs.pop("ic", None)

        az.plot_compare(comparisons, *args, **kwargs)
        fig = plt.gcf()
        state.add_plot(
            plot=fig,
            plot_name="model_comparison",
        )
        return state, "pass"

    return communicate
