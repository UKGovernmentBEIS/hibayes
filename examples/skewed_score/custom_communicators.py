from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hibayes.analysis_state import AnalysisState
from hibayes.communicate import CommunicateResult, Communicator, communicate
from hibayes.ui import ModellingDisplay


@communicate
def cutpoint_plot(
    figsize: tuple[int, int] = (8, 4),
    title: str = "Cutpoint positions on latent scale",
    save_path: Optional[str] = None,
) -> Communicator:
    """
    Plot cutpoint positions for ordered logistic model.
    """

    def communicate(
        state: AnalysisState,
        display: ModellingDisplay | None = None,
    ) -> Tuple[AnalysisState, CommunicateResult]:
        for model_analysis in state.models:
            if not model_analysis.is_fitted:
                continue

            if "cutpoints" not in model_analysis.inference_data.posterior.data_vars:
                if display:
                    display.logger.warning(
                        "No cutpoints found - skipping cutpoint plot"
                    )
                continue

            # Extract cutpoints
            cutpoints_posterior = model_analysis.inference_data.posterior.cutpoints
            cutpoints_mean = cutpoints_posterior.mean(dim=["chain", "draw"]).values

            fig, ax = plt.subplots(figsize=figsize)

            # Draw horizontal line representing latent scale
            ax.axhline(y=0, color="black", linewidth=2)

            # Plot cutpoints as circles
            ax.scatter(
                cutpoints_mean,
                np.zeros_like(cutpoints_mean),
                s=100,
                color="blue",
                zorder=3,
            )

            # Add cutpoint values
            for i, cp in enumerate(cutpoints_mean):
                ax.text(cp, -0.1, f"{cp:.2f}", ha="center", fontsize=10)
                ax.text(cp, 0.1, f"{i}|{i + 1}", ha="center", fontsize=10)

            # Add score labels
            for i in range(len(cutpoints_mean) + 1):
                if i == 0:
                    position = cutpoints_mean[0] - 2
                elif i == len(cutpoints_mean):
                    position = cutpoints_mean[-1] + 2
                else:
                    position = (cutpoints_mean[i - 1] + cutpoints_mean[i]) / 2

                ax.text(
                    position,
                    0,
                    f"{i}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"),
                )

            # Styling
            ax.set_yticks([])
            ax.set_ylabel("")
            min_x = cutpoints_mean.min() - 3
            max_x = cutpoints_mean.max() + 3
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(-0.5, 0.5)
            ax.set_title(title, fontsize=14)
            ax.set_xlabel("Latent scale", fontsize=12)
            ax.grid(axis="x", alpha=0.3)

            plt.tight_layout()

            state.add_plot(
                plot=fig, plot_name=f"model_{model_analysis.model_name}_cutpoints"
            )

        return state, "pass"

    return communicate


@communicate
def ordered_residuals_plot(
    figsize: tuple[int, int] = (10, 4),
) -> Communicator:
    """
    Plot residuals for ordered logistic model.
    """

    def communicate(
        state: AnalysisState,
        display: ModellingDisplay | None = None,
    ) -> Tuple[AnalysisState, CommunicateResult]:
        for model_analysis in state.models:
            if not model_analysis.is_fitted:
                continue

            if "obs" not in model_analysis.inference_data.posterior.data_vars:
                if display:
                    display.logger.warning(
                        "No posterior predictive samples - skipping residuals plot"
                    )
                continue

            # Calculate residuals
            obs = model_analysis.features["obs"]
            pred = model_analysis.inference_data.posterior.obs.mean(
                dim=["chain", "draw"]
            ).values
            residuals = obs - pred

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

            # Residuals vs fitted
            ax1.scatter(pred, residuals, alpha=0.6)
            ax1.axhline(y=0, color="red", linestyle="--")
            ax1.set_xlabel("Fitted values")
            ax1.set_ylabel("Residuals")
            ax1.set_title("Residuals vs Fitted")

            # Residuals histogram
            ax2.hist(residuals, bins=20, alpha=0.7, edgecolor="black")
            ax2.axvline(x=0, color="red", linestyle="--")
            ax2.set_xlabel("Residuals")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Residuals Distribution")

            plt.tight_layout()

            state.add_plot(
                plot=fig, plot_name=f"model_{model_analysis.model_name}_residuals"
            )

        return state, "pass"

    return communicate


@communicate
def category_frequency_plot(
    figsize: tuple[int, int] = (8, 6),
    show_predictions: bool = True,
) -> Communicator:
    """
    Plot observed vs predicted category frequencies.
    """

    def communicate(
        state: AnalysisState,
        display: ModellingDisplay | None = None,
    ) -> Tuple[AnalysisState, CommunicateResult]:
        for model_analysis in state.models:
            if not model_analysis.is_fitted:
                continue

            obs = model_analysis.features["obs"]

            fig, ax = plt.subplots(figsize=figsize)

            # Observed frequencies
            obs_counts = pd.Series(obs).value_counts().sort_index()
            obs_props = obs_counts / len(obs)

            x_pos = np.arange(len(obs_counts))
            width = 0.35

            _ = ax.bar(
                x_pos - width / 2,
                obs_props,
                width,
                label="Observed",
                alpha=0.7,
                color="blue",
            )

            # Predicted frequencies if available
            if (
                show_predictions
                and "obs"
                in model_analysis.inference_data.posterior_predictive.data_vars
            ):
                pred_samples = (
                    model_analysis.inference_data.posterior_predictive.obs.values
                )
                all_preds = pred_samples.flatten()
                pred_counts = pd.Series(all_preds).value_counts().sort_index()
                pred_props = pred_counts / len(all_preds)

                # Align with observed categories
                pred_props_aligned = pred_props.reindex(obs_counts.index, fill_value=0)

                _ = ax.bar(
                    x_pos + width / 2,
                    pred_props_aligned,
                    width,
                    label="Predicted",
                    alpha=0.7,
                    color="orange",
                )

            ax.set_xlabel("Category")
            ax.set_ylabel("Proportion")
            ax.set_title("Observed vs Predicted Category Frequencies")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(obs_counts.index)
            ax.legend()

            plt.tight_layout()

            state.add_plot(
                plot=fig,
                plot_name=f"model_{model_analysis.model_name}_category_frequencies",
            )

        return state, "pass"

    return communicate
