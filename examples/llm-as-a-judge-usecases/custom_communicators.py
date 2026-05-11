from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import krippendorff
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpyro.infer import Predictive

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


# Krippendorff's alpha plot


def _finite(arr: np.ndarray) -> np.ndarray:
    """Return finite values only (drop NaN/Inf)."""
    arr = np.asarray(arr).astype(float)
    return arr[np.isfinite(arr)]


def _mean_ci(x: np.ndarray, ci: float = 0.95) -> tuple[float, tuple[float, float]]:
    """Mean and central CI from samples (finite only)."""
    x = _finite(x)
    if x.size == 0:
        return np.nan, (np.nan, np.nan)
    lo = (1 - ci) / 2 * 100
    hi = (1 + ci) / 2 * 100
    return float(np.mean(x)), (float(np.percentile(x, lo)), float(np.percentile(x, hi)))


def plot_alpha_comparison(
    alpha_observed: float | None,
    series: Sequence[tuple[str, np.ndarray]],
    *,
    title: str = "Comparison of Krippendorff's α",
    xlabel: str = "Krippendorff's α",
    figsize: tuple[int, int] = (10, 4),
    ci_level: float = 0.95,
    ref_lines: Sequence[tuple[float, str]] = (
        (0.0, "α = 0 (random agreement)"),
        (0.4, "α = 0.4 (moderate agreement)"),
    ),
    violin_width: float = 0.5,
    violin_alpha: float = 0.75,
    point_size_observed: int = 240,
    point_size_series: int = 90,
    line_width_series: int = 2,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """

    Args:
    alpha_observed : float | None
        If provided, plotted as a single point above the violins.
    series : sequence of (label, samples)
        Each element is a tuple: label string and 1D array of posterior α samples.
    ci_level : float
        Central credible interval for series (default 0.95).
    ref_lines : sequence of (x, label)
        Vertical reference lines with legend entries.

    """
    n = len(series)
    y_positions = []
    y_labels = []

    top_index = n  # observed goes at y = n (top), violins at y = 0..n-1

    fig, ax = plt.subplots(figsize=figsize)

    # Color cycle for violins/points
    color_cycle = (
        plt.rcParams["axes.prop_cycle"]
        .by_key()
        .get("color", ["C0", "C1", "C2", "C3", "C4", "C5"])
    )

    # 1) Observed point (optional)
    x_all = []
    if alpha_observed is not None and np.isfinite(alpha_observed):
        ax.scatter(
            alpha_observed, top_index, marker="X", s=point_size_observed, zorder=10
        )
        y_positions.append(top_index)
        y_labels.append(f"Observed (α = {alpha_observed:.3f})")
        x_all.append(alpha_observed)

    # 2) Violins for each series (bottom to top)
    for i, (label, samples) in enumerate(series):
        samples = _finite(np.asarray(samples))
        ypos = i  # stack from bottom upwards
        if samples.size > 0:
            parts = ax.violinplot(
                [samples],
                positions=[ypos],
                vert=False,
                widths=violin_width,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            # Style the single body
            body = parts["bodies"][0]
            col = color_cycle[i % len(color_cycle)]
            body.set_facecolor(col)
            body.set_edgecolor(col)
            body.set_alpha(violin_alpha)

            # Mean & CI bar
            m, (lo, hi) = _mean_ci(samples, ci=ci_level)
            if np.isfinite(m):
                ax.scatter(m, ypos, s=point_size_series, zorder=9)
            if np.isfinite(lo) and np.isfinite(hi):
                ax.plot(
                    [lo, hi],
                    [ypos, ypos],
                    linewidth=line_width_series,
                    solid_capstyle="round",
                    alpha=0.5,
                )

            y_positions.append(ypos)
            y_labels.append(
                f"{label} (mean = {m:.3f})"
                if np.isfinite(m)
                else f"{label} (mean = NA)"
            )

            x_all.extend(samples.tolist())
            if np.isfinite(m):
                x_all.append(m)
                x_all.extend([lo, hi])

        else:
            # Empty/NaN-only series: still reserve a line with a note
            ax.text(
                0.01,
                ypos,
                "No finite samples",
                va="center",
                fontsize=9,
                transform=ax.get_yaxis_transform(),
            )
            y_positions.append(ypos)
            y_labels.append(f"{label} (no finite samples)")

    # 3) Reference lines
    for xref, lab in ref_lines:
        if np.isfinite(xref):
            ax.axvline(xref, linestyle="--", alpha=0.5, label=lab)

    # Axis labels and ticks
    ax.set_xlabel(xlabel)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_title(title)

    # X limits with padding
    x_all = _finite(np.asarray(x_all))
    if x_all.size > 0:
        x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
        if np.isfinite(x_min) and np.isfinite(x_max) and x_min != x_max:
            pad = 0.15 * (x_max - x_min)
            ax.set_xlim(x_min - pad, x_max + pad)
        else:
            ax.set_xlim(-1.0, 1.0)  # safe default for α
    else:
        ax.set_xlim(-1.0, 1.0)

    # Y limits to fit all rows
    if alpha_observed is not None and np.isfinite(alpha_observed):
        ax.set_ylim(-0.5, n + 0.5)
    else:
        ax.set_ylim(-0.5, max(n - 1, 0) + 0.5)

    ax.grid(axis="x", alpha=0.3)
    # Only show legend if there are reference lines with labels
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def compute_krippendorff(df, values_col: str = "score") -> float:
    """Krippendorff's alpha on observed ordinal ratings (grader x item)."""

    matrix = df.pivot_table(
        index="grader", columns="item", values=values_col, aggfunc="median"
    )  # we are taking the median as we have multiple ratings per grader-item pair. Noting that this will result in less variance.
    matrix_array = matrix.values.astype(float)
    return krippendorff.alpha(matrix_array, level_of_measurement="ordinal")


def compute_bias_corrected_alpha(
    posterior_sample_idx,
    df,
    idata,
    features,
    model,
    parameters_to_zero=None,
    parameter_values=None,
):
    """
    Compute a after optionally modifying parameter values in posterior samples.

    Args:
    posterior_sample_idx : int
        Index of the posterior sample to use
    df : pandas.DataFrame
        Data frame with grader and item columns
    idata : arviz.InferenceData
        Posterior samples from the model
    features: dict
        Features used for predictions
    model : callable
        The model function used for predictions
    parameters_to_zero : list of str, optional
        List of parameter names to set to zero. Default: ['b_grader', 'b_grader_full']
    parameter_values : dict, optional
        Dictionary mapping parameter names to specific values to set them to.
        Takes precedence over parameters_to_zero.

    Returns:
    float
        Krippendorff's alpha coefficient
    """
    # Default parameters to zero out (grader main effects)
    if parameters_to_zero is None:
        return None

    if parameter_values is None:
        parameter_values = {}

    # Extract single sample from posterior
    single_sample = extract_posterior_sample(
        idata, posterior_sample_idx, parameters_to_zero, parameter_values
    )

    # Generate counterfactual predictions
    predicted_scores = generate_predictions(
        single_sample,
        posterior_sample_idx,
        features,
        model,
    )

    df["score_pred"] = predicted_scores

    return compute_krippendorff(df, "score_pred")


def extract_posterior_sample(idata, sample_idx, parameters_to_zero, parameter_values):
    """Extract a single sample from posterior, with optional parameter modifications."""
    single_sample = {}

    for param in idata.posterior.data_vars:
        param_data = idata.posterior[param].values

        # Check if we need to set this parameter to a specific value
        if param in parameter_values:
            single_sample[param] = set_parameter_value(
                param_data[0, sample_idx], parameter_values[param]
            )
        # Check if we need to zero out this parameter
        elif param in parameters_to_zero:
            single_sample[param] = zero_parameter(param_data[0, sample_idx])
        # Otherwise, extract the parameter value normally
        else:
            single_sample[param] = extract_parameter(param_data, sample_idx)

    return single_sample


def set_parameter_value(param_shape_reference, value):
    """Set parameter to a specific value, preserving shape."""
    if hasattr(param_shape_reference, "shape"):
        return jnp.full_like(param_shape_reference, value)
    else:
        return value


def zero_parameter(param_data):
    """Zero out a parameter, preserving its shape."""
    if hasattr(param_data, "shape"):
        return jnp.zeros_like(param_data)
    else:
        return 0.0


def extract_parameter(param_data, sample_idx):
    """Extract parameter value based on its dimensionality."""
    # param_data shape is (chain, sample, *param_dims)
    # We always use chain 0 and the specified sample
    if param_data.ndim == 2:  # (chain, sample)
        return param_data[0, sample_idx]
    elif param_data.ndim == 3:  # (chain, sample, param)
        return param_data[0, sample_idx, :]
    elif param_data.ndim == 4:  # (chain, sample, param1, param2)
        return param_data[0, sample_idx, :, :]
    else:
        raise ValueError(f"Unexpected parameter dimensionality: {param_data.ndim}")


def generate_predictions(single_sample, sample_idx, features, model):
    """Generate counterfactual predictions using the posterior sample."""
    rng_key = jax.random.PRNGKey(42 + sample_idx)

    predictive = Predictive(
        model,
        posterior_samples={k: jnp.array([v]) for k, v in single_sample.items()},
        return_sites=["obs"],
        parallel=False,  # see note in posterior_predictive_plot
    )

    counterfactual_samples = predictive(rng_key, features)

    return counterfactual_samples["obs"][0]


@communicate
def krippendorff_alpha(
    *,
    max_draws_per_model: int = 1000,
    ci_level: float = 0.95,
    figsize: tuple[int, int] = (10, 4),
    title_prefix: str = "Krippendorff's α (observed vs posterior): ",
) -> Communicator:
    """
    Compute observed α and per-model posterior α distributions, and plot a separate
    figure for each model.
    """

    def communicate(
        state: AnalysisState,
        display: ModellingDisplay | None = None,
    ) -> Tuple[AnalysisState, CommunicateResult]:
        # Observed / "classical" α from raw data (shared across figs)
        classical_alpha = compute_krippendorff(state.processed_data)

        for model_analysis in state.models:
            if not model_analysis.is_fitted:
                continue

            if not model_analysis.inference_data.get("posterior_predictive"):
                if display:
                    display.logger.warning(
                        f"No posterior predictive samples for model {model_analysis.model_name} - skipping Krippendorff's alpha for this model. "
                        "If you want this, run with posterior predictive sampling enabled."
                    )
                continue

            # get alpha from posterior draws

            pred = (
                model_analysis.inference_data.posterior_predictive.obs.values
            )  # (chain, draw, N)
            draws_flat = pred.reshape(-1, pred.shape[-1])  # (chain*draw, N)

            if draws_flat.shape[1] != len(state.processed_data):
                if display:
                    display.logger.error(
                        f"Predictions length ({draws_flat.shape[1]}) does not match data length ({len(state.processed_data)}) "
                        f"for model {model_analysis.model_name}."
                    )
                continue

            # Subsample draws for speed/reasonable CI
            rng = np.random.default_rng(0)
            take = min(max_draws_per_model, draws_flat.shape[0])
            idx = rng.choice(draws_flat.shape[0], size=take, replace=False)

            alphas = []
            for i in idx:  # TODO this should be vectorised!
                df_i = state.processed_data[["grader", "item"]].copy()
                df_i["score_pred"] = draws_flat[i]
                a = compute_krippendorff(df_i, values_col="score_pred")
                if np.isfinite(a):
                    alphas.append(float(a))
            alphas = np.asarray(alphas, dtype=float)

            # now lets explore the counterfactual! see paper for more details
            # here we set the systematic grader effects to zero
            take = min(
                max_draws_per_model,
                model_analysis.inference_data.posterior.sizes["draw"],
            )
            idx = rng.choice(
                model_analysis.inference_data.posterior.sizes["draw"],
                size=take,
                replace=False,
            )
            bias_corrected_alphas = []
            for i in idx:
                alpha = compute_bias_corrected_alpha(
                    posterior_sample_idx=i,
                    df=state.processed_data[["grader", "item"]].copy(),
                    idata=model_analysis.inference_data,
                    features=model_analysis.prior_features,  # no observed features
                    model=model_analysis.model,
                    parameters_to_zero=["grader_effects", "grader_effects_constrained"],
                )

                if np.isfinite(alpha):
                    bias_corrected_alphas.append(float(alpha))

            # Build the per-model figure: observed point + single violin for this model
            fig = plot_alpha_comparison(
                alpha_observed=classical_alpha,
                series=[
                    (f"{model_analysis.model_name} posterior α", alphas),
                    (
                        f"{model_analysis.model_name} bias-corrected α",
                        bias_corrected_alphas,
                    ),
                ],
                title=title_prefix + model_analysis.model_name,
                xlabel="Krippendorff's α",
                figsize=figsize,
                ci_level=ci_level,
            )

            state.add_plot(
                plot=fig,
                plot_name=f"krippendorff_alpha_{model_analysis.model_name}",
            )

        return state, "pass"

    return communicate
