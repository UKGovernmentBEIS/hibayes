from typing import List, Optional, Tuple

import arviz as az
import jax
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from numpyro.infer import Predictive

from hibayes.ui import display

from ..analysis_state import ModelAnalysisState
from ..ui import ModellingDisplay
from ._check import Checker, CheckerResult, checker


@checker(when="before")
def prior_predictive_check(
    num_samples: int = 1000,
    predictive_kwargs: dict = None,
) -> Checker:
    """
    Check if the prior predictive distribution looks reasonable.

    Args:
        num_samples: Number of prior predictive samples to generate
        predictive_kwargs: Additional keyword arguments to pass to Predictive
    Returns:
        Checker function
    """
    if predictive_kwargs is None:
        predictive_kwargs = {}

    def check(
        state: ModelAnalysisState, display: ModellingDisplay | None = None
    ) -> Tuple[ModelAnalysisState, CheckerResult]:
        rng_key = jax.random.PRNGKey(state.model_config.fit.seed)
        predictive = Predictive(
            state.model,
            num_samples=num_samples,
            parallel=state.platform_config.chain_method != "sequential",
            **predictive_kwargs,
        )
        prior_pred_samples = predictive(rng_key, state.prior_features)
        ds = az.from_numpyro(prior=prior_pred_samples)
        pp_ds = ds.prior

        state.inference_data.add_groups(prior_predictive=pp_ds)
        return state, "NA"  # no automatic pass/fail, user must eyeball

    return check


@checker(when="before")
def prior_predictive_plot(
    variables: Optional[List[str]] = None,
    kind: str = "kde",
    figsize: Tuple[int, int] = (12, 8),
    plot_proportion: bool = False,
    save_path: Optional[str] = None,
    plot_kwargs: Optional[dict] = None,
    predictive_kwargs: Optional[dict] = None,
    interactive: bool = True,
) -> Checker:
    """
    Check by creating prior predictive plots and getting user feedback.

    Args:
        variables: Specific variables to plot. If None, plots all observable variables and main effect params.
        kind: Type of plot ('kde', 'hist', 'cumulative', 'scatter')
        figsize: Figure size
        plot_proportion: If True, plot proportion of successes not counts
        save_path: If provided, save plots to this path (will append variable names)
        plot_kwargs: Additional keyword arguments to pass to az.plot_ppc
        predictive_kwargs: Additional keyword arguments to pass to Predictive
        interactive: If True, prompt user for approval on each plot
    Returns:
        Checker function
    """
    if plot_kwargs is None:
        plot_kwargs = {}
    if predictive_kwargs is None:
        predictive_kwargs = {}

    def check(
        state: ModelAnalysisState, display: ModellingDisplay | None = None
    ) -> Tuple[ModelAnalysisState, CheckerResult]:
        # make sure the prior predictive group exists

        plot_kind = kind
        if state.inference_data.get("prior_predictive") is None:
            state, res = prior_predictive_check(predictive_kwargs=predictive_kwargs)(
                state, display
            )
            if res == "error":
                return state, "error"

        # Handle proportion plotting for prior predictive
        if plot_proportion:
            if "n_total" not in state.prior_features:
                raise ValueError(
                    "Cannot plot proportion without 'n_total' in prior_features."
                )
            # Calculate proportion for prior predictive samples
            pp_ds = state.inference_data.prior_predictive
            if "obs" in pp_ds.data_vars:
                pp_obs = pp_ds["obs"]
                pp_prop = pp_obs / state.prior_features["n_total"]

                # Add proportion to the prior predictive dataset
                state.inference_data.prior_predictive["prop_pred"] = pp_prop

        # Create dummy observed_data if it doesn't exist (required for az.plot_ppc)
        if (
            not hasattr(state.inference_data, "observed_data")
            or state.inference_data.observed_data is None
        ):
            # Create dummy observed data with same structure as prior_predictive but empty
            pp_ds = state.inference_data.prior_predictive
            dummy_observed = {}

            for var_name in pp_ds.data_vars:
                # Create dummy data with same dimensions but just one observation
                var_data = pp_ds[var_name]
                dummy_shape = tuple(
                    1 if dim in ["chain", "draw"] else var_data.sizes[dim]
                    for dim in var_data.dims
                )
                dummy_coords = {
                    dim: var_data.coords[dim] if dim not in ["chain", "draw"] else [0]
                    for dim in var_data.dims
                }

                # Create dummy data filled with NaN (won't be plotted anyway)
                dummy_observed[var_name] = xr.DataArray(
                    np.full(dummy_shape, np.nan),
                    dims=var_data.dims,
                    coords=dummy_coords,
                )

            # Add dummy observed_data to inference_data
            state.inference_data.add_groups(observed_data=xr.Dataset(dummy_observed))

        # Determine which variables to plot
        plot_params = variables
        if plot_params is None:
            if plot_proportion:
                plot_params = ["prop_pred"]
            else:
                plot_params = [
                    v
                    for v in state.inference_data.prior_predictive.data_vars
                    if "obs" in v
                ]

            # Add model parameters if available
            if state.model_config.get_plot_params():
                plot_params.extend(state.model_config.get_plot_params())

        user_ok = True
        for var in plot_params:
            # Check if variable exists in prior_predictive
            if var not in state.inference_data.prior_predictive.data_vars:
                display.logger.warning(
                    f"Warning: Variable '{var}' not found in prior_predictive data"
                )
                continue

            fig, ax = plt.subplots(figsize=figsize)

            try:
                az.plot_ppc(
                    state.inference_data,
                    ax=ax,
                    group="prior",
                    var_names=[var],
                    kind=plot_kind,
                    figsize=figsize,
                    **plot_kwargs,
                )
            except ValueError as e:
                if ("Too many bins for data range" in str(e) or 
                    "inhomogeneous shape" in str(e) or
                    "setting an array element with a sequence" in str(e)):
                    # Fall back to matplotlib histogram when KDE fails
                    display.logger.warning(
                        f"KDE plot failed for '{var}' (likely due to data shape issues or limited data range). "
                        "Using histogram instead."
                    )

                    # Extract prior predictive samples
                    pp_data = state.inference_data.prior_predictive[var].values

                    # Flatten the data if it has multiple dimensions
                    if pp_data.ndim > 1:
                        pp_data = pp_data.flatten()

                    # Remove NaN values if any
                    pp_data = pp_data[~np.isnan(pp_data)]

                    # Create histogram
                    counts, bins, _ = ax.hist(
                        pp_data,
                        bins="auto",
                        alpha=0.7,
                        density=True,
                        edgecolor="black",
                        linewidth=0.5,
                    )
                    ax.set_xlabel(var)
                    ax.set_ylabel("Density")
                    ax.set_title(f"Prior Predictive Distribution: {var}")

                    # Update plot_kind for consistency
                    plot_kind = "histogram"
                else:
                    raise

            if display and interactive:
                if plot_kind == "kde":
                    # Extract data for display
                    pp_data = state.inference_data.prior_predictive[
                        var
                    ].values.flatten()
                    x, y = az.kde(pp_data)
                    display.add_plot({"x": x, "y": y}, title=f"Prior check – {var}")
                elif plot_kind == "histogram":
                    # Use the histogram data from the plot
                    # Create x values as bin centers
                    x = (bins[:-1] + bins[1:]) / 2
                    display.add_plot(
                        {"x": x, "y": counts}, title=f"Prior check – {var} (histogram)"
                    )

                if not display.prompt_user(
                    f"Is the prior predictive distribution for '{np.var}' acceptable?"
                ):
                    user_ok = False
                    break
            # Store diagnostic
            state.add_diagnostic(f"{var}_prior_predictive", fig)

            # Save plot if requested
            if save_path:
                fig.savefig(
                    f"{save_path}_prior_pred_{var}.png",
                    dpi=300,
                    bbox_inches="tight",
                )

        return state, ("pass" if user_ok else "fail")

    return check


@checker
def r_hat(threshold: float = 1.01):
    def check(state: ModelAnalysisState, display: ModellingDisplay = None):
        if "summary" not in state.diagnostics:
            state.diagnostics["summary"] = az.summary(state.inference_data)

        da = state.diagnostics["summary"].r_hat.values
        state.add_diagnostic("r_hat", da)

        if np.all(da < threshold):
            return state, "pass"
        if display:
            display.logger.info(f"High R-hat values: {da}")
        return state, "fail"

    return check


@checker
def ess_bulk(threshold: float = 1_000):
    def check(state: ModelAnalysisState, display: ModellingDisplay = None):
        if "summary" not in state.diagnostics:
            state.diagnostics["summary"] = az.summary(state.inference_data)

        da = state.diagnostics["summary"].ess_bulk.values
        state.add_diagnostic("ess_bulk", da)

        if np.all(da > threshold):
            return state, "pass"
        if display:
            display.logger.info(f"Low ESS-bulk: {da}")
        return state, "fail"

    return check


@checker
def ess_tail(threshold: float = 1_000):
    def check(state: ModelAnalysisState, display: ModellingDisplay = None):
        if "summary" not in state.diagnostics:
            state.diagnostics["summary"] = az.summary(state.inference_data)

        da = state.diagnostics["summary"].ess_tail.values
        state.add_diagnostic("ess_tail", da)

        if np.all(da > threshold):
            return state, "pass"
        if display:
            display.logger.info(f"Low ESS-tail: {da}")
        return state, "fail"

    return check


@checker
def divergences(threshold: int = 0):
    """
    Check the number of divergences in the model.

    Args:
        threshold: Divergence threshold for convergence
    """

    def check(state: ModelAnalysisState, display: ModellingDisplay = None):
        divs = int(state.inference_data.sample_stats["diverging"].values.sum())
        state.add_diagnostic("divergences", divs)

        if divs <= threshold:
            return state, "pass"
        if display:
            display.logger.info(f"{divs} divergences detected.")
        return state, "fail"

    return check


@checker
def loo(
    reff_threshold: float = 0.7,
    scale: str = "log"
) -> Checker:
    """
    Pareto-smoothed importance sampling LOO:
    - computes az.loo(...)
    - flags any high Pareto k > `reff_threshold` as failures
    """

    def check(state: ModelAnalysisState, display: ModellingDisplay = None) -> Tuple[ModelAnalysisState, CheckerResult]:
        if "loo" in state.diagnostics:
            loo_res = state.diagnostics["loo"]
        else:
            loo_res = az.loo(state.inference_data, pointwise=True, scale=scale)

        state.add_diagnostic("loo", loo_res)
        state.add_diagnostic("elpd_loo", loo_res.elpd_loo)
        state.add_diagnostic("se_elpd_loo", loo_res.se)
        state.add_diagnostic("p_loo", loo_res.p_loo)

        bad = int((loo_res.pareto_k.values > reff_threshold).sum())
        if bad == 0:
            return state, "pass"

        if display:
            display.logger.warning(f"{bad} points with Pareto k > {reff_threshold}")
        return state, "fail"

    return check


@checker
def bfmi(threshold: float = 0.20):
    def check(state: ModelAnalysisState, display: ModellingDisplay = None):
        if "potential_energy" not in state.inference_data.sample_stats:
            if display:
                display.logger.info("BFMI skipped: no potential_energy in sample_stats")
            return state, "NA"

        if "bfmi" not in state.diagnostics:
            energy = state.inference_data.sample_stats["potential_energy"].values
            da = az.stats.bfmi(energy)
        else:
            da = state.inference_data["bfmi"]

        state.add_diagnostic("bfmi", da)
        if np.all(da.values() >= threshold):
            return state, "pass"

        if display:
            chains = np.where(da.values() < threshold)[0].tolist()
            display.logger.warning(f"Low BFMI in chains: {chains}")
        return state, "fail"

    return check


@checker
def posterior_predictive_plot(
    num_samples: int = 500,
    kind: str = "kde",
    figsize: Tuple[int, int] = (12, 8),
    plot_proportion: bool = False,
    plot_kwargs: dict = None,
    predictive_kwargs: dict = None,
    interactive: bool = True,
) -> Checker:
    """
    Posterior-predictive check.

    Args:
        num_samples: Number of posterior predictive samples to generate
        kind: Type of plot ('kde', 'hist')
        figsize: Figure size
        plot_proportion: If True, plot proportion of successes not counts
        save_path: If provided, save plot to this path
        plot_kwargs: Additional keyword arguments to pass to plotting functions
        predictive_kwargs: Additional keyword arguments to pass to Predictive
        interactive: If True, prompt user for approval on each plot
    """
    if plot_kwargs is None:
        plot_kwargs = {}
    if predictive_kwargs is None:
        predictive_kwargs = {}

    def _posterior_samples_as_dict(idata: az.InferenceData):
        """Convert `idata.posterior` into the dict format numpyro Predictive expects."""
        dct = {}
        for v in idata.posterior.data_vars:
            arr = idata.posterior[v].values  # (chain, draw, ...)
            dct[v] = arr.reshape(-1, *arr.shape[2:])
        return dct

    def check(
        state: ModelAnalysisState, display: ModellingDisplay | None = None
    ) -> Tuple[ModelAnalysisState, CheckerResult]:
        if not state.inference_data.get("posterior_predictive"):
            rng_key = jax.random.PRNGKey(state.model_config.fit.seed + 1)
            predictive = Predictive(
                state.model,
                posterior_samples=_posterior_samples_as_dict(state.inference_data),
                num_samples=num_samples,
                parallel=False,  # vectorised in mcmc actually samples sequentially as each sample is dependent on the previous one. In Predictive, this is not the case. This can introduce significnat memory requirements. Disabling now to avoid OOM errors.
                **predictive_kwargs,
            )
            pp_samples = predictive(
                rng_key, state.prior_features
            )  # all features but none of the observables!

            # Store as a proper InferenceData group
            pp_ds = az.from_numpyro(posterior_predictive=pp_samples)
            state.inference_data.extend(pp_ds)

        if plot_proportion:
            if "n_total" not in state.features:
                raise ValueError(
                    "Cannot plot proportion without 'n_total' in features."
                )
            # Observed proportion
            obs_prop = state.features["obs"] / state.features["n_total"]

            # Predicted proportion
            pp_obs = state.inference_data.posterior_predictive["obs"]
            pp_prop = pp_obs / state.features["n_total"]

            # Attach to InferenceData
            if "observed_data" in state.inference_data.groups():
                state.inference_data.observed_data["prop_pred"] = (
                    ("obs_dim_0",),
                    obs_prop,
                )
            else:
                state.inference_data.add_groups(
                    observed_data={"prop_pred": (("obs_dim_0",), obs_prop)},
                    inplace=True,
                )

            state.inference_data.posterior_predictive["prop_pred"] = pp_prop

        fig, ax = plt.subplots(figsize=figsize)
        if kind == "kde":
            # Combine plot_kwargs with defaults for plot_ppc
            az.plot_ppc(
                state.inference_data,
                ax=ax,
                var_names=["prop_pred"] if plot_proportion else ["obs"],
                **plot_kwargs,
            )
            if display and interactive:
                lines = ax.get_lines()
                obs_line = next(
                    line for line in lines if line.get_label() == "Observed"
                )
                posterior_line = lines[
                    -1
                ]  # the mean of the posterior predictive is the last line

                display.add_plot(
                    [
                        {
                            "x": posterior_line.get_xdata(),
                            "y": posterior_line.get_ydata(),
                            "label": "posterior-pred",
                        },
                        {
                            "x": obs_line.get_xdata(),
                            "y": obs_line.get_ydata(),
                            "label": "observed",
                        },
                    ],
                    ylim=(
                        ax.get_ylim()[0],
                        ax.get_ylim()[1] * 1.2,
                    ),  # annoyingly you cannot move the legend position in plottext....https://github.com/piccolomo/plotext/issues/220
                )
                ok = display.prompt_user(
                    "Does the posterior-predictive distribution align with the data?"
                )
                verdict = "pass" if ok else "fail"
            else:
                verdict = "NA"
        else:  # hist
            # Combine plot_kwargs with hist defaults for posterior
            sims = np.asarray(pp_samples["obs"]).ravel()
            obs = np.asarray(state.features["obs"]).ravel()
            post_hist_kwargs = {"bins": 30, "alpha": 0.7, "label": "posterior-pred"}
            post_hist_kwargs.update(plot_kwargs)
            ax.hist(sims, **post_hist_kwargs)

            # Create a separate dict for observed data histogram
            obs_hist_kwargs = post_hist_kwargs.copy()
            obs_hist_kwargs["alpha"] = 0.5
            obs_hist_kwargs["label"] = "observed"
            ax.hist(obs, **obs_hist_kwargs)
            verdict = "NA"

        ax.set_title("Posterior-predictive vs observed")
        ax.legend()
        state.add_diagnostic("posterior_predictive_plot", fig)

        return state, verdict

    return check


@checker
def waic(scale: str = "log") -> Checker:
    """
    Compute the Widely Applicable Information Criterion (WAIC).

    Args:
        scale: Output scale for WAIC; one of "deviance" or "log" (default).
    Returns:
        Checker function that computes WAIC and records it for inspection.
    """

    def check(state: ModelAnalysisState, display: ModellingDisplay | None = None):
        # avoid recomputing if we already did
        if "waic" in state.diagnostics:
            waic_res = state.diagnostics["waic"]
        else:
            waic_res = az.waic(state.inference_data, scale=scale)
        
        state.add_diagnostic("waic", waic_res)
        state.add_diagnostic("elpd_waic", waic_res.elpd_waic)
        state.add_diagnostic("se_elpd_waic", waic_res.se)
        state.add_diagnostic("p_waic", waic_res.p_waic)

        if display:
            display.logger.info(
                f"elpd_waic: {waic_res['elpd_waic']:.2f} ± {waic_res['se']:.2f}, "
                f"p_waic: {waic_res['p_waic']:.2f}"
            )
        return state, "NA"

    return check
