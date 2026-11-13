from typing import List, Optional, Tuple

import arviz as az
import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from numpyro.infer import Predictive


from ..analysis_state import ModelAnalysisState
from ..ui import ModellingDisplay
from ._check import Checker, CheckerResult, checker


def _display_can_prompt(display) -> bool:
    """Whether the display can actually prompt the user for input.

    Displays expose ``can_prompt()``; custom displays which do not implement it
    are assumed to be able to prompt (previous behaviour).
    """
    can_prompt = getattr(display, "can_prompt", None)
    if callable(can_prompt):
        return bool(can_prompt())
    return True


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

        prompt_unavailable = False
        if interactive and display and not _display_can_prompt(display):
            prompt_unavailable = True
            display.logger.warning(
                "prior_predictive_plot: interactive approval was requested but no "
                "interactive session is available to prompt through. Skipping the "
                "user prompt and returning 'NA'. Review the saved plots offline, "
                "or set 'interactive: false' to silence this warning."
            )

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
                if (
                    "Too many bins for data range" in str(e)
                    or "inhomogeneous shape" in str(e)
                    or "setting an array element with a sequence" in str(e)
                ):
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

            if display and interactive and not prompt_unavailable:
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
                    f"Is the prior predictive distribution for '{var}' acceptable?"
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

        if prompt_unavailable:
            return state, "NA"
        return state, ("pass" if user_ok else "fail")

    return check


@checker
def r_hat(threshold: float = 1.01):
    """
    Check the R-hat convergence diagnostic for all parameters.

    Category: MCMC convergence

    R-hat (potential scale reduction factor) measures convergence by comparing
    between-chain and within-chain variance. Values close to 1.0 indicate
    convergence; values above the threshold suggest the chains have not converged.

    Computed using the rank-normalized split R-hat from Vehtari et al. (2021):
    https://arxiv.org/abs/1903.08008

    Args:
        threshold: Maximum acceptable R-hat value. Default is 1.01.

    Returns:
        Checker function that passes if all R-hat values are below threshold.
    """

    def check(state: ModelAnalysisState, display: ModellingDisplay = None):
        summary = state.get_summary()
        da = summary.r_hat.values
        state.add_diagnostic("r_hat", da)

        if np.all(da < threshold):
            return state, "pass"
        if display:
            display.logger.info(f"High R-hat values: {da}")
        return state, "fail"

    return check


@checker
def ess_bulk(threshold: float = 1_000):
    """
    Check the bulk effective sample size (ESS) for all parameters.

    Category: MCMC convergence

    Bulk ESS estimates the effective number of independent samples for
    estimating the mean of the distribution. Low values indicate high
    autocorrelation and unreliable posterior estimates.

    Computed using the rank-normalized ESS from Vehtari et al. (2021):
    https://arxiv.org/abs/1903.08008

    Args:
        threshold: Minimum acceptable bulk ESS. Default is 1000.

    Returns:
        Checker function that passes if all bulk ESS values exceed threshold.
    """

    def check(state: ModelAnalysisState, display: ModellingDisplay = None):
        summary = state.get_summary()
        da = summary.ess_bulk.values
        state.add_diagnostic("ess_bulk", da)

        if np.all(da > threshold):
            return state, "pass"
        if display:
            display.logger.info(f"Low ESS-bulk: {da}")
        return state, "fail"

    return check


@checker
def ess_tail(threshold: float = 1_000):
    """
    Check the tail effective sample size (ESS) for all parameters.

    Category: MCMC convergence

    Tail ESS estimates the effective number of independent samples for
    estimating tail quantiles (e.g., 5th and 95th percentiles). Low values
    indicate unreliable credible interval estimates.

    Computed using the rank-normalized ESS from Vehtari et al. (2021):
    https://arxiv.org/abs/1903.08008

    Args:
        threshold: Minimum acceptable tail ESS. Default is 1000.

    Returns:
        Checker function that passes if all tail ESS values exceed threshold.
    """

    def check(state: ModelAnalysisState, display: ModellingDisplay = None):
        summary = state.get_summary()
        da = summary.ess_tail.values
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
    Check the number of divergent transitions in the model.

    Category: Hamiltonian Monte Carlo convergence

    Divergent transitions indicate the sampler encountered regions where the
    Hamiltonian dynamics broke down, often due to high curvature in the posterior.
    Any divergences suggest potential bias in the posterior estimates.

    Args:
        threshold: Maximum acceptable number of divergences. Default is 0.

    Returns:
        Checker function that passes if divergence count is at or below threshold.
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
def loo(reff_threshold: float = 0.7, scale: str = "log") -> Checker:
    """
    Compute Leave-One-Out cross-validation using Pareto-Smoothed Importance Sampling.

    Category: Model comparison

    LOO-PSIS estimates out-of-sample predictive accuracy without refitting the model.
    High Pareto k values indicate observations that are difficult to predict and
    where the importance sampling approximation may be unreliable.

    Based on Vehtari, Gelman & Gabry (2017):
    https://arxiv.org/abs/1507.04544

    Args:
        reff_threshold: Maximum acceptable Pareto k value. Default is 0.7.

    Returns:
        Checker function that passes if all Pareto k values are below threshold.
    """

    def check(
        state: ModelAnalysisState, display: ModellingDisplay = None
    ) -> Tuple[ModelAnalysisState, CheckerResult]:
        if "loo" in state.diagnostics:
            loo_res = state.diagnostics["loo"]
        else:
            loo_res = az.loo(state.inference_data, pointwise=True, scale=scale)

        state.add_diagnostic("loo", loo_res)
        state.add_diagnostic("elpd_loo", loo_res.elpd_loo)
        state.add_diagnostic("se_elpd_loo", loo_res.se)
        state.add_diagnostic("p_loo", loo_res.p_loo)
        # Per-observation arrays so they survive CSV export (the ELPDData
        # summary stringifies these xarray DataArrays). See issue #77.
        state.add_diagnostic("loo_i", loo_res.loo_i)
        state.add_diagnostic("pareto_k", loo_res.pareto_k)

        bad = int((loo_res.pareto_k.values > reff_threshold).sum())
        if bad == 0:
            return state, "pass"

        if display:
            display.logger.warning(f"{bad} points with Pareto k > {reff_threshold}")
        return state, "fail"

    return check


@checker
def bfmi(threshold: float = 0.20):
    """
    Check the Bayesian Fraction of Missing Information (BFMI) for each chain.

    Category: Hamiltonian Monte Carlo convergence

    BFMI measures how well the sampler explores the energy distribution.
    Low BFMI values suggest the sampler may be having difficulty exploring
    the posterior, often due to problematic geometry.

    Based on Betancourt (2016):
    https://arxiv.org/abs/1604.00695

    Args:
        threshold: Minimum acceptable BFMI value. Default is 0.20.

    Returns:
        Checker function that passes if all chains have BFMI above threshold.
    """

    def check(state: ModelAnalysisState, display: ModellingDisplay = None):
        sample_stats = getattr(state.inference_data, "sample_stats", None)
        if sample_stats is None or "energy" not in sample_stats:
            if display:
                display.logger.info("BFMI skipped: no energy in sample_stats")
            return state, "NA"

        if "bfmi" in state.diagnostics:
            values = np.asarray(state.diagnostics["bfmi"])
        else:
            # az.bfmi reads sample_stats.energy and returns one value per chain
            values = az.bfmi(state.inference_data)

        state.add_diagnostic("bfmi", values)
        if np.all(values >= threshold):
            return state, "pass"

        if display:
            chains = np.where(values < threshold)[0].tolist()
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
            if display and interactive and not _display_can_prompt(display):
                display.logger.warning(
                    "posterior_predictive_plot: interactive approval was requested "
                    "but no interactive session is available to prompt through. "
                    "Skipping the user prompt and returning 'NA'. Review the saved "
                    "plots offline, or set 'interactive: false' to silence this "
                    "warning."
                )
                verdict = "NA"
            elif display and interactive:
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

    Category: Model comparison

    WAIC estimates out-of-sample predictive accuracy using the log pointwise
    predictive density. It provides similar information to LOO-PSIS but uses
    a different approximation method.

    Based on Vehtari, Gelman & Gabry (2017):
    https://arxiv.org/abs/1507.04544

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
            waic_res = az.waic(state.inference_data, pointwise=True, scale=scale)

        state.add_diagnostic("waic", waic_res)
        state.add_diagnostic("elpd_waic", waic_res.elpd_waic)
        state.add_diagnostic("se_elpd_waic", waic_res.se)
        state.add_diagnostic("p_waic", waic_res.p_waic)
        # Per-observation array so it survives CSV export (the ELPDData summary
        # stringifies this xarray DataArray). See issue #77.
        if "waic_i" in waic_res:
            state.add_diagnostic("waic_i", waic_res.waic_i)

        if display:
            display.logger.info(
                f"elpd_waic: {waic_res['elpd_waic']:.2f} ± {waic_res['se']:.2f}, "
                f"p_waic: {waic_res['p_waic']:.2f}"
            )
        return state, "NA"

    return check


_AGREEMENT_PATTERNS = (
    "both right",
    "only reference right",
    "only compared right",
    "both wrong",
)


def _agreement_shares(p_ref: np.ndarray, p_other: np.ndarray) -> np.ndarray:
    """
    Expected share of the four agreement patterns, averaged over items.

    ``p_ref``/``p_other`` are each model's share correct per item (last axis).
    With one answer per item and model this is the exact share of answer
    pairs in each pattern; with repeats (epochs) it pairs every answer of one
    model with every answer of the other on the same item.
    """
    return np.stack(
        [
            p_ref * p_other,
            p_ref * (1 - p_other),
            (1 - p_ref) * p_other,
            (1 - p_ref) * (1 - p_other),
        ]
    ).mean(axis=-1)


@checker
def agreement_check(
    num_sims: int = 400,
    credible_mass: float = 0.95,
    seed: int = 1,
) -> Checker:
    """
    Posterior predictive check of how often compared models agree.

    Category: Posterior predictive

    For ``accuracy_comparison`` fits with an item term. Every answer pair
    (the reference model and one compared model on the same item) falls into
    one of four patterns: both right, only the reference right, only the
    compared model right, both wrong. The two "only one right" patterns are
    what drive the estimated difference, and the balance between "both
    right" and "both wrong" is what the item spread has to explain.

    For each of ``num_sims`` posterior draws, the check simulates *new*
    items from the fitted difficulty distribution, as many as each group has
    and with the same number of answers per item, and computes the pattern
    shares. New items rather than the fitted ones, so the check isn't
    circular: the fitted item offsets have already seen the answers.

    Passes if every observed share falls inside its ``credible_mass``
    interval. Each share is checked separately, so with many groups or
    compared models expect the occasional share just outside by chance (at
    95%, roughly one in twenty); look at how far outside it is before
    concluding the model misfits. Saves a plot (``agreement_check``) and a table
    (``agreement_check_table``) to the model's diagnostics. Returns NA for
    other models.

    Args:
        num_sims: Posterior draws to simulate from.
        credible_mass: Mass of the simulated interval each share must fall in.
        seed: Random seed for the simulation.
    """

    def check(state: ModelAnalysisState, display: ModellingDisplay = None):
        columns = getattr(state.model, "comparison_columns", None)
        if not columns or not columns.get("item") or state.inference_data is None:
            return state, "NA"
        post = state.inference_data.posterior
        if "item_sigma" not in post:
            return state, "NA"

        from scipy.special import expit

        f = state.features
        model_col, item_col, group_col = (
            columns["model"],
            columns["item"],
            columns["group"],
        )
        model_idx = np.asarray(f[f"{model_col}_index"])
        item_idx = np.asarray(f[f"{item_col}_index"])
        num_model = int(f[f"num_{model_col}"])
        num_item = int(f[f"num_{item_col}"])
        num_group = int(f[f"num_{group_col}"]) if group_col else 1
        group_idx = (
            np.asarray(f[f"{group_col}_index"])
            if group_col
            else np.zeros_like(item_idx)
        )
        obs = np.asarray(f["obs"], dtype=float)
        trials = (
            np.asarray(f["n_total"], dtype=float)
            if "n_total" in f
            else np.ones_like(obs)
        )

        # Successes and trials per (item, model).
        cell = item_idx * num_model + model_idx
        k = np.bincount(cell, weights=obs, minlength=num_item * num_model)
        n = np.bincount(cell, weights=trials, minlength=num_item * num_model)
        k, n = k.reshape(num_item, num_model), n.reshape(num_item, num_model)
        item_group = np.zeros(num_item, dtype=int)
        item_group[item_idx] = group_idx

        # Posterior draws, flattened and thinned: (S, G) and (S, G, K-1).
        stacked = post.stack(sample=("chain", "draw"))
        keep = np.linspace(0, stacked.sizes["sample"] - 1, num_sims).astype(int)

        def draws(name: str, extra: int) -> np.ndarray:
            arr = np.moveaxis(np.asarray(stacked[name]), -1, 0)[keep]
            return arr.reshape((num_sims, num_group) + ((extra,) if extra else ()))

        baseline = draws("baseline", 0)
        sigma = draws("item_sigma", 0)
        difference = draws("difference", num_model - 1)

        coords = state.coords or {}
        model_names = list(coords.get(model_col, range(num_model)))
        group_names = (
            list(coords.get(group_col, range(num_group))) if group_col else [""]
        )

        rng = np.random.default_rng(seed)
        tail = (1 - credible_mass) / 2
        rows, panels = [], []
        for g in range(num_group):
            for m in range(1, num_model):
                items = (item_group == g) & (n[:, 0] > 0) & (n[:, m] > 0)
                if not items.any():
                    continue
                n_ref, n_oth = n[items, 0], n[items, m]
                observed = _agreement_shares(k[items, 0] / n_ref, k[items, m] / n_oth)
                eta = baseline[:, g, None] + sigma[:, g, None] * rng.standard_normal(
                    (num_sims, items.sum())
                )
                sim_ref = rng.binomial(n_ref.astype(int), expit(eta)) / n_ref
                sim_oth = (
                    rng.binomial(
                        n_oth.astype(int), expit(eta + difference[:, g, m - 1, None])
                    )
                    / n_oth
                )
                lo, hi = np.quantile(
                    _agreement_shares(sim_ref, sim_oth), [tail, 1 - tail], axis=1
                )
                for pattern, o, a, b in zip(_AGREEMENT_PATTERNS, observed, lo, hi):
                    rows.append(
                        {
                            "group": str(group_names[g]),
                            "reference": str(model_names[0]),
                            "compared": str(model_names[m]),
                            "pattern": pattern,
                            "observed": o,
                            "lower": a,
                            "upper": b,
                            "inside": bool(a <= o <= b),
                        }
                    )
                panels.append((g, m, observed, lo, hi))

        if not rows:
            return state, "NA"
        table = pd.DataFrame(rows)
        state.add_diagnostic("agreement_check_table", table)

        fig_height = 3.6 if group_col else 3.3
        fig, axes = plt.subplots(
            1,
            len(panels),
            figsize=(3.2 * len(panels) + 1.5, fig_height),
            sharey=True,
            squeeze=False,
        )
        y = np.arange(len(_AGREEMENT_PATTERNS))[::-1]
        for ax, (g, m, observed, lo, hi) in zip(axes[0], panels):
            ax.hlines(
                y,
                lo,
                hi,
                color="#2a78d6",
                linewidth=10,
                alpha=0.45,
                label=f"new items ({credible_mass:.0%} interval)",
            )
            ax.plot(observed, y, "o", color="#0b0b0b", markersize=4.5, label="observed")
            title = f"reference: {model_names[0]}\ncompared: {model_names[m]}"
            if group_col:
                title = f"{group_names[g]}\n{title}"
            ax.set_title(title, loc="left", fontsize=9)
            ax.set_yticks(y, _AGREEMENT_PATTERNS)
            ax.set_xscale("function", functions=(np.sqrt, np.square))
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.05, 0.25, 0.5, 1])
            ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
            ax.grid(axis="x", color="#e4e3dd", linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            ncols=2,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            frameon=False,
            fontsize=9,
        )
        fig.supxlabel("Share of answer pairs (square-root scale)", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig_height))
        state.add_diagnostic("agreement_check", fig)

        outside = table.loc[~table["inside"]]
        if display:
            display.logger.info(
                f"agreement_check: {len(table) - len(outside)} of {len(table)} "
                f"observed pattern shares inside the {credible_mass:.0%} interval"
                + (
                    ""
                    if outside.empty
                    else "; outside: "
                    + ", ".join(
                        f"{r.group} {r.compared}/{r.pattern}".strip()
                        for r in outside.itertuples()
                    )
                )
            )
        return state, "pass" if outside.empty else "fail"

    return check
