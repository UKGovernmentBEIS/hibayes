from __future__ import annotations

from typing import TYPE_CHECKING

import arviz as az
import jax
from numpyro.infer import HMC, MCMC, NUTS

from .model_config import INIT_FUNCTION_MAP

if TYPE_CHECKING:
    from ..analysis_state import ModelAnalysisState
    from ..ui import ModellingDisplay


def fit(
    model_analysis_state: ModelAnalysisState, display: ModellingDisplay | None = None
) -> None:
    """
    Run an MCMC fit and store the result.
    """
    if display:
        display.update_header(f"Fitting {model_analysis_state.model_name}")

    cfg = model_analysis_state.model_config.fit
    init_fn = INIT_FUNCTION_MAP[cfg.init_strategy]
    if cfg.method == "NUTS":
        kernel = NUTS(
            model_analysis_state.model,
            target_accept_prob=cfg.target_accept,
            max_tree_depth=cfg.max_tree_depth,
            init_strategy=init_fn(),
        )
    elif cfg.method == "HMC":
        kernel = HMC(model_analysis_state.model)
    else:
        raise ValueError(f"Unsupported inference method: {cfg.method}")

    mcmc = MCMC(
        kernel,
        num_samples=cfg.samples,
        num_warmup=cfg.warmup,
        num_chains=cfg.chains,
        progress_bar=cfg.progress_bar,
        chain_method=model_analysis_state.platform_config.chain_method,
    )

    rng_key = jax.random.PRNGKey(cfg.seed)

    if display:
        display.update_stat("Statistical Models", model_analysis_state.model_name)
        display.update_stat("Chains", cfg.chains)
        display.update_stat("Samples", cfg.samples)
        display.update_stat("Method", cfg.method)
        display.update_stat("Status", "Running")
        display.set_fit_context(
            chain_method=model_analysis_state.platform_config.chain_method,
            num_chains=cfg.chains,
        )

    try:
        mcmc.run(
            rng_key,
            model_analysis_state.features,
            extra_fields=("potential_energy", "energy"),
        )
        if display:
            display.update_stat("Status", "Completed")
    except KeyboardInterrupt:  # pragma: no cover
        if display:
            display.logger.info("MCMC interrupted by user")
        raise
    except Exception as e:  # pragma: no cover
        if display:
            display.update_stat("Status", "Failed")
            display.update_stat("Errors encountered", str(e))
        raise

    idata: az.InferenceData = az.from_numpyro(
        mcmc,
        coords=model_analysis_state.coords,
        dims=model_analysis_state.dims,
    )
    idata.extend(
        model_analysis_state.inference_data, join="right"
    )  # if we calculated prior through some other method use taht.

    # good point to store the observed data as well (overwriting any dummy data from prior checks)
    idata.extend(
        az.from_dict(
            observed_data={"obs": model_analysis_state.features.get("obs", None)},
        ),
        join="right",
    )
    model_analysis_state.inference_data = idata
    model_analysis_state.is_fitted = True
