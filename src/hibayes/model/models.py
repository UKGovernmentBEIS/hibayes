from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from ..process import Features
from ._model import Model, model
from .utils import create_interaction_effects


def check_features(features: Features, required: List[str]) -> None:
    """Check that the features contain all required keys."""
    missing = [key for key in required if key not in features]
    if missing:
        raise ValueError(
            f"Missing required features: {', '.join(missing)}. Add a data processor to extract these features. Current features: {list(features.keys())}"
        )


@model
def two_level_group_binomial(
    prior_mu_overall_loc: float = 0.0,
    prior_mu_overall_scale: float = 1.0,
    prior_sigma_overall_scale: float = 0.5,
    prior_mu_group_scale: float = 0.5,
    prior_sigma_group_scale: float = 0.1,
) -> Model:
    """
    Two-level group binomial model with customisable priors.

    Args:
        prior_mu_overall_loc: Mean of the normal prior for overall mu
        prior_mu_overall_scale: Scale of the normal prior for overall mu
        prior_sigma_overall_scale: Scale of the half-normal prior for overall sigma
        prior_mu_group_scale: Scale of the normal prior for group mu (centered on overall_mean)
        prior_sigma_group_scale: Scale of the half-normal prior for group sigma
    """

    def model(features: Features) -> None:
        check_features(features, ["obs", "num_group", "group_index", "n_total"])

        mu_overall = numpyro.sample(
            "mu_overall",
            dist.Normal(prior_mu_overall_loc, prior_mu_overall_scale),
        )
        sigma_overall = numpyro.sample(
            "sigma_overall",
            dist.HalfNormal(prior_sigma_overall_scale),
        )
        z_overall = numpyro.sample("z_overall", dist.Normal(0, 1))
        overall_mean = mu_overall + sigma_overall * z_overall
        numpyro.deterministic("overall_mean", overall_mean)

        mu_group = numpyro.sample(
            "mu_group",
            dist.Normal(overall_mean, prior_mu_group_scale),
        )
        sigma_group = numpyro.sample(
            "sigma_group",
            dist.HalfNormal(prior_sigma_group_scale),
        )
        z_group = numpyro.sample(
            "z_group", dist.Normal(0, 1).expand([features["num_group"]])
        )
        group_effects = mu_group + sigma_group * z_group
        numpyro.deterministic("group_effects", group_effects)

        logit_p = group_effects[features["group_index"]]

        # likelihood
        numpyro.sample(
            "obs",
            dist.Binomial(
                total_count=features["n_total"],
                probs=jax.nn.sigmoid(logit_p),
            ),
            obs=features["obs"],
        )

    return model


@model
def ordered_logistic_model(
    main_effects: Optional[List[str]] = None,
    interactions: Optional[List[tuple]] = None,
    num_classes: int = 11,
    effect_coding_for_main_effects: bool = True,
    prior_intercept_loc: float = 0.0,
    prior_intercept_scale: float = 1.0,
    prior_main_effects_loc: float = 0.0,
    prior_main_effects_scale: float = 1.0,
    prior_interaction_loc: float = 0.0,
    prior_interaction_scale: float = 1.0,
    prior_first_cutpoint_loc: float = -4.0,
    prior_first_cutpoint_scale: float = 0.2,
    prior_cutpoint_diffs_loc: float = -0.5,
    prior_cutpoint_diffs_scale: float = 0.3,
    min_cutpoint_spacing: float = 0.3,
) -> Model:
    """
    Ordered logistic regression model with configurable main effects and interactions.

    Args:
        main_effects: List of categorical variables to include as main effects
        interactions: List of tuples specifying interactions (e.g., [('var1', 'var2')])
        num_classes: Number of ordered categories in the outcome
        effect_coding_for_main_effects: Whether to use effect coding (sum-to-zero constraint)
        prior_intercept_loc: Mean of the normal prior for intercept
        prior_intercept_scale: Scale of the normal prior for intercept
        prior_main_effects_loc: Mean of the normal prior for main effects
        prior_main_effects_scale: Scale of the normal prior for main effects
        prior_interaction_loc: Mean of the normal prior for interactions
        prior_interaction_scale: Scale of the normal prior for interactions
        prior_first_cutpoint_loc: Mean of the normal prior for first cutpoint
        prior_first_cutpoint_scale: Scale of the normal prior for first cutpoint
        prior_cutpoint_diffs_loc: Mean of the log-normal prior for cutpoint differences
        prior_cutpoint_diffs_scale: Scale of the log-normal prior for cutpoint differences
        min_cutpoint_spacing: Minimum spacing between cutpoints to ensure identifiability
    """

    def model(features: Features) -> None:
        # Check required features
        required_features = ["obs"]

        # Add requirements for main effects
        if main_effects:
            for effect in main_effects:
                required_features.extend([f"{effect}_index", f"num_{effect}"])

        # Add requirements for interactions
        if interactions:
            for var1, var2 in interactions:
                required_features.extend(
                    [f"{var1}_index", f"num_{var1}", f"{var2}_index", f"num_{var2}"]
                )

        check_features(features, required_features)

        # Priors
        prior_intercept = dist.Normal(prior_intercept_loc, prior_intercept_scale)
        prior_main_effect = dist.Normal(
            prior_main_effects_loc, prior_main_effects_scale
        )
        prior_interaction = dist.Normal(prior_interaction_loc, prior_interaction_scale)
        prior_first_cutpoint = dist.Normal(
            prior_first_cutpoint_loc, prior_first_cutpoint_scale
        )
        prior_cutpoint_diffs = dist.LogNormal(
            prior_cutpoint_diffs_loc, prior_cutpoint_diffs_scale
        )

        # Intercept
        intercept = numpyro.sample("intercept", prior_intercept)
        eta = intercept

        # Add main effects
        if main_effects:
            for effect in main_effects:
                n_levels = features[f"num_{effect}"]
                idx = features[f"{effect}_index"]

                if effect_coding_for_main_effects and n_levels > 1:
                    # Effect coding with sum-to-zero constraint
                    free_coefs = numpyro.sample(
                        f"{effect}_effects_constrained",
                        prior_main_effect.expand([n_levels - 1]),
                    )
                    last_coef = -jnp.sum(free_coefs)
                    coefs = jnp.concatenate([free_coefs, jnp.array([last_coef])])
                    numpyro.deterministic(f"{effect}_effects", coefs)
                else:
                    # Standard dummy coding
                    coefs = numpyro.sample(
                        f"{effect}_effects", prior_main_effect.expand([n_levels])
                    )

                eta += coefs[idx]

        # Add interaction effects
        if interactions:
            for var1, var2 in interactions:
                interaction_matrix = create_interaction_effects(
                    var1, var2, features, prior_interaction
                )
                idx1 = features[f"{var1}_index"]
                idx2 = features[f"{var2}_index"]
                eta += interaction_matrix[idx1, idx2]

        # Cutpoints with ordered constraint
        first_cutpoint = numpyro.sample("first_cutpoint", prior_first_cutpoint)

        if num_classes > 2:
            cutpoint_diffs = numpyro.sample(
                "cutpoint_diffs", prior_cutpoint_diffs, sample_shape=(num_classes - 2,)
            )

            # Ensure minimum spacing between cutpoints
            adjusted_diffs = cutpoint_diffs + min_cutpoint_spacing

            # Build cutpoints
            cutpoints = jnp.concatenate(
                [
                    jnp.array([first_cutpoint]),
                    first_cutpoint + jnp.cumsum(adjusted_diffs),
                ]
            )
        else:
            # For binary case
            cutpoints = jnp.array([first_cutpoint])

        numpyro.deterministic("cutpoints", cutpoints)

        # Likelihood
        numpyro.sample("obs", dist.OrderedLogistic(eta, cutpoints), obs=features["obs"])

    return model
