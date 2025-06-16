from typing import List, Optional

import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist

from hibayes.model import Model, check_features, model
from hibayes.process import Features


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
                        f"b_{effect}", prior_main_effect.expand([n_levels - 1])
                    )
                    last_coef = -jnp.sum(free_coefs)
                    coefs = jnp.concatenate([free_coefs, jnp.array([last_coef])])
                    numpyro.deterministic(f"b_{effect}_full", coefs)
                else:
                    # Standard dummy coding
                    coefs = numpyro.sample(
                        f"b_{effect}", prior_main_effect.expand([n_levels])
                    )

                eta += coefs[idx]

        # Add interaction effects
        if interactions:
            for var1, var2 in interactions:
                interaction_matrix = _create_interaction_effects(
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


def _create_interaction_effects(
    name1: str, name2: str, features: Features, prior: dist.Distribution
) -> jnp.ndarray:
    """Create interaction effects matrix with sum-to-zero constraints."""
    n1 = features[f"num_{name1}"]
    n2 = features[f"num_{name2}"]

    if n1 == 1 or n2 == 1:
        return jnp.zeros((n1, n2))

    # Sample free parameters (excluding last row and column)
    raw = numpyro.sample(
        f"b_interaction_{name1}_{name2}", prior.expand([(n1 - 1) * (n2 - 1)])
    ).reshape((n1 - 1, n2 - 1))

    # Initialize full matrix
    b_full = jnp.zeros((n1, n2))

    # Fill in the free parameters
    b_full = b_full.at[: n1 - 1, : n2 - 1].set(raw)

    # Set last row to satisfy row sum-to-zero constraint
    b_full = b_full.at[n1 - 1, : n2 - 1].set(
        -jnp.sum(b_full[: n1 - 1, : n2 - 1], axis=0)
    )

    # Set last column to satisfy column sum-to-zero constraint
    b_full = b_full.at[:, n2 - 1].set(-jnp.sum(b_full[:, : n2 - 1], axis=1))

    numpyro.deterministic(f"b_interaction_{name1}_{name2}_full", b_full)
    return b_full
