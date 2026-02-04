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
def simplified_group_binomial_exponential(
    prior_mu_overall_loc: float = 0.0,
    prior_mu_overall_scale: float = 1.0,
    prior_sigma_group_rate: float = 1.0,
) -> Model:
    """
    Simplified group binomial with exponential prior on group variance.
    Here we drop the redundant reparameterisation step and introduce less
    informative priors.

    Args:
        prior_mu_overall_loc: Mean of the normal prior for overall mu
        prior_mu_overall_scale: Scale of the normal prior for overall mu
        prior_sigma_group_rate: Rate of the exponential prior for group sigma
    """

    def model(features: Features) -> None:
        check_features(features, ["obs", "num_group", "group_index", "n_total"])

        overall_mean = numpyro.sample(
            "overall_mean",
            dist.Normal(prior_mu_overall_loc, prior_mu_overall_scale),
        )

        # Group-level variability with Exponential
        sigma_group = numpyro.sample(
            "sigma_group",
            dist.Exponential(rate=prior_sigma_group_rate),
        )

        z_group = numpyro.sample(
            "z_group", dist.Normal(0, 1).expand([features["num_group"]])
        )
        group_effects = overall_mean + sigma_group * z_group
        numpyro.deterministic("group_effects", group_effects)

        logit_p = group_effects[features["group_index"]]

        # Likelihood
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
def two_level_group_binomial(
    prior_mu_overall_loc: float = 0.0,
    prior_mu_overall_scale: float = 1.0,
    prior_sigma_overall_scale: float = 0.5,
    prior_sigma_group_scale: float = 0.1,
) -> Model:
    """
    Two-level group binomial model with customisable priors.

    Args:
        prior_mu_overall_loc: Mean of the normal prior for overall mu
        prior_mu_overall_scale: Scale of the normal prior for overall mu
        prior_sigma_overall_scale: Scale of the half-normal prior for overall sigma
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

        sigma_group = numpyro.sample(
            "sigma_group",
            dist.HalfNormal(prior_sigma_group_scale),
        )
        z_group = numpyro.sample(
            "z_group", dist.Normal(0, 1).expand([features["num_group"]])
        )
        group_effects = overall_mean + sigma_group * z_group
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
    continuous_effects: Optional[List[str]] = None,
    interactions: Optional[List[tuple]] = None,
    num_classes: int = 11,
    effect_coding_for_main_effects: bool = True,
    prior_intercept_loc: float = 0.0,
    prior_intercept_scale: float = 1.0,
    prior_main_effects_loc: float = 0.0,
    prior_main_effects_scale: float = 1.0,
    prior_continuous_loc: float = 0.0,
    prior_continuous_scale: float = 1.0,
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
        continuous_effects: List of continuous variables to include as main effects
        interactions: List of tuples specifying interactions (e.g., [('var1', 'var2')])
        num_classes: Number of ordered categories in the outcome
        effect_coding_for_main_effects: Whether to use effect coding
        (sum-to-zero constraint)
        prior_intercept_loc: Mean of the normal prior for intercept
        prior_intercept_scale: Scale of the normal prior for intercept
        prior_main_effects_loc: Mean of the normal prior for main effects
        prior_main_effects_scale: Scale of the normal prior for main effects
        prior_continuous_loc: Mean of the normal prior for continuous effects
        prior_continuous_scale: Scale of the normal prior for continuous effects
        prior_interaction_loc: Mean of the normal prior for interactions
        prior_interaction_scale: Scale of the normal prior for interactions
        prior_first_cutpoint_loc: Mean of the normal prior for first cutpoint
        prior_first_cutpoint_scale: Scale of the normal prior for first cutpoint
        prior_cutpoint_diffs_loc: Mean of the log-normal prior for cutpoint differences
        prior_cutpoint_diffs_scale: Scale of the log-normal prior for cutpoint
        differences
        min_cutpoint_spacing: Minimum spacing between cutpoints to ensure
        identifiability
    """

    def model(features: Features) -> None:
        # Check required features
        required_features = ["obs"]

        # Add requirements for categorical main effects
        if main_effects:
            for effect in main_effects:
                required_features.extend([f"{effect}_index", f"num_{effect}"])

        # Add requirements for continuous effects
        if continuous_effects:
            for effect in continuous_effects:
                required_features.append(effect)

        # Add requirements for interactions
        if interactions:
            for var1, var2 in interactions:
                # Determine if each variable is continuous or categorical
                is_var1_continuous = continuous_effects and var1 in continuous_effects
                is_var2_continuous = continuous_effects and var2 in continuous_effects

                if not is_var1_continuous:
                    required_features.extend([f"{var1}_index", f"num_{var1}"])
                else:
                    required_features.append(var1)

                if not is_var2_continuous:
                    required_features.extend([f"{var2}_index", f"num_{var2}"])
                else:
                    required_features.append(var2)

        check_features(features, required_features)

        # Priors
        prior_intercept = dist.Normal(prior_intercept_loc, prior_intercept_scale)
        prior_main_effect = dist.Normal(
            prior_main_effects_loc, prior_main_effects_scale
        )
        prior_continuous = dist.Normal(prior_continuous_loc, prior_continuous_scale)
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

        # Add categorical main effects
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
                    # Standard dummy coding (reference category is 0)
                    coefs = jnp.concatenate(
                        [
                            jnp.array([0.0]),  # Reference category
                            numpyro.sample(
                                f"{effect}_effects_raw",
                                prior_main_effect.expand([n_levels - 1]),
                            ),
                        ]
                    )
                    numpyro.deterministic(f"{effect}_effects", coefs)
                    
                eta += coefs[idx]

        # Add continuous main effects
        if continuous_effects:
            for effect in continuous_effects:
                coef = numpyro.sample(f"{effect}_coef", prior_continuous)
                eta += coef * features[effect]

        # Add interaction effects
        if interactions:
            for var1, var2 in interactions:
                is_var1_continuous = continuous_effects and var1 in continuous_effects
                is_var2_continuous = continuous_effects and var2 in continuous_effects

                if is_var1_continuous and is_var2_continuous:
                    # Continuous × Continuous
                    coef = numpyro.sample(f"{var1}_{var2}_effects", prior_interaction)
                    eta += coef * features[var1] * features[var2]

                elif is_var1_continuous and not is_var2_continuous:
                    # Continuous × Categorical: one slope per category
                    idx2 = features[f"{var2}_index"]
                    n2 = features[f"num_{var2}"]
                    slopes = numpyro.sample(
                        f"{var1}_{var2}_effects", prior_interaction.expand([n2])
                    )
                    eta += slopes[idx2] * features[var1]

                elif not is_var1_continuous and is_var2_continuous:
                    # Categorical × Continuous: one slope per category
                    idx1 = features[f"{var1}_index"]
                    n1 = features[f"num_{var1}"]
                    slopes = numpyro.sample(
                        f"{var1}_{var2}_effects", prior_interaction.expand([n1])
                    )
                    eta += slopes[idx1] * features[var2]

                else:
                    # Categorical × Categorical
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


@model
def three_level_group_binomial(
    prior_overall_mean_loc: float = 0.0,
    prior_overall_mean_scale: float = 1.0,
    prior_sigma_group_scale: float = 0.1,
    prior_sigma_subgroup_scale: float = 0.1,
    prior_sigma_subsubgroup_scale: float = 0.1,
) -> Model:
    """
    Three-level hierarchical binomial model for aggregated data.

    Args:
        prior_overall_mean_loc: Mean of the normal prior for overall mean
        prior_overall_mean_scale: Scale of the normal prior for overall mean
        prior_sigma_group_scale: Scale of the half-normal prior for group-level sigma
        prior_sigma_subgroup_scale: Scale of the half-normal prior for subgroup-level sigma
        prior_sigma_subsubgroup_scale: Scale of the half-normal prior for subsubgroup-level sigma
    """

    def model(features: Features) -> None:
        check_features(
            features,
            [
                "obs",
                "n_total",
                "group_index",
                "num_group",
                "subgroup_index",
                "num_subgroup",
                "subsubgroup_index",
                "num_subsubgroup",
            ],
        )

        # Overall mean
        overall_mean = numpyro.sample(
            "overall_mean",
            dist.Normal(prior_overall_mean_loc, prior_overall_mean_scale),
        )

        # Group-level effects from overall mean
        sigma_group = numpyro.sample(
            "sigma_group", dist.HalfNormal(prior_sigma_group_scale)
        )
        z_group = numpyro.sample(
            "z_group", dist.Normal(0, 1).expand([features["num_group"]])
        )
        group_effects = overall_mean + sigma_group * z_group
        numpyro.deterministic("group_effects", group_effects)

        # Subgroup-level effects drawn from group effects
        sigma_subgroup = numpyro.sample(
            "sigma_subgroup", dist.HalfNormal(prior_sigma_subgroup_scale)
        )
        z_subgroup = numpyro.sample(
            "z_subgroup",
            dist.Normal(0, 1).expand([features["num_group"], features["num_subgroup"]]),
        )
        subgroup_effects = group_effects[:, None] + sigma_subgroup * z_subgroup
        numpyro.deterministic("subgroup_effects", subgroup_effects)

        # Subsubgroup-level effects drawn from subgroup effects
        sigma_subsubgroup = numpyro.sample(
            "sigma_subsubgroup", dist.HalfNormal(prior_sigma_subsubgroup_scale)
        )
        z_subsubgroup = numpyro.sample(
            "z_subsubgroup",
            dist.Normal(0, 1).expand(
                [
                    features["num_group"],
                    features["num_subgroup"],
                    features["num_subsubgroup"],
                ]
            ),
        )
        subsubgroup_effects = (
            subgroup_effects[:, :, None] + sigma_subsubgroup * z_subsubgroup
        )
        numpyro.deterministic("subsubgroup_effects", subsubgroup_effects)

        # Linear predictor
        logit_p = subsubgroup_effects[
            features["group_index"],
            features["subgroup_index"],
            features["subsubgroup_index"],
        ]

        # Likelihood using aggregated data
        numpyro.sample(
            "obs",
            dist.Binomial(
                total_count=features["n_total"], probs=jax.nn.sigmoid(logit_p)
            ),
            obs=features["obs"],
        )

    return model


@model
def three_level_group_binomial_exponential(
    prior_overall_mean_loc: float = 0.0,
    prior_overall_mean_scale: float = 1.0,
    prior_sigma_group_rate: float = 1.0,  # Rate parameter for group-level exponential
    prior_sigma_subgroup_rate: float = 1.0,  # Rate parameter for subgroup-level exponential
    prior_sigma_subsubgroup_rate: float = 1.0,  # Rate parameter for subsubgroup-level exponential
) -> Model:
    """
    Three-level hierarchical binomial model with exponential priors on variance components.

    Uses Exponential distributions for variance components - simple, always positive,
    with moderate tail behavior.

    Args:
        prior_overall_mean_loc: Mean of the normal prior for overall mean
        prior_overall_mean_scale: Scale of the normal prior for overall mean
        prior_sigma_group_rate: Rate of the exponential prior for group-level sigma
        prior_sigma_subgroup_rate: Rate of the exponential prior for subgroup-level sigma
        prior_sigma_subsubgroup_rate: Rate of the exponential prior for subsubgroup-level sigma
    """

    def model(features: Features) -> None:
        check_features(
            features,
            [
                "obs",
                "n_total",
                "group_index",
                "num_group",
                "subgroup_index",
                "num_subgroup",
                "subsubgroup_index",
                "num_subsubgroup",
            ],
        )

        overall_mean = numpyro.sample(
            "overall_mean",
            dist.Normal(prior_overall_mean_loc, prior_overall_mean_scale),
        )

        # Group-level effects with exponential prior
        sigma_group = numpyro.sample(
            "sigma_group",
            dist.Exponential(rate=prior_sigma_group_rate),
        )
        z_group = numpyro.sample(
            "z_group", dist.Normal(0, 1).expand([features["num_group"]])
        )
        group_effects = overall_mean + sigma_group * z_group
        numpyro.deterministic("group_effects", group_effects)

        # Subgroup-level effects with exponential prior
        sigma_subgroup = numpyro.sample(
            "sigma_subgroup",
            dist.Exponential(rate=prior_sigma_subgroup_rate),
        )
        z_subgroup = numpyro.sample(
            "z_subgroup",
            dist.Normal(0, 1).expand([features["num_group"], features["num_subgroup"]]),
        )
        subgroup_effects = group_effects[:, None] + sigma_subgroup * z_subgroup
        numpyro.deterministic("subgroup_effects", subgroup_effects)

        # Subsubgroup-level effects with exponential prior
        sigma_subsubgroup = numpyro.sample(
            "sigma_subsubgroup",
            dist.Exponential(rate=prior_sigma_subsubgroup_rate),
        )
        z_subsubgroup = numpyro.sample(
            "z_subsubgroup",
            dist.Normal(0, 1).expand(
                [
                    features["num_group"],
                    features["num_subgroup"],
                    features["num_subsubgroup"],
                ]
            ),
        )
        subsubgroup_effects = (
            subgroup_effects[:, :, None] + sigma_subsubgroup * z_subsubgroup
        )
        numpyro.deterministic("subsubgroup_effects", subsubgroup_effects)

        # Linear predictor
        logit_p = subsubgroup_effects[
            features["group_index"],
            features["subgroup_index"],
            features["subsubgroup_index"],
        ]

        # Likelihood using aggregated data
        numpyro.sample(
            "obs",
            dist.Binomial(
                total_count=features["n_total"], probs=jax.nn.sigmoid(logit_p)
            ),
            obs=features["obs"],
        )

    return model


@model
def linear_group_binomial(
    main_effects: Optional[List[str]] = None,
    interactions: Optional[List[tuple]] = None,
    effect_coding_for_main_effects: bool = True,
    prior_intercept_loc: float = 0.0,
    prior_intercept_scale: float = 1.0,
    prior_main_effects_loc: float = 0.0,
    prior_main_effects_scale: float = 0.5,
    prior_interaction_loc: float = 0.0,
    prior_interaction_scale: float = 0.3,
) -> Model:
    """
    Linear binomial regression model with configurable main effects and interactions.

    This is a non-hierarchical model that treats groups, subgroups, and subsubgroups
    as categorical predictors without nesting structure.

    Args:
        main_effects: List of categorical variables to include as main effects
            e.g., ['group', 'subgroup', 'subsubgroup']
        interactions: List of tuples specifying interactions
            e.g., [('group', 'subgroup'), ('subgroup', 'subsubgroup')]
        effect_coding_for_main_effects: Whether to use effect coding (sum-to-zero constraint)
        prior_intercept_loc: Mean of the normal prior for intercept
        prior_intercept_scale: Scale of the normal prior for intercept
        prior_main_effects_loc: Mean of the normal prior for main effects
        prior_main_effects_scale: Scale of the normal prior for main effects
        prior_interaction_loc: Mean of the normal prior for interactions
        prior_interaction_scale: Scale of the normal prior for interactions
    """

    def model(features: Features) -> None:
        # Check required features
        required_features = ["obs", "n_total"]

        # Add requirements for main effects
        if main_effects:
            for effect in main_effects:
                required_features.extend([f"{effect}_index", f"num_{effect}"])

        # Add requirements for interactions
        if interactions:
            for var1, var2 in interactions:
                # Only add if not already in required_features
                for feat in [
                    f"{var1}_index",
                    f"num_{var1}",
                    f"{var2}_index",
                    f"num_{var2}",
                ]:
                    if feat not in required_features:
                        required_features.append(feat)

        check_features(features, required_features)

        # Priors
        prior_intercept = dist.Normal(prior_intercept_loc, prior_intercept_scale)
        prior_main_effect = dist.Normal(
            prior_main_effects_loc, prior_main_effects_scale
        )
        prior_interaction = dist.Normal(prior_interaction_loc, prior_interaction_scale)

        # Intercept (on logit scale)
        intercept = numpyro.sample("intercept", prior_intercept)
        logit_p = intercept

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
                    # Standard dummy coding (reference category is 0)
                    coefs = jnp.concatenate(
                        [
                            jnp.array([0.0]),  # Reference category
                            numpyro.sample(
                                f"{effect}_effects_raw",
                                prior_main_effect.expand([n_levels - 1]),
                            ),
                        ]
                    )
                    numpyro.deterministic(f"{effect}_effects", coefs)

                logit_p = logit_p + coefs[idx]

        # Add interaction effects using create_interaction_effects
        if interactions:
            for var1, var2 in interactions:
                interaction_matrix = create_interaction_effects(
                    var1, var2, features, prior_interaction
                )
                idx1 = features[f"{var1}_index"]
                idx2 = features[f"{var2}_index"]
                logit_p = logit_p + interaction_matrix[idx1, idx2]

        # Likelihood using binomial distribution
        numpyro.sample(
            "obs",
            dist.Binomial(
                total_count=features["n_total"], probs=jax.nn.sigmoid(logit_p)
            ),
            obs=features["obs"],
        )

    return model
