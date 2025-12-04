from typing import List, Optional, Tuple

import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist

from hibayes.model import Model, check_features, model
from hibayes.model.utils import create_interaction_effects
from hibayes.process import Features


@model
def hierarchical_ordered_logistic_model(
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
    prior_h_mean_type_loc: float = 0.0,
    prior_h_mean_type_scale: float = 3.0,
    prior_h_sigma_type_scale: float = 1.0,
    prior_raw_effect_loc: float = 0.0,
    prior_raw_effect_scale: float = 1.0,
) -> Model:
    """
    Hierarchical ordered logistic regression model with separate means and variances
    for each grader type. Uses non-centered parameterization for better sampling.
    """

    def model(features: Features) -> None:
        # Check required features
        required_features = ["obs"]

        # Hierarchical model requires grader and grader_type
        required_features.extend(
            ["grader_index", "num_grader", "grader_type_index", "num_grader_type"]
        )

        # Add requirements for main effects
        if main_effects:
            for effect in main_effects:
                if effect != "grader":  # grader handled hierarchically
                    required_features.extend([f"{effect}_index", f"num_{effect}"])

        # Add requirements for interactions
        if interactions:
            for var1, var2 in interactions:
                if var1 not in ["grader"]:
                    required_features.extend([f"{var1}_index", f"num_{var1}"])
                if var2 not in ["grader"]:
                    required_features.extend([f"{var2}_index", f"num_{var2}"])

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
        prior_h_mean_type = dist.Normal(prior_h_mean_type_loc, prior_h_mean_type_scale)
        prior_h_sigma_type = dist.HalfCauchy(prior_h_sigma_type_scale)
        prior_raw_effect = dist.Normal(prior_raw_effect_loc, prior_raw_effect_scale)

        # Intercept
        intercept = numpyro.sample("intercept", prior_intercept)
        eta = intercept
        filtered_main_effects = [e for e in (main_effects or []) if e != "grader"]

        # Get number of grader types
        n_grader_types = features["num_grader_type"]

        # Sample means for each grader type
        grader_type_means = numpyro.sample(
            "grader_type_effects", prior_h_mean_type.expand([n_grader_types])
        )

        # Sample type-specific standard deviations
        grader_type_sigmas = numpyro.sample(
            "grader_type_sigmas", prior_h_sigma_type.expand([n_grader_types])
        )

        # Get grader and type indices
        grader_indices = features["grader_index"]
        grader_type_indices = features["grader_type_index"]
        n_graders = features["num_grader"]

        # Create mapping from grader index to grader type
        # We need to build this mapping for all possible grader indices
        grader_type_mapping = jnp.zeros(n_graders, dtype=jnp.int32)
        grader_type_mapping = grader_type_mapping.at[grader_indices].set(
            grader_type_indices
        )

        # Sample individual grader effects with non-centered parameterization
        raw_effects = numpyro.sample(
            "grader_raw_effects", prior_raw_effect.expand([n_graders])
        )

        # Get the type-specific means and sigmas for each grader
        grader_means = grader_type_means[grader_type_mapping]
        grader_sigmas = grader_type_sigmas[grader_type_mapping]

        # Compute actual effects using non-centered parameterization
        grader_effects = grader_means + grader_sigmas * raw_effects
        numpyro.deterministic("grader_effects", grader_effects)

        # Save pairwise differences between grader types (for interpretability)
        # This creates a matrix of differences: type_i - type_j
        if n_grader_types > 1:
            type_diffs = grader_type_means[:, None] - grader_type_means[None, :]
            numpyro.deterministic("grader_type_differences", type_diffs)

        # Add grader effect to linear predictor
        eta += grader_effects[grader_indices]

        # Add main effects
        if filtered_main_effects:
            for effect in filtered_main_effects:
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

        # Add interaction effects
        if interactions:
            for var1, var2 in interactions:
                if (var1 == "grader" and var2 == "LLM") or (
                    var1 == "LLM" and var2 == "grader"
                ):
                    # Handle grader-LLM interaction specially
                    n_graders = features["num_grader"]
                    n_llms = features["num_LLM"]

                    # Sample interaction effects
                    grader_llm_interaction = numpyro.sample(
                        "grader_LLM_effects",
                        prior_interaction.expand([n_graders, n_llms]),
                    )

                    # Add to linear predictor
                    grader_idx = features["grader_index"]
                    llm_idx = features["LLM_index"]
                    eta += grader_llm_interaction[grader_idx, llm_idx]
                else:
                    # Handle other interactions using the helper function
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
def pairwise_logistic_model(
    main_effects: Optional[List[str]] = None,
    interactions: Optional[List[Tuple[str, str]]] = None,
    effect_coding_for_main_effects: bool = True,
    # Priors
    prior_intercept_loc: float = 0.0,
    prior_intercept_scale: float = 1.0,
    prior_main_effects_loc: float = 0.0,
    prior_main_effects_scale: float = 1.0,
    prior_interaction_loc: float = 0.0,
    prior_interaction_scale: float = 1.0,
    # Grader-specific length slope hyperpriors (used iff "length_diff" in main_effects)
    prior_length_slope_mean_loc: float = 0.0,
    prior_length_slope_mean_scale: float = 0.5,
    prior_length_slope_sigma_scale: float = 1.0,
) -> Model:
    """
    Binary (pairwise) logistic regression with:
      - Intercept
      - LLM-pair effect
      - Grader effect
      - Optional grader-specific slope for a numeric covariate 'length_diff'
      - Optional additional main effects (effect-coded or dummy)
      - Optional pairwise interactions via create_interaction_effects

    Likelihood: Bernoulli with logits link.
    """

    def model(features: Features) -> None:
        #  Required features
        required = [
            "obs",  # binary outcomes (0/1)
            "llm_pair_index",
            "num_llm_pair",
            "grader_index",
            "num_grader",
        ]

        # Optional numeric covariate used in the original: length_diff
        filtered_main = list(main_effects or [])
        uses_length_diff = "length_diff" in filtered_main
        if uses_length_diff:
            required.append("length_diff")
            # do not treat 'length_diff' as a categorical main effect
            filtered_main = [e for e in filtered_main if e != "length_diff"]

        # Add requirements for any additional categorical main effects
        for effect in filtered_main:
            required.extend([f"{effect}_index", f"num_{effect}"])

        # Add requirements for interactions
        if interactions:
            for v1, v2 in interactions:
                if v1 not in ["length_diff"]:
                    required.extend([f"{v1}_index", f"num_{v1}"])
                if v2 not in ["length_diff"]:
                    required.extend([f"{v2}_index", f"num_{v2}"])
                # If either is 'length_diff', we’ll treat it as numeric below.

        check_features(features, required)

        #  Priors
        prior_intercept = dist.Normal(prior_intercept_loc, prior_intercept_scale)
        prior_main = dist.Normal(prior_main_effects_loc, prior_main_effects_scale)
        prior_inter = dist.Normal(prior_interaction_loc, prior_interaction_scale)

        #  Intercept
        intercept = numpyro.sample("intercept", prior_intercept)
        eta = intercept

        #  Optional grader-specific slope for numeric 'length_diff'
        if uses_length_diff:
            length_slope_mean = numpyro.sample(
                "length_slope_mean",
                dist.Normal(prior_length_slope_mean_loc, prior_length_slope_mean_scale),
            )
            sigma_slope = numpyro.sample(
                "length_slope_sigma", dist.HalfNormal(prior_length_slope_sigma_scale)
            )
            grader_length_slopes = numpyro.sample(
                "grader_length_diff_effects",
                dist.Normal(length_slope_mean, sigma_slope)
                .expand([features["num_grader"]])
                .to_event(1),
            )
            length_diff = features["length_diff"]
            eta = eta + grader_length_slopes[features["grader_index"]] * length_diff

        #  Additional main effects (categorical)
        for effect in filtered_main:
            n_levels = features[f"num_{effect}"]
            idx = features[f"{effect}_index"]
            if effect_coding_for_main_effects and n_levels > 1:
                free = numpyro.sample(
                    f"{effect}_effects_constrained", prior_main.expand([n_levels - 1])
                )
                last = -jnp.sum(free)
                coefs = jnp.concatenate([free, jnp.array([last])])
                numpyro.deterministic(f"{effect}_effects", coefs)
            else:
                # Standard dummy coding (reference category is 0)
                coefs = jnp.concatenate(
                    [
                        jnp.array([0.0]),  # Reference category
                        numpyro.sample(
                            f"{effect}_effects_raw", prior_main.expand([n_levels - 1]),
                        ),
                    ]
                )
                numpyro.deterministic(f"{effect}_effects", coefs)
            eta = eta + coefs[idx]

        #  Interactions
        if interactions:
            for var1, var2 in interactions:
                # Numeric × categorical: allow length_diff in either slot
                if var1 == "length_diff" and var2 != "length_diff":
                    idx2 = features[f"{var2}_index"]
                    n2 = features[f"num_{var2}"]
                    # one slope per level of var2
                    slopes = numpyro.sample(
                        f"length_diff_{var2}", prior_inter.expand([n2])
                    )
                    eta = eta + slopes[idx2] * features["length_diff"]

                elif var2 == "length_diff" and var1 != "length_diff":
                    idx1 = features[f"{var1}_index"]
                    n1 = features[f"num_{var1}"]
                    slopes = numpyro.sample(
                        f"{var1}_length_diff", prior_inter.expand([n1])
                    )
                    eta = eta + slopes[idx1] * features["length_diff"]

                elif var1 != "length_diff" and var2 != "length_diff":
                    # Categorical × categorical via helper (sum-to-zero constrained)
                    interaction_matrix = create_interaction_effects(
                        var1, var2, features, prior_inter
                    )
                    i1 = features[f"{var1}_index"]
                    i2 = features[f"{var2}_index"]
                    eta = eta + interaction_matrix[i1, i2]
                else:
                    raise ValueError(
                        "Cannot have length_diff x length_diff interaction"
                    )

        #  Likelihood (Bernoulli with logits)
        numpyro.sample("obs", dist.Bernoulli(logits=eta), obs=features["obs"])

    return model


@model
def ordered_logistic_model_with_continuous(
    main_effects: List[str] | None = None,
    continuous_effects: List[str] | None = None, 
    interactions: List[tuple] | None = None,
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
                    # Standard dummy coding
                    coefs = numpyro.sample(
                        f"{effect}_effects", prior_main_effect.expand([n_levels])
                    )

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




