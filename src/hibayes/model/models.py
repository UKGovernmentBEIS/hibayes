from math import exp
from typing import List, Literal, Optional

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from ..process import Features
from ._model import Model, model
from .utils import create_interaction_effects, ordered_logistic_log_probs


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
    Group binomial with an exponential prior on the group standard deviation.

    ``overall_mean`` is a location on the log-odds scale: its sigmoid is the
    median group probability, not the population average probability. With
    few groups (especially at all-zero/all-success boundaries), the location
    and group spread can remain sensitive to their priors even when each
    group's rate is well estimated. More trials in one group do not provide
    more independent groups. Use ``binomial_estimands_table`` for explicitly
    weighted evaluated-group rates and an integrated population rate.

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

    The default HalfNormal(0.1) group standard deviation strongly pools
    groups. Its population estimates therefore encode different assumptions
    from the exponential model; neither model is universally preferable.
    ``overall_mean`` is a log-odds location, not an average probability.
    Use ``binomial_estimands_table`` to distinguish evaluated-group averages
    from inference about a broader population of exchangeable groups.

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
    prior_intercept_scale: Optional[float] = None,
    prior_main_effects_loc: float = 0.0,
    prior_main_effects_scale: float = 1.0,
    prior_continuous_loc: float = 0.0,
    prior_continuous_scale: float = 1.0,
    prior_interaction_loc: float = 0.0,
    prior_interaction_scale: float = 1.0,
    prior_first_cutpoint_loc: Optional[float] = None,
    prior_first_cutpoint_scale: float = 0.2,
    prior_cutpoint_diffs_loc: float = -0.5,
    prior_cutpoint_diffs_scale: float = 0.3,
    min_cutpoint_spacing: float = 0.3,
    prior_preset: Literal["default", "boundary"] = "default",
) -> Model:
    """
    Ordered logistic regression model with configurable main effects and interactions.

    The default priors favor scores spread across the rubric and strongly
    disfavor almost all scores being zero. For evaluations where concentration
    at either end is plausible, select ``prior_preset="boundary"`` and check
    prior predictions and sensitivity. This preset broadens the intercept to
    Normal(0, 5) and centers the expected cutpoints on zero for the supplied
    number of classes. It does not relax the spacing or effect priors, and is
    not intended for arbitrary concentrations in interior categories.

    Explicit prior arguments override the preset. Omitting the preset preserves
    the original Normal(0, 1) intercept and Normal(-4, 0.2) first cutpoint.

    Args:
        main_effects: List of categorical variables to include as main effects
        continuous_effects: List of continuous variables to include as main effects
        interactions: List of tuples specifying interactions (e.g., [('var1', 'var2')])
        num_classes: Number of ordered categories in the outcome
        effect_coding_for_main_effects: Whether to use effect coding
            (sum-to-zero constraint)
        prior_intercept_loc: Mean of the normal prior for intercept
        prior_intercept_scale: Intercept scale; None uses 1 (default) or 5 (boundary)
        prior_main_effects_loc: Mean of the normal prior for main effects
        prior_main_effects_scale: Scale of the normal prior for main effects
        prior_continuous_loc: Mean of the normal prior for continuous effects
        prior_continuous_scale: Scale of the normal prior for continuous effects
        prior_interaction_loc: Mean of the normal prior for interactions
        prior_interaction_scale: Scale of the normal prior for interactions
        prior_first_cutpoint_loc: First cutpoint mean; None uses the preset
        prior_first_cutpoint_scale: Scale of the normal prior for first cutpoint
        prior_cutpoint_diffs_loc: Mean of the log-normal prior for cutpoint
            differences
        prior_cutpoint_diffs_scale: Scale of the log-normal prior for cutpoint
            differences
        min_cutpoint_spacing: Minimum spacing between cutpoints to ensure
            identifiability
        prior_preset: "default" for the original priors, or "boundary" for
            plausible concentrations at either end of the rubric
    """

    if prior_preset not in ("default", "boundary"):
        raise ValueError("prior_preset must be 'default' or 'boundary'")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    if prior_intercept_scale is None:
        prior_intercept_scale = 5.0 if prior_preset == "boundary" else 1.0
    if prior_first_cutpoint_loc is None:
        if prior_preset == "boundary":
            expected_spacing = (
                exp(prior_cutpoint_diffs_loc + prior_cutpoint_diffs_scale**2 / 2)
                + min_cutpoint_spacing
            )
            prior_first_cutpoint_loc = -(num_classes - 2) * expected_spacing / 2
        else:
            prior_first_cutpoint_loc = -4.0

    if main_effects and continuous_effects:
        overlap = set(main_effects) & set(continuous_effects)
        if overlap:
            raise ValueError(
                f"Variables cannot be in both main_effects and continuous_effects: {overlap}"
            )

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
        # The same ordered-logistic likelihood, evaluated in log space to
        # avoid CDF cancellation and divergent gradients near rubric ends.
        numpyro.sample(
            "obs",
            dist.Categorical(
                logits=ordered_logistic_log_probs(jnp.atleast_1d(eta), cutpoints)
            ),
            obs=features["obs"],
        )

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


def _item_groups(
    item_index: np.ndarray, group_index: np.ndarray, num_item: int, num_group: int
) -> np.ndarray:
    """Group of each item; raises if an item appears in more than one group."""
    item_group = np.zeros(num_item, dtype=np.int32)
    item_group[item_index] = group_index
    if not np.array_equal(item_group[item_index], group_index):
        raise ValueError(
            "Some items appear in more than one group. Item ids must be unique "
            "within the whole dataset, e.g. build them with "
            "combine_columns: {new_column: item, columns: [dataset, task]}."
        )
    if np.bincount(item_index, minlength=num_item).min() == 0:
        raise ValueError("Some item levels have no rows.")
    return item_group


@model
def accuracy_comparison(
    model_column: str = "model",
    item_column: Optional[str] = "item",
    group_column: Optional[str] = None,
    prior_baseline_loc: float = 0.0,
    prior_baseline_scale: float = 1.5,
    prior_difference_scale: float = 1.0,
    prior_item_sigma_scale: float = 2.0,
) -> Model:
    """
    Compare the accuracy of several AI models against a reference model.

    Category: Comparison

    The question this answers is "is model B better than model A on this
    benchmark, and by how much?". On the log-odds scale:

    ``logit P(correct) = baseline + difference[model] + item_offset[item]``

    - ``baseline``: log-odds of a correct answer for the reference model (the
      first ``model_column`` category; choose it with ``reference_categories``
      in ``extract_features``) on an item of average difficulty.
    - ``difference``: how many log-odds each other model adds, one per
      non-reference model.
    - ``item_offset = item_sigma * item_z``: a difficulty offset per item,
      shared by every model (non-centred). This makes the comparison
      *paired*: items every model gets right, or every model gets wrong, say
      little about the difference, and the items they disagree on drive it
      (Miller, 2024, arXiv:2411.00640, §4.2). Repeated answers to the same
      item (epochs) share the offset, so they are not counted as independent
      evidence. Set ``item_column: null`` to drop the term and fit an
      unpaired comparison, which treats every answer as independent.

    With ``group_column`` (e.g. ``benchmark``) each group gets its own
    baseline, differences and item spread. Groups are a handful of named
    datasets that matter in their own right, so they are not pooled.

    The model reports what people ask about on the probability scale:

    - ``accuracy``: per model (and group), averaged over the group's items.
    - ``accuracy_difference``: each model minus the reference, with dim
      ``{model_column}_contrast``.
    - ``mean_accuracy_difference`` (grouped only): the difference averaged
      over groups with equal weight per group.

    Works on one row per answer (Bernoulli) or on aggregated counts
    (Binomial, if ``n_total`` is present, e.g. after ``groupby`` over model
    and item). Summarise with the ``comparison_summary`` communicator and
    check the fit with the ``agreement_check`` checker.

    Note: with only one or a few answers per item, an individual item's
    fitted offset mostly reflects its own outcome pattern. Use the fitted
    ``item_sigma`` (the population spread) rather than per-item offsets when
    describing difficulty.

    Args:
        model_column: Categorical feature identifying the AI model.
        item_column: Categorical feature identifying the item (question).
            ``None`` fits the unpaired model.
        group_column: Optional categorical feature for separate estimates per
            group (e.g. benchmark). Items must not span groups.
        prior_baseline_loc: Mean of the normal prior on ``baseline``.
        prior_baseline_scale: Scale of that prior. 1.5 is roughly uniform on
            accuracy.
        prior_difference_scale: Scale of the normal prior on each
            ``difference``. 1.0 allows large gaps (e.g. 50% vs 85%) while
            ruling out absurd ones.
        prior_item_sigma_scale: Scale of the half-normal prior on
            ``item_sigma``, the spread of item difficulty in log-odds.
            Benchmarks mix trivial and very hard items, so this is wider than
            for most effects; raise it if the posterior presses against it.
    """
    grouped = group_column is not None
    paired = item_column is not None

    def model_fn(features: Features) -> None:
        required = ["obs", f"{model_column}_index", f"num_{model_column}"]
        if paired:
            required += [f"{item_column}_index", f"num_{item_column}"]
        if grouped:
            required += [f"{group_column}_index", f"num_{group_column}"]
        check_features(features, required)

        num_model = int(features[f"num_{model_column}"])
        if num_model < 2:
            raise ValueError(
                f"accuracy_comparison needs at least two levels of '{model_column}', "
                f"got {num_model}."
            )
        model_idx = features[f"{model_column}_index"]
        num_group = int(features[f"num_{group_column}"]) if grouped else 1
        # Plain numpy (not jnp): this runs under jit, and the item/group
        # bookkeeping below needs concrete values.
        group_idx = (
            np.asarray(features[f"{group_column}_index"])
            if grouped
            else np.zeros(np.shape(model_idx), dtype=np.int32)
        )
        shape = (num_group,) if grouped else ()

        baseline = numpyro.sample(
            "baseline",
            dist.Normal(prior_baseline_loc, prior_baseline_scale).expand(shape),
        )
        difference = numpyro.sample(
            "difference",
            dist.Normal(0.0, prior_difference_scale).expand(shape + (num_model - 1,)),
        )
        # Reference model's difference is 0; shape (num_group, num_model).
        base_g = jnp.reshape(baseline, (num_group,))
        diff_g = jnp.concatenate(
            [
                jnp.zeros((num_group, 1)),
                jnp.reshape(difference, (num_group, num_model - 1)),
            ],
            axis=-1,
        )
        logits = base_g[group_idx] + diff_g[group_idx, model_idx]

        if paired:
            num_item = int(features[f"num_{item_column}"])
            item_idx = features[f"{item_column}_index"]
            item_group = _item_groups(
                np.asarray(item_idx), np.asarray(group_idx), num_item, num_group
            )
            item_sigma = numpyro.sample(
                "item_sigma", dist.HalfNormal(prior_item_sigma_scale).expand(shape)
            )
            with numpyro.plate(item_column, num_item):
                item_z = numpyro.sample("item_z", dist.Normal(0.0, 1.0))
            item_offset = jnp.reshape(item_sigma, (num_group,))[item_group] * item_z
            logits = logits + item_offset[item_idx]

            # Accuracy averaged over each group's items, per model.
            p_item = jax.nn.sigmoid(
                (base_g[item_group] + item_offset)[:, None] + diff_g[item_group]
            )
            items_per_group = np.bincount(item_group, minlength=num_group)
            accuracy = (
                jax.ops.segment_sum(p_item, item_group, num_group)
                / items_per_group[:, None]
            )
        else:
            accuracy = jax.nn.sigmoid(base_g[:, None] + diff_g)

        if "n_total" in features:
            numpyro.sample(
                "obs",
                dist.Binomial(total_count=features["n_total"], logits=logits),
                obs=features["obs"],
            )
        else:
            numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=features["obs"])

        accuracy_difference = accuracy[:, 1:] - accuracy[:, :1]
        if grouped:
            numpyro.deterministic("accuracy", accuracy)
            numpyro.deterministic("accuracy_difference", accuracy_difference)
            numpyro.deterministic(
                "mean_accuracy_difference", accuracy_difference.mean(axis=0)
            )
        else:
            numpyro.deterministic("accuracy", accuracy[0])
            numpyro.deterministic("accuracy_difference", accuracy_difference[0])

    def coords_and_dims(coords, dims):
        contrast = f"{model_column}_contrast"
        if model_column in coords:
            reference, *others = coords[model_column]
            coords[contrast] = [f"{m} - {reference}" for m in others]
        g = [group_column] if grouped else []
        dims.update(
            {
                "difference": g + [contrast],
                "accuracy": g + [model_column],
                "accuracy_difference": g + [contrast],
            }
        )
        if grouped:
            dims.update(
                {
                    "baseline": g,
                    "mean_accuracy_difference": [contrast],
                }
            )
        if paired:
            dims["item_z"] = [item_column]
            if grouped:
                dims["item_sigma"] = g
        # Only name dims whose coords exist (e.g. not if features were built
        # by hand without extract_features).
        dims = {k: v for k, v in dims.items() if all(d in coords for d in v)}
        return coords, dims

    model_fn.coords_and_dims = coords_and_dims
    model_fn.comparison_columns = {
        "model": model_column,
        "item": item_column,
        "group": group_column,
    }
    return model_fn
