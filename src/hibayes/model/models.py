from typing import List

import jax
import numpyro
import numpyro.distributions as dist

from ..process import Features
from ._model import Model, model


def check_features(features: Features, required: List[str]) -> None:
    """Check that the features contain all required keys."""
    missing = [key for key in required if key not in features]
    if missing:
        raise ValueError(
            f"Missing required features: {', '.join(missing)}. Add a data processor to extract these features. Current features: {list(features.keys())}"
        )


@model
def two_level_group_binomial() -> Model:
    """Two-level group binomial model. For this simple example there are no customisable arguments."""

    def model(features: Features) -> None:
        check_features(features, ["obs", "num_group", "group_index", "n_total"])

        mu_overall = numpyro.sample(
            "mu_overall",
            dist.Normal(0, 1),
        )
        sigma_overall = numpyro.sample(
            "sigma_overall",
            dist.HalfNormal(0.5),
        )
        z_overall = numpyro.sample("z_overall", dist.Normal(0, 1))
        overall_mean = mu_overall + sigma_overall * z_overall
        numpyro.deterministic("overall_mean", overall_mean)

        mu_group = numpyro.sample(
            "mu_group",
            dist.Normal(overall_mean, 0.5),
        )
        sigma_group = numpyro.sample(
            "sigma_group",
            dist.HalfNormal(0.1),  # notice expect lower variance
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
