"""Prior and likelihood checks independent of MCMC sampling error."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.handlers import seed, trace
from numpyro.infer import Predictive
from scipy.special import expit, logsumexp
import numpyro.distributions as dist

from hibayes.model import ModelsToRunConfig
from hibayes.model.models import ordered_logistic_model
from hibayes.model.utils import ordered_logistic_log_probs


def model_trace(**kwargs):
    return trace(seed(ordered_logistic_model(**kwargs), 0)).get_trace(
        {"obs": jnp.array([0])}
    )


def test_original_priors_and_explicit_overrides_are_preserved():
    original = model_trace()
    assert original["obs"]["fn"].batch_shape == (1,)
    explicit = model_trace(prior_intercept_scale=1.0, prior_first_cutpoint_loc=-4.0)
    overridden = model_trace(
        prior_preset="boundary",
        prior_intercept_scale=1.0,
        prior_first_cutpoint_loc=-4.0,
    )
    for name in original:
        np.testing.assert_array_equal(original[name]["value"], explicit[name]["value"])
        np.testing.assert_array_equal(
            original[name]["value"], overridden[name]["value"]
        )


@pytest.mark.parametrize("classes", [2, 5, 11])
def test_boundary_prior_predictive_supports_both_ends(classes):
    model = ordered_logistic_model(num_classes=classes, prior_preset="boundary")
    samples = Predictive(model, num_samples=4000)(jax.random.PRNGKey(1), {"obs": None})
    cutpoints = np.asarray(samples["cutpoints"])
    intercept = np.asarray(samples["intercept"])
    low = expit(cutpoints[:, 0] - intercept)
    high = expit(intercept - cutpoints[:, -1])
    # Concentration is plausible at either end without dominating the prior.
    assert 0.02 < np.mean(low > 0.99) < 0.25
    assert 0.02 < np.mean(high > 0.99) < 0.25
    assert np.isfinite(cutpoints).all()
    assert (np.diff(cutpoints, axis=-1) > 0).all()
    assert (samples["obs"] >= 0).all()
    assert (samples["obs"] < classes).all()


def zero_posterior(n, **kwargs):
    sites = model_trace(**kwargs)
    first, intercept = sites["first_cutpoint"]["fn"], sites["intercept"]["fn"]
    loc = float(first.loc - intercept.loc)
    variance = float(first.scale**2 + intercept.scale**2)
    x = np.linspace(-30, 60, 90001)
    log_weight = -((x - loc) ** 2) / (2 * variance) - n * np.logaddexp(0, -x)
    weight = np.exp(log_weight - logsumexp(log_weight))
    return float(weight @ expit(-x))


@pytest.mark.parametrize("n", [20, 200, 1000])
def test_all_zero_evidence_overcomes_boundary_prior(n):
    original = zero_posterior(n)
    boundary = zero_posterior(n, prior_preset="boundary")
    # These bounds concern posterior predictions, not agreement with a reference prior.
    assert boundary < original / 3
    assert n * boundary < 1


def test_more_zero_observations_reduce_estimated_nonzero_rate():
    means = [zero_posterior(n, prior_preset="boundary") for n in (20, 200, 1000)]
    assert means[0] > means[1] > means[2] > 0


@pytest.mark.parametrize("n", [20, 200, 1000])
def test_binary_boundary_preset_handles_both_ends_and_mixed_scores(n):
    sites = model_trace(num_classes=2, prior_preset="boundary")
    variance = float(
        sites["first_cutpoint"]["fn"].scale ** 2 + sites["intercept"]["fn"].scale ** 2
    )
    x = np.linspace(-40, 40, 80001)
    results = []
    for successes in (0, int(0.35 * n), n):
        log_weight = (
            -(x**2) / (2 * variance)
            - successes * np.logaddexp(0, x)
            - (n - successes) * np.logaddexp(0, -x)
        )
        weight = np.exp(log_weight - logsumexp(log_weight))
        results.append(float(weight @ expit(-x)))
    assert results[0] == pytest.approx(1 - results[2], abs=1e-8)
    assert results[0] < 1 / n
    assert abs(results[1] - 0.35) < 0.03


def test_preset_uses_configured_cutpoint_spacing():
    sites = model_trace(
        prior_preset="boundary",
        num_classes=5,
        prior_cutpoint_diffs_loc=0.0,
        prior_cutpoint_diffs_scale=0.5,
        min_cutpoint_spacing=0.4,
    )
    expected_width = 3 * (np.exp(0.5**2 / 2) + 0.4)
    assert float(sites["first_cutpoint"]["fn"].loc) == pytest.approx(
        -expected_width / 2
    )


def test_preset_can_be_selected_through_config():
    config = ModelsToRunConfig.from_dict(
        {
            "models": [
                {
                    "name": "ordered_logistic_model",
                    "config": {"prior_preset": "boundary", "num_classes": 5},
                }
            ]
        }
    )
    model, _ = config.enabled_models[0]
    sites = trace(seed(model, 0)).get_trace({"obs": jnp.array([0])})
    assert float(sites["intercept"]["fn"].scale) == 5


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"prior_preset": "typo"}, "prior_preset"),
        ({"num_classes": 1}, "num_classes"),
    ],
)
def test_invalid_preset_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ordered_logistic_model(**kwargs)


@pytest.mark.parametrize("classes", [2, 5, 11])
def test_stable_likelihood_matches_ordered_logistic(classes):
    cuts = jnp.linspace(-4, 4, classes - 1)
    eta = jnp.array([-2.0, 0.0, 2.0])
    actual = np.exp(np.asarray(ordered_logistic_log_probs(eta, cuts)))
    expected = np.asarray(dist.OrderedLogistic(eta, cuts).probs)
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=2e-5)
    np.testing.assert_allclose(actual.sum(-1), 1, atol=2e-7)


@pytest.mark.parametrize("eta", [-100.0, -25.0, 25.0, 100.0])
def test_extreme_likelihood_and_gradients_remain_finite_in_float32(eta):
    cuts = jnp.linspace(-4, 4, 10, dtype=jnp.float32)
    predictor = jnp.float32(eta)
    logits = ordered_logistic_log_probs(predictor, cuts)
    assert np.isfinite(logits).all()
    for category in range(11):

        def fn(value):
            return dist.Categorical(
                logits=ordered_logistic_log_probs(value, cuts)
            ).log_prob(category)

        assert np.isfinite(jax.grad(fn)(predictor))
    # Analytic derivative for an all-zero score; no artificial zero-rate floor.
    derivative = jax.grad(lambda value: ordered_logistic_log_probs(value, cuts)[0])(
        predictor
    )
    np.testing.assert_allclose(derivative, -expit(eta + 4), atol=1e-7)


def test_boundary_preset_builds_without_explicit_first_cutpoint_loc():
    """Regression for #86: the preset computed its default loc with an unimported `exp`."""
    sites = model_trace(num_classes=5, prior_preset="boundary")
    assert float(sites["first_cutpoint"]["fn"].loc) == pytest.approx(-1.4017, abs=1e-4)
