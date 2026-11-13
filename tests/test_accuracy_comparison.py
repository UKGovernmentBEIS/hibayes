"""Tests for comparing AI models on shared items.

Covers the ``accuracy_comparison`` model, the ``combine_columns`` and
``set_dims`` processors, the ``comparison_summary`` communicator, the
``agreement_check`` checker, ``format_probability`` and the model-declared
coords/dims hook.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import numpyro.distributions as dist
import pandas as pd
import pytest
from numpyro.handlers import seed, trace
from scipy.special import expit

from hibayes.analysis_state import AnalysisState, ModelAnalysisState
from hibayes.check import agreement_check
from hibayes.communicate import comparison_summary
from hibayes.communicate.utils import format_probability
from hibayes.model import ModelConfig, fit, model_coords_and_dims
from hibayes.model.models import accuracy_comparison, two_level_group_binomial
from hibayes.platform import PlatformConfig
from hibayes.process import (
    combine_columns,
    extract_features,
    extract_observed_feature,
    groupby,
    set_dims,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def simulate_answers(
    num_items: int = 300,
    models: tuple = ("ref", "other"),
    baseline: float = 0.3,
    differences: tuple = (-0.8,),
    item_sigma: float = 1.5,
    epochs: int = 1,
    groups: tuple | None = None,
    seed_value: int = 0,
) -> pd.DataFrame:
    """One row per answer from the accuracy_comparison generative model."""
    rng = np.random.default_rng(seed_value)
    rows = []
    for g in groups or (None,):
        offsets = item_sigma * rng.standard_normal(num_items)
        for i, offset in enumerate(offsets):
            for m, name in enumerate(models):
                eta = baseline + (0.0 if m == 0 else differences[m - 1]) + offset
                for e in range(epochs):
                    rows.append(
                        {
                            "model": name,
                            "task": i,
                            "dataset": g or "bench",
                            "epoch": e,
                            "score": int(rng.random() < expit(eta)),
                        }
                    )
    return pd.DataFrame(rows)


def process(
    data: pd.DataFrame,
    reference: str = "ref",
    grouped: bool = False,
    aggregate: bool = False,
) -> AnalysisState:
    state = AnalysisState(data=data, processed_data=data.copy())
    state = combine_columns(new_column="item", columns=["dataset", "task"])(state)
    if grouped:
        state.processed_data["benchmark"] = state.processed_data["dataset"]
    cats = ["model", "item"] + (["benchmark"] if grouped else [])
    if aggregate:
        state = groupby(groupby_columns=cats)(state)
        state = extract_features(
            categorical_features=cats,
            continuous_features=["n_total"],
            reference_categories={"model": reference},
        )(state)
        state = extract_observed_feature(feature_name="n_correct")(state)
    else:
        state = extract_features(
            categorical_features=cats, reference_categories={"model": reference}
        )(state)
        state = extract_observed_feature(feature_name="score")(state)
    return state


def fit_model(state: AnalysisState, samples: int = 400, **model_kwargs):
    model = accuracy_comparison(**model_kwargs)
    coords, dims = model_coords_and_dims(model, state.coords, state.dims)
    mas = ModelAnalysisState(
        model=model,
        model_config=ModelConfig.from_dict(
            {"fit": {"samples": samples, "warmup": samples, "chains": 1, "seed": 0}}
        ),
        platform_config=PlatformConfig(chain_method="sequential"),
        features=state.features,
        coords=coords,
        dims=dims,
    )
    fit(mas)
    state.add_model(mas)
    return mas


def model_trace(state: AnalysisState, **model_kwargs) -> dict:
    model = accuracy_comparison(**model_kwargs)
    return trace(seed(model, 0)).get_trace(state.features)


@pytest.fixture(scope="module")
def paired_fit():
    state = process(simulate_answers())
    return state, fit_model(state)


# ---------------------------------------------------------------------------
# Processors
# ---------------------------------------------------------------------------


class TestCombineColumns:
    def test_joins_columns(self):
        df = pd.DataFrame({"dataset": ["a", "b"], "task": [1, 1]})
        state = AnalysisState(data=df, processed_data=df.copy())
        state = combine_columns(new_column="item", columns=["dataset", "task"])(state)
        assert state.processed_data["item"].tolist() == ["a::1", "b::1"]

    def test_custom_separator(self):
        df = pd.DataFrame({"dataset": ["a"], "task": [3]})
        state = AnalysisState(data=df, processed_data=df.copy())
        state = combine_columns(
            new_column="item", columns=["dataset", "task"], separator="/"
        )(state)
        assert state.processed_data["item"].iloc[0] == "a/3"

    def test_missing_column_raises(self):
        df = pd.DataFrame({"dataset": ["a"]})
        state = AnalysisState(data=df, processed_data=df.copy())
        with pytest.raises(ValueError, match="not found"):
            combine_columns(new_column="item", columns=["dataset", "task"])(state)

    def test_missing_values_raise(self):
        df = pd.DataFrame({"dataset": ["a", None], "task": [1, 2]})
        state = AnalysisState(data=df, processed_data=df.copy())
        with pytest.raises(ValueError, match="missing values"):
            combine_columns(new_column="item", columns=["dataset", "task"])(state)


class TestSetDims:
    def test_merges_into_existing_dims(self):
        df = pd.DataFrame({"model": ["a", "b"], "score": [1, 0]})
        state = AnalysisState(data=df, processed_data=df.copy())
        state = extract_features(categorical_features=["model"])(state)
        state = set_dims(dims={"accuracy": ["model"]})(state)
        assert state.dims["accuracy"] == ["model"]
        assert state.dims["model_effects"] == ["model"]

    def test_unknown_coord_raises(self):
        df = pd.DataFrame({"model": ["a", "b"], "score": [1, 0]})
        state = AnalysisState(data=df, processed_data=df.copy())
        state = extract_features(categorical_features=["model"])(state)
        with pytest.raises(ValueError, match="unknown coords"):
            set_dims(dims={"accuracy": ["benchmark"]})(state)


# ---------------------------------------------------------------------------
# format_probability
# ---------------------------------------------------------------------------


class TestFormatProbability:
    def test_ordinary_share(self):
        assert format_probability(np.array([True, False, False, False])) == "0.250"

    def test_never_true_is_floored(self):
        assert format_probability(np.zeros(4000, dtype=bool)) == "< 0.001"

    def test_always_true_is_floored(self):
        assert format_probability(np.ones(4000, dtype=bool)) == "> 0.999"

    def test_floor_is_one_over_n_for_few_draws(self):
        assert format_probability(np.zeros(100, dtype=bool)) == "< 0.01"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            format_probability(np.array([], dtype=bool))


# ---------------------------------------------------------------------------
# Model structure
# ---------------------------------------------------------------------------


class TestAccuracyComparisonStructure:
    def test_paired_sites_and_shapes(self):
        state = process(simulate_answers(num_items=20))
        tr = model_trace(state)
        assert tr["baseline"]["value"].shape == ()
        assert tr["difference"]["value"].shape == (1,)
        assert tr["item_sigma"]["value"].shape == ()
        assert tr["item_z"]["value"].shape == (20,)
        assert tr["accuracy"]["value"].shape == (2,)
        assert tr["accuracy_difference"]["value"].shape == (1,)
        assert isinstance(tr["obs"]["fn"], dist.BernoulliLogits)

    def test_unpaired_has_no_item_term(self):
        state = process(simulate_answers(num_items=20))
        tr = model_trace(state, item_column=None)
        assert "item_sigma" not in tr and "item_z" not in tr
        assert tr["accuracy"]["value"].shape == (2,)

    def test_unpaired_accuracy_is_sigmoid_of_logits(self):
        state = process(simulate_answers(num_items=20))
        tr = model_trace(state, item_column=None)
        b, d = tr["baseline"]["value"], tr["difference"]["value"][0]
        np.testing.assert_allclose(
            tr["accuracy"]["value"], expit(np.array([b, b + d])), rtol=1e-5
        )

    def test_paired_accuracy_averages_over_items(self):
        state = process(simulate_answers(num_items=20))
        tr = model_trace(state)
        offset = tr["item_sigma"]["value"] * tr["item_z"]["value"]
        b, d = tr["baseline"]["value"], tr["difference"]["value"][0]
        expected = [expit(b + offset).mean(), expit(b + d + offset).mean()]
        np.testing.assert_allclose(tr["accuracy"]["value"], expected, rtol=1e-5)
        np.testing.assert_allclose(
            tr["accuracy_difference"]["value"][0], expected[1] - expected[0], rtol=1e-4
        )

    def test_several_models_get_one_difference_each(self):
        data = simulate_answers(
            num_items=20, models=("ref", "b", "c"), differences=(-0.5, 0.5)
        )
        tr = model_trace(process(data))
        assert tr["difference"]["value"].shape == (2,)
        assert tr["accuracy"]["value"].shape == (3,)
        assert tr["accuracy_difference"]["value"].shape == (2,)

    def test_grouped_shapes(self):
        data = simulate_answers(num_items=15, groups=("g1", "g2", "g3"))
        tr = model_trace(process(data, grouped=True), group_column="benchmark")
        assert tr["baseline"]["value"].shape == (3,)
        assert tr["difference"]["value"].shape == (3, 1)
        assert tr["item_sigma"]["value"].shape == (3,)
        assert tr["item_z"]["value"].shape == (45,)
        assert tr["accuracy"]["value"].shape == (3, 2)
        assert tr["accuracy_difference"]["value"].shape == (3, 1)
        np.testing.assert_allclose(
            tr["mean_accuracy_difference"]["value"],
            tr["accuracy_difference"]["value"].mean(axis=0),
            rtol=1e-5,
        )

    def test_aggregated_counts_use_binomial(self):
        data = simulate_answers(num_items=20, epochs=3)
        tr = model_trace(process(data, aggregate=True))
        assert isinstance(tr["obs"]["fn"], dist.BinomialLogits)

    def test_reference_category_is_first(self):
        state = process(simulate_answers(num_items=5), reference="other")
        assert state.coords["model"][0] == "other"

    def test_single_model_raises(self):
        data = simulate_answers(num_items=5)
        state = process(data[data["model"] == "ref"])
        with pytest.raises(ValueError, match="at least two"):
            model_trace(state)

    def test_items_spanning_groups_raise(self):
        data = simulate_answers(num_items=5, groups=("g1", "g2"))
        state = AnalysisState(data=data, processed_data=data.copy())
        # task ids restart per group, so "task" alone is not a valid item id
        state.processed_data["item"] = state.processed_data["task"]
        state.processed_data["benchmark"] = state.processed_data["dataset"]
        state = extract_features(categorical_features=["model", "item", "benchmark"])(
            state
        )
        state = extract_observed_feature(feature_name="score")(state)
        with pytest.raises(ValueError, match="more than one group"):
            model_trace(state, group_column="benchmark")

    def test_missing_item_feature_raises(self):
        state = process(simulate_answers(num_items=5))
        with pytest.raises(ValueError, match="Missing required features"):
            model_trace(state, item_column="question")

    def test_custom_column_names(self):
        data = simulate_answers(num_items=10).rename(columns={"model": "llm"})
        state = AnalysisState(data=data, processed_data=data.copy())
        state = combine_columns(new_column="question", columns=["dataset", "task"])(
            state
        )
        state = extract_features(categorical_features=["llm", "question"])(state)
        state = extract_observed_feature(feature_name="score")(state)
        tr = model_trace(state, model_column="llm", item_column="question")
        assert tr["item_z"]["value"].shape == (10,)


class TestCoordsAndDims:
    def test_paired_dims(self):
        state = process(simulate_answers(num_items=5))
        coords, dims = model_coords_and_dims(
            accuracy_comparison(), state.coords, state.dims
        )
        assert coords["model_contrast"] == ["other - ref"]
        assert dims["accuracy"] == ["model"]
        assert dims["accuracy_difference"] == ["model_contrast"]
        assert dims["difference"] == ["model_contrast"]
        assert dims["item_z"] == ["item"]
        # the processed coords are not mutated
        assert "model_contrast" not in state.coords

    def test_grouped_dims(self):
        state = process(simulate_answers(num_items=5, groups=("a", "b")), grouped=True)
        _, dims = model_coords_and_dims(
            accuracy_comparison(group_column="benchmark"), state.coords, state.dims
        )
        assert dims["accuracy"] == ["benchmark", "model"]
        assert dims["accuracy_difference"] == ["benchmark", "model_contrast"]
        assert dims["baseline"] == ["benchmark"]
        assert dims["item_sigma"] == ["benchmark"]
        assert dims["mean_accuracy_difference"] == ["model_contrast"]

    def test_user_dims_win(self):
        state = process(simulate_answers(num_items=5))
        state.dims["accuracy"] = ["model"]
        state.dims["difference"] = ["model"]  # deliberately different
        _, dims = model_coords_and_dims(accuracy_comparison(), state.coords, state.dims)
        assert dims["difference"] == ["model"]

    def test_dims_without_coords_are_dropped(self):
        coords, dims = model_coords_and_dims(accuracy_comparison(), {}, {})
        assert dims == {} and coords == {}

    def test_models_without_hook_are_unchanged(self):
        coords, dims = {"a": [1]}, {"x": ["a"]}
        assert model_coords_and_dims(two_level_group_binomial(), coords, dims) == (
            coords,
            dims,
        )


# ---------------------------------------------------------------------------
# Fitting: parameter recovery on simulated data
# ---------------------------------------------------------------------------


class TestAccuracyComparisonFit:
    def test_recovers_parameters(self, paired_fit):
        _, mas = paired_fit
        post = az.extract(mas.inference_data)
        for name, truth in [("baseline", 0.3), ("item_sigma", 1.5)]:
            lo, hi = np.quantile(post[name].values, [0.005, 0.995])
            assert lo < truth < hi, name
        lo, hi = np.quantile(post["difference"].values.ravel(), [0.005, 0.995])
        assert lo < -0.8 < hi

    def test_accuracy_matches_observed(self, paired_fit):
        state, mas = paired_fit
        observed = state.processed_data.groupby("model")["score"].mean()
        acc = az.extract(mas.inference_data)["accuracy"].mean("sample")
        np.testing.assert_allclose(acc.sel(model="ref"), observed["ref"], atol=0.03)
        np.testing.assert_allclose(acc.sel(model="other"), observed["other"], atol=0.03)

    def test_inference_data_is_labelled(self, paired_fit):
        _, mas = paired_fit
        post = mas.inference_data.posterior
        assert list(post["accuracy"].coords["model"].values) == ["ref", "other"]
        assert list(post["accuracy_difference"].coords["model_contrast"].values) == [
            "other - ref"
        ]

    def test_paired_interval_is_narrower_than_unpaired(self, paired_fit):
        state, paired = paired_fit
        unpaired = fit_model(state, item_column=None)

        def width(mas):
            d = az.extract(mas.inference_data)["accuracy_difference"].values.ravel()
            lo, hi = np.quantile(d, [0.025, 0.975])
            return hi - lo

        assert width(paired) < width(unpaired)

    def test_aggregated_and_row_level_agree(self):
        data = simulate_answers(num_items=150, epochs=3, seed_value=3)
        rows = fit_model(process(data), samples=300)
        counts = fit_model(process(data, aggregate=True), samples=300)
        a = az.extract(rows.inference_data)["accuracy_difference"].values.mean()
        b = az.extract(counts.inference_data)["accuracy_difference"].values.mean()
        assert abs(a - b) < 0.02


# ---------------------------------------------------------------------------
# comparison_summary
# ---------------------------------------------------------------------------


def _fake_comparison_state(grouped: bool = False, n: int = 1000) -> AnalysisState:
    """State with a hand-made accuracy_comparison posterior."""
    rng = np.random.default_rng(0)
    models, contrasts = ["ref", "b", "c"], ["b - ref", "c - ref"]
    groups = ["g1", "g2"]
    shape = (1, n) + ((2,) if grouped else ())
    acc = np.stack(
        [
            np.full(shape, 0.5),
            0.5 + 0.10 + 0.01 * rng.standard_normal(shape),  # clearly better
            0.5 + 0.002 * rng.standard_normal(shape),  # equivalent
        ],
        axis=-1,
    )
    diff = acc[..., 1:] - acc[..., :1]
    g = ["benchmark"] if grouped else []
    posterior = {"accuracy": acc, "accuracy_difference": diff}
    dims = {"accuracy": g + ["model"], "accuracy_difference": g + ["model_contrast"]}
    coords = {"model": models, "model_contrast": contrasts}
    if grouped:
        posterior["mean_accuracy_difference"] = diff.mean(axis=2)
        dims["mean_accuracy_difference"] = ["model_contrast"]
        coords["benchmark"] = groups
    idata = az.from_dict(posterior=posterior, coords=coords, dims=dims)
    mas = ModelAnalysisState(
        model=accuracy_comparison(),
        model_config=ModelConfig.from_dict({}),
        inference_data=idata,
        is_fitted=True,
    )
    state = AnalysisState(data=pd.DataFrame())
    state.add_model(mas)
    return state


class TestComparisonSummary:
    def test_one_row_per_compared_model(self):
        state, result = comparison_summary()(_fake_comparison_state())
        table = state.communicate["comparison_summary"]
        assert result == "pass"
        assert table["compared"].tolist() == ["b", "c"]
        assert set(table["reference"]) == {"ref"}
        b, c = table.iloc[0], table.iloc[1]
        assert b["difference (mean)"] == pytest.approx(0.10, abs=0.005)
        assert b["P(compared better)"] == "> 0.999"
        assert c["P(|difference| < 0.01)"] == "> 0.999"
        assert b["interval width"] == pytest.approx(
            b["difference 95% upper"] - b["difference 95% lower"], abs=1e-4
        )

    def test_grouped_adds_average_rows(self):
        state, _ = comparison_summary()(_fake_comparison_state(grouped=True))
        table = state.communicate["comparison_summary"]
        assert table["group"].tolist() == [
            "g1",
            "g2",
            "average over groups",
            "g1",
            "g2",
            "average over groups",
        ]

    def test_credible_mass_changes_column_names(self):
        state, _ = comparison_summary(credible_mass=0.9)(_fake_comparison_state())
        assert "difference 90% lower" in state.communicate["comparison_summary"]

    def test_invalid_credible_mass(self):
        with pytest.raises(ValueError):
            comparison_summary(credible_mass=1.5)

    def test_na_without_comparison_model(self):
        idata = az.from_dict(posterior={"mu": np.zeros((1, 10))})
        mas = ModelAnalysisState(
            model=two_level_group_binomial(),
            model_config=ModelConfig.from_dict({}),
            inference_data=idata,
            is_fitted=True,
        )
        state = AnalysisState(data=pd.DataFrame())
        state.add_model(mas)
        _, result = comparison_summary()(state)
        assert result == "NA"

    def test_on_real_fit(self, paired_fit):
        state, _ = paired_fit
        state, result = comparison_summary()(state)
        assert result == "pass"
        table = state.communicate["comparison_summary"]
        assert (table["compared"] == "other").all()


# ---------------------------------------------------------------------------
# agreement_check
# ---------------------------------------------------------------------------


def _state_with_posterior(state: AnalysisState, baseline, difference, sigma, n=200):
    """ModelAnalysisState whose posterior is a point mass at given values."""
    model = accuracy_comparison()
    coords, dims = model_coords_and_dims(model, state.coords, state.dims)
    posterior = {
        "baseline": np.full((1, n), baseline),
        "difference": np.full((1, n, 1), difference),
        "item_sigma": np.full((1, n), sigma),
    }
    idata = az.from_dict(
        posterior=posterior,
        coords={"model_contrast": coords["model_contrast"]},
        dims={"difference": ["model_contrast"]},
    )
    return ModelAnalysisState(
        model=model,
        model_config=ModelConfig.from_dict({}),
        features=state.features,
        coords=coords,
        dims=dims,
        inference_data=idata,
        is_fitted=True,
    )


class TestAgreementCheck:
    def test_passes_at_true_parameters(self):
        state = process(simulate_answers(num_items=500, seed_value=0))
        mas = _state_with_posterior(state, 0.3, -0.8, 1.5)
        mas, result = agreement_check()(mas)
        assert result == "pass"
        table = mas.diagnostic("agreement_check_table")
        assert len(table) == 4 and table["inside"].all()
        assert mas.diagnostic("agreement_check") is not None

    def test_intervals_are_calibrated(self):
        # At the true parameters each observed share should fall inside its
        # 95% interval about 95% of the time.
        inside = []
        for s in range(20):
            state = process(simulate_answers(num_items=300, seed_value=100 + s))
            mas = _state_with_posterior(state, 0.3, -0.8, 1.5)
            mas, _ = agreement_check(num_sims=200)(mas)
            inside.extend(mas.diagnostic("agreement_check_table")["inside"])
        assert 0.85 <= np.mean(inside) <= 1.0

    def test_fails_when_item_spread_is_wrong(self):
        # Data with large item spread, posterior claiming none: the models
        # agree far more often than independent answers would.
        state = process(simulate_answers(num_items=500, item_sigma=3.0))
        mas = _state_with_posterior(state, 0.3, -0.8, 0.0)
        _, result = agreement_check()(mas)
        assert result == "fail"

    def test_handles_epochs(self):
        state = process(simulate_answers(num_items=300, epochs=4, seed_value=5))
        mas = _state_with_posterior(state, 0.3, -0.8, 1.5)
        _, result = agreement_check()(mas)
        assert result == "pass"

    def test_handles_aggregated_counts(self):
        state = process(
            simulate_answers(num_items=300, epochs=4, seed_value=5), aggregate=True
        )
        mas = _state_with_posterior(state, 0.3, -0.8, 1.5)
        _, result = agreement_check()(mas)
        assert result == "pass"

    def test_na_for_unpaired_fit(self):
        state = process(simulate_answers(num_items=20))
        mas = ModelAnalysisState(
            model=accuracy_comparison(item_column=None),
            model_config=ModelConfig.from_dict({}),
            features=state.features,
            inference_data=az.from_dict(posterior={"baseline": np.zeros((1, 5))}),
            is_fitted=True,
        )
        _, result = agreement_check()(mas)
        assert result == "NA"

    def test_na_for_other_models(self):
        mas = ModelAnalysisState(
            model=two_level_group_binomial(),
            model_config=ModelConfig.from_dict({}),
            inference_data=az.from_dict(posterior={"mu": np.zeros((1, 5))}),
            is_fitted=True,
        )
        _, result = agreement_check()(mas)
        assert result == "NA"

    def test_on_real_fit(self, paired_fit):
        _, mas = paired_fit
        _, result = agreement_check()(mas)
        assert result == "pass"


# ---------------------------------------------------------------------------
# Through the model stage, as a config would run it
# ---------------------------------------------------------------------------


def test_model_stage_applies_model_dims_and_runs_check():
    from hibayes.analysis import model as run_model_stage
    from hibayes.check import CheckerConfig
    from hibayes.model import ModelsToRunConfig

    from hibayes.ui import ModellingDisplay

    display = ModellingDisplay()
    state = process(
        simulate_answers(num_items=40, groups=("a", "b"), seed_value=7), grouped=True
    )
    fit_cfg = {"samples": 200, "warmup": 200, "chains": 1, "seed": 0}
    models = ModelsToRunConfig.from_dict(
        {
            "models": [
                {
                    "name": "accuracy_comparison",
                    "config": {"group_column": "benchmark", "fit": fit_cfg},
                }
            ]
        }
    )
    checks = CheckerConfig.from_dict({"checkers": ["divergences", "agreement_check"]})
    state = run_model_stage(
        analysis_state=state,
        models_to_run_config=models,
        checker_config=checks,
        platform_config=PlatformConfig(chain_method="sequential"),
        display=display,
    )
    mas = state.models[0]
    post = mas.inference_data.posterior
    assert post["accuracy"].dims[2:] == ("benchmark", "model")
    assert list(post["model_contrast"].values) == ["other - ref"]
    assert mas.diagnostic("agreement_check_table") is not None
    # the processed state's coords are untouched
    assert "model_contrast" not in state.coords

    state, result = comparison_summary()(state)
    table = state.communicate["comparison_summary"]
    assert table["group"].tolist() == ["a", "b", "average over groups"]
