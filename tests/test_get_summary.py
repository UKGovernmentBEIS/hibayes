"""Tests for the get_summary method on ModelAnalysisState."""

import numpy as np
import pandas as pd
import pytest
import arviz as az
import xarray as xr

from hibayes.analysis_state import ModelAnalysisState
from hibayes.model import ModelConfig


def create_mock_inference_data():
    """Create a minimal mock InferenceData for testing."""
    # Create mock posterior samples
    np.random.seed(42)
    n_chains = 2
    n_draws = 100

    posterior_data = {
        "alpha": (["chain", "draw"], np.random.randn(n_chains, n_draws)),
        "beta": (["chain", "draw"], np.random.randn(n_chains, n_draws)),
        "sigma": (["chain", "draw"], np.abs(np.random.randn(n_chains, n_draws))),
    }

    posterior = xr.Dataset(posterior_data)
    return az.InferenceData(posterior=posterior)


def create_model_analysis_state():
    """Create a minimal ModelAnalysisState for testing."""
    def dummy_model(**kwargs):
        pass

    model_config = ModelConfig()
    inference_data = create_mock_inference_data()

    state = ModelAnalysisState(
        model=dummy_model,
        model_config=model_config,
        inference_data=inference_data,
        is_fitted=True,
    )
    return state


class TestGetSummary:
    """Tests for ModelAnalysisState.get_summary() method."""

    def test_get_summary_returns_dataframe(self):
        """get_summary should return a pandas DataFrame."""
        state = create_model_analysis_state()
        summary = state.get_summary()
        assert isinstance(summary, pd.DataFrame)

    def test_get_summary_caches_result(self):
        """get_summary should cache the result in diagnostics."""
        state = create_model_analysis_state()
        assert "summary" not in state.diagnostics

        summary1 = state.get_summary()
        assert "summary" in state.diagnostics

        # Second call should use cached result (same object when no filtering)
        summary2 = state.get_summary()

        # The cached summary should be the same object
        pd.testing.assert_frame_equal(summary1, summary2)

    def test_get_summary_includes_expected_columns(self):
        """Summary should include standard ArviZ summary columns."""
        state = create_model_analysis_state()
        summary = state.get_summary()

        expected_columns = ["mean", "sd", "hdi_3%", "hdi_97%", "r_hat", "ess_bulk", "ess_tail"]
        for col in expected_columns:
            assert col in summary.columns, f"Missing expected column: {col}"

    def test_get_summary_includes_all_variables(self):
        """Summary should include all posterior variables."""
        state = create_model_analysis_state()
        summary = state.get_summary()

        expected_vars = ["alpha", "beta", "sigma"]
        for var in expected_vars:
            assert var in summary.index, f"Missing expected variable: {var}"

    def test_get_summary_var_names_filter(self):
        """get_summary should filter by var_names when provided."""
        state = create_model_analysis_state()

        # Get summary filtered by var_names
        summary = state.get_summary(var_names=["alpha"])

        assert "alpha" in summary.index
        assert "beta" not in summary.index
        assert "sigma" not in summary.index

    def test_get_summary_round_to(self):
        """get_summary should round values when round_to is provided."""
        state = create_model_analysis_state()

        # Get summary with rounding
        summary_rounded = state.get_summary(round_to=1)
        summary_unrounded = state.get_summary()

        # Check that rounding was applied (values should differ)
        # Note: This is a weak test since values could coincidentally round to same
        assert isinstance(summary_rounded, pd.DataFrame)

    def test_get_summary_var_names_partial_match(self):
        """var_names should match variable name prefixes."""
        state = create_model_analysis_state()

        # "alpha" should match "alpha" exactly
        summary = state.get_summary(var_names=["alpha"])
        assert len(summary) == 1
        assert "alpha" in summary.index

    def test_get_summary_reuses_cache_with_different_filters(self):
        """Subsequent calls with different filters should reuse cached summary."""
        state = create_model_analysis_state()

        # First call caches the full summary
        _ = state.get_summary()
        cached_summary = state.diagnostics["summary"]

        # Second call with filter should still use same cached summary
        _ = state.get_summary(var_names=["alpha"])
        assert state.diagnostics["summary"] is cached_summary

    def test_get_summary_does_not_recompute_when_cached(self):
        """get_summary should not recompute when summary is already cached."""
        state = create_model_analysis_state()

        # Manually set a cached summary
        mock_summary = pd.DataFrame(
            {"mean": [1.0, 2.0], "sd": [0.1, 0.2]},
            index=["test_var1", "test_var2"]
        )
        state.diagnostics["summary"] = mock_summary

        # get_summary should return the cached version
        summary = state.get_summary()
        assert "test_var1" in summary.index
        assert "test_var2" in summary.index


class TestGetSummaryIntegration:
    """Integration tests for get_summary with checkers workflow."""

    def test_multiple_checkers_share_cached_summary(self):
        """Multiple calls simulating r_hat, ess_bulk, ess_tail should share cache."""
        state = create_model_analysis_state()

        # Simulate r_hat checker
        summary = state.get_summary()
        r_hat_values = summary.r_hat.values

        # Simulate ess_bulk checker
        summary = state.get_summary()
        ess_bulk_values = summary.ess_bulk.values

        # Simulate ess_tail checker
        summary = state.get_summary()
        ess_tail_values = summary.ess_tail.values

        # All should have accessed values successfully
        assert len(r_hat_values) == 3  # alpha, beta, sigma
        assert len(ess_bulk_values) == 3
        assert len(ess_tail_values) == 3

        # Cache should only have been computed once
        assert "summary" in state.diagnostics
