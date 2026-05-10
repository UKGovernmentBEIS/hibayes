"""Tests for LOO/WAIC pointwise diagnostic export.

Regression tests for issue #77: the ELPDData summary returned by ``arviz.loo``
and ``arviz.waic`` stringifies its xarray DataArray fields when written to CSV,
so the per-observation values (``loo_i``, ``pareto_k``, ``waic_i``) used to be
unrecoverable after a HiBayes run. The checkers now store those arrays as
separate diagnostics and ``ModelAnalysisState.save`` writes each as its own
CSV.
"""

from __future__ import annotations

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hibayes.analysis_state import ModelAnalysisState
from hibayes.check.checkers import loo, waic
from hibayes.model import ModelConfig


def _dummy_model(features):
    """Picklable stub model — save() pickles the model attribute."""
    pass


def _make_inference_data(seed: int = 0, n_obs: int = 30) -> az.InferenceData:
    """Build a tiny InferenceData with posterior + log_likelihood.

    arviz.loo / arviz.waic only need ``log_likelihood``; the posterior is
    included so the object resembles a real fit.
    """
    rng = np.random.default_rng(seed)
    chains, draws = 2, 200
    log_lik = rng.normal(loc=-1.0, scale=0.1, size=(chains, draws, n_obs))
    posterior = rng.normal(size=(chains, draws))
    return az.from_dict(
        posterior={"theta": posterior},
        log_likelihood={"obs": log_lik},
        coords={"obs_dim_0": np.arange(n_obs)},
        dims={"obs": ["obs_dim_0"]},
    )


@pytest.fixture
def fitted_state() -> ModelAnalysisState:
    return ModelAnalysisState(
        model=_dummy_model,
        model_config=ModelConfig(),
        inference_data=_make_inference_data(),
    )


class TestLooDiagnostics:
    def test_loo_stores_pointwise_arrays(self, fitted_state: ModelAnalysisState):
        """loo_i and pareto_k must land in diagnostics as xarray DataArrays
        — not just embedded inside the ELPDData summary."""
        loo()(fitted_state)

        assert "loo_i" in fitted_state.diagnostics
        assert "pareto_k" in fitted_state.diagnostics
        assert isinstance(fitted_state.diagnostics["loo_i"], xr.DataArray)
        assert isinstance(fitted_state.diagnostics["pareto_k"], xr.DataArray)

    def test_loo_save_writes_pointwise_csvs(
        self, fitted_state: ModelAnalysisState, tmp_path: Path
    ):
        loo()(fitted_state)
        fitted_state.save(tmp_path)

        diag_dir = tmp_path / "diagnostics"
        assert (diag_dir / "inference_data_loo.csv").exists(), (
            "summary CSV should still be written"
        )
        assert (diag_dir / "inference_data_loo_i.csv").exists()
        assert (diag_dir / "inference_data_pareto_k.csv").exists()

    def test_pointwise_csvs_contain_per_observation_values(
        self, fitted_state: ModelAnalysisState, tmp_path: Path
    ):
        """The CSVs hold a row per observation with finite numeric values
        — regression for the bug where the data was stringified."""
        loo()(fitted_state)
        fitted_state.save(tmp_path)

        diag_dir = tmp_path / "diagnostics"
        n_obs = fitted_state.inference_data.log_likelihood.sizes["obs_dim_0"]

        loo_i_df = pd.read_csv(diag_dir / "inference_data_loo_i.csv", index_col=0)
        pareto_df = pd.read_csv(diag_dir / "inference_data_pareto_k.csv", index_col=0)

        assert len(loo_i_df) == n_obs
        assert len(pareto_df) == n_obs
        assert np.isfinite(loo_i_df.to_numpy(dtype=float)).all()
        assert np.isfinite(pareto_df.to_numpy(dtype=float)).all()

    def test_csv_values_match_in_memory_diagnostic(
        self, fitted_state: ModelAnalysisState, tmp_path: Path
    ):
        loo()(fitted_state)
        fitted_state.save(tmp_path)

        loaded = pd.read_csv(
            tmp_path / "diagnostics" / "inference_data_loo_i.csv", index_col=0
        )
        expected = fitted_state.diagnostics["loo_i"].to_dataframe()
        np.testing.assert_allclose(
            loaded.to_numpy(dtype=float), expected.to_numpy(dtype=float)
        )


class TestWaicDiagnostics:
    def test_waic_stores_pointwise_array(self, fitted_state: ModelAnalysisState):
        waic()(fitted_state)

        assert "waic_i" in fitted_state.diagnostics
        assert isinstance(fitted_state.diagnostics["waic_i"], xr.DataArray)

    def test_waic_save_writes_pointwise_csv(
        self, fitted_state: ModelAnalysisState, tmp_path: Path
    ):
        waic()(fitted_state)
        fitted_state.save(tmp_path)

        diag_dir = tmp_path / "diagnostics"
        assert (diag_dir / "inference_data_waic.csv").exists()
        assert (diag_dir / "inference_data_waic_i.csv").exists()

    def test_waic_csv_contains_per_observation_values(
        self, fitted_state: ModelAnalysisState, tmp_path: Path
    ):
        waic()(fitted_state)
        fitted_state.save(tmp_path)

        n_obs = fitted_state.inference_data.log_likelihood.sizes["obs_dim_0"]
        waic_i_df = pd.read_csv(
            tmp_path / "diagnostics" / "inference_data_waic_i.csv", index_col=0
        )

        assert len(waic_i_df) == n_obs
        assert np.isfinite(waic_i_df.to_numpy(dtype=float)).all()


class TestIncrementalSave:
    def test_incremental_save_writes_pointwise_csvs(
        self, fitted_state: ModelAnalysisState, tmp_path: Path
    ):
        """Diagnostics are always rewritten on save — incremental or not —
        because they can change during ``communicate``."""
        loo()(fitted_state)
        waic()(fitted_state)
        fitted_state.save(tmp_path, incremental=True)

        diag_dir = tmp_path / "diagnostics"
        for fname in (
            "inference_data_loo_i.csv",
            "inference_data_pareto_k.csv",
            "inference_data_waic_i.csv",
        ):
            assert (diag_dir / fname).exists(), f"missing {fname} after incremental save"

    def test_incremental_save_refreshes_changed_pointwise_values(
        self, tmp_path: Path
    ):
        """If diagnostics change between two incremental saves, the CSVs on
        disk must reflect the new values — not the stale ones."""
        state = ModelAnalysisState(
            model=_dummy_model,
            model_config=ModelConfig(),
            inference_data=_make_inference_data(seed=0),
        )
        loo()(state)
        state.save(tmp_path, incremental=False)

        first = pd.read_csv(
            tmp_path / "diagnostics" / "inference_data_loo_i.csv", index_col=0
        ).to_numpy(dtype=float)

        # Recompute against different data and incremental-save again.
        state._inference_data = _make_inference_data(seed=1)
        # Force recomputation by clearing the cached entry the checker reads.
        del state._diagnostics["loo"]
        loo()(state)
        state.save(tmp_path, incremental=True)

        second = pd.read_csv(
            tmp_path / "diagnostics" / "inference_data_loo_i.csv", index_col=0
        ).to_numpy(dtype=float)

        assert not np.allclose(first, second), (
            "incremental save did not refresh loo_i CSV"
        )
        np.testing.assert_allclose(
            second, state.diagnostics["loo_i"].to_dataframe().to_numpy(dtype=float)
        )


class TestRoundtrip:
    def test_load_recovers_pointwise_data(
        self, fitted_state: ModelAnalysisState, tmp_path: Path
    ):
        """After save+load the per-observation values are recoverable as a
        DataFrame (the load path reads ``inference_data_<name>.csv`` for any
        diagnostic key that has a CSV on disk)."""
        loo()(fitted_state)
        waic()(fitted_state)
        fitted_state.save(tmp_path)

        reloaded = ModelAnalysisState.load(tmp_path)

        for key in ("loo_i", "pareto_k", "waic_i"):
            assert key in reloaded.diagnostics
            assert isinstance(reloaded.diagnostics[key], pd.DataFrame)

        n_obs = fitted_state.inference_data.log_likelihood.sizes["obs_dim_0"]
        assert len(reloaded.diagnostics["loo_i"]) == n_obs
        np.testing.assert_allclose(
            reloaded.diagnostics["loo_i"].to_numpy(dtype=float),
            fitted_state.diagnostics["loo_i"].to_dataframe().to_numpy(dtype=float),
        )

    def test_save_load_save_preserves_pointwise_data(
        self, fitted_state: ModelAnalysisState, tmp_path: Path
    ):
        """Mirrors the ``hibayes-model`` → ``hibayes-comm`` flow: model stage
        saves DataArray diagnostics, communicate stage loads them as DataFrames
        and saves again incrementally. The pointwise CSVs must survive the
        second save.
        """
        loo()(fitted_state)
        waic()(fitted_state)
        fitted_state.save(tmp_path)

        reloaded = ModelAnalysisState.load(tmp_path)
        # On reload these are pd.DataFrame, so save() must fall through to the
        # DataFrame/Series branch — not the DataArray branch — and still write
        # a valid CSV.
        reloaded.save(tmp_path, incremental=True)

        # All three CSVs must still exist with the same length and values
        # (modulo float CSV roundtrip noise).
        n_obs = fitted_state.inference_data.log_likelihood.sizes["obs_dim_0"]
        for key in ("loo_i", "pareto_k", "waic_i"):
            csv = tmp_path / "diagnostics" / f"inference_data_{key}.csv"
            assert csv.exists(), f"{key} CSV missing after save→load→save"
            df = pd.read_csv(csv, index_col=0)
            assert len(df) == n_obs

        # Numeric content should be preserved through the roundtrip.
        first = fitted_state.diagnostics["loo_i"].to_dataframe().to_numpy(dtype=float)
        second = pd.read_csv(
            tmp_path / "diagnostics" / "inference_data_loo_i.csv", index_col=0
        ).to_numpy(dtype=float)
        np.testing.assert_allclose(first, second, rtol=1e-12, atol=1e-12)
