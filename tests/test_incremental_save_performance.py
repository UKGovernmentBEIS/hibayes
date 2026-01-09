"""Test incremental save performance vs full save."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from arviz import InferenceData
import xarray as xr

from hibayes.analysis_state import AnalysisState, ModelAnalysisState
from hibayes.model import ModelConfig


def _dummy_model(features):
    """A simple picklable model function for testing."""
    pass


def create_mock_inference_data(n_samples: int = 2000, n_chains: int = 4, n_params: int = 50) -> InferenceData:
    """Create a mock InferenceData object with realistic sizes."""
    # Create posterior samples - this is the heavy part
    posterior_data = {}
    for i in range(n_params):
        posterior_data[f"param_{i}"] = (
            ["chain", "draw"],
            np.random.randn(n_chains, n_samples)
        )

    posterior = xr.Dataset(posterior_data)

    return InferenceData(posterior=posterior)


def create_test_analysis_state(
    n_rows: int = 10000,
    n_models: int = 3,
    n_samples: int = 2000,
    n_chains: int = 4,
    n_params: int = 50,
) -> AnalysisState:
    """Create a test AnalysisState with realistic data sizes."""
    # Create data
    data = pd.DataFrame({
        "col1": np.random.randn(n_rows),
        "col2": np.random.randn(n_rows),
        "col3": np.random.randint(0, 100, n_rows),
        "category": np.random.choice(["A", "B", "C", "D"], n_rows),
    })

    # Create features
    features = {
        "obs": np.random.randint(0, 2, n_rows),
        "n_total": np.ones(n_rows) * 10,
        "group_index": np.random.randint(0, 5, n_rows),
        "num_group": 5,
    }

    coords = {"group": ["A", "B", "C", "D", "E"]}
    dims = {"group_effects": ["group"]}

    analysis_state = AnalysisState(
        data=data,
        processed_data=data.copy(),
        features=features,
        coords=coords,
        dims=dims,
    )

    # Create mock models
    for i in range(n_models):
        model_state = ModelAnalysisState(
            model=_dummy_model,
            model_config=ModelConfig(tag=f"v{i}"),
            features=features.copy(),
            coords=coords,
            dims=dims,
            inference_data=create_mock_inference_data(n_samples, n_chains, n_params),
            diagnostics={"rhat": 1.01, "ess": 1000},
            is_fitted=True,
        )
        analysis_state.add_model(model_state)

    return analysis_state


class TestIncrementalSavePerformance:
    """Test that incremental save is faster than full save."""

    @pytest.mark.slow
    def test_incremental_save_faster_than_full_save(self, tmp_path: Path):
        """Test that incremental save is significantly faster after initial save."""
        # Create test state with realistic sizes
        analysis_state = create_test_analysis_state(
            n_rows=5000,
            n_models=2,
            n_samples=1000,
            n_chains=2,
            n_params=20,
        )

        out_path = tmp_path / "analysis_output"

        # First save (full) - establishes baseline files
        start = time.perf_counter()
        analysis_state.save(out_path, incremental=False)
        first_full_save_time = time.perf_counter() - start

        # Second full save
        start = time.perf_counter()
        analysis_state.save(out_path, incremental=False)
        second_full_save_time = time.perf_counter() - start

        # Incremental save
        start = time.perf_counter()
        analysis_state.save(out_path, incremental=True)
        incremental_save_time = time.perf_counter() - start

        print(f"\nFirst full save: {first_full_save_time:.3f}s")
        print(f"Second full save: {second_full_save_time:.3f}s")
        print(f"Incremental save: {incremental_save_time:.3f}s")
        print(f"Speedup: {second_full_save_time / incremental_save_time:.1f}x")

        # Incremental should be faster than full save
        # We expect significant speedup since we skip data.parquet, features.pkl, model.pkl
        assert incremental_save_time < second_full_save_time, (
            f"Incremental save ({incremental_save_time:.3f}s) should be faster than "
            f"full save ({second_full_save_time:.3f}s)"
        )

    @pytest.mark.slow
    def test_incremental_save_speedup_factor(self, tmp_path: Path):
        """Test that incremental save achieves at least 10x speedup when data is unchanged.

        This test verifies the performance benefit of incremental saves by measuring
        the time taken for full saves vs incremental saves when no data has changed.
        """
        # Create test state with larger sizes to make timing differences more pronounced
        analysis_state = create_test_analysis_state(
            n_rows=10000,
            n_models=3,
            n_samples=2000,
            n_chains=4,
            n_params=50,
        )

        out_path = tmp_path / "analysis_output"

        # First full save to establish baseline files
        analysis_state.save(out_path, incremental=False)

        # Measure full save time (multiple runs for stability)
        full_save_times = []
        for _ in range(3):
            start = time.perf_counter()
            analysis_state.save(out_path, incremental=False)
            full_save_times.append(time.perf_counter() - start)
        avg_full_save_time = sum(full_save_times) / len(full_save_times)

        # Measure incremental save time (multiple runs for stability)
        incremental_save_times = []
        for _ in range(3):
            start = time.perf_counter()
            analysis_state.save(out_path, incremental=True)
            incremental_save_times.append(time.perf_counter() - start)
        avg_incremental_save_time = sum(incremental_save_times) / len(incremental_save_times)

        speedup = avg_full_save_time / avg_incremental_save_time

        print(f"\nAverage full save time: {avg_full_save_time:.3f}s")
        print(f"Average incremental save time: {avg_incremental_save_time:.3f}s")
        print(f"Speedup factor: {speedup:.1f}x")

        # We expect at least 10x speedup when nothing has changed
        # This is because incremental saves skip:
        # - data.parquet, processed_data.parquet
        # - features.pkl, test_features.pkl
        # - model.pkl for each model
        # - inference_data.nc for each model (when signature unchanged)
        min_expected_speedup = 10
        assert speedup >= min_expected_speedup, (
            f"Expected at least {min_expected_speedup}x speedup, but got {speedup:.1f}x. "
            f"Full save: {avg_full_save_time:.3f}s, Incremental: {avg_incremental_save_time:.3f}s"
        )

    @pytest.mark.slow
    def test_frequent_incremental_saves_vs_single_save_at_end(self, tmp_path: Path):
        """Compare total time of frequent incremental saves vs single save at end.

        This compares:
        - OLD behavior: No frequent saves, single full save at the very end
        - NEW behavior: Frequent incremental saves after each model/communicator

        We want to measure the overhead of doing multiple incremental saves compared
        to the old approach of saving once at the end.
        """
        n_models = 3
        n_communicators = 5  # Simulating 5 communicator runs

        analysis_state = create_test_analysis_state(
            n_rows=10000,
            n_models=n_models,
            n_samples=2000,
            n_chains=4,
            n_params=50,
        )

        # --- OLD BEHAVIOR: Single full save at the very end ---
        out_path_old = tmp_path / "old_behavior_single_save"
        start = time.perf_counter()
        analysis_state.save(out_path_old, incremental=False)
        old_behavior_time = time.perf_counter() - start

        # --- NEW BEHAVIOR: Frequent incremental saves after each model/communicator ---
        out_path_new = tmp_path / "new_behavior_frequent_incremental"
        total_saves = n_models + n_communicators  # Save after each model and communicator

        start = time.perf_counter()
        # First save is full (establishes the baseline files)
        analysis_state.save(out_path_new, incremental=False)
        # Subsequent saves are incremental (after each model fit / communicator)
        for _ in range(total_saves - 1):
            analysis_state.save(out_path_new, incremental=True)
        new_behavior_time = time.perf_counter() - start

        overhead = new_behavior_time / old_behavior_time

        print(f"\n=== Old vs New Behavior Comparison ===")
        print(f"Scenario: {n_models} models + {n_communicators} communicators = {total_saves} save points")
        print(f"")
        print(f"OLD (single save at end):              {old_behavior_time:.3f}s")
        print(f"NEW (frequent incremental, {total_saves}x saves): {new_behavior_time:.3f}s")
        print(f"")
        print(f"Overhead of new behavior: {overhead:.2f}x ({(overhead - 1) * 100:.1f}% slower)")

        # The new behavior with frequent incremental saves should have minimal overhead
        # compared to single save at end (less than 50% slower is acceptable for the
        # benefit of not losing progress on crashes)
        max_acceptable_overhead = 1.5
        assert overhead < max_acceptable_overhead, (
            f"Frequent incremental saves overhead ({overhead:.2f}x) exceeds "
            f"acceptable threshold ({max_acceptable_overhead}x). "
            f"Old: {old_behavior_time:.3f}s, New: {new_behavior_time:.3f}s"
        )

    def test_incremental_save_produces_same_loadable_state(self, tmp_path: Path):
        """Test that state saved incrementally can be loaded correctly."""
        analysis_state = create_test_analysis_state(
            n_rows=1000,
            n_models=1,
            n_samples=100,
            n_chains=2,
            n_params=5,
        )

        out_path = tmp_path / "analysis_output"

        # Save full first
        analysis_state.save(out_path, incremental=False)

        # Then incremental
        analysis_state.save(out_path, incremental=True)

        # Load and verify
        loaded = AnalysisState.load(out_path)

        assert len(loaded.data) == len(analysis_state.data)
        assert len(loaded.models) == len(analysis_state.models)
        assert loaded.models[0].is_fitted == analysis_state.models[0].is_fitted

    def test_first_incremental_save_creates_all_files(self, tmp_path: Path):
        """Test that first save with incremental=True still creates all files."""
        analysis_state = create_test_analysis_state(
            n_rows=100,
            n_models=1,
            n_samples=50,
            n_chains=1,
            n_params=3,
        )

        out_path = tmp_path / "analysis_output"

        # First save with incremental=True should still create everything
        # since files don't exist yet
        analysis_state.save(out_path, incremental=True)

        # Verify all essential files exist
        assert (out_path / "data.parquet").exists()
        assert (out_path / "features.pkl").exists()
        assert (out_path / "models").exists()

        model_dir = out_path / "models" / analysis_state.models[0].model_name
        assert (model_dir / "inference_data.nc").exists()
        assert (model_dir / "model.pkl").exists()


class TestIncrementalSaveCorrectness:
    """Test that incremental save correctly skips unchanged files."""

    def test_incremental_skips_data_parquet(self, tmp_path: Path):
        """Test that incremental save skips data.parquet if it exists."""
        analysis_state = create_test_analysis_state(n_rows=100, n_models=1, n_samples=50, n_chains=1, n_params=3)
        out_path = tmp_path / "analysis_output"

        # First full save
        analysis_state.save(out_path, incremental=False)
        original_mtime = (out_path / "data.parquet").stat().st_mtime

        # Small delay to ensure mtime would differ
        time.sleep(0.01)

        # Incremental save
        analysis_state.save(out_path, incremental=True)
        new_mtime = (out_path / "data.parquet").stat().st_mtime

        # File should not have been modified
        assert original_mtime == new_mtime, "data.parquet should not be rewritten on incremental save"

    def test_incremental_skips_unchanged_inference_data(self, tmp_path: Path):
        """Test that incremental save skips inference_data when unchanged."""
        analysis_state = create_test_analysis_state(n_rows=100, n_models=1, n_samples=50, n_chains=1, n_params=3)
        out_path = tmp_path / "analysis_output"

        # First full save
        analysis_state.save(out_path, incremental=False)

        model_dir = out_path / "models" / analysis_state.models[0].model_name
        original_mtime = (model_dir / "inference_data.nc").stat().st_mtime

        # Small delay
        time.sleep(0.01)

        # Incremental save - should skip inference_data since it hasn't changed
        analysis_state.save(out_path, incremental=True)
        new_mtime = (model_dir / "inference_data.nc").stat().st_mtime

        # inference_data should NOT be rewritten (signature unchanged)
        assert new_mtime == original_mtime, "inference_data.nc should NOT be rewritten when unchanged"

    def test_incremental_saves_modified_inference_data(self, tmp_path: Path):
        """Test that incremental save saves inference_data when it has been modified."""
        analysis_state = create_test_analysis_state(n_rows=100, n_models=1, n_samples=50, n_chains=1, n_params=3)
        out_path = tmp_path / "analysis_output"

        # First full save
        analysis_state.save(out_path, incremental=False)

        model_dir = out_path / "models" / analysis_state.models[0].model_name
        original_mtime = (model_dir / "inference_data.nc").stat().st_mtime

        # Modify inference_data by adding a new group
        new_group_data = xr.Dataset({"new_var": (["x"], [1, 2, 3])})
        analysis_state.models[0].inference_data.add_groups(predictions=new_group_data)

        # Small delay
        time.sleep(0.01)

        # Incremental save - should save inference_data since it has changed
        analysis_state.save(out_path, incremental=True)
        new_mtime = (model_dir / "inference_data.nc").stat().st_mtime

        # inference_data should be rewritten (signature changed)
        assert new_mtime > original_mtime, "inference_data.nc should be rewritten when modified"

    def test_incremental_still_saves_diagnostics(self, tmp_path: Path):
        """Test that incremental save still saves diagnostics."""
        analysis_state = create_test_analysis_state(n_rows=100, n_models=1, n_samples=50, n_chains=1, n_params=3)
        out_path = tmp_path / "analysis_output"

        # First full save
        analysis_state.save(out_path, incremental=False)

        model_dir = out_path / "models" / analysis_state.models[0].model_name
        original_mtime = (model_dir / "diagnostics.json").stat().st_mtime

        # Add a new diagnostic
        analysis_state.models[0].add_diagnostic("new_metric", 42.0)

        # Small delay
        time.sleep(0.01)

        # Incremental save
        analysis_state.save(out_path, incremental=True)
        new_mtime = (model_dir / "diagnostics.json").stat().st_mtime

        # diagnostics should be rewritten
        assert new_mtime > original_mtime, "diagnostics.json should be rewritten on incremental save"


def run_timing_comparison():
    """Run a manual timing comparison for development purposes."""
    import tempfile

    print("Creating test analysis state...")
    analysis_state = create_test_analysis_state(
        n_rows=10000,
        n_models=3,
        n_samples=2000,
        n_chains=4,
        n_params=50,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "analysis_output"

        print("\n--- Timing Results ---")

        # First full save
        start = time.perf_counter()
        analysis_state.save(out_path, incremental=False)
        t1 = time.perf_counter() - start
        print(f"First full save:     {t1:.3f}s")

        # Second full save
        start = time.perf_counter()
        analysis_state.save(out_path, incremental=False)
        t2 = time.perf_counter() - start
        print(f"Second full save:    {t2:.3f}s")

        # Incremental save
        start = time.perf_counter()
        analysis_state.save(out_path, incremental=True)
        t3 = time.perf_counter() - start
        print(f"Incremental save:    {t3:.3f}s")

        print(f"\nSpeedup (full vs incremental): {t2/t3:.1f}x")

        # Check file sizes
        print("\n--- File Sizes ---")
        data_size = (out_path / "data.parquet").stat().st_size / 1024 / 1024
        print(f"data.parquet: {data_size:.2f} MB")

        for model_state in analysis_state.models:
            model_dir = out_path / "models" / model_state.model_name
            idata_size = (model_dir / "inference_data.nc").stat().st_size / 1024 / 1024
            print(f"{model_state.model_name}/inference_data.nc: {idata_size:.2f} MB")

        # Verify incremental skipped data.parquet
        print("\n--- Verifying incremental behavior ---")
        data_mtime_before = (out_path / "data.parquet").stat().st_mtime
        time.sleep(0.01)
        analysis_state.save(out_path, incremental=True)
        data_mtime_after = (out_path / "data.parquet").stat().st_mtime
        if data_mtime_before == data_mtime_after:
            print("PASS: data.parquet was NOT rewritten on incremental save")
        else:
            print("FAIL: data.parquet was rewritten on incremental save")

        model_dir = out_path / "models" / analysis_state.models[0].model_name
        idata_mtime_before = (model_dir / "inference_data.nc").stat().st_mtime
        time.sleep(0.01)
        analysis_state.save(out_path, incremental=True)
        idata_mtime_after = (model_dir / "inference_data.nc").stat().st_mtime
        if idata_mtime_after == idata_mtime_before:
            print("PASS: inference_data.nc was NOT rewritten (unchanged, as expected)")
        else:
            print("FAIL: inference_data.nc was rewritten even though it didn't change")

        # Test that modifying inference_data triggers a save
        print("\n--- Testing that modified inference_data gets saved ---")
        idata_mtime_before = (model_dir / "inference_data.nc").stat().st_mtime

        # Add a new group to inference_data (simulating what a communicator might do)
        import xarray as xr
        new_group_data = xr.Dataset({"new_var": (["x"], [1, 2, 3])})
        analysis_state.models[0].inference_data.add_groups(predictions=new_group_data)

        time.sleep(0.01)
        analysis_state.save(out_path, incremental=True)
        idata_mtime_after = (model_dir / "inference_data.nc").stat().st_mtime
        if idata_mtime_after > idata_mtime_before:
            print("PASS: inference_data.nc WAS rewritten after modification")
        else:
            print("FAIL: inference_data.nc was not rewritten after modification")


if __name__ == "__main__":
    run_timing_comparison()
