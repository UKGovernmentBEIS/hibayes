from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import yaml

from hibayes.analysis_state import AnalysisState
from hibayes.process import (
    DataProcessor,
    ProcessConfig,
    drop_rows_with_missing_features,
    extract_features,
    extract_observed_feature,
    groupby,
    map_columns,
    process,
)
from hibayes.process.utils import infer_jax_dtype
from hibayes.ui import ModellingDisplay


class TestDataProcessorProtocol:
    """Test the DataProcessor protocol and decorator."""

    def test_process_decorator_registration(self):
        """Test that @process decorator registers processors correctly."""

        @process
        def test_processor() -> DataProcessor:
            def processor_impl(
                state: AnalysisState, display: ModellingDisplay | None = None
            ) -> AnalysisState:
                return state

            return processor_impl

        # Should be registered and callable
        processor = test_processor()
        assert callable(processor)

        # Should have registry info
        from hibayes.registry import registry_info

        info = registry_info(processor)
        assert info.type == "processor"
        assert info.name == "test_processor"

    def test_process_decorator_enforces_interface(
        self, sample_analysis_state: AnalysisState
    ):
        """Test that process decorator enforces the correct interface."""

        @process
        def test_processor() -> DataProcessor:
            def processor_impl(
                state: AnalysisState, display: ModellingDisplay | None = None
            ) -> AnalysisState:
                # Should have processed_data copied from data
                assert state.processed_data is not None
                assert state.processed_data.equals(state.data)
                return state

            return processor_impl

        processor = test_processor()
        result = processor(sample_analysis_state)

        # Original data should be untouched
        assert sample_analysis_state.data is not None
        # Processed data should be a copy
        assert result.processed_data is not None
        assert result.processed_data.equals(sample_analysis_state.data)
        assert result.processed_data is not sample_analysis_state.data

    def test_process_decorator_preserves_existing_processed_data(
        self, sample_analysis_state: AnalysisState
    ):
        """Test that decorator doesn't overwrite existing processed_data."""

        # Set up state with existing processed_data
        sample_analysis_state.processed_data = pd.DataFrame({"existing": [1, 2, 3]})

        @process
        def test_processor() -> DataProcessor:
            def processor_impl(
                state: AnalysisState, display: ModellingDisplay | None = None
            ) -> AnalysisState:
                assert "existing" in state.processed_data.columns
                return state

            return processor_impl

        processor = test_processor()
        result = processor(sample_analysis_state)

        # Should preserve existing processed_data
        assert "existing" in result.processed_data.columns


class TestExtractObservedFeature:
    """Test extract_observed_feature processor."""

    def test_extract_observed_feature_default(
        self, sample_analysis_state: AnalysisState, mock_display: ModellingDisplay
    ):
        """Test extracting default 'score' feature."""
        processor = extract_observed_feature()
        result = processor(sample_analysis_state, mock_display)

        assert result.features is not None
        assert "obs" in result.features
        assert isinstance(result.features["obs"], jnp.ndarray)
        assert len(result.features["obs"]) == 4
        assert jnp.allclose(result.features["obs"], jnp.array([0.8, 0.9, 0.7, 0.85]))

        # Check logging
        mock_display.logger.info.assert_called_once()
        call_args = mock_display.logger.info.call_args[0][0]
        assert "Extracted 'score' -> 'obs'" in call_args

    def test_extract_observed_feature_custom_name(
        self, sample_analysis_state: AnalysisState
    ):
        """Test extracting custom feature name."""
        processor = extract_observed_feature(feature_name="difficulty")
        result = processor(sample_analysis_state)

        assert result.features is not None
        assert "obs" in result.features
        assert jnp.allclose(result.features["obs"], jnp.array([0.2, 0.4, 0.3, 0.5]))

    def test_extract_observed_feature_missing_column(
        self, sample_analysis_state: AnalysisState
    ):
        """Test error when feature column doesn't exist."""
        processor = extract_observed_feature(feature_name="nonexistent")

        with pytest.raises(ValueError, match=r".*nonexistent.*"):
            processor(sample_analysis_state)

    def test_extract_observed_feature_dtype_inference(
        self, sample_analysis_state: AnalysisState
    ):
        """Test that JAX dtype is inferred correctly."""
        # Modify data to have integer scores
        sample_analysis_state.data["int_score"] = [1, 0, 1, 0]

        processor = extract_observed_feature(feature_name="int_score")
        result = processor(sample_analysis_state)

        assert result.features["obs"].dtype == jnp.int32

    def test_extract_observed_feature_updates_existing_features(
        self, sample_analysis_state: AnalysisState
    ):
        """Test that processor updates existing features dict."""
        sample_analysis_state.features = {"existing": jnp.array([1, 2, 3])}

        processor = extract_observed_feature()
        result = processor(sample_analysis_state)

        assert "existing" in result.features
        assert "obs" in result.features

    def test_extract_observed_feature_no_display(
        self, sample_analysis_state: AnalysisState
    ):
        """Test processor works without display object."""
        processor = extract_observed_feature()
        result = processor(sample_analysis_state, display=None)

        assert result.features is not None
        assert "obs" in result.features


class TestExtractFeatures:
    """Test extract_features processor."""

    def test_extract_features_default(
        self, sample_analysis_state: AnalysisState, mock_display: ModellingDisplay
    ):
        """Test extracting default 'score' feature."""
        processor = extract_features(continuous_features=["score"])
        result = processor(sample_analysis_state, mock_display)

        assert result.features is not None
        assert "score" in result.features
        assert isinstance(result.features["score"], jnp.ndarray)
        assert len(result.features["score"]) == 4
        assert jnp.allclose(result.features["score"], jnp.array([0.8, 0.9, 0.7, 0.85]))

        # Check logging
        mock_display.logger.info.assert_called_once()
        call_args = mock_display.logger.info.call_args[0][0]
        assert "Extracted continuous 'score' with dtype:" in call_args

    def test_extract_features_multiple_features(
        self, sample_analysis_state: AnalysisState, mock_display: ModellingDisplay
    ):
        """Test extracting multiple features."""
        processor = extract_features(continuous_features=["score", "difficulty"])
        result = processor(sample_analysis_state, mock_display)

        assert result.features is not None
        assert "score" in result.features
        assert "difficulty" in result.features
        assert isinstance(result.features["score"], jnp.ndarray)
        assert isinstance(result.features["difficulty"], jnp.ndarray)
        assert len(result.features["score"]) == 4
        assert len(result.features["difficulty"]) == 4

        # Check values
        assert jnp.allclose(result.features["score"], jnp.array([0.8, 0.9, 0.7, 0.85]))
        assert jnp.allclose(
            result.features["difficulty"], jnp.array([0.2, 0.4, 0.3, 0.5])
        )

        # Check logging - should log for each feature
        assert mock_display.logger.info.call_count == 2

    def test_extract_features_custom_feature_names(
        self, sample_analysis_state: AnalysisState
    ):
        """Test extracting custom feature names."""
        processor = extract_features(continuous_features=["difficulty"])
        result = processor(sample_analysis_state)

        assert result.features is not None
        assert "difficulty" in result.features
        assert "score" not in result.features  # Should not be included
        assert jnp.allclose(
            result.features["difficulty"], jnp.array([0.2, 0.4, 0.3, 0.5])
        )

    def test_extract_features_missing_column(
        self, sample_analysis_state: AnalysisState
    ):
        """Test error when feature column doesn't exist."""
        processor = extract_features(continuous_features=["nonexistent"])

        with pytest.raises(ValueError, match=r".*nonexistent.*"):
            processor(sample_analysis_state)

    def test_extract_features_partial_missing_columns(
        self, sample_analysis_state: AnalysisState
    ):
        """Test error when some feature columns don't exist."""
        processor = extract_features(continuous_features=["score", "nonexistent"])

        with pytest.raises(ValueError, match=r".*nonexistent.*"):
            processor(sample_analysis_state)

    def test_extract_features_dtype_inference(
        self, sample_analysis_state: AnalysisState
    ):
        """Test that JAX dtype is inferred correctly for different types."""
        # Add integer column to test data
        sample_analysis_state.processed_data = sample_analysis_state.data.copy()
        sample_analysis_state.processed_data["int_feature"] = [1, 0, 1, 0]
        sample_analysis_state.processed_data["float32_feature"] = pd.Series(
            [1.0, 2.0, 3.0, 4.0], dtype="float32"
        )

        print(
            sample_analysis_state.processed_data["score"].dtype
        )  # Should be float64 by default
        print(sample_analysis_state.processed_data["score"].dtype.itemsize)

        processor = extract_features(
            continuous_features=["int_feature", "float32_feature", "score"]
        )
        result = processor(sample_analysis_state)
        print(result.features["score"].dtype)

        assert result.features["int_feature"].dtype == jnp.int32
        assert result.features["float32_feature"].dtype == jnp.float32
        assert result.features["score"].dtype == jnp.float32  # Default float type

    def test_extract_features_updates_existing_features(
        self, sample_analysis_state: AnalysisState
    ):
        """Test that processor updates existing features dict."""
        sample_analysis_state.features = {"existing": jnp.array([1, 2, 3, 4])}

        processor = extract_features(continuous_features=["score"])
        result = processor(sample_analysis_state)

        assert "existing" in result.features
        assert "score" in result.features
        assert len(result.features) == 2

    def test_extract_features_no_existing_features(
        self, sample_analysis_state: AnalysisState
    ):
        """Test processor works when no existing features dict."""
        sample_analysis_state.features = None

        processor = extract_features(continuous_features=["score"])
        result = processor(sample_analysis_state)

        assert result.features is not None
        assert "score" in result.features
        assert len(result.features) == 1

    def test_extract_features_no_display(self, sample_analysis_state: AnalysisState):
        """Test processor works without display object."""
        processor = extract_features(continuous_features=["score"])
        result = processor(sample_analysis_state, display=None)

        assert result.features is not None
        assert "score" in result.features

    def test_extract_features_empty_feature_list(
        self, sample_analysis_state: AnalysisState
    ):
        """Test processor with empty feature list."""
        processor = extract_features(continuous_features=[])
        result = processor(sample_analysis_state)

        # Should not modify features dict if it exists
        if sample_analysis_state.features is not None:
            assert result.features == sample_analysis_state.features
        else:
            assert result.features == {}

    def test_extract_features_preserves_feature_order(
        self, sample_analysis_state: AnalysisState
    ):
        """Test that features are extracted in the specified order."""
        processor = extract_features(continuous_features=["difficulty", "score"])
        result = processor(sample_analysis_state)

        # Check that both features are present
        assert "difficulty" in result.features
        assert "score" in result.features

        # In Python 3.7+, dict order is preserved
        feature_keys = list(result.features.keys())
        assert feature_keys.index("difficulty") < feature_keys.index("score")

    def test_extract_features_with_nan_values(
        self, sample_analysis_state: AnalysisState
    ):
        """Test processor handles NaN values correctly."""
        # Add NaN values to test data
        sample_analysis_state.processed_data = sample_analysis_state.data.copy()
        sample_analysis_state.processed_data.loc[0, "score"] = float("nan")

        processor = extract_features(continuous_features=["score"])
        result = processor(sample_analysis_state)

        assert result.features is not None
        assert "score" in result.features
        assert jnp.isnan(result.features["score"][0])

    def test_extract_features_with_unsupported_dtype(
        self, sample_analysis_state: AnalysisState
    ):
        """Test error when column has unsupported dtype for JAX conversion."""
        # Add string column which can't be converted to JAX array
        sample_analysis_state.processed_data = sample_analysis_state.data.copy()
        sample_analysis_state.processed_data["string_feature"] = ["a", "b", "c", "d"]

        processor = extract_features(continuous_features=["string_feature"])

        with pytest.raises(ValueError, match="Unsupported pandas dtype"):
            processor(sample_analysis_state)

    def test_extract_features_error_message_format(
        self, sample_analysis_state: AnalysisState
    ):
        """Test that error message format is correct and informative."""
        processor = extract_features(continuous_features=["missing1", "missing2"])

        with pytest.raises(ValueError) as exc_info:
            processor(sample_analysis_state)

        error_msg = str(exc_info.value)
        assert "missing1" in error_msg
        assert "missing2" in error_msg
        assert "score" in error_msg  # Should show available columns
        assert "model" in error_msg
        assert "task" in error_msg
        assert "difficulty" in error_msg

    def test_extract_features_with_boolean_column(
        self, sample_analysis_state: AnalysisState
    ):
        """Test processor with boolean column."""
        sample_analysis_state.processed_data = sample_analysis_state.data.copy()
        sample_analysis_state.processed_data["bool_feature"] = [
            True,
            False,
            True,
            False,
        ]

        processor = extract_features(continuous_features=["bool_feature"])
        result = processor(sample_analysis_state)

        assert result.features is not None
        assert "bool_feature" in result.features
        # Boolean should be converted to int (True=1, False=0)
        expected = jnp.array([1, 0, 1, 0], dtype=jnp.int32)
        assert jnp.array_equal(result.features["bool_feature"], expected)

    def test_extract_features_logging_content(
        self, sample_analysis_state: AnalysisState, mock_display: ModellingDisplay
    ):
        """Test that logging contains correct information."""
        processor = extract_features(continuous_features=["score"])
        _ = processor(sample_analysis_state, mock_display)

        # Check the logged message contains expected information
        mock_display.logger.info.assert_called_once()
        call_args = mock_display.logger.info.call_args[0][0]
        assert "Extracted continuous 'score'" in call_args
        assert "dtype:" in call_args
        assert "float32" in call_args  # Expected dtype for float columns

    def test_extract_features_with_duplicate_names(
        self, sample_analysis_state: AnalysisState
    ):
        """Test processor handles duplicate feature names."""
        processor = extract_features(continuous_features=["score", "score"])
        result = processor(sample_analysis_state)

        # Should only have one 'score' feature (dict keys are unique)
        assert result.features is not None
        assert "score" in result.features
        assert len([k for k in result.features.keys() if k == "score"]) == 1


class TestExtractPredictors:
    """Test extract_features processor with categorical_features."""

    def test_extract_features_categorical_default(
        self, sample_analysis_state: AnalysisState, mock_display: ModellingDisplay
    ):
        """Test extracting default predictors (model, task)."""
        processor = extract_features(categorical_features=["model", "task"])
        result = processor(sample_analysis_state, mock_display)

        # Check features
        assert result.features is not None
        assert "model_index" in result.features
        assert "task_index" in result.features
        assert "num_model" in result.features
        assert "num_task" in result.features

        # Check values
        assert result.features["num_model"] == 2  # gpt-4, claude
        assert result.features["num_task"] == 2  # math, reading
        assert len(result.features["model_index"]) == 4
        assert len(result.features["task_index"]) == 4

        # Check coords
        assert result.coords is not None
        assert "model" in result.coords
        assert "task" in result.coords
        assert set(result.coords["model"]) == {"claude", "gpt-4"}  # sorted
        assert set(result.coords["task"]) == {"math", "reading"}  # sorted

        # Check dims
        assert result.dims is not None
        assert result.dims["model_effects"] == ["model"]
        assert result.dims["task_effects"] == ["task"]

        # Check logging
        assert mock_display.logger.info.call_count == 2

    def test_extract_features_categorical_custom_names(
        self, sample_analysis_state: AnalysisState
    ):
        """Test extracting custom predictor names."""
        processor = extract_features(categorical_features=["task"])
        result = processor(sample_analysis_state)

        assert result.features is not None
        assert "task_index" in result.features
        assert "num_task" in result.features
        assert "model_index" not in result.features  # Should not be included

    def test_extract_features_categorical_missing_columns(
        self, sample_analysis_state: AnalysisState
    ):
        """Test error when predictor columns don't exist."""
        processor = extract_features(categorical_features=["nonexistent"])

        with pytest.raises(ValueError, match=r".*nonexistent.*"):
            processor(sample_analysis_state)

    def test_extract_features_categorical_factorization(
        self, sample_analysis_state: AnalysisState
    ):
        """Test that factorization produces correct indices."""
        processor = extract_features(categorical_features=["model"])
        result = processor(sample_analysis_state)

        # Should have indices 0, 0, 1, 1 (claude=0, gpt-4=1 when sorted)
        expected_indices = jnp.array(
            [1, 1, 0, 0], dtype=jnp.int32
        )  # gpt-4, gpt-4, claude, claude
        assert jnp.array_equal(result.features["model_index"], expected_indices)

    def test_extract_features_categorical_updates_existing_state(
        self, sample_analysis_state: AnalysisState
    ):
        """Test that processor updates existing features/coords/dims."""
        sample_analysis_state.features = {"existing": jnp.array([1])}
        sample_analysis_state.coords = {"existing_coord": ["a"]}
        sample_analysis_state.dims = {"existing_dim": ["x"]}

        processor = extract_features(categorical_features=["model"])
        result = processor(sample_analysis_state)

        # Should preserve existing
        assert "existing" in result.features
        assert "existing_coord" in result.coords
        assert "existing_dim" in result.dims

        # Should add new
        assert "model_index" in result.features
        assert "model" in result.coords
        assert "model_effects" in result.dims

    def test_extract_features_categorical_with_interactions(
        self, sample_analysis_state: AnalysisState, mock_display: ModellingDisplay
    ):
        """Test extracting predictors with interactions enabled."""
        processor = extract_features(categorical_features=["model", "task"], interactions=True)
        result = processor(sample_analysis_state, mock_display)

        # Check that basic features are still there
        assert result.features is not None
        assert "model_index" in result.features
        assert "task_index" in result.features

        # Check interaction dimensions
        assert result.dims is not None
        assert "model_task_effects" in result.dims
        assert result.dims["model_task_effects"] == ["model", "task"]

        # Check logging includes interaction info
        log_calls = [call[0][0] for call in mock_display.logger.info.call_args_list]
        interaction_logs = [log for log in log_calls if "interaction" in log.lower()]
        assert len(interaction_logs) > 0
        assert any("model_task" in log for log in interaction_logs)

    def test_extract_features_categorical_with_effect_coding(
        self, sample_analysis_state: AnalysisState, mock_display: ModellingDisplay
    ):
        """Test extracting predictors with effect coding for main effects."""
        processor = extract_features(categorical_features=["model", "task"], effect_coding_for_main_effects=True)
        result = processor(sample_analysis_state, mock_display)

        # Check that constrained coordinates are created
        assert result.coords is not None
        assert "model_constrained" in result.coords
        assert "task_constrained" in result.coords

        # Check constrained coords have n-1 elements
        assert len(result.coords["model_constrained"]) == 1
        assert len(result.coords["task_constrained"]) == 1

        # Check that both constrained and full dims are created
        assert result.dims is not None
        assert result.dims["model_effects"] == ["model"]
        assert result.dims["task_effects"] == ["task"]
        assert result.dims["model_effects_constrained"] == ["model_constrained"]
        assert result.dims["task_effects_constrained"] == ["task_constrained"]

        # Check logging mentions constrained effects
        log_calls = [call[0][0] for call in mock_display.logger.info.call_args_list]
        constrained_logs = [log for log in log_calls if "Constrained:" in log]
        assert len(constrained_logs) == 2  # One for model, one for task

    def test_extract_features_categorical_effect_coding_single_level(self):
        """Test effect coding with single-level predictor (should not create constrained coords)."""
        # Create data with single-level predictor
        data = pd.DataFrame(
            {
                "model": ["gpt-4", "gpt-4", "gpt-4"],
                "task": ["math", "reading", "math"],
                "score": [0.8, 0.7, 0.9],
            }
        )
        state = AnalysisState(data=data)

        processor = extract_features(
            categorical_features=["model"], effect_coding_for_main_effects=True
        )
        result = processor(state)

        # Single level predictor should not have constrained coords
        assert "model_constrained" not in result.coords
        assert result.dims["model_effects"] == ["model"]  # Should use standard coding
        assert "model_effects" in result.dims

    def test_extract_features_categorical_interactions_and_effect_coding(
        self, sample_analysis_state: AnalysisState, mock_display: ModellingDisplay
    ):
        """Test extracting predictors with both interactions and effect coding."""
        processor = extract_features(
            categorical_features=["model", "task"],
            interactions=True, effect_coding_for_main_effects=True
        )
        result = processor(sample_analysis_state, mock_display)

        # Check main effects have both constrained and full
        assert result.dims["model_effects_constrained"] == ["model_constrained"]
        assert result.dims["model_effects"] == ["model"]
        assert result.dims["task_effects_constrained"] == ["task_constrained"]
        assert result.dims["task_effects"] == ["task"]

        # Check interaction effects
        assert result.dims["model_task_effects"] == ["model", "task"]

        # Check logging mentions both constrained effects and interactions
        log_calls = [call[0][0] for call in mock_display.logger.info.call_args_list]

        constrained_logs = [log for log in log_calls if "Constrained:" in log]
        assert len(constrained_logs) == 2

        interaction_logs = [log for log in log_calls if "interaction" in log.lower()]
        assert len(interaction_logs) > 0

    def test_extract_features_categorical_three_way_interactions(self):
        """Test that only pairwise interactions are created with 3+ predictors."""
        # Create data with three predictors
        data = pd.DataFrame(
            {
                "model": ["gpt-4", "claude", "gpt-4", "claude"],
                "task": ["math", "reading", "math", "reading"],
                "dataset": ["A", "B", "A", "B"],
                "score": [0.8, 0.7, 0.9, 0.6],
            }
        )
        state = AnalysisState(data=data)

        processor = extract_features(
            categorical_features=["model", "task", "dataset"], interactions=True
        )
        result = processor(state)

        # Should have all pairwise interactions but no three-way
        expected_interactions = [
            "model_task_effects",
            "model_dataset_effects",
            "task_dataset_effects",
        ]

        for interaction in expected_interactions:
            assert interaction in result.dims

        # Should not have three-way interaction
        assert "model_task_dataset_effects" not in result.dims

    def test_extract_features_categorical_constrained_coords_content(
        self, sample_analysis_state: AnalysisState
    ):
        """Test that constrained coordinates contain the correct values (all but last)."""
        processor = extract_features(categorical_features=["model", "task"], effect_coding_for_main_effects=True)
        result = processor(sample_analysis_state)

        # Check that constrained coords are all but the last (when sorted)
        full_model_coords = result.coords[
            "model"
        ]  # Should be ['claude', 'gpt-4'] (sorted)
        constrained_model_coords = result.coords["model_constrained"]

        assert constrained_model_coords == full_model_coords[:-1]
        assert len(constrained_model_coords) == len(full_model_coords) - 1

        full_task_coords = result.coords["task"]
        constrained_task_coords = result.coords["task_constrained"]

        assert constrained_task_coords == full_task_coords[:-1]
        assert len(constrained_task_coords) == len(full_task_coords) - 1


class TestDropRowsWithMissingFeatures:
    """Test drop_rows_with_missing_features processor."""

    def test_drop_rows_with_missing_default(
        self, mock_display: ModellingDisplay, sample_data_with_missing: pd.DataFrame
    ):
        """Test dropping rows with missing default features."""
        data = sample_data_with_missing
        state = AnalysisState(data=data)

        processor = drop_rows_with_missing_features()
        result = processor(state, mock_display)

        # Should drop rows where model or task is None
        assert len(result.processed_data) == 2  # Only 2 complete rows
        assert result.processed_data["model"].isna().sum() == 0
        assert result.processed_data["task"].isna().sum() == 0

        # Check logging
        mock_display.logger.info.assert_called_once()
        call_args = mock_display.logger.info.call_args[0][0]
        assert "Dropping rows with missing features: ['model', 'task']" in call_args

    def test_drop_rows_with_missing_custom_features(self):
        """Test dropping rows with custom feature names."""
        data = pd.DataFrame(
            {"col1": [1, None, 3], "col2": [1, 2, None], "score": [0.1, 0.2, 0.3]}
        )
        state = AnalysisState(data=data)

        processor = drop_rows_with_missing_features(feature_names=["col1"])
        result = processor(state)

        # Should only drop row where col1 is None
        assert len(result.processed_data) == 2
        assert result.processed_data["col1"].isna().sum() == 0


class TestMapColumns:
    """Test map_columns processor."""

    def test_map_columns_basic(
        self, sample_analysis_state: AnalysisState, mock_display: ModellingDisplay
    ):
        """Test basic column mapping."""
        mapping = {"model": "llm", "task": "benchmark"}

        processor = map_columns(column_mapping=mapping)
        result = processor(sample_analysis_state, mock_display)

        assert "llm" in result.processed_data.columns
        assert "benchmark" in result.processed_data.columns
        assert "model" not in result.processed_data.columns
        assert "task" not in result.processed_data.columns

        # Check logging
        mock_display.logger.info.assert_called_once()
        call_args = mock_display.logger.info.call_args[0][0]
        assert "Mapping columns: {'model': 'llm', 'task': 'benchmark'}" in call_args

    def test_map_columns_partial_mapping(self, sample_analysis_state: AnalysisState):
        """Test mapping only some columns."""
        mapping = {"model": "llm"}

        processor = map_columns(column_mapping=mapping)
        result = processor(sample_analysis_state)

        assert "llm" in result.processed_data.columns
        assert "model" not in result.processed_data.columns
        assert "task" in result.processed_data.columns

    def test_map_columns_nonexistent_columns(
        self, sample_analysis_state: AnalysisState
    ):
        """Test mapping nonexistent columns (should not raise error)."""
        mapping = {"nonexistent": "new_name"}

        processor = map_columns(column_mapping=mapping)
        result = processor(sample_analysis_state)

        # Should not crash, just ignore nonexistent columns
        assert "new_name" not in result.processed_data.columns
        assert "nonexistent" not in result.processed_data.columns


class TestGroupby:
    """Test groupby processor."""

    def test_groupby_basic(
        self, mock_display: ModellingDisplay, sample_binomial_data: pd.DataFrame
    ):
        """Test basic groupby aggregation."""
        data = sample_binomial_data
        state = AnalysisState(data=data)

        processor = groupby(groupby_columns=["model", "task"])
        result = processor(state, mock_display)

        # Should have aggregated data
        assert len(result.processed_data) == 2  # 2 unique (model, task) pairs
        assert "n_correct" in result.processed_data.columns
        assert "n_total" in result.processed_data.columns
        assert "model" in result.processed_data.columns
        assert "task" in result.processed_data.columns

        # Check aggregation values
        grouped = result.processed_data.set_index(["model", "task"])
        assert grouped.loc[("gpt-4", "math"), "n_correct"] == 2  # 2 out of 3 correct
        assert grouped.loc[("gpt-4", "math"), "n_total"] == 3  # 3 total attempts
        assert (
            grouped.loc[("claude", "reading"), "n_correct"] == 1
        )  # 1 out of 2 correct
        assert grouped.loc[("claude", "reading"), "n_total"] == 2  # 2 total attempts

        # Check logging
        mock_display.logger.info.assert_called_once()
        call_args = mock_display.logger.info.call_args[0][0]
        assert "Grouping data by: ['model', 'task']" in call_args

    def test_groupby_single_column(self, sample_binomial_data: pd.DataFrame):
        """Test groupby with single column."""
        data = sample_binomial_data
        state = AnalysisState(data=data)

        processor = groupby(groupby_columns=["model"])
        result = processor(state)

        # Should have 2 rows (2 unique models)
        assert len(result.processed_data) == 2
        grouped = result.processed_data.set_index("model")
        assert grouped.loc["gpt-4", "n_total"] == 3  # gpt-4 has 3 attempts
        assert grouped.loc["claude", "n_total"] == 2  # claude has 2 attempts

    def test_groupby_missing_columns(self, sample_analysis_state: AnalysisState):
        """Test error when groupby columns don't exist."""
        processor = groupby(groupby_columns=["nonexistent"])

        with pytest.raises(ValueError, match=r".*nonexistent.*"):
            processor(sample_analysis_state)

    def test_groupby_missing_score_column(self):
        """Test error when score column doesn't exist."""
        data = pd.DataFrame(
            {
                "model": ["gpt-4", "claude"],
                "task": ["math", "reading"],
                # Missing 'score' column
            }
        )
        state = AnalysisState(data=data)

        processor = groupby(groupby_columns=["model"])

        with pytest.raises(ValueError, match=r".*score.*"):
            processor(state)

    def test_groupby_with_kwargs(self, sample_binomial_data: pd.DataFrame):
        """Test groupby with additional arguments."""
        data = sample_binomial_data
        state = AnalysisState(data=data)

        # Test with dropna=False (though this is default)
        processor = groupby(groupby_columns=["model"], dropna=False)
        result = processor(state)

        assert len(result.processed_data) == 2

    def test_groupby_preserves_reset_index(self, sample_binomial_data: pd.DataFrame):
        """Test that groupby resets index properly."""
        data = sample_binomial_data
        state = AnalysisState(data=data)

        processor = groupby(groupby_columns=["model", "task"])
        result = processor(state)

        # Index should be reset (default integer index)
        assert list(result.processed_data.index) == [0, 1]


class TestUtils:
    """Test utility functions."""

    def test_infer_jax_dtype_float(self):
        """Test JAX dtype inference for float types."""
        series_float32 = pd.Series([1.0, 2.0, 3.0], dtype="float32")
        series_float64 = pd.Series([1.0, 2.0, 3.0], dtype="float64")

        assert infer_jax_dtype(series_float32) == jnp.float32
        assert (
            infer_jax_dtype(series_float64) == jnp.float32
        )  # 32 preferred for GPU performance

    def test_infer_jax_dtype_int(self):
        """Test JAX dtype inference for integer types."""
        series_int32 = pd.Series([1, 2, 3], dtype="int32")
        series_int64 = pd.Series([1, 2, 3], dtype="int64")
        series_int8 = pd.Series([1, 2, 3], dtype="int8")

        assert infer_jax_dtype(series_int32) == jnp.int32
        assert infer_jax_dtype(series_int64) == jnp.int32  # All signed ints -> int32
        assert infer_jax_dtype(series_int8) == jnp.int32

    def test_infer_jax_dtype_uint(self):
        """Test JAX dtype inference for unsigned integer types."""
        series_uint32 = pd.Series([1, 2, 3], dtype="uint32")
        series_uint64 = pd.Series([1, 2, 3], dtype="uint64")

        assert infer_jax_dtype(series_uint32) == jnp.uint32
        assert infer_jax_dtype(series_uint64) == jnp.uint64

    def test_infer_jax_dtype_unsupported(self):
        """Test error for unsupported dtypes."""
        series_object = pd.Series(["a", "b", "c"], dtype="object")

        with pytest.raises(ValueError, match="Unsupported pandas dtype"):
            infer_jax_dtype(series_object)


class TestProcessConfig:
    """Test ProcessConfig class."""

    def test_default_process_config(self):
        """Test default process configuration."""
        config = ProcessConfig()

        assert len(config.enabled_processors) == 2
        # Should have default processors
        processor_names = [p.__name__ for p in config.enabled_processors]
        assert "extract_observed_feature" in str(processor_names)
        assert "extract_features" in str(processor_names)

    def test_process_config_custom_processors(self):
        """Test ProcessConfig with custom processors."""
        custom_processor = extract_observed_feature(feature_name="custom")
        config = ProcessConfig(enabled_processors=[custom_processor])

        assert len(config.enabled_processors) == 1
        assert config.enabled_processors[0] == custom_processor

    def test_process_config_from_dict_list(self):
        """Test ProcessConfig.from_dict with list format."""
        config_dict = {
            "processors": [
                "extract_observed_feature",
                {"extract_features": {"continuous_features": ["model"]}},
                "drop_rows_with_missing_features",
            ]
        }

        config = ProcessConfig.from_dict(config_dict)

        assert len(config.enabled_processors) == 3

    def test_process_config_from_dict_dict_format(self):
        """Test ProcessConfig.from_dict with dict format."""
        config_dict = {
            "processors": {
                "extract_observed_feature": {"feature_name": "accuracy"},
                "map_columns": {"column_mapping": {"old": "new"}},
            }
        }

        config = ProcessConfig.from_dict(config_dict)

        assert len(config.enabled_processors) == 2

    def test_process_config_from_yaml(self, tmp_path: Path):
        """Test ProcessConfig.from_yaml."""
        yaml_content = {
            "processors": [
                "extract_observed_feature",
                {"extract_features": {"continuous_features": ["model", "task"]}},
            ]
        }

        yaml_file = tmp_path / "config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        config = ProcessConfig.from_yaml(str(yaml_file))

        assert len(config.enabled_processors) == 2

    def test_process_config_with_custom_paths(self, tmp_path: Path):
        """Test loading processors from custom paths."""
        # Create a custom processor module
        custom_module = tmp_path / "custom_processors.py"
        custom_module.write_text(
            """
from hibayes.process import process
from hibayes.analysis_state import AnalysisState
from hibayes.ui import ModellingDisplay

@process
def custom_test_processor():
    def processor_impl(state: AnalysisState, display: ModellingDisplay | None = None) -> AnalysisState:
        return state
    return processor_impl
"""
        )

        config_dict = {
            "path": str(custom_module),
            "processors": ["custom_test_processor"],
        }

        with patch("hibayes.process.process_config._import_path"):
            with patch("hibayes.process.process_config.registry_get") as mock_registry:
                mock_processor = MagicMock()
                mock_registry.return_value = mock_processor

                config = ProcessConfig.from_dict(config_dict)

                mock_registry.assert_called_once()
                mock_processor.assert_called_once()
                assert len(config.enabled_processors) == 1

    def test_process_config_with_multiple_custom_paths(self, tmp_path: Path):
        """Test loading processors from multiple custom paths."""
        config_dict = {
            "path": ["/fake/path1", "/fake/path2"],
            "processors": ["custom_processor"],
        }

        with patch("hibayes.process.process_config._import_path") as mock_import:
            with patch("hibayes.process.process_config.registry_get") as mock_registry:
                mock_processor = MagicMock()
                mock_registry.return_value = mock_processor

                _ = ProcessConfig.from_dict(config_dict)

                # Should import both paths
                assert mock_import.call_count == 2
                mock_import.assert_any_call("/fake/path1")
                mock_import.assert_any_call("/fake/path2")

    def test_process_config_handles_missing_processor(self, tmp_path: Path):
        """Test that missing processors raise KeyError."""
        config_dict = {
            "path": "/fake/path",
            "processors": ["nonexistent_processor"],
        }

        with patch("hibayes.process.process_config._import_path"):
            with patch(
                "hibayes.process.process_config.registry_get", side_effect=KeyError
            ):
                with pytest.raises(KeyError):
                    ProcessConfig.from_dict(config_dict)

    def test_process_config_from_none(self):
        """Test ProcessConfig.from_dict with None input."""
        config = ProcessConfig.from_dict(None)

        # Should use defaults
        assert len(config.enabled_processors) == 2

    def test_process_config_invalid_processor_format(self, tmp_path: Path):
        """Test error handling for invalid processor format."""
        config_dict = {
            "processors": [{"invalid": "format", "too_many": "keys"}],
        }

        with pytest.raises(
            ValueError, match="Each process must be either a string or a dict"
        ):
            ProcessConfig.from_dict(config_dict)

    def test_process_config_empty_processors_list(self):
        """Test ProcessConfig with empty processors list."""
        config_dict = {"processors": []}

        config = ProcessConfig.from_dict(config_dict)

        # Should use default processors when list is empty
        assert len(config.enabled_processors) == 2

    def test_process_config_processors_dict_format(self):
        """Test ProcessConfig with processors in dict format."""
        config_dict = {
            "processors": {
                "extract_observed_feature": {"feature_name": "score"},
                "extract_features": {"continuous_features": ["model", "task"]},
            }
        }

        config = ProcessConfig.from_dict(config_dict)

        assert len(config.enabled_processors) == 2


class TestProcessIntegration:
    """Integration tests for the process functionality."""

    def test_full_processing_pipeline(self, mock_display: ModellingDisplay):
        """Test a complete processing pipeline."""
        data = pd.DataFrame(
            {
                "llm": ["gpt-4", "gpt-4", "claude", "claude"],
                "benchmark": ["math", "reading", "math", "reading"],
                "accuracy": [0.8, 0.9, 0.7, 0.85],
                "extra_col": [1, 2, 3, 4],
            }
        )

        state = AnalysisState(data=data)

        # Create processing pipeline
        processors = [
            map_columns(
                column_mapping={
                    "llm": "model",
                    "benchmark": "task",
                    "accuracy": "score",
                }
            ),
            extract_observed_feature(),
            extract_features(continuous_features=["score"]),
            extract_features(categorical_features=["model", "task"]),
        ]

        # Run pipeline
        for processor in processors:
            state = processor(state, mock_display)

        # Verify final state
        assert "obs" in state.features
        assert "model_index" in state.features
        assert "task_index" in state.features
        assert "model" in state.coords
        assert "task" in state.coords
        assert len(state.features["obs"]) == 4

    def test_binomial_processing_pipeline(self):
        """Test processing pipeline for binomial data."""
        # Data with multiple observations per group
        data = pd.DataFrame(
            {
                "model": ["gpt-4"] * 5 + ["claude"] * 3,
                "task": ["math"] * 3 + ["reading"] * 2 + ["math"] * 2 + ["reading"] * 1,
                "score": [1, 0, 1, 1, 0, 0, 1, 1],
            }
        )

        state = AnalysisState(data=data)

        # Processing pipeline for binomial data
        processors = [
            groupby(groupby_columns=["model", "task"]),
            map_columns(
                column_mapping={"n_correct": "score"}
            ),  # Map for extract_observed_feature
            extract_observed_feature(),
            extract_features(continuous_features=["score"]),
        ]

        # Run pipeline
        for processor in processors:
            state = processor(state)

        # Should have aggregated data
        assert len(state.processed_data) == 4
        assert "obs" in state.features
        assert len(state.features["obs"]) == 4

    def test_error_handling_in_pipeline(self):
        """Test error handling when processors fail."""
        data = pd.DataFrame({"col1": [1, 2, 3]})  # Missing required columns
        state = AnalysisState(data=data)

        # This should fail because 'score' column doesn't exist
        processor = extract_observed_feature()

        with pytest.raises(
            ValueError, match="Processed data must contain 'score' column"
        ):
            processor(state)

    def test_processor_state_preservation(self):
        """Test that processors preserve original data."""
        original_data = pd.DataFrame(
            {
                "model": ["gpt-4", "claude"],
                "task": ["math", "reading"],
                "score": [0.8, 0.9],
            }
        )

        state = AnalysisState(data=original_data.copy())

        # Run multiple processors
        processors = [
            extract_observed_feature(),
            extract_features(continuous_features=["score"]),
            map_columns(column_mapping={"model": "llm"}),
        ]

        for processor in processors:
            state = processor(state)

        # Original data should be unchanged
        assert state.data.equals(original_data)
        # But processed_data should be modified
        assert "llm" in state.processed_data.columns
        assert "model" not in state.processed_data.columns

    def test_processor_chaining_state_updates(self):
        """Test that processors correctly chain and update state."""
        data = pd.DataFrame(
            {
                "model": ["gpt-4", "claude"],
                "task": ["math", "reading"],
                "score": [0.8, 0.9],
            }
        )

        state = AnalysisState(data=data)

        # First processor
        processor1 = extract_observed_feature()
        state = processor1(state)

        assert state.features is not None
        assert "obs" in state.features
        assert state.coords is None
        assert state.dims is None

        # Second processor should build on first
        processor2 = extract_features(categorical_features=["model", "task"])
        state = processor2(state)

        assert "obs" in state.features  # From first processor
        assert "model_index" in state.features  # From second processor
        assert state.coords is not None
        assert state.dims is not None

    def test_processor_with_duplicate_data(self):
        """Test processors handle duplicate data correctly."""
        data = pd.DataFrame(
            {
                "model": ["gpt-4", "gpt-4", "gpt-4"],
                "task": ["math", "math", "math"],
                "score": [1, 0, 1],
            }
        )

        state = AnalysisState(data=data)

        # Group by should aggregate duplicates
        processor = groupby(groupby_columns=["model", "task"])
        result = processor(state)

        assert len(result.processed_data) == 1  # Should have only 1 row after grouping
        assert result.processed_data.iloc[0]["n_correct"] == 2  # 2 out of 3 correct
        assert result.processed_data.iloc[0]["n_total"] == 3

    def test_processor_order_independence_where_applicable(self):
        """Test that some processors can be applied in different orders."""
        data = pd.DataFrame(
            {
                "old_model": ["gpt-4", "claude"],
                "old_task": ["math", "reading"],
                "old_score": [0.8, 0.9],
            }
        )

        state1 = AnalysisState(data=data.copy())
        state2 = AnalysisState(data=data.copy())

        # Order 1: map columns first, then extract
        map_processor = map_columns(
            column_mapping={
                "old_model": "model",
                "old_task": "task",
                "old_score": "score",
            }
        )
        extract_processor = extract_observed_feature()

        state1 = map_processor(state1)
        state1 = extract_processor(state1)

        # Order 2: This should fail because extract looks for 'score' column
        state2 = AnalysisState(data=data.copy())
        with pytest.raises(ValueError):
            state2 = extract_processor(state2)  # Should fail - no 'score' column yet


class TestProcessorRegistration:
    """Test processor registration and retrieval."""

    def test_custom_processor_registration(self):
        """Test that custom processors can be registered."""

        @process
        def my_custom_processor(param1: str = "default") -> DataProcessor:
            def processor_impl(
                state: AnalysisState, display: ModellingDisplay | None = None
            ) -> AnalysisState:
                # Simple processor that adds a feature
                if state.features is None:
                    state.features = {}
                state.features["custom_param"] = param1
                return state

            return processor_impl

        # Should be callable
        processor = my_custom_processor(param1="test_value")
        assert callable(processor)

        # Should work with AnalysisState
        state = AnalysisState(data=pd.DataFrame({"col": [1, 2, 3]}))
        result = processor(state)

        assert result.features is not None
        assert result.features["custom_param"] == "test_value"

    def test_processor_registry_info(self):
        """Test that registered processors have correct registry info."""

        @process
        def test_registry_processor() -> DataProcessor:
            def processor_impl(
                state: AnalysisState, display: ModellingDisplay | None = None
            ) -> AnalysisState:
                return state

            return processor_impl

        processor = test_registry_processor()

        from hibayes.registry import registry_info

        info = registry_info(processor)

        assert info.type == "processor"
        assert info.name == "test_registry_processor"

    def test_builtin_processors_are_registered(self):
        """Test that built-in processors are properly registered."""
        from hibayes.registry import RegistryInfo, registry_get

        # Test that built-in processors can be retrieved from registry
        processor_names = [
            "extract_observed_feature",
            "extract_features",
            "drop_rows_with_missing_features",
            "map_columns",
            "groupby",
        ]

        for name in processor_names:
            info = RegistryInfo(type="processor", name=name)
            processor_builder = registry_get(info)
            assert callable(processor_builder)

            # Should be able to create processor instance
            processor = processor_builder()
            assert callable(processor)


class TestProcessorErrorHandling:
    """Test error handling in processors."""

    def test_processor_with_invalid_jax_dtype(self):
        """Test processor handling of invalid data types for JAX conversion."""
        data = pd.DataFrame(
            {
                "model": ["gpt-4", "claude"],
                "task": ["math", "reading"],
                "score": ["high", "low"],  # String values can't be converted to numeric
            }
        )

        state = AnalysisState(data=data)
        processor = extract_observed_feature()

        # This should raise an error when trying to infer JAX dtype
        with pytest.raises(ValueError):
            processor(state)

    def test_processor_with_none_values_in_features(self):
        """Test processor handling when existing features/coords/dims are None."""
        data = pd.DataFrame(
            {
                "model": ["gpt-4", "claude"],
                "task": ["math", "reading"],
                "score": [0.8, 0.9],
            }
        )

        state = AnalysisState(data=data)

        # Explicitly set to None to test initialization
        state.features = None
        state.coords = None
        state.dims = None

        processor = extract_features(categorical_features=["model", "task"])
        result = processor(state)

        # Should initialize new dicts
        assert result.features is not None
        assert result.coords is not None
        assert result.dims is not None

    def test_processor_with_empty_feature_names(self):
        """Test extract_features with empty feature list."""
        data = pd.DataFrame({"score": [0.8, 0.9]})
        state = AnalysisState(data=data)

        processor = extract_features(continuous_features=[])
        result = processor(state)

        # Should handle empty list gracefully
        assert result.features is not None or result.features == {}

    def test_groupby_with_no_score_variations(self):
        """Test groupby when all scores are the same."""
        data = pd.DataFrame(
            {
                "model": ["gpt-4", "gpt-4"],
                "task": ["math", "math"],
                "score": [1, 1],  # All same values
            }
        )

        state = AnalysisState(data=data)
        processor = groupby(groupby_columns=["model", "task"])
        result = processor(state)

        assert len(result.processed_data) == 1
        assert result.processed_data.iloc[0]["n_correct"] == 2
        assert result.processed_data.iloc[0]["n_total"] == 2


class TestProcessorPerformance:
    """Test processor performance characteristics."""

    def test_large_dataset_processing(self):
        """Test processors with larger datasets."""
        n_rows = 1000
        data = pd.DataFrame(
            {
                "model": ["gpt-4", "claude", "palm"] * (n_rows // 3 + 1),
                "task": ["math", "reading"] * (n_rows // 2 + 1),
                "score": [0.8, 0.9] * (n_rows // 2 + 1),
            }
        )[:n_rows]

        state = AnalysisState(data=data)
        processor = extract_features(categorical_features=["model", "task"])
        result = processor(state)

        assert len(result.features["model_index"]) == n_rows
        assert len(result.features["task_index"]) == n_rows

    def test_memory_efficiency_processed_data_copy(self):
        """Test that processed_data is efficiently copied."""
        data = pd.DataFrame(
            {"model": ["gpt-4"] * 100, "task": ["math"] * 100, "score": [0.8] * 100}
        )
        state = AnalysisState(data=data)

        processor1 = extract_observed_feature()
        result = processor1(state)

        # Should be a copy, not the same object
        assert result.processed_data is not state.data
        assert result.processed_data.equals(state.data)

        # Second processor should reuse existing processed_data
        original_processed_id = id(result.processed_data)
        processor2 = extract_features(continuous_features=["score"])
        result2 = processor2(result)

        assert id(result2.processed_data) == original_processed_id


class TestProcessConfigYAML:
    """Test ProcessConfig YAML configuration loading."""

    def test_process_config_from_yaml_list_format(self, tmp_path: Path):
        """Test loading ProcessConfig from YAML with list format."""
        yaml_content = {
            "processors": [
                "extract_observed_feature",
                {"extract_features": {"continuous_features": ["model", "task"]}},
                "drop_rows_with_missing_features",
            ]
        }

        yaml_file = tmp_path / "config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        config = ProcessConfig.from_yaml(str(yaml_file))
        assert len(config.enabled_processors) == 3

    def test_process_config_from_yaml_with_processor_parameters(self, tmp_path: Path):
        """Test loading ProcessConfig from YAML with processor parameters."""
        yaml_content = {
            "processors": [
                {"extract_observed_feature": {"feature_name": "accuracy"}},
                {
                    "extract_features": {
                        "continuous_features": ["model", "task", "difficulty"]
                    }
                },
                {
                    "map_columns": {
                        "column_mapping": {"old_model": "model", "old_task": "task"}
                    }
                },
                {"groupby": {"groupby_columns": ["model", "task"]}},
            ]
        }

        yaml_file = tmp_path / "config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        config = ProcessConfig.from_yaml(str(yaml_file))
        assert len(config.enabled_processors) == 4

    def test_process_config_from_yaml_with_paths_and_processors(self, tmp_path: Path):
        """Test loading ProcessConfig from YAML with custom paths."""
        yaml_content = {
            "path": ["/fake/custom/path1.py", "/fake/custom/path2.py"],
            "processors": [
                "extract_observed_feature",
                {"extract_features": {"continuous_features": ["score"]}},
            ],
        }

        yaml_file = tmp_path / "config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        with patch("hibayes.process.process_config._import_path") as mock_import:
            config = ProcessConfig.from_yaml(str(yaml_file))
            assert mock_import.call_count == 2
            assert len(config.enabled_processors) == 2

    def test_process_config_from_yaml_error_handling(self, tmp_path: Path):
        """Test error handling for invalid configurations."""
        # Test file not found
        with pytest.raises(FileNotFoundError):
            ProcessConfig.from_yaml(str(tmp_path / "nonexistent.yaml"))

        # Test invalid YAML syntax
        invalid_yaml_file = tmp_path / "invalid.yaml"
        invalid_yaml_file.write_text("invalid: yaml: content: [")
        with pytest.raises(yaml.YAMLError):
            ProcessConfig.from_yaml(str(invalid_yaml_file))

        # Test invalid processor format
        yaml_content = {
            "processors": [{"invalid_processor": "config", "another_key": "value"}]
        }
        yaml_file = tmp_path / "config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        with pytest.raises(
            ValueError, match=r".*Each process must be either a string or a dict*."
        ):
            ProcessConfig.from_yaml(str(yaml_file))

    def test_process_config_from_yaml_integration_with_analysis_state(
        self, tmp_path: Path, sample_analysis_state: AnalysisState
    ):
        """Test that processors loaded from YAML work with AnalysisState."""
        yaml_content = {
            "processors": [
                "extract_observed_feature",
                {"extract_features": {"continuous_features": ["score"]}},
            ]
        }

        yaml_file = tmp_path / "config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        config = ProcessConfig.from_yaml(str(yaml_file))

        state = sample_analysis_state
        for processor in config.enabled_processors:
            state = processor(state)

        assert state.features is not None
        assert "obs" in state.features
        assert "score" in state.features


class TestProcessConfigCustomProcessorIntegration:
    """Test ProcessConfig with custom processor files and argument passing."""

    def test_process_config_custom_processor_with_args(self, tmp_path: Path):
        """Test loading custom processor from file with arguments."""
        custom_module = tmp_path / "custom_processors.py"
        custom_module.write_text(
            """
from hibayes.process import process
from hibayes.analysis_state import AnalysisState
from hibayes.ui import ModellingDisplay

@process
def custom_feature_multiplier(multiplier: float = 1.0, feature_name: str = "score"):
    def processor_impl(state: AnalysisState, display: ModellingDisplay | None = None) -> AnalysisState:
        if feature_name not in state.processed_data.columns:
            raise ValueError(f"Feature {feature_name} not found in processed data")

        state.processed_data[feature_name] = state.processed_data[feature_name] * multiplier

        if state.features and feature_name in state.features:
            state.features[feature_name] = state.features[feature_name] * multiplier

        return state
    return processor_impl
"""
        )

        yaml_content = {
            "path": str(custom_module),
            "processors": [
                {
                    "custom_feature_multiplier": {
                        "multiplier": 2.5,
                        "feature_name": "score",
                    }
                },
                "extract_observed_feature",
            ],
        }

        yaml_file = tmp_path / "config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        config = ProcessConfig.from_yaml(str(yaml_file))
        assert len(config.enabled_processors) == 2

        data = pd.DataFrame(
            {
                "model": ["gpt-4", "claude"],
                "task": ["math", "reading"],
                "score": [0.8, 0.6],
            }
        )
        state = AnalysisState(data=data)

        state = config.enabled_processors[0](state)  # custom_feature_multiplier

        expected_scores = [0.8 * 2.5, 0.6 * 2.5]
        assert state.processed_data["score"].tolist() == expected_scores

    def test_process_config_custom_processor_full_pipeline(self, tmp_path: Path):
        """Test a full pipeline with custom and built-in processors."""
        custom_module = tmp_path / "pipeline_processors.py"
        custom_module.write_text(
            """
from hibayes.process import process
from hibayes.analysis_state import AnalysisState
from hibayes.ui import ModellingDisplay

@process
def custom_data_cleaner(min_score: float = 0.0, max_score: float = 1.0):
    def processor_impl(state: AnalysisState, display: ModellingDisplay | None = None) -> AnalysisState:
        mask = (state.processed_data["score"] >= min_score) & (state.processed_data["score"] <= max_score)
        state.processed_data = state.processed_data[mask].reset_index(drop=True)
        return state
    return processor_impl

@process
def custom_feature_engineer(score_threshold: float = 0.7):
    def processor_impl(state: AnalysisState, display: ModellingDisplay | None = None) -> AnalysisState:
        state.processed_data["high_performer"] = (state.processed_data["score"] >= score_threshold).astype(int)
        state.processed_data["score_squared"] = state.processed_data["score"] ** 2
        return state
    return processor_impl
"""
        )

        yaml_content = {
            "path": str(custom_module),
            "processors": [
                {
                    "map_columns": {
                        "column_mapping": {"llm_name": "model", "performance": "score"}
                    }
                },
                {"custom_data_cleaner": {"min_score": 0.1, "max_score": 0.9}},
                {"custom_feature_engineer": {"score_threshold": 0.6}},
            ],
        }

        yaml_file = tmp_path / "config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        config = ProcessConfig.from_yaml(str(yaml_file))

        assert len(config.enabled_processors) == 3

        data = pd.DataFrame(
            {
                "llm_name": ["gpt-4", "claude", "palm", "gpt-3"],
                "performance": [0.95, 0.05, 0.75, 0.65],  # One will be filtered out
            }
        )

        state = AnalysisState(data=data)
        for processor in config.enabled_processors:
            state = processor(state)

        # Check column mapping worked
        assert "model" in state.processed_data.columns
        assert "score" in state.processed_data.columns

        # Check data cleaning worked
        assert (
            len(state.processed_data) == 2
        )  # 4 -> 2 rows (filtered out 0.05 and 0.95)
        assert state.processed_data["score"].min() >= 0.1

        # Check feature engineering worked
        assert "high_performer" in state.processed_data.columns
        assert "score_squared" in state.processed_data.columns

    def test_process_config_custom_processor_error_handling(self, tmp_path: Path):
        """Test custom processor error handling."""
        custom_module = tmp_path / "error_processors.py"
        custom_module.write_text(
            """
from hibayes.process import process
from hibayes.analysis_state import AnalysisState
from hibayes.ui import ModellingDisplay

@process
def custom_error_processor(required_column: str = "nonexistent"):
    def processor_impl(state: AnalysisState, display: ModellingDisplay | None = None) -> AnalysisState:
        if required_column not in state.processed_data.columns:
            raise ValueError(f"Required column {required_column} not found in data")
        return state
    return processor_impl
"""
        )

        yaml_content = {
            "path": str(custom_module),
            "processors": [
                {"custom_error_processor": {"required_column": "missing_column"}}
            ],
        }

        yaml_file = tmp_path / "config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        config = ProcessConfig.from_yaml(str(yaml_file))
        data = pd.DataFrame({"model": ["gpt-4"], "score": [0.8]})
        state = AnalysisState(data=data)

        with pytest.raises(
            ValueError, match="Required column missing_column not found"
        ):
            config.enabled_processors[0](state)


class TestProcessConfigAdvanced:
    """Advanced tests for ProcessConfig functionality."""

    def test_process_config_processor_order_matters(self, tmp_path: Path):
        """Test that processor order affects results."""
        custom_module = tmp_path / "order_processors.py"
        custom_module.write_text(
            """
from hibayes.process import process
from hibayes.analysis_state import AnalysisState
from hibayes.ui import ModellingDisplay

@process
def multiply_score(factor: float = 2.0):
    def processor_impl(state: AnalysisState, display: ModellingDisplay | None = None) -> AnalysisState:
        state.processed_data["score"] = state.processed_data["score"] * factor
        return state
    return processor_impl

@process
def add_to_score(value: float = 0.1):
    def processor_impl(state: AnalysisState, display: ModellingDisplay | None = None) -> AnalysisState:
        state.processed_data["score"] = state.processed_data["score"] + value
        return state
    return processor_impl
"""
        )

        # Order 1: multiply then add
        yaml_content_1 = {
            "path": str(custom_module),
            "processors": [
                {"multiply_score": {"factor": 2.0}},
                {"add_to_score": {"value": 0.1}},
            ],
        }

        # Order 2: add then multiply
        yaml_content_2 = {
            "path": str(custom_module),
            "processors": [
                {"add_to_score": {"value": 0.1}},
                {"multiply_score": {"factor": 2.0}},
            ],
        }

        yaml_file_1 = tmp_path / "config1.yaml"
        yaml_file_2 = tmp_path / "config2.yaml"

        with open(yaml_file_1, "w") as f:
            yaml.dump(yaml_content_1, f)
        with open(yaml_file_2, "w") as f:
            yaml.dump(yaml_content_2, f)

        config_1 = ProcessConfig.from_yaml(str(yaml_file_1))
        config_2 = ProcessConfig.from_yaml(str(yaml_file_2))

        data = pd.DataFrame({"model": ["gpt-4"], "score": [0.5]})

        # Order 1: (0.5 * 2.0) + 0.1 = 1.1
        state_1 = AnalysisState(data=data.copy())
        for processor in config_1.enabled_processors:
            state_1 = processor(state_1)

        # Order 2: (0.5 + 0.1) * 2.0 = 1.2
        state_2 = AnalysisState(data=data.copy())
        for processor in config_2.enabled_processors:
            state_2 = processor(state_2)

        assert state_1.processed_data["score"].iloc[0] == pytest.approx(1.1)
        assert state_2.processed_data["score"].iloc[0] == pytest.approx(1.2)

    def test_process_config_comprehensive_integration(self, tmp_path: Path):
        """Comprehensive test combining multiple processor features."""
        custom_module = tmp_path / "comprehensive_processors.py"
        custom_module.write_text(
            """
from hibayes.process import process
from hibayes.analysis_state import AnalysisState
from hibayes.ui import ModellingDisplay
import numpy as np

@process
def comprehensive_processor(
    normalize_scores: bool = True,
    add_derived_features: bool = True,
    quality_threshold: float = 0.1
):
    def processor_impl(state: AnalysisState, display: ModellingDisplay | None = None) -> AnalysisState:
        # Filter low quality
        mask = state.processed_data["score"] >= quality_threshold
        state.processed_data = state.processed_data[mask].reset_index(drop=True)

        if normalize_scores:
            mean_score = state.processed_data["score"].mean()
            std_score = state.processed_data["score"].std()
            state.processed_data["normalized_score"] = (state.processed_data["score"] - mean_score) / std_score

        if add_derived_features:
            state.processed_data["score_percentile"] = state.processed_data["score"].rank(pct=True) * 100

            # Add performance levels
            score_col = state.processed_data["score"]
            conditions = [
                score_col <= score_col.quantile(0.25),
                (score_col > score_col.quantile(0.25)) & (score_col <= score_col.quantile(0.75)),
                score_col > score_col.quantile(0.75)
            ]
            choices = ["Low", "Medium", "High"]
            state.processed_data["performance_level"] = np.select(conditions, choices, default="Medium")

        return state
    return processor_impl
"""
        )

        yaml_content = {
            "path": str(custom_module),
            "processors": [
                {
                    "map_columns": {
                        "column_mapping": {"llm_model": "model", "performance": "score"}
                    }
                },
                {
                    "comprehensive_processor": {
                        "normalize_scores": True,
                        "add_derived_features": True,
                        "quality_threshold": 0.2,
                    }
                },
                "extract_observed_feature",
                {"extract_features": {"continuous_features": ["score", "normalized_score"]}},
                {"extract_features": {"categorical_features": ["model"]}},
            ],
        }

        yaml_file = tmp_path / "config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        config = ProcessConfig.from_yaml(str(yaml_file))

        # Create test data
        np.random.seed(42)
        n_samples = 200
        data = pd.DataFrame(
            {
                "llm_model": np.random.choice(
                    ["gpt-4", "claude", "palm"], size=n_samples
                ),
                "performance": np.random.beta(2, 1.5, size=n_samples),
            }
        )

        # Add some low-quality scores to be filtered
        low_quality_indices = np.random.choice(n_samples, size=20, replace=False)
        data.loc[low_quality_indices, "performance"] = np.random.uniform(
            0.0, 0.15, size=20
        )

        state = AnalysisState(data=data)

        # Apply full pipeline
        for processor in config.enabled_processors:
            state = processor(state)

        # Verify results
        assert "model" in state.processed_data.columns
        assert "score" in state.processed_data.columns
        assert len(state.processed_data) < n_samples  # Some rows filtered
        assert state.processed_data["score"].min() >= 0.2  # Quality threshold

        # Check derived features
        assert "normalized_score" in state.processed_data.columns
        assert "score_percentile" in state.processed_data.columns
        assert "performance_level" in state.processed_data.columns

        # Check hibayes features
        assert state.features is not None
        assert "obs" in state.features
        assert "score" in state.features
        assert "normalized_score" in state.features
        assert "model_index" in state.features


class TestProcessConfigEdgeCases:
    """Test edge cases and limits of ProcessConfig functionality."""

    def test_process_config_empty_data_handling(self):
        """Test processors with empty datasets."""
        empty_data = pd.DataFrame(columns=["model", "task", "score"])
        state = AnalysisState(data=empty_data)

        config_dict = {
            "processors": [
                "extract_observed_feature",
                {"extract_features": {"continuous_features": ["score"]}},
            ]
        }

        config = ProcessConfig.from_dict(config_dict)

        with pytest.raises(ValueError):
            for processor in config.enabled_processors:
                state = processor(state)

    def test_process_config_unicode_handling(self, tmp_path: Path):
        """Test configuration with unicode characters."""
        yaml_content = {
            "processors": [
                {
                    "map_columns": {
                        "column_mapping": {
                            "modèle": "model",  # Unicode
                            "tâche": "task",
                            "نتيجة": "score",  # Arabic
                        }
                    }
                }
            ]
        }

        yaml_file = tmp_path / "unicode_config.yaml"
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, allow_unicode=True)

        config = ProcessConfig.from_yaml(str(yaml_file))
        assert len(config.enabled_processors) == 1

        # Test with unicode data
        data = pd.DataFrame(
            {
                "modèle": ["gpt-4"],
                "tâche": ["math"],
                "نتيجة": [0.85],
            }
        )

        state = AnalysisState(data=data)
        result = config.enabled_processors[0](state)

        assert "model" in result.processed_data.columns
        assert "task" in result.processed_data.columns
        assert "score" in result.processed_data.columns

    def test_process_config_memory_stress_test(self):
        """Test processor with large datasets."""
        n_rows = 10000
        large_data = pd.DataFrame(
            {
                "model": np.random.choice(
                    ["model_a", "model_b", "model_c"], size=n_rows
                ),
                "score": np.random.beta(2, 2, size=n_rows),
            }
        )

        state = AnalysisState(data=large_data)

        config_dict = {
            "processors": [
                "extract_observed_feature",
                {"extract_features": {"continuous_features": ["score"]}},
                {"extract_features": {"categorical_features": ["model"]}},
            ]
        }

        config = ProcessConfig.from_dict(config_dict)

        for processor in config.enabled_processors:
            state = processor(state)

        assert len(state.features["obs"]) == n_rows
        assert len(state.features["score"]) == n_rows
