from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
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
        """Test extracting default features (model, task)."""
        processor = extract_features()
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
        assert result.dims["model_index"] == ["model"]
        assert result.dims["task_index"] == ["task"]

        # Check logging
        assert mock_display.logger.info.call_count == 2

    def test_extract_features_custom_names(self, sample_analysis_state: AnalysisState):
        """Test extracting custom feature names."""
        processor = extract_features(feature_names=["task"])
        result = processor(sample_analysis_state)

        assert result.features is not None
        assert "task_index" in result.features
        assert "num_task" in result.features
        assert "model_index" not in result.features  # Should not be included

    def test_extract_features_missing_columns(
        self, sample_analysis_state: AnalysisState
    ):
        """Test error when feature columns don't exist."""
        processor = extract_features(feature_names=["nonexistent"])

        with pytest.raises(ValueError, match=r".*nonexistent.*"):
            processor(sample_analysis_state)

    def test_extract_features_factorization(self, sample_analysis_state: AnalysisState):
        """Test that factorization produces correct indices."""
        processor = extract_features(feature_names=["model"])
        result = processor(sample_analysis_state)

        # Should have indices 0, 0, 1, 1 (claude=0, gpt-4=1 when sorted)
        expected_indices = jnp.array(
            [1, 1, 0, 0], dtype=jnp.int32
        )  # gpt-4, gpt-4, claude, claude
        assert jnp.array_equal(result.features["model_index"], expected_indices)

    def test_extract_features_updates_existing_state(
        self, sample_analysis_state: AnalysisState
    ):
        """Test that processor updates existing features/coords/dims."""
        sample_analysis_state.features = {"existing": jnp.array([1])}
        sample_analysis_state.coords = {"existing_coord": ["a"]}
        sample_analysis_state.dims = {"existing_dim": ["x"]}

        processor = extract_features(feature_names=["model"])
        result = processor(sample_analysis_state)

        # Should preserve existing
        assert "existing" in result.features
        assert "existing_coord" in result.coords
        assert "existing_dim" in result.dims

        # Should add new
        assert "model_index" in result.features
        assert "model" in result.coords
        assert "model_index" in result.dims


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
        assert infer_jax_dtype(series_float64) == jnp.float64

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
            "data_process": [
                "extract_observed_feature",
                {"extract_features": {"feature_names": ["model"]}},
                "drop_rows_with_missing_features",
            ]
        }

        config = ProcessConfig.from_dict(config_dict)

        assert len(config.enabled_processors) == 3

    def test_process_config_from_dict_dict_format(self):
        """Test ProcessConfig.from_dict with dict format."""
        config_dict = {
            "data_process": {
                "extract_observed_feature": {"feature_name": "accuracy"},
                "map_columns": {"column_mapping": {"old": "new"}},
            }
        }

        config = ProcessConfig.from_dict(config_dict)

        assert len(config.enabled_processors) == 2

    def test_process_config_from_yaml(self, tmp_path: Path):
        """Test ProcessConfig.from_yaml."""
        yaml_content = {
            "data_process": [
                "extract_observed_feature",
                {"extract_features": {"feature_names": ["model", "task"]}},
            ]
        }

        yaml_file = tmp_path / "config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        config = ProcessConfig.from_yaml(str(yaml_file))

        assert len(config.enabled_processors) == 2

    def test_process_config_custom_processors_loading(self, tmp_path: Path):
        """Test loading custom processors from external module."""
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
            "custom_data_process": {
                "path": str(custom_module),
                "processors": ["custom_test_processor"],
            }
        }

        with patch("hibayes.process.process_config._import_path"):
            with patch("hibayes.process.process_config.registry_get") as mock_registry:
                mock_processor = MagicMock()
                mock_registry.return_value = mock_processor

                _ = ProcessConfig.from_dict(config_dict)

                mock_registry.assert_called_once()
                mock_processor.assert_called_once()

    def test_process_config_handles_missing_processor(self, tmp_path: Path):
        """Test that missing processors are skipped with warning."""
        config_dict = {
            "custom_data_process": {
                "path": "/fake/path",
                "processors": ["nonexistent_processor"],
            }
        }

        with patch("hibayes.process.process_config._import_path"):
            with patch(
                "hibayes.process.process_config.registry_get", side_effect=KeyError
            ):
                with patch("hibayes.process.process_config.logger") as mock_logger:
                    with patch.object(ProcessConfig, "DEFAULT_PROCESS", []):
                        config = ProcessConfig.from_dict(config_dict)

                        mock_logger.warning.assert_called_once()
                        warning_msg = mock_logger.warning.call_args[0][0]
                        assert "nonexistent_processor not found" in warning_msg

                        # Should have no processors since custom one failed and defaults are empty
                        assert len(config.enabled_processors) == 0

    def test_process_config_from_none(self):
        """Test ProcessConfig.from_dict with None input."""
        config = ProcessConfig.from_dict(None)

        # Should use defaults
        assert len(config.enabled_processors) == 2

    def test_process_config_custom_processors_dict_format(self, tmp_path: Path):
        """Test custom processors with dict format for processors."""
        config_dict = {
            "custom_data_process": {
                "path": "/fake/path",
                "processors": {"custom_processor": {"param1": "value1"}},
            }
        }

        with patch("hibayes.process.process_config._import_path"):
            with patch("hibayes.process.process_config.registry_get") as mock_registry:
                mock_processor_class = MagicMock()
                mock_registry.return_value = mock_processor_class

                ProcessConfig.from_dict(config_dict)

                mock_processor_class.assert_called_once_with(param1="value1")

    def test_process_config_invalid_processor_format(self, tmp_path: Path):
        """Test error handling for invalid processor format."""
        config_dict = {
            "custom_data_process": {
                "path": "/fake/path",
                "processors": [{"invalid": "format", "too_many": "keys"}],
            }
        }

        with patch("hibayes.process.process_config._import_path"):
            with pytest.raises(
                ValueError, match="Each process must be either a string or a dict"
            ):
                ProcessConfig.from_dict(config_dict)


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
            extract_features(),
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
            extract_features(),
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
            extract_features(),
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
        processor2 = extract_features()
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

        processor = extract_features()
        result = processor(state)

        # Should initialize new dicts
        assert result.features is not None
        assert result.coords is not None
        assert result.dims is not None

    def test_processor_with_empty_feature_names(self):
        """Test extract_features with empty feature list."""
        data = pd.DataFrame({"score": [0.8, 0.9]})
        state = AnalysisState(data=data)

        processor = extract_features(feature_names=[])
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
        # Create larger dataset
        n_rows = 1000
        data = pd.DataFrame(
            {
                "model": ["gpt-4", "claude", "palm"] * (n_rows // 3 + 1),
                "task": ["math", "reading"] * (n_rows // 2 + 1),
                "score": [0.8, 0.9] * (n_rows // 2 + 1),
            }
        )[:n_rows]

        state = AnalysisState(data=data)

        # Should process without issues
        processor = extract_features()
        result = processor(state)

        assert len(result.features["model_index"]) == n_rows
        assert len(result.features["task_index"]) == n_rows

    def test_memory_efficiency_processed_data_copy(self):
        """Test that processed_data is efficiently copied."""
        data = pd.DataFrame(
            {"model": ["gpt-4"] * 100, "task": ["math"] * 100, "score": [0.8] * 100}
        )

        state = AnalysisState(data=data)

        # First processor should create processed_data copy
        processor1 = extract_observed_feature()
        result = processor1(state)

        # Should be a copy, not the same object
        assert result.processed_data is not state.data
        assert result.processed_data.equals(state.data)

        # Second processor should reuse existing processed_data
        original_processed_id = id(result.processed_data)
        processor2 = extract_features()
        result2 = processor2(result)

        # Should be the same object (no unnecessary copying)
        assert id(result2.processed_data) == original_processed_id
