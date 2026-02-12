"""Tests for analysis.py module functions (load_data, process_data, model, communicate)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from hibayes.analysis import communicate, load_data, model, process_data
from hibayes.analysis_state import AnalysisState
from hibayes.check import CheckerConfig
from hibayes.communicate import CommunicateConfig
from hibayes.load import DataLoaderConfig
from hibayes.model import ModelsToRunConfig
from hibayes.platform import PlatformConfig
from hibayes.process import ProcessConfig


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Basic sample data for AnalysisState."""
    return pd.DataFrame(
        {
            "model": ["gpt-4", "gpt-4", "claude", "claude"],
            "task": ["math", "reading", "math", "reading"],
            "score": [0.8, 0.9, 0.7, 0.85],
        }
    )


@pytest.fixture
def analysis_state(sample_data) -> AnalysisState:
    """Create a basic AnalysisState for testing."""
    return AnalysisState(data=sample_data)


@pytest.fixture
def mock_display():
    """Create a mock ModellingDisplay."""
    display = MagicMock()
    display.is_live = False
    display.capture_logs = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
    display.get_all_logs = MagicMock(return_value=[])
    display.get_stats_for_persistence = MagicMock(return_value={})
    return display


@pytest.fixture
def empty_models_config():
    """Create a ModelsToRunConfig with no models."""
    config = MagicMock(spec=ModelsToRunConfig)
    config.enabled_models = []
    return config


@pytest.fixture
def empty_checker_config():
    """Create a CheckerConfig with no checkers."""
    return CheckerConfig.from_dict({})


@pytest.fixture
def empty_communicate_config():
    """Create a CommunicateConfig with no communicators."""
    return CommunicateConfig.from_dict({})


@pytest.fixture
def platform_config():
    """Create a basic PlatformConfig."""
    return PlatformConfig.from_dict({})


class TestModelFunction:
    """Tests for the model() function in analysis.py."""

    def test_model_raises_error_when_frequent_save_true_and_out_none(
        self,
        analysis_state,
        mock_display,
        empty_models_config,
        empty_checker_config,
        platform_config,
    ):
        """Test that model() raises ValueError when frequent_save=True but out=None."""
        with pytest.raises(ValueError, match="'out' path must be provided when frequent_save=True"):
            model(
                analysis_state=analysis_state,
                models_to_run_config=empty_models_config,
                checker_config=empty_checker_config,
                platform_config=platform_config,
                display=mock_display,
                out=None,
                frequent_save=True,
            )

    def test_model_works_when_frequent_save_false_and_out_none(
        self,
        analysis_state,
        mock_display,
        empty_models_config,
        empty_checker_config,
        platform_config,
    ):
        """Test that model() works when frequent_save=False and out=None."""
        result = model(
            analysis_state=analysis_state,
            models_to_run_config=empty_models_config,
            checker_config=empty_checker_config,
            platform_config=platform_config,
            display=mock_display,
            out=None,
            frequent_save=False,
        )
        assert result is analysis_state

    def test_model_works_when_frequent_save_true_and_out_provided(
        self,
        analysis_state,
        mock_display,
        empty_models_config,
        empty_checker_config,
        platform_config,
    ):
        """Test that model() works when frequent_save=True and out is provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "output"
            out_path.mkdir()

            result = model(
                analysis_state=analysis_state,
                models_to_run_config=empty_models_config,
                checker_config=empty_checker_config,
                platform_config=platform_config,
                display=mock_display,
                out=out_path,
                frequent_save=True,
            )
            assert result is analysis_state

    def test_model_default_frequent_save_is_false(
        self,
        analysis_state,
        mock_display,
        empty_models_config,
        empty_checker_config,
        platform_config,
    ):
        """Test that model() defaults to frequent_save=False (so out is not required)."""
        # Should not raise when out is not provided and frequent_save defaults to False
        result = model(
            analysis_state=analysis_state,
            models_to_run_config=empty_models_config,
            checker_config=empty_checker_config,
            platform_config=platform_config,
            display=mock_display,
        )
        assert result is analysis_state


class TestCommunicateFunction:
    """Tests for the communicate() function in analysis.py."""

    def test_communicate_raises_error_when_frequent_save_true_and_out_none(
        self,
        analysis_state,
        mock_display,
        empty_communicate_config,
    ):
        """Test that communicate() raises ValueError when frequent_save=True but out=None."""
        with pytest.raises(ValueError, match="'out' path must be provided when frequent_save=True"):
            communicate(
                analysis_state=analysis_state,
                communicate_config=empty_communicate_config,
                display=mock_display,
                out=None,
                frequent_save=True,
            )

    def test_communicate_works_when_frequent_save_false_and_out_none(
        self,
        analysis_state,
        mock_display,
        empty_communicate_config,
    ):
        """Test that communicate() works when frequent_save=False and out=None."""
        result = communicate(
            analysis_state=analysis_state,
            communicate_config=empty_communicate_config,
            display=mock_display,
            out=None,
            frequent_save=False,
        )
        assert result is analysis_state

    def test_communicate_works_when_frequent_save_true_and_out_provided(
        self,
        analysis_state,
        mock_display,
        empty_communicate_config,
    ):
        """Test that communicate() works when frequent_save=True and out is provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "output"
            out_path.mkdir()

            result = communicate(
                analysis_state=analysis_state,
                communicate_config=empty_communicate_config,
                display=mock_display,
                out=out_path,
                frequent_save=True,
            )
            assert result is analysis_state

    def test_communicate_default_frequent_save_is_false(
        self,
        analysis_state,
        mock_display,
        empty_communicate_config,
    ):
        """Test that communicate() defaults to frequent_save=False (so out is not required)."""
        # Should not raise when out is not provided and frequent_save defaults to False
        result = communicate(
            analysis_state=analysis_state,
            communicate_config=empty_communicate_config,
            display=mock_display,
        )
        assert result is analysis_state


class TestModelAndCommunicateAPICompatibility:
    """Tests to ensure API compatibility for library users."""

    def test_model_can_be_called_without_out_parameter(
        self,
        analysis_state,
        mock_display,
        empty_models_config,
        empty_checker_config,
        platform_config,
    ):
        """
        Test that model() can be called without the out parameter.

        This is a regression test for GitHub issue #69 - users calling model()
        directly should not be required to provide 'out' if they don't want
        frequent saves.
        """
        # This should work without providing out
        result = model(
            analysis_state=analysis_state,
            models_to_run_config=empty_models_config,
            checker_config=empty_checker_config,
            platform_config=platform_config,
            display=mock_display,
        )
        assert isinstance(result, AnalysisState)

    def test_communicate_can_be_called_without_out_parameter(
        self,
        analysis_state,
        mock_display,
        empty_communicate_config,
    ):
        """
        Test that communicate() can be called without the out parameter.

        This is a regression test for GitHub issue #69 - users calling communicate()
        directly should not be required to provide 'out' if they don't want
        frequent saves.
        """
        # This should work without providing out
        result = communicate(
            analysis_state=analysis_state,
            communicate_config=empty_communicate_config,
            display=mock_display,
        )
        assert isinstance(result, AnalysisState)


class TestLoadDataFunction:
    """Tests for the load_data() function in analysis.py."""

    def test_load_data_with_extracted_csv_data(self, mock_display, tmp_path):
        """Test that load_data() can load pre-extracted CSV data."""
        # Create a sample CSV file
        csv_path = tmp_path / "test_data.csv"
        sample_df = pd.DataFrame({
            "model": ["gpt-4", "claude"],
            "score": [0.8, 0.9],
        })
        sample_df.to_csv(csv_path, index=False)

        config = DataLoaderConfig.from_dict({
            "paths": {"extracted_data": [str(csv_path)]}
        })

        result = load_data(config=config, display=mock_display)

        assert isinstance(result, AnalysisState)
        assert len(result.data) == 2
        assert "model" in result.data.columns
        assert "score" in result.data.columns

    def test_load_data_with_extracted_parquet_data(self, mock_display, tmp_path):
        """Test that load_data() can load pre-extracted parquet data."""
        # Create a sample parquet file
        parquet_path = tmp_path / "test_data.parquet"
        sample_df = pd.DataFrame({
            "model": ["gpt-4", "claude"],
            "score": [0.8, 0.9],
        })
        sample_df.to_parquet(parquet_path, index=False)

        config = DataLoaderConfig.from_dict({
            "paths": {"extracted_data": [str(parquet_path)]}
        })

        result = load_data(config=config, display=mock_display)

        assert isinstance(result, AnalysisState)
        assert len(result.data) == 2

    def test_load_data_with_multiple_files(self, mock_display, tmp_path):
        """Test that load_data() can concatenate multiple data files."""
        # Create multiple CSV files
        csv1 = tmp_path / "data1.csv"
        csv2 = tmp_path / "data2.csv"

        pd.DataFrame({"model": ["gpt-4"], "score": [0.8]}).to_csv(csv1, index=False)
        pd.DataFrame({"model": ["claude"], "score": [0.9]}).to_csv(csv2, index=False)

        config = DataLoaderConfig.from_dict({
            "paths": {"extracted_data": [str(csv1), str(csv2)]}
        })

        result = load_data(config=config, display=mock_display)

        assert isinstance(result, AnalysisState)
        assert len(result.data) == 2  # Combined from both files

    def test_load_data_captures_logs(self, mock_display, tmp_path):
        """Test that load_data() captures logs in the analysis state."""
        csv_path = tmp_path / "test_data.csv"
        pd.DataFrame({"model": ["gpt-4"], "score": [0.8]}).to_csv(csv_path, index=False)

        mock_display.get_all_logs.return_value = ["Log entry 1", "Log entry 2"]

        config = DataLoaderConfig.from_dict({
            "paths": {"extracted_data": [str(csv_path)]}
        })

        result = load_data(config=config, display=mock_display)

        assert result.logs.get("load") == ["Log entry 1", "Log entry 2"]


class TestProcessDataFunction:
    """Tests for the process_data() function in analysis.py."""

    def test_process_data_raises_error_when_both_data_and_analysis_state_provided(
        self,
        sample_data,
        analysis_state,
        mock_display,
    ):
        """Test that process_data() raises ValueError when both data and analysis_state are provided."""
        config = ProcessConfig.from_dict({})

        with pytest.raises(ValueError, match="Provide either 'data' or 'analysis_state', not both"):
            process_data(
                config=config,
                display=mock_display,
                data=sample_data,
                analysis_state=analysis_state,
            )

    def test_process_data_raises_error_when_neither_data_nor_analysis_state_provided(
        self,
        mock_display,
    ):
        """Test that process_data() raises ValueError when neither data nor analysis_state is provided."""
        config = ProcessConfig.from_dict({})

        with pytest.raises(ValueError, match="Must provide either 'data' or 'analysis_state'"):
            process_data(
                config=config,
                display=mock_display,
            )

    def test_process_data_works_with_data_parameter(
        self,
        sample_data,
        mock_display,
    ):
        """Test that process_data() works when data is provided."""
        config = ProcessConfig.from_dict({})

        result = process_data(
            config=config,
            display=mock_display,
            data=sample_data,
        )

        assert isinstance(result, AnalysisState)
        assert len(result.data) == len(sample_data)

    def test_process_data_works_with_analysis_state_parameter(
        self,
        analysis_state,
        mock_display,
    ):
        """Test that process_data() works when analysis_state is provided."""
        config = ProcessConfig.from_dict({})

        result = process_data(
            config=config,
            display=mock_display,
            analysis_state=analysis_state,
        )

        assert isinstance(result, AnalysisState)
        assert result is analysis_state

    def test_process_data_works_with_path_to_analysis_state(
        self,
        sample_data,
        mock_display,
        tmp_path,
    ):
        """Test that process_data() works when a path to analysis_state is provided."""
        # Save an analysis state to disk
        state = AnalysisState(data=sample_data)
        state_path = tmp_path / "analysis_state"
        state.save(state_path)

        config = ProcessConfig.from_dict({})

        result = process_data(
            config=config,
            display=mock_display,
            analysis_state=state_path,
        )

        assert isinstance(result, AnalysisState)
        assert len(result.data) == len(sample_data)

    def test_process_data_works_with_string_path_to_analysis_state(
        self,
        sample_data,
        mock_display,
        tmp_path,
    ):
        """Test that process_data() works when a string path to analysis_state is provided."""
        # Save an analysis state to disk
        state = AnalysisState(data=sample_data)
        state_path = tmp_path / "analysis_state"
        state.save(state_path)

        config = ProcessConfig.from_dict({})

        result = process_data(
            config=config,
            display=mock_display,
            analysis_state=str(state_path),
        )

        assert isinstance(result, AnalysisState)
        assert len(result.data) == len(sample_data)

    def test_process_data_captures_logs(
        self,
        sample_data,
        mock_display,
    ):
        """Test that process_data() captures logs in the analysis state."""
        mock_display.get_all_logs.return_value = ["Process log 1", "Process log 2"]

        config = ProcessConfig.from_dict({})

        result = process_data(
            config=config,
            display=mock_display,
            data=sample_data,
        )

        assert result.logs.get("process") == ["Process log 1", "Process log 2"]


class TestAnalysisConfigFromYaml:
    """Tests for AnalysisConfig.from_yaml() and from_dict()."""

    def test_analysis_config_from_yaml(self, tmp_path):
        """Test that AnalysisConfig can be loaded from a YAML file."""
        from hibayes.analysis import AnalysisConfig

        yaml_content = """
data_loader: {}
data_process: {}
model: {}
check: {}
communicate: {}
platform: {}
"""
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml_content)

        config = AnalysisConfig.from_yaml(str(yaml_path))

        assert isinstance(config.data_loader, DataLoaderConfig)
        assert isinstance(config.data_process, ProcessConfig)
        assert isinstance(config.models, ModelsToRunConfig)
        assert isinstance(config.checkers, CheckerConfig)
        assert isinstance(config.communicate, CommunicateConfig)
        assert isinstance(config.platform, PlatformConfig)

    def test_analysis_config_from_dict_with_empty_dict(self):
        """Test that AnalysisConfig can be created from an empty dict (uses defaults)."""
        from hibayes.analysis import AnalysisConfig

        config = AnalysisConfig.from_dict({})

        assert isinstance(config.data_loader, DataLoaderConfig)
        assert isinstance(config.data_process, ProcessConfig)
        assert isinstance(config.models, ModelsToRunConfig)
        assert isinstance(config.checkers, CheckerConfig)
        assert isinstance(config.communicate, CommunicateConfig)
        assert isinstance(config.platform, PlatformConfig)
