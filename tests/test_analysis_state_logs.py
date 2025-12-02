"""Tests for AnalysisState logs functionality."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from hibayes.analysis_state import AnalysisState


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


class TestAnalysisStateLogs:
    """Tests for AnalysisState logs functionality."""

    def test_logs_initialized_as_empty_dict(self, sample_data):
        """Test that logs are initialized as an empty dict."""
        state = AnalysisState(data=sample_data)
        assert state.logs == {}
        assert isinstance(state.logs, dict)

    def test_add_log_creates_stage_key(self, sample_data):
        """Test that add_log creates the stage key if it doesn't exist."""
        state = AnalysisState(data=sample_data)
        state.add_log("Test log entry", stage="load")

        assert "load" in state.logs
        assert state.logs["load"] == ["Test log entry"]

    def test_add_log_appends_to_existing_stage(self, sample_data):
        """Test that add_log appends to existing stage logs."""
        state = AnalysisState(data=sample_data)
        state.add_log("First entry", stage="process")
        state.add_log("Second entry", stage="process")

        assert state.logs["process"] == ["First entry", "Second entry"]

    def test_add_log_multiple_stages(self, sample_data):
        """Test that logs can be added to multiple stages."""
        state = AnalysisState(data=sample_data)
        state.add_log("Load log", stage="load")
        state.add_log("Process log", stage="process")
        state.add_log("Model log", stage="model")
        state.add_log("Communicate log", stage="communicate")

        assert len(state.logs) == 4
        assert state.logs["load"] == ["Load log"]
        assert state.logs["process"] == ["Process log"]
        assert state.logs["model"] == ["Model log"]
        assert state.logs["communicate"] == ["Communicate log"]

    def test_get_stage_logs_existing_stage(self, sample_data):
        """Test get_stage_logs returns logs for existing stage."""
        state = AnalysisState(data=sample_data)
        state.add_log("Test entry", stage="load")

        assert state.get_stage_logs("load") == ["Test entry"]

    def test_get_stage_logs_nonexistent_stage(self, sample_data):
        """Test get_stage_logs returns empty list for nonexistent stage."""
        state = AnalysisState(data=sample_data)

        assert state.get_stage_logs("nonexistent") == []

    def test_logs_setter(self, sample_data):
        """Test that logs can be set directly."""
        state = AnalysisState(data=sample_data)
        new_logs = {
            "load": ["Log 1", "Log 2"],
            "process": ["Log 3"],
        }
        state.logs = new_logs

        assert state.logs == new_logs

    def test_logs_initialized_from_constructor(self, sample_data):
        """Test that logs can be passed in constructor."""
        initial_logs = {"load": ["Initial log"]}
        state = AnalysisState(data=sample_data, logs=initial_logs)

        assert state.logs == initial_logs


class TestAnalysisStateLogsSaveLoad:
    """Tests for saving and loading logs."""

    def test_save_creates_logs_directory(self, sample_data):
        """Test that save creates the logs directory."""
        state = AnalysisState(data=sample_data)
        state.add_log("Test log", stage="load")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "output"
            state.save(path)

            logs_dir = path / "logs"
            assert logs_dir.exists()
            assert logs_dir.is_dir()

    def test_save_creates_stage_log_files(self, sample_data):
        """Test that save creates separate log files for each stage."""
        state = AnalysisState(data=sample_data)
        state.add_log("Load entry", stage="load")
        state.add_log("Process entry", stage="process")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "output"
            state.save(path)

            assert (path / "logs" / "logs_load.txt").exists()
            assert (path / "logs" / "logs_process.txt").exists()

    def test_save_writes_log_content(self, sample_data):
        """Test that save writes the correct log content."""
        state = AnalysisState(data=sample_data)
        state.add_log("First load entry", stage="load")
        state.add_log("Second load entry", stage="load")
        state.add_log("Process entry", stage="process")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "output"
            state.save(path)

            load_logs = (path / "logs" / "logs_load.txt").read_text()
            assert "First load entry" in load_logs
            assert "Second load entry" in load_logs

            process_logs = (path / "logs" / "logs_process.txt").read_text()
            assert "Process entry" in process_logs

    def test_save_no_logs_directory_when_empty(self, sample_data):
        """Test that save doesn't create logs directory when there are no logs."""
        state = AnalysisState(data=sample_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "output"
            state.save(path)

            # logs directory should not exist if there are no logs
            assert not (path / "logs").exists()

    def test_save_overwrites_existing_log_files(self, sample_data):
        """Test that save overwrites existing log files for the same stage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "output"

            # First save
            state1 = AnalysisState(data=sample_data)
            state1.add_log("Original entry", stage="load")
            state1.save(path)

            # Second save with different content
            state2 = AnalysisState(data=sample_data)
            state2.add_log("New entry", stage="load")
            state2.save(path)

            load_logs = (path / "logs" / "logs_load.txt").read_text()
            assert "New entry" in load_logs
            assert "Original entry" not in load_logs

    def test_load_does_not_restore_logs(self, sample_data):
        """Test that load does not restore logs (they start fresh)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "output"

            # Save state with logs
            state1 = AnalysisState(data=sample_data)
            state1.add_log("Saved log", stage="load")
            state1.save(path)

            # Load state
            state2 = AnalysisState.load(path)

            # Logs should be empty after load
            assert state2.logs == {}

    def test_save_preserves_logs_from_other_stages(self, sample_data):
        """Test that saving one stage doesn't affect log files from other stages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "output"

            # First save with load stage
            state1 = AnalysisState(data=sample_data)
            state1.add_log("Load entry", stage="load")
            state1.save(path)

            # Second save with process stage (simulating pipeline continuation)
            state2 = AnalysisState.load(path)
            state2.add_log("Process entry", stage="process")
            state2.save(path)

            # Both log files should exist
            assert (path / "logs" / "logs_load.txt").exists()
            assert (path / "logs" / "logs_process.txt").exists()

            # Load logs should still contain original content
            load_logs = (path / "logs" / "logs_load.txt").read_text()
            assert "Load entry" in load_logs


class TestAnalysisStateLogsIntegration:
    """Integration tests for logs with analysis.py stage functions."""

    def test_logs_dict_assignment(self, sample_data):
        """Test that logs can be assigned as a dict (as done in analysis.py)."""
        state = AnalysisState(data=sample_data)

        # Simulate what analysis.py does
        state.logs["load"] = ["[10:00:00] Loading data...", "[10:00:01] Data loaded"]
        state.logs["process"] = ["[10:00:02] Processing..."]

        assert len(state.logs) == 2
        assert len(state.logs["load"]) == 2
        assert len(state.logs["process"]) == 1
