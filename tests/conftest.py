from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from hibayes.analysis_state import AnalysisState
from hibayes.ui import ModellingDisplay


@pytest.fixture
def sample_analysis_state() -> AnalysisState:
    """Basic AnalysisState with sample data."""
    data = pd.DataFrame(
        {
            "model": ["gpt-4", "gpt-4", "claude", "claude"],
            "task": ["math", "reading", "math", "reading"],
            "score": [0.8, 0.9, 0.7, 0.85],
            "difficulty": [0.2, 0.4, 0.3, 0.5],
        }
    )
    return AnalysisState(data=data)


@pytest.fixture
def sample_data_with_missing() -> pd.DataFrame:
    """Sample data with missing values."""
    return pd.DataFrame(
        {
            "model": ["gpt-4", "gpt-4", None, "claude"],
            "task": ["math", None, "math", "reading"],
            "score": [0.8, 0.9, 0.7, 0.85],
        }
    )


@pytest.fixture
def sample_binomial_data() -> pd.DataFrame:
    """Sample data for binomial groupby tests."""
    return pd.DataFrame(
        {
            "model": ["gpt-4", "gpt-4", "gpt-4", "claude", "claude"],
            "task": ["math", "math", "math", "reading", "reading"],
            "score": [1, 0, 1, 1, 0],
        }
    )


@pytest.fixture
def mock_display() -> ModellingDisplay:
    """Mock display object."""
    display = MagicMock(spec=ModellingDisplay)
    display.logger = MagicMock()
    return display
