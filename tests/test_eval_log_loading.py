"""Tests for loading inspect-ai eval logs using the modern API.

These tests verify that the modern inspect API (read_eval_log_sample_summaries,
read_eval_log_sample, read_eval_log) correctly reads .eval files.
"""

from pathlib import Path

import pytest
from inspect_ai.log import (
    EvalSample,
    read_eval_log,
    read_eval_log_sample,
    read_eval_log_sample_summaries,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MINIMAL_EVAL = FIXTURES_DIR / "minimal_test.eval"


@pytest.fixture
def minimal_eval_path():
    assert MINIMAL_EVAL.exists(), f"Test fixture not found: {MINIMAL_EVAL}"
    return str(MINIMAL_EVAL)


class TestEvalLogLoading:
    def test_sample_summaries_returns_correct_count(self, minimal_eval_path):
        """read_eval_log_sample_summaries returns 2 summaries."""
        summaries = read_eval_log_sample_summaries(minimal_eval_path)
        assert len(summaries) == 2

    def test_sample_summaries_extracts_ids(self, minimal_eval_path):
        """Summary ids are {"q1", "q2"}."""
        summaries = read_eval_log_sample_summaries(minimal_eval_path)
        sample_ids = {s.id for s in summaries}
        assert sample_ids == {"q1", "q2"}

    def test_sample_summaries_extracts_epochs(self, minimal_eval_path):
        """All epochs are 1."""
        summaries = read_eval_log_sample_summaries(minimal_eval_path)
        assert all(s.epoch == 1 for s in summaries)

    def test_read_full_sample(self, minimal_eval_path):
        """read_eval_log_sample returns an EvalSample."""
        summaries = read_eval_log_sample_summaries(minimal_eval_path)
        sample = read_eval_log_sample(minimal_eval_path, summaries[0].id, summaries[0].epoch)
        assert isinstance(sample, EvalSample)

    def test_read_log_has_eval_metadata(self, minimal_eval_path):
        """read_eval_log(header_only=True) has eval spec with model."""
        log = read_eval_log(minimal_eval_path, header_only=True)
        assert log.eval is not None
        assert log.eval.model is not None
