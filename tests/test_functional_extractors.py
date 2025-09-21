from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from inspect_ai.log import (
    EvalDataset,
    EvalLog,
    EvalPlan,
    EvalPlanStep,
    EvalSample,
    EvalSpec,
    EvalStats,
)
from inspect_ai.model import ModelUsage
from inspect_ai.scorer import Score

from hibayes.load import (
    base_extractor,
    cyber_extractor,
    extractor,
    message_count_extractor,
    metadata_field_extractor,
    score_normalizer,
    token_extractor,
    tools_extractor,
)


@pytest.fixture
def score() -> Score:
    return Score(value=0.75)


@pytest.fixture
def eval_sample(score):
    s = MagicMock(spec=EvalSample)
    s.id = "sample-1"
    s.epoch = 1
    s.target = "target-1"
    s.messages = [
        MagicMock(role="system", content="System message"),
        MagicMock(role="user", content="User message"),
        MagicMock(role="assistant", content="Assistant message"),
        MagicMock(role="user", content="Another user message"),
    ]
    s.scores = {"default": score}
    s.score = score
    s.metadata = {
        "challenge_metadata": {
            "category": "cat",
            "max_messages": 10,
            "split": "train",
            "full_name": "Full Name",
            "description": "Description",
            "source": "Source",
            "capabilities": [
                {"name": "capability1", "level": "high"},
                {"name": "capability2", "level": "low"},
            ],
        },
        "nested": {"field": "value", "deep": {"field": "deep_value"}},
    }
    s.model_usage = {
        "gpt-4o": ModelUsage(
            total_tokens=100,
            input_tokens=50,
            output_tokens=50,
            input_tokens_cache_write=10,
            input_tokens_cache_read=5,
        )
    }
    return s


@pytest.fixture
def eval_log(eval_sample):
    dataset = MagicMock(spec=EvalDataset)
    dataset.name = "unit-dataset"

    spec = MagicMock(spec=EvalSpec)
    spec.model = "gpt-4o"
    spec.dataset = dataset
    spec.task = "test-task"

    stats = MagicMock(spec=EvalStats)
    stats.completed_at = "2024-05-01T12:34:56+00:00"

    plan = EvalPlan(
        steps=[
            EvalPlanStep(
                solver="use_tools",
                params={"tools": [{"name": "grep"}, {"name": "sed"}]},
            )
        ]
    )

    log = MagicMock(spec=EvalLog)
    log.version = 2
    log.status = "success"
    log.eval = spec
    log.plan = plan
    log.stats = stats
    log.samples = [eval_sample]
    log.location = "in-mem.eval"
    return log


def test_base_extractor_functional(eval_sample, eval_log):
    """Test the functional base_extractor."""
    my_extractor = base_extractor()
    result = my_extractor(eval_sample, eval_log)

    assert result["score"] == 0.75
    assert result["target"] == "target-1"
    assert result["model"] == "gpt-4o"
    assert result["dataset"] == "unit-dataset"
    assert result["task"] == "sample-1"
    assert result["epoch"] == 1
    assert result["num_messages"] == 4
    assert result["log_path"] == "in-mem.eval"


def test_base_extractor_score_normalization():
    """Test score normalization in base_extractor."""
    my_extractor = base_extractor()

    # Mock sample with "I" score
    sample_i = MagicMock(spec=EvalSample)
    sample_i.scores = {"default": MagicMock(value="I")}
    sample_i.messages = []
    sample_i.target = "target"
    sample_i.id = "id"
    sample_i.epoch = 1

    # Mock sample with "C" score
    sample_c = MagicMock(spec=EvalSample)
    sample_c.scores = {"default": MagicMock(value="C")}
    sample_c.messages = []
    sample_c.target = "target"
    sample_c.id = "id"
    sample_c.epoch = 1

    # Mock log with proper structure
    dataset = MagicMock()
    dataset.name = "dataset"

    spec = MagicMock()
    spec.model = "model"
    spec.dataset = dataset

    log = MagicMock(spec=EvalLog)
    log.eval = spec
    log.location = "location"

    result_i = my_extractor(sample_i, log)
    assert result_i["score"] == 0.0

    result_c = my_extractor(sample_c, log)
    assert result_c["score"] == 1.0


def test_token_extractor_functional(eval_sample, eval_log):
    """Test the functional token_extractor."""
    my_extractor = token_extractor()
    result = my_extractor(eval_sample, eval_log)

    assert result["total_tokens"] == 100
    assert result["input_tokens"] == 50
    assert result["output_tokens"] == 50
    assert result["cache_write_tokens"] == 10
    assert result["cache_read_tokens"] == 5


def test_token_extractor_no_usage(eval_log):
    """Test token_extractor with no model usage."""
    sample = MagicMock(spec=EvalSample)
    sample.model_usage = {}

    my_extractor = token_extractor()
    result = my_extractor(sample, eval_log)

    assert result == {}


def test_tools_extractor_functional(eval_sample, eval_log):
    """Test the functional tools_extractor."""
    my_extractor = tools_extractor()
    result = my_extractor(eval_sample, eval_log)

    assert result["tools"] == ["grep", "sed"]


def test_tools_extractor_no_tools(eval_sample):
    """Test tools_extractor with no tools."""
    log = MagicMock(spec=EvalLog)
    log.plan = EvalPlan(steps=[])

    my_extractor = tools_extractor()
    result = my_extractor(eval_sample, log)

    assert result["tools"] == [None]


def test_cyber_extractor_functional(eval_sample, eval_log):
    """Test the functional cyber_extractor."""
    my_extractor = cyber_extractor()
    result = my_extractor(eval_sample, eval_log)

    assert result["max_messages"] == 10
    assert result["category"] == "cat"
    assert result["split"] == "train"
    assert result["full_name"] == "Full Name"
    assert result["description"] == "Description"
    assert result["source"] == "Source"
    assert result["capability1"] == "high"
    assert result["capability2"] == "low"


def test_cyber_extractor_no_metadata(eval_log):
    """Test cyber_extractor with no metadata."""
    sample = MagicMock(spec=EvalSample)
    sample.metadata = {}

    my_extractor = cyber_extractor()
    result = my_extractor(sample, eval_log)

    assert result == {}


def test_message_count_extractor_functional(eval_sample, eval_log):
    """Test the functional message_count_extractor."""
    my_extractor = message_count_extractor()
    result = my_extractor(eval_sample, eval_log)

    assert result["system_messages"] == 1
    assert result["user_messages"] == 2
    assert result["assistant_messages"] == 1
    assert result["total_messages"] == 4


def test_message_count_extractor_with_params(eval_sample, eval_log):
    """Test message_count_extractor with custom parameters."""
    my_extractor = message_count_extractor(
        include_system=False, include_user=True, include_assistant=False
    )
    result = my_extractor(eval_sample, eval_log)

    assert "system_messages" not in result
    assert result["user_messages"] == 2
    assert "assistant_messages" not in result
    assert result["total_messages"] == 4


def test_metadata_field_extractor_functional(eval_sample, eval_log):
    """Test the functional metadata_field_extractor."""
    # Test simple field
    my_extractor = metadata_field_extractor("nested.field")
    result = my_extractor(eval_sample, eval_log)
    assert result["nested_field"] == "value"

    # Test deep nested field
    deep_extractor = metadata_field_extractor("nested.deep.field")
    result = deep_extractor(eval_sample, eval_log)
    assert result["nested_deep_field"] == "deep_value"

    # Test with rename
    renamed_extractor = metadata_field_extractor(
        "nested.field", rename_to="custom_name"
    )
    result = renamed_extractor(eval_sample, eval_log)
    assert result["custom_name"] == "value"


def test_metadata_field_extractor_default_value(eval_sample, eval_log):
    """Test metadata_field_extractor with default value."""
    my_extractor = metadata_field_extractor(
        "nonexistent.field", default_value="default"
    )
    result = my_extractor(eval_sample, eval_log)
    assert result["nonexistent_field"] == "default"


def test_score_normalizer_functional(eval_sample, eval_log):
    """Test the functional score_normalizer."""
    my_extractor = score_normalizer()
    result = my_extractor(eval_sample, eval_log)
    assert result["score"] == 0.75


def test_score_normalizer_custom_mapping(eval_log):
    """Test score_normalizer with custom mapping."""
    my_extractor = score_normalizer(
        score_field="custom_score", custom_mapping={"A": 1.0, "B": 0.5, "F": 0.0}
    )

    # Test custom mapping
    sample = MagicMock(spec=EvalSample)
    sample.scores = {"default": MagicMock(value="B")}

    result = my_extractor(sample, eval_log)
    assert result["custom_score"] == 0.5


def test_score_normalizer_percentage(eval_log):
    """Test score_normalizer with percentage normalization."""
    my_extractor = score_normalizer(normalize_to_01=True)

    sample = MagicMock(spec=EvalSample)
    sample.scores = {"default": MagicMock(value=85.0)}

    result = my_extractor(sample, eval_log)
    assert result["score"] == 0.85


def test_custom_extractor_decorator():
    """Test creating a custom extractor with the @extractor decorator."""

    @extractor
    def custom_test_extractor(multiplier: float = 2.0):
        def extract(sample: EvalSample, eval_log: EvalLog) -> Dict[str, Any]:
            score = float(next(iter(sample.scores.values())).value)
            return {
                "custom_score": score * multiplier,
                "model_type": eval_log.eval.model.split("-")[0],
            }

        return extract

    # Create instance with custom multiplier
    my_extractor = custom_test_extractor(multiplier=3.0)

    # Mock sample and log
    sample = MagicMock(spec=EvalSample)
    sample.scores = {"default": MagicMock(value=0.5)}

    spec = MagicMock()
    spec.model = "gpt-4o"

    log = MagicMock(spec=EvalLog)
    log.eval = spec

    result = my_extractor(sample, log)
    assert result["custom_score"] == 1.5
    assert result["model_type"] == "gpt"


def test_extractor_returns_non_dict():
    """Test that extractor enforces dict return type."""

    @extractor
    def bad_extractor():
        def extract(sample: EvalSample, eval_log: EvalLog):
            return "not a dict"  # This should fail

        return extract

    my_extractor = bad_extractor()

    sample = MagicMock(spec=EvalSample)
    log = MagicMock(spec=EvalLog)

    with pytest.raises(ValueError, match="Extractor must return a dictionary"):
        my_extractor(sample, log)


def test_functional_extractors_with_logprocessor():
    """Test that functional extractors work with LogProcessor."""
    from hibayes.load.configs.config import DataLoaderConfig
    from hibayes.load.load import LogProcessor

    # Create config with string references to extractors
    config = DataLoaderConfig.from_dict(
        {
            "extractors": {"enabled": ["base_extractor", "token_extractor"]},
            "paths": {"files_to_process": ["/test"]},
        }
    )

    processor = LogProcessor(config)
    assert len(processor.extractors) == 2


def test_mixed_extractors_with_logprocessor():
    """Test that LogProcessor can handle parameterized extractors."""
    from hibayes.load.configs.config import DataLoaderConfig
    from hibayes.load.load import LogProcessor

    # Create config with parameterized extractors
    config = DataLoaderConfig.from_dict(
        {
            "extractors": {
                "enabled": [
                    "base_extractor",
                    {"message_count_extractor": {"include_system": False}},
                    {"metadata_field_extractor": {"field_path": "test.field"}},
                ]
            },
            "paths": {"files_to_process": ["/test"]},
        }
    )

    processor = LogProcessor(config)
    assert len(processor.extractors) == 3
