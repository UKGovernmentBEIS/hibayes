import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import pytz
from inspect_ai.log import (
    EvalDataset,
    EvalLog,
    EvalLogInfo,
    EvalPlan,
    EvalPlanStep,
    EvalResults,
    EvalSample,
    EvalSpec,
    EvalStats,
)
from inspect_ai.model import ModelUsage
from inspect_ai.scorer import Score

from hibayes.analysis import _load_extracted_data
from hibayes.load import (
    base_extractor,
    token_extractor,
    tools_extractor,
)
from hibayes.load.configs.config import DataLoaderConfig
from hibayes.load.load import (
    _apply_cutoff,
    _apply_extractors,
    get_file_list,
    get_sample_df,
    is_after_timestamp,
)


@pytest.fixture
def score() -> Score:
    return Score(value=0.75)


@pytest.fixture
def eval_sample(score):
    s = MagicMock(spec=EvalSample)
    s.id = "sample‑1"
    s.epoch = 1
    s.target = "target‑1"
    # Create proper message objects for extractors that expect them
    msg = MagicMock()
    msg.role = "user"
    msg.content = "dummy-msg"
    s.messages = [msg]
    s.scores = {"default": score}
    s.score = score
    s.metadata = {"challenge_metadata": {"category": "cat"}}
    s.model_usage = {
        "gpt‑4o": ModelUsage(
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
    dataset.name = "unit‑dataset"

    spec = MagicMock(spec=EvalSpec)
    spec.model = "gpt‑4o"
    spec.dataset = dataset

    stats = MagicMock(spec=EvalStats)
    stats.completed_at = "2024-05-01T12:34:56+00:00"

    plan = EvalPlan(
        steps=[EvalPlanStep(solver="use_tools", params={"tools": [{"name": "grep"}]})]
    )

    log = MagicMock(spec=EvalLog)
    log.version = 2
    log.status = "success"
    log.eval = spec
    log.plan = plan
    log.results = MagicMock(spec=EvalResults)
    log.stats = stats
    log.samples = [eval_sample]
    log.location = "in‑mem.eval"
    return log


@pytest.fixture
def log_info():
    info = MagicMock(spec=EvalLogInfo)
    info.name = "in‑mem.eval"
    return info


@pytest.fixture
def dl_cfg(tmp_path: Path):
    """A default DataLoaderConfig pointing at a tmp logs dir."""
    return DataLoaderConfig.from_dict(
        {
            "extractors": {
                "enabled": ["base_extractor", "tools_extractor", "token_extractor"]
            },
            "paths": {"files_to_process": [str(tmp_path / "logs")]},
        }
    )


def test_dataloaderconfig_init(tmp_path: Path):
    cfg = DataLoaderConfig.from_dict(
        {
            "extractors": {"enabled": ["base_extractor", "tools_extractor"]},
            "paths": {"files_to_process": [str(tmp_path)]},
        }
    )
    assert len(cfg.enabled_extractors) == 2
    assert cfg.files_to_process == [str(tmp_path)]


def test_dataloaderconfig_from_yaml(tmp_path: Path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(
        """
extractors:
  enabled: [base_extractor, token_extractor]
paths:
  files_to_process: [/data]
"""
    )

    with patch(
        "yaml.safe_load",
        return_value={
            "extractors": {"enabled": ["base_extractor", "token_extractor"]},
            "paths": {"files_to_process": ["/data"]},
        },
    ):
        cfg = DataLoaderConfig.from_yaml(str(cfg_file))
    assert len(cfg.enabled_extractors) == 2
    assert cfg.files_to_process == ["/data"]


def test_dataloaderconfig_extracted_data(tmp_path: Path):
    """Test that extracted_data can be configured."""
    cfg = DataLoaderConfig.from_dict(
        {
            "paths": {
                "extracted_data": [
                    str(tmp_path / "data1.csv"),
                    str(tmp_path / "data2.parquet"),
                ]
            },
        }
    )
    assert cfg.extracted_data == [
        str(tmp_path / "data1.csv"),
        str(tmp_path / "data2.parquet"),
    ]
    assert cfg.files_to_process == []


def test_dataloaderconfig_mutual_exclusivity():
    """Test that files_to_process and extracted_data cannot both be specified."""
    with pytest.raises(ValueError, match="Cannot specify both"):
        DataLoaderConfig.from_dict(
            {
                "paths": {
                    "files_to_process": ["/path/to/logs"],
                    "extracted_data": ["/path/to/data.csv"],
                }
            }
        )


def test_dataloaderconfig_mutual_exclusivity_direct():
    """Test mutual exclusivity when creating DataLoaderConfig directly."""
    with pytest.raises(ValueError, match="Cannot specify both"):
        DataLoaderConfig(
            files_to_process=["/path/to/logs"],
            extracted_data=["/path/to/data.csv"],
        )


def test_dataloaderconfig_empty_warns(caplog):
    """Test that a warning is logged when no data source is specified."""
    import logging

    with caplog.at_level(logging.WARNING):
        DataLoaderConfig.from_dict({})
    assert "No data source specified" in caplog.text


def test_dataloaderconfig_cache_path_no_warning(caplog):
    """Test that no warning is logged when cache_path is provided."""
    import logging

    with caplog.at_level(logging.WARNING):
        cfg = DataLoaderConfig.from_dict(
            {"paths": {"cache_path": "/path/to/cache.jsonl"}}
        )
    assert "No data source specified" not in caplog.text
    assert cfg.cache_path == "/path/to/cache.jsonl"


def test_processor_unknown_extractor(tmp_path: Path):
    # This should raise an error when trying to get a non-existent extractor from registry
    with pytest.raises(KeyError, match="No registered does-not-exist found"):
        cfg = DataLoaderConfig.from_dict(
            {
                "extractors": {"enabled": ["base_extractor", "does-not-exist"]},
                "paths": {"files_to_process": [str(tmp_path)]},
            }
        )


def test_is_after_timestamp(eval_log):
    before = datetime.datetime(2024, 4, 1, tzinfo=pytz.UTC)
    after = datetime.datetime(2025, 1, 1, tzinfo=pytz.UTC)
    assert is_after_timestamp(before, eval_log)
    assert not is_after_timestamp(after, eval_log)


def test_is_after_timestamp_tzaware(eval_log):
    cutoff = datetime.datetime(2024, 4, 30, tzinfo=pytz.UTC)
    assert is_after_timestamp(cutoff, eval_log)


def test_get_file_list(tmp_path: Path):
    p1 = tmp_path / "a.eval"
    p1.touch()
    p2 = tmp_path / "b.eval"
    p2.touch()
    manifest = tmp_path / "list.txt"
    manifest.write_text(f"{p1}\n{p2}\n")
    assert get_file_list([str(manifest)]) == [str(p1), str(p2)]


def test_get_file_list_empty():
    assert get_file_list([]) == []


def test_get_sample_df_from_path_cached(tmp_path: Path, dl_cfg):
    cached = tmp_path / "cached.jsonl"
    cached.write_text("{score: 0.9}\n")
    dl_cfg.cache_path = str(cached)
    with patch("pandas.read_json", return_value=pd.DataFrame([{"score": 0.9}])):
        df = get_sample_df(
            config=dl_cfg,
        )
    assert df.iloc[0]["score"] == 0.9


def test_apply_cutoff(eval_log):
    """Test _apply_cutoff filters logs by completed_at timestamp."""
    df = pd.DataFrame(
        {
            "log": ["/path/log1.eval", "/path/log1.eval", "/path/log2.eval"],
            "id": ["s1", "s2", "s3"],
            "epoch": [1, 1, 1],
        }
    )

    # log1 passes cutoff (before May 2024), log2 does not
    log1_header = MagicMock(spec=EvalLog)
    log1_header.stats = MagicMock()
    log1_header.stats.completed_at = "2024-05-01T12:34:56+00:00"

    log2_header = MagicMock(spec=EvalLog)
    log2_header.stats = MagicMock()
    log2_header.stats.completed_at = "2024-03-01T12:34:56+00:00"

    def mock_read(path, header_only=False):
        if path == "/path/log1.eval":
            return log1_header
        return log2_header

    cutoff = datetime.datetime(2024, 4, 1, tzinfo=pytz.UTC)
    with patch("hibayes.load.load.read_eval_log", side_effect=mock_read):
        result = _apply_cutoff(df, cutoff)

    assert len(result) == 2
    assert list(result["log"]) == ["/path/log1.eval", "/path/log1.eval"]


def test_apply_extractors(eval_sample, eval_log):
    """Test _apply_extractors applies extractors to samples from bulk log reads."""
    df = pd.DataFrame(
        {
            "log": ["/path/log1.eval"],
            "id": [eval_sample.id],
            "epoch": [eval_sample.epoch],
        }
    )

    def mock_extractor(sample, log):
        return {"custom_field": "custom_value", "score": 0.75}

    with patch("hibayes.load.load.read_eval_log", return_value=eval_log):
        result = _apply_extractors(df, [mock_extractor])

    assert "custom_field" in result.columns
    assert result.iloc[0]["custom_field"] == "custom_value"
    assert result.iloc[0]["score"] == 0.75


def test_apply_extractors_error_handling(eval_sample, eval_log):
    """Test that _apply_extractors handles extractor errors gracefully."""
    df = pd.DataFrame(
        {
            "log": ["/path/log1.eval"],
            "id": [eval_sample.id],
            "epoch": [eval_sample.epoch],
        }
    )

    def failing_extractor(sample, log):
        raise RuntimeError("bang")

    failing_extractor.__name__ = "failing_extractor"

    def good_extractor(sample, log):
        return {"good_field": "ok"}

    with patch("hibayes.load.load.read_eval_log", return_value=eval_log):
        result = _apply_extractors(df, [failing_extractor, good_extractor])

    # Good extractor's data should still be present
    assert result.iloc[0]["good_field"] == "ok"


def test_end_to_end_pipeline(tmp_path: Path, eval_sample, eval_log, dl_cfg):
    """Test end-to-end pipeline using samples_df + _apply_extractors."""
    # Mock samples_df to return a base DataFrame
    base_df = pd.DataFrame(
        {
            "id": [eval_sample.id],
            "epoch": [eval_sample.epoch],
            "log": ["/fake/log.eval"],
            "model": ["gpt‑4o"],
        }
    )

    with (
        patch("hibayes.load.load.samples_df", return_value=base_df),
        patch("hibayes.load.load.read_eval_log", return_value=eval_log),
    ):
        df = get_sample_df(config=dl_cfg)

    assert len(df) == 1
    # base_extractor should have added score
    assert "score" in df.columns
    assert df.iloc[0]["score"] == 0.75


def test_end_to_end_with_fixture_and_custom_extractor():
    """Test full pipeline against real fixture with both built-in and custom extractors."""
    from hibayes.load import extractor

    fixture_path = str(Path(__file__).parent / "fixtures" / "minimal_test.eval")

    @extractor
    def task_label_extractor():
        def extract(sample, eval_log):
            return {"task_label": f"{eval_log.eval.task}_{sample.id}"}

        return extract

    config = DataLoaderConfig(
        files_to_process=[fixture_path],
        enabled_extractors=[base_extractor(), task_label_extractor()],
    )
    df = get_sample_df(config=config)

    assert len(df) == 2
    # Built-in extractor columns
    assert "score" in df.columns
    assert "model" in df.columns
    assert set(df["model"]) == {"model"}  # base_extractor strips "mockllm/"
    # Custom extractor columns
    assert "task_label" in df.columns
    assert set(df["task_label"]) == {"minimal_task_q1", "minimal_task_q2"}
    # samples_df base columns should also be present
    assert "log" in df.columns


# Tests for _load_extracted_data function


def test_load_extracted_data_csv(tmp_path: Path):
    """Test loading data from a CSV file."""
    csv_file = tmp_path / "data.csv"
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df.to_csv(csv_file, index=False)

    result = _load_extracted_data([str(csv_file)])
    assert len(result) == 3
    assert list(result.columns) == ["a", "b"]
    assert result["a"].tolist() == [1, 2, 3]


def test_load_extracted_data_parquet(tmp_path: Path):
    """Test loading data from a parquet file."""
    parquet_file = tmp_path / "data.parquet"
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df.to_parquet(parquet_file, index=False)

    result = _load_extracted_data([str(parquet_file)])
    assert len(result) == 3
    assert list(result.columns) == ["a", "b"]


def test_load_extracted_data_jsonl(tmp_path: Path):
    """Test loading data from a JSONL file."""
    jsonl_file = tmp_path / "data.jsonl"
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df.to_json(jsonl_file, orient="records", lines=True)

    result = _load_extracted_data([str(jsonl_file)])
    assert len(result) == 3
    assert list(result.columns) == ["a", "b"]


def test_load_extracted_data_multiple_files(tmp_path: Path):
    """Test loading and concatenating multiple data files."""
    csv_file = tmp_path / "data1.csv"
    parquet_file = tmp_path / "data2.parquet"

    df1 = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df2 = pd.DataFrame({"a": [3, 4], "b": ["z", "w"]})

    df1.to_csv(csv_file, index=False)
    df2.to_parquet(parquet_file, index=False)

    result = _load_extracted_data([str(csv_file), str(parquet_file)])
    assert len(result) == 4
    assert result["a"].tolist() == [1, 2, 3, 4]
    assert result["b"].tolist() == ["x", "y", "z", "w"]


def test_load_extracted_data_unsupported_format(tmp_path: Path):
    """Test that unsupported file formats raise an error."""
    txt_file = tmp_path / "data.txt"
    txt_file.write_text("some text")

    with pytest.raises(ValueError, match="Unsupported file extension"):
        _load_extracted_data([str(txt_file)])


def test_load_extracted_data_empty_list():
    """Test that an empty list raises an error."""
    with pytest.raises(ValueError, match="No data files were loaded"):
        _load_extracted_data([])
