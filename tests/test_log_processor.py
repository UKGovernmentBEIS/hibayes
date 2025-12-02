import datetime
from pathlib import Path
from typing import Any
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
    LogProcessor,
    get_file_list,
    get_sample_df,
    is_after_timestamp,
    process_eval_logs_parallel,
    row_generator,
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


def test_processor_setup(dl_cfg):
    proc = LogProcessor(dl_cfg)
    # With the new functional extractors, the config uses functional extractors by default
    # Check that we have 3 extractors (base, tools, tokens)
    assert len(proc.extractors) == 3


def test_processor_process_sample(eval_log, log_info, dl_cfg):
    proc = LogProcessor(dl_cfg)
    with patch(
        "hibayes.load.load.read_eval_log_sample", return_value=eval_log.samples[0]
    ):
        row = proc.process_sample(eval_log.samples[0], eval_log, log_info)
    assert row["tools"] == ["grep"]
    assert row["score"] == 0.75


def test_processor_unknown_extractor(tmp_path: Path):
    # This should raise an error when trying to get a non-existent extractor from registry
    with pytest.raises(KeyError, match="No registered does-not-exist found"):
        cfg = DataLoaderConfig.from_dict(
            {
                "extractors": {"enabled": ["base_extractor", "does-not-exist"]},
                "paths": {"files_to_process": [str(tmp_path)]},
            }
        )


def test_processor_error_capture(eval_log, log_info):
    # Test error capture with a mock that will fail
    cfg = DataLoaderConfig.from_dict(
        {
            "extractors": {"enabled": ["base_extractor"]},
            "paths": {"files_to_process": ["/logs"]},
        }
    )
    proc = LogProcessor(cfg)

    # Mock an extractor to fail
    def failing_extractor(sample, eval_log):
        raise RuntimeError("bang")

    failing_extractor.__name__ = "failing_extractor"
    proc.extractors.append(failing_extractor)

    with patch(
        "hibayes.load.load.read_eval_log_sample", return_value=eval_log.samples[0]
    ):
        row = proc.process_sample(eval_log.samples[0], eval_log, log_info)
    assert "processing_errors" in row


def test_processor_with_multiple_custom_extractors(eval_log, log_info):
    """Test processor with multiple extractors."""
    # With registry pattern, use multiple registered extractors
    config = DataLoaderConfig.from_dict(
        {
            "extractors": {
                "enabled": ["base_extractor", "token_extractor", "tools_extractor"]
            },
            "paths": {"files_to_process": ["test_dir"]},
        }
    )

    proc = LogProcessor(config)

    assert len(proc.extractors) == 3

    with patch(
        "hibayes.load.load.read_eval_log_sample", return_value=eval_log.samples[0]
    ):
        row = proc.process_sample(eval_log.samples[0], eval_log, log_info)

    # Check that extractors produced expected fields
    assert "score" in row  # From base_extractor
    assert "total_tokens" in row  # From token_extractor
    assert "tools" in row  # From tools_extractor


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


def test_row_generator_flow(dl_cfg):
    proc = LogProcessor(dl_cfg)
    with (
        patch("hibayes.load.load.list_eval_logs", return_value=["foobar"]),
        patch(
            "hibayes.load.load.process_eval_logs_parallel",
            return_value=iter([{"score": 0.75, "model": "gpt‑4o"}]),
        ),
    ):
        rows = list(row_generator(processor=proc, files_to_process=["/fake"]))
    assert rows[0]["score"] == 0.75


def test_row_generator_empty_validation(dl_cfg):
    proc = LogProcessor(dl_cfg)
    with pytest.raises(ValueError):
        next(row_generator(processor=proc, files_to_process=[]))


def test_row_generator_s3_path(dl_cfg):
    proc = LogProcessor(dl_cfg)
    with patch("hibayes.load.load.list_eval_logs", return_value=[]):
        list(row_generator(processor=proc, files_to_process=["s3://bucket/key"]))


def test_row_generator_nonexistent_path(dl_cfg):
    proc = LogProcessor(dl_cfg)
    with (
        patch("os.path.exists", return_value=False),
        patch("hibayes.load.load.list_eval_logs", return_value=[]),
    ):
        rows = list(row_generator(processor=proc, files_to_process=["/nope"]))
    assert rows == []


def test_get_sample_df_from_path_cached(tmp_path: Path, dl_cfg):
    cached = tmp_path / "cached.jsonl"
    cached.write_text("{score: 0.9}\n")
    dl_cfg.cache_path = str(cached)
    with patch("pandas.read_json", return_value=pd.DataFrame([{"score": 0.9}])):
        df = get_sample_df(
            config=dl_cfg,
        )
    assert df.iloc[0]["score"] == 0.9


def test_process_eval_logs_parallel(eval_sample, eval_log):
    info = MagicMock(spec=EvalLogInfo)
    with (
        patch("hibayes.load.load.read_eval_log", return_value=eval_log),
        patch("hibayes.load.load.read_eval_log_sample", return_value=eval_sample),
    ):
        config = DataLoaderConfig.from_dict(
            {
                "extractors": {"enabled": ["base_extractor"]},
                "paths": {"files_to_process": ["/"]},
            }
        )
        proc = LogProcessor(config)
        rows = list(
            process_eval_logs_parallel(eval_logs=[info], processor=proc, cutoff=None)
        )
    assert rows[0]["score"] == 0.75


def test_end_to_end_pipeline(tmp_path: Path, eval_sample, eval_log, log_info, dl_cfg):
    with (
        patch("hibayes.load.load.list_eval_logs", return_value=[log_info]),
        patch("hibayes.load.load.read_eval_log", return_value=eval_log),
        patch("hibayes.load.load.read_eval_log_sample", return_value=eval_sample),
        patch("os.path.exists", return_value=True),
    ):
        df = get_sample_df(config=dl_cfg)
    assert len(df) == 1 and df.loc[0, "score"] == 0.75


def test_processor_with_functional_extractors(eval_log, log_info):
    """Test processor with functional extractors."""
    # Use string references to extractors
    config = DataLoaderConfig.from_dict(
        {
            "extractors": {"enabled": ["base_extractor", "token_extractor"]},
            "paths": {"files_to_process": ["test_dir"]},
        }
    )

    proc = LogProcessor(config)
    assert len(proc.extractors) == 2

    with patch(
        "hibayes.load.load.read_eval_log_sample", return_value=eval_log.samples[0]
    ):
        row = proc.process_sample(eval_log.samples[0], eval_log, log_info)

    # Should have results from both extractors
    assert "score" in row  # From base_extractor
    assert "total_tokens" in row  # From token_extractor


def test_processor_with_mixed_extractors(eval_log, log_info):
    """Test processor with parameterized extractors."""
    # Use extractors with parameters
    config = DataLoaderConfig.from_dict(
        {
            "extractors": {
                "enabled": [
                    "base_extractor",
                    {"message_count_extractor": {"include_system": False}},
                ]
            },
            "paths": {"files_to_process": ["test_dir"]},
        }
    )

    proc = LogProcessor(config)
    assert len(proc.extractors) == 2

    with patch(
        "hibayes.load.load.read_eval_log_sample", return_value=eval_log.samples[0]
    ):
        row = proc.process_sample(eval_log.samples[0], eval_log, log_info)

    # Should have results from both extractors
    assert "score" in row  # From base_extractor
    assert "total_messages" in row  # From message_count_extractor


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
