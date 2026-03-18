import datetime
import os
import traceback
from contextlib import nullcontext
from typing import List, Optional

import pandas as pd
import pytz
from inspect_ai.analysis import EvalInfo, EvalModel, SampleSummary, samples_df
from inspect_ai.log import (
    EvalLog,
    read_eval_log,
)

from hibayes.utils import init_logger

from ..ui.display import ModellingDisplay
from .configs.config import DataLoaderConfig
from .utils import check_mixed_types

logger = init_logger()


def is_after_timestamp(timestamp: Optional[datetime.datetime], log: EvalLog) -> bool:
    """Check if a log was completed after a given timestamp"""
    if timestamp is None:
        return True

    cutoff = pytz.utc.localize(timestamp) if not timestamp.tzinfo else timestamp
    log_time = datetime.datetime.strptime(log.stats.completed_at, "%Y-%m-%dT%H:%M:%S%z")
    return log_time > cutoff


def get_file_list(files_to_process: List[str]) -> List[str]:
    """
    Process a list of file paths or text files containing paths

    Args:
        files_to_process: List of file paths or .txt files containing paths

    Returns:
        List of unique file paths
    """
    files = []
    for item in files_to_process:
        if item.endswith(".txt"):
            with open(item, "r") as f:
                files.extend([line.strip() for line in f if line.strip()])
        else:
            files.append(item)
    # Remove duplicates while preserving order
    return list(dict.fromkeys(files))


def _apply_cutoff(df: pd.DataFrame, cutoff: datetime.datetime) -> pd.DataFrame:
    """Filter logs by stats.completed_at, preserving existing cutoff semantics."""
    passing_logs = set()
    for log_path in df["log"].unique():
        try:
            header = read_eval_log(log_path, header_only=True)
            if is_after_timestamp(cutoff, header):
                passing_logs.add(log_path)
        except Exception as e:
            logger.error(f"Error reading log header {log_path}: {e}")
    return df[df["log"].isin(passing_logs)]


def _apply_extractors(
    df: pd.DataFrame,
    extractors: list,
    display: Optional[ModellingDisplay] = None,
) -> pd.DataFrame:
    """Apply extractors to samples by loading each log file exactly once."""
    results = []
    log_paths = df["log"].unique()
    samples_processed = 0

    if display:
        display.add_task("Applying extractors", total=len(df))
        display.update_stat("Samples found", len(df))

    for log_path in log_paths:
        try:
            eval_log = read_eval_log(log_path)
        except Exception as e:
            logger.error(
                f"Error reading log {log_path}:\n{traceback.format_exc()}"
            )
            if display:
                display.update_stat(
                    "Errors encountered",
                    display.stats.get("Errors encountered", 0) + 1,
                )
            continue

        sample_lookup = {(str(s.id), int(s.epoch)): s for s in eval_log.samples}
        log_rows = df[df["log"] == log_path]

        for idx, row in log_rows.iterrows():
            sample = sample_lookup.get((str(row["id"]), int(row["epoch"])))
            if sample is None:
                if display:
                    display.update_task("Applying extractors", advance=1)
                continue
            extracted = {}
            for extractor in extractors:
                try:
                    extracted.update(extractor(sample, eval_log))
                except Exception as e:
                    extractor_name = getattr(extractor, "__name__", "unknown_extractor")
                    logger.error(
                        f"Error processing sample {row['id']} with {extractor_name}:\n"
                        f"{traceback.format_exc()}"
                    )
                    if display:
                        display.update_stat(
                            "Extractor errors",
                            display.stats.get("Extractor errors", 0) + 1,
                        )
            results.append((idx, extracted))
            samples_processed += 1
            if display:
                display.update_stat("Samples processed", samples_processed)
                display.update_task("Applying extractors", advance=1)

    if results:
        ext_df = pd.DataFrame.from_dict(
            {idx: data for idx, data in results},
            orient="index",
        )
        # Drop overlapping columns from base df before joining extractor results,
        # so extractor-produced columns take precedence
        overlap = ext_df.columns.intersection(df.columns)
        if len(overlap) > 0:
            df = df.drop(columns=overlap)
        df = df.join(ext_df)
    return df


def get_sample_df(
    config: DataLoaderConfig,
    display: Optional[ModellingDisplay] = None,
) -> pd.DataFrame:
    """
    Load evaluation samples via inspect_ai.analysis.samples_df() and optionally
    apply custom extractors for additional columns.

    Args:
        config: DataLoaderConfig instance with configuration
        display: Optional ModellingDisplay instance for progress visualisation

    Returns:
        DataFrame containing the processed logs
    """
    # Set up display and log capture context at the start
    if display:
        if not display.live:
            display.start()
        display.update_header("Processing Logs")
        capture_context = display.capture_logs()
    else:
        capture_context = nullcontext()

    with capture_context:
        logger.info("HiBayES - Loading Data")
        logger.info(f"Files to process: {config.files_to_process}")

        # If cached data path is provided and exists, read from it
        if config.cache_path and os.path.exists(config.cache_path):
            logger.info(f"Reading data from existing cache: {config.cache_path}")
            return pd.read_json(config.cache_path, lines=True)

        if isinstance(config.files_to_process, str):
            files_to_process = [config.files_to_process]
        else:
            files_to_process = config.files_to_process

        # Process file list (expand .txt manifests, deduplicate)
        processed_files = get_file_list(files_to_process)
        if display:
            display.update_stat("Files to process", len(processed_files))

        # Phase 1: Get base DataFrame via samples_df()
        if display:
            display.add_task("Loading samples", total=len(processed_files))
        columns = list(SampleSummary) + list(EvalInfo) + list(EvalModel)
        base_df = samples_df(
            logs=processed_files,
            columns=columns,
            parallel=True,
            quiet=True,
        )
        logger.info(f"samples_df returned {len(base_df)} rows")
        if display:
            display.update_task("Loading samples", advance=len(processed_files))
            display.update_stat("Samples found", len(base_df))
            display.update_stat(
                "AI Models detected", set(base_df["model"].unique())
            )
            display.update_stat("Logs found", len(base_df["log"].unique()))

        # Phase 2: Apply cutoff filter if configured
        if config.cutoff:
            base_df = _apply_cutoff(base_df, config.cutoff)
            logger.info(f"After cutoff filter: {len(base_df)} rows")

        # Phase 3: If extractors configured, apply them via bulk log reads
        if config.enabled_extractors:
            if display:
                display.update_stat("Logs to process", len(base_df["log"].unique()))
            base_df = _apply_extractors(base_df, config.enabled_extractors, display)

        logger.info(f"Final dataframe shape: {base_df.shape}")
        logger.info("Finished processing logs.")

        # Check for mixed types in the DataFrame that will error parquet write
        check_mixed_types(base_df)

        if display:
            if display.live:
                display.stop()
        return base_df
