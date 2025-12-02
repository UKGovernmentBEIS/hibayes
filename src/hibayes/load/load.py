import datetime
import json
import os
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd
import pytz
from inspect_ai.log import (
    EvalLog,
    EvalLogInfo,
    list_eval_logs,
    read_eval_log,
    read_eval_log_sample,
)

from hibayes.utils import init_logger

from ..ui.display import ModellingDisplay
from .configs.config import DataLoaderConfig
from .utils import (
    LogSample,
    check_mixed_types,
    update_inspect_recorders,
)

logger = init_logger()


class LogProcessor:
    """Processes evaluation logs and extracts metadata using configured extractors"""

    def __init__(self, config: None | DataLoaderConfig = None) -> None:
        """
        Initialise the log processor with extractors

        Args:
            config: Configuration to customise extractor behavior
        """
        self.config = config if config else DataLoaderConfig.from_dict({})
        self.extractors = self._setup_extractors()

    def _setup_extractors(self):
        """Configure extractors based on the provided configuration"""
        # The extractors are now already instantiated in the config
        return self.config.enabled_extractors

    def process_sample(
        self,
        sample: LogSample,
        eval_log_header: EvalLog,
        log_info: EvalLogInfo,
        display: Optional[ModellingDisplay] = None,
    ) -> Dict[str, Any]:
        """
        Process a single sample using all configured extractors

        Args:
            sample: The evaluation sample to process
            eval_log_header: The evaluation log header
            log_info: Log information
            display: Optional ModellingDisplay for tracking progress

        Returns:
            Dictionary containing extracted metadata
        """
        row = {}
        errors = []

        loaded_sample = read_eval_log_sample(log_info, sample.id, sample.epoch)

        for extractor in self.extractors:
            try:
                extracted_data = extractor(loaded_sample, eval_log_header)
                row.update(extracted_data)
            except Exception as e:
                extractor_name = getattr(extractor, "__name__", "unknown_extractor")
                error_msg = f"Error in {extractor_name}: {str(e)}"
                errors.append(error_msg)
                logger.error(
                    f"Error processing sample {sample.id} with {extractor_name}:\n"
                    f"{traceback.format_exc()}"
                )
                if display:
                    display.update_stat(
                        "Extractor errors",
                        display.stats.get("Extractor errors", 0) + 1,
                    )

        if errors:
            row["processing_errors"] = "; ".join(errors)

        return row


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


def process_eval_logs_parallel(
    eval_logs: List[EvalLogInfo],
    processor: LogProcessor,
    cutoff: Optional[datetime.datetime],
    max_workers: Optional[int] = None,
    display: Optional[ModellingDisplay] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Process evaluation logs in parallel with Rich display

    Args:
        eval_logs: List of evaluation log entries
        processor: LogProcessor instance to use
        cutoff: Optional datetime cutoff
        max_workers: Maximum number of parallel workers
        display: Optional ModellingDisplay for visual progress tracking

    Yields:
        Rows of processed data
    """
    # First, build a complete list of (model, sample, extractor, log) tuples to process
    tasks = []
    sample_count = 0
    models_seen = set()

    from tqdm import tqdm

    if display:
        display.add_task("Identifying processing tasks", total=len(eval_logs))
        eval_logs_iterable = eval_logs
    else:
        eval_logs_iterable = tqdm(eval_logs, desc="Identifying processing tasks")

    for log_info in eval_logs_iterable:
        try:
            eval_log_header = read_eval_log(log_info, header_only=True)
            if not is_after_timestamp(cutoff, eval_log_header):
                continue

            # Track models seen
            if hasattr(eval_log_header.eval, "model"):
                model_name = eval_log_header.eval.model
                models_seen.add(model_name)
                if display:
                    display.update_stat("AI Models detected", models_seen)

            for sample in eval_log_header.samples:
                tasks.append((sample, eval_log_header, log_info))
                sample_count += 1
                if display:
                    display.update_stat("Samples found", sample_count)
            if display:
                display.update_task("Identifying processing tasks", advance=1)

        except Exception as e:
            logger.error(
                f"Error preparing log {log_info} for processing:\n"
                f"{traceback.format_exc()}"
            )
            if display:
                display.update_stat(
                    "Errors encountered",
                    display.stats.get("Errors encountered", 0) + 1,
                )

    logger.info(f"Created {len(tasks)} sample-extractor tasks to process")

    # Set up the processing task
    if display:
        process_task = display.add_task(
            "Processing samples",
            total=len(tasks),
            worker=max_workers
            if max_workers
            else min(32, os.cpu_count() + 4),  # default for threadpoolexecutor
        )
    samples_processed = 0
    errors = 0

    def process_task(task_tuple):
        sample: LogSample = task_tuple[0]
        eval_log_header: EvalLog = task_tuple[1]
        log_info: EvalLogInfo = task_tuple[2]

        try:
            result = processor.process_sample(
                sample, eval_log_header, log_info, display
            )
            return {
                "sample_error": False,
                **result,
            }
        except Exception as e:
            error_msg = f"Error in {sample.id}: {str(e)}"
            logger.error(
                f"Error processing sample {sample.id}:\n"
                f"{traceback.format_exc()}"
            )
            return {
                "model": eval_log_header.eval.model,
                "sample_id": sample.id,
                "sample_epoch": sample.epoch,
                "sample_error_message": error_msg,
                "sample_error": True,
            }

    # Process all tasks in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if display:
            task_iterator = executor.map(process_task, tasks)
        else:
            # Use tqdm for progress bar when display is False
            task_iterator = tqdm(
                executor.map(process_task, tasks),
                total=len(tasks),
                desc="Processing samples",
                unit="sample",
            )

        for result in task_iterator:
            # Either add results or track the error
            if not result["sample_error"]:
                samples_processed += 1
                if display:
                    display.update_stat("Samples processed", samples_processed)
                yield result
            else:
                errors += 1
                if display:
                    display.update_stat("Sample errors", errors)
                yield result
            if display:
                display.update_task("Processing samples", advance=1)


def row_generator(
    processor: LogProcessor,
    files_to_process: List[str],
    cutoff: Optional[datetime.datetime] = None,
    max_workers: Optional[int] = None,
    display: Optional[ModellingDisplay] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Generate rows from a list of files to process.
    Uses parallelism to speed up processing with Rich display.

    Args:
        processor: LogProcessor instance to process the logs
        files_to_process: List of paths to logs or directories containing logs
        cutoff: Optional datetime cutoff to filter logs
        max_workers: Maximum number of workers for parallel processing (defaults to CPU count)
        display: Optional ModellingDisplay instance for progress visualization

    Yields:
        Dictionary containing processed log information
    """
    if not files_to_process:
        raise ValueError("You must provide files_to_process.")

    try:
        # Process file list
        processed_files = get_file_list(files_to_process)
        if display:
            display.update_stat("Files to process", len(processed_files))

        total_logs = []

        for file_path in processed_files:
            # Check if file exists (unless it's an S3 path)
            if not ("s3://" in str(file_path) or os.path.exists(file_path)):
                logger.warning(f"File or directory does not exist: {file_path}")
                if display:
                    display.update_stat(
                        "Warnings", display.stats.get("Warnings", 0) + 1
                    )
                continue

            logs = list_eval_logs(log_dir=file_path)
            total_logs.extend(logs)

        if display:
            display.update_stat("Logs found", len(total_logs))

        yield from process_eval_logs_parallel(
            total_logs, processor, cutoff, max_workers, display
        )
    finally:
        pass


def get_sample_df(
    config: DataLoaderConfig,
    display: Optional[ModellingDisplay] = None,
) -> pd.DataFrame:
    """
    Memory-efficient version that writes to JSONL in batches before loading as DataFrame.
    Uses ModellingDisplay for visualisation if provided.

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

    # Set up the log processor
    processor = LogProcessor(config=config)

    # Use the capture context for the entire function
    with capture_context:
        logger.info("HiBayES - Loading Data")
        logger.info(f"Files to process: {config.files_to_process}")

        # If cached data path is provided and exists, read from it
        if config.cache_path and os.path.exists(config.cache_path):
            logger.info(f"Reading data from existing cache: {config.cache_path}")
            return pd.read_json(config.cache_path, lines=True)

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl").name
        logger.info(f"Using temporary JSONL file: {output_path}")

        if isinstance(config.files_to_process, str):
            files_to_process = [config.files_to_process]
        else:
            files_to_process = config.files_to_process

        # patch inspects loaders to also log sample id when in header only mode
        update_inspect_recorders()

        try:
            # Setup row generator with the display
            row_gen = row_generator(
                processor, files_to_process, config.cutoff, config.max_workers, display
            )

            # Process and write to JSONL file
            rows_processed = 0
            with open(output_path, "w", encoding="utf-8") as jsonl_file:
                logger.info(f"Writing data to: {output_path}")
                batch = []

                for row in row_gen:
                    # Convert any non-serializable objects to strings
                    serializable_row = {}
                    for k, v in row.items():
                        if isinstance(v, (datetime.datetime, datetime.date)):
                            serializable_row[k] = v.isoformat()
                        else:
                            serializable_row[k] = v

                    batch.append(serializable_row)
                    rows_processed += 1

                    if len(batch) >= config.batch_size:
                        # Write batch to file
                        for r in batch:
                            jsonl_file.write(json.dumps(r) + "\n")
                        batch = []

                # Write remaining rows
                for r in batch:
                    jsonl_file.write(json.dumps(r) + "\n")

            # Load the JSONL to DataFrame
            try:
                df = pd.read_json(output_path, lines=True, orient="records")
                logger.info(f"Loaded dataframe with shape: {df.shape}")

                # Log dataframe info
                logger.info("Data Preview:")
                logger.info(df.head().to_string())

                # Log memory usage
                memory_usage = df.memory_usage(deep=True).sum() / 1024**2
                logger.info(f"Memory usage: {memory_usage:.2f} MB")

                # If this is a temporary file and not an explicitly set output directory, delete it
                if not config.output_dir:
                    try:
                        os.unlink(output_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file: {str(e)}")
            except Exception as e:
                logger.error(f"Error loading JSONL file: {str(e)}")
                if not config.output_dir:
                    logger.info(f"JSONL file was saved at: {output_path}")
                df = pd.DataFrame()
        finally:
            # We don't stop the display as it was passed in
            logger.info("Finished processing logs.")

        # Check for mixed types in the DataFrame that will error parquet write
        check_mixed_types(df)

        if display:
            if display.live:
                display.stop()
        return df
