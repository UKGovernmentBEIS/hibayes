from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml

from .analysis_state import AnalysisState, ModelAnalysisState
from .check import CheckerConfig
from .communicate import CommunicateConfig
from .load import (
    DataLoaderConfig,
    get_sample_df,
)
from .model import ModelsToRunConfig, fit
from .platform import PlatformConfig
from .process import ProcessConfig
from .registry import registry_info
from .ui import ModellingDisplay
from .utils import init_logger

logger = init_logger()

# TODO: data loader which checks if .json or .eval is passed
# TODO: before data is loaded from eval logs check that the extractors are going
# to extract the required variables.


def _load_extracted_data(file_paths: List[str]) -> pd.DataFrame:
    """
    Load pre-extracted data from CSV, parquet, or JSONL files.

    Args:
        file_paths: List of paths to data files.

    Returns:
        Concatenated DataFrame from all files.
    """
    dfs = []
    for path in file_paths:
        path = Path(path)
        ext = path.suffix.lower()
        if ext == ".parquet":
            dfs.append(pd.read_parquet(path))
        elif ext == ".csv":
            dfs.append(pd.read_csv(path))
        elif ext in (".jsonl", ".json"):
            dfs.append(pd.read_json(path, lines=True))
        else:
            raise ValueError(
                f"Unsupported file extension: {ext}. "
                "Supported formats: .csv, .parquet, .jsonl, .json"
            )
        logger.info(f"Loaded {path} with shape {dfs[-1].shape}")

    if not dfs:
        raise ValueError("No data files were loaded from extracted_data paths.")

    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined DataFrame shape: {df.shape}")
    return df


@dataclass
class AnalysisConfig:
    """Optional configuration object for the analysis pipeline."""

    data_loader: DataLoaderConfig
    data_process: ProcessConfig
    models: ModelsToRunConfig
    checkers: CheckerConfig
    communicate: CommunicateConfig
    platform: PlatformConfig

    @classmethod
    def from_yaml(cls, path: str) -> "AnalysisConfig":
        """Load configuration from a yaml file."""
        with open(path, "r") as f:
            config: dict = yaml.safe_load(f)

        return cls.from_dict(config)

    @classmethod
    def from_dict(cls, config: dict) -> "AnalysisConfig":
        """Load configuration from a dictionary."""

        return cls(
            data_loader=DataLoaderConfig.from_dict(
                config["data_loader"] if "data_loader" in config else {}
            ),
            data_process=ProcessConfig.from_dict(
                config["data_process"] if "data_process" in config else {}
            ),
            models=ModelsToRunConfig.from_dict(
                config["model"] if "model" in config else {},
            ),
            checkers=CheckerConfig.from_dict(
                config["check"] if "check" in config else {}
            ),
            communicate=CommunicateConfig.from_dict(
                config["communicate"] if "communicate" in config else {}
            ),
            platform=PlatformConfig.from_dict(
                config["platform"] if "platform" in config else {}
            ),
        )


def _extract_df_stats(df: pd.DataFrame, display: ModellingDisplay) -> None:
    """
    Extract summary statistics from a DataFrame and update the display.

    Looks for common columns like 'model', 'dataset', etc. and reports
    unique values found. Treats each row as a sample.
    """
    # Total samples (rows)
    display.update_stat("Samples found", len(df))

    # Check for model column (AI models detected)
    if "model" in df.columns:
        models = set(df["model"].dropna().unique())
        if models:
            display.update_stat("AI Models detected", models)
            logger.info(f"AI Models detected: {models}")

    # Check for dataset column
    if "dataset" in df.columns:
        datasets = set(df["dataset"].dropna().unique())
        if datasets:
            display.update_stat("Datasets", datasets)
            logger.info(f"Datasets: {datasets}")

    # Check for task column
    if "task" in df.columns:
        tasks = set(df["task"].dropna().unique())
        if tasks:
            display.update_stat("Tasks", tasks)
            logger.info(f"Tasks: {tasks}")

    # Log column summary
    logger.info(f"DataFrame columns: {list(df.columns)}")
    logger.info(f"DataFrame shape: {df.shape}")


def load_data(
    config: DataLoaderConfig,
    display: ModellingDisplay,
) -> AnalysisState:
    # Check if we're loading pre-extracted data or processing eval logs
    if config.extracted_data:
        # Load pre-extracted CSV/parquet/JSONL files
        if not display.is_live:
            display.start()
        display.update_header("Loading Pre-Extracted Data")
        with display.capture_logs():
            logger.info(f"Loading extracted data from: {config.extracted_data}")
            df = _load_extracted_data(config.extracted_data)
            display.update_stat("Files loaded", len(config.extracted_data))

            # Extract summary statistics from the loaded data
            _extract_df_stats(df, display)
    else:
        # Process eval logs using extractors
        df = get_sample_df(
            display=display,
            config=config,
        )

    if display.is_live:
        display.stop()

    # Create AnalysisState with the loaded data
    analysis_state = AnalysisState(data=df)

    # Capture logs and display stats from the loading process
    analysis_state.logs["load"] = display.get_all_logs()
    analysis_state.display_stats = display.get_stats_for_persistence()

    return analysis_state


def process_data(
    config: ProcessConfig,
    display: ModellingDisplay,
    data: Optional[pd.DataFrame] = None,
    analysis_state: Optional[AnalysisState | str | Path] = None,
) -> AnalysisState:
    # Handle input options
    if data is not None and analysis_state is not None:
        raise ValueError("Provide either 'data' or 'analysis_state', not both")
    elif data is None and analysis_state is None:
        raise ValueError("Must provide either 'data' or 'analysis_state'")

    # Create or load the analysis state
    if data is not None:
        analysis_state = AnalysisState(data=data)
    elif isinstance(analysis_state, (str, Path)):
        analysis_state = AnalysisState.load(Path(analysis_state))
    # else: use the provided AnalysisState object as-is

    if not display.is_live:
        display.start()

    display.update_header("Running data processing methods")
    display.update_logs(
        f"Enabled processors: {[registry_info(proc).name for proc in config.enabled_processors]}"
    )
    with display.capture_logs():
        for processor in config.enabled_processors:
            display.update_header(f"Running processor {registry_info(processor).name}")
            analysis_state = processor(analysis_state, display=display)

    if display.is_live:
        display.stop()

    # Capture all logs from display and add to analysis state
    analysis_state.logs["process"] = display.get_all_logs()
    # Capture display stats for persistence
    analysis_state.display_stats = display.get_stats_for_persistence()
    return analysis_state


def model(
    analysis_state: AnalysisState,
    models_to_run_config: ModelsToRunConfig,
    checker_config: CheckerConfig,
    platform_config: PlatformConfig,
    display: ModellingDisplay,
    out: Path,
    frequent_save: bool = False,
) -> AnalysisState:
    display.setupt_for_modelling()

    if not display.is_live:
        display.start()

    with display.capture_logs():
        for model, model_config in models_to_run_config.enabled_models:
            model_analysis_state = ModelAnalysisState(
                model=model,
                model_config=model_config,
                platform_config=platform_config,
                features=analysis_state.features,
                coords=analysis_state.coords,
                dims=analysis_state.dims,
            )

            display.update_header(f"Running model {model_analysis_state.model_name}")
            # checks before fitting e.g. prior predictive checks
            model_checks(
                model_analysis_state=model_analysis_state,
                checker_config=checker_config,
                display=display,
            )
            if not display.is_live:
                display.start()

            fit(model_analysis_state=model_analysis_state, display=display)

            # checks after fitting e.g. posterior predictive checks
            model_checks(
                model_analysis_state=model_analysis_state,
                checker_config=checker_config,
                display=display,
            )

            analysis_state.add_model(model_analysis_state)

            if frequent_save:
                # save model after each fit (incremental after first save)
                incremental = (out / "data.parquet").exists()
                analysis_state.save(path=out, incremental=incremental)

    if display.is_live:
        display.stop()

    # Update logs from display
    analysis_state.logs["model"] = display.get_all_logs()
    # Capture display stats for persistence
    analysis_state.display_stats = display.get_stats_for_persistence()
    return analysis_state


def model_checks(
    model_analysis_state: ModelAnalysisState,
    checker_config: CheckerConfig,
    display: ModellingDisplay,
):
    """Run checks on the model."""
    if not display.is_live:
        display.start()

    display.update_header(f"Running checks for {model_analysis_state.model_name}")

    when = "after" if model_analysis_state.is_fitted else "before"

    display.update_logs(
        f"Enabled checks: {[registry_info(check).name for check in checker_config.get_checkers(when=when)]}"
    )
    with display.capture_logs():
        for checker in checker_config.get_checkers(when=when):
            model_analysis_state, outcome = checker(
                model_analysis_state, display=display
            )
            display.update_logs(
                f"Checker {registry_info(checker).name} for model {model_analysis_state.model_name} returned: {outcome}"
            )
            display.add_check(
                registry_info(checker).name,
                outcome,
            )

    if display.is_live:
        display.stop()

    return model_analysis_state


def communicate(
    analysis_state: AnalysisState,
    communicate_config: CommunicateConfig,
    display: ModellingDisplay,
    out: Path,
    frequent_save: bool = False
):
    """Run communication on the model."""

    if not display.is_live:
        display.start()

    display.update_header("Running communication methods")

    display.update_logs(
        f"Enabled communicators: {[registry_info(check).name for check in communicate_config.enabled_communicators]}"
    )

    with display.capture_logs():
        for communicator in communicate_config.enabled_communicators:
            display.update_header(
                f"Running communicator for {registry_info(communicator).name}"
            )
            analysis_state, outcome = communicator(analysis_state, display=display)
            display.update_logs(
                f"Communicators {registry_info(communicator).name} returned: {outcome}"
            )

            if frequent_save:
                # save after each communicator (incremental after first save)
                incremental = (out / "data.parquet").exists()
                analysis_state.save(path=out, incremental=incremental)

    if display.is_live:
        display.stop()

    # Update logs from display
    analysis_state.logs["communicate"] = display.get_all_logs()
    # Capture display stats for persistence
    analysis_state.display_stats = display.get_stats_for_persistence()
    return analysis_state


def stamp():
    """
    Use DVC to version the data, model, and outputs.
    """
    pass


def check_attributes():
    pass
