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
from .ui import Display, ModellingDisplay
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


def _extract_idata_summary(idata) -> dict:
    """Extract a rich summary of InferenceData groups mirroring xarray repr.

    Each group contains:
    - dims: {dim_name: size, ...}
    - coords: {coord_name: {dims: [...], dtype: str, preview: str}, ...}
    - vars: {var_name: {dims: [...], dtype: str, shape: [...]}, ...}
    """
    summary = {}
    try:
        for group_name in idata._groups + idata._groups_warmup:
            ds = getattr(idata, group_name, None)
            if ds is None:
                continue
            group_info: dict = {}

            # Dimensions
            if hasattr(ds, "dims"):
                group_info["dims"] = {str(k): int(v) for k, v in ds.dims.items()}

            # Coordinates
            if hasattr(ds, "coords"):
                coords_info = {}
                for coord_name, coord_data in ds.coords.items():
                    coord_entry: dict = {
                        "dims": list(str(d) for d in coord_data.dims),
                        "dtype": str(coord_data.dtype),
                    }
                    # Include a short preview of coordinate values
                    try:
                        vals = coord_data.values
                        if len(vals) <= 6:
                            coord_entry["values"] = [str(v) for v in vals]
                        else:
                            head = [str(v) for v in vals[:3]]
                            tail = [str(v) for v in vals[-3:]]
                            coord_entry["values"] = head + ["..."] + tail
                    except Exception:
                        pass
                    coords_info[str(coord_name)] = coord_entry
                group_info["coords"] = coords_info

            # Data variables
            if hasattr(ds, "data_vars"):
                vars_info = {}
                for var_name, var_data in ds.data_vars.items():
                    vars_info[str(var_name)] = {
                        "dims": list(str(d) for d in var_data.dims),
                        "dtype": str(var_data.dtype),
                        "shape": list(var_data.shape),
                    }
                group_info["vars"] = vars_info

            summary[group_name] = group_info
    except Exception:
        pass
    return summary


def _extract_check_details(
    checker_name: str, model_analysis_state: ModelAnalysisState
) -> dict:
    """Extract diagnostic details relevant to a checker for display."""
    diag = model_analysis_state.diagnostics or {}
    if checker_name == "r_hat":
        vals = diag.get("r_hat")
        return {"values": vals.tolist() if hasattr(vals, "tolist") else vals, "threshold": "< 1.01"} if vals is not None else {}
    elif checker_name == "ess_bulk":
        vals = diag.get("ess_bulk")
        return {"values": vals.tolist() if hasattr(vals, "tolist") else vals, "threshold": "> 1000"} if vals is not None else {}
    elif checker_name == "ess_tail":
        vals = diag.get("ess_tail")
        return {"values": vals.tolist() if hasattr(vals, "tolist") else vals, "threshold": "> 1000"} if vals is not None else {}
    elif checker_name == "divergences":
        count = diag.get("divergences")
        return {"count": count, "threshold": "<= 0"} if count is not None else {}
    elif checker_name == "loo":
        return {
            k: v for k, v in {
                "elpd_loo": diag.get("elpd_loo"),
                "se": diag.get("se_elpd_loo"),
                "p_loo": diag.get("p_loo"),
            }.items() if v is not None
        }
    elif checker_name == "waic":
        return {
            k: v for k, v in {
                "elpd_waic": diag.get("elpd_waic"),
                "se": diag.get("se_elpd_waic"),
                "p_waic": diag.get("p_waic"),
            }.items() if v is not None
        }
    elif checker_name == "bfmi":
        vals = diag.get("bfmi")
        return {"values": vals, "threshold": ">= 0.20"} if vals is not None else {}
    else:
        # Return any scalar diagnostics we can find
        return {k: v for k, v in diag.items() if isinstance(v, (int, float, str))}


def _extract_comm_details(
    communicator_name: str,
    analysis_state: AnalysisState,
    only_keys: set[str] | None = None,
) -> dict:
    """Extract details about a communicator result for display.

    Parameters
    ----------
    only_keys:
        If given, only include items whose key is in this set.
        This prevents showing outputs from earlier communicators.
    """
    details: dict = {}
    comm = analysis_state.communicate
    if not comm:
        return details

    items = {k: v for k, v in comm.items() if only_keys is None or k in only_keys}

    tables = [k for k, v in items.items() if isinstance(v, pd.DataFrame)]
    plots = [k for k, v in items.items() if not isinstance(v, pd.DataFrame)]
    if tables:
        details["tables"] = tables
    if plots:
        details["plots"] = plots

    # Include actual table data so the TUI can render it
    table_data = {}
    for k in tables:
        df = items[k]
        table_data[k] = {
            "columns": list(df.columns),
            "index": [str(i) for i in df.index],
            "data": [
                [f"{v:.2f}" if isinstance(v, float) else str(v) for v in row]
                for row in df.values
            ],
        }
    if table_data:
        details["table_data"] = table_data

    return details


def _extract_df_stats(df: pd.DataFrame, display: Display) -> None:
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
    display: Display,
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
    display: Display,
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
    display: Display,
    out: Optional[Path] = None,
    frequent_save: bool = False,
) -> AnalysisState:
    if frequent_save and out is None:
        raise ValueError("'out' path must be provided when frequent_save=True")

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

            display.set_active_model(model_analysis_state.model_name)
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

            # Show InferenceData summary in the display
            if model_analysis_state.inference_data is not None:
                idata_summary = _extract_idata_summary(model_analysis_state.inference_data)
                if idata_summary:
                    display.show_inference_summary(idata_summary)

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
    display: Display,
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
                details=_extract_check_details(
                    registry_info(checker).name, model_analysis_state
                ),
            )

    if display.is_live:
        display.stop()

    return model_analysis_state


def _match_model_for_comm_key(key: str, model_names: list[str]) -> str | None:
    """Try to match a communicate item key to a model name.

    Communicate items follow the convention ``model_{model_name}_{suffix}``.
    Returns the matched model name, or ``None`` for cross-model items.
    """
    for name in model_names:
        prefix = f"model_{name}_"
        if key.startswith(prefix):
            return name
    return None


def communicate(
    analysis_state: AnalysisState,
    communicate_config: CommunicateConfig,
    display: Display,
    out: Optional[Path] = None,
    frequent_save: bool = False,
):
    """Run communication on the model."""
    if frequent_save and out is None:
        raise ValueError("'out' path must be provided when frequent_save=True")

    if not display.is_live:
        display.start()

    display.update_header("Running communication methods")

    # Collect known model names so we can route results per-model
    model_names = [m.model_name for m in analysis_state.models]

    display.update_logs(
        f"Enabled communicators: {[registry_info(check).name for check in communicate_config.enabled_communicators]}"
    )

    with display.capture_logs():
        for communicator in communicate_config.enabled_communicators:
            display.update_header(
                f"Running communicator for {registry_info(communicator).name}"
            )
            # Snapshot communicate keys before running
            keys_before = set(analysis_state.communicate.keys()) if analysis_state.communicate else set()

            analysis_state, outcome = communicator(analysis_state, display=display)
            display.update_logs(
                f"Communicators {registry_info(communicator).name} returned: {outcome}"
            )

            # Find new items added by this communicator
            keys_after = set(analysis_state.communicate.keys()) if analysis_state.communicate else set()
            new_keys = keys_after - keys_before

            # Route each new item to the matching model section
            model_to_keys: dict[str, set[str]] = {}
            unrouted_keys: set[str] = set()
            for key in sorted(new_keys):
                matched = _match_model_for_comm_key(key, model_names)
                if matched:
                    model_to_keys.setdefault(matched, set()).add(key)
                else:
                    unrouted_keys.add(key)

            # Emit per-model results (filtered to only this model's keys)
            for model_name, keys in model_to_keys.items():
                display.set_active_model(model_name)
                display.add_communicate_result(
                    registry_info(communicator).name,
                    outcome,
                    details=_extract_comm_details(
                        registry_info(communicator).name, analysis_state,
                        only_keys=keys,
                    ),
                )

            # Emit unrouted (cross-model) results to the first model
            if unrouted_keys and not model_to_keys:
                if model_names:
                    display.set_active_model(model_names[0])
                display.add_communicate_result(
                    registry_info(communicator).name,
                    outcome,
                    details=_extract_comm_details(
                        registry_info(communicator).name, analysis_state,
                        only_keys=unrouted_keys,
                    ),
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
