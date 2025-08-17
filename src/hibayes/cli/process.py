import argparse
import pathlib

import pandas as pd

from ..analysis import AnalysisConfig, AnalysisState, process_data
from ..platform import configure_computation_platform
from ..ui import ModellingDisplay


def _read_df(path: pathlib.Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".jsonl", ".json"):
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported file extension: {ext}")


def run_process(args):
    config = AnalysisConfig.from_yaml(args.config)

    # Handle different input options
    if hasattr(args, "analysis_state") and args.analysis_state:
        # Load from existing analysis state
        analysis_state_input = AnalysisState.load(
            path=pathlib.Path(args.analysis_state)
        )
        data_input = None
        initial_stats = (
            analysis_state_input.display_stats
            if analysis_state_input.display_stats
            else None
        )
    else:
        # Load from data file
        data_input = _read_df(pathlib.Path(args.data))
        analysis_state_input = None
        initial_stats = None

    display = ModellingDisplay(initial_stats=initial_stats)
    out = pathlib.Path(args.out)

    configure_computation_platform(
        platform_config=config.platform,
        display=display,
    )

    out.mkdir(parents=True, exist_ok=True)

    analysis_state = process_data(
        config=config.data_process,
        display=display,
        data=data_input,
        analysis_state=analysis_state_input,
    )
    analysis_state.save(path=out)


def main():
    parser = argparse.ArgumentParser(
        description="Extract features from loaded data and save in analysis state."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to the configuration file (YAML format). See examples",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--data",
        help="Path to the processed data file (Parquet, csv, jsonl format accepted).",
    )
    input_group.add_argument(
        "--analysis-state",
        dest="analysis_state",
        help="Path to existing analysis state directory to load and process.",
    )

    parser.add_argument(
        "--out", required=True, help="dir path to save the analysis state to."
    )
    args = parser.parse_args()
    run_process(args)


if __name__ == "__main__":
    main()
