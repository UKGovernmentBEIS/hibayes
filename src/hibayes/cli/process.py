import argparse
import pathlib

import pandas as pd

from ..analysis import AnalysisConfig, process_data
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
    data = _read_df(pathlib.Path(args.data))
    config = AnalysisConfig.from_yaml(args.config)
    display = ModellingDisplay()
    out = pathlib.Path(args.out)

    out.mkdir(parents=True, exist_ok=True)

    analysis_state = process_data(
        data=data,
        config=config.data_process,
        display=display,
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
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the processed data file (Parquet, csv, jsonl format accepted).",
    )
    parser.add_argument(
        "--out", required=True, help="dir path to save the analysis state to."
    )
    args = parser.parse_args()
    run_process(args)


if __name__ == "__main__":
    main()
