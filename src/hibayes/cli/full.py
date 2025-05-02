import argparse
import pathlib

from ..analysis import AnalysisConfig, communicate, model
from ..load import get_sample_df
from ..ui import ModellingDisplay


def run_full(args):
    config = AnalysisConfig.from_yaml(args.config)
    display = ModellingDisplay()
    out = pathlib.Path(args.out)

    print(config.data_loader)

    out.mkdir(parents=True, exist_ok=True)
    df = get_sample_df(
        display=display,
        config=config.data_loader,
    )
    df.to_parquet(
        out / "data.parquet",
        engine="pyarrow",  # auto might result in different engines in different setups)
        compression="snappy",
    )

    analysis_state = model(
        data=df,
        model_config=config.models,
        checker_config=config.checkers,
        platform_config=config.platform,
        display=display,
    )
    analysis_state.save(path=out)

    analysis_state = communicate(
        analysis_state=analysis_state,
        communicate_config=config.communicate,
        display=display,
    )
    analysis_state.save(path=out)


def main():
    parser = argparse.ArgumentParser(
        description="Fit statistical models and run quality checks using hibayes."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the configuration file (YAML format). See examples/*/*.yaml for examples",
    )
    parser.add_argument(
        "--out", required=True, help="dir path to write the DVC tracking files"
    )

    args = parser.parse_args()
    args.data = args.out + "/data.parquet"
    args.analysis_state = args.out + "/model"
    run_full(args)
