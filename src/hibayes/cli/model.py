import argparse
import pathlib

from ..analysis import AnalysisConfig, AnalysisState, model
from ..ui import ModellingDisplay


def run_model(args):
    analysis_state = AnalysisState.load(path=pathlib.Path(args.analysis_state))
    config = AnalysisConfig.from_yaml(args.config)
    display = ModellingDisplay()
    out = pathlib.Path(args.out)

    out.mkdir(parents=True, exist_ok=True)

    analysis_state = model(
        analysis_state=analysis_state,
        model_config=config.models,
        checker_config=config.checkers,
        platform_config=config.platform,
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
        "--analysis-state",
        required=True,
        help="Path to the dir containing the analysis state.",
    )
    parser.add_argument(
        "--out", required=True, help="dir path to write the DVC tracking files"
    )

    args = parser.parse_args()

    run_model(args)


if __name__ == "__main__":
    main()
