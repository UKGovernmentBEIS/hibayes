import argparse
import pathlib


from ..analysis import AnalysisConfig, AnalysisState, communicate
from ..ui import ModellingDisplay


def run_communicate(args):
    analysis_state = AnalysisState.load(path=pathlib.Path(args.analysis_state))
    config = AnalysisConfig.from_yaml(args.config)
    display = ModellingDisplay()
    out = pathlib.Path(args.out)

    out.mkdir(parents=True, exist_ok=True)

    analysis_state = communicate(
        analysis_state=analysis_state,
        communicate_config=config.communicate,
        display=display,
    )
    analysis_state.save(path=out)


def main():
    parser = argparse.ArgumentParser(
        description="Communicate results from a fitted model"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the configuration file (YAML format). See examples/*/*.yaml for examples",
    )
    parser.add_argument(
        "--analysis_state",
        required=True,
        help="Path to the analysis state dir",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="dir path to write the DVC tracking files (plots and tables)",
    )

    args = parser.parse_args()
    run_communicate(args)  # will save the results in the out dir.


if __name__ == "__main__":
    main()
