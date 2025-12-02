import argparse
import pathlib

from ..analysis import AnalysisConfig, AnalysisState, process_data
from ..platform import configure_computation_platform
from ..ui import ModellingDisplay


def run_process(args):
    config = AnalysisConfig.from_yaml(args.config)

    # Load from existing analysis state
    analysis_state_input = AnalysisState.load(path=pathlib.Path(args.analysis_state))
    initial_stats = (
        analysis_state_input.display_stats
        if analysis_state_input.display_stats
        else None
    )

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

    parser.add_argument(
        "--analysis-state",
        dest="analysis_state",
        required=True,
        help="Path to existing analysis state directory to load and process.",
    )

    parser.add_argument(
        "--out", required=True, help="dir path to save the analysis state to."
    )
    args = parser.parse_args()
    run_process(args)


if __name__ == "__main__":
    main()
