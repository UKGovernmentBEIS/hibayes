import argparse
import pathlib

from . import setup_platform


def run_process(args, display=None):
    from ..analysis import AnalysisConfig, AnalysisState, process_data
    from ..platform import configure_computation_platform
    from ..ui import ModellingDisplay

    config = AnalysisConfig.from_yaml(args.config)

    # Load from existing analysis state
    analysis_state_input = AnalysisState.load(path=pathlib.Path(args.analysis_state))
    initial_stats = (
        analysis_state_input.display_stats
        if analysis_state_input.display_stats
        else None
    )

    if display is None:
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
    parser.add_argument(
        "--no-tui",
        dest="use_tui",
        action="store_false",
        help="Use classic Rich display instead of interactive TUI",
    )
    parser.set_defaults(use_tui=True)

    args = parser.parse_args()
    setup_platform(args.config)
    if args.use_tui:
        from ..ui.textual.app import run_with_tui
        run_with_tui(run_process, args)
    else:
        run_process(args)


if __name__ == "__main__":
    main()
