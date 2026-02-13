import argparse
import pathlib

from ..analysis import AnalysisConfig, AnalysisState, communicate
from ..ui import ModellingDisplay


def run_communicate(args, display=None):
    analysis_state = AnalysisState.load(path=pathlib.Path(args.analysis_state))
    config = AnalysisConfig.from_yaml(args.config)

    if display is None:
        display = ModellingDisplay(
            initial_stats=analysis_state.display_stats
            if analysis_state.display_stats
            else None
        )
    out = pathlib.Path(args.out)

    out.mkdir(parents=True, exist_ok=True)

    analysis_state = communicate(
        analysis_state=analysis_state,
        communicate_config=config.communicate,
        display=display,
        out=out,
        frequent_save=args.frequent_save
    )
    # Final save - use incremental if files already exist (from frequent saves or previous run)
    incremental = (out / "data.parquet").exists()
    analysis_state.save(path=out, incremental=incremental)


def main():
    parser = argparse.ArgumentParser(
        description="Communicate results from a fitted model"
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
        help="Path to the analysis state dir",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="dir path to write the DVC tracking files (plots and tables)",
    )

    parser.add_argument(
        "--no-frequent-save",
        dest="frequent_save",
        action="store_false",
        help="If set, disables saving after each communicator run. By default, frequent saves are enabled."
    )
    parser.set_defaults(frequent_save=True)

    parser.add_argument(
        "--no-tui",
        dest="use_tui",
        action="store_false",
        help="Use classic Rich display instead of interactive TUI",
    )
    parser.set_defaults(use_tui=True)

    args = parser.parse_args()
    if args.use_tui:
        from ..ui.textual.app import run_with_tui
        run_with_tui(run_communicate, args)
    else:
        run_communicate(args)


if __name__ == "__main__":
    main()
