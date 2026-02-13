import argparse
import pathlib

from . import setup_platform


def run_model(args, display=None):
    from ..analysis import AnalysisConfig, AnalysisState, model
    from ..platform import configure_computation_platform
    from ..ui import ModellingDisplay

    config = AnalysisConfig.from_yaml(args.config)

    # Load analysis state first to get display stats
    analysis_state = AnalysisState.load(path=pathlib.Path(args.analysis_state))

    if display is None:
        display = ModellingDisplay(
            initial_stats=analysis_state.display_stats
            if analysis_state.display_stats
            else None
        )
    out = pathlib.Path(args.out)

    configure_computation_platform(
        platform_config=config.platform,
        display=display,
    )

    out.mkdir(parents=True, exist_ok=True)

    analysis_state = model(
        analysis_state=analysis_state,
        models_to_run_config=config.models,
        checker_config=config.checkers,
        platform_config=config.platform,
        display=display,
        out=out,
        frequent_save=args.frequent_save
    )
    # Final save - use incremental if files already exist (from frequent saves or previous run)
    incremental = (out / "data.parquet").exists()
    analysis_state.save(path=out, incremental=incremental)


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
        dest="analysis_state",
        required=True,
        help="Path to the dir containing the analysis state.",
    )
    parser.add_argument(
        "--out", required=True, help="dir path to write the DVC tracking files"
    )

    parser.add_argument(
        "--no-frequent-save",
        dest="frequent_save",
        action="store_false",
        help="If set, disables saving after each model fit. By default, frequent saves are enabled."
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
    setup_platform(args.config)
    if args.use_tui:
        from ..ui.textual.app import run_with_tui
        run_with_tui(run_model, args)
    else:
        run_model(args)


if __name__ == "__main__":
    main()
