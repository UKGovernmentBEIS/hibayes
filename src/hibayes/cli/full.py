import argparse
import pathlib

from . import setup_platform


def run_full(args):
    from ..analysis import AnalysisConfig, communicate, load_data, model, process_data
    from ..platform import configure_computation_platform
    from ..ui import ModellingDisplay

    config = AnalysisConfig.from_yaml(args.config)
    display = ModellingDisplay()
    out = pathlib.Path(args.out)

    configure_computation_platform(
        platform_config=config.platform,
        display=display,
    )

    out.mkdir(parents=True, exist_ok=True)

    # Load data (handles both extracted_data and files_to_process)
    analysis_state = load_data(
        config=config.data_loader,
        display=display,
    )

    # Process the loaded data
    analysis_state = process_data(
        analysis_state=analysis_state,
        config=config.data_process,
        display=display,
    )

    analysis_state.save(path=out)

    analysis_state = model(
        analysis_state=analysis_state,
        models_to_run_config=config.models,
        checker_config=config.checkers,
        platform_config=config.platform,
        display=display,
        out=out,
        frequent_save=args.frequent_save,
    )
    # Save after model - incremental since files already exist
    analysis_state.save(path=out, incremental=True)

    analysis_state = communicate(
        analysis_state=analysis_state,
        communicate_config=config.communicate,
        display=display,
        out=out,
        frequent_save=args.frequent_save,
    )
    # Final save - incremental since files already exist
    analysis_state.save(path=out, incremental=True)


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

    parser.add_argument(
        "--no-frequent-save",
        dest="frequent_save",
        action="store_false",
        help="If set, disables saving after each model fit and communicator run. By default, frequent saves are enabled."
    )
    parser.set_defaults(frequent_save=True)

    args = parser.parse_args()
    setup_platform(args.config)
    run_full(args)
