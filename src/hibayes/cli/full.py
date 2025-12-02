import argparse
import pathlib

from ..analysis import AnalysisConfig, communicate, load_data, model, process_data
from ..platform import configure_computation_platform
from ..ui import ModellingDisplay


def run_full(args):
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
    run_full(args)
