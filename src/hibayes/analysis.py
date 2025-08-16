from dataclasses import dataclass, field

import pandas as pd
import yaml

from .analysis_state import AnalysisState, ModelAnalysisState
from .check import CheckerConfig
from .communicate import CommunicateConfig
from .load import (
    DataLoaderConfig,
    get_sample_df,
)
from .model import ModelsToRunConfig, fit
from .platform import PlatformConfig
from .process import ProcessConfig
from .registry import registry_info
from .ui import ModellingDisplay

# TODO: data loader which checks if .json or .eval is passed
# TODO: before data is loaded from eval logs check that the extractors are going
# to extract the required variables.


@dataclass
class AnalysisConfig:
    """Optional configuration object for the analysis pipeline."""

    data_loader: DataLoaderConfig
    data_process: ProcessConfig
    models: ModelsToRunConfig
    checkers: CheckerConfig
    communicate: CommunicateConfig
    platform: PlatformConfig

    @classmethod
    def from_yaml(cls, path: str) -> "AnalysisConfig":
        """Load configuration from a yaml file."""
        with open(path, "r") as f:
            config: dict = yaml.safe_load(f)

        return cls.from_dict(config)

    @classmethod
    def from_dict(cls, config: dict) -> "AnalysisConfig":
        """Load configuration from a dictionary."""

        return cls(
            data_loader=DataLoaderConfig.from_dict(
                config["data_loader"] if "data_loader" in config else {}
            ),
            data_process=ProcessConfig.from_dict(
                config["data_process"] if "data_process" in config else {}
            ),
            models=ModelsToRunConfig.from_dict(
                config["model"] if "model" in config else {},
            ),
            checkers=CheckerConfig.from_dict(
                config["check"] if "check" in config else {}
            ),
            communicate=CommunicateConfig.from_dict(
                config["communicate"] if "communicate" in config else {}
            ),
            platform=PlatformConfig.from_dict(
                config["platform"] if "platform" in config else {}
            ),
        )


def load_data(
    config: DataLoaderConfig,
    display: ModellingDisplay,
) -> pd.DataFrame:
    df = get_sample_df(
        display=display,
        config=config,
    )
    if display.is_live:
        display.stop()

    return df


def process_data(
    data: pd.DataFrame,
    config: ProcessConfig,
    display: ModellingDisplay,
) -> AnalysisState:
    if not display.is_live:
        display.start()
    analysis_state = AnalysisState(data=data)

    display.update_header("Running data processing methods")
    display.update_logs(
        f"Enabled processors: {[registry_info(proc).name for proc in config.enabled_processors]}"
    )
    with display.capture_logs():
        for processor in config.enabled_processors:
            display.update_header(f"Running processor {registry_info(processor).name}")
            analysis_state = processor(analysis_state, display=display)

    if display.is_live:
        display.stop()

    # Capture all logs from display and add to analysis state
    analysis_state.logs = display.get_all_logs()
    return analysis_state


def model(
    analysis_state: AnalysisState,
    models_to_run_config: ModelsToRunConfig,
    checker_config: CheckerConfig,
    platform_config: PlatformConfig,
    display: ModellingDisplay,
):
    display.setupt_for_modelling()

    if not display.is_live:
        display.start()

    with display.capture_logs():
        for model, model_config in models_to_run_config.enabled_models:
            model_analysis_state = ModelAnalysisState(
                model=model,
                model_config=model_config,
                platform_config=platform_config,
                features=analysis_state.features,
                coords=analysis_state.coords,
                dims=analysis_state.dims,
            )

            display.update_header(f"Running model {model_analysis_state.model_name}")
            # checks before fitting e.g. prior predictive checks
            model_checks(
                model_analysis_state=model_analysis_state,
                checker_config=checker_config,
                display=display,
            )
            if not display.is_live:
                display.start()

            fit(model_analysis_state=model_analysis_state, display=display)

            # checks after fitting e.g. posterior predictive checks
            model_checks(
                model_analysis_state=model_analysis_state,
                checker_config=checker_config,
                display=display,
            )

            analysis_state.add_model(model_analysis_state)

    if display.is_live:
        display.stop()

    # Update logs from display
    analysis_state.logs = display.get_all_logs()
    return analysis_state


def model_checks(
    model_analysis_state: ModelAnalysisState,
    checker_config: CheckerConfig,
    display: ModellingDisplay,
):
    """Run checks on the model."""
    if not display.is_live:
        display.start()

    display.update_header(f"Running checks for {model_analysis_state.model_name}")

    when = "after" if model_analysis_state.is_fitted else "before"

    display.update_logs(
        f"Enabled checks: {[registry_info(check).name for check in checker_config.get_checkers(when=when)]}"
    )
    with display.capture_logs():
        for checker in checker_config.get_checkers(when=when):
            model_analysis_state, outcome = checker(
                model_analysis_state, display=display
            )
            display.update_logs(
                f"Checker {registry_info(checker).name} for model {model_analysis_state.model_name} returned: {outcome}"
            )
            display.add_check(
                registry_info(checker).name,
                outcome,
            )

    if display.is_live:
        display.stop()

    return model_analysis_state


def communicate(
    analysis_state: AnalysisState,
    communicate_config: CommunicateConfig,
    display: ModellingDisplay,
):
    """Run communication on the model."""

    if not display.is_live:
        display.start()

    display.update_header("Running communication methods")

    display.update_logs(
        f"Enabled commincators: {[registry_info(check).name for check in communicate_config.enabled_communicators]}"
    )

    with display.capture_logs():
        for communicator in communicate_config.enabled_communicators:
            display.update_header(
                f"Running communicator for {registry_info(communicator).name}"
            )
            analysis_state, outcome = communicator(analysis_state, display=display)
            display.update_logs(
                f"Communicators {registry_info(communicator).name} returned: {outcome}"
            )

    if display.is_live:
        display.stop()

    # Update logs from display
    analysis_state.logs = display.get_all_logs()
    return analysis_state


def stamp():
    """
    Use DVC to version the data, model, and outputs.
    """
    pass


def check_attributes():
    pass
