from dataclasses import dataclass, field
from typing import ClassVar, List

import yaml

from ..registry import RegistryInfo, _import_path, registry_get
from ..utils import init_logger
from ._communicate import Communicator

logger = init_logger()


@dataclass
class CommunicateConfig:
    """Configuration which determins what to plot/tabulate."""

    DEFAULT_COMMUNICATE: ClassVar[List[str]] = [
        "trace_plot",
        "forest_plot",
        "pair_plot",
        "model_comparison_plot",
        "summary_table",
    ]

    enabled_communicators: List[Communicator] = field(
        default_factory=list,
    )

    def __post_init__(self) -> None:
        """Set up default checks if none specified."""
        if not self.enabled_communicators:
            self.enabled_communicators = [
                registry_get(RegistryInfo(type="communicator", name=check))()
                for check in self.DEFAULT_COMMUNICATE
            ]

    @classmethod
    def from_yaml(cls, path: str) -> "CommunicateConfig":
        """Load configuration from a yaml file."""
        with open(path, "r") as f:
            config: dict = yaml.safe_load(f)

        return cls.from_dict(config)

    @classmethod
    def from_dict(cls, config: dict) -> "CommunicateConfig":
        """Load configuration from a dictionary."""
        if config is None:
            config = {}

        enabled_communicators = []

        if custom_communicate_config := config.get("custom_communicate", None):
            enabled_communicators.extend(
                cls._load_custom_communicate(custom_communicate_config)
            )

        communicate_config = config.get("communicate", None)
        if isinstance(communicate_config, list):
            for communicate in communicate_config:
                if isinstance(communicate, dict):
                    communicate_name, communicate_config = next(
                        iter(communicate.items())
                    )
                    enabled_communicators.append(
                        registry_get(
                            RegistryInfo(type="communicator", name=communicate_name)
                        )(**communicate_config)
                    )
                else:
                    communicate_name = communicate
                    enabled_communicators.append(
                        registry_get(
                            RegistryInfo(type="communicator", name=communicate_name)
                        )()
                    )
        elif isinstance(communicate_config, dict):
            for communicate_name, kwargs in communicate_config.items():
                communicator = registry_get(
                    RegistryInfo(type="communicator", name=communicate_name)
                )
                enabled_communicators.append(communicator(**kwargs))

        return cls(enabled_communicators=enabled_communicators)

    @classmethod
    def _load_custom_communicate(cls, config: dict) -> List[Communicator]:
        """config from custom communicate.
        each mapping has a key path (required) and an optional list of communicate names.
        """
        entries = config if isinstance(config, list) else [config]
        loaded: List[Communicator] = []

        for entry in entries:
            _import_path(entry["path"])
            communicators = entry.get("communicate", None)
            if communicators is None:
                continue
            communicators = (
                communicators if isinstance(communicators, list) else [communicators]
            )
            for communicator in communicators:
                if isinstance(communicator, str):
                    name, kwargs = communicator, {}
                elif isinstance(communicator, dict) and len(communicator) == 1:
                    name, kwargs = next(iter(communicator.items()))
                else:
                    raise ValueError(
                        "Each communicator must be either a string or a dict with kwargs. e.g. communicator1: {kwargs1: x, kwargs2: y}"
                    )
                try:
                    communicator = registry_get(
                        RegistryInfo(type="communicator", name=name)
                    )
                except KeyError:
                    logger.warning(
                        f"Communicator {name} not found in registry. Skipping communicator."
                    )
                    continue
                loaded.append(communicator(**kwargs))
        return loaded
