from dataclasses import dataclass, field
from typing import ClassVar, List

import yaml

from ..registry import RegistryInfo, _import_path, registry_get
from ..utils import init_logger
from ._process import DataProcessor

logger = init_logger()


@dataclass
class ProcessConfig:
    """Configuration which determines the data processing steps."""

    DEFAULT_PROCESS: ClassVar[List[str]] = [
        "extract_observed_feature",
        "extract_features",
    ]

    enabled_processors: List[DataProcessor] = field(
        default_factory=list,
    )

    def __post_init__(self) -> None:
        """Set up default checks if none specified."""
        if not self.enabled_processors:
            self.enabled_processors = [
                registry_get(RegistryInfo(type="processor", name=process))()
                for process in self.DEFAULT_PROCESS
            ]

    @classmethod
    def from_yaml(cls, path: str) -> "ProcessConfig":
        """Load configuration from a yaml file."""
        with open(path, "r") as f:
            config: dict = yaml.safe_load(f)

        return cls.from_dict(config)

    @classmethod
    def from_dict(cls, config: dict) -> "ProcessConfig":
        """Load configuration from a dictionary."""
        if config is None:
            config = {}

        enabled_processors = []

        # so custom processors are registered and can be treated as defailt
        if custom_path := config.get("path", None):
            if not isinstance(custom_path, list):
                custom_path = [custom_path]
            for path in custom_path:
                _import_path(path)

        process_section = config.get("processors", None)
        if isinstance(process_section, list):
            for process in process_section:
                if isinstance(process, dict):
                    if len(process) != 1:
                        raise ValueError(
                            "Each process must be either a string or a dict with kwargs. e.g. process1: {kwargs1: x, kwargs2: y}"
                        )
                    process_name, process_config = next(iter(process.items()))
                    enabled_processors.append(
                        registry_get(RegistryInfo(type="processor", name=process_name))(
                            **process_config
                        )
                    )
                else:
                    process_name = process
                    enabled_processors.append(
                        registry_get(
                            RegistryInfo(type="processor", name=process_name)
                        )()
                    )
        elif isinstance(process_section, dict):
            for process_name, process_config in process_section.items():
                processor = registry_get(
                    RegistryInfo(type="processor", name=process_name)
                )
                enabled_processors.append(processor(**process_config))

        return cls(enabled_processors=enabled_processors)
