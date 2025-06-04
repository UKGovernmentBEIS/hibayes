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
                registry_get(RegistryInfo(type="processor", name=check))()
                for check in self.DEFAULT_PROCESS
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

        if custom_process_config := config.get("custom_data_process", None):
            enabled_processors.extend(cls._load_custom_process(custom_process_config))

        process_config = config.get("data_process", None)
        if isinstance(process_config, list):
            for process in process_config:
                if isinstance(process, dict):
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
        elif isinstance(process_config, dict):
            for process_name, kwargs in process_config.items():
                processor = registry_get(
                    RegistryInfo(type="processor", name=process_name)
                )
                enabled_processors.append(processor(**kwargs))

        return cls(enabled_processors=enabled_processors)

    @classmethod
    def _load_custom_process(cls, config: dict) -> List[DataProcessor]:
        """config from custom data processors.
        each mapping has a key path (required) and an optional list of process names.
        """
        entries = config if isinstance(config, list) else [config]
        loaded: List[DataProcessor] = []

        for entry in entries:
            _import_path(entry["path"])
            processes = entry.get("processors", None)
            if processes is None:
                continue
            processes = processes if isinstance(processes, list) else [processes]
            for process in processes:
                if isinstance(process, str):
                    name, kwargs = process, {}
                elif isinstance(process, dict) and len(process) == 1:
                    name, kwargs = next(iter(process.items()))
                else:
                    raise ValueError(
                        "Each process must be either a string or a dict with kwargs. e.g. process1: {kwargs1: x, kwargs2: y}"
                    )
                try:
                    processor = registry_get(RegistryInfo(type="processor", name=name))
                except KeyError:
                    logger.warning(
                        f"Processor {name} not found in registry. Skipping processor."
                    )
                    continue
                loaded.append(processor(**kwargs))
        return loaded
