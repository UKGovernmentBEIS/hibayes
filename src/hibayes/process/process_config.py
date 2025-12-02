from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Any, ClassVar, Dict, List, Set

import yaml

from ..registry import RegistryInfo, _import_path, registry_get
from ..utils import init_logger
from ._process import DataProcessor

logger = init_logger()


def _validate_config_keys(
    config: Dict[str, Any], allowed_keys: Set[str], config_name: str
) -> None:
    """Validate that config only contains allowed keys, with suggestions for typos."""
    unknown_keys = set(config.keys()) - allowed_keys
    if unknown_keys:
        suggestions = []
        for key in unknown_keys:
            matches = get_close_matches(key, allowed_keys, n=1, cutoff=0.6)
            if matches:
                suggestions.append(f"'{key}' (did you mean '{matches[0]}'?)")
            else:
                suggestions.append(f"'{key}'")

        raise ValueError(
            f"Unknown keys in {config_name}: {', '.join(suggestions)}. "
            f"Allowed keys are: {sorted(allowed_keys)}"
        )


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

        # Validate config keys
        allowed_keys = {"path", "processors"}
        _validate_config_keys(config, allowed_keys, "ProcessConfig")

        enabled_processors = []

        # so custom processors are registered and can be treated as default
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
