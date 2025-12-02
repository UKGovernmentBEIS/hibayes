from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Any, ClassVar, Dict, List, Set

import yaml

from ..registry import RegistryInfo, _import_path, registry_get
from ..utils import init_logger
from ._communicate import Communicator

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

        # Validate config keys
        allowed_keys = {"path", "communicators"}
        _validate_config_keys(config, allowed_keys, "CommunicateConfig")

        enabled_communicators = []

        # so custom communicators are registered and can be treated as default
        if custom_path := config.get("path", None):
            if not isinstance(custom_path, list):
                custom_path = [custom_path]
            for path in custom_path:
                _import_path(path)

        communicate_section = config.get("communicators", None)
        if isinstance(communicate_section, list):
            for communicate in communicate_section:
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
        elif isinstance(communicate_section, dict):
            for communicate_name, communicate_config in communicate_section.items():
                communicator = registry_get(
                    RegistryInfo(type="communicator", name=communicate_name)
                )
                enabled_communicators.append(communicator(**communicate_config))

        return cls(enabled_communicators=enabled_communicators)
