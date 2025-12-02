from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Any, ClassVar, Dict, List, Set

import yaml

from ..registry import RegistryInfo, _import_path, registry_get, registry_info
from ..utils import init_logger
from ._check import Checker

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
class CheckerConfig:
    """Configuration for which checks to run on the model."""

    DEFAULT_CHECKS: ClassVar[List[str]] = [
        "prior_predictive_plot",
        "r_hat",
        "divergences",
        "ess_bulk",
        "ess_tail",
        "loo",
        "bfmi",
        "posterior_predictive_plot",
        "waic",
    ]

    enabled_checks: List[Checker] = field(
        default_factory=list,
    )

    def __post_init__(self) -> None:
        """Set up default checks if none specified."""
        if not self.enabled_checks:
            self.enabled_checks = [
                registry_get(RegistryInfo(type="checker", name=check))()
                for check in self.DEFAULT_CHECKS
            ]  # build default checks with default args.

    @classmethod
    def from_yaml(cls, path: str) -> "CheckerConfig":
        """Load configuration from a yaml file."""
        with open(path, "r") as f:
            config: dict = yaml.safe_load(f)

        return cls.from_dict(config)

    @classmethod
    def from_dict(cls, config: dict) -> "CheckerConfig":
        """Load configuration from a dictionary."""
        if config is None:
            config = {}

        # Validate config keys
        allowed_keys = {"path", "checkers"}
        _validate_config_keys(config, allowed_keys, "CheckerConfig")

        enabled_checks = []

        # so custom checks are registered and can be treated as default
        if custom_path := config.get("path", None):
            if not isinstance(custom_path, list):
                custom_path = [custom_path]
            for path in custom_path:
                _import_path(path)

        checks_config = config.get("checkers", None)
        if isinstance(checks_config, list):
            # ["check1", "check2"] format
            for check in checks_config:
                if isinstance(check, dict):
                    check_name, check_config = next(iter(check.items()))
                    enabled_checks.append(
                        registry_get(RegistryInfo(type="checker", name=check_name))(
                            **check_config
                        )
                    )
                else:
                    # use default args
                    enabled_checks.append(
                        registry_get(RegistryInfo(type="checker", name=check))()
                    )
        elif isinstance(checks_config, dict):
            # {"check1": {config}, "check2": {config}} format
            for check_name, check_config in checks_config.items():
                enabled_checks.append(
                    registry_get(RegistryInfo(type="checker", name=check_name))(
                        **check_config
                    )
                )

        return cls(enabled_checks=enabled_checks)

    def get_checkers(self, when: str = "after") -> List[Checker]:
        """Get the checkers to run."""
        if when == "after":
            return [
                checker
                for checker in self.enabled_checks
                if registry_info(checker).metadata.get("when") == "after"
            ]
        if when == "before":
            return [
                checker
                for checker in self.enabled_checks
                if registry_info(checker).metadata.get("when") == "before"
            ]
        raise ValueError("Only 'after' and 'before' is supported for now.")
