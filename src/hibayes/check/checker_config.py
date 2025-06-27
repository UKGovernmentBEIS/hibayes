from dataclasses import dataclass, field
from typing import ClassVar, List

import yaml

from ..registry import RegistryInfo, _import_path, registry_get, registry_info
from ..utils import init_logger
from ._check import Checker

logger = init_logger()


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
