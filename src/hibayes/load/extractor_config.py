from dataclasses import dataclass, field
from typing import ClassVar, List

import yaml

from ..registry import RegistryInfo, _import_path, registry_get
from ..utils import init_logger
from ._extractor import Extractor

logger = init_logger()


@dataclass
class ExtractorConfig:
    """Configuration which determines the extractors to use."""

    DEFAULT_EXTRACTORS: ClassVar[List[str]] = ["base_extractor"]

    enabled_extractors: List[Extractor] = field(
        default_factory=list,
    )

    def __post_init__(self) -> None:
        """Set up default extractors if none specified."""
        if not self.enabled_extractors:
            self.enabled_extractors = [
                registry_get(RegistryInfo(type="extractor", name=extractor))()
                for extractor in self.DEFAULT_EXTRACTORS
            ]

    @classmethod
    def from_yaml(cls, path: str) -> "ExtractorConfig":
        """Load configuration from a yaml file."""
        with open(path, "r") as f:
            config: dict = yaml.safe_load(f)

        return cls.from_dict(config)

    @classmethod
    def from_dict(cls, config: dict) -> "ExtractorConfig":
        """Load configuration from a dictionary."""
        if config is None:
            config = {}

        enabled_extractors = []

        # Import custom extractors if specified
        if custom_path := config.get("path", None):
            if not isinstance(custom_path, list):
                custom_path = [custom_path]
            for path in custom_path:
                _import_path(path)

        extractors_section = config.get("extractors", None)
        if isinstance(extractors_section, list):
            for extractor in extractors_section:
                if isinstance(extractor, dict):
                    if len(extractor) != 1:
                        raise ValueError(
                            "Each extractor must be either a string or a dict with kwargs. e.g. extractor1: {kwargs1: x, kwargs2: y}"
                        )
                    extractor_name, extractor_config = next(iter(extractor.items()))
                    enabled_extractors.append(
                        registry_get(
                            RegistryInfo(type="extractor", name=extractor_name)
                        )(**extractor_config)
                    )
                else:
                    extractor_name = extractor
                    enabled_extractors.append(
                        registry_get(
                            RegistryInfo(type="extractor", name=extractor_name)
                        )()
                    )
        elif isinstance(extractors_section, dict):
            for extractor_name, extractor_config in extractors_section.items():
                extractor_fn = registry_get(
                    RegistryInfo(type="extractor", name=extractor_name)
                )
                enabled_extractors.append(extractor_fn(**extractor_config))

        return cls(enabled_extractors=enabled_extractors)
