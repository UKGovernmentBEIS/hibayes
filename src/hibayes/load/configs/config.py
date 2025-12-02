import datetime
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List

import yaml

from hibayes.utils import init_logger

from ...registry import RegistryInfo, _import_path, registry_get
from .._extractor import Extractor

logger = init_logger()


@dataclass
class DataLoaderConfig:
    """Configuration for the log processor."""

    # Define a list of default extractors that are used if nothing specified
    DEFAULT_EXTRACTORS: ClassVar[List[str]] = ["base_extractor"]

    # Configuration properties
    enabled_extractors: List[Extractor] = field(default_factory=list)
    files_to_process: List[str] = field(default_factory=list)
    extracted_data: List[str] = field(default_factory=list)
    cache_path: str | None = None
    output_dir: str | None = None
    max_workers: int = 10
    batch_size: int = 1000
    cutoff: datetime.datetime | None = None

    def __post_init__(self) -> None:
        """Set up default extractors if none specified and validate config."""
        if not self.enabled_extractors:
            self.enabled_extractors = [
                registry_get(RegistryInfo(type="extractor", name=extractor))()
                for extractor in self.DEFAULT_EXTRACTORS
            ]

        # Validate mutual exclusivity
        if self.files_to_process and self.extracted_data:
            raise ValueError(
                "Cannot specify both 'files_to_process' and 'extracted_data'. "
                "Use 'files_to_process' for inspect eval logs, or 'extracted_data' "
                "for pre-extracted CSV/parquet files."
            )

        # Validate at least one data source is provided
        if not self.files_to_process and not self.extracted_data and not self.cache_path:
            logger.warning(
                "No data source specified. Provide 'files_to_process' for eval logs, "
                "'extracted_data' for pre-extracted files, or 'cache_path' for cached data."
            )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DataLoaderConfig":
        """
        Create a DataLoaderConfig from a YAML file

        The YAML should have this structure:
        ```yaml
        extractors:
          path: my_custom_extractors.py
          enabled:
            - base
            - domain
            - tools
            - custom_extractor
        ```
        """
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any] | None) -> "DataLoaderConfig":
        if config_dict is None:
            config_dict = {}
        # get paths for logs and processed data
        config_paths = config_dict.get("paths", {})
        files_to_process = config_paths.get("files_to_process", [])
        extracted_data = config_paths.get("extracted_data", [])
        cache_path = config_paths.get("cache_path")
        output_dir = config_paths.get("output_dir")

        enabled_extractors = []

        # Import custom extractors if specified
        if custom_path := config_dict.get("extractors", {}).get("path", None):
            if not isinstance(custom_path, list):
                custom_path = [custom_path]
            for path in custom_path:
                _import_path(path)

        extractors_section = config_dict.get("extractors", {}).get("enabled", None)
        if isinstance(extractors_section, list):
            for extractor in extractors_section:
                if isinstance(extractor, dict):
                    if len(extractor) != 1:
                        raise ValueError(
                            "Each extractor must be either a string or a dict with kwargs"
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

        return cls(
            enabled_extractors=enabled_extractors,
            files_to_process=files_to_process,
            extracted_data=extracted_data,
            cache_path=cache_path,
            output_dir=output_dir,
        )
