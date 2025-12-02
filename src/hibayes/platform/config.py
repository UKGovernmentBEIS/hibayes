import os
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any, Dict, Literal, Set

import jax
import numpyro
import yaml

from ..utils import init_logger

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

ChainMethod = Literal["parallel", "sequential", "vectorized"]


@dataclass
class PlatformConfig:
    device_type: str = "cpu"  # Device type (cpu, gpu, tpu)
    num_devices: int | None = None  # Number of devices to use (None = auto-detect)
    gpu_memory_fraction: float = 0.9  # Fraction of GPU memory to use (0.1-1.0)
    chain_method: ChainMethod = "parallel"  # Method for running chains

    def __post_init__(self):
        # Auto-detect number of devices if not explicitly provided
        if self.num_devices is None:
            if self.device_type == "cpu":
                self.num_devices = os.cpu_count()
                numpyro.set_host_device_count(self.num_devices)
            elif self.device_type == "gpu":
                # Set CPU device count BEFORE checking for GPUs
                # This ensures if we fall back to CPU, the device count is already configured
                cpu_count = os.cpu_count()
                numpyro.set_host_device_count(cpu_count)

                try:
                    gpu_devices = jax.devices("gpu")
                    self.num_devices = len(gpu_devices) if gpu_devices else 0
                    if self.num_devices == 0:
                        raise RuntimeError("No GPU devices found")
                except Exception:
                    # Fallback to CPU if GPU detection fails
                    logger.warning("No GPU devices found, falling back to CPU.")
                    self.device_type = "cpu"
                    self.num_devices = cpu_count
            else:
                raise ValueError(f"Unsupported device type: {self.device_type}")

        # Validate gpu_memory_fraction
        if not 0.1 <= self.gpu_memory_fraction <= 1.0:
            raise ValueError("gpu_memory_fraction must be between 0.1 and 1.0")

    def merge_in_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Merge a dictionary into the existing configuration.
        """
        if not config_dict:
            return

        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(
                    f"Invalid configuration for {self.__class__.__name__} key: {key}"
                )

    @classmethod
    def from_yaml(cls, path: str) -> "PlatformConfig":
        """
        Load configuration from a yaml file.
        """
        with open(path, "r") as f:
            config: dict = yaml.safe_load(f)

        return cls.from_dict(config)

    @classmethod
    def from_dict(cls, config: dict | None) -> "PlatformConfig":
        """
        Load configuration from a dictionary.
        """
        if config is None:
            return cls()

        # Validate config keys
        allowed_keys = {"device_type", "num_devices", "gpu_memory_fraction", "chain_method"}
        _validate_config_keys(config, allowed_keys, "PlatformConfig")

        return cls(
            device_type=config.get("device_type", "cpu"),
            num_devices=config.get("num_devices", None),
            gpu_memory_fraction=config.get("gpu_memory_fraction", 0.9),
            chain_method=config.get("chain_method", "parallel"),
        )
