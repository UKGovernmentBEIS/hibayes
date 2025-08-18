import os
from dataclasses import dataclass
from typing import Any, Dict, Literal

import jax
import numpyro
import yaml

from ..utils import init_logger

logger = init_logger()

ChainMethod = Literal["parallel", "sequential", "vectorized"]


@dataclass
class PlatformConfig:
    device_type: str = "cpu"  # Device type (cpu, gpu, tpu)
    num_devices: int | None = None  # Number of devices to use (None = auto-detect)
    devices: list[str] | None = None  # List of device names (e.g., ["gpu:0", "gpu:1"])
    gpu_memory_fraction: float = 0.9  # Fraction of GPU memory to use (0.1-1.0)
    chain_method: ChainMethod = "parallel"  # Method for running chains

    def __post_init__(self):
        # Auto-detect number of devices if not explicitly provided
        if self.num_devices is None:
            if self.device_type == "cpu":
                self.num_devices = os.cpu_count()
            elif self.device_type == "gpu":
                # Set CPU device count BEFORE checking for GPUs
                # This ensures if we fall back to CPU, the device count is already configured
                cpu_count = os.cpu_count()
                numpyro.set_host_device_count(cpu_count)

                try:
                    gpu_devices = jax.devices("gpu")
                    self.num_devices = len(gpu_devices) if gpu_devices else 0
                    self.devices = gpu_devices if gpu_devices else []
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
        return cls(
            device_type=config.get("device_type", "cpu"),
            num_devices=config.get("num_devices", None),
            gpu_memory_fraction=config.get("gpu_memory_fraction", 0.9),
            devices=config.get("devices", None),
            chain_method=config.get("chain_method", "parallel"),
        )
