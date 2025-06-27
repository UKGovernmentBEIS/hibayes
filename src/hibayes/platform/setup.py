import jax
import numpyro

from ..platform import PlatformConfig
from ..ui import ModellingDisplay


def configure_computation_platform(
    platform_config: PlatformConfig,
    display: ModellingDisplay,
) -> None:
    """
    Configure the computation platform (CPU/GPU/TPU) and parallelization settings.
    Should be called once during model initialization.
    """
    # Check for GPU/TPU support
    if platform_config.device_type in ["gpu", "tpu"]:
        raise NotImplementedError(
            f"{platform_config.device_type.upper()} support is not yet implemented. Please use CPU for now."
        )

    try:
        # For CPU, use process-based parallelism
        # We only want to set this once per program execution
        display.update_logs(
            f"Setting host device count to {platform_config.num_devices}"
        )
        numpyro.set_host_device_count(platform_config.num_devices)
        assert (
            platform_config.num_devices == jax.device_count()
        ), f"Mismatch in config device count {platform_config.num_devices} and actual jax device count {jax.device_count()} this might be due to set_device_count not being called before some jax code....."
    except Exception as e:
        display.update_logs(f"Failed to configure parallelization: {str(e)}")
        display.update_logs("Falling back to sequential execution")
