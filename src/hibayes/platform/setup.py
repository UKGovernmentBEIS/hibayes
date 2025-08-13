import os
from typing import Optional

import jax
import numpyro

from ..platform import PlatformConfig
from ..ui import ModellingDisplay


def _configure_gpu_memory(
    platform_config: PlatformConfig, display: ModellingDisplay
) -> None:
    """
    Configure GPU memory settings for JAX.
    """
    try:
        # Set memory preallocation for better memory management
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(
            platform_config.gpu_memory_fraction
        )

        display.update_logs(
            f"Configured GPU to use {platform_config.gpu_memory_fraction * 100:.1f}% of available memory"
        )
    except Exception as e:
        display.update_logs(f"Warning: Failed to configure GPU memory: {str(e)}")


def _get_device_info(device_type: str) -> str:
    """
    Get information about available devices.
    """
    try:
        if device_type == "gpu":
            devices = jax.devices("gpu")
            if devices:
                device_info = [f"{d.device_kind} ({d.id})" for d in devices]
                return f"GPUs: {', '.join(device_info)}"
            return "No GPUs found"
        elif device_type == "tpu":
            devices = jax.devices("tpu")
            if devices:
                device_info = [f"{d.device_kind} ({d.id})" for d in devices]
                return f"TPUs: {', '.join(device_info)}"
            return "No TPUs found"
        else:
            return f"CPUs: {jax.device_count()} cores"
    except Exception as e:
        return f"Error getting device info: {str(e)}"


def configure_computation_platform(
    platform_config: PlatformConfig,
    display: ModellingDisplay,
) -> None:
    """
    Configure the computation platform (CPU/GPU/TPU) and parallelization settings.
    Should be called once during model initialization.
    """
    display.update_logs(f"Configuring {platform_config.device_type.upper()} platform")
    display.update_logs(_get_device_info(platform_config.device_type))

    try:
        if platform_config.device_type == "gpu":
            # Configure GPU memory before any JAX operations
            _configure_gpu_memory(platform_config, display)

            # Get available GPU devices
            gpu_devices = jax.devices("gpu")
            if not gpu_devices:
                raise RuntimeError("No GPU devices available")

            display.update_logs(f"Using {len(gpu_devices)} GPU(s) for computation")

            # For multi-GPU setups, JAX automatically distributes computation
            # across all available GPUs. For single GPU, we can set it as default
            if len(gpu_devices) == 1:
                jax.config.update("jax_default_device", gpu_devices[0])
                display.update_logs("Single GPU setup - set as default device")
            else:
                display.update_logs(
                    "Multi-GPU setup - JAX will automatically distribute workload across GPUs"
                )

        else:
            # For CPU, use process-based parallelism
            display.update_logs(
                f"Setting host device count to {platform_config.num_devices}"
            )
            numpyro.set_host_device_count(platform_config.num_devices)

        # Verify device configuration
        current_devices = jax.devices()
        display.update_logs(
            f"Active devices: {len(current_devices)} {platform_config.device_type.upper()}(s)"
        )

        # Validate expected vs actual device count for CPU
        if platform_config.device_type == "cpu":
            assert platform_config.num_devices == jax.device_count(), (
                f"Mismatch in config device count {platform_config.num_devices} and actual jax device count {jax.device_count()}"
            )

    except Exception as e:
        display.update_logs(
            f"Failed to configure {platform_config.device_type.upper()}: {str(e)}"
        )
        display.update_logs("Falling back to CPU execution")

        # Fallback to CPU
        try:
            numpyro.set_host_device_count(os.cpu_count() or 1)
            display.update_logs("Successfully configured CPU fallback")
        except Exception as fallback_e:
            display.update_logs(f"Fallback also failed: {str(fallback_e)}")
            raise


def get_device_memory_info() -> Optional[dict]:
    """
    Get memory information for current devices.
    Returns device information or None if unavailable.
    """
    try:
        devices = jax.devices()
        if not devices:
            return None

        return {
            "device_count": len(devices),
            "platform": devices[0].platform,
            "device_kind": devices[0].device_kind
            if hasattr(devices[0], "device_kind")
            else "unknown",
            "devices": [
                {"id": d.id, "kind": getattr(d, "device_kind", "unknown")}
                for d in devices
            ],
        }
    except Exception:
        return None
