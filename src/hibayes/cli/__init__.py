import os


def setup_platform(config_path=None, *, device_type=None):
    """Configure the JAX/XLA environment. **Must be called before importing
    other hibayes modules.**

    With CUDA-enabled jaxlib (installed via ``numpyro[cuda]``), importing JAX
    triggers XLA backend initialisation which locks the CPU device count to 1
    and sets GPU as the default backend.  After that,
    ``numpyro.set_host_device_count()`` has no effect and numpyro's parallel
    chain method sees only 1 device — running chains sequentially.

    The CLI entry points call this automatically.  For **programmatic** use::

        from hibayes.cli import setup_platform
        setup_platform("path/to/config.yaml")   # or: setup_platform(device_type="gpu")

        # Now safe to import the rest of hibayes
        from hibayes.analysis import AnalysisConfig, ...

    Args:
        config_path: Path to a hibayes YAML config file.  The
            ``platform.device_type`` field is read to decide CPU vs GPU.
        device_type: Explicit device type (``"cpu"`` or ``"gpu"``).
            Overrides ``config_path`` if both are given.
    """
    cpu_count = os.cpu_count() or 1

    # Determine device type
    resolved_device_type = "cpu"
    if device_type is not None:
        resolved_device_type = device_type
    elif config_path is not None:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        if config and isinstance(config.get("platform"), dict):
            resolved_device_type = config["platform"].get("device_type", "cpu")

    # CPU mode: restrict JAX to CPU-only so it doesn't grab GPU memory
    # and numpyro parallel chains can see all CPU devices.
    # GPU mode: leave JAX_PLATFORMS unset so the GPU backend is available.
    if resolved_device_type != "gpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    # Always ensure multiple CPU host devices are available (harmless for GPU).
    xla_flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_force_host_platform_device_count" not in xla_flags:
        os.environ["XLA_FLAGS"] = (
            f"{xla_flags} --xla_force_host_platform_device_count={cpu_count}"
        )
