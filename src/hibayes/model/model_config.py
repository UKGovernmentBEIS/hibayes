import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Literal, Optional, Tuple

import numpy as np
import yaml

from ..registry import RegistryInfo, _import_path, registry_get
from ..utils import init_logger
from ._model import Model
from .utils import cloglog_to_prob, link_to_key, logit_to_prob, probit_to_prob

logger = init_logger()

Method = Literal["NUTS", "HMC"]  # MCMC sampler type
ChainMethod = Literal["parallel", "sequential", "vectorised"]

LINK_FUNCTION_MAP: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "identity": lambda x: x,
    "logit": logit_to_prob,
    "sigmoid": logit_to_prob,
    "probit": probit_to_prob,
    "cloglog": cloglog_to_prob,
}


@dataclass(frozen=True, slots=True)
class FitConfig:
    method: Method = "NUTS"
    samples: int = 4000
    warmup: int = 500
    chains: int = 4
    seed: int = 0
    progress_bar: bool = True
    parallel: bool = True
    chain_method: ChainMethod = "parallel"
    target_accept: float = 0.8
    max_tree_depth: int = 10

    def merged(self, **updates: Any) -> "FitConfig":
        """Return a *new* FitConfig with updates applied."""
        return replace(self, **updates)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    fit: FitConfig = field(default_factory=FitConfig)

    main_effect_params: Optional[
        List[str]
    ] = None  # a list of the main effect parameters in the model which you would like to plot. If None then all parameters are treated as main.
    tag: Optional[str] = None  # a tag for the model config - e.g. version 1
    link_function: Callable[[np.ndarray], np.ndarray] = field(
        default=logit_to_prob  # link function for the model
    )
    extra_kwargs: Dict[str, Any] = field(
        default_factory=dict
    )  # to allow for high degrees of customisation in the creation of new @model functions the user can provide additional kwargs that will be passed to the model builder.

    def get_plot_params(self) -> List[str] | None:
        """Get a list of parameters to plot based on the configuration."""
        return self.main_effect_params

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        """Load configuration from a yaml file."""
        with open(path, "r") as f:
            config: dict = yaml.safe_load(f)

        return cls.from_dict(config)

    @classmethod
    def from_dict(cls, config: dict) -> "ModelConfig":
        """Load configuration from a dictionary."""
        if config is None:
            config = {}

        config = config.copy()

        known_fields = set(cls.__dataclass_fields__.keys())

        # Extract known fields
        structured_args = {}
        for field_name in known_fields:
            if field_name in config:
                structured_args[field_name] = config.pop(field_name)

        # Everything left goes to extra_kwargs
        extra_kwargs = structured_args.get("extra_kwargs", {})
        extra_kwargs.update(config)

        # Process fit config
        fit_config = FitConfig(**structured_args.get("fit", {}))

        # Process link function
        link_arg = structured_args.get("link_function", "logit")

        if isinstance(link_arg, str):
            if link_arg not in LINK_FUNCTION_MAP:
                raise ValueError(
                    f"Link function {link_arg} not recognised. Must be one of {list(LINK_FUNCTION_MAP.keys())}."
                )
            link_function = LINK_FUNCTION_MAP[link_arg]
        elif callable(link_arg):
            if link_arg not in LINK_FUNCTION_MAP.values():
                raise ValueError(
                    "Link function must be one of the predefined functions or a custom callable."
                )
            link_function = link_arg
        else:
            raise ValueError(
                "Link function must be a string or a callable. "
                f"Got {type(link_arg)} instead."
            )

        return cls(
            fit=fit_config,
            main_effect_params=structured_args.get("main_effect_params", None),
            tag=structured_args.get("tag", None),
            link_function=link_function,
            extra_kwargs=extra_kwargs,
        )

    def save(self, path: Path) -> None:
        """Save the model configuration as a json file."""
        state = asdict(self)
        state["link_function"] = link_to_key(state["link_function"], LINK_FUNCTION_MAP)
        with path.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)


@dataclass
class ModelsToRunConfig:
    """Configuration which determines the models to run and loads in args"""

    DEFAULT_MODELS: ClassVar[List[str]] = ["two_level_group_binomial"]

    enabled_models: List[Tuple[Model, ModelConfig]] = field(
        default_factory=list,
    )

    def __post_init__(self) -> None:
        """Set up default models with default args if none specified."""
        if not self.enabled_models:
            self.enabled_models = [
                (registry_get(RegistryInfo(type="model", name=model))(), ModelConfig())
                for model in self.DEFAULT_MODELS
            ]

    @classmethod
    def from_yaml(cls, path: str) -> "ModelsToRunConfig":
        """Load configuration from a yaml file."""
        with open(path, "r") as f:
            config: dict = yaml.safe_load(f)

        return cls.from_dict(config)

    @classmethod
    def from_dict(cls, config: dict) -> "ModelsToRunConfig":
        """Load configuration from a dictionary."""
        if config is None:
            config = {}

        enabled_models = []

        # so custom models are registered and can be treated as default
        if custom_path := config.get("path", None):
            if not isinstance(custom_path, list):
                custom_path = [custom_path]
            for path in custom_path:
                _import_path(path)

        model_section = config.get("models", None)
        if isinstance(model_section, list):
            for model in model_section:
                if isinstance(model, dict):
                    model_name = model["name"]

                    model_config = ModelConfig.from_dict(model["config"])
                    enabled_models.append(
                        (
                            registry_get(RegistryInfo(type="model", name=model_name))(
                                **model_config.extra_kwargs
                            ),
                            model_config,
                        )
                    )
                else:
                    model_name = model
                    enabled_models.append(
                        (
                            registry_get(RegistryInfo(type="model", name=model_name))(),
                            ModelConfig(),
                        )
                    )
        elif isinstance(model_section, dict):
            for model_name, model_config in model_section.items():
                model_config = ModelConfig.from_dict(model_config)
                enabled_models.append(
                    (
                        registry_get(RegistryInfo(type="model", name=model_name))(
                            **model_config.extra_kwargs
                        ),
                        model_config,
                    )
                )

        return cls(enabled_models=enabled_models)
