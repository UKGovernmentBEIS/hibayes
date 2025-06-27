from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
)

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.distributions.distribution import DistributionLike

from hibayes.utils import init_logger

from .utils import _link_to_key, cloglog_to_prob, logit_to_prob, probit_to_prob

logger = init_logger()

Features = Dict[str, float | int | jnp.ndarray | np.ndarray]
Coords = Dict[
    str, List[Any]
]  # arviz information on how to map indexes to names. Here we store both the coords and the dims values
Dims = Dict[str, List[str]]
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
        """Return a *new* FitConfig with `updates` applied."""
        return replace(self, **updates)


@dataclass(frozen=True, slots=True)
class PriorConfig:
    distribution: DistributionLike  # The prior distribution
    distribution_args: Dict[
        str, Any
    ] | None = None  # Arguments for the prior distribution

    DISTRIBUTION_MAP: ClassVar[Dict[str, Type[DistributionLike]]] = {
        "normal": dist.Normal,
        "uniform": dist.Uniform,
        "exponential": dist.Exponential,
        "gamma": dist.Gamma,
        "bernoulli": dist.Bernoulli,
        "half_normal": dist.HalfNormal,
        "beta": dist.Beta,
    }

    def get_distribution(self) -> DistributionLike:
        """
        Get the prior distribution with the specified arguments.
        """
        if self.distribution_args is None:
            return self.distribution()
        return self.distribution(**self.distribution_args)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PriorConfig":
        """
        Create a PriorConfig from a dictionary.
        """
        distribution_name = config_dict.pop("distribution", None)
        distribution_args = config_dict.pop("distribution_args", None)

        if distribution_name not in cls.DISTRIBUTION_MAP:
            raise ValueError(
                f"Invalid distribution name '{distribution_name}'. Valid options are: {list(cls.DISTRIBUTION_MAP.keys())}"
            )
        distribution_class = cls.DISTRIBUTION_MAP[distribution_name]
        return cls(
            distribution=distribution_class,
            distribution_args=distribution_args,
        )


@dataclass(frozen=True, slots=True)
class RandomEffectConfig:
    """Describe one random effect term.
    So far this makes sense to keep separate from ParameterConfig due to
    requring information from data such as num models, num tasks, etc."""

    groups: List[
        str, ...
    ]  # grouping columns defining the levels - this is used in _prepare_data. len(groups) == 1 for random effects, len(groups) > 1 for interaction
    name: str
    prior: PriorConfig
    main_effect: bool = False  # Whether this re should feature in headline analysis

    def merged(self, **updates: Any) -> "RandomEffectConfig":
        if "name" in updates and updates["name"] != self.name:
            raise ValueError("Cannot change name for an existing RE config.")
        if "prior" in updates:
            updates["prior"] = self.prior.from_dict(updates.pop("prior"))
        return replace(self, **updates)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RandomEffectConfig":
        """
        Create a RandomEffectConfig from a dictionary.
        """
        groups = config_dict.pop("groups", None)
        if not isinstance(groups, list):
            raise ValueError("You need to specify groups as a list.")
        name = config_dict.pop("name", None)
        if name is None:
            raise ValueError("Missing 'name' key in random effect configuration.")
        prior = config_dict.pop("prior", None)
        if prior is None:
            raise ValueError("Missing 'prior' key in random effect configuration.")
        prior = PriorConfig.from_dict(prior)

        return cls(groups=sorted(groups), name=name, prior=prior)


@dataclass(frozen=True, slots=True)
class ParameterConfig:
    name: str  # Name of the parameter in the model
    prior: PriorConfig  # Prior distribution for the parameter
    main_effect: bool = (
        False  # Whether this parameter should feature in headline analysis
    )

    def merge_in_dict(self, config_dict: Dict[str, Any]) -> "ParameterConfig":
        new_param_name = config_dict.get("name")
        if new_param_name != self.name:
            raise ValueError(
                f"Parameter name mismatch: '{new_param_name}' does not match '{self.name}'. You cannot update parameter names as they are called in the model."
            )
        if new_prior := config_dict.get("prior", None):
            config_dict["prior"] = self.prior.from_dict(new_prior)
        return replace(self, **config_dict)


@dataclass
class ModelConfig:
    column_map: Optional[
        Dict[str, str]
    ] = None  # the key for mapping from data columns to any required_column names
    configurable_parameters: List[ParameterConfig] = field(
        default_factory=list
    )  # parameters user can adjust
    random_effects: List[RandomEffectConfig] = field(
        default_factory=list
    )  ## random effects user can adjust, add to and remove from.

    fit: FitConfig = field(default_factory=FitConfig)

    main_effect_params: Optional[List[str]] = field(
        default_factory=list
    )  # a list of the main effect parameters in the model which you would like to plot. If None then default to parameter config
    tag: Optional[str] = None  # a tag for the model config - e.g. version 1
    link_function: Union[
        str, Callable[[jnp.ndarray], jnp.ndarray]
    ] = "sigmoid"  # link function for the model, if str then use the mapping in LINK_FUNCTION_MAP

    def __post_init__(self):
        # Ensure no duplicate parameter names
        if self.configurable_parameters is not None:
            param_names = {param.name for param in self.configurable_parameters}
            if len(param_names) != len(self.configurable_parameters):
                raise ValueError(
                    "Duplicate parameter names found in model configuration"
                )
            # make sure intercept  parameters don't collide
            # very edge case but the error is not nice otherwise
            re_param_names = []
            if self.random_effects is not None:
                for re_cfg in self.random_effects:
                    re_param_names.append(re_cfg.name)
                overlap = set(param_names).intersection(re_param_names)
                if overlap:
                    raise ValueError(
                        f"Parameter names reused by random_effects: {', '.join(overlap)}"
                    )
        if isinstance(self.link_function, str):
            if self.link_function not in LINK_FUNCTION_MAP:
                raise ValueError(
                    f"Invalid link function '{self.link_function}'. Valid options are: {list(LINK_FUNCTION_MAP.keys())}"
                )
            self.link_function = LINK_FUNCTION_MAP[self.link_function]

        self.main_effect_params = (
            [
                p.name
                for p in self.configurable_parameters
                if p.main_effect and p.name not in self.main_effect_params
            ]
            + [
                re.name
                for re in self.random_effects
                if re.main_effect and re.name not in self.main_effect_params
            ]
            + self.main_effect_params
        )

    def parameter_to_numpyro(self, param: str) -> Dict[str, Any]:
        """Get the parameter configuration for a specific parameter."""
        if self.configurable_parameters is None and self.random_effects is None:
            raise ValueError("No parameters defined in model configuration")

        parameter = self.get_param(param)
        if parameter is not None:
            numpyro_args = {"name": parameter.name}
            if parameter.prior is not None:
                numpyro_args["fn"] = parameter.prior.get_distribution()
            return numpyro_args
        raise ValueError(f"Parameter '{param}' not found in model configuration")

    def get_plot_params(self) -> List[str] | None:
        """Get a list of parameters to plot based on the configuration."""
        return self.main_effect_params

    def get_params(self) -> List[str] | None:
        """Get a list of all configurable parameters in the model config."""
        combined_params = (self.configurable_parameters or []) + (
            self.random_effects or []
        )
        if not combined_params:
            raise ValueError("No parameters defined in model configuration")
        return [param.name for param in combined_params]

    def get_param(self, param: str) -> ParameterConfig | RandomEffectConfig | None:
        """Get a specific parameter configuration from the configurable parameters."""
        combined_params = (self.configurable_parameters or []) + (
            self.random_effects or []
        )
        if not combined_params:
            raise ValueError("No parameters defined in model configuration")
        for parameter in combined_params:
            if parameter.name == param:
                return parameter
        return None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["link_function"] = _link_to_key(state["link_function"], LINK_FUNCTION_MAP)
        return state

    def __setstate__(self, state):
        lf = state["link_function"]
        state["link_function"] = LINK_FUNCTION_MAP[lf] if isinstance(lf, str) else lf
        object.__setattr__(self, "__dict__", state)

    @staticmethod
    def _apply_delete_flag(item: Dict[str, Any]) -> bool:
        """Return True if this dict carries _delete_: true."""
        return bool(item.pop("_delete_", False))

    def merge_in_dict(self, config_dict: Dict[str, Any]) -> "ModelConfig":
        """
        Return a config with a merge of default and user specified values.

        the user can:
        fit: modify
        column_map: override
        main_effect_params: override
        configurable_parameters: modify - to add or delete the user should create a new class
        random_effects: delete, modify and add
        link_function: override
        """
        fit_updates = config_dict.get("fit", {})
        new_fit = self.fit.merged(**fit_updates)

        new_column_map = config_dict.get("column_map", self.column_map)

        new_main_effect_params = config_dict.get(
            "main_effect_params", self.main_effect_params
        )
        new_params = self.configurable_parameters

        new_tag = config_dict.get("tag", self.tag)

        new_link = config_dict.get("link_function", self.link_function)

        if "configurable_parameters" in config_dict:
            param_updates = config_dict["configurable_parameters"]
            if isinstance(param_updates, dict):  # only update a single parameter
                param_updates_iter = [param_updates]
            elif isinstance(param_updates, list):  # assuming a list of Mapping
                if not all(isinstance(param, dict) for param in param_updates):
                    raise ValueError(
                        "To update parameters you need to specify either a list of dicts or a single dict where for each parameter at least the parameter name and what you would like to change is detailed."
                    )
                param_updates_iter = param_updates
            else:
                raise ValueError(
                    "To update parameters you need to specify either a list of dicts or a single dict where for each parameter at least the parameter name and what you would like to change is detailed."
                )
            param_map = {param.name: param for param in self.configurable_parameters}
            for param in param_updates_iter:
                name = param.get("name", None)
                if name not in param_map:
                    raise ValueError(
                        f"Parameter name '{param['name']}' not found in model configuration"
                    )
                param_map[name] = param_map[name].merge_in_dict(param)
            new_params = list(param_map.values())

        new_res: list[RandomEffectConfig] = self.random_effects
        if "random_effects" in config_dict:
            updates = config_dict["random_effects"]
            updates = updates if isinstance(updates, list) else [updates]

            re_map = {r.name: r for r in self.random_effects}
            for item in updates:
                name = item.get("name")
                if name is None:
                    raise ValueError("Each random-effect patch needs a name.")

                if self._apply_delete_flag(item):
                    re_map.pop(name, None)  # silently ignore if absent
                    continue

                if name in re_map:  # modify
                    re_map[name] = re_map[name].merged(**item)
                else:  # add new re
                    re_map[name] = RandomEffectConfig.from_dict(item)
            new_res = list(re_map.values())

        return replace(
            self,
            column_map=new_column_map,
            fit=new_fit,
            main_effect_params=new_main_effect_params,
            configurable_parameters=new_params,
            random_effects=new_res,
            tag=new_tag,
            link_function=new_link,
        )


class BaseModel(ABC):
    """Base class for methods for statistical modelling and analysis of inspect data logs."""

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], ModelConfig]] = None,
    ) -> None:
        """
        Return the numpyro model callable

        Parameters
        ----------
        config : Optional[Union[Dict[str, Any], ModelConfig]]
            Configuration dictionary or ModelConfig instance to override defaults

        """
        # Merge default config with provided config
        default_config = self.get_default_config()
        if isinstance(config, dict):
            self.config = default_config.merge_in_dict(config)
        elif isinstance(config, ModelConfig):
            self.config = config
        else:
            self.config = default_config

    @classmethod
    @abstractmethod
    def get_default_config(cls) -> ModelConfig:
        return ModelConfig()

    @abstractmethod
    def build_model(self) -> Callable[..., Any]:
        """
        Build the numpyro model

        Returns:
            Callable: A NumPyro model function that must accept at least:
                - obs (optional): Observed data, defaults to None
        """

        def model(obs=None):
            # Numpyro model implementation that uses obs - observational data
            # Why kwarg with none? this is for prior predictive checks!
            pass

        return model

    def prepare_data(self, data: pd.DataFrame) -> Tuple[Features, Coords, Dims]:
        """
        Prepare data for modeling. Wraps the abstract method _prepare_data so
        to enforce that observed variable in features.
        """
        data_copy = (
            data.rename(columns=self.config.column_map)
            if self.config.column_map
            else data.copy()
        )
        features, coords, dims = self._prepare_data(data_copy)

        if "obs" not in features:
            raise ValueError(
                "The model must have an observation feature named 'obs' in the configuration."
            )
        return features, coords, dims

    @abstractmethod
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[Features, Coords, Dims]:
        """
        Abstract method for preparing data. This should be implemented in subclasses.
        """
        pass

    @classmethod
    def name(cls) -> str:
        """
        Get the name of the model.
        """
        return cls.__name__

    def sample_param(self, name: str) -> jnp.ndarray:
        return numpyro.sample(**self.config.parameter_to_numpyro(name))

    def sample_plate(
        self,
        *,
        plate_name: str,
        size: int,
        param_name: str,
        index: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Helper function for a simple plate
        """
        with numpyro.plate(plate_name, size):
            vals = self.sample_param(param_name)

        return vals[index] if index is not None else vals


class BetaBinomial(BaseModel):
    """flexible BetaBinomial GLM with independent random effects

    The class is generic. The set of random effects is defined in the config.
    The default config models 1 random effects with no interaction.
    """

    @classmethod
    def get_default_config(cls) -> ModelConfig:
        # Binomial needs at least one random effect - otherwise just Bernoulli
        re_1 = RandomEffectConfig(
            groups=["task"],
            name="task_effects",
            prior=PriorConfig(
                distribution=dist.Normal,
                distribution_args={"loc": 0.0, "scale": 0.3},
            ),
        )

        params = [
            ParameterConfig(
                name="overall_mean",
                prior=PriorConfig(
                    distribution=dist.Normal,
                    distribution_args={"loc": 0.0, "scale": 0.3},
                ),
            ),
            ParameterConfig(
                name="dispersion_phi",
                prior=PriorConfig(
                    distribution=dist.Gamma,
                    distribution_args={"concentration": 1.0, "rate": 0.1},
                ),
            ),
        ]

        return ModelConfig(
            configurable_parameters=params,
            random_effects=[re_1],
            main_effect_params=["success_prob"],
        )

    def _prepare_data(self, data: pd.DataFrame) -> Tuple[Features, Coords, Dims]:
        # TODO: Magda to check this assumption.
        # collect all grouping columns that appear in any RE
        # each RE should contain only one value per linear predictor.
        random_effect_cols = {g for re in self.config.random_effects for g in re.groups}

        group_cols = sorted(random_effect_cols)

        if not group_cols:
            raise ValueError(
                "No random effect columns found in the data. For a BetaBinomial model you must have at least one variable to group by otherwise you should use Bernoulli."
            )
        data = (
            data.groupby(group_cols, observed=True)
            .agg(
                n_correct=("score", "sum"),
                n_total=("score", "count"),
            )
            .reset_index()
        )

        features: Features = {
            "total_count": jnp.asarray(data["n_total"].values, dtype=jnp.int32),
            "n_correct": jnp.asarray(data["n_correct"].values, dtype=jnp.int32),
            "obs": jnp.asarray(
                data["n_correct"].values / data["n_total"].values, dtype=jnp.float32
            ),
        }
        coords: Coords = {}
        dims: Dims = {}

        for re_cfg in self.config.random_effects:
            key_name = "_x_".join(re_cfg.groups)  # note just the name if only one group
            if len(re_cfg.groups) > 1:
                codes, cats = pd.factorize(
                    tuple(zip(*(data[g] for g in re_cfg.groups))), sort=True
                )  # assume our groups are categorical.
                cats = ["_x_".join(cat) for cat in cats]
            else:
                codes, cats = pd.factorize(data[re_cfg.groups[0]], sort=True)
                cats = cats.tolist()

            idx_name = f"{re_cfg.name}_index"
            size_name = f"num_{re_cfg.name}"

            features[idx_name] = jnp.asarray(codes, dtype=jnp.int32)
            features[size_name] = len(cats)

            coords[key_name] = cats
            dims[re_cfg.name] = [key_name]

        return features, coords, dims

    def linear_component(self, **features) -> jnp.ndarray:
        """
        Build the linear component of the model.
        """
        linear = self.sample_param("overall_mean")

        # loop over random effects
        for re_cfg in self.config.random_effects:
            size = features[f"num_{re_cfg.name}"]
            index = features[f"{re_cfg.name}_index"]

            # intercept
            re_int = self.sample_plate(
                plate_name=f"{re_cfg.name}_plate",
                size=size,
                param_name=re_cfg.name,
                index=index,
            )
            linear = linear + re_int

        return linear

    def build_model(self):
        def model(**features):
            linear = self.linear_component(**features)
            # likelihood
            p_bar = numpyro.deterministic("p_bar", jax.nn.sigmoid(linear))

            phi = self.sample_param("dispersion_phi")
            alpha = p_bar * phi
            beta = (1.0 - p_bar) * phi

            success_prob = numpyro.sample("success_prob", dist.Beta(alpha, beta))

            n_tot = features["total_count"]
            if features["obs"] is not None:
                obs_cnt = jnp.round(features["obs"] * n_tot).astype(jnp.int32)

                numpyro.sample(
                    "n_correct",
                    dist.Binomial(n_tot, success_prob),
                    obs=obs_cnt,
                )
            else:
                count_pred = numpyro.sample(
                    "n_correct",
                    dist.Binomial(n_tot, success_prob),
                )
                # we like to plot proportions
                numpyro.deterministic("obs", count_pred / n_tot)

        return model


class Binomial(BetaBinomial):
    """Binomial GLM (optionally with independent random effects)."""

    @classmethod
    def get_default_config(cls) -> ModelConfig:
        params = [
            ParameterConfig(
                name="overall_mean",
                prior=PriorConfig(
                    distribution=dist.Normal,
                    distribution_args={"loc": 0.0, "scale": 0.3},
                ),
            ),
        ]

        # Binomial needs at least one random effect - otherwise just Bernoulli
        re_1 = RandomEffectConfig(
            groups=["task"],
            name="task_effects",
            prior=PriorConfig(
                distribution=dist.Normal,
                distribution_args={"loc": 0.0, "scale": 0.3},
            ),
        )

        return ModelConfig(
            configurable_parameters=params,
            random_effects=[re_1],
            main_effect_params=["success_prob"],
        )

    # same prep data as BetaBinomial

    def build_model(self):
        def model(**features):
            linear = self.linear_component(**features)

            # likelihood
            success_prob = numpyro.deterministic("success_prob", jax.nn.sigmoid(linear))

            n_tot = features["total_count"]
            if features["obs"] is not None:
                obs_cnt = jnp.round(features["obs"] * n_tot).astype(jnp.int32)

                numpyro.sample(
                    "n_correct",
                    dist.Binomial(n_tot, success_prob),
                    obs=obs_cnt,
                )
            else:
                count_pred = numpyro.sample(
                    "n_correct",
                    dist.Binomial(n_tot, success_prob),
                )
                # we like to plot proportions
                numpyro.deterministic("obs", count_pred / n_tot)

        return model


class Bernoulli(BaseModel):
    """Bernoulli GLM (optionally with independent random effects)."""

    @classmethod
    def get_default_config(cls) -> ModelConfig:
        params = [
            ParameterConfig(
                name="overall_mean",
                prior=PriorConfig(
                    distribution=dist.Normal,
                    distribution_args={"loc": 0.0, "scale": 0.3},
                ),
                main_effect=True,
            ),
        ]

        return ModelConfig(
            configurable_parameters=params,
            random_effects=[],
            main_effect_params=["success_prob"],
        )

    def _prepare_data(self, data: pd.DataFrame) -> Tuple[Features, Coords, Dims]:
        """Convert a df of binary trials into NumPyro ready tensors.
        score 0/1 outcome for each trial
        """

        features: Features = {}
        coords: Coords = {}
        dims: Dims = {}

        for re_cfg in self.config.random_effects:
            key_name = "_x_".join(re_cfg.groups)
            if len(re_cfg.groups) == 1:
                codes, cats = pd.factorize(data[re_cfg.groups[0]], sort=True)
                cats = cats.tolist()
            else:
                # interaction term
                codes, cats = pd.factorize(
                    tuple(zip(*(data[g] for g in re_cfg.groups))), sort=True
                )
                cats = ["_x_".join(cat) for cat in cats]

            idx_name = f"{re_cfg.name}_index"
            size_name = f"num_{re_cfg.name}"

            features[idx_name] = jnp.asarray(codes, dtype=jnp.int32)
            features[size_name] = len(cats)

            coords[key_name] = cats
            dims[re_cfg.name] = [key_name]

        features["obs"] = jnp.asarray(data["score"].values, dtype=jnp.int32)

        return features, coords, dims

    def linear_component(self, **features) -> jnp.ndarray:
        """Intercept + (optional) random-effects intercepts."""
        linear = self.sample_param("overall_mean")

        for re_cfg in self.config.random_effects:
            size = features[f"num_{re_cfg.name}"]
            index = features[f"{re_cfg.name}_index"]

            re_int = self.sample_plate(
                plate_name=f"{re_cfg.name}_plate",
                size=size,
                param_name=re_cfg.name,
                index=index,
            )
            linear = linear + re_int

        return linear

    def build_model(self):
        def model(**features):
            # fixed + random effects
            linear = self.linear_component(**features)

            # success probability on (0,1)
            success_prob = numpyro.deterministic("success_prob", jax.nn.sigmoid(linear))

            # likelihood
            if features["obs"] is not None:
                numpyro.sample(
                    "obs", dist.Bernoulli(probs=success_prob), obs=features["obs"]
                )
            else:  # prior / posterior predictive
                numpyro.sample("obs", dist.Bernoulli(probs=success_prob))

        return model


class TwoLevelGroupBinomial(BaseModel):
    """
    Hierarchical Binomial model, aggregated at group level, with
    non-centred parameterisation for the global mean and the group-specific effects.
    """

    @classmethod
    def get_default_config(cls) -> ModelConfig:
        params = [
            ParameterConfig(
                name="mu_overall",
                prior=PriorConfig(
                    distribution=dist.Normal,
                    distribution_args={"loc": 0.0, "scale": 1.0},
                ),
            ),
            ParameterConfig(
                name="sigma_overall",
                prior=PriorConfig(
                    distribution=dist.HalfNormal,
                    distribution_args={"scale": 0.5},
                ),
            ),
            ParameterConfig(
                name="z_overall",
                prior=PriorConfig(
                    distribution=dist.Normal,
                    distribution_args={"loc": 0.0, "scale": 1.0},
                ),
            ),
            ParameterConfig(
                name="sigma_domain",
                prior=PriorConfig(
                    distribution=dist.HalfNormal,
                    distribution_args={"scale": 0.1},
                ),
            ),
        ]

        return ModelConfig(
            configurable_parameters=params,
            random_effects=[],  # TODO: set up is still not flexible enough for this multilevel model....
            main_effect_params=[
                "overall_mean",
                "group_effects",
            ],
        )

    def _prepare_data(self, data: pd.DataFrame) -> Tuple[Features, Coords, Dims]:
        agg = (
            data.groupby(["group"], observed=True)
            .agg(
                n_correct=("score", "sum"),
                n_total=("score", "count"),
            )
            .reset_index()
        )

        group_codes, group_levels = pd.factorize(agg["group"], sort=True)

        features: Features = {
            "group_index": jnp.asarray(group_codes, dtype=jnp.int32),
            "num_group": int(len(group_levels)),
            "total_count": jnp.asarray(agg["n_total"].values, dtype=jnp.int32),
            "obs": jnp.asarray(agg["n_correct"].values, dtype=jnp.int32),
        }

        coords: Coords = {"group": group_levels.tolist()}
        dims: Dims = {"group_effects": ["group"]}

        return features, coords, dims

    def build_model(self):
        def model(**features):
            mu_overall = self.sample_param("mu_overall")
            sigma_overall = self.sample_param("sigma_overall")
            z_overall = self.sample_param("z_overall")

            overall_mean = mu_overall + sigma_overall * z_overall
            numpyro.deterministic("overall_mean", overall_mean)

            mu_group = numpyro.sample("mu_group", dist.Normal(overall_mean, 0.5))
            sigma_group = self.sample_param("sigma_domain")

            z_group = numpyro.sample(
                "z_group",
                dist.Normal(0.0, 1).expand([features["num_group"]]),
            )

            group_effects = mu_group + sigma_group * z_group
            numpyro.deterministic("group_effects", group_effects)

            # likelihood
            logit_p = group_effects[features["group_index"]]

            numpyro.sample(
                "obs",
                dist.Binomial(
                    total_count=features["total_count"],
                    probs=jax.nn.sigmoid(logit_p),
                ),
                obs=features["obs"],
            )

        return model
