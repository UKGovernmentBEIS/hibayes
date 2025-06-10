from __future__ import annotations

from functools import wraps
from typing import Callable, Protocol

from ..process import Features
from ..registry import RegistryInfo, registry_add, registry_tag


class Model(Protocol):
    """
    Numpyro model
    """

    def __call__(
        self,
        features: Features,
    ) -> None:
        """
        A NumPyro model is a callable that defines a probabilistic program using
        NumPyro primitives (numpyro.sample, numpyro.param, etc.) to specify
        the joint probability distribution over random variables and observed data.

            Args:
                features: Features to be passed to the model.
        """
        ...


def model(
    model_builder: Callable[..., Model],
) -> Callable[..., Model]:
    """
    Decorator to register a model and enforce an agreed upon interface for the model.

    Args:
        model_builder: builder which creates a model. Through decorating the builder, the model is registered.
    Returns:
        Model with registration.
    """

    @wraps(model_builder)
    def model_wrapper(*args, **kwargs) -> Model:
        modeller = model_builder(*args, **kwargs)

        @wraps(model_builder)
        def model_interface(
            features: Features,
        ) -> None:
            """
            Wrapper to enforce the interface of the model.
            """
            if "obs" not in features:
                raise ValueError("Model must have 'obs' in features.")

            return modeller(features)

        kwargs_pass = {k: v for k, v in kwargs.items() if k != "info"}

        registry_tag(
            model_builder,
            model_interface,
            *args,
            info=registry_info,
            **kwargs_pass,
        )

        return model_interface

    registry_info = RegistryInfo(type="model", name=model_builder.__name__)
    registry_add(model_wrapper, registry_info)

    return model_wrapper
