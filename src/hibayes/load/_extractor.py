"""Functional extractor interface for HiBayES."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Dict, Protocol

from inspect_ai.log import EvalLog, EvalSample

from ..registry import RegistryInfo, registry_add, registry_tag


class Extractor(Protocol):
    """Extract metadata from evaluation logs."""

    def __call__(
        self,
        sample: EvalSample,
        eval_log: EvalLog,
    ) -> Dict[str, Any]:
        """
        Extract metadata from a sample and eval log.

        Args:
            sample: The evaluation sample to extract data from.
            eval_log: The complete evaluation log minus the samples.

        Returns:
            Dictionary of extracted metadata.
        """
        ...


def extractor(
    extractor_builder: Callable[..., Extractor],
) -> Callable[..., Extractor]:
    """
    Register an extractor and enforce an agreed upon interface for extractors.

    Args:
        extractor_builder: Builder which creates an extractor. Through decorating the builder,
                          the extractor is registered.

    Returns:
        Extractor with registration.
    """

    @wraps(extractor_builder)
    def extractor_wrapper(*args, **kwargs) -> Extractor:
        metadata_extractor = extractor_builder(*args, **kwargs)

        @wraps(extractor_builder)
        def extractor_interface(
            sample: EvalSample,
            eval_log: EvalLog,
        ) -> Dict[str, Any]:
            """Enforce the interface of the extractor."""
            # Call the extractor implementation
            result = metadata_extractor(sample, eval_log)

            # Ensure result is a dictionary
            if not isinstance(result, dict):
                raise ValueError(
                    f"Extractor must return a dictionary, got {type(result)}"
                )

            return result

        registry_tag(
            extractor_builder,
            extractor_interface,
            registry_info,
            *args,
            **kwargs,
        )

        return extractor_interface

    registry_info = RegistryInfo(type="extractor", name=extractor_builder.__name__)
    registry_add(extractor_wrapper, registry_info)

    return extractor_wrapper