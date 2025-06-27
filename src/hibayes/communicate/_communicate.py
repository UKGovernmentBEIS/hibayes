from functools import wraps
from typing import (
    Callable,
    Literal,
    Protocol,
    Tuple,
)

from ..analysis_state import AnalysisState
from ..registry import RegistryInfo, registry_add, registry_tag
from ..ui import ModellingDisplay

CommunicateResult = Literal["pass", "fail", "error", "NA"]


class Communicator(Protocol):
    """
    Communicate the results of a model analysis.
    """

    def __call__(
        self,
        state: AnalysisState,
        display: ModellingDisplay | None = None,
    ) -> Tuple[AnalysisState, Literal["pass", "fail", "error", "NA"]]:
        """
        Perform the communication.

        Args:
            state: The analysis state of the model.
            display: The display object used to get user interaction IF required.
        Returns:
            The updated analysis state (with information about the results which were used to yield the checkererresult) and the result of the analysis.
        """
        ...


def communicate(
    communicate_builder: Callable[..., Communicator],
) -> Callable[..., Communicator]:
    """
    Decorator to register a communicate and enforce an agreed upon interface for the communicators.

    Args:
        communicate_builder: builder which creates a communicator. Through decorating the builder, the communicator is registered.
    Returns:
        Communicator with registration.
    """

    @wraps(communicate_builder)
    def communicator_wrapper(*args, **kwargs) -> Communicator:
        communicator = communicate_builder(*args, **kwargs)

        @wraps(communicate_builder)
        def communicator_interface(
            state: AnalysisState,
            display: ModellingDisplay | None = None,
        ) -> Tuple[AnalysisState, CommunicateResult]:
            """
            Wrapper to enforce the interface of the communicator.
            """
            return communicator(state, display)

        registry_tag(
            communicate_builder,
            communicator_interface,
            info=registry_info,
            *args,
            **kwargs,
        )

        return communicator_interface

    registry_info = RegistryInfo(type="communicator", name=communicate_builder.__name__)
    registry_add(communicator_wrapper, registry_info)

    return communicator_wrapper
