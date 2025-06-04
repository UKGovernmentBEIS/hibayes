from functools import wraps
from typing import (
    Callable,
    Protocol,
)

from ..analysis_state import AnalysisState
from ..registry import RegistryInfo, registry_add, registry_tag
from ..ui import ModellingDisplay


class DataProcessor(Protocol):
    """
    Process the extracted data for modelling.
    """

    def __call__(
        self,
        state: AnalysisState,
        display: ModellingDisplay | None = None,
    ) -> AnalysisState:
        """
        Perform the processing.

        Args:
            state: The analysis state of the model.
            display: The display object used to get user interaction IF required.
        Returns:
            state: with processed data and/or with extracted features, coords and dims."""
        ...


def process(
    process_builder: Callable[..., DataProcessor],
) -> Callable[..., DataProcessor]:
    """
    Decorator to register a process and enforce an agreed upon interface for the data processor.

    Args:
        process_builder: builder which creates a data processor. Through decorating the builder, the data processor is registered.
    Returns:
        DataProcessor with registration.
    """

    @wraps(process_builder)
    def process_wrapper(*args, **kwargs) -> DataProcessor:
        processor = process_builder(*args, **kwargs)

        @wraps(processor)
        def process_interface(
            state: AnalysisState,
            display: ModellingDisplay | None = None,
        ) -> AnalysisState:
            """
            Wrapper to enforce the interface of the processor.
            """
            # we only modify the processed_data attr, keeping the original extracted data intact
            if state.processed_data is None:  # processors only act on processed data
                state.processed_data = state.data.copy()
            return processor(state, display)

        registry_tag(
            process_builder,
            process_interface,
            info=registry_info,
            *args,
            **kwargs,
        )

        return process_interface

    registry_info = RegistryInfo(type="processor", name=process_builder.__name__)
    registry_add(process_wrapper, registry_info)

    return process_wrapper
