from functools import wraps
from typing import (
    Callable,
    Literal,
    Optional,
    Protocol,
    Tuple,
)

from ..analysis_state import ModelAnalysisState
from ..registry import RegistryInfo, registry_add, registry_tag
from ..ui import ModellingDisplay

CheckerResult = Literal["pass", "fail", "error", "NA"]


class Checker(Protocol):
    """
    Check the fit of a GLM to the data based on ModelAnalysisState which contains
    information on the data, GLM and maybe the fit.
    """

    def __call__(
        self, state: ModelAnalysisState, display: ModellingDisplay | None = None
    ) -> Tuple[ModelAnalysisState, CheckerResult]:
        """
        Perform the check.

        Args:
            state: The analysis state of the model.
            display: The display object used to get user interaction IF required.
        Returns:
            The updated analysis state (with information about the results which were used to yield the checkererresult) and the result of the analysis.
        """
        ...


def checker(
    checker_builder: Optional[Callable[..., Checker]] = None,
    *,
    when: Literal["before", "after"] = "after",
) -> Callable[..., Checker]:
    """
    Decorator to register a checker and enforce an agreed upon interface for the checkers.

    Args:
        checker_builder: builder which creates a checker. Through decorating the builder, the checker is registered.
        when: whether this checker should run 'before' or 'after' model fitting.
    Returns:
        Checker with registration.
    """

    def decorate(cb: Callable[..., Checker]) -> Callable[..., Checker]:
        # include 'when' in your registry info / metadata
        registry_info = RegistryInfo(
            type="checker", name=cb.__name__, metadata={"when": when}
        )

        @wraps(cb)
        def checker_wrapper(*args, **kwargs) -> Checker:
            checker = cb(*args, **kwargs)

            @wraps(checker)
            def checker_interface(
                state: ModelAnalysisState,
                display: ModellingDisplay | None = None,
            ) -> Tuple[ModelAnalysisState, CheckerResult]:
                """
                Wrapper to enforce the interface of the checker.
                """
                if not isinstance(state, ModelAnalysisState):
                    raise TypeError("state must be an instance of ModelAnalysisState")
                if display is not None and not isinstance(display, ModellingDisplay):
                    raise TypeError("display must be an instance of ModellingDisplay")

                return checker(state, display)

            # register with the builder-args and our extra metadata
            registry_tag(
                cb,
                checker_interface,
                info=registry_info,
                *args,
                **kwargs,
            )
            return checker_interface

        registry_add(checker_wrapper, registry_info)
        return checker_wrapper

    # support both @checker and @checker(when="before")
    if checker_builder is None:
        return decorate
    return decorate(checker_builder)
