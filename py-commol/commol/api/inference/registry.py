"""The runners of uncertainty methods, looked up by configuration type.

A method registers its runner with `register`; ``Calibrator.uncertainty``
finds it with `run_method`. The entry point knows no method.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar, cast

from pydantic import BaseModel

from commol.context.inference.method import UncertaintyMethod

if TYPE_CHECKING:
    from commol.api.calibrator import Calibrator

MethodT = TypeVar("MethodT", bound=UncertaintyMethod[Any])
ResultT = TypeVar("ResultT", bound=BaseModel)

_RUNNERS: dict[type[UncertaintyMethod[Any]], Callable[["Calibrator", Any], Any]] = {}


def register(
    method_type: type[MethodT],
    runner: Callable[["Calibrator", MethodT], BaseModel],
) -> None:
    """
    Register the runner of an uncertainty method.

    Parameters
    ----------
    method_type : type[UncertaintyMethod]
        The configuration type of the method.
    runner : Callable[[Calibrator, UncertaintyMethod], BaseModel]
        Runs the method on a calibrator and returns its result.

    Raises
    ------
    ValueError
        If the type already has a runner.
    """
    if method_type in _RUNNERS:
        raise ValueError(f"{method_type.__name__} already has a runner.")
    _RUNNERS[method_type] = runner


def registered() -> list[str]:
    """
    The names of the configuration types that have a runner.

    Returns
    -------
    list[str]
        Sorted names.
    """
    return sorted(method_type.__name__ for method_type in _RUNNERS)


def run_method(calibrator: "Calibrator", method: UncertaintyMethod[ResultT]) -> ResultT:
    """
    Run `method` on `calibrator` with the runner of its type.

    Parameters
    ----------
    calibrator : Calibrator
        The calibration problem.
    method : UncertaintyMethod[ResultT]
        The configuration of the method.

    Returns
    -------
    ResultT
        The method's result.

    Raises
    ------
    ValueError
        If no runner is registered for the configuration's type.
    """
    runner = _RUNNERS.get(type(method))
    if runner is None:
        raise ValueError(
            f"No uncertainty method is registered for {type(method).__name__}. "
            f"Registered: {registered()}."
        )
    return cast("ResultT", runner(calibrator, method))
