"""The runners of inference methods, looked up by configuration type.

A method registers its runner with `register`; ``Calibrator.uncertainty``,
``Calibrator.identifiability`` and ``SensitivityAnalysis.run`` find it with
`run_method`, and pass the runner themselves. The entry points know no method.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar, cast

from pydantic import BaseModel

from commol.context.inference.method import (
    IdentifiabilityMethod,
    InferenceMethod,
    SensitivityMethod,
    UncertaintyMethod,
)

if TYPE_CHECKING:
    from commol.api.calibrator import Calibrator
    from commol.api.sensitivity import SensitivityAnalysis

MethodT = TypeVar("MethodT", bound=InferenceMethod[BaseModel])
ResultT = TypeVar("ResultT", bound=BaseModel)

type EntryPoint = Calibrator | SensitivityAnalysis
"""What a runner runs on: the calibrator of an uncertainty or identifiability
method, or the analysis of a sensitivity method."""

type Runner = Callable[[EntryPoint, InferenceMethod[BaseModel]], BaseModel]

_RUNNERS: dict[type[InferenceMethod[BaseModel]], Runner] = {}

_ENTRY_POINTS: dict[type[InferenceMethod[BaseModel]], str] = {
    UncertaintyMethod: "Calibrator.uncertainty",
    IdentifiabilityMethod: "Calibrator.identifiability",
    SensitivityMethod: "SensitivityAnalysis.run",
}


def register(
    method_type: type[MethodT],
    runner: Callable[[Any, MethodT], BaseModel],
) -> None:
    """
    Register the runner of an inference method.

    Parameters
    ----------
    method_type : type[InferenceMethod]
        The configuration type of the method.
    runner : Callable[[Calibrator | SensitivityAnalysis, InferenceMethod], BaseModel]
        Runs the method on the object of its entry point (a `Calibrator`,
        or a `SensitivityAnalysis` for a sensitivity method) and returns its
        result.

    Raises
    ------
    ValueError
        If the type already has a runner.
    """
    if method_type in _RUNNERS:
        raise ValueError(f"{method_type.__name__} already has a runner.")
    _RUNNERS[method_type] = cast("Runner", runner)


def registered() -> list[str]:
    """
    The names of the configuration types that have a runner.

    Returns
    -------
    list[str]
        Sorted names.
    """
    return sorted(method_type.__name__ for method_type in _RUNNERS)


def run_method(
    entry_point: EntryPoint,
    method: InferenceMethod[ResultT],
    kind: type[InferenceMethod[BaseModel]],
) -> ResultT:
    """
    Run `method` on `entry_point` with the runner of its type.

    Parameters
    ----------
    entry_point : Calibrator | SensitivityAnalysis
        The calibration problem, or the sensitivity analysis.
    method : InferenceMethod[ResultT]
        The configuration of the method.
    kind : type[InferenceMethod]
        The kind of method the entry point takes: `UncertaintyMethod`,
        `IdentifiabilityMethod` or `SensitivityMethod`.

    Returns
    -------
    ResultT
        The method's result.

    Raises
    ------
    TypeError
        If `method` is not of `kind`; the message names its entry point.
    ValueError
        If no runner is registered for the configuration's type.
    """
    if not isinstance(method, kind):
        right = next(
            (
                name
                for other, name in _ENTRY_POINTS.items()
                if isinstance(method, other)
            ),
            None,
        )
        hint = f" Run it with {right}." if right else ""
        raise TypeError(f"{type(method).__name__} is not a {kind.__name__}.{hint}")
    runner = _RUNNERS.get(type(method))
    if runner is None:
        raise ValueError(
            f"No method is registered for {type(method).__name__}. "
            f"Registered: {registered()}."
        )
    return cast("ResultT", runner(entry_point, method))
