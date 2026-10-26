"""The configuration of an inference method, generic in its result, and what
its runner reads of a calibrator.

Three kinds of method share one base: an uncertainty method
(``Calibrator.uncertainty``), an identifiability method
(``Calibrator.identifiability``) and a sensitivity method
(``SensitivityAnalysis.run``). Each method declares its runner; each entry
point takes only its kind, and imports no method.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar, Generic, Protocol, TypeVar, cast

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from commol._problem.problem import CalibrationProblem
    from commol.commol_rs._commol_rs import CalibrationProblemProtocol

ResultT = TypeVar("ResultT", bound=BaseModel, covariant=True)
OutT = TypeVar("OutT", bound=BaseModel)


class CalibrationPort(Protocol):
    """
    What a method reads of a calibrator: the problem, the names of its
    parameters, its seed and the problem the engine built from them.

    `Calibrator` gives it; a runner, and a member of a result that reads the
    calibration problem, such as `LocalResult.delta_band`, read only it.
    """

    @property
    def problem(self) -> "CalibrationProblem":
        """The calibration problem."""
        ...

    @property
    def parameter_names(self) -> list[str]:
        """The names of the parameters, in the order of the problem."""
        ...

    @property
    def seed(self) -> int:
        """The seed of the random numbers of the calibration."""
        ...

    @property
    def _engine_problem(self) -> "CalibrationProblemProtocol":
        """The problem the engine built, which every method reads."""
        ...


class InferenceMethod(BaseModel, Generic[ResultT]):
    """
    The configuration of a method that reads a calibration problem or a
    sensitivity analysis, whose result is `ResultT`.

    A method subclasses one of its kinds, `UncertaintyMethod`,
    `IdentifiabilityMethod` or `SensitivityMethod`, with its result type, and
    declares its runner in ``_runner``: a function of the object of its entry
    point and the method that returns the result. The runner sends the method
    whole to the engine, as the JSON of its dump
    (`commol._data.engine_json.engine_json`); a field that holds a result
    with behavior in the engine, such as a sample, is named in
    ``_engine_handles`` and crosses as its handle.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    _engine_handles: ClassVar[tuple[str, ...]] = ()
    _runner: ClassVar[Callable[..., BaseModel] | None] = None
    _entry_point: ClassVar[str] = ""


class UncertaintyMethod(InferenceMethod[ResultT], Generic[ResultT]):
    """
    The configuration of an uncertainty method, whose result is `ResultT`.

    A method subclasses it with its result type, for example
    ``class BayesianBootstrap(UncertaintyMethod[BootstrapResult])``, and
    declares its runner. ``Calibrator.uncertainty(method)`` returns that
    result type, so a static type checker infers it from the configuration's
    type.
    """

    _entry_point: ClassVar[str] = "Calibrator.uncertainty"


class IdentifiabilityMethod(InferenceMethod[ResultT], Generic[ResultT]):
    """
    The configuration of an identifiability method, whose result is
    `ResultT`.

    ``Calibrator.identifiability(method)`` runs it, and returns its result
    type. The methods of commol return an `IdentifiabilityReport`.
    """

    _entry_point: ClassVar[str] = "Calibrator.identifiability"


class SensitivityMethod(InferenceMethod[ResultT], Generic[ResultT]):
    """
    The configuration of a sensitivity method, whose result is `ResultT`.

    ``SensitivityAnalysis.run(method)`` runs it on the inputs of the
    analysis, with no data, and returns its result type.
    """

    _entry_point: ClassVar[str] = "SensitivityAnalysis.run"


def run_method(
    entry_point: object,
    method: InferenceMethod[OutT],
    kind: type[InferenceMethod[BaseModel]],
) -> OutT:
    """
    Run `method` on `entry_point` with the runner it declares.

    Parameters
    ----------
    entry_point : Calibrator | SensitivityAnalysis
        The calibration problem, or the sensitivity analysis.
    method : InferenceMethod[OutT]
        The configuration of the method.
    kind : type[InferenceMethod]
        The kind of method the entry point takes: `UncertaintyMethod`,
        `IdentifiabilityMethod` or `SensitivityMethod`.

    Returns
    -------
    OutT
        The method's result.

    Raises
    ------
    TypeError
        If `method` is not of `kind`; the message names its entry point.
    ValueError
        If the configuration's type declares no runner.
    """
    if not isinstance(method, kind):
        entry = getattr(method, "_entry_point", "")
        hint = f" Run it with {entry}." if entry else ""
        raise TypeError(f"{type(method).__name__} is not a {kind.__name__}.{hint}")
    runner = type(method)._runner
    if runner is None:
        raise ValueError(
            f"{type(method).__name__} declares no runner: set its _runner to the "
            "function that runs it."
        )
    return cast("OutT", runner(entry_point, method))
