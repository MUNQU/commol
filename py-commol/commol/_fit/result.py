"""The result of a calibration: the best of its starts, each start, and the
evaluations of an optimization."""

from typing import override

from pydantic import ConfigDict, Field

from commol._data.ranges import RangeChecked
from commol._data.result_model import ResultModel


class StartSummary(ResultModel):
    """
    One start of a multi-start optimization.

    Distinct ends with equal losses are a sign that the data do not identify
    the parameters.

    Attributes
    ----------
    start : dict[str, float]
        The point the optimizer started from.
    end : dict[str, float]
        The best point it found.
    loss : float
        The loss at the end.
    converged : bool
        Whether the optimizer met its convergence criterion.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: dict[str, float] = Field(description="Start point")
    end: dict[str, float] = Field(description="Best point found")
    loss: float = Field(description="Loss at the end")
    converged: bool = Field(description="Whether the optimizer converged")


class CalibrationResult(ResultModel, RangeChecked):
    """
    Result of a calibration: the best of its starts.

    Attributes
    ----------
    best_parameters : dict[str, float]
        The calibrated value of each parameter, in the order of the problem's
        parameters.
    final_loss : float
        Loss at the best parameters.
    iterations : int
        Iterations of the best start.
    converged : bool
        Whether the best start converged.
    termination_reason : str
        Why the best start stopped.
    starts : list[StartSummary]
        Every start that finished, in order; the first is the initial guess.
    failed : list[str]
        Each start that failed: ``"Start <index> failed: <reason>"``, also
        logged as a warning.
    seed : int
        The seed of the calibration's random numbers (`Calibrator.seed`):
        ``problem.seed``, or the seed the calibrator drew. Set as
        ``problem.seed``, it repeats the calibration and its uncertainty
        methods.
    """

    model_config = ConfigDict(extra="forbid")

    best_parameters: dict[str, float] = Field(
        default=..., description="Calibrated parameter values"
    )
    final_loss: float = Field(default=..., description="Final loss value")
    iterations: int = Field(default=..., description="Number of iterations performed")
    converged: bool = Field(default=..., description="Whether optimization converged")
    termination_reason: str = Field(
        default=..., description="Reason for optimization termination"
    )
    starts: list[StartSummary] = Field(
        default_factory=list, description="Every start that finished"
    )
    failed: list[str] = Field(
        default_factory=list, description="Each start that failed"
    )
    seed: int = Field(
        default=..., description="Seed of the calibration's random numbers"
    )

    @override
    def __str__(self) -> str:
        """String representation of calibration result."""
        return (
            f"CalibrationResult(\n"
            f"  converged={self.converged},\n"
            f"  final_loss={self.final_loss:.6f},\n"
            f"  iterations={self.iterations},\n"
            f"  best_parameters={self.best_parameters},\n"
            f"  termination_reason='{self.termination_reason}',\n"
            f"  starts={len(self.starts)},\n"
            f"  failed={len(self.failed)},\n"
            f"  seed={self.seed}\n"
            f")"
        )


class CalibrationEvaluation(ResultModel):
    """
    One evaluation of a feasible point during an optimization.

    Attributes
    ----------
    parameters : dict[str, float]
        The values the objective scored.
    loss : float
        The loss at these values.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    parameters: dict[str, float] = Field(description="Values the objective scored")
    loss: float = Field(description="Loss at these values")


class CalibrationHistory(ResultModel):
    """
    A calibration and every evaluation of a feasible point it made.

    Attributes
    ----------
    result : CalibrationResult
        The result of the calibration.
    evaluations : list[CalibrationEvaluation]
        Every evaluation of a feasible point, in the order they were made.
    """

    model_config = ConfigDict(extra="forbid")

    result: CalibrationResult = Field(description="The result")
    evaluations: list[CalibrationEvaluation] = Field(
        description="Every evaluation of a feasible point"
    )
