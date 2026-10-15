"""Calibration problems that many tests share."""

from collections.abc import Iterable

from commol import (
    CalibrationParameter,
    CalibrationProblem,
    NelderMeadConfig,
    ObservedDataPoint,
)


def initial_condition_problem(
    *ids: str, remainder: str | None = None
) -> CalibrationProblem:
    """
    A problem that declares each id as an initial condition in ``[0, 1]``.

    It is meant for writing values back, not for a run: it has one placeholder
    observation and one Nelder-Mead iteration.
    """
    return CalibrationProblem(
        initial_condition_remainder=remainder,
        observed_data=[ObservedDataPoint(at=1, output="A", value=1.0)],
        parameters=[
            CalibrationParameter(
                id=entry_id,
                parameter_type="initial_condition",
                min_bound=0.0,
                max_bound=1.0,
            )
            for entry_id in ids
        ],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=1),
    )


def bounded(
    entry_id: str,
    low: float = 0.0,
    high: float = 1.0,
    *,
    kind: str = "parameter",
    initial_guess: float | None = None,
) -> CalibrationParameter:
    """A calibration parameter of ``kind`` searched in ``[low, high]``."""
    return CalibrationParameter(
        id=entry_id,
        parameter_type=kind,
        min_bound=low,
        max_bound=high,
        initial_guess=initial_guess,
    )


def observed(
    results: dict[str, list[float]],
    output: str,
    at: Iterable[int],
    *,
    factor: float = 1.0,
    factor_id: str | None = None,
) -> list[ObservedDataPoint]:
    """Observations of one output of a run, at the given integer positions."""
    return [
        ObservedDataPoint(
            at=step,
            output=output,
            value=results[output][step] * factor,
            factor_id=factor_id,
        )
        for step in at
    ]
