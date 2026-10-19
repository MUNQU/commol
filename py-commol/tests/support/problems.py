"""Calibration problems that many tests share."""

from collections.abc import Iterable, Sequence

from commol import (
    CalibrationParameter,
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    Simulation,
)

from .models import a_to_b_builder, build


def initial_condition_problem(
    *ids: str, remainder: str | None = None
) -> CalibrationProblem:
    """
    A problem that declares each id as an initial condition in ``[0, 10⁶]``,
    wide enough for an amount and for a category split.

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
                max_bound=1e6,
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


def flow_observations(
    values: Sequence[float], *, windowed: bool, start: int = 1
) -> list[ObservedDataPoint]:
    """
    Observations of B at positions ``start``, ``start + 1``, …: with
    `windowed`, of the flow of one step each; otherwise of the amount of B.
    """
    return [
        ObservedDataPoint(
            at=start + index,
            output="B",
            value=float(value),
            window=1 if windowed else None,
        )
        for index, value in enumerate(values)
    ]


def flow_calibrator(
    values: Sequence[float],
    *,
    windowed: bool,
    seed: int = 1,
    bounds: tuple[float, float] = (0.0, 50.0),
) -> Calibrator:
    """
    Fit a constant flow ``c`` from A to B (``a_to_b_builder`` with the rate
    ``c * A / A``, absolute because it names A, and 10⁶ in A) to
    `values` at positions 1, 2, ….

    A model linear in its parameter: with `windowed`, each observation is the
    flow of one step and its prediction is ``c``, so the fit is the mean of
    the values; otherwise each observation is the amount of B at ``t`` and
    its prediction is ``c·t``, a regression through the origin.
    """
    model = build(
        a_to_b_builder(
            parameters={"c": None}, rate="c * A / A", amounts={"A": 1e6, "B": 0}
        )
    )
    problem = CalibrationProblem(
        observed_data=flow_observations(values, windowed=windowed),
        parameters=[bounded("c", *bounds, initial_guess=sum(bounds) / 2)],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=1000, sd_tolerance=1e-14),
        seed=seed,
    )
    return Calibrator(Simulation(model), problem)
