"""Calibrators with no feasible point in their bounds: the fit of each ends
on the penalty of the objective, converged all the same."""

from collections.abc import Callable

from commol import (
    CalibrationConstraint,
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    Simulation,
)
from commol.observations import BinomialObservations

from .models import a_to_b_builder, build
from .problems import bounded, flow_calibrator, flow_observations, observed

_FLOWS = [9.0, 11.0, 12.0, 10.0, 8.0, 9.0, 13.0, 11.0, 10.0, 9.0, 10.0, 12.0]
_REFUSED = "^The refits cannot start from the fit: Infeasible point: "


def _constrained(constraint: CalibrationConstraint) -> Calibrator:
    """The flow ``c`` of ``flow_calibrator``, in ``[0, 50]``, under
    `constraint`."""
    calibrator = flow_calibrator(_FLOWS, windowed=True)
    problem = calibrator.problem.model_copy(update={"constraints": [constraint]})
    return Calibrator(calibrator.simulation, problem)


def _binomial() -> Calibrator:
    """The flow ``c``, in ``[10, 50]``, observed by a binomial model of 5
    trials, whose mean is at most 5."""
    return flow_calibrator(
        [2.0, 3.0, 1.0, 4.0] * 3,
        windowed=True,
        bounds=(10.0, 50.0),
        observation_models={"B": BinomialObservations(trials=5)},
    )


def no_run_calibrator() -> Calibrator:
    """A fit on the plateau of the penalty with no run: the first step moves
    5000 out of the 1000 of A, at every point the optimizer reads."""
    truth = Simulation(build(a_to_b_builder(amounts={"A": 1000, "B": 0}))).run(20)
    problem = CalibrationProblem(
        observed_data=observed(truth, "B", range(1, 21)),
        parameters=[bounded("k", 0.0, 10.0, initial_guess=5.0)],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=300),
        seed=1,
    )
    model = a_to_b_builder(parameters={"k": None}, amounts={"A": 1000, "B": 0})
    return Calibrator(Simulation(build(model)), problem)


def _overflow() -> Calibrator:
    """The decay of ``a_to_b_builder`` from 10²⁰⁰ in A at the rate ``k``, in
    ``[0.1, 0.5]``: each squared residual is above the largest float."""
    model = build(a_to_b_builder(parameters={"k": None}, amounts={"A": 1e200, "B": 0}))
    problem = CalibrationProblem(
        observed_data=flow_observations(_FLOWS, windowed=False),
        parameters=[bounded("k", 0.1, 0.5)],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=500),
        seed=1,
    )
    return Calibrator(Simulation(model), problem)


INFEASIBLE_FITS: dict[str, tuple[Callable[[], Calibrator], str]] = {
    "constraint": (
        lambda: _constrained(
            CalibrationConstraint(id="c_above_100", expression="c - 100")
        ),
        _REFUSED + "the constraint 'c_above_100' is below zero",
    ),
    "constraint_at_positions": (
        lambda: _constrained(
            CalibrationConstraint(id="b_above_1000", expression="B - 1000", at=[5.0])
        ),
        _REFUSED + "the constraint 'b_above_1000' is below zero",
    ),
    "mean_outside_model": (
        _binomial,
        _REFUSED + "a predicted mean is outside the means its observation model allows",
    ),
    "loss_not_finite": (
        _overflow,
        _REFUSED + "the run, the loss or the likelihood is not a finite number",
    ),
}
"""Each kind of fit that is not feasible: a calibrator whose fit is of that
kind, and the pattern of the error of a method that refits around it."""
