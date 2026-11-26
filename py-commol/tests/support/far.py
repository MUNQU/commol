"""A flow read by an observation of a large scale, whose curvature is near the
range of a number."""

from commol import (
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    Simulation,
)
from commol.observations import NormalObservations

from .models import a_to_b_builder, build
from .problems import bounded, flow_observations


def far_calibrator(
    scale: float,
    *,
    rate: str = "c * A / A",
    low: float = 0.0,
    amount: float = 0.0,
    far: float = 1000.0,
) -> Calibrator:
    """
    A flow ``rate`` from A to B, from ``amount`` in B, with ``c`` in
    ``[low, low + 100]``, and the observation "far" at 20 of ``scale`` times
    B, seen at ``scale · far``.

    With the defaults the fit is ``c = 50``, and the curvature of ``c`` on the
    unbounded scale is ``2.5e5 · scale²``.
    """
    model = build(
        a_to_b_builder(
            parameters={"c": None}, rate=rate, amounts={"A": 1e15, "B": amount}
        )
    )
    values = [0.01 * t + 0.01 * (-1) ** t for t in range(1, 9)]
    observed = ObservedDataPoint(
        at=20, output="B", label="far", value=far * scale, scale=scale
    )
    problem = CalibrationProblem(
        observed_data=[*flow_observations(values, windowed=False), observed],
        parameters=[bounded("c", low, low + 100.0, initial_guess=low + 50.0)],
        observation_models={
            "B": NormalObservations(sd=0.1),
            "far": NormalObservations(sd=1.0),
        },
        optimization_config=NelderMeadConfig(max_iterations=5000, sd_tolerance=1e-12),
        seed=2,
    )
    return Calibrator(Simulation(model), problem)
