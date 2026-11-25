"""An observation of the model plus a known background, which every reader of
the observation adds after its scale and its factor."""

import math

import pytest
from pydantic import ValidationError

from commol import (
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    Simulation,
)
from commol.observations import PoissonObservations
from commol.profiles import PredictionSpec
from tests.support import abc_builder, bounded, build, sum_calibrator

# The factor gives at most 48 of the model, so half of what is observed is
# background.
FACTOR, BACKGROUND = 0.3, 50.0

# log p(3) of a Poisson of mean 2.
THREE_AT_A_MEAN_OF_TWO = 3 * math.log(2) - 2 - math.log(6)


def _fitted(run: dict[str, list[float]], background: float) -> dict[str, float]:
    """The fit of ``k1`` and of a factor to ``FACTOR · B + BACKGROUND`` of
    `run`, each observation with `background`."""
    observations = [
        ObservedDataPoint(
            at=at,
            output="B",
            value=FACTOR * run["B"][at] + BACKGROUND,
            factor_id="f",
            background=background,
        )
        for at in range(0, 100, 5)
    ]
    problem = CalibrationProblem(
        observed_data=observations,
        parameters=[
            bounded("k1", initial_guess=0.3),
            bounded("f", 0.01, 2.0, kind="factor", initial_guess=1.0),
        ],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=2000, verbose=False),
        seed=42,
    )
    model = build(abc_builder(parameters={"k1": None, "k2": 0.05}))
    return Calibrator(Simulation(model), problem).run().best_parameters


def test_a_count_model_reads_the_background_where_the_model_gives_zero() -> None:
    # B is 0 at the start of the run.
    calibrator = sum_calibrator(
        [ObservedDataPoint(at=0, output="B", value=3.0, background=2.0)],
        {"B": PoissonObservations()},
    )

    (density,) = calibrator._engine_problem.pointwise_log_likelihood([0.1])

    assert density == pytest.approx(THREE_AT_A_MEAN_OF_TWO, rel=1e-12)


def test_a_calibration_reads_the_background(truth_run: dict[str, list[float]]) -> None:
    truth = {"k1": 0.1, "f": FACTOR}

    assert _fitted(truth_run, BACKGROUND) == pytest.approx(truth, rel=1e-4)

    without = _fitted(truth_run, 0.0)
    for name, value in truth.items():
        assert abs(without[name] / value - 1) > 0.1, name


@pytest.mark.parametrize("background", [-0.5, math.inf, math.nan])
@pytest.mark.parametrize(
    "reader",
    [
        lambda background: ObservedDataPoint(
            at=1, output="B", value=1.0, background=background
        ),
        lambda background: PredictionSpec(at=1, output="B", background=background),
    ],
    ids=["observation", "prediction"],
)
def test_an_invalid_background_is_refused(reader, background: float) -> None:
    message = "background is a finite number of at least 0, got"

    with pytest.raises(ValidationError, match=message):
        reader(background)
