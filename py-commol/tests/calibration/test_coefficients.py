"""An observation of a weighted sum of outputs: each output times its known
coefficient, the share of it that the observation sees."""

import math

import numpy as np
import pytest
from pydantic import ValidationError

from commol import (
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    Simulation,
)
from commol._model.model import Model
from commol._simulation.results import SimulationResults
from commol.observations import NoNoise
from tests.support import abc_builder, bounded, build
from tests.support.plotted import predictive_of

COEFFICIENTS = [1.0, 0.6]


def _seen(
    run: dict[str, list[float]], at: int, label: str = "seen", **fields: object
) -> ObservedDataPoint:
    """The observation at `at` of ``B + 0.6 · C`` of `run`, in the series
    `label`."""
    return ObservedDataPoint(
        at=at,
        outputs=["B", "C"],
        coefficients=COEFFICIENTS,
        label=label,
        value=run["B"][at] + 0.6 * run["C"][at],
        **fields,
    )


def _calibrator(observations: list[ObservedDataPoint]) -> Calibrator:
    problem = CalibrationProblem(
        observed_data=observations,
        parameters=[
            bounded("k1", initial_guess=0.3),
            bounded("k2", high=0.5, initial_guess=0.2),
        ],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=2000, verbose=False),
        seed=42,
    )
    model = build(abc_builder(parameters={"k1": None, "k2": None}))
    return Calibrator(Simulation(model), problem)


def test_the_prediction_is_the_weighted_sum(truth_run) -> None:
    observations = [_seen(truth_run, at) for at in (20, 50)] + [
        _seen(truth_run, 50, window=10, scale=0.5)
    ]
    calibrator = _calibrator(observations)

    [[predictions]] = calibrator._engine_problem.predict_observations([[0.1, 0.05]])

    def weighted(at: int) -> float:
        return truth_run["B"][at] + 0.6 * truth_run["C"][at]

    expected = [weighted(20), weighted(50), 0.5 * (weighted(50) - weighted(40))]
    np.testing.assert_allclose(predictions, expected, rtol=1e-12)


def test_a_calibration_reads_the_weighted_sum(truth_run) -> None:
    calibrator = _calibrator([_seen(truth_run, at) for at in range(0, 100, 5)])

    fitted = calibrator.run().best_parameters

    assert fitted == pytest.approx({"k1": 0.1, "k2": 0.05}, rel=1e-4)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        (
            {"outputs": ["B", "C"], "coefficients": [1.0]},
            "Observation 'seen' has 1 coefficients for 2 outputs",
        ),
        (
            {"output": "B", "coefficients": [0.6]},
            "Observation 'seen' gives 'coefficients' with 'output'",
        ),
        (
            {"outputs": ["B", "C"], "coefficients": [1.0, -0.6]},
            "Observation 'seen' has the coefficient -0.6 for output 'C'",
        ),
        (
            {"outputs": ["B", "C"], "coefficients": [math.inf, 1.0]},
            "Observation 'seen' has the coefficient inf for output 'B'",
        ),
        (
            {"outputs": ["B", "C"], "coefficients": [math.nan, 1.0]},
            "Observation 'seen' has the coefficient NaN for output 'B'",
        ),
        (
            {"outputs": ["B", "C"], "coefficients": [0.0, 0.0]},
            "Observation 'seen' has every coefficient 0",
        ),
    ],
)
def test_invalid_coefficients_are_refused(
    fields: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        ObservedDataPoint(at=1, label="seen", value=1.0, **fields)


def test_a_coefficient_of_0_is_an_output_the_observation_does_not_see() -> None:
    point = ObservedDataPoint(
        at=1, outputs=["B", "C"], coefficients=[0.0, 0.6], label="seen", value=1.0
    )

    assert point.observed_coefficients == [0.0, 0.6]
    assert ObservedDataPoint(at=1, output="B", value=1.0).observed_coefficients == [1.0]


class TestTheReadersUseTheCoefficients:
    def test_the_model_line_of_a_run_is_the_weighted_sum(self, truth: Model) -> None:
        """A series panel whose observations weigh its outputs draws the
        weighted sum at every position of the run."""
        run = Simulation(truth).run(60)

        fig = run.plot.series(
            observed_data=[_seen(run, at, "B") for at in (20, 40)], series=["B"]
        )

        (ax,) = [ax for ax in fig.axes if ax.get_visible()]
        (line,) = [line for line in ax.lines if len(line.get_xdata()) == 61]
        expected = np.asarray(run["B"]) + 0.6 * np.asarray(run["C"])
        np.testing.assert_allclose(line.get_ydata(), expected, rtol=1e-12)

    def test_each_observation_of_a_run_is_read_with_its_coefficients(
        self, truth: Model
    ) -> None:
        run = Simulation(truth).run(60)
        weighted = np.asarray(run["B"]) + 0.6 * np.asarray(run["C"])
        expected = [weighted[20], weighted[40] - weighted[30]]

        of_observations = run.plot.observations(
            [_seen(run, 20), _seen(run, 40, window=10)]
        )
        of_series = run.plot.series(
            observed_data=[_seen(run, 20, "B"), _seen(run, 40, "B", window=10)],
            series=["B"],
        )

        (markers,) = [
            line for line in of_observations.axes[0].lines if line.get_marker() == "o"
        ]
        np.testing.assert_allclose(markers.get_ydata(), expected, rtol=1e-12)
        (ax,) = [ax for ax in of_series.axes if ax.get_visible()]
        (model,) = [line for line in ax.lines if line.get_label() == "Simulation"]
        np.testing.assert_allclose(model.get_ydata(), expected, rtol=1e-12)

    def test_the_band_of_a_predictive_reads_the_coefficients(self) -> None:
        """A series panel whose observations weigh its outputs draws the band
        of the weighted sum at every position of the predictive."""
        predictive = predictive_of([0.02, 0.05, 0.1, 0.2])

        def points(at: list[float]) -> list[ObservedDataPoint]:
            return [
                ObservedDataPoint(
                    at=position,
                    outputs=["A", "B"],
                    coefficients=[0.5, 1.0],
                    label="B",
                    value=1.0,
                )
                for position in at
            ]

        fig = predictive.plot.series(
            observed_data=points([10, 20]), series=["B"], confidence_level=0.9
        )

        expected = predictive.at_observations(
            points(predictive.positions), NoNoise()
        ).band(0.9)
        (ax,) = [ax for ax in fig.axes if ax.get_visible()]
        (median,) = [line for line in ax.lines if len(line.get_xdata()) == 41]
        np.testing.assert_allclose(median.get_ydata(), expected.median)

    def test_the_report_records_the_coefficients(self) -> None:
        observed = [
            ObservedDataPoint(
                at=5,
                outputs=["B", "C"],
                coefficients=COEFFICIENTS,
                label="seen",
                value=1.0,
            ),
            ObservedDataPoint(at=5, output="B", value=1.0),
        ]

        report = SimulationResults([]).report([], 5, 2, observed)["observed"]

        assert report["seen"]["outputs"] == ["B", "C"]
        assert report["seen"]["coefficients"] == COEFFICIENTS
        assert "coefficients" not in report["B"]
