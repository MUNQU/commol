"""Scores of a predictive equal their closed forms.

The predictive is normal: one observation of B at 10 times a factor ``f``,
whose draws are the normal quantiles ``μ + σ Φ⁻¹((i − ½)/n)`` with equal
weights. The prediction is ``B(10)·f``, so it is normal with mean
``m = B(10)·μ`` and standard deviation ``s = B(10)·σ``.
"""

import math
from statistics import NormalDist

import pytest

from commol import (
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    ParameterSample,
    Predictive,
    SampleMethod,
    Simulation,
    scores,
)
from tests.support import abc_builder, bounded, build

MU, SIGMA, DRAWS = 1.0, 0.1, 4001
STANDARD = NormalDist()


def _observation(value: float) -> ObservedDataPoint:
    return ObservedDataPoint(at=10, output="B", value=value, factor_id="f")


@pytest.fixture
def normal() -> tuple[Predictive, NormalDist]:
    """A normal predictive of B(10)·f, and its distribution."""
    simulation = Simulation(build(abc_builder(parameters={"k1": 0.1, "k2": 0.05})))
    b10 = simulation.run(10)["B"][10]
    problem = CalibrationProblem(
        observed_data=[_observation(b10)],
        parameters=[bounded("f", 0.01, 10.0, kind="factor")],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=1),
    )
    calibrator = Calibrator(simulation, problem)
    values = [[MU + SIGMA * STANDARD.inv_cdf((i + 0.5) / DRAWS)] for i in range(DRAWS)]
    sample = ParameterSample.equally_weighted(
        ["f"], [(0.01, 10.0)], values, SampleMethod(name="normal")
    )
    return Predictive(calibrator, sample, [10]), NormalDist(b10 * MU, b10 * SIGMA)


def _interval(dist: NormalDist, level: float) -> tuple[float, float]:
    return dist.inv_cdf((1 - level) / 2), dist.inv_cdf((1 + level) / 2)


def _interval_score(dist: NormalDist, level: float, y: float) -> float:
    lower, upper = _interval(dist, level)
    alpha = 1 - level
    return (
        (upper - lower)
        + 2 / alpha * max(lower - y, 0.0)
        + 2 / alpha * max(y - upper, 0.0)
    )


class TestClosedForms:
    @pytest.mark.parametrize("offset", [0.0, 1.0, 2.5])
    def test_coverage_is_inside_and_outside_the_normal_interval(
        self, normal: tuple[Predictive, NormalDist], offset: float
    ) -> None:
        predictive, dist = normal
        y = dist.mean + offset * dist.stdev
        expected = float(_interval(dist, 0.9)[0] <= y <= _interval(dist, 0.9)[1])

        score = scores.coverage(predictive, [_observation(y)], 0.9)

        assert score.per_observation == [expected]

    @pytest.mark.parametrize("offset", [0.0, 2.5, -3.0])
    def test_the_interval_score_is_its_closed_form(
        self, normal: tuple[Predictive, NormalDist], offset: float
    ) -> None:
        predictive, dist = normal
        y = dist.mean + offset * dist.stdev

        score = scores.interval_score(predictive, [_observation(y)], 0.8)

        assert score.mean == pytest.approx(_interval_score(dist, 0.8, y), rel=1e-3)

    def test_the_weighted_interval_score_is_its_closed_form(
        self, normal: tuple[Predictive, NormalDist]
    ) -> None:
        predictive, dist = normal
        y = dist.mean + 1.3 * dist.stdev
        levels = [0.5, 0.8, 0.95]
        expected = (
            0.5 * abs(y - dist.mean)
            + sum((1 - c) / 2 * _interval_score(dist, c, y) for c in levels)
        ) / (len(levels) + 0.5)

        score = scores.weighted_interval_score(predictive, [_observation(y)], levels)

        assert score.mean == pytest.approx(expected, rel=1e-3)

    @pytest.mark.parametrize("offset", [0.0, 0.7, -2.0])
    def test_the_crps_is_the_normal_closed_form(
        self, normal: tuple[Predictive, NormalDist], offset: float
    ) -> None:
        predictive, dist = normal
        z = offset
        y = dist.mean + z * dist.stdev
        expected = dist.stdev * (
            z * (2 * STANDARD.cdf(z) - 1) + 2 * STANDARD.pdf(z) - 1 / math.sqrt(math.pi)
        )

        score = scores.crps(predictive, [_observation(y)])

        assert score.mean == pytest.approx(expected, rel=1e-3)

    @pytest.mark.parametrize("offset", [-1.0, 0.0, 1.5])
    def test_the_pit_is_the_normal_distribution_function(
        self, normal: tuple[Predictive, NormalDist], offset: float
    ) -> None:
        predictive, dist = normal

        score = scores.pit(
            predictive, [_observation(dist.mean + offset * dist.stdev)], seed=1
        )

        assert score.mean == pytest.approx(STANDARD.cdf(offset), abs=1e-3)


class TestSeries:
    def test_scores_average_per_series(
        self, normal: tuple[Predictive, NormalDist]
    ) -> None:
        predictive, dist = normal
        inside = _observation(dist.mean)
        outside = ObservedDataPoint(
            at=10, output="B", value=dist.mean * 3, factor_id="f", label="far"
        )

        score = scores.coverage(predictive, [inside, outside, inside], 0.9)

        assert score.per_observation == [1.0, 0.0, 1.0]
        assert score.per_series == {"B": 1.0, "far": 0.0}
        assert score.mean == pytest.approx(2 / 3)

    def test_the_sample_crps_of_a_known_ensemble(
        self, normal: tuple[Predictive, NormalDist]
    ) -> None:
        predictive, dist = normal
        calibrator = predictive.calibrator
        values = [0.5, 2.0, 1.0, 4.0]
        sample = ParameterSample.equally_weighted(
            ["f"], [(0.01, 10.0)], [[v] for v in values], SampleMethod(name="four")
        )
        b10 = dist.mean / MU
        draws = [b10 * v for v in values]
        y = b10 * 1.5
        expected = (
            sum(abs(x - y) for x in draws) / 4
            - 0.5 * sum(abs(a - b) for a in draws for b in draws) / 16
        )

        score = scores.crps(Predictive(calibrator, sample, [10]), [_observation(y)])

        assert score.mean == pytest.approx(expected, rel=1e-12)
