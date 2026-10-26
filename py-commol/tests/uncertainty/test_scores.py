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
    Simulation,
    scores,
)
from commol.observations import (
    DistributionNoise,
    NoNoise,
    PoissonObservations,
    ResidualNoise,
)
from commol.uncertainty import ParameterSample, Predictive, SampleMethod
from tests.support import (
    abc_builder,
    bounded,
    build,
    flow_calibrator,
    flow_observations,
)

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

        score = scores.coverage(predictive, [_observation(y)], NoNoise(), 0.9)

        assert score.per_observation == [expected]

    @pytest.mark.parametrize("offset", [0.0, 2.5, -3.0])
    def test_the_interval_score_is_its_closed_form(
        self, normal: tuple[Predictive, NormalDist], offset: float
    ) -> None:
        predictive, dist = normal
        y = dist.mean + offset * dist.stdev

        score = scores.interval_score(predictive, [_observation(y)], NoNoise(), 0.8)

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

        score = scores.weighted_interval_score(
            predictive, [_observation(y)], NoNoise(), levels
        )

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

        score = scores.crps(predictive, [_observation(y)], NoNoise())

        assert score.mean == pytest.approx(expected, rel=1e-3)

    @pytest.mark.parametrize("offset", [-1.0, 0.0, 1.5])
    def test_the_pit_is_the_normal_distribution_function(
        self, normal: tuple[Predictive, NormalDist], offset: float
    ) -> None:
        predictive, dist = normal

        score = scores.pit(
            predictive,
            [_observation(dist.mean + offset * dist.stdev)],
            NoNoise(),
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

        score = scores.coverage(predictive, [inside, outside, inside], NoNoise(), 0.9)

        assert score.per_observation == [1.0, 0.0, 1.0]
        assert score.per_series == {"B": 1.0, "far": 0.0}
        assert score.mean == pytest.approx(2 / 3)

    def test_the_sample_crps_of_a_known_ensemble(
        self, normal: tuple[Predictive, NormalDist]
    ) -> None:
        predictive, dist = normal
        calibrator = predictive.source
        assert isinstance(calibrator, Calibrator)
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

        score = scores.crps(
            Predictive(calibrator, sample, [10]), [_observation(y)], NoNoise()
        )

        assert score.mean == pytest.approx(expected, rel=1e-12)


class TestPit:
    def test_the_pit_takes_the_seed_of_the_source(self) -> None:
        def calibrator(seed: int) -> Calibrator:
            simulation = Simulation(
                build(abc_builder(parameters={"k1": 0.1, "k2": 0.05}))
            )
            problem = CalibrationProblem(
                observed_data=[_observation(1.0)],
                parameters=[bounded("f", 0.01, 10.0, kind="factor")],
                loss_function="sse",
                optimization_config=NelderMeadConfig(max_iterations=1),
                seed=seed,
            )
            return Calibrator(simulation, problem)

        # Two values of f, so every draw ties with the observations of one.
        sample = ParameterSample.equally_weighted(
            ["f"],
            [(0.01, 10.0)],
            [[1.0], [2.0]] * 4,
            SampleMethod(name="two"),
        )
        draws = Predictive(calibrator(1), sample, [10]).at_observations(
            [_observation(1.0)], NoNoise()
        )
        observations = [_observation(draws.values[0][0])] * 20

        def pit(seed: int) -> list[float]:
            predictive = Predictive(calibrator(seed), sample, [10])
            return scores.pit(predictive, observations, NoNoise()).per_observation

        first = pit(1)
        # F(y⁻) = 0 and F(y) = 1/2: each value is uniform in between.
        assert all(0.0 <= value <= 0.5 for value in first)
        assert len(set(first)) == len(first)
        assert pit(1) == first
        other = pit(2)
        assert all(a != b for a, b in zip(first, other, strict=True))


class TestLogScore:
    def test_log_score_of_one_draw_is_the_log_density(self) -> None:
        calibrator = flow_calibrator(
            [3.0, 7.0], windowed=True, observation_models={"B": PoissonObservations()}
        )
        sample = ParameterSample.equally_weighted(
            ["c"], [(0.0, 50.0)], [[5.0]], SampleMethod(name="point")
        )
        observations = flow_observations([3.0, 7.0], windowed=True)

        score = scores.log_score(
            Predictive(calibrator, sample, [0, 1]), observations, DistributionNoise()
        )

        expected = [y * math.log(5.0) - 5.0 - math.lgamma(y + 1) for y in (3.0, 7.0)]
        assert score.per_observation == pytest.approx(expected, rel=1e-12)
        assert score.per_series == {"B": pytest.approx(sum(expected) / 2)}

    @pytest.mark.parametrize("noise", ["none", "residual"])
    def test_log_score_needs_distribution_noise(
        self, normal: tuple[Predictive, NormalDist], noise: str
    ) -> None:
        predictive, _ = normal
        calibrator = predictive.source
        assert isinstance(calibrator, Calibrator)
        other = (
            NoNoise()
            if noise == "none"
            else ResidualNoise(
                fit=calibrator.run(),
                block_length=1,
                residual_scale="additive",
            )
        )

        with pytest.raises(TypeError, match="takes DistributionNoise"):
            scores.log_score(predictive, [_observation(1.0)], other)  # type: ignore[arg-type]
