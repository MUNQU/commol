"""Local approximations through ``Calibrator.uncertainty``: the Laplace
approximation and the delta method, the sandwich, the Laplace mixture, and
information matching from the curvature.

Most tests fit a constant flow ``c`` (``flow_calibrator``): with windowed
observations each prediction is ``c``, so with a normal model of known ``sd``
the information of ``c`` is ``n / sd²``; without windows the prediction at
``t`` is ``c·t``, a line through the origin.
"""

import logging
import math

import numpy as np
import pytest
from pydantic import BaseModel

from commol import (
    CalibrationProblem,
    Calibrator,
    GibbsTarget,
    ImportanceResult,
    Laplace,
    LaplaceMixture,
    LaplaceResult,
    LikelihoodTarget,
    NelderMeadConfig,
    NormalObservations,
    Predictive,
    Priors,
    Sandwich,
    SandwichResult,
    Simulation,
    UncertaintyMethod,
)
from tests.support import (
    a_to_b_builder,
    bounded,
    build,
    flow_calibrator,
    flow_observations,
)


def _normal_data(count: int, seed: int, sd: float, level: float = 10.0) -> list[float]:
    rng = np.random.default_rng(seed)
    return (level + rng.normal(0.0, sd, count)).tolist()


def _square_calibrator(values: list[float], low: float, high: float) -> Calibrator:
    """A flow ``c²`` from A to B, with a normal model of ``sd = 1``: two
    optima, ``±√ȳ``, when the bounds of ``c`` hold both."""
    model = build(
        a_to_b_builder(
            parameters={"c": None}, rate="c * c * A / A", amounts={"A": 1e6, "B": 0}
        )
    )
    problem = CalibrationProblem(
        observed_data=flow_observations(values, windowed=True),
        parameters=[bounded("c", low, high, initial_guess=1.0)],
        observation_models={"B": NormalObservations(sd=1.0)},
        optimization_config=NelderMeadConfig(max_iterations=1000, sd_tolerance=1e-12),
        seed=2,
    )
    return Calibrator(Simulation(model), problem)


class TestCurvature:
    def test_a_fit_on_a_bound_is_refused(self) -> None:
        # Values below 5 put the fit of c on its lower bound.
        calibrator = flow_calibrator(
            [1.0, 2.0, 1.0],
            windowed=True,
            bounds=(5.0, 50.0),
            observation_models={"B": NormalObservations(sd=1.0)},
        )

        with pytest.raises(
            ValueError, match=r"'c' is on its bound 5.*profile likelihood"
        ):
            calibrator.uncertainty(Laplace(draws=10))


class TestLaplace:
    def test_delta_band_of_a_linear_model_equals_the_draws(self) -> None:
        # B(t) = c·t and A(t) = 10⁶ − c·t: the band of the draws is the
        # linearized band, up to the Monte Carlo error of its quantiles.
        draws, level = 4000, 0.9
        values = [10.0 * t + noise for t, noise in enumerate([0.3, -0.5, 0.2] * 4, 1)]
        calibrator = flow_calibrator(
            values,
            windowed=False,
            observation_models={"B": NormalObservations(sd=1.0)},
        )
        result = calibrator.uncertainty(Laplace(draws=draws))
        positions = [float(t) for t in range(13)]

        band = result.delta_band(positions, level)
        drawn = Predictive(calibrator, result.sample, positions).band(level)

        # The error of a quantile at 0.05 of n normal draws, in standard
        # deviations: √(p(1 − p)/n) / φ(z_p).
        quantile_error = math.sqrt(0.05 * 0.95 / draws) / 0.10313
        sd_c = 1.0 / math.sqrt(sum(t * t for t in range(1, 13)))
        for output in ("A", "B"):
            for t, (lower, upper, low, high) in enumerate(
                zip(
                    band.lower[output],
                    band.upper[output],
                    drawn.lower[output],
                    drawn.upper[output],
                    strict=True,
                )
            ):
                tolerance = 4 * quantile_error * sd_c * t + 1e-9
                assert lower == pytest.approx(low, abs=tolerance), (output, t)
                assert upper == pytest.approx(high, abs=tolerance), (output, t)
        half = 1.6449 * sd_c * 12
        assert band.upper["B"][12] - band.center["B"][12] == pytest.approx(
            half, rel=1e-3
        )

    def test_a_result_read_from_json_has_no_delta_band(self) -> None:
        calibrator = flow_calibrator(
            [9.0, 11.0, 10.0],
            windowed=True,
            observation_models={"B": NormalObservations(sd=1.0)},
        )
        result = calibrator.uncertainty(Laplace(draws=10))
        read = LaplaceResult.model_validate_json(result.model_dump_json())

        assert read.covariance == result.covariance
        with pytest.raises(ValueError, match="needs the calibration problem"):
            read.delta_band([1.0], 0.9)


class TestSandwich:
    def test_sandwich_equals_laplace_under_gaussian_noise(self) -> None:
        count = 200
        calibrator = flow_calibrator(
            _normal_data(count, 3, sd=1.0),
            windowed=True,
            observation_models={"B": NormalObservations(sd=1.0)},
        )

        laplace = calibrator.uncertainty(Laplace(draws=10))
        sandwich = calibrator.uncertainty(Sandwich(block_length=1, draws=10))

        # B/H is the mean of the squared residuals over sd²: its relative
        # error is about √(2/n).
        assert sandwich.covariance[0][0] == pytest.approx(
            laplace.covariance[0][0], rel=3 * math.sqrt(2 / count)
        )
        assert sandwich.blocks == count

    def test_sandwich_refuses_too_few_blocks(self) -> None:
        calibrator = flow_calibrator(_normal_data(6, 4, sd=1.0), windowed=True)

        with pytest.raises(
            ValueError, match=r"block_length 6 makes 1 block for 1 parameter.*shorter"
        ):
            calibrator.uncertainty(Sandwich(block_length=6, draws=10))

    def test_sandwich_refuses_the_absolute_error(self) -> None:
        model = build(
            a_to_b_builder(
                parameters={"c": None}, rate="c * A / A", amounts={"A": 1e6, "B": 0}
            )
        )
        problem = CalibrationProblem(
            observed_data=flow_observations(_normal_data(10, 5, sd=1.0), windowed=True),
            parameters=[bounded("c", 0.0, 50.0, initial_guess=25.0)],
            loss_function="mae",
            optimization_config=NelderMeadConfig(max_iterations=500),
            seed=1,
        )
        calibrator = Calibrator(Simulation(model), problem)

        with pytest.raises(ValueError, match=r"density of the residuals.*bootstrap"):
            calibrator.uncertainty(Sandwich(block_length=1, draws=10))


class TestResults:
    @pytest.mark.parametrize(
        ("method", "result_type"),
        [
            (Laplace(draws=10), LaplaceResult),
            (Sandwich(block_length=2, draws=10), SandwichResult),
            (
                LaplaceMixture(
                    starts=4,
                    draws=100,
                    target=LikelihoodTarget(priors=Priors.uniform_on_bounds()),
                ),
                ImportanceResult,
            ),
            (
                LaplaceMixture(
                    starts=4,
                    draws=100,
                    target=GibbsTarget(priors=Priors.uniform_on_bounds(), eta=0.5),
                ),
                ImportanceResult,
            ),
        ],
        ids=["laplace", "sandwich", "mixture_of_a_likelihood", "mixture_of_a_loss"],
    )
    def test_each_method_returns_its_result(
        self, method: UncertaintyMethod[BaseModel], result_type: type[BaseModel]
    ) -> None:
        data = _normal_data(12, 6, sd=1.0)
        models = (
            None
            if isinstance(method, LaplaceMixture)
            and isinstance(method.target, GibbsTarget)
            else {"B": NormalObservations(sd=1.0)}
        )
        calibrator = flow_calibrator(data, windowed=True, observation_models=models)

        result = calibrator.uncertainty(method)

        assert type(result) is result_type
        assert result.sample.names == ["c"]
        mean = float(np.mean(data))
        assert result.sample.mean()["c"] == pytest.approx(mean, abs=1.0)

    def test_an_end_on_a_bound_gives_no_component(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # c² near 4 has the optima ±2; the bound 1.5 holds the positive one.
        calibrator = _square_calibrator([4.2, 3.8, 4.1, 3.9], -4.0, 1.5)
        method = LaplaceMixture(
            starts=8,
            draws=200,
            target=LikelihoodTarget(priors=Priors.uniform_on_bounds()),
        )

        with caplog.at_level(logging.WARNING):
            result = calibrator.uncertainty(method)

        assert len(result.components) == 1
        assert result.components[0].center["c"] == pytest.approx(-2.0, abs=0.1)
        assert any("'c' is on its bound" in note for note in result.notes)
        assert "'c' is on its bound" in caplog.text
