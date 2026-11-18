"""Coverage studies of the tests of dependence: their level under
independence, and their power against an AR(1).

`TestValues` reads values with no model: independent normal values whose
standard deviation grows 30 times along 35 positions, against which the
robust portmanteau ``Q̃`` holds its level, and an AR(1) of ``ρ = 0.5`` on
100 positions, which it finds. `TestReferences` reads the residuals of fits
of `smooth_model`, a flow ``exp(Σ c_k T_k(x))`` of 3 or 10 coefficients:
counts of independent Poisson data at 35 positions, read with a Poisson
model (`DistributionNoise`) or with the sum of squares (`NoNoise`); the
`Refitted` reference holds its level at the fit whatever the parameters,
and finds normal AR(1) noise. The rate of `AtParameters` in the same study
is logged for the guide. The studies carry the marker ``coverage_study``.
"""

import logging
import math

import numpy as np
import pytest

from commol import ObservedDataPoint, Simulation
from commol.dependence import (
    AtParameters,
    Refitted,
    residual_dependence,
    series_dependence,
)
from commol.observations import DistributionNoise, NoNoise, PoissonObservations
from commol.uncertainty import Predictive
from tests.support import smooth_calibrator, smooth_model

LEVEL = 0.05
REFITS = 39
"""The refits of a `Refitted` reference: ``(1 + #{Q_b ≥ Q}) / 40`` is at or
below 0.05 when at most one refit reaches ``Q``."""

logger = logging.getLogger(__name__)


def _points(length: int) -> list[ObservedDataPoint]:
    return [ObservedDataPoint(at=t, output="x", value=0.0) for t in range(length)]


def _autoregression(rng: np.random.Generator, rho: float, length: int) -> np.ndarray:
    """A stationary AR(1) of `rho` and of variance 1."""
    innovations = rng.standard_normal(length) * math.sqrt(1 - rho**2)
    series = np.empty(length)
    series[0] = rng.standard_normal()
    for t in range(1, length):
        series[t] = rho * series[t - 1] + innovations[t]
    return series


def _rejects(values: np.ndarray, seed: int) -> bool:
    result = series_dependence(
        list(values),
        _points(len(values)),
        max_lag=7,
        replicates=1,
        confidence_level=0.9,
        seed=seed,
    )
    p_value = result.series[0].p_value
    assert p_value is not None
    return p_value <= LEVEL


@pytest.mark.coverage_study
class TestValues:
    def test_heteroscedastic_values_hold_their_level(self) -> None:
        rng = np.random.default_rng(11)
        sd = 30.0 ** np.linspace(0, 1, 35)

        rate = np.mean(
            [_rejects(sd * rng.standard_normal(35), seed) for seed in range(1000)]
        )

        logger.info("heteroscedastic values: rejected %.3f", rate)
        assert 0.03 <= rate <= 0.08, rate

    def test_dependent_values_are_found(self) -> None:
        rng = np.random.default_rng(12)

        rate = np.mean(
            [_rejects(_autoregression(rng, 0.5, 100), seed) for seed in range(200)]
        )

        logger.info("AR(1) of 0.5 on 100 positions: rejected %.3f", rate)
        assert rate >= 0.8, rate


START = [math.log(20.0), 0.3, -0.2]
"""The coefficients of the truth; those after them are 0."""


def _truth(parameters: int) -> list[float]:
    return (START + [0.0] * parameters)[:parameters]


def _means(parameters: int, length: int) -> np.ndarray:
    """The flow of each step at the truth."""
    model = smooth_model(parameters, length, _truth(parameters))
    return np.diff(np.array(Simulation(model).run(length)["B"]))


def _p_values(
    flows: np.ndarray, parameters: int, counts: bool, seed: int
) -> tuple[float, float]:
    """The p-values of ``Q̃`` of the residuals of the fit to `flows`, against
    `Refitted` and `AtParameters`: of a Poisson model with `counts`, of the
    sum of squares otherwise."""
    calibrator = smooth_calibrator(
        list(flows),
        _truth(parameters),
        observation_models={"B": PoissonObservations()} if counts else None,
        seed=seed,
    )
    predictive = Predictive(calibrator, calibrator.point_sample(calibrator.run()), [1])
    noise = DistributionNoise() if counts else NoNoise()

    def p_value(reference: Refitted | AtParameters) -> float:
        result = residual_dependence(
            predictive,
            calibrator.problem.observed_data,
            noise,
            max_lag=7,
            randomizations=4 if counts else 1,
            reference=reference,
            confidence_level=0.9,
        )
        value = result.series[0].p_value
        assert value is not None
        return value

    return p_value(Refitted(replicates=REFITS)), p_value(
        AtParameters(replicates=REFITS)
    )


@pytest.mark.coverage_study
class TestReferences:
    @pytest.mark.parametrize("counts", [True, False], ids=["poisson", "loss"])
    @pytest.mark.parametrize("parameters", [3, 10])
    def test_a_refitted_reference_holds_at_a_fit(
        self, parameters: int, counts: bool
    ) -> None:
        rng = np.random.default_rng(parameters)
        means = _means(parameters, 35)

        p_values = np.array(
            [
                _p_values(rng.poisson(means), parameters, counts, seed)
                for seed in range(400)
            ]
        )

        refitted, at_parameters = np.mean(p_values <= LEVEL, axis=0)
        logger.info(
            "%d parameters, %s: Refitted %.3f, AtParameters %.3f",
            parameters,
            "Poisson" if counts else "loss",
            refitted,
            at_parameters,
        )
        assert 0.02 <= refitted <= 0.09, (refitted, at_parameters)

    def test_dependent_noise_is_found(self) -> None:
        rng = np.random.default_rng(13)
        means = _means(3, 100)

        rejected = [
            _p_values(means + 2 * _autoregression(rng, 0.5, 100), 3, False, seed)[0]
            <= LEVEL
            for seed in range(100)
        ]

        rate = np.mean(rejected)
        logger.info("AR(1) of 0.5, 100 positions, 3 parameters: rejected %.3f", rate)
        assert rate >= 0.8, rate
