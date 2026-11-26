"""Coverage studies of the tests of dependence: their level under
independence, and their power against an AR(1).

`TestValues` reads values with no model, whose robust portmanteau ``Q̃``
is read against 199 permutations of the values of each series:
independent normal and ``t₃`` values of 6 to 100 positions, with gaps or
without, at ``max_lag`` 1, 4 and the last lag of 4 pairs, and skewed
values, against which ``Q̃`` holds 4–6 %; values 0 or 1, against which it
rejects less; normal values whose standard deviation grows 30 times,
counts of a growing mean less their mean and an ARCH(1), against which it
holds 3–8 %; independent values with ``max_lag`` at the last lag of a
short grid, where the lags of fewer than 4 pairs leave ``Q̃``; and an
AR(1) of ``ρ = 0.5`` on 100 positions, which it finds. Each band holds
within 1.5 standard errors of a rate of 5 %. `TestReferences` reads the
residuals of fits of `smooth_model`, a flow ``exp(Σ c_k T_k(x))`` of 3 or
10 coefficients: counts of independent Poisson data at 35 positions, read
with a Poisson model (`DistributionNoise`) or with the sum of squares
(`NoNoise`); the `Refitted` reference holds its level at the fit whatever
the parameters, and finds normal AR(1) noise. The rate of `AtParameters` in
the same study is logged for the guide. The studies carry the marker
``coverage_study``.
"""

import logging
import math
from collections.abc import Callable

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
REPLICATES = 199
"""The replicates of `series_dependence`: ``(1 + #{Q_b ≥ Q}) / 200`` is at or
below 0.05 when at most 9 of them reach ``Q``."""
REFITS = 39
"""The refits of a `Refitted` reference: ``(1 + #{Q_b ≥ Q}) / 40`` is at or
below 0.05 when at most one refit reaches ``Q``."""

logger = logging.getLogger(__name__)


GAPS_ON_35 = [0, 1, 3, 4, 6, 9, 10, 13, 15, 16, 19, 21, 22, 25, 27, 28, 30, 32, 34]
"""19 positions on a grid of 35 steps."""
GAPS_ON_24 = [0, 1, 2, 4, 5, 7, 8, 11, 12, 13, 16, 18, 20, 23]
"""14 positions on a grid of 24 steps."""


def _autoregression(rng: np.random.Generator, rho: float, length: int) -> np.ndarray:
    """A stationary AR(1) of `rho` and of variance 1."""
    innovations = rng.standard_normal(length) * math.sqrt(1 - rho**2)
    series = np.empty(length)
    series[0] = rng.standard_normal()
    for t in range(1, length):
        series[t] = rho * series[t - 1] + innovations[t]
    return series


def _rejects(
    values: np.ndarray,
    seed: int,
    positions: list[int] | None = None,
    max_lag: int = 7,
) -> bool:
    at = range(len(values)) if positions is None else positions
    result = series_dependence(
        list(values),
        [ObservedDataPoint(at=t, output="x", value=0.0) for t in at],
        max_lag=max_lag,
        replicates=REPLICATES,
        confidence_level=0.9,
        seed=seed,
    )
    p_value = result.series[0].p_value
    assert p_value is not None
    return p_value <= LEVEL


def _gappy(values: int, gaps: float) -> list[int]:
    """`values` positions of a grid of which a share `gaps` is missing, drawn
    once with a seed of their count, the first and the last point kept."""
    if not gaps:
        return list(range(values))
    grid = round(values / (1 - gaps))
    rng = np.random.default_rng(values)
    inner = rng.choice(np.arange(1, grid - 1), size=values - 2, replace=False)
    return sorted([0, grid - 1, *(int(p) for p in inner)])


def _lags(positions: list[int]) -> list[int]:
    """The lags with 4 pairs or more on `positions`."""
    at = set(positions)
    return [
        lag
        for lag in range(1, positions[-1] + 1)
        if sum(p + lag in at for p in positions) >= 4
    ]


SERIES = 10
"""The series of one call of `series_dependence` in a study of the level:
each has its own label and its own test."""


def _rejection_rate(
    draw: Callable[[np.random.Generator, int], np.ndarray],
    positions: list[int],
    max_lag: int,
    sets: int,
    seed: int,
) -> float:
    """The share of `sets` series of `draw` at `positions` whose ``Q̃`` is
    rejected at 5 % with `max_lag`."""
    rng = np.random.default_rng(seed)
    observations = [
        ObservedDataPoint(at=t, output=f"x{s}", value=0.0)
        for s in range(SERIES)
        for t in positions
    ]
    p_values = []
    for call in range(sets // SERIES):
        values = np.concatenate([draw(rng, len(positions)) for _ in range(SERIES)])
        result = series_dependence(
            list(values),
            observations,
            max_lag=max_lag,
            replicates=REPLICATES,
            confidence_level=0.9,
            seed=call,
        )
        p_values.extend(series.p_value for series in result.series)
    assert None not in p_values
    return float(np.mean(np.array(p_values) <= LEVEL))


def _holds(rate: float, low: float, high: float, sets: int) -> bool:
    """Whether `rate`, of `sets` series, is in ``[low, high]`` within 1.5
    standard errors of a rate of 5 %, as the study of the engine reads it."""
    slack = 1.5 * math.sqrt(LEVEL * (1 - LEVEL) / sets)
    return low - slack <= rate <= high + slack


def _normal(rng: np.random.Generator, length: int) -> np.ndarray:
    return rng.standard_normal(length)


def _t3(rng: np.random.Generator, length: int) -> np.ndarray:
    return rng.standard_t(3, length)


@pytest.mark.coverage_study
class TestValues:
    def test_the_robust_portmanteau_holds_its_level_at_every_max_lag(self) -> None:
        # 6000 series per design: a rate of 5 % has a standard error of 0.28
        # points.
        cells = []
        for draw in (_normal, _t3):
            for values in (6, 12, 35, 100):
                for gaps in (0.0, 0.3, 0.5):
                    positions = _gappy(values, gaps)
                    lags = _lags(positions)
                    if not lags:
                        continue
                    # max_lag 1, 4 and the last lag, each at least the first
                    # lag in the sum of Q̃.
                    first, last = lags[0], lags[-1]
                    for max_lag in sorted({first, min(max(4, first), last), last}):
                        rate = _rejection_rate(
                            draw, positions, max_lag, 6000, 1000 * values + max_lag
                        )
                        cells.append((draw.__name__, values, gaps, max_lag, rate))
                        logger.info(
                            "%s, %d values, gaps %.1f, max_lag %d: rejected %.4f",
                            *cells[-1],
                        )

        wrong = [cell for cell in cells if not _holds(cell[4], 0.04, 0.06, 6000)]
        assert not wrong, wrong

    @pytest.mark.parametrize("draw", ["chi_square", "lognormal"])
    @pytest.mark.parametrize(("values", "max_lag"), [(35, 7), (100, 17)])
    def test_skewed_values_hold_their_level(
        self, draw: str, values: int, max_lag: int
    ) -> None:
        def skewed(rng: np.random.Generator, length: int) -> np.ndarray:
            if draw == "chi_square":
                return rng.standard_normal(length) ** 2
            return rng.lognormal(0.0, 1.0, length)

        rate = _rejection_rate(skewed, list(range(values)), max_lag, 4000, 16)

        logger.info("%s, %d values: rejected %.4f", draw, values, rate)
        assert _holds(rate, 0.04, 0.06, 4000), rate

    def test_equal_values_reject_less_than_the_level(self) -> None:
        # Values 0 or 1: many permutations give the observed Q̃, and a tie
        # counts as a replicate that reaches it.
        def binary(rng: np.random.Generator, length: int) -> np.ndarray:
            return (rng.random(length) < 0.3).astype(float)

        rate = _rejection_rate(binary, list(range(35)), 7, 4000, 17)

        logger.info("values 0 or 1: rejected %.4f", rate)
        assert _holds(rate, 0.0, 0.06, 4000), rate

    @pytest.mark.parametrize(
        ("values", "gaps", "max_lag"), [(20, 0.0, 7), (35, 0.3, 7), (100, 0.5, 7)]
    )
    def test_heteroscedastic_values_hold_their_level(
        self, values: int, gaps: float, max_lag: int
    ) -> None:
        # The standard deviation grows 30 times along the grid. The
        # permutations hold 3–8 % from 20 values without gaps, 35 with 30 %
        # gaps and 100 with 50 %. Below, they reject up to about 10 % without
        # gaps and up to about 18 % with them.
        positions = _gappy(values, gaps)
        sd = 30.0 ** (np.array(positions) / positions[-1])

        def draw(rng: np.random.Generator, length: int) -> np.ndarray:
            return sd[:length] * rng.standard_normal(length)

        rate = _rejection_rate(draw, positions, max_lag, 4000, 11)

        logger.info("heteroscedastic values: rejected %.4f", rate)
        assert _holds(rate, 0.03, 0.08, 4000), rate

    def test_skewed_heteroscedastic_values_hold_their_level(self) -> None:
        # Counts of a mean that grows from 1 to 40, less their mean.
        mean = np.linspace(1.0, 40.0, 35)

        def draw(rng: np.random.Generator, length: int) -> np.ndarray:
            return rng.poisson(mean[:length]) - mean[:length]

        rate = _rejection_rate(draw, list(range(35)), 7, 4000, 15)

        logger.info("skewed heteroscedastic values: rejected %.4f", rate)
        assert _holds(rate, 0.03, 0.08, 4000), rate

    def test_a_variance_of_the_past_holds_the_level(self) -> None:
        # An ARCH(1) of 0.7: uncorrelated values whose variance follows the
        # last value.
        def draw(rng: np.random.Generator, length: int) -> np.ndarray:
            values = np.empty(length)
            last = 0.0
            for t in range(length):
                last = math.sqrt(0.2 + 0.7 * last**2) * rng.standard_normal()
                values[t] = last
            return values

        rate = _rejection_rate(draw, list(range(100)), 7, 2000, 18)

        logger.info("ARCH(1): rejected %.4f", rate)
        assert _holds(rate, 0.03, 0.08, 2000), rate

    def test_lags_with_few_pairs_keep_the_level(self) -> None:
        # max_lag at the last lag of each grid: the lags of fewer than 4
        # pairs leave Q̃ and its degrees of freedom.
        rng = np.random.default_rng(14)
        designs = [(list(range(12)), 11), (GAPS_ON_35, 34), (GAPS_ON_24, 23)]

        rates = [
            float(
                np.mean(
                    [
                        _rejects(
                            rng.standard_normal(len(positions)),
                            seed,
                            positions,
                            max_lag,
                        )
                        for seed in range(3000)
                    ]
                )
            )
            for positions, max_lag in designs
        ]

        logger.info(
            "max_lag at the last lag: rejected %s", ", ".join(f"{r:.3f}" for r in rates)
        )
        assert all(_holds(rate, 0.04, 0.06, 3000) for rate in rates), rates

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
