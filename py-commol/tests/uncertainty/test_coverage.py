"""The coverage harness accepts calibrated intervals and rejects narrow ones,
and the coverage studies of the conformal prediction intervals.

The model is linear, ``y_j = θ x_j + ε_j`` with ``ε_j ~ N(0, σ²)`` and a flat
prior, so the posterior of ``θ`` is ``N(θ̂, σ²/Σx²)`` with ``θ̂ = Σxy/Σx²``.
The oracle draws from it; a 90 % interval of its draws covers ``θ`` in 90 % of
the data sets.

A prediction interval holds a new observation, not a fixed value: a study of
one shifts the interval by the observation, so it holds 0 when it holds the
observation.
"""

import math

import numpy as np
import pytest

from commol import EnbPI, JackknifePlus, SplitConformal, weighted_quantiles
from tests.support import flow_calibrator, flow_observations
from tests.support.coverage import CoverageStudy, Interval, StudyMethod, coverage_study

THETA, NOISE, LEVEL = 2.0, 0.5, 0.9
X = np.arange(1.0, 11.0)


def _posterior_interval(rng: np.random.Generator, width: float) -> dict[str, Interval]:
    y = THETA * X + rng.normal(0.0, NOISE, X.size)
    estimate = float(X @ y / (X @ X))
    draws = rng.normal(estimate, NOISE / math.sqrt(X @ X), 2000)
    lower, upper = weighted_quantiles(
        draws.tolist(), [(1 - LEVEL) / 2, (1 + LEVEL) / 2]
    )
    return {
        "theta": (
            estimate - width * (estimate - lower),
            estimate + width * (upper - estimate),
        )
    }


@pytest.mark.coverage_study
class TestHarness:
    def test_an_oracle_of_the_true_posterior_is_calibrated(self) -> None:
        study = coverage_study(
            lambda _, rng: _posterior_interval(rng, 1.0),
            {"theta": THETA},
            LEVEL,
            data_sets=400,
            seed=1,
        )

        assert study.calibrated(), study

    def test_intervals_half_as_wide_are_rejected(self) -> None:
        study = coverage_study(
            lambda _, rng: _posterior_interval(rng, 0.5),
            {"theta": THETA},
            LEVEL,
            data_sets=400,
            seed=1,
        )

        assert not study.calibrated(), study
        assert study.coverage("theta") < LEVEL - study.tolerance


def _held(lower: float, upper: float, observed: float) -> dict[str, Interval]:
    """The interval shifted by the observation it should hold."""
    return {"observation": (lower - observed, upper - observed)}


def _conformal_study(
    make: StudyMethod, level: float, data_sets: int = 200
) -> CoverageStudy:
    return coverage_study(make, {"observation": 0.0}, level, data_sets, seed=11)


@pytest.mark.coverage_study
class TestSplitConformal:
    def test_split_conformal_covers_exchangeable_data(self) -> None:
        level = 0.9

        def make(index: int, rng: np.random.Generator) -> dict[str, Interval]:
            # Flows of one step around c = 10: 30 fitted, 30 held out, 1 new.
            values = 10.0 + rng.normal(0.0, 1.0, 61)
            fitted = values[:30].tolist()
            calibrator = flow_calibrator(fitted, windowed=True, seed=index)
            result = calibrator.uncertainty(
                SplitConformal(
                    calibration_data=flow_observations(
                        values[30:60].tolist(), windowed=True, start=31
                    ),
                    score="absolute",
                    by="series",
                    level=level,
                    positions=[61.0],
                )
            )
            series = result.intervals.series["B"]
            return _held(series.lower[0], series.upper[0], float(values[60]))

        study = _conformal_study(make, level)

        assert study.coverage("observation") >= level - study.tolerance, study


def _jackknife_plus_study(block_length: int, level: float) -> CoverageStudy:
    def make(index: int, rng: np.random.Generator) -> dict[str, Interval]:
        # Flows of one step around c = 10: 30 fitted, 1 new.
        values = 10.0 + rng.normal(0.0, 1.0, 31)
        calibrator = flow_calibrator(values[:30].tolist(), windowed=True, seed=index)
        result = calibrator.uncertainty(
            JackknifePlus(
                block_length=block_length,
                score="absolute",
                by="series",
                level=level,
                positions=[31.0],
            )
        )
        series = result.intervals.series["B"]
        return _held(series.lower[0], series.upper[0], float(values[30]))

    return _conformal_study(make, level)


@pytest.mark.coverage_study
class TestJackknifePlus:
    def test_jackknife_plus_is_calibrated(self) -> None:
        study = _jackknife_plus_study(block_length=1, level=0.9)

        bound = 1 - 2 * 0.1
        assert study.coverage("observation") >= bound - study.tolerance, study

    def test_jackknife_plus_with_long_blocks_is_calibrated(self) -> None:
        study = _jackknife_plus_study(block_length=5, level=0.9)

        # CV+ with K = 6 blocks of m = 5 of n = 30 observations (Barber et
        # al. 2021, Theorem 4): 1 − 2α − min(2(1 − 1/K)/(m + 1), (1 − K/n)/(K + 1)).
        n, k, alpha = 30, 6, 0.1
        bound = (
            1 - 2 * alpha - min(2 * (1 - 1 / k) / (n / k + 1), (1 - k / n) / (k + 1))
        )
        assert study.coverage("observation") >= bound - study.tolerance, study


def _ar1(rng: np.random.Generator, phi: float, count: int) -> np.ndarray:
    noise = rng.normal(0.0, 1.0, count + 50)
    series = np.zeros_like(noise)
    for i in range(1, noise.size):
        series[i] = phi * series[i - 1] + noise[i]
    return series[50:]


@pytest.mark.coverage_study
class TestEnbPI:
    def test_enbpi_holds_under_autocorrelation(self) -> None:
        level = 0.9

        def make(index: int, rng: np.random.Generator) -> dict[str, Interval]:
            # Flows of one step around c = 10 with AR(1) noise: 60 fitted, 1 new.
            values = 10.0 + _ar1(rng, 0.5, 61)
            fitted = values[:60].tolist()
            calibrator = flow_calibrator(fitted, windowed=True, seed=index)
            result = calibrator.uncertainty(
                EnbPI(
                    replicates=40,
                    block_length=5,
                    score="absolute",
                    by="series",
                    level=level,
                    positions=[61.0],
                )
            )
            series = result.intervals.series["B"]
            return _held(series.lower[0], series.upper[0], float(values[60]))

        study = _conformal_study(make, level)

        assert study.calibrated(), study
