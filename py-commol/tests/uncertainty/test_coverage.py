"""The coverage harness accepts calibrated intervals and rejects narrow ones.

The model is linear, ``y_j = θ x_j + ε_j`` with ``ε_j ~ N(0, σ²)`` and a flat
prior, so the posterior of ``θ`` is ``N(θ̂, σ²/Σx²)`` with ``θ̂ = Σxy/Σx²``.
The oracle draws from it; a 90 % interval of its draws covers ``θ`` in 90 % of
the data sets.
"""

import math

import numpy as np
import pytest

from commol import weighted_quantiles
from tests.support.coverage import Interval, coverage_study

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
