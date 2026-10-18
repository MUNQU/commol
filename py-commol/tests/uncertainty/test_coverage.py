"""The coverage harness, and the coverage studies of the sequential Monte
Carlo methods.

The harness's model is linear, ``y_j = θ x_j + ε_j`` with ``ε_j ~ N(0, σ²)``
and a flat prior, so the posterior of ``θ`` is ``N(θ̂, σ²/Σx²)`` with
``θ̂ = Σxy/Σx²``. The oracle draws from it; a 90 % interval of its draws
covers ``θ`` in 90 % of the data sets. The Gibbs posterior fits a constant
flow with Gaussian noise (``flow_calibrator``); ABC fits the probability of a
chain binomial from the units that moved in one step.
"""

import math

import numpy as np
import pytest

from commol import (
    AbcSmc,
    CalibrationProblem,
    Calibrator,
    EtaCalibration,
    GibbsPosterior,
    LossDistance,
    NelderMeadConfig,
    ObservedDataPoint,
    Priors,
    Simulation,
    weighted_quantiles,
)
from commol.context.inference.smc import BootstrapCoverage, InformationMatching
from tests.support import a_to_b_builder, bounded, build_stochastic, flow_calibrator
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


FLOW, FLOW_NOISE, FLOW_COUNT = 10.0, 1.0, 30


@pytest.mark.coverage_study
class TestGibbsPosterior:
    @pytest.mark.parametrize(
        "eta",
        [
            EtaCalibration.information_matching(),
            EtaCalibration.bootstrap_coverage(
                replicates=20, block_length=1, confidence_level=LEVEL
            ),
        ],
        ids=["information_matching", "bootstrap_coverage"],
    )
    def test_gibbs_posterior_is_calibrated(
        self, eta: InformationMatching | BootstrapCoverage
    ) -> None:
        method = GibbsPosterior(
            priors=Priors.uniform_on_bounds(),
            eta=eta,
            particles=400,
            target_cess=0.5,
            resample_threshold=0.5,
            stay_probability=0.05,
        )

        def interval(index: int, rng: np.random.Generator) -> dict[str, Interval]:
            values = FLOW + rng.normal(0.0, FLOW_NOISE, FLOW_COUNT)
            calibrator = flow_calibrator(values.tolist(), windowed=True, seed=index)
            return {"c": calibrator.uncertainty(method).sample.interval(LEVEL)["c"]}

        study = coverage_study(interval, {"c": FLOW}, LEVEL, data_sets=200, seed=3)

        assert study.calibrated(), study


UNITS, PROBABILITY = 200, 0.3


@pytest.mark.coverage_study
class TestAbcSmc:
    def test_abc_smc_with_a_stochastic_engine_is_calibrated(self) -> None:
        method = AbcSmc(
            priors=Priors.uniform_on_bounds(),
            distance=LossDistance(),
            particles=300,
            target_cess=0.5,
            min_acceptance=0.01,
            stay_probability=0.05,
        )
        model = build_stochastic(
            a_to_b_builder(parameters={"k": None}, total=UNITS), "StochasticSteps"
        )

        def interval(index: int, rng: np.random.Generator) -> dict[str, Interval]:
            moved = float(rng.binomial(UNITS, PROBABILITY))
            problem = CalibrationProblem(
                observed_data=[ObservedDataPoint(at=1, output="B", value=moved)],
                parameters=[bounded("k", 0.0, 1.0, initial_guess=0.5)],
                loss_function="mae",
                replicates=1,
                replicate_summary="mean",
                optimization_config=NelderMeadConfig(max_iterations=10),
                seed=index,
            )
            result = Calibrator(Simulation(model), problem).uncertainty(method)
            return {"k": result.sample.interval(LEVEL)["k"]}

        study = coverage_study(
            interval, {"k": PROBABILITY}, LEVEL, data_sets=200, seed=11
        )

        assert study.calibrated(), study
