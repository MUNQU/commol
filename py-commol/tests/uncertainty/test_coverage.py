"""The coverage harness accepts calibrated intervals and rejects narrow ones,
and the coverage studies of the sequential Monte Carlo methods and of the
conformal prediction intervals.

The harness's model is linear, ``y_j = θ x_j + ε_j`` with ``ε_j ~ N(0, σ²)``
and a flat prior, so the posterior of ``θ`` is ``N(θ̂, σ²/Σx²)`` with
``θ̂ = Σxy/Σx²``. The oracle draws from it; a 90 % interval of its draws
covers ``θ`` in 90 % of the data sets. The Gibbs posterior fits a constant
flow with Gaussian noise (``flow_calibrator``); ABC fits the probability of a
chain binomial from the units that moved in one step.

The Laplace approximation fits the same flow with a normal model of known
``sd``, whose likelihood is exactly normal in ``c``. The sandwich fits a line
through the origin, ``B(t) = c·t``, whose noise grows with ``t``: a normal
model of one ``sd`` is wrong there, and so are its Laplace intervals. The
Laplace mixture fits a flow ``c²``, whose likelihood has the two optima
``±√ȳ``: with a flat prior its central interval runs from the negative mode
to the upper quantile of the positive one, so it holds ``c > 0`` when that
quantile does.

A prediction interval holds a new observation, not a fixed value: a study of
one shifts the interval by the observation, so it holds 0 when it holds the
observation. The posterior predictive of a normal observation model, with
the priors ``c`` uniform and ``sd`` log-uniform, gives the Student-t
prediction interval, which covers a new observation at its level.
"""

import math

import numpy as np
import pytest

from commol import (
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    Simulation,
)
from commol.conformal import EnbPI, JackknifePlus, SplitConformal
from commol.distributions import LogUniform, Priors, Uniform
from commol.local import Laplace, LaplaceMixture, LikelihoodTarget, Sandwich
from commol.observations import DistributionNoise, NormalObservations
from commol.smc import (
    ABCSMC,
    BayesianPosterior,
    BootstrapCoverageEta,
    GibbsPosterior,
    InformationMatchingEta,
    LossDistance,
)
from commol.uncertainty import Predictive, weighted_quantiles
from tests.support import (
    a_to_b_builder,
    bounded,
    build,
    build_stochastic,
    flow_calibrator,
    flow_observations,
)
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


FLOW, FLOW_NOISE, FLOW_COUNT = 10.0, 1.0, 30


@pytest.mark.coverage_study
class TestGibbsPosterior:
    @pytest.mark.parametrize(
        "eta",
        [
            InformationMatchingEta(block_length=1),
            BootstrapCoverageEta(replicates=20, block_length=1, confidence_level=LEVEL),
        ],
        ids=["information_matching", "bootstrap_coverage"],
    )
    def test_gibbs_posterior_is_calibrated(
        self, eta: InformationMatchingEta | BootstrapCoverageEta
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
class TestABCSMC:
    def test_abc_smc_with_a_stochastic_engine_is_calibrated(self) -> None:
        method = ABCSMC(
            priors=Priors.uniform_on_bounds(),
            distance=LossDistance(),
            particles=300,
            target_cess=0.5,
            min_acceptance=0.01,
            stay_probability=0.05,
        )
        model = build_stochastic(
            a_to_b_builder(parameters={"k": None}, amounts={"A": UNITS, "B": 0}),
            "StochasticSteps",
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
                    confidence_level=level,
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
                confidence_level=level,
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
                    confidence_level=level,
                    positions=[61.0],
                )
            )
            series = result.intervals.series["B"]
            return _held(series.lower[0], series.upper[0], float(values[60]))

        study = _conformal_study(make, level)

        assert study.calibrated(), study


@pytest.mark.coverage_study
class TestPosteriorPredictive:
    def test_posterior_predictive_with_noise_is_calibrated(self) -> None:
        level = 0.9
        method = BayesianPosterior(
            priors=Priors(distributions={"c": Uniform(), "sd": LogUniform()}),
            particles=400,
            target_cess=0.5,
            resample_threshold=0.5,
            stay_probability=0.05,
        )

        def make(index: int, rng: np.random.Generator) -> dict[str, Interval]:
            values = FLOW + rng.normal(0.0, FLOW_NOISE, 16)
            calibrator = flow_calibrator(
                values[:15].tolist(),
                windowed=True,
                seed=index,
                observation_models={"B": NormalObservations(sd="sd")},
                noise=[bounded("sd", 0.05, 20.0, kind="noise")],
            )
            sample = calibrator.uncertainty(method).sample
            held_out = flow_observations([float(values[15])], windowed=True, start=16)
            band = (
                Predictive(calibrator, sample, [0, 1])
                .at_observations(held_out, DistributionNoise())
                .band(level)
            )
            return _held(band.lower[0], band.upper[0], float(values[15]))

        study = _conformal_study(make, level)

        assert study.calibrated(), study


@pytest.mark.coverage_study
class TestLaplace:
    def test_laplace_is_calibrated_near_gaussian(self) -> None:
        method = Laplace(draws=2000)

        def interval(index: int, rng: np.random.Generator) -> dict[str, Interval]:
            values = FLOW + rng.normal(0.0, FLOW_NOISE, 20)
            calibrator = flow_calibrator(
                values.tolist(),
                windowed=True,
                seed=index,
                observation_models={"B": NormalObservations(sd=FLOW_NOISE)},
            )
            return {"c": calibrator.uncertainty(method).sample.interval(LEVEL)["c"]}

        study = coverage_study(interval, {"c": FLOW}, LEVEL, data_sets=200, seed=21)

        assert study.calibrated(), study


SLOPE, SLOPE_POSITIONS = 10.0, np.arange(1.0, 31.0)


def _changing_noise_study(method: Laplace | Sandwich, models: bool) -> CoverageStudy:
    def interval(index: int, rng: np.random.Generator) -> dict[str, Interval]:
        noise = 0.02 * SLOPE_POSITIONS**1.5 * rng.normal(0.0, 1.0, SLOPE_POSITIONS.size)
        values = SLOPE * SLOPE_POSITIONS + noise
        calibrator = flow_calibrator(
            values.tolist(),
            windowed=False,
            seed=index,
            observation_models={"B": NormalObservations(sd="sd")} if models else None,
            noise=[bounded("sd", 1e-3, 100.0, kind="noise")] if models else (),
        )
        return {"c": calibrator.uncertainty(method).sample.interval(LEVEL)["c"]}

    return coverage_study(interval, {"c": SLOPE}, LEVEL, data_sets=300, seed=31)


@pytest.mark.coverage_study
class TestSandwich:
    def test_sandwich_holds_under_changing_noise(self) -> None:
        sandwich = _changing_noise_study(
            Sandwich(block_length=1, draws=2000), models=False
        )
        laplace = _changing_noise_study(Laplace(draws=2000), models=True)

        assert sandwich.calibrated(), sandwich
        assert laplace.coverage("c") < LEVEL - laplace.tolerance, laplace


SQUARE_ROOT, SQUARE_COUNT = 2.0, 10


@pytest.mark.coverage_study
class TestLaplaceMixture:
    def test_laplace_mixture_is_calibrated_with_two_optima(self) -> None:
        method = LaplaceMixture(
            starts=8,
            draws=2000,
            target=LikelihoodTarget(priors=Priors.uniform_on_bounds()),
        )
        model = build(
            a_to_b_builder(
                parameters={"c": None}, rate="c * c * A / A", amounts={"A": 1e6, "B": 0}
            )
        )

        def interval(index: int, rng: np.random.Generator) -> dict[str, Interval]:
            values = SQUARE_ROOT**2 + rng.normal(0.0, FLOW_NOISE, SQUARE_COUNT)
            problem = CalibrationProblem(
                observed_data=flow_observations(values.tolist(), windowed=True),
                parameters=[bounded("c", -4.0, 4.0, initial_guess=1.0)],
                observation_models={"B": NormalObservations(sd=FLOW_NOISE)},
                optimization_config=NelderMeadConfig(
                    max_iterations=1000, sd_tolerance=1e-12
                ),
                seed=index,
            )
            result = Calibrator(Simulation(model), problem).uncertainty(method)
            return {"c": result.sample.interval(LEVEL)["c"]}

        study = coverage_study(
            interval, {"c": SQUARE_ROOT}, LEVEL, data_sets=200, seed=41
        )

        assert study.calibrated(), study
