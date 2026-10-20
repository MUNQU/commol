"""Coverage studies of the profile likelihood.

Each study fits synthetic data sets of a decay, ``A(t) = a (1 − k)ᵗ`` at
``t = 1, …, 20`` with normal noise (``a_to_b_builder``, the initial amount
``a`` of A an initial condition), and counts how often the profile interval
of ``k``, ``a`` or the amount of A at 30 holds the truth. The model is
nonlinear in ``k``, so the intervals are those of Wilks' approximation, or of
the bootstrap. The studies carry the marker ``coverage_study``.
"""

import numpy as np
import pytest

from commol import (
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    NormalObservations,
    ObservedDataPoint,
    PredictionProfile,
    PredictionSpec,
    ProfileLikelihood,
    ProfileResult,
    ResidualBootstrap,
    Simulation,
    Threshold,
    UncertaintyMethod,
)
from tests.support import a_to_b_builder, bounded, build
from tests.support.coverage import Interval, coverage_study

K, A0, NOISE, LEVEL = 0.15, 100.0, 2.0, 0.9
POSITIONS = range(1, 21)


def _calibrator(values: np.ndarray, *, likelihood: bool, seed: int) -> Calibrator:
    model = build(a_to_b_builder(parameters={"k": None}, amounts={"A": None, "B": 0.0}))
    problem = CalibrationProblem(
        observed_data=[
            ObservedDataPoint(at=t, output="A", value=float(value))
            for t, value in zip(POSITIONS, values, strict=True)
        ],
        parameters=[
            bounded("k", 0.01, 0.5, initial_guess=0.2),
            bounded("A", 50.0, 150.0, kind="initial_condition", initial_guess=90.0),
        ],
        loss_function=None if likelihood else "sse",
        observation_models={"A": NormalObservations(sd=NOISE)} if likelihood else None,
        optimization_config=NelderMeadConfig(max_iterations=2000, sd_tolerance=1e-12),
        seed=seed,
    )
    return Calibrator(Simulation(model), problem)


def _data(rng: np.random.Generator) -> np.ndarray:
    truth = A0 * (1.0 - K) ** np.asarray(POSITIONS, dtype=float)
    return np.maximum(truth + rng.normal(0.0, NOISE, truth.size), 0.0)


def _intervals(result: ProfileResult) -> dict[str, Interval]:
    return {
        name: (
            -np.inf if interval.lower is None else interval.lower,
            np.inf if interval.upper is None else interval.upper,
        )
        for name, interval in result.intervals.items()
    }


def _study(
    method: UncertaintyMethod[ProfileResult],
    truth: dict[str, float],
    *,
    likelihood: bool,
    data_sets: int = 200,
):
    def run(index: int, rng: np.random.Generator) -> dict[str, Interval]:
        calibrator = _calibrator(_data(rng), likelihood=likelihood, seed=index)
        return _intervals(calibrator.uncertainty(method))

    return coverage_study(run, truth, LEVEL, data_sets=data_sets, seed=29)


@pytest.mark.coverage_study
class TestProfileLikelihood:
    def test_chi_square_intervals_are_calibrated(self) -> None:
        study = _study(
            ProfileLikelihood(
                parameters=["k", "A"], threshold=Threshold.chi_square(LEVEL)
            ),
            {"k": K, "A": A0},
            likelihood=True,
        )

        assert study.calibrated(), study

    def test_bootstrap_calibrated_intervals_are_calibrated(self) -> None:
        bootstrap = ResidualBootstrap(
            replicates=50,
            block_length=1,
            residual_scale="additive",
            scheme="moving",
        )
        study = _study(
            ProfileLikelihood(
                parameters=["k", "A"],
                threshold=Threshold.bootstrap_calibrated(bootstrap, LEVEL),
            ),
            {"k": K, "A": A0},
            likelihood=False,
        )

        assert study.calibrated(), study


@pytest.mark.coverage_study
class TestPredictionProfile:
    def test_chi_square_interval_of_a_prediction_is_calibrated(self) -> None:
        study = _study(
            PredictionProfile(
                specs=[PredictionSpec(at=30, output="A")],
                threshold=Threshold.chi_square(LEVEL),
            ),
            {"A": A0 * (1.0 - K) ** 30},
            likelihood=True,
        )

        assert study.calibrated(), study
