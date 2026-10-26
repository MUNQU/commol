"""Coverage studies of the resampling methods.

Each study fits synthetic data sets made from a known flow ``c`` with
``flow_calibrator`` and counts how often the interval of ``c`` (or of a
held-out observation) holds the truth. They carry the marker
``coverage_study``.
"""

from collections.abc import Callable

import numpy as np
import pytest

from commol._calibration.method import UncertaintyMethod
from commol.observations import NoNoise, ResidualNoise
from commol.resampling import BayesianBootstrap, ResidualBootstrap, WildBootstrap
from commol.resampling.results import BootstrapResult
from commol.uncertainty import Predictive
from tests.support import flow_calibrator, flow_observations
from tests.support.coverage import Interval, coverage_study

C = 10.0


def _interval(
    values: np.ndarray,
    method: UncertaintyMethod[BootstrapResult],
    *,
    windowed: bool,
    level: float,
    seed: int,
) -> dict[str, Interval]:
    calibrator = flow_calibrator(values.tolist(), windowed=windowed, seed=seed)
    result = calibrator.uncertainty(method)
    return {"c": result.sample.interval(level)["c"]}


def _ar1(rng: np.random.Generator, phi: float, count: int) -> np.ndarray:
    noise = rng.normal(0.0, 1.0, count + 50)
    series = np.zeros_like(noise)
    for i in range(1, noise.size):
        series[i] = phi * series[i - 1] + noise[i]
    return series[50:]


def _study(
    make: Callable[[int, np.random.Generator], dict[str, Interval]],
    level: float,
    data_sets: int = 200,
):
    return coverage_study(make, {"c": C}, level, data_sets=data_sets, seed=17)


@pytest.mark.coverage_study
class TestBayesianBootstrap:
    @pytest.mark.parametrize("level", [0.9, 0.95])
    def test_bayesian_bootstrap_is_calibrated(self, level: float) -> None:
        method = BayesianBootstrap(replicates=200, block_length=1)

        study = _study(
            lambda i, rng: _interval(
                C + rng.normal(0.0, 1.0, 80), method, windowed=True, level=level, seed=i
            ),
            level,
        )

        assert study.calibrated(), study


@pytest.mark.coverage_study
class TestResidualBlockBootstrap:
    def _method(self, block_length: int) -> ResidualBootstrap:
        return ResidualBootstrap(
            replicates=200,
            block_length=block_length,
            residual_scale="additive",
            scheme="moving",
        )

    def test_residual_block_bootstrap_holds_under_autocorrelation(self) -> None:
        # A long series: on 200 positions the block version still covers
        # about 0.8, since block bootstraps underestimate the variance by a
        # term of order 1/block_length on short series.
        def make(block_length: int):
            return lambda i, rng: _interval(
                C + _ar1(rng, 0.5, 400),
                self._method(block_length),
                windowed=True,
                level=0.9,
                seed=i,
            )

        blocks = _study(make(10), 0.9)
        single = _study(make(1), 0.9)

        assert blocks.calibrated(), blocks
        assert single.coverage("c") < 0.9 - single.tolerance, single


@pytest.mark.coverage_study
class TestWildBootstrap:
    def test_wild_bootstrap_holds_under_changing_noise(self) -> None:
        # y = c·t with a noise standard deviation of 0.5·t: the size of the
        # noise grows with the model value.
        t = np.arange(1.0, 41.0)

        def make(method: UncertaintyMethod[BootstrapResult]):
            return lambda i, rng: _interval(
                C * t + rng.normal(0.0, 0.5 * t),
                method,
                windowed=False,
                level=0.9,
                seed=i,
            )

        wild = _study(
            make(
                WildBootstrap(replicates=200, block_length=1, multipliers="rademacher")
            ),
            0.9,
        )
        residual = _study(
            make(
                ResidualBootstrap(
                    replicates=200,
                    block_length=1,
                    residual_scale="additive",
                    scheme="moving",
                )
            ),
            0.9,
        )

        assert wild.calibrated(), wild
        assert residual.coverage("c") < 0.9 - residual.tolerance, residual


@pytest.mark.coverage_study
class TestResidualPredictionInterval:
    def test_residual_prediction_interval_is_calibrated(self) -> None:
        # The interval of a held-out observation y, shifted by y: it holds 0
        # when it holds the observation.
        def make(noisy: bool):
            def interval(i: int, rng: np.random.Generator) -> dict[str, Interval]:
                values = C + rng.normal(0.0, 1.0, 61)
                calibrator = flow_calibrator(
                    values[:60].tolist(), windowed=True, seed=i
                )
                result = calibrator.uncertainty(
                    BayesianBootstrap(replicates=200, block_length=1)
                )
                held_out = flow_observations([values[60]], windowed=True, start=61)
                noise = (
                    ResidualNoise(
                        fit=result.fit, block_length=1, residual_scale="additive"
                    )
                    if noisy
                    else NoNoise()
                )
                band = (
                    Predictive(calibrator, result.sample, [1.0])
                    .at_observations(held_out, noise)
                    .band(0.9)
                )
                return {"c": (band.lower[0] - values[60], band.upper[0] - values[60])}

            return interval

        prediction = coverage_study(make(True), {"c": 0.0}, 0.9, data_sets=200, seed=3)
        model = coverage_study(make(False), {"c": 0.0}, 0.9, data_sets=200, seed=3)

        assert prediction.calibrated(), prediction
        assert model.coverage("c") < 0.9 - model.tolerance, model
