"""Local approximations through ``Calibrator.uncertainty``: the Laplace
approximation and the delta method, the sandwich, the Laplace mixture, and
information matching from the curvature.

Most tests fit a constant flow ``c`` (``flow_calibrator``): with windowed
observations each prediction is ``c``, so with a normal model of known ``sd``
the information of ``c`` is ``n / sd²``; without windows the prediction at
``t`` is ``c·t``, a line through the origin.
"""

import math

import numpy as np
import pytest

from commol import Laplace, LaplaceResult, NormalObservations, Predictive
from tests.support import flow_calibrator


def _normal_data(count: int, seed: int, sd: float, level: float = 10.0) -> list[float]:
    rng = np.random.default_rng(seed)
    return (level + rng.normal(0.0, sd, count)).tolist()


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
