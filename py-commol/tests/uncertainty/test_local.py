"""Local approximations through ``Calibrator.uncertainty``: the Laplace
approximation and the delta method, the sandwich, the Laplace mixture, and
information matching from the curvature.

Most tests fit a constant flow ``c`` (``flow_calibrator``): with windowed
observations each prediction is ``c``, so with a normal model of known ``sd``
the information of ``c`` is ``n / sd²``; without windows the prediction at
``t`` is ``c·t``, a line through the origin.
"""

import numpy as np
import pytest

from commol import Laplace, NormalObservations
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
