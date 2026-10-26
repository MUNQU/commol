"""
Growth limited by crowding: the logistic equation and an exponential birth law.

``y' = r y (1 - y / K)`` tends to ``K`` from every ``y0 > 0``, with the closed
form ``K y0 / (y0 + (K - y0) e^{-r t})``. ``y' = y (r e^{-y} - d)``, a birth rate
that falls exponentially with crowding, tends to ``log(r / d)`` if ``r > d`` and
to zero if ``r < d``.

References
----------
.. [1] Verhulst, P.-F. (1838). Notice sur la loi que la population suit dans
   son accroissement. *Correspondance Mathematique et Physique*, 10, 113-121.
.. [2] Ricker, W. E. (1954). Stock and recruitment. *Journal of the Fisheries
   Research Board of Canada*, 11, 559-623. https://doi.org/10.1139/f54-039
"""

import math

import pytest

from commol import Simulation
from commol._model.model import Model
from tests.support import ODE_REL, Flow, assert_close, flow_model, settled


@pytest.mark.parametrize("y0", [1, 5])
def test_logistic_growth_tends_to_the_carrying_capacity(y0: float) -> None:
    # [1], with r = 2 and K = 2, from below and from above K.
    r, k = 2.0, 2.0
    flows = [
        Flow("births", None, "Y", "r * Y"),
        Flow("crowding", "Y", None, "r * Y * Y / K"),
    ]
    times = [0.25, 1.0, 3.0, 20.0]

    results = Simulation(flow_model({"Y": y0}, flows, {"r": r, "K": k})).run_at(times)

    expected = [k * y0 / (y0 + (k - y0) * math.exp(-r * t)) for t in times]
    assert_close(results["Y"], expected, rel=ODE_REL)
    assert_close(results["Y"][-1], k, rel=ODE_REL)


def _exponential_births(r: float, d: float) -> Model:
    """``y' = y (r e^{-y} - d)`` from ``y = 1`` [2]_."""
    flows = [
        Flow("births", None, "Y", "r * Y * exp(-Y)"),
        Flow("deaths", "Y", None, "d * Y"),
    ]
    return flow_model({"Y": 1}, flows, {"r": r, "d": d})


def test_exponential_birth_law_tends_to_log_r_over_d() -> None:
    r, d = 0.5 * math.exp(1.5), 0.5

    limit = settled(_exponential_births(r, d), 200)

    assert_close(limit["Y"], math.log(r / d), rel=ODE_REL)


def test_exponential_birth_law_dies_out_when_deaths_win() -> None:
    # r < d: the per capita rate is below -(d - r) = -0.1, so y(400) < e^-40.
    results = Simulation(_exponential_births(0.4, 0.5)).run_at([400])

    assert abs(results["Y"][0]) < 1e-9
