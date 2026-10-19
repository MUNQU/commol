"""
Quadratic decay: ``y' = -k y^2``, the law of mass action for ``2A -> products``.

With ``k = 1`` every solution is ``y = 1 / (t + 1 / y0)``, and ``y0 = 0`` gives
the zero solution.

References
----------
.. [1] Guldberg, C. M., & Waage, P. (1864). Studier over affiniteten.
   *Forhandlinger i Videnskabs-Selskabet i Christiania*, 35-45.
"""

import pytest

from commol import Simulation
from tests.support import ODE_REL, Flow, assert_close, flow_model

TIMES = [0.5, 1.0, 4.0, 20.0]


def _decay(y0: float) -> dict[str, list[float]]:
    flows = [Flow("reaction", "Y", None, "Y * Y")]
    return Simulation(flow_model({"Y": y0}, flows)).run_at(TIMES)


@pytest.mark.parametrize("y0", [1, 2, 4, 0.5])
def test_every_initial_value_gives_one_over_t_plus_c(y0: float) -> None:
    assert_close(_decay(y0)["Y"], [1 / (t + 1 / y0) for t in TIMES], rel=ODE_REL)


def test_zero_is_the_solution_from_zero() -> None:
    assert _decay(0.0)["Y"] == [0.0] * len(TIMES)
