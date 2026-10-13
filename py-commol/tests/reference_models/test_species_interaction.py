"""
Two interacting species: competition and predation with a saturating response.

Competition: ``y' = y (1 - 2y - z)``, ``z' = z (1 - y - 2z)``. Of its four
equilibria only the coexistence point ``(1/3, 1/3)`` is asymptotically stable.

Predation: ``y' = y (2 - y) - y z / (y + 1)``, ``z' = 4 y z / (y + 1) - z``. Its
three equilibria are unstable and every orbit in the open quadrant is bounded,
so every such orbit approaches a periodic orbit.

References
----------
.. [1] Volterra, V. (1926). Fluctuations in the abundance of a species
   considered mathematically. *Nature*, 118, 558-560.
   https://doi.org/10.1038/118558a0
.. [2] Rosenzweig, M. L., & MacArthur, R. H. (1963). Graphical representation
   and stability conditions of predator-prey interactions. *The American
   Naturalist*, 97, 209-223. https://doi.org/10.1086/282272
"""

import pytest

from commol import Simulation
from tests.support import ODE_REL, Flow, assert_close, flow_model, settled


@pytest.mark.parametrize(("y0", "z0"), [(0.25, 0.75), (0.9, 0.1), (0.001, 0.999)])
def test_competition_settles_at_coexistence(y0: float, z0: float) -> None:
    # [1]: the boundary equilibria (0, 1/2) and (1/2, 0) repel, so every start
    # in the open quadrant ends at (1/3, 1/3).
    flows = [
        Flow("y_growth", None, "Y", "Y"),
        Flow("y_competition", "Y", None, "2 * Y * Y + Y * Z"),
        Flow("z_growth", None, "Z", "Z"),
        Flow("z_competition", "Z", None, "Y * Z + 2 * Z * Z"),
    ]

    limit = settled(flow_model({"Y": y0, "Z": z0}, flows), 200)

    assert_close(limit, {"Y": 1 / 3, "Z": 1 / 3}, rel=ODE_REL)


def test_saturating_predation_approaches_a_periodic_orbit() -> None:
    # [2]: from (1, 2) the orbit neither settles at the interior equilibrium
    # (1/3, 20/9) nor grows: two windows of 100 time units, each longer than a
    # period, see the same range of y.
    flows = [
        Flow("prey_growth", None, "Y", "2 * Y"),
        Flow("prey_crowding", "Y", None, "Y * Y"),
        Flow("predation", "Y", None, "Y * Z / (Y + 1)"),
        Flow("predator_growth", None, "Z", "4 * Y * Z / (Y + 1)"),
        Flow("predator_deaths", "Z", None, "Z"),
    ]
    times = [300 + 0.01 * i for i in range(20001)]

    results = Simulation(flow_model({"Y": 1, "Z": 2}, flows)).run_at(times)

    first, second = results["Y"][:10001], results["Y"][10000:]
    assert max(first) - min(first) > 0.5
    assert_close(max(second), max(first), rel=1e-4)
    assert_close(min(second), min(first), rel=1e-4)
    assert max(4 * y + z for y, z in zip(results["Y"], results["Z"], strict=True)) < 9
