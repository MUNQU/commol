"""
The relaxation oscillator ``y'' - mu (1 - y^2) y' + y = 0`` with ``mu = 2``.

As a system, ``y' = z`` and ``z' = mu (1 - y^2) z - y``. Its only equilibrium is
the origin: unstable for ``mu > 0``, where orbits spiral out to a cycle of
amplitude about 2, and asymptotically stable when the sign of the damping term
is reversed, where the linearization has the double eigenvalue -1.

References
----------
.. [1] van der Pol, B. (1926). On "relaxation-oscillations". *The London,
   Edinburgh, and Dublin Philosophical Magazine and Journal of Science*, 2,
   978-992. https://doi.org/10.1080/14786442608564127
"""

from commol import Simulation
from commol._model.model import Model
from tests.support import Flow, flow_model


def _oscillator(damping: float, y0: float, z0: float) -> Model:
    """``y' = z``, ``z' = damping (1 - y^2) z - y`` [1]_."""
    flows = [
        Flow("velocity", None, "Y", "Z"),
        Flow("damping", None, "Z", "c * (1 - Y * Y) * Z"),
        Flow("restoring", "Z", None, "Y"),
    ]
    return flow_model({"Y": y0, "Z": z0}, flows, {"c": damping})


def test_the_origin_repels_orbits_to_a_cycle() -> None:
    # From distance 1 the orbit grows to the cycle, whose amplitude is about 2.
    times = [50 + 0.01 * i for i in range(1001)]

    results = Simulation(_oscillator(2.0, 1, 0)).run_at(times)

    assert max(abs(y) for y in results["Y"]) > 1.9


def test_reversed_damping_makes_the_origin_stable() -> None:
    # The linearization at the origin has trace -2 and determinant 1, so
    # orbits near it decay like t e^-t: below 1e-15 at t = 40.
    results = Simulation(_oscillator(-2.0, 0.5, 0.5)).run_at([40])

    assert abs(results["Y"][0]) < 1e-9
    assert abs(results["Z"][0]) < 1e-9
