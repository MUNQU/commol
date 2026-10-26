"""
SIR with births and deaths: ``y' = A - beta y z - mu y`` and
``z' = beta y z - (gamma + mu) z``.

At the disease-free equilibrium ``(A / mu, 0)`` the linearization gives
``z' = (beta A / mu - (gamma + mu)) z``, so the infection grows or decays at that
rate. The endemic equilibrium has ``beta y = gamma + mu`` and
``A = y (beta z + mu)``.

References
----------
.. [1] Kermack, W. O., & McKendrick, A. G. (1932). Contributions to the
   mathematical theory of epidemics. II. The problem of endemicity.
   *Proceedings of the Royal Society of London. Series A*, 138, 55-83.
   https://doi.org/10.1098/rspa.1932.0171
.. [2] Hethcote, H. W. (1976). Qualitative analyses of communicable disease
   models. *Mathematical Biosciences*, 28, 335-356.
   https://doi.org/10.1016/0025-5564(76)90132-2
"""

import math

import pytest

from commol import Simulation
from commol._model.model import Model
from tests.support import ODE_REL, Flow, assert_close, flow_model, settled

A, MU, GAMMA = 20.0, 0.02, 0.1
"""Births per unit time, the death rate and the recovery rate: ``A / mu = 1000``."""


def _sir(beta: float, z0: float) -> Model:
    flows = [
        Flow("births", None, "Y", "A"),
        Flow("infection", "Y", "Z", "beta * Y * Z"),
        Flow("y_deaths", "Y", None, "mu * Y"),
        Flow("z_removal", "Z", None, "(gamma + mu) * Z"),
    ]
    parameters = {"A": A, "beta": beta, "mu": MU, "gamma": GAMMA}
    return flow_model({"Y": A / MU - z0, "Z": z0}, flows, parameters)


@pytest.mark.parametrize("beta", [2e-4, 1e-4])
def test_the_infection_moves_at_the_rate_of_the_linearization(beta: float) -> None:
    # beta A / mu - (gamma + mu) is 0.08 (invasion) or -0.02 (extinction). From
    # z0 = 1e-6 the susceptibles stay at A / mu to 1e-9 over 10 time units.
    rate = beta * A / MU - (GAMMA + MU)
    z0 = 1e-6

    results = Simulation(_sir(beta, z0)).run_at([5, 10])

    assert_close(
        results["Z"], [z0 * math.exp(rate * 5), z0 * math.exp(rate * 10)], rel=ODE_REL
    )


def test_the_endemic_equilibrium_has_beta_y_equal_to_gamma_plus_mu() -> None:
    beta = 2e-4
    y = (GAMMA + MU) / beta

    limit = settled(_sir(beta, 10), 2000)

    assert_close(limit, {"Y": y, "Z": (A / y - MU) / beta}, rel=ODE_REL)
