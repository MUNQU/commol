"""
SEIR with births and deaths: a latent stage between infection and infectivity.

``S' = b N - mu S - beta S I / N``, ``E' = beta S I / N - (k + mu) E``,
``I' = k E - gamma I`` with ``gamma = r + mu``. An infective causes
``Q0 = beta / gamma`` infections, of which a fraction ``f = k / (k + mu)``
survives the latent stage, so ``R0 = Q0 f``. With ``b = mu`` the endemic
equilibrium has ``S / N = 1 / R0``, ``E = mu N (1 - 1 / R0) / (k + mu)`` and
``I = k E / gamma``.

References
----------
.. [1] Aparicio, J. P., Capurro, A. F., & Castillo-Chavez, C. (2002). Markers
   of disease evolution: the case of tuberculosis. *Journal of Theoretical
   Biology*, 215, 227-237. https://doi.org/10.1006/jtbi.2001.2489
.. [2] Anderson, R. M., & May, R. M. (1991). *Infectious Diseases of Humans:
   Dynamics and Control*. Oxford University Press.
   https://doi.org/10.1093/oso/9780198545996.001.0001
"""

from commol.context.model import Model
from tests.support import (
    ODE_REL,
    Flow,
    assert_close,
    assert_threshold,
    flow_model,
    settled,
)

# Per year: latency progression, death and birth, and recovery.
K, MU, R = 1.0, 0.02, 0.5
GAMMA = R + MU
N = 1000.0


def _seir(r0: float, i0: float) -> Model:
    """[1]_ with ``b = mu`` and ``beta`` set from ``R0 = beta k / (gamma (k + mu))``."""
    flows = [
        Flow("births", None, "S", "mu * N"),
        Flow("infection", "S", "E", "beta * S * I / N"),
        Flow("progression", "E", "I", "k * E"),
        Flow("recovery", "I", "R", "r * I"),
        *(Flow(f"{x.lower()}_deaths", x, None, f"mu * {x}") for x in "SEIR"),
    ]
    beta = r0 * GAMMA * (K + MU) / K
    parameters = {"beta": beta, "k": K, "mu": MU, "r": R}
    return flow_model({"S": N - i0, "E": 0, "I": i0, "R": 0}, flows, parameters)


def test_r0_is_q0_times_the_latent_survival() -> None:
    assert_threshold(lambda r0: _seir(r0, 1e-6), ["E", "I"], 300)


def test_the_endemic_equilibrium_has_s_over_n_equal_to_one_over_r0() -> None:
    r0 = 3.0
    e = MU * N * (1 - 1 / r0) / (K + MU)

    limit = settled(_seir(r0, 10), 3000)

    assert_close(
        {name: limit[name] for name in "SEI"},
        {"S": N / r0, "E": e, "I": K * e / GAMMA},
        rel=ODE_REL,
    )
