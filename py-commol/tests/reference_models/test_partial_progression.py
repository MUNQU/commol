"""
HIV when only a fraction of the infected progress to AIDS.

A fraction ``p`` of new infections enters ``I`` (leaves at ``sigma_I`` and
progresses to AIDS) and ``1 - p`` enters ``Y`` (leaves at ``sigma_Y`` and stops
being sexually active). New infections are ``lambda C(T) S W / T``, with
``W = I + Y``, ``T = S + W`` and a partner rate ``C(T)`` that does not decrease.
``R0 = lambda (p / sigma_I + (1 - p) / sigma_Y) C(Lambda / mu)``; above one
there is a unique endemic equilibrium, and it is stable.

References
----------
.. [1] Castillo-Chavez, C., Cooke, K., Huang, W., & Levin, S. A. (1989). On the
   role of long incubation periods in the dynamics of acquired immunodeficiency
   syndrome (AIDS). Part 1: Single population models. *Journal of Mathematical
   Biology*, 27, 373-398. https://doi.org/10.1007/BF00290636
"""

from commol.context.model import Model
from tests.support import (
    ODE_REL,
    Flow,
    assert_close,
    assert_threshold,
    bisect,
    flow_model,
    settled,
)

# Per year: recruitment, exit, progression, AIDS death and the AIDS fraction.
LAMBDA, MU, ALPHA, D, P = 20.0, 0.02, 0.1, 1.0, 0.3
SIGMA = ALPHA + MU
"""``sigma_I = sigma_Y``: both infected classes leave at the same rate."""


def _partners(t: float) -> float:
    """``C(T) = 4 T / (T + 500)``: more partners in a larger active population."""
    return 4 * t / (t + 500)


def _hiv(lam: float, values: dict[str, float]) -> Model:
    contacts = "lam * 4 * (S + I + Y) / (S + I + Y + 500) * S * (I + Y) / (S + I + Y)"
    flows = [
        Flow("recruitment", None, "S", "Lam"),
        Flow("to_aids", "S", "I", f"p * {contacts}"),
        Flow("not_to_aids", "S", "Y", f"(1 - p) * {contacts}"),
        Flow("s_exit", "S", None, "mu * S"),
        Flow("progression", "I", "A", "alpha * I"),
        Flow("i_exit", "I", None, "mu * I"),
        Flow("inactivity", "Y", "Z", "alpha * Y"),
        Flow("y_exit", "Y", None, "mu * Y"),
        Flow("aids_deaths", "A", None, "(d + mu) * A"),
        Flow("z_exit", "Z", None, "mu * Z"),
    ]
    parameters = {"lam": lam, "Lam": LAMBDA, "mu": MU, "alpha": ALPHA, "d": D, "p": P}
    return flow_model(values, flows, parameters)


R0_PER_LAMBDA = (P / SIGMA + (1 - P) / SIGMA) * _partners(LAMBDA / MU)


def test_r0_weights_the_two_infective_periods() -> None:
    small = {"S": LAMBDA / MU - 1e-6, "I": 1e-6, "Y": 0.0, "A": 0.0, "Z": 0.0}

    assert_threshold(lambda r0: _hiv(r0 / R0_PER_LAMBDA, small), ["I", "Y"], 1000)


def test_above_one_the_unique_endemic_equilibrium_is_stable() -> None:
    # W' = 0 gives lambda C(T) S / T = sigma and S' = 0 gives
    # Lambda = sigma W + mu S, so S = (sigma T - Lambda) / (sigma - mu).
    lam = 2.0 / R0_PER_LAMBDA

    def balance(t: float) -> float:
        s = (SIGMA * t - LAMBDA) / (SIGMA - MU)
        return lam * _partners(t) * s / t - SIGMA

    t = bisect(balance, LAMBDA / SIGMA * (1 + 1e-12), LAMBDA / MU)
    s = (SIGMA * t - LAMBDA) / (SIGMA - MU)
    start = {"S": 900.0, "I": 50.0, "Y": 50.0, "A": 0.0, "Z": 0.0}

    limit = settled(_hiv(lam, start), 3000)

    assert_close(
        {x: limit[x] for x in "SIY"},
        {"S": s, "I": P * (t - s), "Y": (1 - P) * (t - s)},
        rel=ODE_REL,
    )
