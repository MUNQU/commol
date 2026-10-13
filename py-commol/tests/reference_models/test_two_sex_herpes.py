"""
Herpes simplex type 2 in males and in a low- and a high-risk group of females.

In each group ``i``, susceptibles ``S_i`` become acute ``A_i``, acute cases turn
latent ``L_i`` at ``omega_i + theta_i`` and latent ones reactivate at
``gamma_i``; members enter and leave at ``mu_i``. Males infect females and
females infect males. One male causes, through females of group ``j``,
``R_mjm^2 = b_j beta_mj / (omega_j + theta_j + mu_j) P_m b_m c_j beta_jm /
(omega_m + theta_m + mu_m) P_j`` male infections, where
``P_i = (omega_i + theta_i + mu_i) (gamma_i + mu_i) / D_i`` with
``D_i = mu_i (gamma_i + omega_i + theta_i + mu_i)`` counts the returns to the
acute stage, and ``R = sqrt(R_m1m^2 + R_m2m^2)``.

References
----------
.. [1] Feng, Z., Qiu, Z., Sang, Z., Lorenzo, C., & Glasser, J. (2013). Modeling
   the synergy between HSV-2 and HIV and potential impact of HSV-2 therapy.
   *Mathematical Biosciences*, 245, 171-187.
   https://doi.org/10.1016/j.mbs.2013.07.003
"""

import math

import pytest

from commol.context.model import Model
from tests.support import Flow, assert_threshold, flow_model, settled

# Per month, without treatment; mu is a sexual life of 35 years.
SIZES = {"m": 1e7, "f1": 9.9889e6, "f2": 1.1099e4}
PARTNERS = {"m": 0.1, "f1": 0.0901, "f2": 9.01}
OMEGA, MU = 2.5, 1 / 420
REACTIVATION = {"m": 0.436, "f1": 0.339, "f2": 0.339}


def _acute_multiplier(group: str) -> float:
    out, gamma = OMEGA + MU, REACTIVATION[group]
    return out * (gamma + MU) / (MU * (gamma + out))


def _reproduction_number(c: float, beta: float) -> float:
    squares = [
        PARTNERS[f]
        * beta
        / (OMEGA + MU)
        * _acute_multiplier("m")
        * PARTNERS["m"]
        * share
        * beta
        / (OMEGA + MU)
        * _acute_multiplier(f)
        for f, share in (("f1", c), ("f2", 1 - c))
    ]
    return math.sqrt(sum(squares))


def _herpes(c: float, beta: float, acute: dict[str, float]) -> Model:
    """[1]_ with every transmission probability per partner equal to ``beta``."""
    total = {g: f"(S{g} + A{g} + L{g})" for g in SIZES}
    forces = {
        "m": (
            f"bm * (c * beta * Af1 / {total['f1']}"
            f" + (1 - c) * beta * Af2 / {total['f2']})"
        ),
        "f1": f"bf1 * beta * Am / {total['m']}",
        "f2": f"bf2 * beta * Am / {total['m']}",
    }
    flows = []
    for g in SIZES:
        flows += [
            Flow(f"entry_{g}", None, f"S{g}", f"mu * {total[g]}"),
            Flow(f"infection_{g}", f"S{g}", f"A{g}", f"{forces[g]} * S{g}"),
            Flow(f"latency_{g}", f"A{g}", f"L{g}", f"omega * A{g}"),
            Flow(f"reactivation_{g}", f"L{g}", f"A{g}", f"gamma{g} * L{g}"),
            *(Flow(f"exit_{x}{g}", f"{x}{g}", None, f"mu * {x}{g}") for x in "SAL"),
        ]
    parameters = {"c": c, "beta": beta, "omega": OMEGA, "mu": MU}
    parameters |= {f"b{g}": PARTNERS[g] for g in SIZES}
    parameters |= {f"gamma{g}": REACTIVATION[g] for g in SIZES}
    values: dict[str, float] = {}
    for g, size in SIZES.items():
        values |= {f"S{g}": size - acute[g], f"A{g}": acute[g], f"L{g}": 0.0}
    return flow_model(values, flows, parameters)


@pytest.mark.parametrize("c", [1.0, 0.9])
def test_the_male_to_male_numbers_set_the_threshold(c: float) -> None:
    # With c = 1 only the low-risk females transmit back, so R = R_m1m.
    unit = _reproduction_number(c, 1.0)
    acute = {"m": 1e-3, "f1": 0.0, "f2": 0.0}

    assert_threshold(lambda r: _herpes(c, r / unit, acute), ["Am", "Lm"], 2e4)


def test_above_the_threshold_the_endemic_equilibrium_is_stable() -> None:
    beta = 2.0 / _reproduction_number(0.9, 1.0)

    limit = settled(_herpes(0.9, beta, {"m": 100.0, "f1": 100.0, "f2": 10.0}), 2e5)

    assert all(limit[f"A{g}"] > 0 and limit[f"L{g}"] > 0 for g in SIZES)
