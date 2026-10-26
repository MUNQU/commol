"""
Discrete-time SIR epidemics with infection and recovery probabilities.

A susceptible escapes infection in a step with probability
``G = exp(-beta I / N)`` and an infective stays infective with probability
``sigma``. The epidemic ends with ``I = 0`` and the final size relation
``log(S0 / S_inf) = beta sum(I) / N = R0 (1 - S_inf / N)``, ``R0 = beta / (1 - sigma)``.
With ``k`` infected stages, some of them treated, the same relation holds with
the control reproduction number
``Rc = beta sum_i [eps_i m_i / (1 - sigma_i^I) + delta_i n_i / (1 - sigma_i^T)]``.

References
----------
.. [1] Allen, L. J. S. (1994). Some discrete-time SI, SIR, and SIS epidemic
   models. *Mathematical Biosciences*, 124, 83-105.
   https://doi.org/10.1016/0025-5564(94)90025-6
.. [2] Brauer, F., Feng, Z., & Castillo-Chavez, C. (2010). Discrete epidemic
   models. *Mathematical Biosciences and Engineering*, 7, 1-15.
   https://doi.org/10.3934/mbe.2010.7.1
"""

import math

import pytest

from commol import Simulation
from commol._model.model import Model
from tests.support import ENGINE_REL, Flow, assert_close, final_size, flow_model

N, I0 = 1000.0, 10.0
STEPS = 600
"""Long enough for the infection to fall below 1e-12 of ``N``."""


def test_the_sir_epidemic_ends_with_the_final_size_relation() -> None:
    # [1], [2]: beta = 0.5 and sigma = exp(-0.25), so R0 = 2.26.
    beta, sigma = 0.5, math.exp(-0.25)
    flows = [
        Flow("infection", "S", "I", "S * (1 - exp(-beta * I / total))"),
        Flow("recovery", "I", "R", "(1 - sigma) * I"),
    ]
    model = flow_model(
        {"S": N - I0, "I": I0, "R": 0}, flows, {"beta": beta, "sigma": sigma}, ode=False
    )

    results = Simulation(model).run(STEPS)

    s, i = results["S"], results["I"]
    removed = [s[j] + i[j] for j in range(STEPS + 1)]
    assert all(later <= earlier for earlier, later in zip(removed, removed[1:]))
    assert i[-1] < 1e-12 * N
    s_inf = s[-1]
    assert_close(math.log(s[0] / s_inf), beta * sum(i) / N, rel=ENGINE_REL)
    r0 = beta / (1 - sigma)
    assert_close(s_inf, N * final_size(r0, s[0] / N), rel=ENGINE_REL)


P = {"p1": 0.3, "p2": 0.4}
STAY = {"sI1": 0.6, "sI2": 0.5, "sT1": 0.4, "sT2": 0.3}
INFECTIVITY = {"eps1": 1.0, "eps2": 0.8, "dlt1": 0.5, "dlt2": 0.2}
BETA = 0.6


def _staged_treatment(infectivity: dict[str, float]) -> Model:
    """
    [2]_ with k = 2 stages; ``infectivity`` names eps1, eps2, dlt1 and dlt2 in order.

    A fraction p_i of those who leave a stage enters treatment and q_i = 1 - p_i
    stays untreated. The initial infectives split like new ones, so every
    infected member follows the same pathways.
    """
    eps1, eps2, dlt1, dlt2 = infectivity
    infectious = f"{eps1} * I1 + {eps2} * I2 + {dlt1} * T1 + {dlt2} * T2"
    force = f"S * (1 - exp(-beta * ({infectious}) / total))"
    flows = [
        Flow("infection_untreated", "S", "I1", f"(1 - p1) * {force}"),
        Flow("infection_treated", "S", "T1", f"p1 * {force}"),
        Flow("untreated_progression", "I1", "I2", "(1 - p2) * (1 - sI1) * I1"),
        Flow("treatment", "I1", "T2", "p2 * (1 - sI1) * I1"),
        Flow("treated_progression", "T1", "T2", "(1 - sT1) * T1"),
        Flow("untreated_recovery", "I2", "R", "(1 - sI2) * I2"),
        Flow("treated_recovery", "T2", "R", "(1 - sT2) * T2"),
    ]
    values = {
        "S": N - I0,
        "I1": (1 - P["p1"]) * I0,
        "T1": P["p1"] * I0,
        "I2": 0,
        "T2": 0,
        "R": 0,
    }
    parameters = {"beta": BETA} | P | STAY | infectivity
    return flow_model(values, flows, parameters, ode=False)


def _control_reproduction_number() -> float:
    q1, q2 = 1 - P["p1"], 1 - P["p2"]
    m = {1: q1, 2: q1 * q2}
    n = {1: P["p1"], 2: P["p1"] + q1 * P["p2"]}
    eps = (INFECTIVITY["eps1"], INFECTIVITY["eps2"])
    dlt = (INFECTIVITY["dlt1"], INFECTIVITY["dlt2"])
    return BETA * sum(
        eps[i - 1] * m[i] / (1 - STAY[f"sI{i}"])
        + dlt[i - 1] * n[i] / (1 - STAY[f"sT{i}"])
        for i in (1, 2)
    )


def test_staged_treatment_ends_with_the_control_final_size_relation() -> None:
    results = Simulation(_staged_treatment(INFECTIVITY)).run(STEPS)

    expected = N * final_size(_control_reproduction_number(), (N - I0) / N)
    assert_close(results["S"][-1], expected, rel=ENGINE_REL)


@pytest.mark.parametrize(
    "names", [("e1", "e2", "dlt1", "dlt2"), ("eps1", "eps2", "d1T", "d2T")]
)
def test_staged_treatment_with_ids_that_hold_digits(names: tuple[str, ...]) -> None:
    infectivity = dict(zip(names, INFECTIVITY.values(), strict=True))

    results = Simulation(_staged_treatment(infectivity)).run(STEPS)

    expected = N * final_size(_control_reproduction_number(), (N - I0) / N)
    assert_close(results["S"][-1], expected, rel=ENGINE_REL)
