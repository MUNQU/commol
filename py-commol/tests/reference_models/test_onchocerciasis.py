"""
Onchocerciasis with ivermectin treatment: an SEIR-type host model and an SI
vector model.

Exposed hosts ``E`` become infective untreated ``I`` or premedicated ``P`` at
``q sigma`` and ``p sigma``; ``P`` are medicated (``M``) at ``phi`` and medicated
hosts are a fraction ``1 - nu`` as infective. All infected hosts recover at
``gamma`` and die at ``mu``, vectors die at ``d``; both totals ``N`` and ``F`` are
constant. In fractions of ``N`` and ``F`` the model becomes
``y' = beta F x v - (sigma + mu) y``, ..., ``v' / d = a w (1 - v) - v`` with
``a = alpha N / d`` and ``w = i + h - nu m``. When vectors are fast, ``v`` takes
its quasi-steady state ``a w / (1 + a w)``; the reduced model has the
disease-free state ``x = 1``, stable if ``R0 = beta F a D < 1``, where ``D`` is the
time an exposed host spends infective, weighted by infectivity, and an endemic
equilibrium otherwise.

References
----------
.. [1] Dietz, K. (1982). The population dynamics of onchocerciasis. In R. M.
   Anderson (Ed.), *The Population Dynamics of Infectious Diseases: Theory and
   Applications* (pp. 209-241). Springer.
   https://doi.org/10.1007/978-1-4899-2901-3_7
.. [2] Ledder, G., Sylvester, D., Bouchat, R. R., & Thiel, J. A. (2018).
   Continuous and pulsed epidemiological models for onchocerciasis with
   implications for eradication strategy. *Mathematical Biosciences and
   Engineering*, 15, 841-862. https://doi.org/10.3934/mbe.2018038
"""

from commol.context.model import Model
from tests.support import (
    ENGINE_REL,
    Flow,
    assert_close,
    assert_threshold,
    flow_model,
    increments,
    settled,
)

# Per year.
HOST = {"mu": 0.02, "sigma": 1.0, "p": 0.65, "phi": 2.0, "nu": 0.6, "gamma": 0.8}
D_VECTOR = 12.0
N, F = 1000.0, 5000.0
ALPHA, BETA = 1.08 / N, 0.3 / F


def _host_flows(force: str) -> list[Flow]:
    """The host flows; deaths and recoveries return to S, so ``N`` is constant."""
    return [
        Flow("infection", "S", "E", force),
        Flow("untreated", "E", "I", "(1 - p) * sigma * E"),
        Flow("premedicated", "E", "P", "p * sigma * E"),
        Flow("medication", "P", "M", "phi * P"),
        Flow("e_turnover", "E", "S", "mu * E"),
        *(Flow(f"{x.lower()}_turnover", x, "S", f"(gamma + mu) * {x}") for x in "IPM"),
    ]


def test_fractions_of_the_totals_follow_the_rescaled_system() -> None:
    # The right-hand side at one state, in counts, divided by N or F.
    state = {
        "S": 600.0,
        "E": 150.0,
        "I": 120.0,
        "P": 70.0,
        "M": 60.0,
        "U": 3000.0,
        "V": 2000.0,
    }
    flows = _host_flows("beta * S * V") + [
        Flow("vector_infection", "U", "V", "alpha * U * (I + P + (1 - nu) * M)"),
        Flow("vector_turnover", "V", "U", "d * V"),
    ]
    parameters = HOST | {"alpha": ALPHA, "beta": BETA, "d": D_VECTOR}

    rates = increments(state, flows, parameters)

    mu, sigma, p, phi, nu, gamma = HOST.values()
    x, y, i, m = (state[name] / N for name in "SEIM")
    h = (state["P"] + state["M"]) / N
    v = state["V"] / F
    a, w = ALPHA * N / D_VECTOR, i + h - nu * m
    assert_close(rates["E"] / N, BETA * F * x * v - (sigma + mu) * y, rel=ENGINE_REL)
    assert_close(rates["I"] / N, (1 - p) * sigma * y - (gamma + mu) * i, rel=ENGINE_REL)
    assert_close(
        (rates["P"] + rates["M"]) / N, p * sigma * y - (gamma + mu) * h, rel=ENGINE_REL
    )
    assert_close(rates["M"] / N, phi * h - (phi + gamma + mu) * m, rel=ENGINE_REL)
    assert_close(rates["V"] / (F * D_VECTOR), a * w * (1 - v) - v, rel=ENGINE_REL)


def _reduced(beta_f: float, a: float, start: dict[str, float]) -> Model:
    """Hosts in fractions of ``N``, with vectors at their quasi-steady state."""
    w = "(I + P + (1 - nu) * M)"
    flows = _host_flows(f"bF * S * a * {w} / (1 + a * {w})")
    return flow_model(start, flows, HOST | {"bF": beta_f, "a": a})


def _infective_time() -> float:
    """``D``: the time an exposed host spends infective, weighted by infectivity."""
    mu, sigma, p, phi, nu, gamma = HOST.values()
    exposed, infected, premedicated = sigma + mu, gamma + mu, phi + gamma + mu
    return (
        (1 - p) * sigma / infected
        + p * sigma * (gamma + mu + (1 - nu) * phi) / (premedicated * infected)
    ) / exposed


A = ALPHA * N / D_VECTOR


def test_the_disease_free_state_is_stable_below_r0_one() -> None:
    small = {"S": 1 - 1e-9, "E": 1e-9, "I": 0.0, "P": 0.0, "M": 0.0}

    assert_threshold(
        lambda r0: _reduced(r0 / (A * _infective_time()), A, small),
        ["E", "I", "P", "M"],
        600,
    )


def test_above_r0_one_the_endemic_equilibrium_is_stable() -> None:
    start = {"S": 0.9, "E": 0.1, "I": 0.0, "P": 0.0, "M": 0.0}

    limit = settled(_reduced(2.0 / (A * _infective_time()), A, start), 2000)

    assert all(limit[x] > 1e-3 for x in "EIPM")
