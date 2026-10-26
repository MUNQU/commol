"""
An epidemic spread both from person to person and through a pathogen that the
infectives shed into the environment.

``S' = -beta1 S I - beta2 S B``, ``I' = beta1 S I + beta2 S B - gamma I`` and
``B' = r I - delta B`` give ``R0 = beta1 N / gamma + r beta2 N / (gamma delta)``
and the final size relation ``log(S0 / S_inf) = R0 (1 - S_inf / N) + beta2 B0 / delta``.
With an infectivity ``P(tau)`` at infection age ``tau`` and a pathogen that
survives a time with distribution ``Q``,
``R0 = beta1 N int P + r beta2 N int P int Q``; with stage ``j`` of relative
infectivity ``eps_j`` shedding at rate ``r_j`` into its own pool,
``R0 = beta1 N sum eps_j int P_j + beta2 N sum r_j int P_j int Q_j``. In both, the
initial pathogen adds ``beta2 B0 int Q`` to the final size relation.

References
----------
.. [1] Codeco, C. T. (2001). Endemic and epidemic dynamics of cholera: the role
   of the aquatic reservoir. *BMC Infectious Diseases*, 1, 1.
   https://doi.org/10.1186/1471-2334-1-1
.. [2] Tien, J. H., & Earn, D. J. D. (2010). Multiple transmission pathways and
   disease dynamics in a waterborne pathogen model. *Bulletin of Mathematical
   Biology*, 72, 1506-1533. https://doi.org/10.1007/s11538-010-9507-6
.. [3] Brauer, F., Shuai, Z., & van den Driessche, P. (2013). Dynamics of an
   age-of-infection cholera model. *Mathematical Biosciences and Engineering*,
   10, 1335-1349. https://doi.org/10.3934/mbe.2013.10.1335
"""

from commol._model.model import Model
from tests.support import (
    ODE_REL,
    Flow,
    assert_close,
    assert_threshold,
    final_size,
    flow_model,
    settled,
)

N, I0, B0 = 1000.0, 10.0, 5.0
BETA1, BETA2, GAMMA, R, DELTA = 1e-4, 5e-5, 0.2, 0.4, 0.5


def _sir_with_pathogen(scale: float, values: dict[str, float]) -> Model:
    """[1]_, [2]_ with both transmission rates multiplied by ``scale``."""
    flows = [
        Flow("direct", "S", "I", "beta1 * S * I"),
        Flow("indirect", "S", "I", "beta2 * S * B"),
        Flow("recovery", "I", "R", "gamma * I"),
        Flow("shedding", None, "B", "r * I"),
        Flow("decay", "B", None, "delta * B"),
    ]
    parameters = {
        "beta1": scale * BETA1,
        "beta2": scale * BETA2,
        "gamma": GAMMA,
        "r": R,
        "delta": DELTA,
    }
    return flow_model(values, flows, parameters)


R0 = BETA1 * N / GAMMA + R * BETA2 * N / (GAMMA * DELTA)


def test_r0_adds_the_direct_and_the_indirect_routes() -> None:
    small = {"S": N - 1e-6, "I": 1e-6, "R": 0.0, "B": 0.0}

    assert_threshold(lambda r0: _sir_with_pathogen(r0 / R0, small), ["I", "B"], 400)


def test_the_initial_pathogen_enters_the_final_size_relation() -> None:
    start = {"S": N - I0, "I": I0, "R": 0.0, "B": B0}

    limit = settled(_sir_with_pathogen(1.0, start), 1000)

    expected = N * final_size(R0, (N - I0) / N, BETA2 * B0 / DELTA)
    assert_close(limit["S"], expected, rel=ODE_REL)


def test_an_infectivity_that_varies_with_infection_age() -> None:
    # [3]: two stages with exits k1, k2 and infectivities eps1, eps2 give
    # int P = eps1 / k1 + eps2 / k2; shedding follows the infectivity and the
    # pathogen decays at delta, so int Q = 1 / delta.
    k1, k2, eps1, eps2 = 0.5, 0.25, 1.0, 0.4
    infectivity = "(eps1 * I1 + eps2 * I2)"
    flows = [
        Flow("infection", "S", "I1", f"S * (beta1 * {infectivity} + beta2 * B)"),
        Flow("progression", "I1", "I2", "k1 * I1"),
        Flow("recovery", "I2", "R", "k2 * I2"),
        Flow("shedding", None, "B", f"r * {infectivity}"),
        Flow("decay", "B", None, "delta * B"),
    ]
    parameters = {
        "beta1": BETA1,
        "beta2": BETA2,
        "r": R,
        "delta": DELTA,
        "k1": k1,
        "k2": k2,
        "eps1": eps1,
        "eps2": eps2,
    }
    values = {"S": N - I0, "I1": I0, "I2": 0, "R": 0, "B": B0}

    limit = settled(flow_model(values, flows, parameters), 2000)

    int_p = eps1 / k1 + eps2 / k2
    r0 = BETA1 * N * int_p + R * BETA2 * N * int_p / DELTA
    expected = N * final_size(r0, (N - I0) / N, BETA2 * B0 / DELTA)
    assert_close(limit["S"], expected, rel=ODE_REL)


def test_stages_that_shed_into_their_own_pathogen_pools() -> None:
    # Stage j sheds at r_j into a pool B_j that decays at delta_j; the initial
    # pathogen B0 decays at delta0. int P_j = 1 / k_j and int Q_j = 1 / delta_j.
    k = {1: 0.5, 2: 0.25}
    eps = {1: 1.0, 2: 0.4}
    shed = {1: 0.6, 2: 0.2}
    decay = {0: 0.5, 1: 0.4, 2: 0.8}
    flows = [
        Flow(
            "infection",
            "S",
            "I1",
            "S * (beta1 * (eps1 * I1 + eps2 * I2) + beta2 * (B0 + B1 + B2))",
        ),
        Flow("progression", "I1", "I2", "k1 * I1"),
        Flow("recovery", "I2", "R", "k2 * I2"),
        Flow("shedding_1", None, "B1", "r1 * I1"),
        Flow("shedding_2", None, "B2", "r2 * I2"),
        *(Flow(f"decay_{j}", f"B{j}", None, f"d{j} * B{j}") for j in decay),
    ]
    parameters = {"beta1": BETA1, "beta2": BETA2}
    parameters |= {f"k{j}": k[j] for j in k} | {f"eps{j}": eps[j] for j in eps}
    parameters |= {f"r{j}": shed[j] for j in shed} | {f"d{j}": decay[j] for j in decay}
    values = {"S": N - I0, "I1": I0, "I2": 0, "R": 0, "B0": B0, "B1": 0, "B2": 0}

    limit = settled(flow_model(values, flows, parameters), 2000)

    r0 = sum(
        BETA1 * N * eps[j] / k[j] + BETA2 * N * shed[j] / (k[j] * decay[j]) for j in k
    )
    expected = N * final_size(r0, (N - I0) / N, BETA2 * B0 / decay[0])
    assert_close(limit["S"], expected, rel=ODE_REL)
