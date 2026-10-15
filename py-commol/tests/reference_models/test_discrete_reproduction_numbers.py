"""
Reproduction numbers of discrete SEIR models with an arbitrary bounded
infective period.

The infective period ``Y`` has ``q_i = P(Y > i)`` and ``q_M = 0``; one bin per
stage age ``i`` keeps the model memoryless, as the linear chain trick does in
continuous time. With infectivity ``beta_i`` at stage age ``i``,
``R0 = rho(F (I - T)^-1) = sum_i beta_i q_{i-1}``, and the epidemic ends with
``log(S0 / S_inf) = R0 (1 - S_inf / N)``. With two sexes that only infect each
other, ``R = sqrt(R_f R_m)``, ``R_w = sum_i beta_{w,i} q_{w,i-1}``.

References
----------
.. [1] Allen, L. J. S., & van den Driessche, P. (2008). The basic reproduction
   number in some discrete-time epidemic models. *Journal of Difference
   Equations and Applications*, 14, 1127-1147.
   https://doi.org/10.1080/10236190802332308
.. [2] Hernandez-Ceron, N., Feng, Z., & Castillo-Chavez, C. (2013). Discrete
   epidemic models with arbitrary stage distributions and applications to
   disease control. *Bulletin of Mathematical Biology*, 75, 1716-1746.
   https://doi.org/10.1007/s11538-013-9866-x
"""

import math

from commol import Simulation
from commol.context.model import Model
from tests.support import (
    ENGINE_REL,
    Flow,
    assert_close,
    assert_threshold,
    final_size,
    flow_model,
)


def _infective_chain(sex: str, q: list[float], latent: str) -> list[Flow]:
    """The latent bin and one bin per stage age of the infective period [2]_."""
    stages = len(q) - 1
    flows = [Flow(f"onset{sex}", latent, f"I{sex}1", f"(1 - alpha{sex}) * {latent}")]
    for i in range(1, stages + 1):
        stay = q[i] / q[i - 1]
        if i < stages:
            flows.append(
                Flow(
                    f"stage{sex}{i}",
                    f"I{sex}{i}",
                    f"I{sex}{i + 1}",
                    f"{stay} * I{sex}{i}",
                )
            )
        flows.append(
            Flow(
                f"recovery{sex}{i}", f"I{sex}{i}", f"R{sex}", f"{1 - stay} * I{sex}{i}"
            )
        )
    return flows


def test_r0_sums_the_stage_infectivities() -> None:
    # [1], [2]: q = (1, 0.8, 0.5, 0.2, 0) and stage infectivities beta_i.
    n, e0 = 1000.0, 10.0
    q = [1.0, 0.8, 0.5, 0.2, 0.0]
    beta = [0.6, 0.5, 0.3, 0.1]
    force = " + ".join(f"b{i} * I{i}" for i in range(1, 5))
    flows = [Flow("infection", "S", "E", f"S * (1 - exp(-({force}) / total))")]
    flows += _infective_chain("", q, "E")
    values = {"S": n - e0, "E": e0} | {f"I{i}": 0.0 for i in range(1, 5)} | {"R": 0.0}
    parameters = {"alpha": 0.6} | {f"b{i}": beta[i - 1] for i in range(1, 5)}

    results = Simulation(flow_model(values, flows, parameters, ode=False)).run(800)

    r0 = sum(beta[i] * q[i] for i in range(4))
    assert_close(results["S"][-1], n * final_size(r0, (n - e0) / n), rel=ENGINE_REL)


def test_two_sexes_have_the_geometric_mean_of_their_numbers() -> None:
    # Each sex has N members and is infected only by the other; ``beta_m``
    # are the infectivities of infective males, so R_m counts the females one
    # male infects.
    n = 1000.0
    q = {"f": [1.0, 0.7, 0.3, 0.0], "m": [1.0, 0.9, 0.6, 0.2, 0.0]}
    beta = {"f": [0.5, 0.4, 0.2], "m": [0.3, 0.3, 0.2, 0.1]}
    numbers = {w: sum(b * q[w][i] for i, b in enumerate(beta[w])) for w in "fm"}
    r = math.sqrt(numbers["f"] * numbers["m"])

    def model_at(target: float) -> Model:
        scale = target / r
        flows = []
        parameters: dict[str, float] = {"Nsex": n, "alphaf": 0.5, "alpham": 0.4}
        values: dict[str, float] = {}
        for w, other in (("f", "m"), ("m", "f")):
            force = " + ".join(
                f"b{other}{i} * I{other}{i}" for i in range(1, len(beta[other]) + 1)
            )
            flows.append(
                Flow(
                    f"infection{w}",
                    f"S{w}",
                    f"E{w}",
                    f"S{w} * (1 - exp(-({force}) / Nsex))",
                )
            )
            flows += _infective_chain(w, q[w], f"E{w}")
            parameters |= {
                f"b{w}{i}": scale * b for i, b in enumerate(beta[w], start=1)
            }
            values |= {f"S{w}": n, f"E{w}": 0.0}
            values |= {f"I{w}{i}": 0.0 for i in range(1, len(beta[w]) + 1)}
            values |= {f"R{w}": 0.0}
        values["Sf"], values["Ef"] = n - 1e-6, 1e-6
        return flow_model(values, flows, parameters, ode=False)

    infected = [f"I{w}{i}" for w in "fm" for i in range(1, len(beta[w]) + 1)]
    assert_threshold(model_at, infected, 400)
