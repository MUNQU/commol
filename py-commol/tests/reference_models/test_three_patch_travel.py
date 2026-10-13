"""
An SIR model in three patches whose residents travel and return.

``X_ij`` are the residents of patch ``i`` present in patch ``j``. Residents leave
home at ``g_i``, to patch ``j`` with probability ``m_ji``, and return at
``r_ij``; infection happens in the patch where people are, at
``kappa_j beta_j S_ij sum_k I_kj / N_j^p`` with ``N_j^p`` everyone present in
``j``. Deaths at ``mu`` are replaced by births at home, so every resident
population ``N_i^0`` is constant and the travel settles at
``N_ii = N_i^0 / (1 + g_i sum_k m_ki / (mu + r_ik))`` and
``N_ij = g_i m_ji / (mu + r_ij) N_ii``. ``R0`` is the spectral radius of
``F V^-1`` on the nine infective classes.

References
----------
.. [1] Sattenspiel, L., & Dietz, K. (1995). A structured epidemic model
   incorporating geographic mobility among regions. *Mathematical
   Biosciences*, 128, 71-91. https://doi.org/10.1016/0025-5564(94)00068-B
.. [2] Arino, J., & van den Driessche, P. (2003). A multi-city epidemic model.
   *Mathematical Population Studies*, 10, 175-193.
   https://doi.org/10.1080/08898480306720
"""

import itertools

import numpy as np

from commol import Simulation
from commol.context.model import Model
from tests.support import (
    ODE_REL,
    Flow,
    assert_close,
    assert_threshold,
    flow_model,
    settled,
)

PATCHES = (1, 2, 3)
PAIRS = list(itertools.product(PATCHES, PATCHES))
# Per day: recovery in 25 days, a life of 75 years, 1500 residents each.
GAMMA, MU, N0 = 1 / 25, 1 / (75 * 365), 1500.0
LEAVE = {1: 0.01, 2: 0.02, 3: 0.03}
KAPPA = {1: 3.0, 2: 2.0, 3: 1.0}
DESTINATION, RETURN = 0.5, 0.05
"""``m_ji`` and ``r_ij`` for every ``i != j``."""


def _travel(beta: float, values: dict[str, float]) -> Model:
    """[1]_, [2]_ with ``beta_j = beta`` in every patch."""
    flows = []
    for i, j in PAIRS:
        for x in "SIR":
            home, away = f"{x}{i}{i}", f"{x}{i}{j}"
            if i != j:
                flows += [
                    Flow(
                        f"leave_{away}",
                        home,
                        away,
                        f"{LEAVE[i] * DESTINATION} * {home}",
                    ),
                    Flow(f"return_{away}", away, home, f"{RETURN} * {away}"),
                ]
            if away != f"S{i}{i}":
                flows.append(Flow(f"death_{away}", away, f"S{i}{i}", f"mu * {away}"))
        present = " + ".join(f"{x}{k}{j}" for k in PATCHES for x in "SIR")
        infectives = " + ".join(f"I{k}{j}" for k in PATCHES)
        flows += [
            Flow(
                f"infection_{i}{j}",
                f"S{i}{j}",
                f"I{i}{j}",
                f"{KAPPA[j]} * beta * S{i}{j} * ({infectives}) / ({present})",
            ),
            Flow(f"recovery_{i}{j}", f"I{i}{j}", f"R{i}{j}", f"gamma * I{i}{j}"),
        ]
    return flow_model(values, flows, {"beta": beta, "gamma": GAMMA, "mu": MU})


def _settled_travel() -> dict[tuple[int, int], float]:
    """``N_ij`` once the travel has settled."""
    sizes = {}
    for i in PATCHES:
        others = [k for k in PATCHES if k != i]
        home = N0 / (1 + LEAVE[i] * sum(DESTINATION / (MU + RETURN) for _ in others))
        sizes[i, i] = home
        for j in others:
            sizes[i, j] = LEAVE[i] * DESTINATION / (MU + RETURN) * home
    return sizes


def _everyone_home(infected: float = 0.0) -> dict[str, float]:
    values = {f"{x}{i}{j}": 0.0 for x in "SIR" for i, j in PAIRS}
    for i in PATCHES:
        values[f"S{i}{i}"] = N0
    values["S11"] -= infected
    values["I11"] = infected
    return values


def test_every_resident_population_stays_constant() -> None:
    times = [10.0, 100.0, 1000.0]

    results = Simulation(_travel(0.025, _everyone_home(10.0))).run_at(times)

    for i in PATCHES:
        held = [
            sum(results[f"{x}{i}{j}"][t] for x in "SIR" for j in PATCHES)
            for t in range(len(times))
        ]
        assert_close(held, [N0] * len(times), rel=ODE_REL)


def test_travel_settles_at_the_stated_distribution() -> None:
    limit = settled(_travel(0.0, _everyone_home()), 2000)

    expected = {f"S{i}{j}": size for (i, j), size in _settled_travel().items()}
    assert_close({name: limit[name] for name in expected}, expected, rel=ODE_REL)


def _r0_per_beta() -> float:
    """``rho(F V^-1)`` for ``beta = 1``, infectives ordered I11, I12, ..., I33."""
    sizes = _settled_travel()
    present = {q: sum(sizes[k, q] for k in PATCHES) for q in PATCHES}
    index = {pair: n for n, pair in enumerate(PAIRS)}
    new, leaving = np.zeros((9, 9)), np.zeros((9, 9))
    for (i, q), (k, q2) in itertools.product(PAIRS, PAIRS):
        if q == q2:
            new[index[i, q], index[k, q]] = KAPPA[q] * sizes[i, q] / present[q]
    for i, j in PAIRS:
        row = index[i, j]
        if i == j:
            leaving[row, row] = LEAVE[i] + GAMMA + MU
            for k in PATCHES:
                if k != i:
                    leaving[row, index[i, k]] = -RETURN
        else:
            leaving[row, row] = RETURN + GAMMA + MU
            leaving[row, index[i, i]] = -LEAVE[i] * DESTINATION
    return float(max(abs(np.linalg.eigvals(new @ np.linalg.inv(leaving)))))


def test_r0_is_the_spectral_radius_of_the_next_generation_matrix() -> None:
    sizes = _settled_travel()
    unit = _r0_per_beta()

    def model_at(r0: float) -> Model:
        values = _everyone_home()
        values |= {f"S{i}{j}": size for (i, j), size in sizes.items()}
        values["S11"] -= 1e-6
        values["I11"] = 1e-6
        return _travel(r0 / unit, values)

    infectives = [f"I{i}{j}" for i, j in PAIRS]
    assert_threshold(model_at, infectives, 4000)
