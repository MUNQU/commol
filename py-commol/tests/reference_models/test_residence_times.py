"""
Two patches linked by short visits: residents of patch ``i`` spend a fraction
``p_ij`` of their time in patch ``j``, and each patch has its own contact rate.

Every flow is written as the source builds it: a resident of patch ``i`` present
in patch ``j`` meets there the infectives present in ``j``, out of everyone
present in ``j``. Their sum per resident group is the stated system. For an SIR
model with contact rates ``beta_j``,
``S_i' = -S_i sum_j beta_j p_ij (p_1j I_1 + p_2j I_2) / (p_1j N_1 + p_2j N_2)``.
For dengue, hosts of patch ``i`` are bitten by the vectors of the patch they
visit, vectors of patch ``i`` bite the hosts present in ``i``, and a fraction
``q`` of the vectors born to infective mothers is born infective.

References
----------
.. [1] Bichara, D., Kang, Y., Castillo-Chavez, C., Horan, R., & Perrings, C.
   (2015). SIS and SIR epidemic models under virtual dispersal. *Bulletin of
   Mathematical Biology*, 77, 2004-2034.
   https://doi.org/10.1007/s11538-015-0113-5
.. [2] Lee, S., & Castillo-Chavez, C. (2015). The role of residence times in
   two-patch dengue transmission dynamics and optimal strategies. *Journal of
   Theoretical Biology*, 374, 152-164. https://doi.org/10.1016/j.jtbi.2015.03.005
.. [3] Esteva, L., & Vargas, C. (2000). Influence of vertical and mechanical
   transmission on the dynamics of dengue disease. *Mathematical Biosciences*,
   167, 51-64. https://doi.org/10.1016/S0025-5564(00)00024-9
"""

from tests.support import ENGINE_REL, Flow, assert_close, increments

PATCHES = (1, 2)
TIME_SHARE = {(1, 1): 0.7, (1, 2): 0.3, (2, 1): 0.2, (2, 2): 0.8}


def _present(compartment: str, j: int) -> str:
    """Members of ``compartment`` present in patch ``j``: ``p_1j X1 + p_2j X2``."""
    return " + ".join(f"{TIME_SHARE[k, j]} * {compartment}{k}" for k in PATCHES)


def test_two_patch_sir_follows_the_residence_time_equations() -> None:
    # [1]: N_i = S_i + I_i + R_i.
    state = {"S1": 600.0, "I1": 30.0, "R1": 70.0, "S2": 250.0, "I2": 20.0, "R2": 30.0}
    beta, gamma = {1: 0.5, 2: 0.2}, 0.1
    flows = []
    for i in PATCHES:
        for j in PATCHES:
            hosts = f"({_present('S', j)} + {_present('I', j)} + {_present('R', j)})"
            flows.append(
                Flow(
                    f"infection_{i}_in_{j}",
                    f"S{i}",
                    f"I{i}",
                    f"beta{j} * {TIME_SHARE[i, j]} * S{i}"
                    f" * ({_present('I', j)}) / {hosts}",
                )
            )
        flows.append(Flow(f"recovery_{i}", f"I{i}", f"R{i}", f"gamma * I{i}"))
    parameters = {"beta1": beta[1], "beta2": beta[2], "gamma": gamma}

    rates = increments(state, flows, parameters)

    n = {i: state[f"S{i}"] + state[f"I{i}"] + state[f"R{i}"] for i in PATCHES}
    for i in PATCHES:
        incidence = state[f"S{i}"] * sum(
            beta[j]
            * TIME_SHARE[i, j]
            * sum(TIME_SHARE[k, j] * state[f"I{k}"] for k in PATCHES)
            / sum(TIME_SHARE[k, j] * n[k] for k in PATCHES)
            for j in PATCHES
        )
        assert_close(rates[f"S{i}"], -incidence, rel=ENGINE_REL)
        assert_close(rates[f"I{i}"], incidence - gamma * state[f"I{i}"], rel=ENGINE_REL)


def test_two_patch_dengue_follows_the_residence_time_equations() -> None:
    # [2], [3]: per day. Hosts of patch i are S, L (exposed), I and R; vectors
    # are U, W (exposed) and V.
    host = {"muh": 1 / (70 * 365), "etah": 1 / 5, "betah": 0.5}
    vector = {"muv": 1 / 14, "etav": 1 / 10, "betav": 0.4, "q": 0.3}
    bites = {1: 0.8, 2: 0.6}
    recovery = {1: 1 / 7, 2: 1 / 6}
    state = {
        "S1": 800.0,
        "L1": 20.0,
        "I1": 30.0,
        "R1": 150.0,
        "S2": 400.0,
        "L2": 10.0,
        "I2": 15.0,
        "R2": 75.0,
        "U1": 1500.0,
        "W1": 100.0,
        "V1": 80.0,
        "U2": 900.0,
        "W2": 60.0,
        "V2": 40.0,
    }
    hosts_in = {j: " + ".join(_present(x, j) for x in "SLIR") for j in PATCHES}
    flows = []
    for i in PATCHES:
        for j in PATCHES:
            flows.append(
                Flow(
                    f"host_infection_{i}_in_{j}",
                    f"S{i}",
                    f"L{i}",
                    f"betah * a{j} * {TIME_SHARE[i, j]} * S{i} * V{j}"
                    f" / ({hosts_in[j]})",
                )
            )
        flows += [
            Flow(
                f"host_births_{i}", None, f"S{i}", f"muh * (S{i} + L{i} + I{i} + R{i})"
            ),
            Flow(f"host_onset_{i}", f"L{i}", f"I{i}", f"etah * L{i}"),
            Flow(f"host_recovery_{i}", f"I{i}", f"R{i}", f"gamma{i} * I{i}"),
            *(
                Flow(f"host_deaths_{x}{i}", f"{x}{i}", None, f"muh * {x}{i}")
                for x in "SLIR"
            ),
            Flow(
                f"vector_infection_{i}",
                f"U{i}",
                f"W{i}",
                f"betav * U{i} * ({_present('I', i)}) / ({hosts_in[i]})",
            ),
            Flow(
                f"vector_births_{i}",
                None,
                f"U{i}",
                f"muv * (U{i} + W{i} + (1 - q) * V{i})",
            ),
            Flow(f"vertical_{i}", None, f"V{i}", f"q * muv * V{i}"),
            Flow(f"vector_onset_{i}", f"W{i}", f"V{i}", f"etav * W{i}"),
            *(
                Flow(f"vector_deaths_{x}{i}", f"{x}{i}", None, f"muv * {x}{i}")
                for x in "UWV"
            ),
        ]
    parameters = host | vector | {f"a{j}": bites[j] for j in PATCHES}
    parameters |= {f"gamma{i}": recovery[i] for i in PATCHES}

    rates = increments(state, flows, parameters)

    def present(x: str, j: int) -> float:
        return sum(TIME_SHARE[k, j] * state[f"{x}{k}"] for k in PATCHES)

    hosts = {j: sum(present(x, j) for x in "SLIR") for j in PATCHES}
    for i in PATCHES:
        s, e, inf = (state[f"{x}{i}"] for x in "SLI")
        u, w, v = (state[f"{x}{i}"] for x in "UWV")
        n_h = sum(state[f"{x}{i}"] for x in "SLIR")
        n_v = u + w + v
        bitten = (
            host["betah"]
            * s
            * sum(
                bites[j] * TIME_SHARE[i, j] * state[f"V{j}"] / hosts[j] for j in PATCHES
            )
        )
        biting = vector["betav"] * u * present("I", i) / hosts[i]
        mu_h, mu_v, q = host["muh"], vector["muv"], vector["q"]
        expected = {
            f"S{i}": mu_h * n_h - bitten - mu_h * s,
            f"L{i}": bitten - (mu_h + host["etah"]) * e,
            f"I{i}": host["etah"] * e - (mu_h + recovery[i]) * inf,
            f"U{i}": mu_v * (n_v - q * v) - biting - mu_v * u,
            f"W{i}": biting - (mu_v + vector["etav"]) * w,
            f"V{i}": q * mu_v * v + vector["etav"] * w - mu_v * v,
        }
        assert_close({name: rates[name] for name in expected}, expected, rel=ENGINE_REL)
