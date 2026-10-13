"""
A discrete epidemic with quarantine of the latent and isolation of the
infectious, for arbitrarily distributed stage durations.

``p_i``, ``q_i``, ``k_i`` and ``l_i`` are the probabilities that latency,
infectivity, freedom from quarantine and freedom from isolation last more than
``i`` steps. A bounded distribution is exact as a chain of one-step stages, one
per stage age. Isolated members transmit a fraction ``1 - rho``. The final size
relation is ``log(S0 / S_inf) = Rc (1 - S_inf / N)`` with
``Rc = beta [T_Ek D_Il + (1 - rho) T_Ek (D_I - D_Il) + (1 - rho) (1 - T_Ek) D_I]``,
``D_I = sum q_i``, ``D_Il = sum q_i l_i`` and ``T_Ek = sum (p_i - p_{i+1}) k_i``.
With geometric distributions ``p_i = alpha^i``, ``q_i = delta^i``,
``k_i = gamma^i`` and ``l_i = sigma^i`` the model reduces to five equations.

References
----------
.. [1] Hernandez-Ceron, N., Feng, Z., & Castillo-Chavez, C. (2013). Discrete
   epidemic models with arbitrary stage distributions and applications to
   disease control. *Bulletin of Mathematical Biology*, 75, 1716-1746.
   https://doi.org/10.1007/s11538-013-9866-x
"""

import math

from commol import Simulation
from tests.support import ENGINE_REL, Flow, assert_close, final_size, flow_model

N, E0, BETA, RHO = 1000.0, 10.0, 0.4, 0.6
STEPS = 800


def test_arbitrary_stage_durations_end_with_the_control_final_size_relation() -> None:
    # Latency lasts 2, 3 or 4 steps and infectivity 1 to 4 steps; quarantine
    # and isolation are geometric. L_j and Q_j are the latent of age j, free
    # or quarantined; I_i and H_i the infectious of age i, free or isolated.
    p = [1.0, 1.0, 0.6, 0.2, 0.0]
    q = [1.0, 0.9, 0.7, 0.3, 0.0]
    gamma, sigma = 0.8, 0.7
    infectious = " + ".join(f"I{i} + (1 - rho) * H{i}" for i in range(4))
    flows = [Flow("infection", "S", "L0", f"S * (1 - exp(-beta * ({infectious}) / N))")]
    for j in range(4):
        stay = p[j + 1] / p[j]
        if j < 3:
            flows += [
                Flow(f"latent_{j}", f"L{j}", f"L{j + 1}", f"{stay * gamma} * L{j}"),
                Flow(
                    f"quarantine_{j}",
                    f"L{j}",
                    f"Q{j + 1}",
                    f"{stay * (1 - gamma)} * L{j}",
                ),
            ]
        flows.append(Flow(f"onset_{j}", f"L{j}", "I0", f"{1 - stay} * L{j}"))
        if j > 0:
            if j < 3:
                flows.append(
                    Flow(f"quarantined_{j}", f"Q{j}", f"Q{j + 1}", f"{stay} * Q{j}")
                )
            flows.append(Flow(f"admission_{j}", f"Q{j}", "H0", f"{1 - stay} * Q{j}"))
    for i in range(4):
        stay = q[i + 1] / q[i]
        if i < 3:
            flows += [
                Flow(f"infectious_{i}", f"I{i}", f"I{i + 1}", f"{stay * sigma} * I{i}"),
                Flow(
                    f"isolation_{i}",
                    f"I{i}",
                    f"H{i + 1}",
                    f"{stay * (1 - sigma)} * I{i}",
                ),
                Flow(f"isolated_{i}", f"H{i}", f"H{i + 1}", f"{stay} * H{i}"),
            ]
        flows += [
            Flow(f"recovery_{i}", f"I{i}", "R", f"{1 - stay} * I{i}"),
            Flow(f"isolated_recovery_{i}", f"H{i}", "R", f"{1 - stay} * H{i}"),
        ]
    values = {"S": N - E0, "L0": E0}
    values |= {f"L{j}": 0.0 for j in range(1, 4)} | {f"Q{j}": 0.0 for j in range(1, 4)}
    values |= {f"{x}{i}": 0.0 for x in "IH" for i in range(4)} | {"R": 0.0}
    model = flow_model(values, flows, {"beta": BETA, "rho": RHO}, ode=False)

    results = Simulation(model).run(STEPS)

    k = [gamma**j for j in range(5)]
    t_ek = sum((p[j] - p[j + 1]) * k[j] for j in range(4))
    d_i = sum(q)
    d_il = sum(q[i] * sigma**i for i in range(5))
    rc = BETA * (
        t_ek * d_il + (1 - RHO) * t_ek * (d_i - d_il) + (1 - RHO) * (1 - t_ek) * d_i
    )
    assert_close(results["S"][-1], N * final_size(rc, (N - E0) / N), rel=ENGINE_REL)


def test_geometric_durations_reduce_to_five_equations() -> None:
    # The five equations run in commol; the general model runs here, as the
    # sums over the history of new infections A and new infectives B.
    alpha, delta, gamma, sigma, steps = 0.7, 0.8, 0.9, 0.6, 60
    flows = [
        Flow("infection", "S", "E", "S * (1 - exp(-beta * (I + (1 - rho) * H) / N))"),
        Flow("quarantine", "E", "Q", "alpha * (1 - gamma) * E"),
        Flow("onset", "E", "I", "(1 - alpha) * E"),
        Flow("admission", "Q", "H", "(1 - alpha) * Q"),
        Flow("isolation", "I", "H", "delta * (1 - sigma) * I"),
        Flow("recovery", "I", "R", "(1 - delta) * I"),
        Flow("isolated_recovery", "H", "R", "(1 - delta) * H"),
    ]
    parameters = {
        "beta": BETA,
        "rho": RHO,
        "alpha": alpha,
        "delta": delta,
        "gamma": gamma,
        "sigma": sigma,
    }
    values = {"S": N - E0, "E": E0, "Q": 0, "I": 0, "H": 0, "R": 0}

    results = Simulation(flow_model(values, flows, parameters, ode=False)).run(steps)

    p = [alpha**i for i in range(steps + 2)]
    q = [delta**i for i in range(steps + 2)]
    k = [gamma**i for i in range(steps + 2)]
    l = [sigma**i for i in range(steps + 2)]  # noqa: E741
    s, e, quarantined, i, h = [N - E0], [E0], [0.0], [0.0], [0.0]
    a, b, f = [E0], [0.0], [0.0]
    for n in range(steps):
        escape = math.exp(-BETA / N * (i[n] + (1 - RHO) * h[n]))
        a.append(s[n] * (1 - escape))
        s.append(s[n] * escape)
        e.append(a[n + 1] + sum(a[n - j] * p[j + 1] * k[j + 1] for j in range(n + 1)))
        quarantined.append(
            sum(a[n - j] * p[j + 1] * (1 - k[j + 1]) for j in range(n + 1))
        )
        b.append(sum(a[n - j] * (p[j] - p[j + 1]) * k[j] for j in range(n + 1)))
        i.append(b[n + 1] + sum(b[n + 1 - j] * q[j] * l[j] for j in range(1, n + 1)))
        entered = e[n] - e[n + 1] + a[n + 1] - b[n + 1]
        f.append(b[n + 1] + quarantined[n] + entered - quarantined[n + 1])
        h.append(
            f[n + 1] + sum(f[n + 1 - j] * q[j] for j in range(1, n + 1)) - i[n + 1]
        )
    expected = {"S": s, "E": e, "Q": quarantined, "I": i, "H": h}
    assert_close(
        {name: results[name] for name in expected}, expected, rel=ENGINE_REL, abs=1e-9
    )
