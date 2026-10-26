"""
An epidemic with quarantine of the latent and imperfect isolation of the
infectious, for gamma-distributed stage durations.

``S' = -beta S [I + (1 - sigma) H]``: latent members are quarantined at rate
``psi``, infectives isolated at rate ``phi``, and isolation removes a fraction
``sigma`` of their infectivity. With a mean infective period ``mu_I``,
``Rc = beta N L_qE(psi) [(1 - sigma) mu_I + sigma L_PI(phi)]``, where ``L`` are
Laplace transforms. For gamma distributions of shape ``k`` and mean ``mu``,
``L_qE(psi) = (1 + psi mu_E / k_E)^{-k_E}`` and
``L_PI(phi) = [1 - (1 + mu_I phi / k_I)^{-k_I}] / phi``; an integer shape is a
chain of ``k`` exponential stages. The epidemic ends with
``log(S0 / S_inf) = Rc (1 - S_inf / N)``.

References
----------
.. [1] Feng, Z., Xu, D., & Zhao, H. (2007). Epidemiological models with
   non-exponentially distributed disease stages and applications to disease
   control. *Bulletin of Mathematical Biology*, 69, 1511-1536.
   https://doi.org/10.1007/s11538-006-9174-9
"""

import pytest

from commol._model.model import Model
from tests.support import ODE_REL, Flow, assert_close, final_size, flow_model, settled

N, E0 = 1000.0, 10.0
MEAN_E, MEAN_I, BETA_N = 2.0, 5.0, 0.8


def _epidemic(k_e: int, k_i: int, psi: float, phi: float, sigma: float) -> Model:
    """Chains of ``k_e`` latent stages L and ``k_i`` infective stages I and H [1]_."""
    flows = []
    for j in range(1, k_e + 1):
        after = f"L{j + 1}" if j < k_e else "I1"
        flows += [
            Flow(f"latent_{j}", f"L{j}", after, f"{k_e / MEAN_E} * L{j}"),
            Flow(f"quarantine_{j}", f"L{j}", "Q", f"psi * L{j}"),
        ]
    for j in range(1, k_i + 1):
        free, isolated = (f"I{j + 1}", f"H{j + 1}") if j < k_i else ("R", "R")
        flows += [
            Flow(f"infectious_{j}", f"I{j}", free, f"{k_i / MEAN_I} * I{j}"),
            Flow(f"isolated_{j}", f"H{j}", isolated, f"{k_i / MEAN_I} * H{j}"),
            Flow(f"isolation_{j}", f"I{j}", f"H{j}", f"phi * I{j}"),
        ]
    infectious = " + ".join(f"I{j} + (1 - sigma) * H{j}" for j in range(1, k_i + 1))
    flows.append(Flow("infection", "S", "L1", f"beta * S * ({infectious})"))
    values = {"S": N - E0, "L1": E0} | {f"L{j}": 0.0 for j in range(2, k_e + 1)}
    values |= {f"{x}{j}": 0.0 for x in "IH" for j in range(1, k_i + 1)}
    values |= {"Q": 0.0, "R": 0.0}
    parameters = {"beta": BETA_N / N, "psi": psi, "phi": phi, "sigma": sigma}
    return flow_model(values, flows, parameters)


def _reproduction_number(
    k_e: int, k_i: int, psi: float, phi: float, sigma: float
) -> float:
    not_quarantined = (1 + psi * MEAN_E / k_e) ** -k_e
    before_isolation = (1 - (1 + MEAN_I * phi / k_i) ** -k_i) / phi
    return BETA_N * not_quarantined * ((1 - sigma) * MEAN_I + sigma * before_isolation)


def test_quarantine_and_imperfect_isolation_set_the_final_size() -> None:
    shape = {"k_e": 2, "k_i": 3, "psi": 0.1, "phi": 0.2, "sigma": 0.6}

    limit = settled(_epidemic(**shape), 1500)

    rc = _reproduction_number(**shape)
    assert_close(limit["S"], N * final_size(rc, (N - E0) / N), rel=ODE_REL)


@pytest.mark.parametrize("k", [1, 2, 4])
def test_without_quarantine_the_gamma_shape_sets_the_final_size(k: int) -> None:
    shape = {"k_e": k, "k_i": k, "psi": 0.0, "phi": 0.2, "sigma": 0.8}

    limit = settled(_epidemic(**shape), 1500)

    rc = _reproduction_number(**shape)
    assert_close(limit["S"], N * final_size(rc, (N - E0) / N), rel=ODE_REL)
