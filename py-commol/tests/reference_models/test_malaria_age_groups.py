"""
Malaria with two host age groups, SIS in the hosts and SI in the mosquitoes.

Hosts are born into group 1 at ``Lambda`` and age into group 2 at ``alpha1``;
group ``k`` dies at ``mu_k`` and recovers at ``gamma_k``. A mosquito bites at
``b``: hosts of group ``k`` are infected at ``b beta_k I_v / N`` and mosquitoes
at ``b sum_k beta_vk I_k / N``. Without infection the age groups settle at
``N_1 = Lambda / (alpha1 + mu1)`` and ``N_2 = Lambda alpha1 / ((alpha1 + mu1) mu2)``,
and the disease-free equilibrium is stable when the spectral radius of the
next generation matrix ``F V^-1`` of the infected classes is below one.

References
----------
.. [1] Chitnis, N., Hyman, J. M., & Cushing, J. M. (2008). Determining
   important parameters in the spread of malaria through the sensitivity
   analysis of a mathematical model. *Bulletin of Mathematical Biology*, 70,
   1272-1296. https://doi.org/10.1007/s11538-008-9299-0
"""

import numpy as np

from commol.context.model import Model
from tests.support import (
    ODE_REL,
    Flow,
    assert_close,
    assert_threshold,
    flow_model,
    settled,
)

# Per day, with 45 % of the hosts in group 1; alpha1 is aging at 15 years.
# The infectivities and the recovery rates are those of [1].
N = 10000.0
ALPHA1, MU1 = 1 / (15 * 365), 0.0002
LAMBDA = 0.45 * N * (MU1 + ALPHA1)
MU2 = ALPHA1 * 0.45 / 0.55
GAMMA = {1: 1 / 100, 2: 1 / 80}
BETA = {1: 0.03, 2: 0.015}
BETA_V = {1: 0.3, 2: 0.24}
M_V, MU_V = 150.0, 0.03
N_V = M_V / MU_V


def _malaria(b: float, values: dict[str, float]) -> Model:
    """[1]_ in two age groups; ``N`` is the host total at the disease-free state."""
    flows = [
        Flow("births", None, "S1", "Lam"),
        Flow("aging_s", "S1", "S2", "alpha1 * S1"),
        Flow("aging_i", "I1", "I2", "alpha1 * I1"),
        Flow("mosquito_births", None, "U", "Mv"),
        Flow("mosquito_infection", "U", "V", "b * (bv1 * I1 + bv2 * I2) / Nh * U"),
        Flow("mosquito_deaths_u", "U", None, "muv * U"),
        Flow("mosquito_deaths_v", "V", None, "muv * V"),
    ]
    for k in (1, 2):
        flows += [
            Flow(f"infection_{k}", f"S{k}", f"I{k}", f"b * beta{k} * V / Nh * S{k}"),
            Flow(f"recovery_{k}", f"I{k}", f"S{k}", f"gamma{k} * I{k}"),
            Flow(f"deaths_s{k}", f"S{k}", None, f"mu{k} * S{k}"),
            Flow(f"deaths_i{k}", f"I{k}", None, f"mu{k} * I{k}"),
        ]
    parameters = {
        "Lam": LAMBDA,
        "alpha1": ALPHA1,
        "mu1": MU1,
        "mu2": MU2,
        "b": b,
        "Nh": N,
        "Mv": M_V,
        "muv": MU_V,
    }
    for k in (1, 2):
        parameters |= {f"gamma{k}": GAMMA[k], f"beta{k}": BETA[k], f"bv{k}": BETA_V[k]}
    return flow_model(values, flows, parameters)


N1 = LAMBDA / (ALPHA1 + MU1)
N2 = LAMBDA * ALPHA1 / ((ALPHA1 + MU1) * MU2)


def test_without_infection_the_age_groups_settle_at_the_stated_sizes() -> None:
    start = {"S1": 2000.0, "I1": 0.0, "S2": 8000.0, "I2": 0.0, "U": N_V, "V": 0.0}

    limit = settled(_malaria(0.2, start), 2e5)

    assert_close(
        {"S1": limit["S1"], "S2": limit["S2"]}, {"S1": N1, "S2": N2}, rel=ODE_REL
    )


def _next_generation_radius(b: float) -> float:
    """``rho(F V^-1)`` of (I1, I2, V) at the disease-free state."""
    new = np.array(
        [
            [0, 0, b * BETA[1] * N1 / N],
            [0, 0, b * BETA[2] * N2 / N],
            [b * BETA_V[1] * N_V / N, b * BETA_V[2] * N_V / N, 0],
        ]
    )
    leaving = np.array(
        [
            [GAMMA[1] + MU1 + ALPHA1, 0, 0],
            [-ALPHA1, GAMMA[2] + MU2, 0],
            [0, 0, MU_V],
        ]
    )
    return float(max(abs(np.linalg.eigvals(new @ np.linalg.inv(leaving)))))


def test_the_disease_free_state_is_stable_below_the_threshold() -> None:
    unit = _next_generation_radius(1.0)
    small = {"S1": N1 - 1e-6, "I1": 1e-6, "S2": N2, "I2": 0.0, "U": N_V, "V": 0.0}

    assert_threshold(lambda r: _malaria(r / unit, small), ["I1", "I2", "V"], 2e4)
