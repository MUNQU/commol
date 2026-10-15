"""
Tuberculosis with a drug-sensitive and a drug-resistant strain.

Latent ``L_i`` and infectious ``I_i`` of strain ``i``; treated ``T``. Latent
sensitive cases are treated at ``r1``, infectious ones at ``r2``, and a fraction
``p`` of the treated infectious fail and stay latent. The disease-free state
loses stability when one of
``R_S = (c1 + p r2) / (mu + r2) kappa1 / (mu + kappa1 + r1)`` and
``R_R = c2 / mu kappa2 / (mu + kappa2)`` exceeds one.

References
----------
.. [1] Castillo-Chavez, C., & Feng, Z. (1997). To treat or not to treat: the
   case of tuberculosis. *Journal of Mathematical Biology*, 35, 629-656.
   https://doi.org/10.1007/s002850050069
"""

import pytest

from commol.context.model import Model
from tests.support import Flow, assert_threshold, flow_model

# Per year, without treatment failure to the resistant strain (q = 0).
MU, KAPPA1, KAPPA2, R1, R2, P, Q = 0.0143, 1.0, 1.0, 1.0, 2.0, 0.5, 0.0
N = 1000.0


def _tuberculosis(c1: float, c2: float, values: dict[str, float]) -> Model:
    flows = [
        Flow("births", None, "S", "mu * total"),
        Flow("sensitive", "S", "L1", "c1 * S * I1 / total"),
        Flow("resistant", "S", "L2", "c2 * S * I2 / total"),
        Flow("superinfection", "L1", "L2", "c2 * L1 * I2 / total"),
        Flow("treated_sensitive", "T", "L1", "cstar * T * I1 / total"),
        Flow("treated_resistant", "T", "L2", "c2 * T * I2 / total"),
        Flow("activation1", "L1", "I1", "kappa1 * L1"),
        Flow("prophylaxis", "L1", "T", "r1 * L1"),
        Flow("failure", "I1", "L1", "p * r2 * I1"),
        Flow("acquired", "I1", "L2", "q * r2 * I1"),
        Flow("cure", "I1", "T", "(1 - p - q) * r2 * I1"),
        Flow("activation2", "L2", "I2", "kappa2 * L2"),
        *(
            Flow(f"deaths_{x}", x, None, f"mu * {x}")
            for x in ("S", "L1", "I1", "L2", "I2", "T")
        ),
    ]
    parameters = {
        "mu": MU,
        "c1": c1,
        "c2": c2,
        "cstar": 0.75 * c1,
        "kappa1": KAPPA1,
        "kappa2": KAPPA2,
        "r1": R1,
        "r2": R2,
        "p": P,
        "q": Q,
    }
    return flow_model(values, flows, parameters)


def _start(strain: int) -> dict[str, float]:
    values = {"S": N - 1e-6, "L1": 0.0, "I1": 0.0, "L2": 0.0, "I2": 0.0, "T": 0.0}
    return values | {f"I{strain}": 1e-6}


@pytest.mark.parametrize("strain", [1, 2])
def test_each_strain_invades_above_its_reproduction_number(strain: int) -> None:
    def model_at(r: float) -> Model:
        if strain == 1:
            c1 = r * (MU + R2) * (MU + KAPPA1 + R1) / KAPPA1 - P * R2
            return _tuberculosis(c1, 0.0, _start(1))
        return _tuberculosis(13.0, r * MU * (MU + KAPPA2) / KAPPA2, _start(2))

    infected = ["L1", "I1"] if strain == 1 else ["L2", "I2"]
    assert_threshold(model_at, infected, 2e4 if strain == 2 else 400)
