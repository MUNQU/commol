"""
Early epidemic growth with a power of the infectives in the incidence.

``I' = beta (S / N) I^alpha - gamma I`` with ``0 < alpha < 1``. While
``S`` is near ``N``, ``I' = beta I^alpha - gamma I``: one infective causes
``beta`` infections per unit time for ``1 / gamma``, so ``R0 = beta / gamma``,
and a single infective starts growth only if ``R0 > 1``.

References
----------
.. [1] Liu, W., Levin, S. A., & Iwasa, Y. (1986). Influence of nonlinear
   incidence rates upon the behavior of SIRS epidemiological models. *Journal
   of Mathematical Biology*, 23, 187-204. https://doi.org/10.1007/BF00276956
.. [2] Viboud, C., Simonsen, L., & Chowell, G. (2016). A generalized-growth
   model to characterize the early ascending phase of infectious disease
   outbreaks. *Epidemics*, 15, 27-37. https://doi.org/10.1016/j.epidem.2016.01.002
"""

from commol.context.model import Model
from tests.support import Flow, assert_threshold, flow_model

N, GAMMA, ALPHA = 1e6, 0.2, 0.7


def test_a_single_infective_grows_only_if_beta_exceeds_gamma() -> None:
    flows = [
        Flow("infection", "S", "I", "beta * S / N * I ^ alpha"),
        Flow("recovery", "I", "R", "gamma * I"),
    ]

    def model_at(r0: float) -> Model:
        parameters = {"beta": r0 * GAMMA, "gamma": GAMMA, "alpha": ALPHA}
        return flow_model({"S": N - 1, "I": 1, "R": 0}, flows, parameters)

    # From I = 1 the infectives move monotonically towards R0^(1 / (1 - alpha)).
    assert_threshold(model_at, ["I"], 10)
