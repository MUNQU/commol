"""The one rule of a singular curvature of the methods that invert it: the
Laplace approximation, the sandwich, the Laplace mixture and information
matching. What the data determine does not depend on the scale of a parameter,
and a covariance of what they determine is refused only when it is too large or
too small for a number."""

import re

import numpy as np
import pytest

from commol import CalibrationProblem, Calibrator, NelderMeadConfig, Simulation
from commol.distributions import Priors
from commol.local import Laplace, LaplaceMixture, LikelihoodTarget, Sandwich
from commol.observations import NormalObservations
from commol.smc import GibbsPosterior, InformationMatchingEta
from tests.support import (
    a_to_b_builder,
    bounded,
    build,
    flow_observations,
    weak_parameter_calibrator,
)

PRIORS = Priors.uniform_on_bounds()
METHODS = {
    "laplace": Laplace(draws=2),
    "sandwich": Sandwich(block_length=2, draws=2),
    "mixture": LaplaceMixture(draws=25, target=LikelihoodTarget(priors=PRIORS)),
    "information matching": GibbsPosterior(
        priors=PRIORS,
        eta=InformationMatchingEta(block_length=2),
        particles=50,
        target_cess=0.5,
        resample_threshold=0.5,
        stay_probability=0.1,
    ),
}
SINGULAR = re.escape(
    "The curvature at the fit is not positive definite: the data do not determine "
    "every parameter there, or the fit is not a minimum. Calibrator.identifiability"
    "(...) shows which parameters the data determine"
)
REFUSALS = {name: f"^{SINGULAR}$" for name in METHODS} | {
    "mixture": "^No end of the starts gives a component of the mixture: "
    rf"The end \[[^\]]*\] gives no component: {SINGULAR}$"
}
"""The whole message of each method for a curvature the data do not determine,
from the one end of the mixture that its starts reach."""
every_method = pytest.mark.parametrize("name", list(METHODS))


def _product(count: int, own: float = 0.0) -> Calibrator:
    """
    ``A → B`` at the rate ``k1 · k2 · A``, read at `count` positions with noise
    of sd 5: the data read ``k1`` and ``k2`` only as their product. An `own` above
    0 adds the flow ``own · k2 · A · step``, which tells them apart.
    """
    values = 1000 * (1 - np.exp(-0.005 * np.arange(1, count + 1)))
    values = values + np.random.default_rng(1).normal(0, 5, count)
    model = build(
        a_to_b_builder(
            parameters={"k1": None, "k2": None},
            rate=f"k1 * k2 * A + {own!r} * k2 * A * step",
        )
    )
    problem = CalibrationProblem(
        observed_data=flow_observations(np.maximum(values, 0).tolist(), windowed=False),
        parameters=[
            bounded("k1", 0.01, 1.0, initial_guess=0.3),
            bounded("k2", 0.01, 1.0, initial_guess=0.3),
        ],
        observation_models={"B": NormalObservations(sd=5.0)},
        optimization_config=NelderMeadConfig(max_iterations=400),
        seed=1,
    )
    return Calibrator(Simulation(model), problem)


@every_method
def test_a_parameter_the_data_do_not_read_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match=REFUSALS[name]):
        weak_parameter_calibrator(0.0).uncertainty(METHODS[name])


@every_method
@pytest.mark.parametrize("count", [20, 200])
def test_parameters_read_through_their_product_are_refused(
    name: str, count: int
) -> None:
    # At these numbers of observations the finite differences are exact enough
    # for the direction that the data do not read to have an eigenvalue of 0
    # but for rounding. With many more, their error can hide it, which the rule
    # does not read.
    with pytest.raises(ValueError, match=REFUSALS[name]):
        _product(count).uncertainty(METHODS[name])


@pytest.mark.parametrize("name", ["laplace", "sandwich"])
def test_parameters_told_apart_by_little_are_not_refused(name: str) -> None:
    # The derivatives of k1 and k2 differ in direction by about 1e-4: the
    # smallest eigenvalue of the equilibrated curvature is about 1e-8.
    result = _product(20, own=1e-5).uncertainty(METHODS[name])

    assert np.all(np.linalg.eigvalsh(result.covariance) > 0)


def _own_bin_variance(k2: float) -> float:
    """
    The sandwich variance of ``k2`` on the unbounded scale for
    ``weak_parameter_calibrator(scale, own_bin=True)`` in blocks of 2.

    D at ``t`` is ``k2 · scale · S_t``, ``S_t = t(t − 1)/2``, observed at 0 with
    sd 1, and no other observation reads ``k2``: the score of block ``b`` over
    the curvature is ``k2 Σ_b S_t² / Σ S_t²`` at any scale, and ``dk2/du`` is
    ``k2 (1 − k2)``.
    """
    squares = np.array([t * (t - 1) / 2 for t in range(1, 21)]) ** 2
    shares = k2 * squares.reshape(10, 2).sum(axis=1) / squares.sum()
    return 10 / 9 * float(np.sum(shares**2)) / (k2 * (1 - k2)) ** 2


@pytest.mark.parametrize("scale", [1e-6, 1e-160, 1e-300])
def test_a_sandwich_does_not_depend_on_the_scale_of_a_parameter(scale: float) -> None:
    # The scores and the curvature of k2 change with the scale, and the
    # covariance H⁻¹ B H⁻¹ does not: the curvature of k2 on the unbounded
    # scale, about 8e3 scale², is below the smallest normal number at 1e-160
    # and below any number at 1e-300.
    result = weak_parameter_calibrator(scale, own_bin=True).uncertainty(
        METHODS["sandwich"]
    )

    k2 = result.fit.best_parameters["k2"]
    assert result.covariance[1][1] == pytest.approx(_own_bin_variance(k2), rel=1e-9)


@pytest.mark.parametrize("name", ["laplace", "mixture"])
def test_a_variance_above_the_largest_number_is_refused(name: str) -> None:
    # The variance of k2 is of the order of 1e311.
    with pytest.raises(ValueError, match="The covariance is not positive definite"):
        weak_parameter_calibrator(1e-160).uncertainty(METHODS[name])
