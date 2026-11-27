"""The local approximations around a fit: the Laplace approximation, the
sandwich and the Laplace mixture, their settings and their results.

Each reads the curvature of the objective at the fit on the unbounded scale
of the parameters (the logit of a parameter between two bounds, a log with
one bound, the identity with none), so every draw maps back inside the
bounds. A curvature that the data do not determine is refused.
"""

from typing import Annotated, Literal

from pydantic import Field

from commol._calibration.method import UncertaintyMethod
from commol._data.ranges import RangeChecked
from commol.distributions.distributions import WithPriors
from commol.local.results import (
    ImportanceResult,
    LaplaceResult,
    SandwichResult,
)
from commol.local.runners import run_laplace, run_laplace_mixture, run_sandwich


class Laplace(UncertaintyMethod[LaplaceResult], RangeChecked):
    """
    The Laplace approximation of the problem's observation models: the normal
    ``N(θ̂, H⁻¹)`` on the unbounded scale of the parameters, centred on the
    fit, with ``H`` the curvature of ``−log L`` there.

    It needs ``observation_models``: a loss has no noise scale, so its
    curvature is no inverse variance (use `Sandwich`). A fit on a bound of
    its parameter is refused: on the unbounded scale its variance is
    infinite there. A bound less than 3 sd from the fit is a note: the draws,
    mapped back, are skewed by it.

    Attributes
    ----------
    draws : int
        Number of draws of the normal, at least 1.
    curvature : {"gauss_newton", "hessian"}
        The curvature (default ``"gauss_newton"``).
    starts : int
        Starts of the fit (default 1): the initial guess, then rows of a Latin
        hypercube over the bounds.
    """

    _runner = staticmethod(run_laplace)

    draws: int = Field(description="Number of draws.")
    curvature: Literal["gauss_newton", "hessian"] = Field(
        default="gauss_newton", description="The curvature."
    )
    starts: int = Field(default=1, description="Starts of the fit.")


class Sandwich(UncertaintyMethod[SandwichResult], RangeChecked):
    """
    The sandwich covariance at the fit, with the meat summed over blocks of
    positions: ``√(f_j f_k) · (H⁻¹ Σ_b s̃_b s̃_bᵀ H⁻¹)_jk``, with ``s_b`` the
    sum of the gradients of the terms of the observations of block ``b``,
    taken after one Gauss–Newton step to the zero of their sum ``g``:
    ``s̃_b = s_b − J_b J⁻¹ g``. The factor ``f_j`` and the degrees of freedom
    ``ν_j`` of each parameter come from the share ``J_b`` of its Gauss–Newton
    curvature ``J`` that each block holds (Bell and McCaffrey, 2002): with
    equal shares, ``f_j = G/(G − 1)`` and ``ν_j = G − 1`` for ``G`` blocks; a
    parameter that few blocks read has fewer, and a block with no curvature
    counts for nothing.

    It needs no noise model: with a loss, ``H`` is the curvature of the loss;
    with observation models, of ``−log L``. It holds when the noise depends
    on its neighbours inside a block, when its size changes, and when the
    noise model is wrong but the mean is right. Intervals read the Student t
    of each parameter, and draws a grouped t whose margins are those t. With
    no more blocks with curvature than parameters it is refused, since the
    meat would have rank below them; so it is when the blocks each hold all
    the curvature of a parameter. ``loss_function="mae"`` is refused: its
    sandwich needs a density of the residuals at 0 and a bandwidth (use a
    bootstrap). A bound less than 3 sd from the fit is a note: the draws,
    mapped back, are skewed by it. The distance of its Student t is that of
    a normal with the same mass past the bound, so with few degrees of
    freedom a bound beyond 3 units of the scale of the t can be noted.

    Attributes
    ----------
    block_length : int
        Positions per block of the meat; they must make more blocks with
        curvature than parameters.
    draws : int
        Number of draws of the grouped t, at least 1.
    curvature : {"gauss_newton", "hessian"}
        The curvature ``H`` (default ``"gauss_newton"``). With the Hessian,
        the sandwich also holds when the mean is wrong.
    starts : int
        Starts of the fit (default 1): the initial guess, then rows of a Latin
        hypercube over the bounds.
    """

    _runner = staticmethod(run_sandwich)

    block_length: int = Field(description="Positions per block.")
    draws: int = Field(description="Number of draws.")
    curvature: Literal["gauss_newton", "hessian"] = Field(
        default="gauss_newton", description="The curvature."
    )
    starts: int = Field(default=1, description="Starts of the fit.")


class LikelihoodTarget(WithPriors):
    """
    The posterior of the problem's observation models,
    ``π(θ) ∝ prior(θ) · L(θ)``, as the target of a `LaplaceMixture`.

    Attributes
    ----------
    priors : Priors | dict[str, distribution]
        The prior of each calibrated parameter, or
        ``Priors.uniform_on_bounds()``.
    """

    kind: Literal["likelihood"] = "likelihood"


class GibbsTarget(WithPriors, RangeChecked):
    """
    The Gibbs posterior of the problem's loss,
    ``π(θ) ∝ prior(θ) · exp(−η · loss(θ))``, as the target of a
    `LaplaceMixture`.

    Attributes
    ----------
    priors : Priors | dict[str, distribution]
        The prior of each calibrated parameter, or
        ``Priors.uniform_on_bounds()``.
    eta : float
        The scale of the loss, above 0.
    """

    kind: Literal["gibbs"] = "gibbs"
    eta: float = Field(description="Scale of the loss.")


type _MixtureTarget = Annotated[
    LikelihoodTarget | GibbsTarget, Field(discriminator="kind")
]
"""The posterior a `LaplaceMixture` samples."""


class LaplaceMixture(UncertaintyMethod[ImportanceResult], RangeChecked):
    """
    The Laplace mixture: a Laplace approximation at each distinct optimum of
    a multi-start fit, weighted by its Laplace evidence, as the proposal of
    importance sampling of a posterior, with the largest weights smoothed
    (Pareto-smoothed importance sampling, Vehtari et al. 2024).

    The ends of the starts merge into a component when they are closer than
    one Mahalanobis unit to it. Every draw is weighted by the target over
    the mixture, on the unbounded scale. ``k_hat`` says whether the mixture
    covers the target: above 0.7 the sample is marked unreliable. A bound
    less than 3 sd from the centre of a component is a note: it skews the
    component, which the importance weights correct as far as ``k_hat``
    says.

    Attributes
    ----------
    starts : int
        Starts of the fit (default 1): the initial guess, then rows of a Latin
        hypercube over the bounds.
    draws : int
        Number of draws of the mixture, at least 25, so that the tail of the
        weights has 5.
    target : LikelihoodTarget | GibbsTarget
        The posterior of the observation models with priors, or the Gibbs
        posterior of the loss with priors and ``eta``.
    curvature : {"gauss_newton", "hessian"}
        The curvature of each component (default ``"gauss_newton"``).
    """

    _runner = staticmethod(run_laplace_mixture)

    starts: int = Field(default=1, description="Starts of the fit.")
    draws: int = Field(description="Number of draws.")
    target: _MixtureTarget = Field(description="The posterior to sample.")
    curvature: Literal["gauss_newton", "hessian"] = Field(
        default="gauss_newton", description="The curvature."
    )
