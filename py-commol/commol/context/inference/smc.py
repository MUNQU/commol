"""The sequential Monte Carlo methods: the Gibbs posterior of the loss, ABC by
sequential Monte Carlo and ABC rejection, their settings and their results.

One sampler runs every method: it starts from the priors, moves through a
sequence of levels (a temperature for the Gibbs posterior, a tolerance for
ABC) and keeps a weighted sample of particles. None of them needs a noise
model.
"""

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from commol.context.calibration import CalibrationResult
from commol.context.inference.distributions import WithPriors
from commol.context.inference.method import UncertaintyMethod
from commol.context.inference.resampling import BlockLength
from commol.context.inference.sample import ParameterSample

ETA_CHOICES = (
    "a number above 0, EtaCalibration.information_matching() or "
    "EtaCalibration.bootstrap_coverage(replicates, block_length, confidence_level)"
)


class EtaCalibration(BaseModel):
    """
    A rule that chooses the scale `η` of a Gibbs posterior from the data.

    Use `information_matching` or `bootstrap_coverage`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def information_matching(cls) -> "InformationMatching":
        """
        Match the curvature of the loss to the variance of its terms at the
        fit (Holmes and Walker 2017).

        It needs ``loss_function="sse"`` and a deterministic engine, and is
        cheap and local.

        Returns
        -------
        InformationMatching
            The calibration.
        """
        return InformationMatching()

    @classmethod
    def bootstrap_coverage(
        cls, replicates: int, block_length: BlockLength, confidence_level: float
    ) -> "BootstrapCoverage":
        """
        Choose `η` so that the posterior intervals of Bayesian-bootstrap data
        sets cover the fit at their level (Syring and Martin 2019).

        It works with any loss and engine, and costs one sampler per
        replicate.

        Parameters
        ----------
        replicates : int
            Number of Bayesian-bootstrap data sets, at least 1.
        block_length : int | "auto"
            Positions per block of the bootstrap.
        confidence_level : float
            The level of the intervals, strictly between 0 and 1.

        Returns
        -------
        BootstrapCoverage
            The calibration.
        """
        return BootstrapCoverage(
            replicates=replicates,
            block_length=block_length,
            confidence_level=confidence_level,
        )


class InformationMatching(EtaCalibration):
    """
    `η = tr(J I⁻¹ J) / tr(J)` at the fit, with ``J`` the curvature of the
    loss terms and ``I`` their variance (Holmes and Walker 2017; Lyddon,
    Holmes and Walker 2019). For the sum of squares of a model with Gaussian
    noise of variance ``σ²`` it estimates ``1/(2σ²)``. It needs
    ``loss_function="sse"`` and a deterministic engine.
    """

    kind: Literal["information_matching"] = "information_matching"


class BootstrapCoverage(EtaCalibration):
    """
    The `η` at which the central intervals of `confidence_level` of the Gibbs
    posteriors of Bayesian-bootstrap data sets hold the fit at that rate,
    over the replicates and the parameters (Syring and Martin 2019).

    Attributes
    ----------
    replicates : int
        Number of Bayesian-bootstrap data sets, at least 1.
    block_length : int | "auto"
        Positions per block of the bootstrap.
    confidence_level : float
        The level of the intervals, strictly between 0 and 1.
    """

    kind: Literal["bootstrap_coverage"] = "bootstrap_coverage"
    replicates: int = Field(ge=1, description="Number of bootstrap data sets.")
    block_length: BlockLength = Field(description="Positions per block.")
    confidence_level: float = Field(
        gt=0.0, lt=1.0, description="The level of the intervals."
    )


type Eta = (
    Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
    | Annotated[InformationMatching | BootstrapCoverage, Field(discriminator="kind")]
)
"""The scale of a Gibbs posterior: a number above 0, or a calibration."""


class LossDistance(BaseModel):
    """
    The problem's loss as the distance of ABC: ``loss_function`` with the
    observation weights and ``normalize_observations``. With a stochastic
    engine it reads one fresh replicate per evaluation, so the problem's
    ``replicates`` and ``replicate_summary`` do not apply.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["loss"] = "loss"


class SeriesDistance(BaseModel):
    """
    The largest distance over the series: `metric` of each series on its
    residuals divided by the root mean square of its observed values, so
    series in different units compare on one scale.

    Attributes
    ----------
    metric : {"rmse", "mae"}
        The weighted root mean square or mean absolute value of the scaled
        residuals of a series.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["series"] = "series"
    metric: Literal["rmse", "mae"] = Field(description="The metric of a series.")


type AbcDistance = Annotated[LossDistance | SeriesDistance, Field(discriminator="kind")]
"""How far a run is from the observations."""


class SmcDiagnostics(BaseModel):
    """
    What happened at each step of a sequential Monte Carlo run.

    Attributes
    ----------
    effective_sizes : list[float]
        Kish's effective size of the weights after the reweighting of each
        step. A step that keeps few particles has a small one.
    resampled : list[bool]
        Whether each step resampled.
    acceptance_rates : list[float]
        The share of the proposals of the first move of each step that were
        accepted; for ABC rejection, the share of draws kept.
    move_counts : list[int]
        How many times each particle moved in each step:
        ``⌈log c / log(1 − a)⌉`` with ``c`` the stay probability and ``a``
        the acceptance rate.
    evaluations : int
        The number of model runs.
    stop_reason : str | None
        Why the run stopped before its last level, or None.
    notes : list[str]
        What the reader of the sample should know, such as settings of the
        problem that do not apply.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    effective_sizes: list[float]
    resampled: list[bool]
    acceptance_rates: list[float]
    move_counts: list[int]
    evaluations: int
    stop_reason: str | None
    notes: list[str]


class SmcResult(BaseModel):
    """
    The result of a sequential Monte Carlo method.

    Attributes
    ----------
    sample : ParameterSample
        The weighted particles at the last level.
    levels : list[float]
        The level of each step: the temperature of a Gibbs posterior, from
        above 0 to 1, or the tolerance of ABC, falling.
    kept : list[float | None]
        The value each particle of the sample keeps, in its order: the loss
        of a Gibbs posterior, the log-likelihood of a Bayesian posterior, or
        the distance of ABC; None for a particle outside the support.
    log_evidence : float
        The log of the product of the mean incremental weights: for a Gibbs
        posterior, the log of ``∫ prior(θ) exp(−η loss(θ)) dθ``; for a
        Bayesian posterior, the log marginal likelihood
        ``log ∫ prior(θ) L(θ) dθ``; for ABC, the log of the prior probability
        of the last tolerance.
    diagnostics : SmcDiagnostics
        The effective sizes, acceptance rates and move counts of each step.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, ser_json_inf_nan="constants")

    sample: ParameterSample
    levels: list[float]
    kept: list[float | None]
    log_evidence: float
    diagnostics: SmcDiagnostics


class GibbsResult(SmcResult):
    """
    The result of a Gibbs posterior.

    Attributes
    ----------
    eta : float
        The `η` the posterior ran with.
    fit : CalibrationResult | None
        The fit a calibration of `η` read; None when `η` is a number.
    coverage : float | None
        With `BootstrapCoverage`, the coverage of the fit by the intervals of
        the bootstrap posteriors at `η`; otherwise None.
    """

    eta: float
    fit: CalibrationResult | None
    coverage: float | None


def _require(data: object, name: str, hint: str) -> None:
    if isinstance(data, Mapping) and data.get(name) is None:
        raise ValueError(f"{name} is required: {hint}.")


class GibbsPosterior(WithPriors, UncertaintyMethod[GibbsResult]):
    """
    The Gibbs posterior ``π(θ) ∝ prior(θ) · exp(−η · loss(θ))`` of the
    problem's loss, with its weights and normalization, sampled by sequential
    Monte Carlo from the priors (temperature 0) to the posterior
    (temperature 1).

    `η` sets how much the data weigh against the priors; the loss has no
    scale of its own, so `η` has no default.

    Attributes
    ----------
    priors : Priors | dict[str, Distribution]
        The prior of each calibrated parameter, or
        ``Priors.uniform_on_bounds()``.
    eta : float | InformationMatching | BootstrapCoverage
        A number above 0, ``EtaCalibration.information_matching()`` or
        ``EtaCalibration.bootstrap_coverage(...)``.
    particles : int
        Number of particles, at least 2.
    target_cess : float
        The conditional effective sample size of each step, as a share of
        the particles, strictly between 0 and 1: the next temperature is the
        highest that keeps it.
    resample_threshold : float
        Resample when the effective size falls below this share of the
        particles, in ``(0, 1]``.
    stay_probability : float
        The probability that a particle does not move in a step, strictly
        between 0 and 1: each step moves every particle
        ``⌈log c / log(1 − a)⌉`` times.
    """

    eta: Eta = Field(description="The scale of the loss, or its calibration.")
    particles: int = Field(ge=2, description="Number of particles.")
    target_cess: float = Field(gt=0.0, lt=1.0, description="Share kept per step.")
    resample_threshold: float = Field(
        gt=0.0, le=1.0, description="Resample below this share."
    )
    stay_probability: float = Field(
        gt=0.0, lt=1.0, description="Probability a particle does not move."
    )

    @model_validator(mode="before")
    @classmethod
    def require_eta(cls, data: object) -> object:
        """Name the choices of `eta` when it is missing."""
        _require(data, "eta", f"give {ETA_CHOICES}")
        if isinstance(data, Mapping):
            eta = data["eta"]
            if isinstance(eta, int | float) and not eta > 0:
                raise ValueError(f"eta must be {ETA_CHOICES}; got {eta}.")
        return data


class BayesianPosterior(WithPriors, UncertaintyMethod[SmcResult]):
    """
    The posterior ``π(θ) ∝ prior(θ) · L(θ)`` of the problem's observation
    models, sampled by sequential Monte Carlo from the priors (temperature
    0) to the posterior (temperature 1), with its log evidence.

    It needs a problem with ``observation_models``: their likelihood sets the
    weight of the data, so it has no `η`. The noise parameters are sampled
    with the others, so each needs a prior.

    Attributes
    ----------
    priors : Priors | dict[str, Distribution]
        The prior of each calibrated parameter, or
        ``Priors.uniform_on_bounds()``.
    particles : int
        Number of particles, at least 2.
    target_cess : float
        The conditional effective sample size of each step, as a share of
        the particles, strictly between 0 and 1: the next temperature is the
        highest that keeps it.
    resample_threshold : float
        Resample when the effective size falls below this share of the
        particles, in ``(0, 1]``.
    stay_probability : float
        The probability that a particle does not move in a step, strictly
        between 0 and 1: each step moves every particle
        ``⌈log c / log(1 − a)⌉`` times.
    """

    particles: int = Field(ge=2, description="Number of particles.")
    target_cess: float = Field(gt=0.0, lt=1.0, description="Share kept per step.")
    resample_threshold: float = Field(
        gt=0.0, le=1.0, description="Resample below this share."
    )
    stay_probability: float = Field(
        gt=0.0, lt=1.0, description="Probability a particle does not move."
    )


class AbcSmc(WithPriors, UncertaintyMethod[SmcResult]):
    """
    Approximate Bayesian computation by sequential Monte Carlo (Del Moral,
    Doucet and Jasra 2012): the tolerance ``ε`` falls step by step, and a
    particle is inside when the distance of its run is at most ``ε``.

    Each next tolerance keeps the weighted share `target_cess` of the
    particles: with weights of 0 or 1, the weighted quantile of their
    distances. With a stochastic engine each distance reads one fresh
    replicate, which the particle keeps until a move replaces it. The run
    stops when the first move of a step accepts less than `min_acceptance`,
    and keeps the sample of the tolerance before.

    Attributes
    ----------
    priors : Priors | dict[str, Distribution]
        The prior of each calibrated parameter, or
        ``Priors.uniform_on_bounds()``.
    distance : LossDistance | SeriesDistance
        How far a run is from the observations.
    particles : int
        Number of particles, at least 2.
    target_cess : float
        The weighted share of the particles each tolerance keeps, strictly
        between 0 and 1.
    min_acceptance : float
        The acceptance rate below which the run stops, strictly between 0
        and 1.
    stay_probability : float
        The probability that a particle does not move in a step, strictly
        between 0 and 1.
    """

    distance: AbcDistance = Field(description="The distance.")
    particles: int = Field(ge=2, description="Number of particles.")
    target_cess: float = Field(gt=0.0, lt=1.0, description="Share kept per step.")
    min_acceptance: float = Field(
        gt=0.0, lt=1.0, description="Acceptance rate below which the run stops."
    )
    stay_probability: float = Field(
        gt=0.0, lt=1.0, description="Probability a particle does not move."
    )


class AbcRejection(WithPriors, UncertaintyMethod[SmcResult]):
    """
    ABC rejection: `draws` draws of the priors, of which the closest share
    `quantile` is kept, with equal weights. It is one step of `AbcSmc` with
    no move. With a stochastic engine each distance reads one fresh
    replicate.

    Attributes
    ----------
    priors : Priors | dict[str, Distribution]
        The prior of each calibrated parameter, or
        ``Priors.uniform_on_bounds()``.
    distance : LossDistance | SeriesDistance
        How far a run is from the observations.
    draws : int
        Number of draws of the priors, at least 2.
    quantile : float
        The share of the draws kept, strictly between 0 and 1.
    """

    distance: AbcDistance = Field(description="The distance.")
    draws: int = Field(ge=2, description="Number of draws.")
    quantile: float = Field(gt=0.0, lt=1.0, description="The share of draws kept.")
