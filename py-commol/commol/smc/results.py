"""The results of the methods of `commol.smc`."""

from pydantic import ConfigDict, Field

from commol._data.result_model import ResultModel
from commol._fit.result import CalibrationResult
from commol.uncertainty.sample import ParameterSample


class SMCDiagnostics(ResultModel):
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


class SMCResult(ResultModel):
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
    diagnostics : SMCDiagnostics
        The effective sizes, acceptance rates and move counts of each step.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample: ParameterSample
    levels: list[float]
    kept: list[float | None]
    log_evidence: float
    diagnostics: SMCDiagnostics


class GibbsResult(SMCResult):
    """
    The result of a Gibbs posterior: the attributes of `SMCResult`, and these.

    Attributes
    ----------
    eta : float
        The `η` the posterior ran with.
    fit : CalibrationResult | None
        The fit a calibration of `η` read, ``Calibrator.run(starts)``; None
        when `η` is a number.
    coverage : float | None
        With `BootstrapCoverageEta`, the coverage of the fit by the intervals of
        the bootstrap posteriors at `η`; otherwise None.
    failed : list[str]
        With `BootstrapCoverageEta`, each replicate whose sampler failed:
        ``"Replicate <index> failed: <reason>"``, also logged as a warning.
        Empty otherwise.
    block_length : int | None
        With a calibration of `η`, the positions per block it ran with; None
        when `η` is a number.
    block_length_estimates : dict[str, float] | None
        With a calibration of `η` and ``block_length="auto"``, the estimate
        of each series, by label; otherwise None.
    """

    eta: float
    fit: CalibrationResult | None
    coverage: float | None
    block_length: int | None
    block_length_estimates: dict[str, float] | None
    failed: list[str] = Field(default_factory=list)
