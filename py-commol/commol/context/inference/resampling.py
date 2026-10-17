"""The resampling methods: the Bayesian, residual and wild block bootstraps
and the block jackknife, their settings and their results.

Each fits the model again to reweighted or resampled data, starting from the
fit, with no noise model and no prior. Blocks of consecutive positions, across
every series at once, keep the dependence of the data inside each block.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from commol.commol_rs import _commol_rs as commol_rs
from commol.context.calibration import CalibrationResult
from commol.context.inference.method import UncertaintyMethod
from commol.context.inference.sample import ParameterSample

type BlockLength = Annotated[int, Field(ge=1)] | Literal["auto"]
"""Positions per block: an integer of at least 1, or ``"auto"``, the rule of
Politis and White (2004) with the correction of Patton, Politis and White
(2009) on the residuals of the fit, the largest over the series."""


class ResamplingDiagnostics(BaseModel):
    """
    What happened to the replicates of a resampling method.

    Attributes
    ----------
    block_length : int
        The block length the method ran with.
    block_length_estimates : dict[str, float] | None
        With ``block_length="auto"``, the estimate of each series, by label;
        otherwise None.
    failed : list[str]
        Each replicate that failed, with its reason. A failed replicate is not
        in the sample.
    not_converged : int
        Replicates kept although their optimizer did not converge.
    negative_values : int
        Synthetic observed values below zero, over every replicate. They are
        kept; many of them point to ``residual_scale="multiplicative"``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    block_length: int = Field(description="The block length used.")
    block_length_estimates: dict[str, float] | None = Field(
        description="The estimate of each series with the automatic length."
    )
    failed: list[str] = Field(description="Each replicate that failed.")
    not_converged: int = Field(description="Replicates that did not converge.")
    negative_values: int = Field(description="Synthetic values below zero.")


class BoundError(BaseModel):
    """
    The Monte Carlo error of the bounds of an interval: for each bound, an
    order-statistic interval that holds the bound of infinitely many
    replicates with about 95 % probability.

    Attributes
    ----------
    lower : tuple[float, float]
        The error interval of the lower bound.
    upper : tuple[float, float]
        The error interval of the upper bound.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    lower: tuple[float, float]
    upper: tuple[float, float]


class BootstrapResult(BaseModel):
    """
    The result of a bootstrap.

    Attributes
    ----------
    fit : CalibrationResult
        The fit every replicate starts from: ``Calibrator.run(starts)``.
    sample : ParameterSample
        One draw per replicate that did not fail, with equal weights; the
        group of a draw is its replicate.
    diagnostics : ResamplingDiagnostics
        The block length, and the replicates that failed, did not converge or
        made values below zero.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    fit: CalibrationResult
    sample: ParameterSample
    diagnostics: ResamplingDiagnostics

    def interval_error(self, confidence_level: float) -> dict[str, BoundError]:
        """
        The Monte Carlo error of the central interval of each parameter.

        More replicates make it smaller; when it is wide against the interval,
        run more.

        Parameters
        ----------
        confidence_level : float
            Strictly between 0 and 1.

        Returns
        -------
        dict[str, BoundError]
            The error of each bound, by parameter.

        Raises
        ------
        ValueError
            If the confidence level is not strictly between 0 and 1.
        """
        if not 0.0 < confidence_level < 1.0:
            raise ValueError(
                f"A confidence level is strictly between 0 and 1, got "
                f"{confidence_level}."
            )
        low, high = (1.0 - confidence_level) / 2.0, (1.0 + confidence_level) / 2.0
        errors: dict[str, BoundError] = {}
        for index, name in enumerate(self.sample.names):
            column = [row[index] for row in self.sample.values]
            errors[name] = BoundError(
                lower=commol_rs.inference.quantile_error(column, low),
                upper=commol_rs.inference.quantile_error(column, high),
            )
        return errors


class BayesianBootstrap(UncertaintyMethod[BootstrapResult]):
    """
    The Bayesian (weighted) bootstrap by blocks.

    Each replicate draws one Dirichlet(1, …, 1) weight per block, scaled to
    mean 1, multiplies the weight of every observation of the block by it,
    and fits. With a loss it is the loss-likelihood bootstrap of Lyddon,
    Holmes and Walker (2019); with one position per block, the Bayesian
    bootstrap of Rubin (1981) and the weighted likelihood bootstrap of Newton
    and Raftery (1994). The observed values never change.

    Attributes
    ----------
    replicates : int
        Number of replicates, at least 1.
    block_length : int | "auto"
        Positions per block.
    starts : int
        Starts of the fit and of each replicate (default 1): the fit, then
        rows of a Latin hypercube over the bounds.
    """

    replicates: int = Field(ge=1, description="Number of replicates.")
    block_length: BlockLength = Field(description="Positions per block.")
    starts: int = Field(default=1, ge=1, description="Starts per fit.")


class ResidualBootstrap(UncertaintyMethod[BootstrapResult]):
    """
    The residual block bootstrap.

    Each replicate joins blocks of residuals of the fit, drawn with
    replacement across every series at once, adds them to the fitted values
    on `residual_scale`, and fits the synthetic data.

    Attributes
    ----------
    replicates : int
        Number of replicates, at least 1.
    block_length : int | "auto"
        Positions per block; with ``scheme="stationary"`` the mean length.
    residual_scale : {"additive", "multiplicative"}
        ``"additive"``: ``r = y − ŷ`` and ``ŷ + r*``. ``"multiplicative"``:
        ``r = log(y / ŷ)`` and ``ŷ · exp(r*)``; it needs positive observed and
        fitted values and keeps synthetic values positive.
    scheme : {"moving", "stationary"}
        ``"moving"``: blocks of `block_length` consecutive positions (Künsch
        1989). ``"stationary"``: blocks of geometric length with mean
        `block_length`, wrapped around the end (Politis and Romano 1994).
    starts : int
        Starts of the fit and of each replicate (default 1).
    """

    replicates: int = Field(ge=1, description="Number of replicates.")
    block_length: BlockLength = Field(description="Positions per block.")
    residual_scale: Literal["additive", "multiplicative"] = Field(
        description="The scale of the residuals."
    )
    scheme: Literal["moving", "stationary"] = Field(
        description="How blocks of residuals are drawn."
    )
    starts: int = Field(default=1, ge=1, description="Starts per fit.")


class WildBootstrap(UncertaintyMethod[BootstrapResult]):
    """
    The wild block bootstrap.

    Each replicate multiplies the additive residuals of each block by one
    draw of `multipliers` and fits ``ŷ + r · v``. Every residual stays at its
    own position, so it holds under noise whose size changes along the
    series.

    Attributes
    ----------
    replicates : int
        Number of replicates, at least 1.
    block_length : int | "auto"
        Positions per block.
    multipliers : {"rademacher", "mammen"}
        ``"rademacher"``: ±1 with probability 1/2. ``"mammen"``: two values
        with mean 0, variance 1 and third moment 1 (Mammen 1993), which keeps
        the skewness of the residuals.
    starts : int
        Starts of the fit and of each replicate (default 1).
    """

    replicates: int = Field(ge=1, description="Number of replicates.")
    block_length: BlockLength = Field(description="Positions per block.")
    multipliers: Literal["rademacher", "mammen"] = Field(
        description="The distribution of the multipliers."
    )
    starts: int = Field(default=1, ge=1, description="Starts per fit.")


class JackknifeResult(BaseModel):
    """
    The result of a block jackknife.

    With ``B`` fits ``θ̂₍₋ᵦ₎`` that leave one block out, around the fit
    ``θ̂``, and their mean ``θ̄``: the bias is ``(B − 1)(θ̄ − θ̂)``, the
    standard error ``√((B − 1)/B · Σ (θ̂₍₋ᵦ₎ − θ̄)²)``, and the pseudo-value of
    block ``b`` is ``B θ̂ − (B − 1) θ̂₍₋ᵦ₎``. The pseudo-values are not draws:
    their percentiles are not an interval.

    Attributes
    ----------
    fit : CalibrationResult
        The fit with every observation.
    estimate : dict[str, float]
        The bias-corrected estimate of each parameter.
    bias : dict[str, float]
        The jackknife bias of each parameter.
    standard_error : dict[str, float]
        The jackknife standard error of each parameter.
    blocks : list[tuple[float, float]]
        The first and last position of each block that was left out and
        fitted, in order.
    pseudo_values : dict[str, list[float]]
        The pseudo-values of each parameter, one per block.
    influence : dict[str, list[float]]
        How far leaving each block out moves each parameter, in standard
        errors: ``(θ̂₍₋ᵦ₎ − θ̂) / se``, one per block. NaN or infinite when a
        standard error is 0.
    diagnostics : ResamplingDiagnostics
        The block length and the fits that failed or did not converge.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, ser_json_inf_nan="constants")

    fit: CalibrationResult
    estimate: dict[str, float]
    bias: dict[str, float]
    standard_error: dict[str, float]
    blocks: list[tuple[float, float]]
    pseudo_values: dict[str, list[float]]
    influence: dict[str, list[float]]
    diagnostics: ResamplingDiagnostics


class BlockJackknife(UncertaintyMethod[JackknifeResult]):
    """
    The block jackknife (Künsch 1989): one fit per block left out.

    It gives a standard error and a bias of each parameter, and the influence
    of each block, with as many fits as blocks. A block is left out by giving
    its observations weight 0.

    Attributes
    ----------
    block_length : int | "auto"
        Positions per block; it must leave at least two blocks.
    starts : int
        Starts of the fit and of each leave-out fit (default 1).
    """

    block_length: BlockLength = Field(description="Positions per block.")
    starts: int = Field(default=1, ge=1, description="Starts per fit.")
