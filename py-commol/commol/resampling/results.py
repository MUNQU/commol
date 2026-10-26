"""The results of the methods of `commol.resampling`."""

from pydantic import ConfigDict, Field

from commol._data.result_model import ResultModel
from commol._fit.result import CalibrationResult
from commol.commol_rs import _commol_rs as commol_rs
from commol.uncertainty.sample import ParameterSample


class ResamplingDiagnostics(ResultModel):
    """
    The blocks of a resampling method, and the synthetic values below zero.

    Attributes
    ----------
    block_length : int | None
        The block length the method ran with; None for the parametric
        bootstrap, which draws no blocks.
    block_length_estimates : dict[str, float] | None
        With ``block_length="auto"``, the estimate of each series, by label;
        otherwise None.
    negative_values : int
        Synthetic observed values below zero, over every replicate. They are
        kept; many of them point to ``residual_scale="multiplicative"``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    block_length: int | None = Field(description="The block length used.")
    block_length_estimates: dict[str, float] | None = Field(
        description="The estimate of each series with the automatic length."
    )
    negative_values: int = Field(description="Synthetic values below zero.")


class BoundError(ResultModel):
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


class BootstrapResult(ResultModel):
    """
    The result of a bootstrap.

    Attributes
    ----------
    fit : CalibrationResult
        The fit every replicate starts from: ``Calibrator.run(starts)``.
    sample : ParameterSample
        One draw per replicate that did not fail, with equal weights; the
        group of a draw is its replicate.
    failed : list[str]
        Each replicate that failed: ``"Replicate <index> failed: <reason>"``,
        also logged as a warning. A failed replicate is not in the sample.
    not_converged : int
        Replicates kept although their optimizer did not converge.
    diagnostics : ResamplingDiagnostics
        The block length, and the synthetic values below zero.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    fit: CalibrationResult
    sample: ParameterSample
    failed: list[str] = Field(description="Each replicate that failed.")
    not_converged: int = Field(description="Replicates that did not converge.")
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
        low, high = commol_rs.inference.interval_levels(confidence_level)
        errors: dict[str, BoundError] = {}
        for index, name in enumerate(self.sample.names):
            column = [row[index] for row in self.sample.values]
            errors[name] = BoundError(
                lower=commol_rs.inference.quantile_error(column, low),
                upper=commol_rs.inference.quantile_error(column, high),
            )
        return errors


class JackknifeResult(ResultModel):
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
    failed : list[str]
        Each block whose fit failed: ``"Block <index> failed: <reason>"``,
        with blocks numbered from 0 in order of position; also logged as a
        warning. A block that failed is not in `blocks`.
    not_converged : int
        Fits kept although their optimizer did not converge.
    diagnostics : ResamplingDiagnostics
        The block length.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    fit: CalibrationResult
    estimate: dict[str, float]
    bias: dict[str, float]
    standard_error: dict[str, float]
    blocks: list[tuple[float, float]]
    pseudo_values: dict[str, list[float]]
    influence: dict[str, list[float]]
    failed: list[str] = Field(description="Each block whose fit failed.")
    not_converged: int = Field(description="Fits that did not converge.")
    diagnostics: ResamplingDiagnostics
